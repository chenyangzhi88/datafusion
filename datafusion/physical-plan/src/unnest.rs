// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Define a plan for unnesting values in columns that contain a list type.

use std::cmp::{self, Ordering};
use std::task::{Poll, ready};
use std::{any::Any, sync::Arc};

use super::metrics::{
    self, BaselineMetrics, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet, RecordOutput,
};
use super::{DisplayAs, ExecutionPlanProperties, PlanProperties};
use crate::{
    DisplayFormatType, Distribution, ExecutionPlan, RecordBatchStream, SendableRecordBatchStream,
    check_if_same_properties,
};

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanBufferBuilder, FixedSizeListArray, Int64Array, LargeListArray,
    ListArray, PrimitiveArray, Scalar, StructArray, new_null_array,
};
use arrow::compute::kernels::length::length;
use arrow::compute::kernels::zip::zip;
use arrow::compute::{cast, is_not_null, kernels};
use arrow::datatypes::{DataType, Int64Type, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use arrow_ord::cmp::lt;
use async_trait::async_trait;
use datafusion_common::{
    Constraints, HashMap, HashSet, Result, UnnestOptions, exec_datafusion_err, exec_err,
    internal_err,
};
use datafusion_execution::TaskContext;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation};
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::equivalence::ProjectionMapping;
use datafusion_physical_expr::expressions::Column;
use futures::{Stream, StreamExt};
use log::trace;

/// Checked construction capacity for a typed leaf array. Variable bytes are
/// the actual already-borrowed payload sum, not a displayed SQL width. Builder
/// doubling/alignment and immutable owner/header construction are admitted
/// while the original scalar/input allocation is still live.
pub fn array_construction_bound(
    data_type: &DataType,
    rows: usize,
    variable_bytes: usize,
) -> Result<usize> {
    let width = data_type
        .primitive_width()
        .unwrap_or_else(|| match data_type {
            DataType::Boolean => 1,
            DataType::Utf8 | DataType::Binary => 4,
            DataType::LargeUtf8 | DataType::LargeBinary => 8,
            DataType::Utf8View | DataType::BinaryView => 16,
            DataType::FixedSizeBinary(width) => (*width).max(0) as usize,
            DataType::Null => 0,
            _ => 16,
        });
    rows.checked_add(1)
        .and_then(|n| n.checked_mul(width))
        .and_then(|n| n.checked_add(variable_bytes))
        .and_then(|n| n.checked_add(rows.div_ceil(8)))
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_add(2048))
        .ok_or_else(allocation_overflow)
}

fn allocation_overflow() -> datafusion_common::DataFusionError {
    datafusion_common::DataFusionError::ResourcesExhausted(
        "Arrow construction size overflow".into(),
    )
}

// Each immutable allocation retains its actual original buffer and one
// reservation, shared only between views of that same physical allocation.
// Field drop order releases the buffer before the final reservation owner.
struct ReservedAllocation {
    _buffer: arrow::buffer::Buffer,
    _reservation: std::panic::AssertUnwindSafe<Arc<MemoryReservation>>,
}

const ALLOCATION_OWNER_BYTES: usize = 256;

fn allocation_view_bound(data_type: &DataType) -> Result<usize> {
    let mut views = 4usize; // null bitmap and up to three own buffers
    let mut add = |child: &DataType| -> Result<()> {
        views = views
            .checked_add(allocation_view_bound(child)?)
            .ok_or_else(allocation_overflow)?;
        Ok(())
    };
    match data_type {
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::Map(field, _) => add(field.data_type())?,
        DataType::Struct(fields) => {
            for field in fields {
                add(field.data_type())?;
            }
        }
        DataType::Union(fields, _) => {
            for (_, field) in fields.iter() {
                add(field.data_type())?;
            }
        }
        DataType::Dictionary(_, values) => add(values)?,
        DataType::RunEndEncoded(ends, values) => {
            add(ends.data_type())?;
            add(values.data_type())?;
        }
        _ => {}
    }
    Ok(views)
}

/// Pre-admit the existing retention service's temporary typed views and the
/// maximum new output allocation owners. Input payload is not charged here.
/// Callers add their actual construction bytes before creating output arrays.
pub fn array_retention_bound<'a>(
    output_type: &DataType,
    shared_input_types: impl IntoIterator<Item = &'a DataType>,
) -> Result<usize> {
    let output_views = allocation_view_bound(output_type)?;
    let views = shared_input_types
        .into_iter()
        .try_fold(output_views, |sum, data_type| {
            sum.checked_add(allocation_view_bound(data_type)?)
                .ok_or_else(allocation_overflow)
        })?;
    views
        .checked_mul(128)
        .and_then(|scratch| scratch.checked_add(output_views.checked_mul(ALLOCATION_OWNER_BYTES)?))
        .ok_or_else(allocation_overflow)
}

/// Transfer pre-admitted *new* physical buffers to their last immutable owner.
/// Shared input buffers preserve their existing owner without another charge.
/// This function never grows a reservation after array construction: exceeding
/// a proven bound is an internal construction invariant failure, not admission.
pub fn retain_array_allocations(
    arrays: Vec<ArrayRef>,
    reservation: MemoryReservation,
    shared_inputs: &[ArrayRef],
) -> Result<Vec<ArrayRef>> {
    // ArrayData headers, the bounded pointer sets/maps and reconstructed array
    // containers are temporary construction scratch, not downstream payload.
    // Split their proven bound before the first to_data/map allocation.
    let views = arrays
        .iter()
        .chain(shared_inputs)
        .try_fold(0usize, |sum, array| {
            sum.checked_add(allocation_view_bound(array.data_type())?)
                .ok_or_else(allocation_overflow)
        })?;
    let scratch_bytes = views.checked_mul(128).ok_or_else(allocation_overflow)?;
    if scratch_bytes > reservation.size() {
        return internal_err!("Arrow owner scratch exceeded pre-admitted capacity");
    }
    let _scratch = reservation.split(scratch_bytes);
    fn visit(data: &arrow::array::ArrayData, visitor: &mut impl FnMut(&arrow::buffer::Buffer)) {
        for buffer in data.buffers() {
            visitor(buffer);
        }
        if let Some(nulls) = data.nulls() {
            visitor(nulls.inner().inner());
        }
        for child in data.child_data() {
            visit(child, visitor);
        }
    }
    let mut shared = HashSet::new();
    for input in shared_inputs {
        visit(&input.to_data(), &mut |buffer| {
            shared.insert(buffer.data_ptr().as_ptr() as usize);
        });
    }
    let mut charges = HashMap::<usize, (usize, usize)>::new();
    let mut overflow = false;
    for array in &arrays {
        visit(&array.to_data(), &mut |buffer| {
            let pointer = buffer.data_ptr().as_ptr() as usize;
            if shared.contains(&pointer) || (buffer.is_empty() && buffer.capacity() == 0) {
                return;
            }
            let entry = charges.entry(pointer).or_default();
            entry.0 = entry.0.max(buffer.capacity().max(buffer.len()));
            match entry.1.checked_add(1) {
                Some(views) => entry.1 = views,
                None => overflow = true,
            }
        });
    }
    if overflow {
        return Err(allocation_overflow());
    }
    let total = charges.values().try_fold(0usize, |sum, (bytes, views)| {
        views
            .checked_mul(ALLOCATION_OWNER_BYTES)
            .and_then(|n| n.checked_add(*bytes))
            .and_then(|n| sum.checked_add(n))
            .ok_or_else(allocation_overflow)
    })?;
    if total > reservation.size() {
        return internal_err!(
            "Arrow construction exceeded its pre-admitted capacity: {total} > {}",
            reservation.size()
        );
    }
    let owners = charges
        .into_iter()
        .map(|(pointer, (bytes, views))| {
            (
                pointer,
                Arc::new(reservation.split(bytes + views * ALLOCATION_OWNER_BYTES)),
            )
        })
        .collect::<HashMap<_, _>>();
    fn retain(
        data: arrow::array::ArrayData,
        owners: &HashMap<usize, Arc<MemoryReservation>>,
    ) -> Result<arrow::array::ArrayData> {
        let wrap = |buffer: &arrow::buffer::Buffer| {
            let Some(reservation) = owners.get(&(buffer.data_ptr().as_ptr() as usize)) else {
                return buffer.clone();
            };
            let pointer = std::ptr::NonNull::new(buffer.as_ptr() as *mut u8)
                .expect("Arrow immutable buffers have a non-null pointer");
            let owner = Arc::new(ReservedAllocation {
                _buffer: buffer.clone(),
                _reservation: std::panic::AssertUnwindSafe(Arc::clone(reservation)),
            });
            // SAFETY: this exact immutable clone retains as_ptr() for len()
            // initialized bytes, including its slice offset. Its allocation
            // cannot move or mutate. The custom owner outlives every clone and
            // slice of the returned view. A zero-length allocated buffer still
            // retains its capacity/owner, without reading uninitialized bytes.
            unsafe { arrow::buffer::Buffer::from_custom_allocation(pointer, buffer.len(), owner) }
        };
        let buffers = data.buffers().iter().map(wrap).collect();
        let nulls = data.nulls().map(|nulls| {
            let buffer = arrow::buffer::BooleanBuffer::new(
                wrap(nulls.inner().inner()),
                nulls.offset(),
                nulls.len(),
            );
            // SAFETY: wrapping changes only the lifetime owner. Every bit,
            // offset and length is identical to the original immutable null
            // bitmap, so its existing exact count remains valid. Do not scan
            // all rows again merely to retain their allocation owner.
            unsafe { arrow::buffer::NullBuffer::new_unchecked(buffer, nulls.null_count()) }
        });
        let children = data
            .child_data()
            .iter()
            .cloned()
            .map(|child| retain(child, owners))
            .collect::<Result<Vec<_>>>()?;
        Ok(data
            .into_builder()
            .buffers(buffers)
            .nulls(nulls)
            .child_data(children)
            .build()?)
    }
    arrays
        .into_iter()
        .map(|array| retain(array.to_data(), &owners).map(arrow::array::make_array))
        .collect()
}

/// Same service for a single newly constructed typed array. Caller owns the
/// admission and same-root reservation before invoking any builder/kernel.
pub fn retain_array_allocation(
    array: ArrayRef,
    reservation: MemoryReservation,
) -> Result<ArrayRef> {
    let mut arrays = retain_array_allocations(vec![array], reservation, &[])?;
    Ok(arrays.remove(0))
}

fn take_bound(data: &arrow::array::ArrayData, rows: usize, repeats: usize) -> Result<usize> {
    let payload = match data.data_type() {
        DataType::Utf8 | DataType::Binary | DataType::LargeUtf8 | DataType::LargeBinary => data
            .buffers()
            .get(1)
            .map_or(0, |buffer| buffer.len())
            .checked_mul(repeats)
            .ok_or_else(allocation_overflow)?,
        _ => 0,
    };
    let mut bound = array_construction_bound(data.data_type(), rows, payload)?;
    // Arrow dictionary/view takes reuse their value buffers. Only fresh keys
    // or view records/null bits are admitted above; never charge shared data.
    if matches!(
        data.data_type(),
        DataType::Dictionary(_, _) | DataType::Utf8View | DataType::BinaryView
    ) {
        return Ok(bound);
    }
    for child in data.child_data() {
        let child_rows = match data.data_type() {
            DataType::Struct(_) => rows,
            DataType::FixedSizeList(_, width) => rows
                .checked_mul((*width).max(0) as usize)
                .ok_or_else(allocation_overflow)?,
            DataType::RunEndEncoded(_, _) => rows,
            _ => child
                .len()
                .checked_mul(repeats)
                .ok_or_else(allocation_overflow)?,
        };
        bound = bound
            .checked_add(take_bound(child, child_rows, repeats)?)
            .ok_or_else(allocation_overflow)?;
    }
    Ok(bound)
}

/// Unnest the given columns (either with type struct or list)
/// For list unnesting, each row is vertically transformed into multiple rows
/// For struct unnesting, each column is horizontally transformed into multiple columns,
/// Thus the original RecordBatch with dimension (n x m) may have new dimension (n' x m')
///
/// See [`UnnestOptions`] for more details and an example.
#[derive(Debug, Clone)]
pub struct UnnestExec {
    /// Input execution plan
    input: Arc<dyn ExecutionPlan>,
    /// The schema once the unnest is applied
    schema: SchemaRef,
    /// Indices of the list-typed columns in the input schema
    list_column_indices: Vec<ListUnnest>,
    /// Indices of the struct-typed columns in the input schema
    struct_column_indices: Vec<usize>,
    /// Options
    options: UnnestOptions,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
    /// Cache holding plan properties like equivalences, output partitioning etc.
    cache: Arc<PlanProperties>,
}

impl UnnestExec {
    /// Create a new [UnnestExec].
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        list_column_indices: Vec<ListUnnest>,
        struct_column_indices: Vec<usize>,
        schema: SchemaRef,
        options: UnnestOptions,
    ) -> Result<Self> {
        let cache = Self::compute_properties(
            &input,
            &list_column_indices,
            &struct_column_indices,
            &schema,
        )?;

        Ok(UnnestExec {
            input,
            schema,
            list_column_indices,
            struct_column_indices,
            options,
            metrics: Default::default(),
            cache: Arc::new(cache),
        })
    }

    /// This function creates the cache object that stores the plan properties such as schema, equivalence properties, ordering, partitioning, etc.
    fn compute_properties(
        input: &Arc<dyn ExecutionPlan>,
        list_column_indices: &[ListUnnest],
        struct_column_indices: &[usize],
        schema: &SchemaRef,
    ) -> Result<PlanProperties> {
        // Find out which indices are not unnested, such that they can be copied over from the input plan
        let input_schema = input.schema();
        let mut unnested_indices = BooleanBufferBuilder::new(input_schema.fields().len());
        unnested_indices.append_n(input_schema.fields().len(), false);
        for list_unnest in list_column_indices {
            unnested_indices.set_bit(list_unnest.index_in_input_schema, true);
        }
        for struct_unnest in struct_column_indices {
            unnested_indices.set_bit(*struct_unnest, true)
        }
        let unnested_indices = unnested_indices.finish();
        let non_unnested_indices: Vec<usize> = (0..input_schema.fields().len())
            .filter(|idx| !unnested_indices.value(*idx))
            .collect();

        // Manually build projection mapping from non-unnested input columns to their positions in the output
        let input_schema = input.schema();
        let projection_mapping: ProjectionMapping = non_unnested_indices
            .iter()
            .map(|&input_idx| {
                // Find what index the input column has in the output schema
                let input_field = input_schema.field(input_idx);
                let output_idx = schema
                    .fields()
                    .iter()
                    .position(|output_field| output_field.name() == input_field.name())
                    .ok_or_else(|| {
                        exec_datafusion_err!(
                            "Non-unnested column '{}' must exist in output schema",
                            input_field.name()
                        )
                    })?;

                let input_col =
                    Arc::new(Column::new(input_field.name(), input_idx)) as Arc<dyn PhysicalExpr>;
                let target_col =
                    Arc::new(Column::new(input_field.name(), output_idx)) as Arc<dyn PhysicalExpr>;
                // Use From<Vec<(Arc<dyn PhysicalExpr>, usize)>> for ProjectionTargets
                let targets = vec![(target_col, output_idx)].into();
                Ok((input_col, targets))
            })
            .collect::<Result<ProjectionMapping>>()?;

        // Create the unnest's equivalence properties by copying the input plan's equivalence properties
        // for the unaffected columns. Except for the constraints, which are removed entirely because
        // the unnest operation invalidates any global uniqueness or primary-key constraints.
        let input_eq_properties = input.equivalence_properties();
        let eq_properties = input_eq_properties
            .project(&projection_mapping, Arc::clone(schema))
            .with_constraints(Constraints::default());

        // Output partitioning must use the projection mapping
        let output_partitioning = input
            .output_partitioning()
            .project(&projection_mapping, &eq_properties);

        Ok(PlanProperties::new(
            eq_properties,
            output_partitioning,
            input.pipeline_behavior(),
            input.boundedness(),
        ))
    }

    /// Input execution plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Indices of the list-typed columns in the input schema
    pub fn list_column_indices(&self) -> &[ListUnnest] {
        &self.list_column_indices
    }

    /// Indices of the struct-typed columns in the input schema
    pub fn struct_column_indices(&self) -> &[usize] {
        &self.struct_column_indices
    }

    pub fn options(&self) -> &UnnestOptions {
        &self.options
    }

    fn with_new_children_and_same_properties(
        &self,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Self {
        Self {
            input: children.swap_remove(0),
            metrics: ExecutionPlanMetricsSet::new(),
            ..Self::clone(self)
        }
    }
}

impl DisplayAs for UnnestExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "UnnestExec")
            }
            DisplayFormatType::TreeRender => {
                write!(f, "")
            }
        }
    }
}

impl ExecutionPlan for UnnestExec {
    fn name(&self) -> &'static str {
        "UnnestExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        check_if_same_properties!(self, children);
        Ok(Arc::new(UnnestExec::new(
            children.swap_remove(0),
            self.list_column_indices.clone(),
            self.struct_column_indices.clone(),
            Arc::clone(&self.schema),
            self.options.clone(),
        )?))
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::UnspecifiedDistribution]
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let pool = Arc::clone(context.memory_pool());
        let input = self.input.execute(partition, context)?;
        let metrics = UnnestMetrics::new(partition, &self.metrics);

        Ok(Box::pin(UnnestStream {
            input,
            schema: Arc::clone(&self.schema),
            list_type_columns: self.list_column_indices.clone(),
            struct_column_indices: self.struct_column_indices.iter().copied().collect(),
            options: self.options.clone(),
            metrics,
            pool,
        }))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

#[derive(Clone, Debug)]
struct UnnestMetrics {
    /// Execution metrics
    baseline_metrics: BaselineMetrics,
    /// Number of batches consumed
    input_batches: metrics::Count,
    /// Number of rows consumed
    input_rows: metrics::Count,
}

impl UnnestMetrics {
    fn new(partition: usize, metrics: &ExecutionPlanMetricsSet) -> Self {
        let input_batches = MetricBuilder::new(metrics).counter("input_batches", partition);

        let input_rows = MetricBuilder::new(metrics).counter("input_rows", partition);

        Self {
            baseline_metrics: BaselineMetrics::new(metrics, partition),
            input_batches,
            input_rows,
        }
    }
}

/// A stream that issues [RecordBatch]es with unnested column data.
struct UnnestStream {
    /// Input stream
    input: SendableRecordBatchStream,
    /// Unnested schema
    schema: Arc<Schema>,
    /// represents all unnest operations to be applied to the input (input index, depth)
    /// e.g unnest(col1),unnest(unnest(col1)) where col1 has index 1 in original input schema
    /// then list_type_columns = [ListUnnest{1,1},ListUnnest{1,2}]
    list_type_columns: Vec<ListUnnest>,
    struct_column_indices: HashSet<usize>,
    /// Options
    options: UnnestOptions,
    /// Metrics
    metrics: UnnestMetrics,
    pool: Arc<dyn MemoryPool>,
}

impl RecordBatchStream for UnnestStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

#[async_trait]
impl Stream for UnnestStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.poll_next_impl(cx)
    }
}

impl UnnestStream {
    /// Separate implementation function that unpins the [`UnnestStream`] so
    /// that partial borrows work correctly
    fn poll_next_impl(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Result<RecordBatch>>> {
        loop {
            return Poll::Ready(match ready!(self.input.poll_next_unpin(cx)) {
                Some(Ok(batch)) => {
                    let elapsed_compute = self.metrics.baseline_metrics.elapsed_compute().clone();
                    let timer = elapsed_compute.timer();
                    self.metrics.input_batches.add(1);
                    self.metrics.input_rows.add(batch.num_rows());
                    let result = build_batch(
                        &batch,
                        &self.schema,
                        &self.list_type_columns,
                        &self.struct_column_indices,
                        &self.options,
                        &self.pool,
                    )?;
                    timer.done();
                    let Some(result_batch) = result else {
                        continue;
                    };
                    (&result_batch).record_output(&self.metrics.baseline_metrics);

                    // Empty record batches should not be emitted.
                    // They need to be treated as  [`Option<RecordBatch>`]es and handled separately
                    debug_assert!(result_batch.num_rows() > 0);
                    Some(Ok(result_batch))
                }
                other => {
                    trace!(
                        "Processed {} probe-side input batches containing {} rows and \
                        produced {} output batches containing {} rows in {}",
                        self.metrics.input_batches,
                        self.metrics.input_rows,
                        self.metrics.baseline_metrics.output_batches(),
                        self.metrics.baseline_metrics.output_rows(),
                        self.metrics.baseline_metrics.elapsed_compute(),
                    );
                    other
                }
            });
        }
    }
}

/// Given a set of struct column indices to flatten
/// try converting the column in input into multiple subfield columns
/// For example
/// struct_col: [a: struct(item: int, name: string), b: int]
/// with a batch
/// {a: {item: 1, name: "a"}, b: 2},
/// {a: {item: 3, name: "b"}, b: 4]
/// will be converted into
/// {a.item: 1, a.name: "a", b: 2},
/// {a.item: 3, a.name: "b", b: 4}
fn flatten_struct_cols(
    input_batch: &[Arc<dyn Array>],
    schema: &SchemaRef,
    struct_column_indices: &HashSet<usize>,
) -> Result<RecordBatch> {
    // horizontal expansion because of struct unnest
    let columns_expanded = input_batch
        .iter()
        .enumerate()
        .map(|(idx, column_data)| match struct_column_indices.get(&idx) {
            Some(_) => match column_data.data_type() {
                DataType::Struct(_) => {
                    let struct_arr = column_data.as_any().downcast_ref::<StructArray>().unwrap();
                    Ok(struct_arr.columns().to_vec())
                }
                data_type => internal_err!(
                    "expecting column {} from input plan to be a struct, got {:?}",
                    idx,
                    data_type
                ),
            },
            None => Ok(vec![Arc::clone(column_data)]),
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();
    Ok(RecordBatch::try_new(Arc::clone(schema), columns_expanded)?)
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ListUnnest {
    pub index_in_input_schema: usize,
    pub depth: usize,
}

/// This function is used to execute the unnesting on multiple columns all at once, but
/// one level at a time, and is called n times, where n is the highest recursion level among
/// the unnest exprs in the query.
///
/// For example giving the following query:
/// ```sql
/// select unnest(colA, max_depth:=3) as P1, unnest(colA,max_depth:=2) as P2, unnest(colB, max_depth:=1) as P3 from temp;
/// ```
/// Then the total times this function being called is 3
///
/// It needs to be aware of which level the current unnesting is, because if there exists
/// multiple unnesting on the same column, but with different recursion levels, say
/// **unnest(colA, max_depth:=3)** and **unnest(colA, max_depth:=2)**, then the unnesting
/// of expr **unnest(colA, max_depth:=3)** will start at level 3, while unnesting for expr
/// **unnest(colA, max_depth:=2)** has to start at level 2
///
/// Set *colA* as a 3-dimension columns and *colB* as an array (1-dimension). As stated,
/// this function is called with the descending order of recursion depth
///
/// Depth = 3
/// - colA(3-dimension) unnest into temp column temp_P1(2_dimension) (unnesting of P1 starts
///   from this level)
/// - colA(3-dimension) having indices repeated by the unnesting operation above
/// - colB(1-dimension) having indices repeated by the unnesting operation above
///
/// Depth = 2
/// - temp_P1(2-dimension) unnest into temp column temp_P1(1-dimension)
/// - colA(3-dimension) unnest into temp column temp_P2(2-dimension) (unnesting of P2 starts
///   from this level)
/// - colB(1-dimension) having indices repeated by the unnesting operation above
///
/// Depth = 1
/// - temp_P1(1-dimension) unnest into P1
/// - temp_P2(2-dimension) unnest into P2
/// - colB(1-dimension) unnest into P3 (unnesting of P3 starts from this level)
///
/// The returned array will has the same size as the input batch
/// and only contains original columns that are not being unnested.
fn list_unnest_at_level(
    batch: &[ArrayRef],
    list_type_unnests: &[ListUnnest],
    temp_unnested_arrs: &mut HashMap<ListUnnest, ArrayRef>,
    level_to_unnest: usize,
    options: &UnnestOptions,
    pool: &Arc<dyn MemoryPool>,
) -> Result<Option<Vec<ArrayRef>>> {
    // Reserve container and length-kernel scratch before allocating it. The
    // actual output buffers are separately admitted below and retain that
    // reservation after this invocation (including downstream column slices).
    let scratch = MemoryConsumer::new("UNNEST scratch").register(pool);
    let rows = batch.first().map_or(0, |array| array.len());
    let scratch_bytes = list_type_unnests
        .len()
        .checked_add(1)
        .and_then(|columns| rows.checked_mul(columns))
        .and_then(|rows| rows.checked_mul(64))
        .and_then(|bytes| {
            list_type_unnests
                .len()
                .checked_add(batch.len())
                .and_then(|columns| columns.checked_mul(512))
                .and_then(|containers| bytes.checked_add(containers))
        })
        .ok_or_else(allocation_overflow)?;
    scratch.try_grow(scratch_bytes)?;
    // Extract unnestable columns at this level
    let (arrs_to_unnest, list_unnest_specs): (Vec<Arc<dyn Array>>, Vec<_>) = list_type_unnests
        .iter()
        .filter_map(|unnesting| {
            if level_to_unnest == unnesting.depth {
                return Some((
                    Arc::clone(&batch[unnesting.index_in_input_schema]),
                    *unnesting,
                ));
            }
            // This means the unnesting on this item has started at higher level
            // and need to continue until depth reaches 1
            if level_to_unnest < unnesting.depth {
                return Some((
                    Arc::clone(temp_unnested_arrs.get(unnesting).unwrap()),
                    *unnesting,
                ));
            }
            None
        })
        .unzip();

    // Filter out so that list_arrays only contain column with the highest depth
    // at the same time, during iteration remove this depth so next time we don't have to unnest them again
    let mut total_length = 0usize;
    let mut maximum_repetition = 0usize;
    for row in 0..rows {
        let mut longest = 0usize;
        for array in &arrs_to_unnest {
            let list: &dyn ListArrayType = match array.data_type() {
                DataType::List(_) => array.as_list::<i32>(),
                DataType::LargeList(_) => array.as_list::<i64>(),
                DataType::FixedSizeList(_, _) => array.as_fixed_size_list(),
                other => return exec_err!("Invalid unnest datatype {other}"),
            };
            let length = if list.is_null(row) {
                usize::from(options.preserve_nulls)
            } else {
                let (start, end) = list.value_offsets(row);
                usize::try_from(end.checked_sub(start).ok_or_else(allocation_overflow)?)
                    .map_err(|_| allocation_overflow())?
            };
            longest = longest.max(length);
        }
        total_length = total_length
            .checked_add(longest)
            .ok_or_else(allocation_overflow)?;
        maximum_repetition = maximum_repetition.max(longest);
    }
    if total_length == 0 {
        return Ok(None);
    }
    let indices_bytes = total_length
        .checked_mul(32)
        .ok_or_else(allocation_overflow)?;
    scratch.try_grow(indices_bytes)?;
    let output = MemoryConsumer::new("UNNEST output").register(pool);
    let mut output_bytes = 0usize;
    for array in &arrs_to_unnest {
        let values = match array.data_type() {
            DataType::List(_) => array.as_list::<i32>().values(),
            DataType::LargeList(_) => array.as_list::<i64>().values(),
            DataType::FixedSizeList(_, _) => array.as_fixed_size_list().values(),
            _ => unreachable!("validated list array"),
        };
        output_bytes = output_bytes
            .checked_add(take_bound(&values.to_data(), total_length, 1)?)
            .ok_or_else(allocation_overflow)?;
    }
    // This upper bound includes columns temporarily replaced by typed NULLs;
    // they share the same physical output owner until the next level drops them.
    for array in batch {
        output_bytes = output_bytes
            .checked_add(take_bound(
                &array.to_data(),
                total_length.max(array.len()),
                maximum_repetition.max(1),
            )?)
            .ok_or_else(allocation_overflow)?;
    }
    output.try_grow(output_bytes)?;
    let longest_length = find_longest_length(&arrs_to_unnest, options)?;
    let unnested_length = longest_length.as_primitive::<Int64Type>();

    // Unnest all the list arrays
    let unnested_temp_arrays =
        unnest_list_arrays(arrs_to_unnest.as_ref(), unnested_length, total_length)?;

    // Create the take indices array for other columns
    let take_indices = create_take_indices(unnested_length, total_length);

    let repeat_mask: Vec<bool> = batch
        .iter()
        .enumerate()
        .map(|(i, _)| {
            // Check if the column is needed in future levels (levels below the current one)
            let needed_in_future_levels = list_type_unnests.iter().any(|unnesting| {
                unnesting.index_in_input_schema == i && unnesting.depth < level_to_unnest
            });

            // Check if the column is involved in unnesting at any level
            let is_involved_in_unnesting = list_type_unnests
                .iter()
                .any(|unnesting| unnesting.index_in_input_schema == i);

            // Repeat columns needed in future levels or not unnested.
            needed_in_future_levels || !is_involved_in_unnesting
        })
        .collect();

    // Dimension of arrays in batch is untouched, but the values are repeated
    // as the side effect of unnesting
    let ret = repeat_arrs_from_indices(batch, &take_indices, &repeat_mask)?;
    let selected_count = unnested_temp_arrays.len();
    let mut arrays = unnested_temp_arrays;
    arrays.extend(ret);
    let mut shared = batch.to_vec();
    shared.extend(arrs_to_unnest);
    let mut arrays = retain_array_allocations(arrays, output, &shared)?;
    let repeated = arrays.split_off(selected_count);
    for (array, unnesting) in arrays.into_iter().zip(list_unnest_specs) {
        temp_unnested_arrs.insert(unnesting, array);
    }
    Ok(Some(repeated))
}
struct UnnestingResult {
    arr: ArrayRef,
    depth: usize,
}

/// For each row in a `RecordBatch`, some list/struct columns need to be unnested.
/// - For list columns: We will expand the values in each list into multiple rows,
///   taking the longest length among these lists, and shorter lists are padded with NULLs.
/// - For struct columns: We will expand the struct columns into multiple subfield columns.
///
/// For columns that don't need to be unnested, repeat their values until reaching the longest length.
///
/// Note: unnest has a big difference in behavior between Postgres and DuckDB
///
/// Take this example
///
/// 1. Postgres
/// ```ignored
/// create table temp (
///     i integer[][][], j integer[]
/// )
/// insert into temp values ('{{{1,2},{3,4}},{{5,6},{7,8}}}', '{1,2}');
/// select unnest(i), unnest(j) from temp;
/// ```
///
/// Result
/// ```text
///     1   1
///     2   2
///     3
///     4
///     5
///     6
///     7
///     8
/// ```
/// 2. DuckDB
/// ```ignore
///     create table temp (i integer[][][], j integer[]);
///     insert into temp values ([[[1,2],[3,4]],[[5,6],[7,8]]], [1,2]);
///     select unnest(i,recursive:=true), unnest(j,recursive:=true) from temp;
/// ```
/// Result:
/// ```text
///
///     ┌────────────────────────────────────────────────┬────────────────────────────────────────────────┐
///     │ unnest(i, "recursive" := CAST('t' AS BOOLEAN)) │ unnest(j, "recursive" := CAST('t' AS BOOLEAN)) │
///     │                     int32                      │                     int32                      │
///     ├────────────────────────────────────────────────┼────────────────────────────────────────────────┤
///     │                                              1 │                                              1 │
///     │                                              2 │                                              2 │
///     │                                              3 │                                              1 │
///     │                                              4 │                                              2 │
///     │                                              5 │                                              1 │
///     │                                              6 │                                              2 │
///     │                                              7 │                                              1 │
///     │                                              8 │                                              2 │
///     └────────────────────────────────────────────────┴────────────────────────────────────────────────┘
/// ```
///
/// The following implementation refer to DuckDB's implementation
fn build_batch(
    batch: &RecordBatch,
    schema: &SchemaRef,
    list_type_columns: &[ListUnnest],
    struct_column_indices: &HashSet<usize>,
    options: &UnnestOptions,
    pool: &Arc<dyn MemoryPool>,
) -> Result<Option<RecordBatch>> {
    let scratch = MemoryConsumer::new("UNNEST batch structure").register(pool);
    let containers = batch
        .num_columns()
        .checked_add(list_type_columns.len())
        .and_then(|count| count.checked_add(schema.fields().len()))
        .and_then(|count| count.checked_mul(1024))
        .ok_or_else(allocation_overflow)?;
    scratch.try_grow(containers)?;
    let transformed = match list_type_columns.len() {
        0 => flatten_struct_cols(batch.columns(), schema, struct_column_indices),
        _ => {
            let mut temp_unnested_result = HashMap::new();
            let max_recursion = list_type_columns
                .iter()
                .fold(0, |highest_depth, ListUnnest { depth, .. }| {
                    cmp::max(highest_depth, *depth)
                });

            // This arr always has the same column count with the input batch
            let mut flatten_arrs = vec![];

            // Original batch has the same columns
            // All unnesting results are written to temp_batch
            for depth in (1..=max_recursion).rev() {
                let input = match depth == max_recursion {
                    true => batch.columns(),
                    false => &flatten_arrs,
                };
                let Some(temp_result) = list_unnest_at_level(
                    input,
                    list_type_columns,
                    &mut temp_unnested_result,
                    depth,
                    options,
                    pool,
                )?
                else {
                    return Ok(None);
                };
                flatten_arrs = temp_result;
            }
            let unnested_array_map: HashMap<usize, Vec<UnnestingResult>> =
                temp_unnested_result.into_iter().fold(
                    HashMap::new(),
                    |mut acc,
                     (
                        ListUnnest {
                            index_in_input_schema,
                            depth,
                        },
                        flattened_array,
                    )| {
                        acc.entry(index_in_input_schema)
                            .or_default()
                            .push(UnnestingResult {
                                arr: flattened_array,
                                depth,
                            });
                        acc
                    },
                );
            let output_order: HashMap<ListUnnest, usize> = list_type_columns
                .iter()
                .enumerate()
                .map(|(order, unnest_def)| (*unnest_def, order))
                .collect();

            // One original column may be unnested multiple times into separate columns
            let mut multi_unnested_per_original_index = unnested_array_map
                .into_iter()
                .map(
                    // Each item in unnested_columns is the result of unnesting the same input column
                    // we need to sort them to conform with the original expression order
                    // e.g unnest(unnest(col)) must goes before unnest(col)
                    |(original_index, mut unnested_columns)| {
                        unnested_columns.sort_by(
                            |UnnestingResult { depth: depth1, .. },
                             UnnestingResult { depth: depth2, .. }|
                             -> Ordering {
                                output_order
                                    .get(&ListUnnest {
                                        depth: *depth1,
                                        index_in_input_schema: original_index,
                                    })
                                    .unwrap()
                                    .cmp(
                                        output_order
                                            .get(&ListUnnest {
                                                depth: *depth2,
                                                index_in_input_schema: original_index,
                                            })
                                            .unwrap(),
                                    )
                            },
                        );
                        (
                            original_index,
                            unnested_columns
                                .into_iter()
                                .map(|result| result.arr)
                                .collect::<Vec<_>>(),
                        )
                    },
                )
                .collect::<HashMap<_, _>>();

            let ret = flatten_arrs
                .into_iter()
                .enumerate()
                .flat_map(|(col_idx, arr)| {
                    // Convert original column into its unnested version(s)
                    // Plural because one column can be unnested with different recursion level
                    // and into separate output columns
                    match multi_unnested_per_original_index.remove(&col_idx) {
                        Some(unnested_arrays) => unnested_arrays,
                        None => vec![arr],
                    }
                })
                .collect::<Vec<_>>();

            flatten_struct_cols(&ret, schema, struct_column_indices)
        }
    }?;
    Ok(Some(transformed))
}

/// Find the longest list length among the given list arrays for each row.
///
/// For example if we have the following two list arrays:
///
/// ```ignore
/// l1: [1, 2, 3], null, [], [3]
/// l2: [4,5], [], null, [6, 7]
/// ```
///
/// If `preserve_nulls` is false, the longest length array will be:
///
/// ```ignore
/// longest_length: [3, 0, 0, 2]
/// ```
///
/// whereas if `preserve_nulls` is true, the longest length array will be:
///
///
/// ```ignore
/// longest_length: [3, 1, 1, 2]
/// ```
fn find_longest_length(list_arrays: &[ArrayRef], options: &UnnestOptions) -> Result<ArrayRef> {
    // The length of a NULL list
    let null_length = if options.preserve_nulls {
        Scalar::new(Int64Array::from_value(1, 1))
    } else {
        Scalar::new(Int64Array::from_value(0, 1))
    };
    let list_lengths: Vec<ArrayRef> = list_arrays
        .iter()
        .map(|list_array| {
            let mut length_array = length(list_array)?;
            // Make sure length arrays have the same type. Int64 is the most general one.
            length_array = cast(&length_array, &DataType::Int64)?;
            length_array = zip(&is_not_null(&length_array)?, &length_array, &null_length)?;
            Ok(length_array)
        })
        .collect::<Result<_>>()?;

    let longest_length = list_lengths.iter().skip(1).try_fold(
        Arc::clone(&list_lengths[0]),
        |longest, current| {
            let is_lt = lt(&longest, &current)?;
            zip(&is_lt, &current, &longest)
        },
    )?;
    Ok(longest_length)
}

/// Trait defining common methods used for unnesting, implemented by list array types.
trait ListArrayType: Array {
    /// Returns a reference to the values of this list.
    fn values(&self) -> &ArrayRef;

    /// Returns the start and end offset of the values for the given row.
    fn value_offsets(&self, row: usize) -> (i64, i64);
}

impl ListArrayType for ListArray {
    fn values(&self) -> &ArrayRef {
        self.values()
    }

    fn value_offsets(&self, row: usize) -> (i64, i64) {
        let offsets = self.value_offsets();
        (offsets[row].into(), offsets[row + 1].into())
    }
}

impl ListArrayType for LargeListArray {
    fn values(&self) -> &ArrayRef {
        self.values()
    }

    fn value_offsets(&self, row: usize) -> (i64, i64) {
        let offsets = self.value_offsets();
        (offsets[row], offsets[row + 1])
    }
}

impl ListArrayType for FixedSizeListArray {
    fn values(&self) -> &ArrayRef {
        self.values()
    }

    fn value_offsets(&self, row: usize) -> (i64, i64) {
        let start = self.value_offset(row) as i64;
        (start, start + self.value_length() as i64)
    }
}

/// Unnest multiple list arrays according to the length array.
fn unnest_list_arrays(
    list_arrays: &[ArrayRef],
    length_array: &PrimitiveArray<Int64Type>,
    capacity: usize,
) -> Result<Vec<ArrayRef>> {
    let typed_arrays = list_arrays
        .iter()
        .map(|list_array| match list_array.data_type() {
            DataType::List(_) => Ok(list_array.as_list::<i32>() as &dyn ListArrayType),
            DataType::LargeList(_) => Ok(list_array.as_list::<i64>() as &dyn ListArrayType),
            DataType::FixedSizeList(_, _) => {
                Ok(list_array.as_fixed_size_list() as &dyn ListArrayType)
            }
            other => exec_err!("Invalid unnest datatype {other }"),
        })
        .collect::<Result<Vec<_>>>()?;

    typed_arrays
        .iter()
        .map(|list_array| unnest_list_array(*list_array, length_array, capacity))
        .collect::<Result<_>>()
}

/// Unnest a list array according the target length array.
///
/// Consider a list array like this:
///
/// ```ignore
/// [1], [2, 3, 4], null, [5], [],
/// ```
///
/// and the length array is:
///
/// ```ignore
/// [2, 3, 2, 1, 2]
/// ```
///
/// If the length of a certain list is less than the target length, pad with NULLs.
/// So the unnested array will look like this:
///
/// ```ignore
/// [1, null, 2, 3, 4, null, null, 5, null, null]
/// ```
fn unnest_list_array(
    list_array: &dyn ListArrayType,
    length_array: &PrimitiveArray<Int64Type>,
    capacity: usize,
) -> Result<ArrayRef> {
    let values = list_array.values();
    let mut take_indices_builder = PrimitiveArray::<Int64Type>::builder(capacity);
    for row in 0..list_array.len() {
        let mut value_length = 0;
        if !list_array.is_null(row) {
            let (start, end) = list_array.value_offsets(row);
            value_length = end - start;
            for i in start..end {
                take_indices_builder.append_value(i)
            }
        }
        let target_length = length_array.value(row);
        debug_assert!(
            value_length <= target_length,
            "value length is beyond the longest length"
        );
        // Pad with NULL values
        for _ in value_length..target_length {
            take_indices_builder.append_null();
        }
    }
    Ok(kernels::take::take(
        &values,
        &take_indices_builder.finish(),
        None,
    )?)
}

/// Creates take indices that will be used to expand all columns except for the list type
/// [`columns`](UnnestExec::list_column_indices) that is being unnested.
/// Every column value needs to be repeated multiple times according to the length array.
///
/// If the length array looks like this:
///
/// ```ignore
/// [2, 3, 1]
/// ```
/// Then [`create_take_indices`] will return an array like this
///
/// ```ignore
/// [0, 0, 1, 1, 1, 2]
/// ```
fn create_take_indices(
    length_array: &PrimitiveArray<Int64Type>,
    capacity: usize,
) -> PrimitiveArray<Int64Type> {
    // `find_longest_length()` guarantees this.
    debug_assert!(
        length_array.null_count() == 0,
        "length array should not contain nulls"
    );
    let mut builder = PrimitiveArray::<Int64Type>::builder(capacity);
    for (index, repeat) in length_array.iter().enumerate() {
        // The length array should not contain nulls, so unwrap is safe
        let repeat = repeat.unwrap();
        (0..repeat).for_each(|_| builder.append_value(index as i64));
    }
    builder.finish()
}

/// Create a batch of arrays based on an input `batch` and a `indices` array.
/// The `indices` array is used by the take kernel to repeat values in the arrays
/// that are marked with `true` in the `repeat_mask`. Arrays marked with `false`
/// in the `repeat_mask` will be replaced with arrays filled with nulls of the
/// appropriate length.
///
/// For example if we have the following batch:
///
/// ```ignore
/// c1: [1], null, [2, 3, 4], null, [5, 6]
/// c2: 'a', 'b',  'c', null, 'd'
/// ```
///
/// then the `unnested_list_arrays` contains the unnest column that will replace `c1` in
/// the final batch if `preserve_nulls` is true:
///
/// ```ignore
/// c1: 1, null, 2, 3, 4, null, 5, 6
/// ```
///
/// And the `indices` array contains the indices that are used by `take` kernel to
/// repeat the values in `c2`:
///
/// ```ignore
/// 0, 1, 2, 2, 2, 3, 4, 4
/// ```
///
/// so that the final batch will look like:
///
/// ```ignore
/// c1: 1, null, 2, 3, 4, null, 5, 6
/// c2: 'a', 'b', 'c', 'c', 'c', null, 'd', 'd'
/// ```
///
/// The `repeat_mask` determines whether an array's values are repeated or replaced with nulls.
/// For example, if the `repeat_mask` is:
///
/// ```ignore
/// [true, false]
/// ```
///
/// The final batch will look like:
///
/// ```ignore
/// c1: 1, null, 2, 3, 4, null, 5, 6  // Repeated using `indices`
/// c2: null, null, null, null, null, null, null, null  // Replaced with nulls
fn repeat_arrs_from_indices(
    batch: &[ArrayRef],
    indices: &PrimitiveArray<Int64Type>,
    repeat_mask: &[bool],
) -> Result<Vec<Arc<dyn Array>>> {
    batch
        .iter()
        .zip(repeat_mask.iter())
        .map(|(arr, &repeat)| {
            if repeat {
                Ok(kernels::take::take(arr, indices, None)?)
            } else {
                Ok(new_null_array(arr.data_type(), arr.len()))
            }
        })
        .collect()
}

#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests;
