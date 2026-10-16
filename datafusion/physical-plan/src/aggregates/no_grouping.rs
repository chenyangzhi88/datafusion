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

//! Aggregate without grouping columns

#[cfg(test)]
#[path = "tests/no_grouping.rs"]
mod tests;

use crate::aggregates::{
    AccumulatorItem, AggrDynFilter, AggregateInputMode, AggregateMode,
    DynamicFilterAggregateType, aggregate_expressions, create_accumulators,
    finalize_aggregation,
};
use crate::metrics::{BaselineMetrics, RecordOutput};
use crate::{RecordBatchStream, SendableRecordBatchStream};
use arrow::array::Array;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion_common::{Result, ScalarValue, internal_datafusion_err, internal_err};
use datafusion_execution::TaskContext;
use datafusion_expr::Operator;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::async_scalar_function::AsyncFuncExpr;
use datafusion_physical_expr::expressions::{BinaryExpr, lit};
use futures::stream::BoxStream;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::sync::Arc;
use std::task::{Context, Poll};

use super::AggregateExec;
use crate::filter::batch_filter;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_physical_expr_common::utils::evaluate_expressions_to_arrays;
use futures::stream::{Stream, StreamExt};

/// stream struct for aggregation without grouping columns
pub(crate) struct AggregateStream {
    stream: BoxStream<'static, Result<RecordBatch>>,
    schema: SchemaRef,
}

/// Actual implementation of [`AggregateStream`].
///
/// This is wrapped into yet another struct because we need to interact with the async memory management subsystem
/// during poll. To have as little code "weirdness" as possible, we chose to just use [`BoxStream`] together with
/// [`futures::stream::unfold`].
///
/// The latter requires a state object, which is [`AggregateStreamInner`].
struct AggregateStreamInner {
    // ==== Properties ====
    schema: SchemaRef,
    mode: AggregateMode,
    input: SendableRecordBatchStream,
    aggregate_expressions: Vec<Vec<Arc<dyn PhysicalExpr>>>,
    ordered_arguments: Option<Vec<Vec<Option<AsyncFuncExpr>>>>,
    ordered_filters: Option<Vec<Option<AsyncFuncExpr>>>,
    ordered_input_evaluator: Option<super::OrderedAggregateInputEvaluator>,
    options: Arc<datafusion_common::config::ConfigOptions>,
    filter_expressions: Arc<[Option<Arc<dyn PhysicalExpr>>]>,

    // ==== Runtime States/Buffers ====
    accumulators: Vec<AccumulatorItem>,
    // None if the dynamic filter is not applicable. See details in `AggrDynFilter`.
    agg_dyn_filter_state: Option<Arc<AggrDynFilter>>,
    finished: bool,

    // ==== Execution Resources ====
    baseline_metrics: BaselineMetrics,
    reservation: MemoryReservation,
    // Last: release the same producer after input/state/reservation drop.
    _observation: Option<Arc<dyn crate::sorts::observation::SortExecutionGuard>>,
}

impl AggregateStreamInner {
    // TODO: check if we get Null handling correct
    /// # Examples
    /// - Example 1
    ///   Accumulators: min(c1)
    ///   Current Bounds: min(c1)=10
    ///   --> dynamic filter PhysicalExpr: c1 < 10
    ///
    /// - Example 2
    ///   Accumulators: min(c1), max(c1), min(c2)
    ///   Current Bounds: min(c1)=10, max(c1)=100, min(c2)=20
    ///   --> dynamic filter PhysicalExpr: (c1 < 10) OR (c1>100) OR (c2 < 20)
    ///
    /// # Errors
    /// Returns internal errors if the dynamic filter is not enabled, or other
    /// invariant check fails.
    fn build_dynamic_filter_from_accumulator_bounds(
        &self,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        let Some(filter_state) = self.agg_dyn_filter_state.as_ref() else {
            return internal_err!(
                "`build_dynamic_filter_from_accumulator_bounds()` is only called when dynamic filter is enabled"
            );
        };

        let mut predicates: Vec<Arc<dyn PhysicalExpr>> =
            Vec::with_capacity(filter_state.supported_accumulators_info.len());

        for acc_info in &filter_state.supported_accumulators_info {
            // Skip if we don't yet have a meaningful bound
            let bound = {
                let guard = acc_info.shared_bound.lock();
                if (*guard).is_null() {
                    continue;
                }
                guard.clone()
            };

            let agg_exprs = self
                .aggregate_expressions
                .get(acc_info.aggr_index)
                .ok_or_else(|| {
                    internal_datafusion_err!(
                        "Invalid aggregate expression index {} for dynamic filter",
                        acc_info.aggr_index
                    )
                })?;
            // Only aggregates with a single argument are supported.
            let column_expr = agg_exprs.first().ok_or_else(|| {
                internal_datafusion_err!(
                    "Aggregate expression at index {} expected a single argument",
                    acc_info.aggr_index
                )
            })?;

            let literal = lit(bound);
            let predicate: Arc<dyn PhysicalExpr> = match acc_info.aggr_type {
                DynamicFilterAggregateType::Min => Arc::new(BinaryExpr::new(
                    Arc::clone(column_expr),
                    Operator::Lt,
                    literal,
                )),
                DynamicFilterAggregateType::Max => Arc::new(BinaryExpr::new(
                    Arc::clone(column_expr),
                    Operator::Gt,
                    literal,
                )),
            };
            predicates.push(predicate);
        }

        let combined = predicates.into_iter().reduce(|acc, pred| {
            Arc::new(BinaryExpr::new(acc, Operator::Or, pred)) as Arc<dyn PhysicalExpr>
        });

        Ok(combined.unwrap_or_else(|| lit(true)))
    }

    // If the dynamic filter is enabled, update it using the current accumulator's
    // values
    fn maybe_update_dyn_filter(&mut self) -> Result<()> {
        // Step 1: Update each partition's current bound
        let Some(filter_state) = self.agg_dyn_filter_state.as_ref() else {
            return Ok(());
        };

        let mut bounds_changed = false;

        for acc_info in &filter_state.supported_accumulators_info {
            let acc =
                self.accumulators
                    .get_mut(acc_info.aggr_index)
                    .ok_or_else(|| {
                        internal_datafusion_err!(
                            "Invalid accumulator index {} for dynamic filter",
                            acc_info.aggr_index
                        )
                    })?;
            // First get current partition's bound, then update the shared bound among
            // all partitions.
            let current_bound = acc.evaluate()?;
            {
                let mut bound = acc_info.shared_bound.lock();
                let new_bound = match acc_info.aggr_type {
                    DynamicFilterAggregateType::Max => {
                        scalar_max(&bound, &current_bound)?
                    }
                    DynamicFilterAggregateType::Min => {
                        scalar_min(&bound, &current_bound)?
                    }
                };
                if new_bound != *bound {
                    *bound = new_bound;
                    bounds_changed = true;
                }
            }
        }

        // Step 2: Sync the dynamic filter physical expression with reader,
        // but only if any bound actually changed.
        if bounds_changed {
            let predicate = self.build_dynamic_filter_from_accumulator_bounds()?;
            filter_state.filter.update(predicate)?;
        }

        Ok(())
    }
}

/// Returns the element-wise minimum of two `ScalarValue`s.
///
/// # Null semantics
/// - `min(NULL, NULL)      = NULL`
/// - `min(NULL, x)         = x`
/// - `min(x, NULL)         = x`
///
/// # Errors
/// Returns internal error if v1 and v2 has incompatible types.
fn scalar_min(v1: &ScalarValue, v2: &ScalarValue) -> Result<ScalarValue> {
    if let Some(result) = scalar_cmp_null_short_circuit(v1, v2) {
        return Ok(result);
    }

    match v1.partial_cmp(v2) {
        Some(Ordering::Less | Ordering::Equal) => Ok(v1.clone()),
        Some(Ordering::Greater) => Ok(v2.clone()),
        None => datafusion_common::internal_err!(
            "cannot compare values of different or incompatible types: {v1:?} vs {v2:?}"
        ),
    }
}

/// Returns the element-wise maximum of two `ScalarValue`s.
///
/// # Null semantics
/// - `max(NULL, NULL)      = NULL`
/// - `max(NULL, x)         = x`
/// - `max(x, NULL)         = x`
///
/// # Errors
/// Returns internal error if v1 and v2 has incompatible types.
fn scalar_max(v1: &ScalarValue, v2: &ScalarValue) -> Result<ScalarValue> {
    if let Some(result) = scalar_cmp_null_short_circuit(v1, v2) {
        return Ok(result);
    }

    match v1.partial_cmp(v2) {
        Some(Ordering::Greater | Ordering::Equal) => Ok(v1.clone()),
        Some(Ordering::Less) => Ok(v2.clone()),
        None => datafusion_common::internal_err!(
            "cannot compare values of different or incompatible types: {v1:?} vs {v2:?}"
        ),
    }
}

fn scalar_cmp_null_short_circuit(
    v1: &ScalarValue,
    v2: &ScalarValue,
) -> Option<ScalarValue> {
    match (v1, v2) {
        (ScalarValue::Null, ScalarValue::Null) => Some(ScalarValue::Null),
        (ScalarValue::Null, other) | (other, ScalarValue::Null) => Some(other.clone()),
        _ => None,
    }
}

/// Prepend the grouping ID column to the output columns if present.
///
/// For GROUPING SETS with no GROUP BY expressions, the schema includes a `__grouping_id`
/// column that must be present in the output. This function inserts it at the beginning
/// of the columns array to maintain schema alignment.
fn prepend_grouping_id_column(
    mut columns: Vec<Arc<dyn Array>>,
    grouping_id: Option<&ScalarValue>,
) -> Result<Vec<Arc<dyn Array>>> {
    if let Some(id) = grouping_id {
        let num_rows = columns.first().map(|array| array.len()).unwrap_or(1);
        let grouping_ids = id.to_array_of_size(num_rows)?;
        columns.insert(0, grouping_ids);
    }
    Ok(columns)
}

impl AggregateStream {
    /// Create a new AggregateStream
    pub fn new(
        agg: &AggregateExec,
        context: &Arc<TaskContext>,
        partition: usize,
    ) -> Result<Self> {
        let agg_schema = Arc::clone(&agg.schema);
        let agg_filter_expr = Arc::clone(&agg.filter_expr);

        let baseline_metrics = BaselineMetrics::new(&agg.metrics, partition);
        let input = agg.input.execute(partition, Arc::clone(context))?;

        let aggregate_expressions = aggregate_expressions(&agg.aggr_expr, &agg.mode, 0)?;
        let filter_expressions = match agg.mode.input_mode() {
            AggregateInputMode::Raw => agg_filter_expr,
            AggregateInputMode::Partial => vec![None; agg.aggr_expr.len()].into(),
        };
        let accumulators = create_accumulators(
            &agg.aggr_expr,
            agg.mode.output_mode() == super::AggregateOutputMode::Final,
        )?;

        let reservation = MemoryConsumer::new(format!("AggregateStream[{partition}]"))
            .register(context.memory_pool());
        let ordered_arguments = if agg.requires_ordered_input() {
            Some(bind_ordered_arguments(
                &aggregate_expressions,
                agg.input.schema().as_ref(),
                &reservation,
            )?)
        } else {
            None
        };
        let ordered_filters = if agg.requires_ordered_input() {
            Some(bind_ordered_filters(
                &filter_expressions,
                agg.input.schema().as_ref(),
                &reservation,
            )?)
        } else {
            None
        };

        // Enable dynamic filter if:
        // 1. AggregateExec did the check and ensure it supports the dynamic filter
        //    (its dynamic_filter field will be Some(..))
        // 2. Aggregate dynamic filter is enabled from the config
        let mut maybe_dynamic_filter = match agg.dynamic_filter.as_ref() {
            Some(filter) => Some(Arc::clone(filter)),
            _ => None,
        };

        if !context
            .session_config()
            .options()
            .optimizer
            .enable_aggregate_dynamic_filter_pushdown
        {
            maybe_dynamic_filter = None;
        }

        let inner = AggregateStreamInner {
            schema: Arc::clone(&agg.schema),
            mode: agg.mode,
            input,
            baseline_metrics,
            aggregate_expressions,
            ordered_arguments,
            ordered_filters,
            ordered_input_evaluator: agg.ordered_input_evaluator,
            options: Arc::clone(context.session_config().options()),
            filter_expressions,
            accumulators,
            reservation,
            finished: false,
            agg_dyn_filter_state: maybe_dynamic_filter,
            _observation: crate::sorts::observation::execution_guard(
                context,
                &agg.input,
                crate::sorts::observation::SortExecutionKind::Aggregate,
                &crate::sorts::observation::SortExecutionMetrics::new(
                    &agg.metrics,
                    partition,
                ),
            ),
        };

        let stream = futures::stream::unfold(inner, |mut this| async move {
            if this.finished {
                return None;
            }

            loop {
                let result = match this.input.next().await {
                    Some(Ok(batch)) => {
                        let result = {
                            let elapsed_compute = this.baseline_metrics.elapsed_compute();
                            let _timer = elapsed_compute.timer(); // Stops on drop
                            if let Some(ordered) = &this.ordered_arguments {
                                aggregate_batch_ordered(
                                    &batch,
                                    &mut this.accumulators,
                                    &this.aggregate_expressions,
                                    ordered,
                                    &this.filter_expressions,
                                    this.ordered_filters
                                        .as_deref()
                                        .expect("same ordered owner"),
                                    &this.options,
                                    &this.reservation,
                                    this.ordered_input_evaluator,
                                )
                                .await
                            } else {
                                aggregate_batch(
                                    &this.mode,
                                    &batch,
                                    &mut this.accumulators,
                                    &this.aggregate_expressions,
                                    &this.filter_expressions,
                                )
                            }
                        };

                        let result = result.and_then(|allocated| {
                            this.maybe_update_dyn_filter()?;
                            Ok(allocated)
                        });

                        // allocate memory
                        // This happens AFTER we actually used the memory, but simplifies the whole accounting and we are OK with
                        // overshooting a bit. Also this means we either store the whole record batch or not.
                        match result
                            .and_then(|allocated| this.reservation.try_grow(allocated))
                        {
                            Ok(_) => continue,
                            Err(e) => Err(e),
                        }
                    }
                    Some(Err(e)) => Err(e),
                    None => {
                        this.finished = true;
                        let timer = this.baseline_metrics.elapsed_compute().timer();
                        let result =
                            finalize_aggregation(&mut this.accumulators, &this.mode)
                                .and_then(|columns| {
                                    prepend_grouping_id_column(columns, None)
                                })
                                .and_then(|columns| {
                                    RecordBatch::try_new(
                                        Arc::clone(&this.schema),
                                        columns,
                                    )
                                    .map_err(Into::into)
                                })
                                .record_output(&this.baseline_metrics);

                        timer.done();

                        result
                    }
                };

                this.finished = true;
                return Some((result, this));
            }
        });

        // seems like some consumers call this stream even after it returned `None`, so let's fuse the stream.
        let stream = stream.fuse();
        let stream = Box::pin(stream);

        Ok(Self {
            schema: agg_schema,
            stream,
        })
    }
}

/// Borrow the same original async physical argument once at stream creation.
/// No expression is rebound, simplified or executed; only the original async
/// invocation's return Field is retained, as AsyncFuncExec previously did.
pub(super) fn bind_ordered_arguments(
    arguments: &[Vec<Arc<dyn PhysicalExpr>>],
    schema: &arrow::datatypes::Schema,
    memory: &MemoryReservation,
) -> Result<Vec<Vec<Option<AsyncFuncExpr>>>> {
    let count = arguments.iter().try_fold(0usize, |count, arguments| {
        count.checked_add(arguments.len()).ok_or_else(|| {
            internal_datafusion_err!("ordered aggregate argument capacity")
        })
    })?;
    let bytes = count
        .checked_mul(size_of::<Option<AsyncFuncExpr>>())
        .and_then(|bytes| {
            bytes.checked_add(
                arguments
                    .len()
                    .checked_mul(size_of::<Vec<Option<AsyncFuncExpr>>>())?,
            )
        })
        .ok_or_else(|| {
            internal_datafusion_err!("ordered aggregate argument allocation")
        })?;
    memory.try_grow(bytes)?;
    arguments
        .iter()
        .map(|arguments| {
            arguments
                .iter()
                .map(|argument| {
                    if let Some(function) = argument
                        .as_any()
                        .downcast_ref::<datafusion_physical_expr::ScalarFunctionExpr>(
                    ) && function.fun().as_async().is_some()
                    {
                        AsyncFuncExpr::try_new(
                            String::new(),
                            Arc::clone(argument),
                            schema,
                        )
                        .map(Some)
                    } else {
                        Ok(None)
                    }
                })
                .collect()
        })
        .collect()
}

pub(super) async fn request_ordered_arguments(
    arguments: &[Arc<dyn PhysicalExpr>],
    asynchronous: &[Option<AsyncFuncExpr>],
    row: &RecordBatch,
    options: &Arc<datafusion_common::config::ConfigOptions>,
    memory: &MemoryReservation,
    evaluator: Option<super::OrderedAggregateInputEvaluator>,
) -> Result<Vec<arrow::array::ArrayRef>> {
    if row.num_rows() != 1 || arguments.len() != asynchronous.len() {
        return internal_err!(
            "ordered aggregate request lost its original row/arguments"
        );
    }
    memory.try_grow(
        arguments
            .len()
            .checked_mul(size_of::<arrow::array::ArrayRef>())
            .ok_or_else(|| {
                internal_datafusion_err!("ordered aggregate row argument allocation")
            })?,
    )?;
    let mut values = Vec::with_capacity(arguments.len());
    for (expression, asynchronous) in arguments.iter().zip(asynchronous) {
        if let Some(evaluate) = evaluator {
            values.push(evaluate(expression, row, options).await?);
            continue;
        }
        let value = if let Some(asynchronous) = asynchronous {
            asynchronous
                .invoke_with_args(row, Arc::clone(options))
                .await?
        } else {
            expression.evaluate(row)?
        };
        values.push(ordered_array(value, memory)?);
    }
    Ok(values)
}

fn ordered_array(
    value: datafusion_expr::ColumnarValue,
    memory: &MemoryReservation,
) -> Result<arrow::array::ArrayRef> {
    match value {
        datafusion_expr::ColumnarValue::Array(array) => Ok(array),
        datafusion_expr::ColumnarValue::Scalar(value) => {
            // Same scalar primary, before the normal ScalarValue -> one-row
            // Arrow conversion. Borrowed arrays are never copied/charged.
            let bytes = value
                .size()
                .checked_mul(2)
                .and_then(|bytes| bytes.checked_add(512))
                .ok_or_else(|| internal_datafusion_err!("ordered scalar array layout"))?;
            memory.try_grow(bytes)?;
            value.to_array_of_size(1)
        }
    }
}

pub(super) fn bind_ordered_filters(
    filters: &[Option<Arc<dyn PhysicalExpr>>],
    schema: &arrow::datatypes::Schema,
    memory: &MemoryReservation,
) -> Result<Vec<Option<AsyncFuncExpr>>> {
    let bytes = std::alloc::Layout::array::<Option<AsyncFuncExpr>>(filters.len())
        .map_err(|_| internal_datafusion_err!("ordered filter layout"))?
        .size();
    memory.try_grow(bytes)?;
    filters
        .iter()
        .map(|filter| {
            if let Some(filter) = filter
                && let Some(function) = filter
                    .as_any()
                    .downcast_ref::<datafusion_physical_expr::ScalarFunctionExpr>(
                )
                && function.fun().as_async().is_some()
            {
                AsyncFuncExpr::try_new(String::new(), Arc::clone(filter), schema)
                    .map(Some)
            } else {
                Ok(None)
            }
        })
        .collect()
}

pub(super) async fn ordered_filter_passes(
    filter: &Option<Arc<dyn PhysicalExpr>>,
    asynchronous: &Option<AsyncFuncExpr>,
    row: &RecordBatch,
    options: &Arc<datafusion_common::config::ConfigOptions>,
    memory: &MemoryReservation,
    evaluator: Option<super::OrderedAggregateInputEvaluator>,
) -> Result<bool> {
    let Some(filter) = filter else {
        return Ok(true);
    };
    let value = if let Some(evaluate) = evaluator {
        datafusion_expr::ColumnarValue::Array(evaluate(filter, row, options).await?)
    } else if let Some(asynchronous) = asynchronous {
        asynchronous
            .invoke_with_args(row, Arc::clone(options))
            .await?
    } else {
        filter.evaluate(row)?
    };
    let value = ordered_array(value, memory)?;
    let value = value
        .as_any()
        .downcast_ref::<arrow::array::BooleanArray>()
        .filter(|value| value.len() == 1)
        .ok_or_else(|| internal_datafusion_err!("ordered filter lost its boolean row"))?;
    Ok(value.is_valid(0) && value.value(0))
}

async fn aggregate_batch_ordered(
    batch: &RecordBatch,
    accumulators: &mut [AccumulatorItem],
    expressions: &[Vec<Arc<dyn PhysicalExpr>>],
    asynchronous: &[Vec<Option<AsyncFuncExpr>>],
    filters: &[Option<Arc<dyn PhysicalExpr>>],
    asynchronous_filters: &[Option<AsyncFuncExpr>],
    options: &Arc<datafusion_common::config::ConfigOptions>,
    reservation: &MemoryReservation,
    evaluator: Option<super::OrderedAggregateInputEvaluator>,
) -> Result<usize> {
    let mut allocated = 0usize;
    for row in 0..batch.num_rows() {
        let row = batch.slice(row, 1);
        for ((((accumulator, expressions), asynchronous), filter), asynchronous_filter) in
            accumulators
                .iter_mut()
                .zip(expressions)
                .zip(asynchronous)
                .zip(filters)
                .zip(asynchronous_filters)
        {
            let scratch = reservation.new_empty();
            if !ordered_filter_passes(
                filter,
                asynchronous_filter,
                &row,
                options,
                &scratch,
                evaluator,
            )
            .await?
            {
                continue;
            }
            let values = request_ordered_arguments(
                expressions,
                asynchronous,
                &row,
                options,
                &scratch,
                evaluator,
            )
            .await?;
            let before = accumulator.size();
            accumulator.update_batch(&values)?;
            allocated += accumulator.size().saturating_sub(before);
        }
    }
    Ok(allocated)
}

impl Stream for AggregateStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        this.stream.poll_next_unpin(cx)
    }
}

impl RecordBatchStream for AggregateStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

/// Perform group-by aggregation for the given [`RecordBatch`].
///
/// If successful, this returns the additional number of bytes that were allocated during this process.
///
/// TODO: Make this a member function
fn aggregate_batch(
    mode: &AggregateMode,
    batch: &RecordBatch,
    accumulators: &mut [AccumulatorItem],
    expressions: &[Vec<Arc<dyn PhysicalExpr>>],
    filters: &[Option<Arc<dyn PhysicalExpr>>],
) -> Result<usize> {
    let mut allocated = 0usize;

    // 1.1 iterate accumulators and respective expressions together
    // 1.2 filter the batch if necessary
    // 1.3 evaluate expressions
    // 1.4 update / merge accumulators with the expressions' values

    // 1.1
    accumulators
        .iter_mut()
        .zip(expressions)
        .zip(filters)
        .try_for_each(|((accum, expr), filter)| {
            // 1.2
            let batch = match filter {
                Some(filter) => Cow::Owned(batch_filter(batch, filter)?),
                None => Cow::Borrowed(batch),
            };

            // 1.3
            let values = evaluate_expressions_to_arrays(expr, batch.as_ref())?;

            // 1.4
            let size_pre = accum.size();
            let res = match mode.input_mode() {
                AggregateInputMode::Raw => accum.update_batch(&values),
                AggregateInputMode::Partial => accum.merge_batch(&values),
            };
            let size_post = accum.size();
            allocated += size_post.saturating_sub(size_pre);
            res
        })?;

    Ok(allocated)
}
