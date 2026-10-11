use super::super::*;

fn aggregate_fixture() -> (Arc<Schema>, PhysicalGroupBy, Vec<RecordBatch>) {
    let (mut group, batch) = owner(65);
    group.groups = vec![vec![false; 65], vec![true; 65]];
    let schema = batch.schema();
    let batches = (0..128)
        .map(|batch_index| {
            let first = (0..16)
                .map(|row| {
                    let key = batch_index * 16 + row;
                    if key == 0 { None } else { Some(key as u32) }
                })
                .collect::<Vec<_>>();
            let mut arrays = vec![Arc::new(UInt32Array::from(first)) as ArrayRef];
            arrays.extend(
                (1..65).map(|_| Arc::new(UInt32Array::from(vec![None; 16])) as ArrayRef),
            );
            RecordBatch::try_new(schema.clone(), arrays).unwrap()
        })
        .collect();
    (schema, group, batches)
}

fn aggregate_context(limit: usize) -> Arc<datafusion_execution::TaskContext> {
    use datafusion_execution::config::SessionConfig;
    use datafusion_execution::memory_pool::FairSpillPool;
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::new(FairSpillPool::new(limit)))
        .build_arc()
        .unwrap();
    Arc::new(
        datafusion_execution::TaskContext::default()
            .with_session_config(SessionConfig::new().with_batch_size(16))
            .with_runtime(runtime),
    )
}

fn counted_rows(
    batches: &[RecordBatch],
) -> std::collections::BTreeMap<(Vec<u8>, Option<u32>), i64> {
    use arrow::array::Int64Array;
    let mut result = std::collections::BTreeMap::new();
    for batch in batches {
        assert_eq!(batch.num_columns(), 67);
        assert_eq!(batch.schema().field(65).data_type(), &DataType::Binary);
        assert!(!batch.schema().field(65).is_nullable());
        let keys = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let masks = batch
            .column(65)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        let counts = batch
            .column(66)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for row in 0..batch.num_rows() {
            let key = if keys.is_null(row) {
                None
            } else {
                Some(keys.value(row))
            };
            *result.entry((masks.value(row).to_vec(), key)).or_insert(0) +=
                counts.value(row);
        }
    }
    result
}

#[tokio::test]
async fn wide_mask_single_and_final_spill_preserve_the_complete_group_prefix()
-> Result<()> {
    use super::super::super::{AggregateExec, AggregateMode};
    use crate::common::collect;
    use crate::execution_plan::ExecutionPlan;
    use crate::test::TestMemoryExec;
    use datafusion_functions_aggregate::count::count_udaf;
    use datafusion_physical_expr::aggregate::AggregateExprBuilder;
    let (schema, group, batches) = aggregate_fixture();
    let make_input = || -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(TestMemoryExec::try_new(
            &[batches.clone()],
            schema.clone(),
            None,
        )?))
    };
    let aggregates = vec![Arc::new(
        AggregateExprBuilder::new(count_udaf(), vec![lit(1i8)])
            .schema(schema.clone())
            .alias("count")
            .build()?,
    )];
    let normal = aggregate_context(64 << 20);
    let single = Arc::new(AggregateExec::try_new(
        AggregateMode::Single,
        group.clone(),
        aggregates.clone(),
        vec![None],
        make_input()?,
        schema.clone(),
    )?);
    let expected = counted_rows(&collect(single.execute(0, normal.clone())?).await?);
    assert_eq!(expected.len(), 2049);
    assert_eq!(expected.get(&(vec![0; 9], None)), Some(&1));
    let mut all = vec![255; 9];
    all[8] = 1;
    assert_eq!(expected.get(&(all.clone(), None)), Some(&2048));

    let spilling = aggregate_context(128 << 10);
    let spilled_single = Arc::new(AggregateExec::try_new(
        AggregateMode::Single,
        group.clone(),
        aggregates.clone(),
        vec![None],
        make_input()?,
        schema.clone(),
    )?);
    let rows = collect(spilled_single.execute(0, spilling.clone())?).await?;
    assert_eq!(counted_rows(&rows), expected);
    assert!(
        spilled_single.metrics().unwrap().spill_count().unwrap() > 0,
        "Single must exercise actual spill"
    );
    assert_eq!(spilling.memory_pool().reserved(), 0);

    // Final receives already-built partial states; its complete key prefix
    // already contains the bitmap. as_final must not append it a second time.
    let partial = Arc::new(AggregateExec::try_new(
        AggregateMode::Partial,
        group.clone(),
        aggregates.clone(),
        vec![None],
        make_input()?,
        schema.clone(),
    )?);
    let states = collect(partial.execute(0, normal.clone())?).await?;
    let final_input: Arc<dyn ExecutionPlan> =
        Arc::new(TestMemoryExec::try_new(&[states], partial.schema(), None)?);
    let final_exec = Arc::new(AggregateExec::try_new(
        AggregateMode::Final,
        group.as_final(),
        aggregates.clone(),
        vec![None],
        final_input,
        schema.clone(),
    )?);
    let final_rows = collect(final_exec.execute(0, spilling.clone())?).await?;
    assert_eq!(counted_rows(&final_rows), expected);
    assert!(
        final_exec.metrics().unwrap().spill_count().unwrap() > 0,
        "Final must exercise actual spill independently"
    );
    assert_eq!(spilling.memory_pool().reserved(), 0);

    // Partial uses early emission, not disk-spill semantics. Duplicate emitted
    // total-group states are real partials and are added exactly once.
    let early = Arc::new(AggregateExec::try_new(
        AggregateMode::Partial,
        group,
        aggregates,
        vec![None],
        make_input()?,
        schema,
    )?);
    let early_states = collect(early.execute(0, spilling.clone())?).await?;
    assert_eq!(counted_rows(&early_states), expected);
    let mut total_group_states = 0;
    for batch in &early_states {
        let masks = batch
            .column(65)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        total_group_states += (0..batch.num_rows())
            .filter(|row| masks.value(*row) == all.as_slice())
            .count();
    }
    assert!(
        total_group_states > 1,
        "Partial must early-emit repeated total-group states"
    );
    assert_eq!(early.metrics().unwrap().spill_count().unwrap(), 0);
    assert_eq!(spilling.memory_pool().reserved(), 0);
    Ok(())
}

use arrow::array::{Array, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion_common::ScalarValue;
use datafusion_execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
use datafusion_physical_expr::expressions::{Column, lit};

fn owner(keys: usize) -> (PhysicalGroupBy, RecordBatch) {
    let schema = Arc::new(Schema::new(
        (0..keys)
            .map(|i| Field::new(format!("k{i}"), DataType::UInt32, true))
            .collect::<Vec<_>>(),
    ));
    let expr = (0..keys)
        .map(|i| {
            (
                Arc::new(Column::new(&format!("k{i}"), i)) as _,
                format!("k{i}"),
            )
        })
        .collect();
    let null_expr = (0..keys)
        .map(|i| (lit(ScalarValue::UInt32(None)), format!("k{i}")))
        .collect();
    let groups = (0..=keys)
        .map(|included| (0..keys).map(|i| i >= included).collect())
        .collect();
    let batch = RecordBatch::try_new(
        schema,
        (0..keys)
            .map(|_| Arc::new(UInt32Array::from(vec![Some(7), None])) as ArrayRef)
            .collect(),
    )
    .unwrap();
    (PhysicalGroupBy::new(expr, null_expr, groups, true), batch)
}

#[test]
fn masks_keep_old_primitives_and_exact_wide_bit_orientation() {
    for keys in [
        0usize, 1, 8, 9, 16, 17, 32, 33, 64, 65, 66, 127, 128, 255, 256, 257,
    ] {
        let mut group = vec![false; keys];
        for i in 0..keys {
            group[i] = i == 0 || i == keys / 2 || i == keys - 1;
        }
        let array = group_id_array(&group, 3).unwrap();
        assert_eq!(
            array.data_type(),
            &datafusion_expr::Aggregate::grouping_id_type(keys)
        );
        assert_eq!((array.len(), array.null_count()), (3, 0));
        if keys <= 64 {
            let expected = group
                .iter()
                .fold(0u64, |acc, excluded| (acc << 1) | u64::from(*excluded));
            let actual = if keys <= 8 {
                u64::from(
                    array
                        .as_any()
                        .downcast_ref::<UInt8Array>()
                        .unwrap()
                        .value(0),
                )
            } else if keys <= 16 {
                u64::from(
                    array
                        .as_any()
                        .downcast_ref::<UInt16Array>()
                        .unwrap()
                        .value(0),
                )
            } else if keys <= 32 {
                u64::from(
                    array
                        .as_any()
                        .downcast_ref::<UInt32Array>()
                        .unwrap()
                        .value(0),
                )
            } else {
                array
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap()
                    .value(0)
            };
            assert_eq!(actual, expected);
        } else {
            let values = array.as_any().downcast_ref::<BinaryArray>().unwrap();
            for row in 0..3 {
                let bytes = values.value(row);
                assert_eq!(bytes.len(), keys.div_ceil(8));
                for (index, excluded) in group.iter().enumerate() {
                    let bit = keys - 1 - index;
                    assert_eq!((bytes[bit / 8] & (1 << (bit % 8))) != 0, *excluded);
                }
                let used = keys % 8;
                if used != 0 {
                    assert_eq!(bytes[bytes.len() - 1] >> used, 0);
                }
            }
            assert_eq!(values.value(0), values.value(2));
        }
    }
    let empty = group_id_array(&vec![true; 65], 0).unwrap();
    let empty = empty.as_any().downcast_ref::<BinaryArray>().unwrap();
    assert_eq!(empty.len(), 0);
    assert_eq!(empty.value_offsets(), &[0]);
    assert!(empty.values().is_empty());
}

#[test]
fn real_evaluator_admits_scratch_before_arrays_and_retains_charge_until_caller_drop() {
    let (group_by, batch) = owner(65);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let parent = MemoryConsumer::new("actual aggregate").register(&pool);
    let scratch = parent.new_empty();
    let expected = (2 * 9 + 3 * 4) * 66;
    let arrays =
        super::super::super::evaluate_group_by(&group_by, &batch, Some(&scratch))
            .unwrap();
    assert_eq!(scratch.size(), expected);
    assert_eq!(pool.reserved(), expected);
    assert_eq!(arrays.len(), 66);
    for (included, values) in arrays.iter().enumerate() {
        assert_eq!(values.len(), 66);
        let mask = values[65].as_any().downcast_ref::<BinaryArray>().unwrap();
        for key in 0..65 {
            let bit = 64 - key;
            assert_eq!(
                (mask.value(0)[bit / 8] & (1 << (bit % 8))) != 0,
                key >= included
            );
        }
        // A real input NULL is not marked as rolled out in the full group.
        if included == 65 {
            assert!(values[0].is_null(1));
            assert!(mask.value(1).iter().all(|byte| *byte == 0));
        }
    }
    drop(arrays);
    assert_eq!(pool.reserved(), expected);
    drop(scratch);
    assert_eq!(pool.reserved(), 0);
    assert_eq!(parent.size(), 0);

    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(expected - 1));
    let scratch = MemoryConsumer::new("actual rejected aggregate").register(&pool);
    let error = super::super::super::evaluate_group_by(&group_by, &batch, Some(&scratch))
        .unwrap_err();
    assert!(matches!(error, DataFusionError::ResourcesExhausted(_)));
    assert_eq!((pool.reserved(), scratch.size()), (0, 0));
    assert!(matches!(
        super::super::super::evaluate_group_by(&group_by, &batch, None),
        Err(DataFusionError::Internal(_))
    ));
}

#[test]
fn zero_rows_shape_and_spill_final_prefix_have_one_mask_and_no_budget_escape() {
    let (group_by, batch) = owner(65);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(4096));
    let scratch = MemoryConsumer::new("empty bitmap").register(&pool);
    let empty = batch.slice(0, 0);
    let arrays =
        super::super::super::evaluate_group_by(&group_by, &empty, Some(&scratch))
            .unwrap();
    assert_eq!(scratch.size(), 66 * 4);
    assert!(arrays.iter().all(|values| values[65].len() == 0));
    drop(arrays);
    drop(scratch);
    assert_eq!(pool.reserved(), 0);
    let schema = group_by.group_schema(batch.schema().as_ref()).unwrap();
    assert_eq!(
        (
            schema.fields().len(),
            schema.field(65).data_type(),
            schema.field(65).is_nullable()
        ),
        (66, &DataType::Binary, false)
    );
    let final_group = group_by.as_final();
    assert!(final_group.is_single());
    assert_eq!(final_group.expr().len(), 66);
    let second_final = final_group.as_final();
    assert_eq!(second_final.expr().len(), 66);
    let prefix_scratch = MemoryConsumer::new("prefix").register(&pool);
    let mut expanded =
        super::super::super::evaluate_group_by(&group_by, &batch, Some(&prefix_scratch))
            .unwrap();
    let input = RecordBatch::try_new(schema.clone(), expanded.pop().unwrap()).unwrap();
    // No newly encoded masks in a final/spill merge; bytes remain borrowed input.
    let final_arrays =
        super::super::super::evaluate_group_by(&final_group, &input, None).unwrap();
    assert_eq!(final_arrays[0][65], input.column(65).clone());
    assert_eq!(
        final_group
            .group_schema(schema.as_ref())
            .unwrap()
            .field(65)
            .data_type(),
        &DataType::Binary
    );
    drop(final_arrays);
    drop(input);
    drop(expanded);
    drop(prefix_scratch);
    assert_eq!(pool.reserved(), 0);
    assert!(matches!(
        mask_buffer_bytes(usize::MAX, 1),
        Err(DataFusionError::ResourcesExhausted(_))
    ));
    assert!(matches!(
        mask_buffer_bytes(65, i32::MAX as usize),
        Err(DataFusionError::ResourcesExhausted(_))
    ));
    let mut malformed = group_by.clone();
    malformed.groups[0].pop();
    assert!(matches!(
        admit_grouping_masks(&malformed, 2, None),
        Err(DataFusionError::Internal(_))
    ));
}
