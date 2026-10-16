use super::super::*;
use crate::aggregates::paired_tests::aggregate_exec::{one_row, plan};
use datafusion_execution::memory_pool::{GreedyMemoryPool, MemoryPool};
use datafusion_execution::runtime_env::RuntimeEnvBuilder;

#[test]
fn ordered_binding_layout_remains_charged_across_actual_resize_and_drop() {
    let pool = Arc::new(GreedyMemoryPool::new(1 << 20)) as Arc<dyn MemoryPool>;
    let runtime = RuntimeEnvBuilder::default()
        .with_memory_pool(Arc::clone(&pool))
        .build_arc()
        .unwrap();
    let context = Arc::new(TaskContext::default().with_runtime(runtime));
    let aggregate = plan(true, true, AggregateMode::Single)
        .with_ordered_input_evaluator(Some(one_row));
    let mut stream =
        GroupedHashAggregateStreamInner::new(&aggregate, &context, 0).unwrap();
    let actual = stream.ordered_arguments.as_ref().unwrap();
    let expected = actual.capacity() * size_of::<Vec<Option<AsyncFuncExpr>>>()
        + actual
            .iter()
            .map(|arguments| arguments.capacity() * size_of::<Option<AsyncFuncExpr>>())
            .sum::<usize>()
        + stream.ordered_filters.as_ref().unwrap().capacity()
            * size_of::<Option<AsyncFuncExpr>>();
    assert_eq!(stream.ordered_binding_bytes().unwrap(), expected);
    assert!(expected > 0);
    for capacity in [0, 8, 64, 0] {
        stream.current_group_indices = Vec::with_capacity(capacity);
        stream.update_memory_reservation().unwrap();
        assert!(stream.reservation.size() >= expected);
        assert_eq!(pool.reserved(), stream.reservation.size());
    }
    assert!(matches!(stream.oom_mode, OutOfMemoryMode::ReportError));
    drop(stream);
    assert_eq!(pool.reserved(), 0);
}
#[test]
fn exact_live_ordered_layout_shortfall_is_refused_without_spilling_or_losing_charge() {
    let aggregate = plan(true, true, AggregateMode::Single)
        .with_ordered_input_evaluator(Some(one_row));
    let unbounded = Arc::new(TaskContext::default());
    let mut sizing =
        GroupedHashAggregateStreamInner::new(&aggregate, &unbounded, 0).unwrap();
    sizing.update_memory_reservation().unwrap();
    let required = sizing.reservation.size();
    assert!(required > sizing.ordered_binding_bytes().unwrap());
    drop(sizing);
    let pool = Arc::new(GreedyMemoryPool::new(required)) as Arc<dyn MemoryPool>;
    let runtime = RuntimeEnvBuilder::default()
        .with_memory_pool(Arc::clone(&pool))
        .build_arc()
        .unwrap();
    let context = Arc::new(TaskContext::default().with_runtime(runtime));
    let mut stream =
        GroupedHashAggregateStreamInner::new(&aggregate, &context, 0).unwrap();
    stream.update_memory_reservation().unwrap();
    let held = stream.reservation.size();
    stream.current_group_indices = Vec::with_capacity(64);
    assert!(matches!(
        stream.try_update_memory_reservation().unwrap_err(),
        DataFusionError::ResourcesExhausted(_)
    ));
    assert!(stream.spill_state.spills.is_empty());
    assert_eq!(stream.reservation.size(), held);
    assert_eq!(pool.reserved(), held);
    drop(stream);
    assert_eq!(pool.reserved(), 0);
}
