use super::super::multi_level_merge::MultiLevelMergeBuilder;
use super::super::observation::{SortExecutionGuard, retain_guard};
use super::super::streaming_merge::{SortedSpillFile, StreamingMergeBuilder};
use crate::expressions::{Column, PhysicalSortExpr};
use crate::metrics::{
    BaselineMetrics, Count, ExecutionPlanMetricsSet, MetricBuilder, SpillMetrics,
};
use crate::stream::RecordBatchStreamAdapter;
use crate::{RecordBatchStream, SendableRecordBatchStream, SpillManager};
use arrow::array::Int64Array;
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion_common::DataFusionError;
use datafusion_execution::memory_pool::MemoryConsumer;
use datafusion_execution::runtime_env::RuntimeEnvBuilder;
use datafusion_physical_expr_common::sort_expr::LexOrdering;
use futures::{FutureExt, Stream, StreamExt};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]))
}
fn ordering() -> LexOrdering {
    LexOrdering::new(vec![PhysicalSortExpr::new(
        Arc::new(Column::new("v", 0)),
        SortOptions::default(),
    )])
    .unwrap()
}
fn rows(values: Vec<i64>) -> RecordBatch {
    RecordBatch::try_new(schema(), vec![Arc::new(Int64Array::from(values))]).unwrap()
}
fn stream(values: Vec<i64>) -> SendableRecordBatchStream {
    Box::pin(RecordBatchStreamAdapter::new(
        schema(),
        futures::stream::iter(vec![Ok(rows(values))]),
    ))
}
fn merge(inputs: Vec<SendableRecordBatchStream>, count: Count) -> SendableRecordBatchStream {
    let runtime = RuntimeEnvBuilder::new().build_arc().unwrap();
    let metrics = ExecutionPlanMetricsSet::new();
    StreamingMergeBuilder::new()
        .with_streams(inputs)
        .with_schema(schema())
        .with_expressions(&ordering())
        .with_metrics(BaselineMetrics::new(&metrics, 0))
        .with_batch_size(4)
        .with_reservation(MemoryConsumer::new("test merge").register(&runtime.memory_pool))
        .with_merge_pass_counter(count)
        .build()
        .unwrap()
}

#[tokio::test]
async fn successful_sort_rows_are_capped_at_factory_fetch_and_incomplete_or_failed_input_is_not_published()
 {
    use super::super::observation::{SortExecutionMetrics, observe_sort_output};
    let metrics = ExecutionPlanMetricsSet::new();
    let success = SortExecutionMetrics::new(&metrics, 0);
    success.starts.add(1);
    success.record_input_rows(7);
    let mut output = observe_sort_output(stream(vec![1, 2, 3]), success.clone(), Some(3));
    assert_eq!(success.succeeded.value(), 0);
    assert_eq!(output.next().await.unwrap().unwrap().num_rows(), 3);
    assert_eq!(
        success.succeeded.value(),
        0,
        "output row count is not completion"
    );
    assert!(output.next().await.is_none());
    assert!(output.next().await.is_none());
    assert_eq!(success.succeeded.value(), 1);
    assert_eq!(success.successful_rows.value(), 3);
    assert_eq!(success.input_rows.value(), 7);

    let abandoned = SortExecutionMetrics::new(&metrics, 1);
    abandoned.starts.add(1);
    abandoned.record_input_rows(7);
    let mut output = observe_sort_output(stream(vec![1, 2, 3]), abandoned.clone(), Some(3));
    assert!(output.next().await.unwrap().is_ok());
    drop(output);
    assert_eq!(abandoned.succeeded.value(), 0);
    assert_eq!(abandoned.failed.value(), 0);
    assert_eq!(abandoned.input_rows.value(), 7);

    let failed = SortExecutionMetrics::new(&metrics, 2);
    failed.starts.add(1);
    failed.record_input_rows(3);
    let error_stream = Box::pin(RecordBatchStreamAdapter::new(
        schema(),
        futures::stream::iter([Err(DataFusionError::Execution(
            "real sort reader error".into(),
        ))]),
    ));
    let mut output = observe_sort_output(error_stream, failed.clone(), None);
    assert!(
        output
            .next()
            .await
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("real sort reader error")
    );
    assert!(output.next().await.is_none());
    assert_eq!(failed.failed.value(), 1);
    assert_eq!(failed.succeeded.value(), 0);
    assert_eq!(failed.successful_rows.value(), 0);
    assert_eq!(failed.input_rows.value(), 3);
}

#[tokio::test]
async fn input_row_overflow_is_a_real_counter_fact_not_a_wrapped_zero() {
    let metrics = ExecutionPlanMetricsSet::new();
    let counters = super::super::observation::SortExecutionMetrics::new(&metrics, 0);
    let empty_schema = Arc::new(Schema::empty());
    let batches = [usize::MAX, 1].map(|rows| {
        RecordBatch::try_new_with_options(
            empty_schema.clone(),
            vec![],
            &arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(rows)),
        )
        .unwrap()
    });
    let source = Box::pin(RecordBatchStreamAdapter::new(
        empty_schema,
        futures::stream::iter(batches.into_iter().map(Ok)),
    ));
    let mut input = super::super::observation::observe_sort_input(source, counters.clone());
    assert_eq!(counters.starts.value(), 0);
    assert_eq!(input.next().await.unwrap().unwrap().num_rows(), usize::MAX);
    assert_eq!(input.next().await.unwrap().unwrap().num_rows(), 1);
    assert!(input.next().await.is_none());
    assert_eq!(counters.starts.value(), 1);
    assert_eq!(counters.input_rows_overflow.value(), 1);
    assert_eq!(counters.input_rows.value(), 0);
}

#[tokio::test]
async fn merge_counter_begins_at_real_poll_not_builder_and_keeps_pending_error_drop_prefix() {
    let metrics = ExecutionPlanMetricsSet::new();
    let count = MetricBuilder::new(&metrics).counter("sort_merge_passes", 0);
    let pending = Box::pin(RecordBatchStreamAdapter::new(
        schema(),
        futures::stream::pending(),
    ));
    let mut output = merge(vec![pending, stream(vec![1, 3])], count.clone());
    assert_eq!(count.value(), 0);
    assert!(output.next().now_or_never().is_none());
    assert_eq!(count.value(), 1);
    assert!(output.next().now_or_never().is_none());
    drop(output);
    assert_eq!(count.value(), 1);

    let error_count = MetricBuilder::new(&metrics).counter("sort_merge_passes", 1);
    let failing = Box::pin(RecordBatchStreamAdapter::new(
        schema(),
        futures::stream::iter(vec![Err(DataFusionError::Execution(
            "actual reader failure".into(),
        ))]),
    ));
    let mut output = merge(vec![failing, stream(vec![2])], error_count.clone());
    assert_eq!(error_count.value(), 0);
    assert!(
        output
            .next()
            .await
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("actual reader failure")
    );
    drop(output);
    assert_eq!(error_count.value(), 1);
}

#[tokio::test]
async fn real_sorted_run_merge_counts_passes_not_spills_and_direct_replay_is_zero() {
    for (files, limited) in [(1, false), (6, false), (6, true)] {
        let batch = rows((0..64).collect());
        let batch_bytes = crate::spill::get_record_batch_memory_size(&batch);
        let runtime = if limited {
            // Two one-batch disk inputs fit, six cannot. The existing merge
            // chooser must produce intermediate then final actual passes.
            RuntimeEnvBuilder::new()
                .with_memory_limit(batch_bytes * 4, 1.0)
                .build_arc()
                .unwrap()
        } else {
            RuntimeEnvBuilder::new().build_arc().unwrap()
        };
        let metrics = ExecutionPlanMetricsSet::new();
        let count = MetricBuilder::new(&metrics).counter("sort_merge_passes", 0);
        let spills = SpillMetrics::new(&metrics, 0);
        let manager = SpillManager::new(runtime.clone(), spills.clone(), schema());
        let runs = (0..files)
            .map(|_| SortedSpillFile {
                file: manager
                    .spill_record_batch_and_finish(&[batch.clone()], "actual test sorted run")
                    .unwrap()
                    .unwrap(),
                max_record_batch_memory: batch_bytes,
            })
            .collect();
        assert_eq!(spills.spill_file_count.value(), files);
        let mut output = MultiLevelMergeBuilder::new(
            manager,
            schema(),
            runs,
            vec![],
            ordering(),
            BaselineMetrics::new(&metrics, 0),
            4,
            MemoryConsumer::new("real multilevel merge").register(&runtime.memory_pool),
            None,
            true,
            Some(count.clone()),
        )
        .create_spillable_merge_stream();
        assert_eq!(count.value(), 0);
        let mut values = Vec::new();
        while let Some(batch) = output.next().await {
            let batch = batch.unwrap();
            values.extend(
                batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied(),
            );
        }
        drop(output);
        assert_eq!(values.len(), files * 64);
        assert!(values.windows(2).all(|pair| pair[0] <= pair[1]));
        if files == 1 {
            assert_eq!(count.value(), 0);
        } else if limited {
            assert!(count.value() > 1);
        } else {
            assert_eq!(count.value(), 1);
        }
        assert_eq!(runtime.memory_pool.reserved(), 0);
    }

    // Equal zero spill counts, different actual merges. This cannot be
    // implemented as a formula over spill_file_count.
    for inputs in [1, 2] {
        let runtime = RuntimeEnvBuilder::new().build_arc().unwrap();
        let metrics = ExecutionPlanMetricsSet::new();
        let count = MetricBuilder::new(&metrics).counter("sort_merge_passes", 0);
        let spills = SpillMetrics::new(&metrics, 0);
        let manager = SpillManager::new(runtime.clone(), spills.clone(), schema());
        let mut output = MultiLevelMergeBuilder::new(
            manager,
            schema(),
            vec![],
            (0..inputs).map(|_| stream(vec![1, 2])).collect(),
            ordering(),
            BaselineMetrics::new(&metrics, 0),
            4,
            MemoryConsumer::new("direct replay or memory merge").register(&runtime.memory_pool),
            None,
            true,
            Some(count.clone()),
        )
        .create_spillable_merge_stream();
        while let Some(batch) = output.next().await {
            batch.unwrap();
        }
        drop(output);
        assert_eq!(spills.spill_file_count.value(), 0);
        assert_eq!(count.value(), inputs - 1);
    }
}

struct PendingInput {
    started: Option<tokio::sync::oneshot::Sender<()>>,
    dropped: Arc<AtomicBool>,
}
impl Stream for PendingInput {
    type Item = datafusion_common::Result<RecordBatch>;
    fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(started) = self.started.take() {
            let _ = started.send(());
        }
        Poll::Pending
    }
}
impl RecordBatchStream for PendingInput {
    fn schema(&self) -> SchemaRef {
        schema()
    }
}
impl Drop for PendingInput {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::Release);
    }
}

#[derive(Debug)]
struct FinalGuard {
    count: Count,
    input_dropped: Arc<AtomicBool>,
    polls_started: AtomicUsize,
    polls_finished: AtomicUsize,
    finished: Mutex<Option<tokio::sync::oneshot::Sender<(usize, bool, bool)>>>,
}
impl SortExecutionGuard for FinalGuard {
    fn begin_poll(&self) {
        self.polls_started.fetch_add(1, Ordering::AcqRel);
    }
    fn end_poll(&self) {
        self.polls_finished.fetch_add(1, Ordering::AcqRel);
    }
}
impl Drop for FinalGuard {
    fn drop(&mut self) {
        if let Some(finished) = self.finished.get_mut().unwrap().take() {
            let _ = finished.send((
                self.count.value(),
                self.input_dropped.load(Ordering::Acquire),
                self.polls_finished.load(Ordering::Acquire) != 0
                    && self.polls_finished.load(Ordering::Acquire)
                        == self.polls_started.load(Ordering::Acquire),
            ));
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn buffered_worker_retains_actual_guard_until_pending_merge_input_has_dropped() {
    let metrics = ExecutionPlanMetricsSet::new();
    let count = MetricBuilder::new(&metrics).counter("sort_merge_passes", 0);
    let dropped = Arc::new(AtomicBool::new(false));
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (finished_tx, finished_rx) = tokio::sync::oneshot::channel();
    let guard: Arc<dyn SortExecutionGuard> = Arc::new(FinalGuard {
        count: count.clone(),
        input_dropped: dropped.clone(),
        polls_started: AtomicUsize::new(0),
        polls_finished: AtomicUsize::new(0),
        finished: Mutex::new(Some(finished_tx)),
    });
    let pending = Box::pin(PendingInput {
        started: Some(started_tx),
        dropped: dropped.clone(),
    });
    let input = merge(vec![pending, stream(vec![1])], count.clone());
    let output = crate::common::spawn_buffered(retain_guard(input, Some(guard.clone())), 1);
    tokio::time::timeout(std::time::Duration::from_secs(5), started_rx)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(count.value(), 1);
    drop(guard);
    assert!(!dropped.load(Ordering::Acquire));
    drop(output); // Actual receiver aborts its existing worker.
    assert_eq!(
        tokio::time::timeout(std::time::Duration::from_secs(5), finished_rx)
            .await
            .unwrap()
            .unwrap(),
        (1, true, true)
    );
}
