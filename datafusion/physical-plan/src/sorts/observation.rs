// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file distributed
// with this work for additional information regarding copyright ownership.
// The ASF licenses this file to you under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with it.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Optional execution attribution over the sorter's actual metric owners.
//! No observer changes sorting, memory admission, buffering or scheduling.

use std::fmt::Debug;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion_common::Result;
use datafusion_common::config::{ConfigEntry, ConfigExtension, ExtensionOptions};
use datafusion_execution::TaskContext;
use futures::Stream;

use crate::metrics::{Count, ExecutionPlanMetricsSet, MetricBuilder};
use crate::{ExecutionPlan, RecordBatchStream, SendableRecordBatchStream};

/// Each actual execution/partition creates distinct counters in its existing
/// MetricsSet. Clones share those same atomic owners, never replay totals.
#[derive(Clone, Debug)]
pub struct SortExecutionMetrics {
    pub starts: Count,
    pub input_rows: Count,
    pub merge_passes: Count,
    pub input_rows_overflow: Count,
    pub successful_rows: Count,
    pub succeeded: Count,
    pub failed: Count,
    /// Files actually created by this selected temporary spill owner.
    pub temporary_files: Count,
}

impl SortExecutionMetrics {
    pub fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self::with_merge_counter(
            metrics,
            partition,
            MetricBuilder::new(metrics).counter("sort_merge_passes", partition),
        )
    }

    pub(super) fn with_merge_counter(
        metrics: &ExecutionPlanMetricsSet,
        partition: usize,
        merge_passes: Count,
    ) -> Self {
        Self {
            starts: MetricBuilder::new(metrics).counter("sort_starts", partition),
            input_rows: MetricBuilder::new(metrics).counter("sort_input_rows", partition),
            merge_passes,
            input_rows_overflow: MetricBuilder::new(metrics)
                .counter("sort_input_rows_overflow", partition),
            successful_rows: MetricBuilder::new(metrics).counter("sort_successful_rows", partition),
            succeeded: MetricBuilder::new(metrics).counter("sort_succeeded", partition),
            failed: MetricBuilder::new(metrics).counter("sort_failed", partition),
            temporary_files: MetricBuilder::new(metrics)
                .counter("execution_temporary_files", partition),
        }
    }

    /// One input reader owns each execution/partition group. Preserve its
    /// actual overflow fact separately instead of publishing a wrapped zero.
    pub fn record_input_rows(&self, rows: usize) {
        if self.input_rows.value().checked_add(rows).is_none() {
            self.input_rows_overflow.add(1);
        }
        self.input_rows.add(rows);
    }

    /// Publish only at an actual successful sort frontier, not input read or
    /// final SQL output. `fetch` is this sort factory's max_rows (with any
    /// pushed OFFSET), never a count of emitted/consumed result rows.
    pub fn finish_success(&self, fetch: Option<usize>) {
        if self.succeeded.value() == 0 && self.failed.value() == 0 {
            self.successful_rows
                .add(fetch.map_or(self.input_rows.value(), |max_rows| {
                    self.input_rows.value().min(max_rows)
                }));
            self.succeeded.add(1);
        }
    }
}

/// Lazy sort/merge completion is proven only by actual EOF. An error is a
/// failed sort, while abandoning the stream leaves completion unproven.
pub fn observe_sort_output(
    stream: SendableRecordBatchStream,
    metrics: SortExecutionMetrics,
    fetch: Option<usize>,
) -> SendableRecordBatchStream {
    Box::pin(SortOutputStream {
        stream,
        metrics,
        fetch,
        done: false,
    })
}

struct SortOutputStream {
    stream: SendableRecordBatchStream,
    metrics: SortExecutionMetrics,
    fetch: Option<usize>,
    done: bool,
}
impl Stream for SortOutputStream {
    type Item = Result<RecordBatch>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let result = self.stream.as_mut().poll_next(cx);
        if !self.done {
            match &result {
                Poll::Ready(None) => {
                    self.metrics.finish_success(self.fetch);
                    self.done = true;
                }
                Poll::Ready(Some(Err(_))) => {
                    self.metrics.failed.add(1);
                    self.done = true;
                }
                _ => {}
            }
        }
        result
    }
}
impl RecordBatchStream for SortOutputStream {
    fn schema(&self) -> SchemaRef {
        self.stream.schema()
    }
}

#[derive(Clone, Copy, Debug)]
pub enum SortExecutionKind {
    Sort,
    MergeOnly,
    /// The concrete aggregate stream, including its optional spill sorts.
    Aggregate,
}

/// The final shared guard consumes counters before releasing its publisher.
/// Only actual futures/streams/workers retain it; never a plan or Session.
pub trait SortExecutionGuard: Debug + Send + Sync {
    fn begin_poll(&self) {}
    fn end_poll(&self) {}
}

pub trait SortExecutionObserver: Debug + Send + Sync {
    fn producer(
        &self,
        input: &Arc<dyn ExecutionPlan>,
        kind: SortExecutionKind,
        metrics: SortExecutionMetrics,
    ) -> Option<Arc<dyn SortExecutionGuard>>;
}

#[derive(Clone, Debug)]
pub struct SortObservationOptions(pub Arc<dyn SortExecutionObserver>);
impl ConfigExtension for SortObservationOptions {
    const PREFIX: &'static str = "execution_sort_observer";
}
impl ExtensionOptions for SortObservationOptions {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn cloned(&self) -> Box<dyn ExtensionOptions> {
        Box::new(self.clone())
    }
    fn set(&mut self, _: &str, _: &str) -> Result<()> {
        datafusion_common::config_err!("sort observer is not a configurable setting")
    }
    fn entries(&self) -> Vec<ConfigEntry> {
        Vec::new()
    }
}

pub fn execution_guard(
    context: &TaskContext,
    input: &Arc<dyn ExecutionPlan>,
    kind: SortExecutionKind,
    metrics: &SortExecutionMetrics,
) -> Option<Arc<dyn SortExecutionGuard>> {
    context
        .session_config()
        .options()
        .extensions
        .get::<SortObservationOptions>()?
        .0
        .producer(input, kind, metrics.clone())
}

/// Carry the same guard into buffered worker inputs as well as their caller.
/// Drop the underlying stream before final guard release, including terminal
/// error/EOF; an abandoned worker can retain its own real guard until abort.
pub fn retain_guard(
    stream: SendableRecordBatchStream,
    guard: Option<Arc<dyn SortExecutionGuard>>,
) -> SendableRecordBatchStream {
    let Some(guard) = guard else {
        return stream;
    };
    Box::pin(GuardedSortStream {
        schema: stream.schema(),
        stream: Some(stream),
        guard: Some(guard),
    })
}

pub fn observe_sort_input(
    stream: SendableRecordBatchStream,
    metrics: SortExecutionMetrics,
) -> SendableRecordBatchStream {
    Box::pin(SortInputStream {
        stream,
        metrics,
        started: false,
    })
}

struct SortInputStream {
    stream: SendableRecordBatchStream,
    metrics: SortExecutionMetrics,
    started: bool,
}
impl Stream for SortInputStream {
    type Item = Result<RecordBatch>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if !self.started {
            self.started = true;
            self.metrics.starts.add(1);
        }
        let result = self.stream.as_mut().poll_next(cx);
        if let Poll::Ready(Some(Ok(batch))) = &result {
            self.metrics.record_input_rows(batch.num_rows());
        }
        result
    }
}
impl RecordBatchStream for SortInputStream {
    fn schema(&self) -> SchemaRef {
        self.stream.schema()
    }
}

struct GuardedSortStream {
    schema: SchemaRef,
    stream: Option<SendableRecordBatchStream>,
    guard: Option<Arc<dyn SortExecutionGuard>>,
}

struct SortPollScope<'a>(&'a dyn SortExecutionGuard);
impl Drop for SortPollScope<'_> {
    fn drop(&mut self) {
        self.0.end_poll();
    }
}
impl Stream for GuardedSortStream {
    type Item = Result<RecordBatch>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let guard = self.guard.clone();
        let scope = guard.as_ref().map(|guard| {
            guard.begin_poll();
            SortPollScope(guard.as_ref())
        });
        let Some(stream) = self.stream.as_mut() else {
            return Poll::Ready(None);
        };
        let result = stream.as_mut().poll_next(cx);
        // Include the actual worker's synchronous prefix before final guard
        // release can wake its publisher, also on Pending/error/panic unwind.
        drop(scope);
        drop(guard);
        if matches!(&result, Poll::Ready(None) | Poll::Ready(Some(Err(_)))) {
            self.stream.take();
            self.guard.take();
        }
        result
    }
}
impl RecordBatchStream for GuardedSortStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}
