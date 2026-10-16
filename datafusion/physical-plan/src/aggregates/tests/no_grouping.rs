use super::*;
use arrow::array::{ArrayRef, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion_execution::memory_pool::{GreedyMemoryPool, MemoryPool};
use datafusion_expr::async_udf::{AsyncScalarUDF, AsyncScalarUDFImpl};
use datafusion_expr::{
    Accumulator, ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_expr::expressions::Column;
use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

#[derive(Debug)]
struct Request {
    events: Arc<Mutex<Vec<String>>>,
    label: &'static str,
    filter: bool,
    signature: Signature,
}
impl PartialEq for Request {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.events, &other.events)
            && self.label == other.label
            && self.filter == other.filter
            && self.signature == other.signature
    }
}
impl Eq for Request {}
impl Hash for Request {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.events).hash(state);
        self.label.hash(state);
        self.filter.hash(state);
        self.signature.hash(state);
    }
}

#[test]
fn request_identity_keeps_original_event_owner_and_fixed_roles_not_mutable_events() {
    use std::collections::hash_map::DefaultHasher;

    fn fingerprint(request: &Request) -> u64 {
        let mut hash = DefaultHasher::new();
        request.hash(&mut hash);
        hash.finish()
    }

    let events = Arc::new(Mutex::new(Vec::new()));
    let signature = Signature::exact(vec![DataType::Int64], Volatility::Volatile);
    let make = |events, label, filter, signature| Request {
        events,
        label,
        filter,
        signature,
    };
    let original = make(Arc::clone(&events), "get-a", false, signature.clone());
    let same = make(Arc::clone(&events), "get-a", false, signature.clone());
    assert_eq!(original, same);
    let before = fingerprint(&original);
    assert_eq!(before, fingerprint(&same));

    events.lock().unwrap().push("already observed".into());
    assert_eq!(original, same);
    assert_eq!(fingerprint(&original), before);
    assert_eq!(fingerprint(&same), before);
    // Equal contents in a different physical event owner never unify requests.
    let foreign = Arc::new(Mutex::new(vec!["already observed".into()]));
    assert_ne!(original, make(foreign, "get-a", false, signature.clone()));
    assert_ne!(
        original,
        make(Arc::clone(&events), "get-b", false, signature.clone())
    );
    assert_ne!(
        original,
        make(Arc::clone(&events), "get-a", true, signature.clone())
    );
    assert_ne!(
        original,
        make(
            Arc::clone(&events),
            "get-a",
            false,
            Signature::exact(vec![DataType::Int32], Volatility::Volatile)
        )
    );

    let mut expected = DefaultHasher::new();
    Arc::as_ptr(&events).hash(&mut expected);
    "get-a".hash(&mut expected);
    false.hash(&mut expected);
    signature.hash(&mut expected);
    assert_eq!(before, expected.finish());
    assert_eq!(*events.lock().unwrap(), ["already observed"]);
}

impl ScalarUDFImpl for Request {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        self.label
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(if self.filter {
            DataType::Boolean
        } else {
            DataType::Int64
        })
    }
    fn invoke_with_args(&self, _: ScalarFunctionArgs) -> Result<ColumnarValue> {
        internal_err!("ordered request must use the original async callable")
    }
}
#[async_trait::async_trait]
impl AsyncScalarUDFImpl for Request {
    async fn invoke_async_with_args(
        &self,
        args: ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        assert_eq!(args.number_rows, 1, "no preceding whole-batch getter");
        let value = match &args.args[0] {
            ColumnarValue::Array(value) => ScalarValue::try_from_array(value, 0)?,
            ColumnarValue::Scalar(value) => value.clone(),
        };
        let ScalarValue::Int64(Some(value)) = value else {
            return internal_err!("request's actual row");
        };
        self.events
            .lock()
            .unwrap()
            .push(format!("{}:{value}", self.label));
        Ok(ColumnarValue::Scalar(if self.filter {
            ScalarValue::Boolean(Some(value != 2))
        } else {
            ScalarValue::Int64(Some(value))
        }))
    }
}
#[derive(Debug)]
struct Transition {
    label: &'static str,
    events: Arc<Mutex<Vec<String>>>,
    fail: Option<i64>,
    sum: i64,
}
impl Accumulator for Transition {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        assert_eq!(values[0].len(), 1);
        let ScalarValue::Int64(Some(value)) = ScalarValue::try_from_array(&values[0], 0)?
        else {
            return internal_err!("transition's actual requested primary");
        };
        self.events
            .lock()
            .unwrap()
            .push(format!("{}:{value}", self.label));
        if self.fail == Some(value) {
            return datafusion_common::exec_err!("transition failure {value}");
        }
        self.sum += value;
        Ok(())
    }
    fn merge_batch(&mut self, _: &[ArrayRef]) -> Result<()> {
        internal_err!("no merge in ordered request test")
    }
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![self.evaluate()?])
    }
    fn evaluate(&mut self) -> Result<ScalarValue> {
        Ok(ScalarValue::Int64(Some(self.sum)))
    }
    fn size(&self) -> usize {
        size_of::<Self>()
    }
}
fn argument(
    label: &'static str,
    filter: bool,
    events: &Arc<Mutex<Vec<String>>>,
) -> Arc<dyn PhysicalExpr> {
    let function = Arc::new(
        AsyncScalarUDF::new(Arc::new(Request {
            events: Arc::clone(events),
            label,
            filter,
            signature: Signature::exact(vec![DataType::Int64], Volatility::Volatile),
        }))
        .into_scalar_udf(),
    );
    Arc::new(ScalarFunctionExpr::new(
        label,
        function,
        vec![Arc::new(Column::new("v", 0))],
        Arc::new(Field::new(
            label,
            if filter {
                DataType::Boolean
            } else {
                DataType::Int64
            },
            false,
        )),
        Arc::new(datafusion_common::config::ConfigOptions::default()),
    ))
}
fn batch() -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(vec![1, 2, 3]))],
    )
    .unwrap()
}
fn state(
    label: &'static str,
    fail: Option<i64>,
    events: &Arc<Mutex<Vec<String>>>,
) -> AccumulatorItem {
    Box::new(Transition {
        label,
        events: Arc::clone(events),
        fail,
        sum: 0,
    })
}

#[tokio::test]
async fn ordered_requests_stop_at_the_first_transition_error_before_later_getters() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let row = batch();
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let memory = MemoryConsumer::new("ordered prefix fixture").register(&pool);
    let expressions = vec![
        vec![argument("get-a", false, &events)],
        vec![argument("get-b", false, &events)],
    ];
    let async_arguments =
        bind_ordered_arguments(&expressions, row.schema().as_ref(), &memory).unwrap();
    let filters = vec![None, None];
    let async_filters =
        bind_ordered_filters(&filters, row.schema().as_ref(), &memory).unwrap();
    let mut states = vec![
        state("add-a", Some(2), &events),
        state("add-b", None, &events),
    ];
    let error = aggregate_batch_ordered(
        &row,
        &mut states,
        &expressions,
        &async_arguments,
        &filters,
        &async_filters,
        &Arc::new(datafusion_common::config::ConfigOptions::default()),
        &memory,
        None,
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("transition failure 2"));
    assert_eq!(
        *events.lock().unwrap(),
        [
            "get-a:1", "add-a:1", "get-b:1", "add-b:1", "get-a:2", "add-a:2"
        ]
    );
    assert_eq!(states[0].evaluate().unwrap(), ScalarValue::Int64(Some(1)));
    assert_eq!(states[1].evaluate().unwrap(), ScalarValue::Int64(Some(1)));
    drop(memory);
    assert_eq!(pool.reserved(), 0);
}
#[tokio::test]
async fn ordered_filter_runs_before_its_getter_without_reordering_mixed_aggregates() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let row = batch();
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let memory = MemoryConsumer::new("ordered filter fixture").register(&pool);
    let expressions = vec![
        vec![argument("get-a", false, &events)],
        vec![argument("get-b", false, &events)],
    ];
    let async_arguments =
        bind_ordered_arguments(&expressions, row.schema().as_ref(), &memory).unwrap();
    let filters = vec![Some(argument("filter", true, &events)), None];
    let async_filters =
        bind_ordered_filters(&filters, row.schema().as_ref(), &memory).unwrap();
    let mut states = vec![state("add-a", None, &events), state("add-b", None, &events)];
    aggregate_batch_ordered(
        &row,
        &mut states,
        &expressions,
        &async_arguments,
        &filters,
        &async_filters,
        &Arc::new(datafusion_common::config::ConfigOptions::default()),
        &memory,
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        *events.lock().unwrap(),
        [
            "filter:1", "get-a:1", "add-a:1", "get-b:1", "add-b:1", "filter:2",
            "get-b:2", "add-b:2", "filter:3", "get-a:3", "add-a:3", "get-b:3", "add-b:3"
        ]
    );
    assert_eq!(states[0].evaluate().unwrap(), ScalarValue::Int64(Some(4)));
    assert_eq!(states[1].evaluate().unwrap(), ScalarValue::Int64(Some(6)));
    drop(memory);
    assert_eq!(pool.reserved(), 0);
}
#[test]
fn ordered_argument_binding_budget_refusal_precedes_all_getters_and_transitions() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let row = batch();
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(0));
    let memory = MemoryConsumer::new("ordered zero budget fixture").register(&pool);
    assert!(
        bind_ordered_arguments(
            &[vec![argument("never", false, &events)]],
            row.schema().as_ref(),
            &memory
        )
        .is_err()
    );
    assert!(events.lock().unwrap().is_empty());
    assert_eq!(pool.reserved(), 0);
}
