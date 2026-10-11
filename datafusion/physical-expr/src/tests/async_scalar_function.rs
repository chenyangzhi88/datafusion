use std::any::Any;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use arrow::array::{Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion_common::Result;
use datafusion_common::config::ConfigOptions;
use datafusion_expr::async_udf::{AsyncScalarUDF, AsyncScalarUDFImpl};
use datafusion_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility,
};

use crate::ScalarFunctionExpr;
use crate::async_scalar_function::AsyncFuncExpr;
use crate::expressions::Column;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Output {
    Scalar(Option<i64>),
    Array,
    WrongWidth(usize),
}

#[derive(Debug)]
struct ChunkProbe {
    output: Output,
    ideal: Option<usize>,
    calls: Arc<Mutex<Vec<usize>>>,
    signature: Signature,
}

impl PartialEq for ChunkProbe {
    fn eq(&self, other: &Self) -> bool {
        self.output == other.output
            && self.ideal == other.ideal
            && Arc::ptr_eq(&self.calls, &other.calls)
    }
}
impl Eq for ChunkProbe {}
impl Hash for ChunkProbe {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.output.hash(state);
        self.ideal.hash(state);
        Arc::as_ptr(&self.calls).hash(state);
    }
}
impl ScalarUDFImpl for ChunkProbe {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "actual_chunk_probe"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Int64)
    }
    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<Arc<Field>> {
        Ok(Arc::clone(&args.arg_fields[0]))
    }
    fn invoke_with_args(&self, _: ScalarFunctionArgs) -> Result<ColumnarValue> {
        panic!("actual async entry required")
    }
}
impl AsyncScalarUDFImpl for ChunkProbe {
    fn ideal_batch_size(&self) -> Option<usize> {
        self.ideal
    }

    // Use the trait's boxed async ABI directly: no additional test dependency
    // or production manifest change is needed for this private mirror.
    fn invoke_async_with_args<'life0, 'async_trait>(
        &'life0 self,
        args: ScalarFunctionArgs,
    ) -> Pin<Box<dyn Future<Output = Result<ColumnarValue>> + Send + 'async_trait>>
    where
        'life0: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move {
            self.calls.lock().unwrap().push(args.number_rows);
            tokio::task::yield_now().await;
            Ok(match self.output {
                Output::Scalar(value) => {
                    ColumnarValue::Scalar(datafusion_common::ScalarValue::Int64(value))
                }
                Output::Array => args.args[0].clone(),
                Output::WrongWidth(rows) => {
                    ColumnarValue::Array(Arc::new(Int64Array::from(vec![1; rows])))
                }
            })
        })
    }
}

fn input(values: Vec<Option<i64>>) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "input",
            DataType::Int64,
            true,
        )])),
        vec![Arc::new(Int64Array::from(values))],
    )
    .unwrap()
}

fn expression(
    batch: &RecordBatch,
    output: Output,
    ideal: Option<usize>,
    calls: &Arc<Mutex<Vec<usize>>>,
) -> AsyncFuncExpr {
    let function = AsyncScalarUDF::new(Arc::new(ChunkProbe {
        output,
        ideal,
        calls: Arc::clone(calls),
        signature: Signature::exact(vec![DataType::Int64], Volatility::Volatile),
    }))
    .into_scalar_udf();
    let function = ScalarFunctionExpr::try_new(
        Arc::new(function),
        vec![Arc::new(Column::new("input", 0))],
        batch.schema_ref(),
        Arc::new(ConfigOptions::default()),
    )
    .unwrap();
    AsyncFuncExpr::try_new("output", Arc::new(function), batch.schema_ref()).unwrap()
}

fn values(output: ColumnarValue) -> Vec<Option<i64>> {
    let ColumnarValue::Array(array) = output else {
        panic!("actual async expression publishes its complete result array")
    };
    let array = array.as_any().downcast_ref::<Int64Array>().unwrap();
    (0..array.len())
        .map(|index| (!array.is_null(index)).then(|| array.value(index)))
        .collect()
}

#[tokio::test]
async fn scalar_broadcast_uses_the_actual_full_or_split_chunk_width() {
    let batch = input(vec![Some(1), None, Some(3), Some(4), Some(5)]);
    for (ideal, expected_calls) in [(None, vec![5]), (Some(2), vec![2, 2, 1])] {
        for scalar in [Some(19), None] {
            let calls = Arc::new(Mutex::new(Vec::new()));
            let output = expression(&batch, Output::Scalar(scalar), ideal, &calls)
                .invoke_with_args(&batch, Arc::new(ConfigOptions::default()))
                .await
                .unwrap();
            assert_eq!(values(output), vec![scalar; 5]);
            assert_eq!(*calls.lock().unwrap(), expected_calls);
        }
    }
}

#[tokio::test]
async fn array_results_keep_each_real_chunk_and_batch_in_order() {
    for ideal in [None, Some(2)] {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let first = input(vec![Some(1), None, Some(3), Some(4), Some(5)]);
        let second = input(vec![Some(6), Some(7), None]);
        let function = expression(&first, Output::Array, ideal, &calls);
        let mut actual = Vec::new();
        for batch in [&first, &second] {
            actual.extend(values(
                function
                    .invoke_with_args(batch, Arc::new(ConfigOptions::default()))
                    .await
                    .unwrap(),
            ));
        }
        assert_eq!(
            actual,
            [
                Some(1),
                None,
                Some(3),
                Some(4),
                Some(5),
                Some(6),
                Some(7),
                None
            ]
        );
        assert_eq!(
            *calls.lock().unwrap(),
            if ideal.is_some() {
                vec![2, 2, 1, 2, 1]
            } else {
                vec![5, 3]
            }
        );
    }
}

#[tokio::test]
async fn wrong_array_width_and_zero_ideal_size_fail_without_replaying_a_child() {
    let batch = input(vec![Some(1), Some(2), Some(3), Some(4), Some(5)]);
    for (ideal, width, chunk) in [(None, 1, 5), (Some(2), 1, 2), (Some(2), 3, 2)] {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let error = expression(&batch, Output::WrongWidth(width), ideal, &calls)
            .invoke_with_args(&batch, Arc::new(ConfigOptions::default()))
            .await
            .unwrap_err();
        assert!(
            matches!(error, datafusion_common::DataFusionError::Execution(_)),
            "{error:?}"
        );
        assert!(
            error
                .to_string()
                .contains(&format!("returned {width} rows for a chunk of {chunk}"))
        );
        assert_eq!(*calls.lock().unwrap(), [chunk]);
    }
    let calls = Arc::new(Mutex::new(Vec::new()));
    let error = expression(&batch, Output::Scalar(Some(1)), Some(0), &calls)
        .invoke_with_args(&batch, Arc::new(ConfigOptions::default()))
        .await
        .unwrap_err();
    assert!(
        matches!(error, datafusion_common::DataFusionError::Internal(_)),
        "{error:?}"
    );
    assert!(calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn empty_input_keeps_the_bound_type_without_invoking_any_chunk() {
    let batch = input(Vec::new());
    for ideal in [None, Some(2)] {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let output = expression(&batch, Output::Scalar(None), ideal, &calls)
            .invoke_with_args(&batch, Arc::new(ConfigOptions::default()))
            .await
            .unwrap();
        let ColumnarValue::Array(array) = &output else {
            panic!("actual array")
        };
        assert_eq!(array.data_type(), &DataType::Int64);
        assert!(values(output).is_empty());
        assert!(calls.lock().unwrap().is_empty());
    }
    let calls = Arc::new(Mutex::new(Vec::new()));
    let error = expression(&batch, Output::Scalar(None), Some(0), &calls)
        .invoke_with_args(&batch, Arc::new(ConfigOptions::default()))
        .await
        .unwrap_err();
    assert!(
        matches!(error, datafusion_common::DataFusionError::Internal(_)),
        "{error:?}"
    );
    assert!(calls.lock().unwrap().is_empty());
}
