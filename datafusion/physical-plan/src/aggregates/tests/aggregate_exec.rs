use crate::aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy};
use crate::{ExecutionPlan, test::TestMemoryExec};
use arrow::array::{ArrayRef, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion_common::{Result, ScalarValue, config::ConfigOptions};
use datafusion_execution::{TaskContext, memory_pool::UnboundedMemoryPool};
use datafusion_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion_expr::{
    Accumulator, AggregateUDF, AggregateUDFImpl, Signature, Volatility,
};
use datafusion_physical_expr::{
    PhysicalExpr, aggregate::AggregateExprBuilder, expressions::col,
};
use futures::future::BoxFuture;
use std::any::Any;
use std::sync::Arc;

#[derive(Debug, PartialEq, Eq, Hash)]
struct OrderedSum {
    signature: Signature,
    ordered: bool,
}
impl AggregateUDFImpl for OrderedSum {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "finite_ordered_sum"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Int64)
    }
    fn requires_ordered_input(&self) -> bool {
        self.ordered
    }
    fn accumulator(&self, _: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(Sum(0)))
    }
    fn state_fields(
        &self,
        args: StateFieldsArgs,
    ) -> Result<Vec<arrow::datatypes::FieldRef>> {
        Ok(vec![Arc::new(Field::new(
            format!("{}_sum", args.name),
            DataType::Int64,
            false,
        ))])
    }
}
#[derive(Debug)]
struct Sum(i64);
impl Accumulator for Sum {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        for row in 0..values[0].len() {
            if let ScalarValue::Int64(Some(value)) =
                ScalarValue::try_from_array(&values[0], row)?
            {
                self.0 += value;
            }
        }
        Ok(())
    }
    fn merge_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        self.update_batch(values)
    }
    fn evaluate(&mut self) -> Result<ScalarValue> {
        Ok(ScalarValue::Int64(Some(self.0)))
    }
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![self.evaluate()?])
    }
    fn size(&self) -> usize {
        size_of::<Self>()
    }
}
pub(in crate::aggregates) fn one_row<'a>(
    expression: &'a Arc<dyn PhysicalExpr>,
    row: &'a RecordBatch,
    _: &'a Arc<ConfigOptions>,
) -> BoxFuture<'a, Result<ArrayRef>> {
    Box::pin(async move {
        assert_eq!(row.num_rows(), 1);
        tokio::task::yield_now().await;
        expression.evaluate(row)?.into_array(1)
    })
}
pub(in crate::aggregates) fn plan(
    grouped: bool,
    ordered: bool,
    mode: AggregateMode,
) -> AggregateExec {
    let schema = Arc::new(Schema::new(vec![
        Field::new("g", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 1, 2])),
            Arc::new(Int64Array::from(vec![2, 3, 4])),
        ],
    )
    .unwrap();
    let input = Arc::new(
        TestMemoryExec::try_new(&[vec![batch]], Arc::clone(&schema), None).unwrap(),
    );
    let function = Arc::new(AggregateUDF::from(OrderedSum {
        signature: Signature::exact(vec![DataType::Int64], Volatility::Volatile),
        ordered,
    }));
    let expression = Arc::new(
        AggregateExprBuilder::new(function, vec![col("v", &schema).unwrap()])
            .schema(Arc::clone(&schema))
            .alias("s")
            .build()
            .unwrap(),
    );
    AggregateExec::try_new(
        mode,
        PhysicalGroupBy::new_single(if grouped {
            vec![(col("g", &schema).unwrap(), "g".into())]
        } else {
            vec![]
        }),
        vec![expression],
        vec![None],
        input,
        schema,
    )
    .unwrap()
}
fn same(actual: &AggregateExec) {
    assert!(std::ptr::fn_addr_eq(
        actual.ordered_input_evaluator().unwrap(),
        one_row as crate::aggregates::OrderedAggregateInputEvaluator
    ));
}
#[tokio::test]
async fn raw_callback_survives_clone_child_aggregate_and_limit_rebuilds_and_executes() {
    for grouped in [false, true] {
        let original = plan(grouped, true, AggregateMode::Single)
            .with_ordered_input_evaluator(Some(one_row));
        same(&original);
        same(&original.clone());
        same(&original.with_new_aggr_exprs(original.aggr_expr().to_vec()));
        same(
            &original
                .with_new_limit_options(Some(crate::aggregates::LimitOptions::new(10))),
        );
        let rebuilt = Arc::new(original.clone())
            .with_new_children(vec![Arc::clone(original.input())])
            .unwrap();
        same(rebuilt.as_any().downcast_ref::<AggregateExec>().unwrap());
        let context = Arc::new(TaskContext::default());
        let output = crate::collect(rebuilt, context).await.unwrap();
        let mut values = output
            .iter()
            .flat_map(|batch| {
                (0..batch.num_rows()).map(|row| {
                    ScalarValue::try_from_array(
                        batch.column(if grouped { 1 } else { 0 }),
                        row,
                    )
                    .unwrap()
                })
            })
            .collect::<Vec<_>>();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(
            values,
            if grouped {
                vec![ScalarValue::Int64(Some(4)), ScalarValue::Int64(Some(5))]
            } else {
                vec![ScalarValue::Int64(Some(9))]
            }
        );
    }
}
#[test]
fn ordinary_factory_and_partial_reduce_cannot_acquire_raw_callback() {
    let foreign = plan(false, false, AggregateMode::Single)
        .with_ordered_input_evaluator(Some(one_row));
    assert!(foreign.ordered_input_evaluator().is_none());
    let partial = plan(false, true, AggregateMode::Partial)
        .with_ordered_input_evaluator(Some(one_row));
    for mode in [
        AggregateMode::Final,
        AggregateMode::FinalPartitioned,
        AggregateMode::PartialReduce,
    ] {
        let merge = AggregateExec::try_new(
            mode,
            PhysicalGroupBy::new_single(vec![]),
            partial.aggr_expr().to_vec(),
            vec![None],
            Arc::new(partial.clone()),
            partial.input_schema(),
        )
        .unwrap()
        .with_ordered_input_evaluator(Some(one_row));
        assert!(merge.ordered_input_evaluator().is_none());
    }
    let pool = Arc::new(UnboundedMemoryPool::default());
    assert_eq!(
        datafusion_execution::memory_pool::MemoryPool::reserved(pool.as_ref()),
        0
    );
}
