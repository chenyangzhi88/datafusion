use super::super::*;
use arrow::{
    array::{ArrayRef, Int64Array, RecordBatch},
    datatypes::{DataType, Field, Schema},
};
use datafusion_common::ScalarValue;
use datafusion_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion_expr::{
    Accumulator, AggregateUDF, AggregateUDFImpl, Signature, Volatility,
};
use datafusion_physical_expr::{aggregate::AggregateExprBuilder, expressions::col};
use datafusion_physical_plan::{
    aggregates::OrderedAggregateInputEvaluator, test::TestMemoryExec,
};
use std::any::Any;

#[derive(Debug, PartialEq, Eq, Hash)]
struct FiniteSum {
    signature: Signature,
    ordered: bool,
}
impl AggregateUDFImpl for FiniteSum {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "finite_sum"
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
            format!("{}_state", args.name),
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
pub(crate) fn row<'a>(
    expression: &'a Arc<dyn PhysicalExpr>,
    batch: &'a RecordBatch,
    _: &'a Arc<ConfigOptions>,
) -> std::pin::Pin<Box<dyn Future<Output = Result<ArrayRef>> + Send + 'a>> {
    Box::pin(async move {
        assert_eq!(batch.num_rows(), 1);
        expression.evaluate(batch)?.into_array(1)
    })
}
pub(crate) fn pair(two_keys: bool, ordered: bool) -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 1, 2])),
            Arc::new(Int64Array::from(vec![3, 3, 4])),
            Arc::new(Int64Array::from(vec![2, 3, 4])),
        ],
    )
    .unwrap();
    let input = Arc::new(
        TestMemoryExec::try_new(&[vec![batch]], Arc::clone(&schema), None).unwrap(),
    );
    let function = Arc::new(AggregateUDF::from(FiniteSum {
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
    let group = PhysicalGroupBy::new_single(if two_keys {
        vec![
            (col("a", &schema).unwrap(), "a".into()),
            (col("b", &schema).unwrap(), "b".into()),
        ]
    } else {
        vec![]
    });
    let partial = AggregateExec::try_new(
        AggregateMode::Partial,
        group.clone(),
        vec![expression],
        vec![None],
        input,
        Arc::clone(&schema),
    )
    .unwrap()
    .with_ordered_input_evaluator(Some(row));
    let final_group = group.as_final();
    Arc::new(
        AggregateExec::try_new(
            if two_keys {
                AggregateMode::FinalPartitioned
            } else {
                AggregateMode::Final
            },
            final_group,
            partial.aggr_expr().to_vec(),
            vec![None],
            Arc::new(partial),
            schema,
        )
        .unwrap(),
    )
}
#[tokio::test]
async fn adjacent_partial_final_combine_moves_same_raw_evaluator_and_preserves_values() {
    for ordered in [false, true] {
        let plan = pair(false, ordered);
        assert!(
            plan.as_any()
                .downcast_ref::<AggregateExec>()
                .unwrap()
                .ordered_input_evaluator()
                .is_none()
        );
        let combined = CombinePartialFinalAggregate::new()
            .optimize(plan, &ConfigOptions::default())
            .unwrap();
        let actual = combined.as_any().downcast_ref::<AggregateExec>().unwrap();
        assert_eq!(actual.mode(), &AggregateMode::Single);
        assert_eq!(actual.ordered_input_evaluator().is_some(), ordered);
        if ordered {
            assert!(std::ptr::fn_addr_eq(
                actual.ordered_input_evaluator().unwrap(),
                row as OrderedAggregateInputEvaluator
            ));
        }
        let output = datafusion_physical_plan::collect(
            combined,
            Arc::new(datafusion_execution::TaskContext::default()),
        )
        .await
        .unwrap();
        assert_eq!(
            ScalarValue::try_from_array(output[0].column(0), 0).unwrap(),
            ScalarValue::Int64(Some(9))
        );
    }
}

#[tokio::test]
async fn plain_empty_group_masks_combine_both_modes_without_replacing_input_or_callback()
{
    for ordered in [false, true] {
        let template = pair(false, ordered);
        let final_template = template.as_any().downcast_ref::<AggregateExec>().unwrap();
        let partial_template = final_template
            .input()
            .as_any()
            .downcast_ref::<AggregateExec>()
            .unwrap();
        for partial_masks in [vec![], vec![vec![]]] {
            for final_masks in [vec![], vec![vec![]]] {
                for final_mode in [AggregateMode::Final, AggregateMode::FinalPartitioned]
                {
                    let partial = AggregateExec::try_new(
                        AggregateMode::Partial,
                        PhysicalGroupBy::new(
                            vec![],
                            vec![],
                            partial_masks.clone(),
                            false,
                        ),
                        partial_template.aggr_expr().to_vec(),
                        partial_template.filter_expr().to_vec(),
                        Arc::clone(partial_template.input()),
                        partial_template.input_schema(),
                    )
                    .unwrap()
                    .with_ordered_input_evaluator(
                        partial_template.ordered_input_evaluator(),
                    );
                    let final_agg = AggregateExec::try_new(
                        final_mode,
                        PhysicalGroupBy::new(vec![], vec![], final_masks.clone(), false),
                        final_template.aggr_expr().to_vec(),
                        final_template.filter_expr().to_vec(),
                        Arc::new(partial),
                        final_template.input_schema(),
                    )
                    .unwrap();
                    let combined = CombinePartialFinalAggregate::new()
                        .optimize(Arc::new(final_agg), &ConfigOptions::default())
                        .unwrap();
                    let actual =
                        combined.as_any().downcast_ref::<AggregateExec>().unwrap();
                    assert_eq!(
                        actual.mode(),
                        &if final_mode == AggregateMode::Final {
                            AggregateMode::Single
                        } else {
                            AggregateMode::SinglePartitioned
                        }
                    );
                    assert!(Arc::ptr_eq(actual.input(), partial_template.input()));
                    assert_eq!(actual.group_expr().groups(), partial_masks);
                    assert_eq!(actual.ordered_input_evaluator().is_some(), ordered);
                    if ordered {
                        assert!(std::ptr::fn_addr_eq(
                            actual.ordered_input_evaluator().unwrap(),
                            partial_template.ordered_input_evaluator().unwrap(),
                        ));
                    }
                    let output = datafusion_physical_plan::collect(
                        combined,
                        Arc::new(datafusion_execution::TaskContext::default()),
                    )
                    .await
                    .unwrap();
                    assert_eq!(output.len(), 1);
                    assert_eq!(output[0].num_rows(), 1);
                    assert_eq!(
                        ScalarValue::try_from_array(output[0].column(0), 0).unwrap(),
                        ScalarValue::Int64(Some(9))
                    );
                }
            }
        }
    }
}

#[test]
fn empty_mask_equivalence_keeps_expression_null_filter_and_grouping_set_guards() {
    use datafusion_physical_expr::expressions::lit;
    let template = pair(false, true);
    let final_agg = template.as_any().downcast_ref::<AggregateExec>().unwrap();
    let partial = final_agg
        .input()
        .as_any()
        .downcast_ref::<AggregateExec>()
        .unwrap();
    let input_group = PhysicalGroupBy::new_single(vec![]);
    let final_group = input_group.as_final();
    let check = |final_group: &PhysicalGroupBy, input_group: &PhysicalGroupBy| {
        can_combine(
            (final_group, final_agg.aggr_expr(), final_agg.filter_expr()),
            (input_group, partial.aggr_expr(), partial.filter_expr()),
        )
    };
    assert!(check(&final_group, &input_group));
    for forbidden in [
        PhysicalGroupBy::new(vec![], vec![], vec![vec![], vec![]], false),
        PhysicalGroupBy::new(vec![], vec![], vec![vec![false]], false),
        PhysicalGroupBy::new(vec![], vec![], vec![vec![]], true),
        PhysicalGroupBy::new(vec![], vec![], vec![], true),
    ] {
        assert!(!check(&final_group, &forbidden));
        assert!(!check(&forbidden, &input_group));
    }
    let null_group = PhysicalGroupBy::new(
        vec![],
        vec![(lit(ScalarValue::Int64(None)), "null".into())],
        vec![],
        false,
    );
    assert!(!check(&null_group, &input_group));
    assert!(!check(&final_group, &null_group));
    let different_null = PhysicalGroupBy::new(
        vec![],
        vec![(lit(ScalarValue::Int64(Some(1))), "null".into())],
        vec![vec![]],
        false,
    );
    assert!(!check(&null_group, &different_null));
    let filter = vec![Some(lit(true))];
    assert!(!can_combine(
        (&final_group, final_agg.aggr_expr(), &filter),
        (&input_group, partial.aggr_expr(), partial.filter_expr()),
    ));
    assert!(!can_combine(
        (&final_group, final_agg.aggr_expr(), final_agg.filter_expr()),
        (&input_group, partial.aggr_expr(), &filter),
    ));
    assert!(!can_combine(
        (&final_group, &[], final_agg.filter_expr()),
        (&input_group, partial.aggr_expr(), partial.filter_expr()),
    ));
    let other_template = pair(false, false);
    let different_aggregate = other_template
        .as_any()
        .downcast_ref::<AggregateExec>()
        .unwrap();
    assert!(!can_combine(
        (
            &final_group,
            different_aggregate.aggr_expr(),
            final_agg.filter_expr()
        ),
        (&input_group, partial.aggr_expr(), partial.filter_expr()),
    ));
    assert!(!can_combine(
        (&final_group, final_agg.aggr_expr(), &[]),
        (&input_group, partial.aggr_expr(), partial.filter_expr()),
    ));
    let keyed = pair(true, false);
    let keyed_final = keyed.as_any().downcast_ref::<AggregateExec>().unwrap();
    let keyed_partial = keyed_final
        .input()
        .as_any()
        .downcast_ref::<AggregateExec>()
        .unwrap();
    let missing_masks = PhysicalGroupBy::new(
        keyed_final.group_expr().expr().to_vec(),
        vec![],
        vec![],
        false,
    );
    assert!(!can_combine(
        (
            &missing_masks,
            keyed_final.aggr_expr(),
            keyed_final.filter_expr()
        ),
        (
            keyed_partial.group_expr(),
            keyed_partial.aggr_expr(),
            keyed_partial.filter_expr()
        ),
    ));
}

#[test]
fn actual_grouping_sets_remain_uncombined_and_keep_the_original_partial_callback() {
    use datafusion_physical_expr::expressions::lit;
    for ordered in [false, true] {
        let template = pair(true, ordered);
        let final_template = template.as_any().downcast_ref::<AggregateExec>().unwrap();
        let partial_template = final_template
            .input()
            .as_any()
            .downcast_ref::<AggregateExec>()
            .unwrap();
        let group = PhysicalGroupBy::new(
            partial_template.group_expr().expr().to_vec(),
            vec![
                (lit(ScalarValue::Int64(None)), "a".into()),
                (lit(ScalarValue::Int64(None)), "b".into()),
            ],
            vec![vec![false, false], vec![true, false]],
            true,
        );
        let final_group = group.as_final();
        let partial = Arc::new(
            AggregateExec::try_new(
                AggregateMode::Partial,
                group,
                partial_template.aggr_expr().to_vec(),
                partial_template.filter_expr().to_vec(),
                Arc::clone(partial_template.input()),
                partial_template.input_schema(),
            )
            .unwrap()
            .with_ordered_input_evaluator(partial_template.ordered_input_evaluator()),
        );
        let input: Arc<dyn ExecutionPlan> = partial;
        let plan = Arc::new(
            AggregateExec::try_new(
                AggregateMode::FinalPartitioned,
                final_group,
                final_template.aggr_expr().to_vec(),
                final_template.filter_expr().to_vec(),
                Arc::clone(&input),
                final_template.input_schema(),
            )
            .unwrap(),
        );
        let optimized = CombinePartialFinalAggregate::new()
            .optimize(plan, &ConfigOptions::default())
            .unwrap();
        let final_agg = optimized.as_any().downcast_ref::<AggregateExec>().unwrap();
        assert_eq!(final_agg.mode(), &AggregateMode::FinalPartitioned);
        assert!(Arc::ptr_eq(final_agg.input(), &input));
        let kept_partial = final_agg
            .input()
            .as_any()
            .downcast_ref::<AggregateExec>()
            .unwrap();
        assert!(kept_partial.group_expr().has_grouping_set());
        assert_eq!(kept_partial.ordered_input_evaluator().is_some(), ordered);
        if ordered {
            assert!(std::ptr::fn_addr_eq(
                kept_partial.ordered_input_evaluator().unwrap(),
                row as OrderedAggregateInputEvaluator
            ));
        }
    }
}
