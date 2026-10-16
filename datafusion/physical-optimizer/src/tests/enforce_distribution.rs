use super::super::*;
use crate::combine_partial_final_agg::tests::combine_partial_final_agg::{pair, row};
use datafusion_common::ScalarValue;
use datafusion_physical_plan::aggregates::OrderedAggregateInputEvaluator;

#[tokio::test]
async fn actual_distribution_key_reorder_preserves_raw_callback_only_on_partial() {
    for ordered in [false, true] {
        let plan = pair(true, ordered);
        let aggregate = plan.as_any().downcast_ref::<AggregateExec>().unwrap();
        let required = vec![
            Arc::new(Column::new("b", 1)) as Arc<dyn PhysicalExpr>,
            Arc::new(Column::new("a", 0)) as Arc<dyn PhysicalExpr>,
        ];
        let context = PlanWithKeyRequirements::new(
            plan.clone(),
            required,
            vec![PlanWithKeyRequirements::new_default(Arc::clone(
                aggregate.input(),
            ))],
        );
        let reordered = reorder_aggregate_keys(context, aggregate).unwrap();
        let projection = reordered
            .plan
            .as_any()
            .downcast_ref::<ProjectionExec>()
            .unwrap();
        let final_stage = projection
            .input()
            .as_any()
            .downcast_ref::<AggregateExec>()
            .unwrap();
        assert!(final_stage.ordered_input_evaluator().is_none());
        let partial = final_stage
            .input()
            .as_any()
            .downcast_ref::<AggregateExec>()
            .unwrap();
        assert_eq!(partial.group_expr().expr()[0].1, "b");
        assert_eq!(partial.group_expr().expr()[1].1, "a");
        assert_eq!(partial.ordered_input_evaluator().is_some(), ordered);
        if ordered {
            assert!(std::ptr::fn_addr_eq(
                partial.ordered_input_evaluator().unwrap(),
                row as OrderedAggregateInputEvaluator
            ));
        }
        let output = datafusion_physical_plan::collect(
            reordered.plan,
            Arc::new(datafusion_execution::TaskContext::default()),
        )
        .await
        .unwrap();
        let mut actual = output
            .iter()
            .flat_map(|batch| {
                (0..batch.num_rows()).map(|row| {
                    (
                        ScalarValue::try_from_array(batch.column(0), row).unwrap(),
                        ScalarValue::try_from_array(batch.column(1), row).unwrap(),
                        ScalarValue::try_from_array(batch.column(2), row).unwrap(),
                    )
                })
            })
            .collect::<Vec<_>>();
        actual.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        assert_eq!(
            actual,
            [
                (
                    ScalarValue::Int64(Some(1)),
                    ScalarValue::Int64(Some(3)),
                    ScalarValue::Int64(Some(5))
                ),
                (
                    ScalarValue::Int64(Some(2)),
                    ScalarValue::Int64(Some(4)),
                    ScalarValue::Int64(Some(4))
                ),
            ]
        );
    }
}
