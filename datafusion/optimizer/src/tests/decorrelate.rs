//! Correlation regressions for typed identity edges; no product-name inference.
use std::sync::Arc;

use arrow::datatypes::DataType;
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion, TreeNodeRewriter};
use datafusion_common::{DFSchemaRef, Result};
use datafusion_expr::{
    Expr, Extension, JoinType, Limit, LogicalPlan, LogicalPlanBuilder,
    UserDefinedLogicalNodeCore, col, exists, in_subquery, lit, not_exists,
    not_in_subquery, out_ref_col, placeholder, scalar_subquery,
};
use datafusion_functions_aggregate::count::count;

use crate::decorrelate::PullUpCorrelatedExpr;
use crate::decorrelate_lateral_join::DecorrelateLateralJoin;
use crate::decorrelate_predicate_subquery::DecorrelatePredicateSubquery;
use crate::scalar_subquery_to_join::ScalarSubqueryToJoin;
use crate::test::test_table_scan_with_name;
use crate::{OptimizerContext, OptimizerRule};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
struct Identity {
    input: LogicalPlan,
    occurrence: usize,
    policy: bool,
}
impl UserDefinedLogicalNodeCore for Identity {
    fn name(&self) -> &str {
        "identity"
    }
    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }
    fn schema(&self) -> &DFSchemaRef {
        self.input.schema()
    }
    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }
    fn decorrelation_transparent_input(&self) -> Option<&LogicalPlan> {
        Some(&self.input)
    }
    fn fmt_for_explain(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "identity occurrence={}", self.occurrence)
    }
    fn with_exprs_and_inputs(
        &self,
        expressions: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        assert!(expressions.is_empty());
        assert_eq!(inputs.len(), 1);
        Ok(Self {
            input: inputs.remove(0),
            occurrence: self.occurrence,
            policy: self.policy,
        })
    }
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
struct Opaque {
    input: LogicalPlan,
}
impl UserDefinedLogicalNodeCore for Opaque {
    fn name(&self) -> &str {
        "NativeSourceBoundary"
    }
    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }
    fn schema(&self) -> &DFSchemaRef {
        self.input.schema()
    }
    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }
    fn fmt_for_explain(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "NativeSourceBoundary")
    }
    fn with_exprs_and_inputs(
        &self,
        expressions: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        assert!(expressions.is_empty());
        assert_eq!(inputs.len(), 1);
        Ok(Self {
            input: inputs.remove(0),
        })
    }
}
fn identity(input: LogicalPlan, occurrence: usize, policy: bool) -> LogicalPlan {
    LogicalPlan::Extension(Extension {
        node: Arc::new(Identity {
            input,
            occurrence,
            policy,
        }),
    })
}
fn correlated_input() -> Result<LogicalPlan> {
    LogicalPlanBuilder::from(test_table_scan_with_name("i")?)
        .filter(col("i.a").eq(out_ref_col(DataType::UInt32, "o.a")))?
        .build()
}
fn identity_facts(plan: &LogicalPlan) -> Result<Vec<(usize, bool)>> {
    let mut result = vec![];
    plan.apply(|plan| {
        if let LogicalPlan::Extension(extension) = plan
            && let Some(node) = extension.node.as_any().downcast_ref::<Identity>()
        {
            result.push((node.occurrence, node.policy));
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(result)
}

#[test]
fn typed_identity_preserves_occurrences_and_original_opaque_sort_union_guards()
-> Result<()> {
    let child = LogicalPlanBuilder::from(correlated_input()?)
        .project([col("i.b")])?
        .build()?;
    let plan = identity(identity(child.clone(), 7, false), 8, true);
    let mut pull = PullUpCorrelatedExpr::new().with_exists_sub_query(true);
    let rewritten = plan.rewrite(&mut pull)?.data;
    assert!(pull.can_pull_up);
    assert_eq!(pull.join_filters.len(), 1);
    assert!(rewritten.all_out_ref_exprs().is_empty());
    assert_eq!(identity_facts(&rewritten)?, [(8, true), (7, false)]);
    for forbidden in [
        LogicalPlan::Extension(Extension {
            node: Arc::new(Opaque {
                input: child.clone(),
            }),
        }),
        LogicalPlanBuilder::from(child.clone())
            .sort([col("i.b").sort(true, false)])?
            .build()?,
        LogicalPlanBuilder::from(child.clone())
            .union(child)?
            .build()?,
    ] {
        let original = identity(forbidden, 9, false);
        let mut pull = PullUpCorrelatedExpr::new().with_exists_sub_query(true);
        let rewritten = original.clone().rewrite(&mut pull)?.data;
        assert!(!pull.can_pull_up, "{rewritten}");
        assert_eq!(rewritten, original);
        assert!(pull.join_filters.is_empty());
    }
    Ok(())
}

#[test]
fn exists_limit_pullup_requires_real_zero_skip_and_literal_fetch() -> Result<()> {
    let cases = [
        (None, None, true, false),
        (Some(lit(0_i64)), Some(lit(1_i64)), true, false),
        (
            Some(lit(datafusion_common::ScalarValue::Int64(None))),
            Some(lit(1_i64)),
            true,
            false,
        ),
        (Some(lit(0_i64)), Some(lit(0_i64)), true, true),
        (Some(lit(1_i64)), Some(lit(1_i64)), false, false),
        (Some(lit(1_i64)), Some(lit(0_i64)), false, false),
        // Proof failure must not emit a new optimizer-phase error or erase
        // the original validation frontier's invalid literal.
        (Some(lit(-1_i64)), Some(lit(1_i64)), false, false),
        (Some(lit(0_i64)), Some(lit(-1_i64)), false, false),
        (Some(placeholder("$1")), Some(lit(1_i64)), false, false),
        (Some(lit(0_i64)), Some(placeholder("$2")), false, false),
        (
            Some(lit(0_i64) + lit(0_i64)),
            Some(lit(1_i64)),
            false,
            false,
        ),
    ];
    for (skip, fetch, allowed, empty) in cases {
        let limit = LogicalPlan::Limit(Limit {
            skip: skip.map(Box::new),
            fetch: fetch.map(Box::new),
            input: Arc::new(correlated_input()?),
        });
        let original = identity(limit, 10, false);
        let mut pull = PullUpCorrelatedExpr::new().with_exists_sub_query(true);
        let rewritten = original.clone().rewrite(&mut pull)?.data;
        assert_eq!(pull.can_pull_up, allowed, "{original}");
        assert_eq!(identity_facts(&rewritten)?, [(10, false)]);
        if allowed {
            assert!(rewritten.all_out_ref_exprs().is_empty());
            let LogicalPlan::Extension(extension) = &rewritten else {
                unreachable!()
            };
            assert_eq!(
                matches!(
                    extension.node.decorrelation_transparent_input().unwrap(),
                    LogicalPlan::EmptyRelation(_)
                ),
                empty
            );
        } else {
            assert_eq!(rewritten, original, "fallback retains actual OFFSET/fetch");
            assert!(pull.join_filters.is_empty());
        }
    }
    for exists in [false, true] {
        let original = identity(
            LogicalPlanBuilder::from(correlated_input()?)
                .limit(1, Some(1))?
                .build()?,
            11,
            true,
        );
        let mut pull = PullUpCorrelatedExpr::new().with_exists_sub_query(exists);
        assert_eq!(original.clone().rewrite(&mut pull)?.data, original);
        assert!(!pull.can_pull_up);
    }
    Ok(())
}

#[test]
fn real_count_state_moves_to_identity_without_cloning_payload_and_handles_equal_children()
-> Result<()> {
    let aggregate = LogicalPlanBuilder::from(correlated_input()?)
        .aggregate(Vec::<Expr>::new(), [count(lit(1_i64)).alias("n")])?
        .build()?;
    for occurrence in [21, 22] {
        // The same genuine aggregate prototype can be visited at two occurrences.
        let original = identity(
            identity(aggregate.clone(), occurrence, false),
            occurrence + 100,
            true,
        );
        let mut pull = PullUpCorrelatedExpr::new().with_need_handle_count_bug(true);
        let rewritten = original.rewrite(&mut pull)?.data;
        assert!(pull.can_pull_up);
        assert!(pull.pulled_up_scalar_agg);
        assert_eq!(
            identity_facts(&rewritten)?,
            [(occurrence + 100, true), (occurrence, false)]
        );
        let map = pull
            .collected_count_expr_map
            .remove(&rewritten)
            .expect("count state reaches final root");
        assert!(!map.is_empty());
        let allocation = map.keys().next().unwrap().as_ptr();
        let mut transfer = PullUpCorrelatedExpr::new().with_need_handle_count_bug(true);
        transfer
            .collected_count_expr_map
            .insert(rewritten.clone(), map);
        let old_capacity = transfer.collected_count_expr_map.capacity();
        let parent = identity(rewritten.clone(), occurrence + 200, false);
        let parent = transfer.f_up(parent)?.data;
        assert!(!transfer.collected_count_expr_map.contains_key(&rewritten));
        let moved = transfer.collected_count_expr_map.get(&parent).unwrap();
        assert_eq!(moved.keys().next().unwrap().as_ptr(), allocation);
        assert_eq!(transfer.collected_count_expr_map.len(), 1);
        assert_eq!(transfer.collected_count_expr_map.capacity(), old_capacity);
    }
    // One rewriter visits both occurrences of the same aggregate prototype.
    let left = LogicalPlanBuilder::from(identity(aggregate.clone(), 51, false))
        .alias("count_left")?
        .build()?;
    let right = LogicalPlanBuilder::from(identity(aggregate, 52, true))
        .alias("count_right")?
        .build()?;
    let dag = LogicalPlanBuilder::from(left).cross_join(right)?.build()?;
    let mut pull = PullUpCorrelatedExpr::new().with_need_handle_count_bug(true);
    let rewritten = dag.rewrite(&mut pull)?.data;
    assert!(pull.can_pull_up);
    assert_eq!(identity_facts(&rewritten)?, [(51, false), (52, true)]);
    let mut aliases = 0;
    rewritten.apply(|node| {
        if let LogicalPlan::SubqueryAlias(_) = node {
            aliases += 1;
            assert!(!pull.collected_count_expr_map.get(node).unwrap().is_empty());
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    assert_eq!(aliases, 2);
    Ok(())
}

#[test]
fn predicate_and_scalar_callers_borrow_head_output_and_keep_count_null_contracts()
-> Result<()> {
    let config = OptimizerContext::new();
    let child = LogicalPlanBuilder::from(correlated_input()?)
        .project([col("i.b")])?
        .build()?;
    for (predicate, expected_type) in [
        (
            exists(Arc::new(identity(child.clone(), 31, false))),
            JoinType::LeftSemi,
        ),
        (
            not_exists(Arc::new(identity(child.clone(), 32, true))),
            JoinType::LeftAnti,
        ),
        (
            in_subquery(col("o.b"), Arc::new(identity(child.clone(), 33, false))),
            JoinType::LeftSemi,
        ),
        (
            not_in_subquery(col("o.b"), Arc::new(identity(child.clone(), 34, true))),
            JoinType::LeftAnti,
        ),
    ] {
        let plan = LogicalPlanBuilder::from(test_table_scan_with_name("o")?)
            .filter(predicate)?
            .build()?;
        let rewritten = DecorrelatePredicateSubquery::new()
            .rewrite(plan, &config)?
            .data;
        let LogicalPlan::Join(join) = &rewritten else {
            panic!("original predicate join required: {rewritten}")
        };
        assert_eq!(join.join_type, expected_type);
        assert_eq!(identity_facts(&rewritten)?.len(), 1);
        assert!(rewritten.all_out_ref_exprs().is_empty());
    }
    let aggregate = LogicalPlanBuilder::from(correlated_input()?)
        .aggregate(Vec::<Expr>::new(), [count(lit(1_i64)).alias("n")])?
        .build()?;
    for having in [false, true] {
        let child = if having {
            LogicalPlanBuilder::from(aggregate.clone())
                .filter(col("n").eq(lit(0_i64)))?
                .build()?
        } else {
            aggregate.clone()
        };
        let child = identity(child, 35, false);
        let plan = LogicalPlanBuilder::from(test_table_scan_with_name("o")?)
            .project([col("o.a"), scalar_subquery(Arc::new(child)).alias("n")])?
            .build()?;
        let rewritten = ScalarSubqueryToJoin::new().rewrite(plan, &config)?.data;
        assert_eq!(identity_facts(&rewritten)?, [(35, false)]);
        let mut scalar = false;
        let mut left = false;
        rewritten.apply(|node| {
            left |= matches!(node, LogicalPlan::Join(join) if join.join_type == JoinType::Left);
            for expr in node.expressions() {
                scalar |= expr.exists(|e| Ok(matches!(e, Expr::ScalarSubquery(_))))?;
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        assert!(left && !scalar, "{rewritten}");
    }
    Ok(())
}

#[test]
fn lateral_shared_caller_retains_identity_without_claiming_count_todo_closed()
-> Result<()> {
    use datafusion_expr::Subquery;
    let child = identity(correlated_input()?, 41, false);
    let subquery = LogicalPlan::Subquery(Subquery {
        outer_ref_columns: child.all_out_ref_exprs(),
        subquery: Arc::new(child),
        spans: Default::default(),
    });
    let plan = LogicalPlanBuilder::from(test_table_scan_with_name("o")?)
        .join_on(subquery, JoinType::Inner, Vec::<Expr>::new())?
        .build()?;
    let rewritten = DecorrelateLateralJoin::new()
        .rewrite(plan, &OptimizerContext::new())?
        .data;
    assert_eq!(identity_facts(&rewritten)?, [(41, false)]);
    assert!(rewritten.all_out_ref_exprs().is_empty());
    Ok(())
}
