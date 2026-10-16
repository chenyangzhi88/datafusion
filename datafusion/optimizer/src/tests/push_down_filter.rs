    use std::any::Any;
    use std::cmp::Ordering;
    use std::fmt::{Debug, Formatter};

    use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use async_trait::async_trait;

    use datafusion_common::{DFSchemaRef, DataFusionError, ScalarValue};
    use datafusion_expr::expr::{ScalarFunction, WindowFunction};
    use datafusion_expr::logical_plan::table_scan;
    use datafusion_expr::{
        ColumnarValue, ExprFunctionExt, Extension, LogicalPlanBuilder,
        ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, TableSource, TableType,
        UserDefinedLogicalNodeCore, Volatility, WindowFunctionDefinition, col, in_list,
        in_subquery, lit,
    };

    use crate::OptimizerContext;
    use crate::assert_optimized_plan_eq_snapshot;
    use crate::optimizer::Optimizer;
    use crate::simplify_expressions::SimplifyExpressions;
    use crate::test::udfs::leaf_udf_expr;
    use crate::test::*;
    use datafusion_expr::test::function_stub::sum;
    use insta::assert_snapshot;

    use crate::push_down_filter::PushDownFilter;
    use std::{collections::HashSet, sync::Arc};
    use datafusion_common::{Column, DFSchema, Result};
    use datafusion_expr::logical_plan::{JoinType, LogicalPlan, TableScan};
    use datafusion_expr::{BinaryExpr, Expr, Operator, TableProviderFilterPushDown, and, or};
    use crate::OptimizerRule;

    fn observe(_plan: &LogicalPlan, _rule: &dyn OptimizerRule) {}

    #[derive(Clone, Debug)]
    struct PredicateGateNode { input: LogicalPlan, allow: bool }
    impl datafusion_expr::UserDefinedLogicalNode for PredicateGateNode {
        fn as_any(&self) -> &dyn Any { self }
        fn name(&self) -> &str { "PredicateGateNode" }
        fn inputs(&self) -> Vec<&LogicalPlan> { vec![&self.input] }
        fn schema(&self) -> &DFSchemaRef { self.input.schema() }
        fn check_invariants(&self, _: datafusion_expr::InvariantLevel) -> Result<()> { Ok(()) }
        fn expressions(&self) -> Vec<Expr> { vec![] }
        fn prevent_predicate_push_down_columns(&self) -> HashSet<String> { HashSet::new() }
        fn supports_predicate_pushdown(&self, _: &Expr) -> bool { self.allow }
        fn fmt_for_explain(&self, f: &mut Formatter) -> std::fmt::Result { write!(f,"PredicateGateNode") }
        fn with_exprs_and_inputs(&self, exprs: Vec<Expr>, mut inputs: Vec<LogicalPlan>) -> Result<Arc<dyn datafusion_expr::UserDefinedLogicalNode>> {
            assert!(exprs.is_empty()); assert_eq!(inputs.len(),1);
            Ok(Arc::new(Self { input: inputs.remove(0), allow: self.allow }))
        }
        fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) { state.write_u8(u8::from(self.allow)); }
        fn dyn_eq(&self, other: &dyn datafusion_expr::UserDefinedLogicalNode) -> bool {
            other.as_any().downcast_ref::<Self>().is_some_and(|other| self.allow == other.allow && self.input == other.input)
        }
        fn dyn_ord(&self, other: &dyn datafusion_expr::UserDefinedLogicalNode) -> Option<Ordering> {
            other.as_any().downcast_ref::<Self>().map(|other| self.allow.cmp(&other.allow).then_with(|| self.input.partial_cmp(&other.input).unwrap()))
        }
    }

    #[test]
    fn predicate_gate_blocks_zero_column_constants_and_volatile_calls() -> Result<()> {
        let volatile = ScalarUDF::new_from_impl(TestScalarUDF { signature: Signature::exact(vec![],Volatility::Volatile) });
        for predicate in [lit(false),volatile.call(vec![]).gt(lit(0.5)),col("test.a").gt(lit(1_i64)).and(lit(false))] {
            let extension = LogicalPlan::Extension(Extension { node: Arc::new(PredicateGateNode { input: test_table_scan()?, allow: false }) });
            let plan = LogicalPlanBuilder::from(extension).filter(predicate)?.build()?;
            let result = PushDownFilter::new().rewrite(plan.clone(),&OptimizerContext::new())?;
            assert_eq!(result.data,plan,"whole predicate must stay above the closed extension");
        }
        Ok(())
    }

    #[test]
    fn predicate_gate_open_rewrites_preserve_the_selected_policy() -> Result<()> {
        let extension = LogicalPlan::Extension(Extension { node: Arc::new(PredicateGateNode { input: test_table_scan()?, allow: true }) });
        let plan = LogicalPlanBuilder::from(extension).filter(col("test.a").gt(lit(1_i64)))?.build()?;
        let result = PushDownFilter::new().rewrite(plan,&OptimizerContext::new())?.data;
        let LogicalPlan::Extension(extension) = result else { panic!("filter did not cross the open gate") };
        let node = extension.node.as_any().downcast_ref::<PredicateGateNode>().unwrap();
        assert!(node.allow);
        assert!(matches!(node.input,LogicalPlan::Filter(_)));
        let rebuilt = extension.node.with_exprs_and_inputs(vec![],vec![test_table_scan()?])?;
        assert!(rebuilt.supports_predicate_pushdown(&lit(false)));
        Ok(())
    }

    #[derive(Clone,Debug,Hash,Eq,PartialEq,PartialOrd)]
    struct DefaultPredicateGateNode { input: LogicalPlan }
    impl UserDefinedLogicalNodeCore for DefaultPredicateGateNode {
        fn name(&self) -> &str { "NativeSourceBoundary" }
        fn inputs(&self) -> Vec<&LogicalPlan> { vec![&self.input] }
        fn schema(&self) -> &DFSchemaRef { self.input.schema() }
        fn expressions(&self) -> Vec<Expr> { vec![] }
        fn prevent_predicate_push_down_columns(&self) -> HashSet<String> { HashSet::new() }
        fn fmt_for_explain(&self,f: &mut Formatter) -> std::fmt::Result { write!(f,"NativeSourceBoundary") }
        fn with_exprs_and_inputs(&self, exprs: Vec<Expr>, mut inputs: Vec<LogicalPlan>) -> Result<Self> {
            assert!(exprs.is_empty()); assert_eq!(inputs.len(),1); Ok(Self { input: inputs.remove(0) })
        }
    }

    #[test]
    fn predicate_gate_default_keeps_foreign_extensions_original_behavior() -> Result<()> {
        let extension = LogicalPlan::Extension(Extension { node: Arc::new(DefaultPredicateGateNode { input: test_table_scan()? }) });
        let plan = LogicalPlanBuilder::from(extension).filter(lit(false))?.build()?;
        let result = PushDownFilter::new().rewrite(plan,&OptimizerContext::new())?.data;
        let LogicalPlan::Extension(extension) = result else { panic!("default extension behavior changed") };
        assert!(matches!(extension.node.inputs()[0],LogicalPlan::Filter(_)));
        Ok(())
    }

    macro_rules! assert_optimized_plan_equal {
        (
            $plan:expr,
            @ $expected:literal $(,)?
        ) => {{
            let optimizer_ctx = OptimizerContext::new().with_max_passes(1);
            let rules: Vec<Arc<dyn crate::OptimizerRule + Send + Sync>> = vec![Arc::new(PushDownFilter::new())];
            assert_optimized_plan_eq_snapshot!(
                optimizer_ctx,
                rules,
                $plan,
                @ $expected,
            )
        }};
    }

    macro_rules! assert_optimized_plan_eq_with_rewrite_predicate {
        (
            $plan:expr,
            @ $expected:literal $(,)?
        ) => {{
            let optimizer = Optimizer::with_rules(vec![
                Arc::new(SimplifyExpressions::new()),
                Arc::new(PushDownFilter::new()),
            ]);
            let optimized_plan = optimizer.optimize($plan, &OptimizerContext::new(), observe)?;
            assert_snapshot!(optimized_plan, @ $expected);
            Ok::<(), DataFusionError>(())
        }};
    }

    #[test]
    fn filter_before_projection() -> Result<()> {
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a"), col("b")])?
            .filter(col("a").eq(lit(1i64)))?
            .build()?;
        // filter is before projection
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a, test.b
          TableScan: test, full_filters=[test.a = Int64(1)]
        "
        )
    }

    #[test]
    fn filter_after_limit() -> Result<()> {
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a"), col("b")])?
            .limit(0, Some(10))?
            .filter(col("a").eq(lit(1i64)))?
            .build()?;
        // filter is before single projection
        assert_optimized_plan_equal!(
            plan,
            @r"
        Filter: test.a = Int64(1)
          Limit: skip=0, fetch=10
            Projection: test.a, test.b
              TableScan: test
        "
        )
    }

    #[test]
    fn filter_no_columns() -> Result<()> {
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .filter(lit(0i64).eq(lit(1i64)))?
            .build()?;
        assert_optimized_plan_equal!(
            plan,
            @"TableScan: test, full_filters=[Int64(0) = Int64(1)]"
        )
    }

    #[test]
    fn filter_jump_2_plans() -> Result<()> {
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a"), col("b"), col("c")])?
            .project(vec![col("c"), col("b")])?
            .filter(col("a").eq(lit(1i64)))?
            .build()?;
        // filter is before double projection
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.c, test.b
          Projection: test.a, test.b, test.c
            TableScan: test, full_filters=[test.a = Int64(1)]
        "
        )
    }

    #[test]
    fn filter_move_agg() -> Result<()> {
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("a")], vec![sum(col("b")).alias("total_salary")])?
            .filter(col("a").gt(lit(10i64)))?
            .build()?;
        // filter of key aggregation is commutative
        assert_optimized_plan_equal!(
            plan,
            @r"
        Aggregate: groupBy=[[test.a]], aggr=[[sum(test.b) AS total_salary]]
          TableScan: test, full_filters=[test.a > Int64(10)]
        "
        )
    }

    /// verifies that filters with unusual column names are pushed down through aggregate operators
    #[test]
    fn filter_move_agg_special() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new("$a", DataType::UInt32, false),
            Field::new("$b", DataType::UInt32, false),
            Field::new("$c", DataType::UInt32, false),
        ]);
        let table_scan = table_scan(Some("test"), &schema, None)?.build()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("$a")], vec![sum(col("$b")).alias("total_salary")])?
            .filter(col("$a").gt(lit(10i64)))?
            .build()?;
        // filter of key aggregation is commutative
        assert_optimized_plan_equal!(
            plan,
            @r"
        Aggregate: groupBy=[[test.$a]], aggr=[[sum(test.$b) AS total_salary]]
          TableScan: test, full_filters=[test.$a > Int64(10)]
        "
        )
    }

    #[test]
    fn filter_complex_group_by() -> Result<()> {
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![add(col("b"), col("a"))], vec![sum(col("a")), col("b")])?
            .filter(col("b").gt(lit(10i64)))?
            .build()?;
        assert_optimized_plan_equal!(
            plan,
            @r"
        Filter: test.b > Int64(10)
          Aggregate: groupBy=[[test.b + test.a]], aggr=[[sum(test.a), test.b]]
            TableScan: test
        "
        )
    }

    #[test]
    fn push_agg_need_replace_expr() -> Result<()> {
        let plan = LogicalPlanBuilder::from(test_table_scan()?)
            .aggregate(vec![add(col("b"), col("a"))], vec![sum(col("a")), col("b")])?
            .filter(col("test.b + test.a").gt(lit(10i64)))?
            .build()?;
        assert_optimized_plan_equal!(
            plan,
            @r"
        Aggregate: groupBy=[[test.b + test.a]], aggr=[[sum(test.a), test.b]]
          TableScan: test, full_filters=[test.b + test.a > Int64(10)]
        "
        )
    }

    #[test]
    fn filter_keep_agg() -> Result<()> {
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("a")], vec![sum(col("b")).alias("b")])?
            .filter(col("b").gt(lit(10i64)))?
            .build()?;
        // filter of aggregate is after aggregation since they are non-commutative
        assert_optimized_plan_equal!(
            plan,
            @r"
        Filter: b > Int64(10)
          Aggregate: groupBy=[[test.a]], aggr=[[sum(test.b) AS b]]
            TableScan: test
        "
        )
    }

    /// verifies that when partitioning by 'a' and 'b', and filtering by 'b', 'b' is pushed
    #[test]
    fn filter_move_window() -> Result<()> {
        let table_scan = test_table_scan()?;

        let window = Expr::from(WindowFunction::new(
            WindowFunctionDefinition::WindowUDF(
                datafusion_functions_window::rank::rank_udwf(),
            ),
            vec![],
        ))
        .partition_by(vec![col("a"), col("b")])
        .order_by(vec![col("c").sort(true, true)])
        .build()
        .unwrap();

        let plan = LogicalPlanBuilder::from(table_scan)
            .window(vec![window])?
            .filter(col("b").gt(lit(10i64)))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        WindowAggr: windowExpr=[[rank() PARTITION BY [test.a, test.b] ORDER BY [test.c ASC NULLS FIRST] ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW]]
          TableScan: test, full_filters=[test.b > Int64(10)]
        "
        )
    }

    /// verifies that filters with unusual identifier names are pushed down through window functions
    #[test]
    fn filter_window_special_identifier() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new("$a", DataType::UInt32, false),
            Field::new("$b", DataType::UInt32, false),
            Field::new("$c", DataType::UInt32, false),
        ]);
        let table_scan = table_scan(Some("test"), &schema, None)?.build()?;

        let window = Expr::from(WindowFunction::new(
            WindowFunctionDefinition::WindowUDF(
                datafusion_functions_window::rank::rank_udwf(),
            ),
            vec![],
        ))
        .partition_by(vec![col("$a"), col("$b")])
        .order_by(vec![col("$c").sort(true, true)])
        .build()
        .unwrap();

        let plan = LogicalPlanBuilder::from(table_scan)
            .window(vec![window])?
            .filter(col("$b").gt(lit(10i64)))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        WindowAggr: windowExpr=[[rank() PARTITION BY [test.$a, test.$b] ORDER BY [test.$c ASC NULLS FIRST] ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW]]
          TableScan: test, full_filters=[test.$b > Int64(10)]
        "
        )
    }

    /// verifies that when partitioning by 'a' and 'b', and filtering by 'a' and 'b', both 'a' and
    /// 'b' are pushed
    #[test]
    fn filter_move_complex_window() -> Result<()> {
        let table_scan = test_table_scan()?;

        let window = Expr::from(WindowFunction::new(
            WindowFunctionDefinition::WindowUDF(
                datafusion_functions_window::rank::rank_udwf(),
            ),
            vec![],
        ))
        .partition_by(vec![col("a"), col("b")])
        .order_by(vec![col("c").sort(true, true)])
        .build()
        .unwrap();

        let plan = LogicalPlanBuilder::from(table_scan)
            .window(vec![window])?
            .filter(and(col("a").gt(lit(10i64)), col("b").eq(lit(1i64))))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        WindowAggr: windowExpr=[[rank() PARTITION BY [test.a, test.b] ORDER BY [test.c ASC NULLS FIRST] ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW]]
          TableScan: test, full_filters=[test.a > Int64(10), test.b = Int64(1)]
        "
        )
    }

    /// verifies that when partitioning by 'a' and filtering by 'a' and 'b', only 'a' is pushed
    #[test]
    fn filter_move_partial_window() -> Result<()> {
        let table_scan = test_table_scan()?;

        let window = Expr::from(WindowFunction::new(
            WindowFunctionDefinition::WindowUDF(
                datafusion_functions_window::rank::rank_udwf(),
            ),
            vec![],
        ))
        .partition_by(vec![col("a")])
        .order_by(vec![col("c").sort(true, true)])
        .build()
        .unwrap();

        let plan = LogicalPlanBuilder::from(table_scan)
            .window(vec![window])?
            .filter(and(col("a").gt(lit(10i64)), col("b").eq(lit(1i64))))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Filter: test.b = Int64(1)
          WindowAggr: windowExpr=[[rank() PARTITION BY [test.a] ORDER BY [test.c ASC NULLS FIRST] ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW]]
            TableScan: test, full_filters=[test.a > Int64(10)]
        "
        )
    }

    /// verifies that filters on partition expressions are not pushed, as the single expression
    /// column is not available to the user, unlike with aggregations
    #[test]
    fn filter_expression_keep_window() -> Result<()> {
        let table_scan = test_table_scan()?;

        let window = Expr::from(WindowFunction::new(
            WindowFunctionDefinition::WindowUDF(
                datafusion_functions_window::rank::rank_udwf(),
            ),
            vec![],
        ))
        .partition_by(vec![add(col("a"), col("b"))]) // PARTITION BY a + b
        .order_by(vec![col("c").sort(true, true)])
        .build()
        .unwrap();

        let plan = LogicalPlanBuilder::from(table_scan)
            .window(vec![window])?
            // unlike with aggregations, single partition column "test.a + test.b" is not available
            // to the plan, so we use multiple columns when filtering
            .filter(add(col("a"), col("b")).gt(lit(10i64)))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Filter: test.a + test.b > Int64(10)
          WindowAggr: windowExpr=[[rank() PARTITION BY [test.a + test.b] ORDER BY [test.c ASC NULLS FIRST] ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW]]
            TableScan: test
        "
        )
    }

    /// verifies that filters are not pushed on order by columns (that are not used in partitioning)
    #[test]
    fn filter_order_keep_window() -> Result<()> {
        let table_scan = test_table_scan()?;

        let window = Expr::from(WindowFunction::new(
            WindowFunctionDefinition::WindowUDF(
                datafusion_functions_window::rank::rank_udwf(),
            ),
            vec![],
        ))
        .partition_by(vec![col("a")])
        .order_by(vec![col("c").sort(true, true)])
        .build()
        .unwrap();

        let plan = LogicalPlanBuilder::from(table_scan)
            .window(vec![window])?
            .filter(col("c").gt(lit(10i64)))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Filter: test.c > Int64(10)
          WindowAggr: windowExpr=[[rank() PARTITION BY [test.a] ORDER BY [test.c ASC NULLS FIRST] ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW]]
            TableScan: test
        "
        )
    }

    /// verifies that when we use multiple window functions with a common partition key, the filter
    /// on that key is pushed
    #[test]
    fn filter_multiple_windows_common_partitions() -> Result<()> {
        let table_scan = test_table_scan()?;

        let window1 = Expr::from(WindowFunction::new(
            WindowFunctionDefinition::WindowUDF(
                datafusion_functions_window::rank::rank_udwf(),
            ),
            vec![],
        ))
        .partition_by(vec![col("a")])
        .order_by(vec![col("c").sort(true, true)])
        .build()
        .unwrap();

        let window2 = Expr::from(WindowFunction::new(
            WindowFunctionDefinition::WindowUDF(
                datafusion_functions_window::rank::rank_udwf(),
            ),
            vec![],
        ))
        .partition_by(vec![col("b"), col("a")])
        .order_by(vec![col("c").sort(true, true)])
        .build()
        .unwrap();

        let plan = LogicalPlanBuilder::from(table_scan)
            .window(vec![window1, window2])?
            .filter(col("a").gt(lit(10i64)))? // a appears in both window functions
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        WindowAggr: windowExpr=[[rank() PARTITION BY [test.a] ORDER BY [test.c ASC NULLS FIRST] ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW, rank() PARTITION BY [test.b, test.a] ORDER BY [test.c ASC NULLS FIRST] ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW]]
          TableScan: test, full_filters=[test.a > Int64(10)]
        "
        )
    }

    /// verifies that when we use multiple window functions with different partitions keys, the
    /// filter cannot be pushed
    #[test]
    fn filter_multiple_windows_disjoint_partitions() -> Result<()> {
        let table_scan = test_table_scan()?;

        let window1 = Expr::from(WindowFunction::new(
            WindowFunctionDefinition::WindowUDF(
                datafusion_functions_window::rank::rank_udwf(),
            ),
            vec![],
        ))
        .partition_by(vec![col("a")])
        .order_by(vec![col("c").sort(true, true)])
        .build()
        .unwrap();

        let window2 = Expr::from(WindowFunction::new(
            WindowFunctionDefinition::WindowUDF(
                datafusion_functions_window::rank::rank_udwf(),
            ),
            vec![],
        ))
        .partition_by(vec![col("b"), col("a")])
        .order_by(vec![col("c").sort(true, true)])
        .build()
        .unwrap();

        let plan = LogicalPlanBuilder::from(table_scan)
            .window(vec![window1, window2])?
            .filter(col("b").gt(lit(10i64)))? // b only appears in one window function
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Filter: test.b > Int64(10)
          WindowAggr: windowExpr=[[rank() PARTITION BY [test.a] ORDER BY [test.c ASC NULLS FIRST] ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW, rank() PARTITION BY [test.b, test.a] ORDER BY [test.c ASC NULLS FIRST] ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW]]
            TableScan: test
        "
        )
    }

    /// verifies that a filter is pushed to before a projection, the filter expression is correctly re-written
    #[test]
    fn alias() -> Result<()> {
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a").alias("b"), col("c")])?
            .filter(col("b").eq(lit(1i64)))?
            .build()?;
        // filter is before projection
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a AS b, test.c
          TableScan: test, full_filters=[test.a = Int64(1)]
        "
        )
    }

    fn add(left: Expr, right: Expr) -> Expr {
        Expr::BinaryExpr(BinaryExpr::new(
            Box::new(left),
            Operator::Plus,
            Box::new(right),
        ))
    }

    fn multiply(left: Expr, right: Expr) -> Expr {
        Expr::BinaryExpr(BinaryExpr::new(
            Box::new(left),
            Operator::Multiply,
            Box::new(right),
        ))
    }

    /// verifies that a filter is pushed to before a projection with a complex expression, the filter expression is correctly re-written
    #[test]
    fn complex_expression() -> Result<()> {
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![
                add(multiply(col("a"), lit(2)), col("c")).alias("b"),
                col("c"),
            ])?
            .filter(col("b").eq(lit(1i64)))?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        Filter: b = Int64(1)
          Projection: test.a * Int32(2) + test.c AS b, test.c
            TableScan: test
        ",
        );
        // filter is before projection
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a * Int32(2) + test.c AS b, test.c
          TableScan: test, full_filters=[test.a * Int32(2) + test.c = Int64(1)]
        "
        )
    }

    /// verifies that when a filter is pushed to after 2 projections, the filter expression is correctly re-written
    #[test]
    fn complex_plan() -> Result<()> {
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![
                add(multiply(col("a"), lit(2)), col("c")).alias("b"),
                col("c"),
            ])?
            // second projection where we rename columns, just to make it difficult
            .project(vec![multiply(col("b"), lit(3)).alias("a"), col("c")])?
            .filter(col("a").eq(lit(1i64)))?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        Filter: a = Int64(1)
          Projection: b * Int32(3) AS a, test.c
            Projection: test.a * Int32(2) + test.c AS b, test.c
              TableScan: test
        ",
        );
        // filter is before the projections
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: b * Int32(3) AS a, test.c
          Projection: test.a * Int32(2) + test.c AS b, test.c
            TableScan: test, full_filters=[(test.a * Int32(2) + test.c) * Int32(3) = Int64(1)]
        "
        )
    }

    #[derive(Debug, PartialEq, Eq, Hash)]
    struct NoopPlan {
        input: Vec<LogicalPlan>,
        schema: DFSchemaRef,
    }

    // Manual implementation needed because of `schema` field. Comparison excludes this field.
    impl PartialOrd for NoopPlan {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            self.input
                .partial_cmp(&other.input)
                // TODO (https://github.com/apache/datafusion/issues/17477) avoid recomparing all fields
                .filter(|cmp| *cmp != Ordering::Equal || self == other)
        }
    }

    impl UserDefinedLogicalNodeCore for NoopPlan {
        fn name(&self) -> &str {
            "NoopPlan"
        }

        fn inputs(&self) -> Vec<&LogicalPlan> {
            self.input.iter().collect()
        }

        fn schema(&self) -> &DFSchemaRef {
            &self.schema
        }

        fn expressions(&self) -> Vec<Expr> {
            self.input
                .iter()
                .flat_map(|child| child.expressions())
                .collect()
        }

        fn prevent_predicate_push_down_columns(&self) -> HashSet<String> {
            HashSet::from_iter(vec!["c".to_string()])
        }

        fn fmt_for_explain(&self, f: &mut Formatter) -> std::fmt::Result {
            write!(f, "NoopPlan")
        }

        fn with_exprs_and_inputs(
            &self,
            _exprs: Vec<Expr>,
            inputs: Vec<LogicalPlan>,
        ) -> Result<Self> {
            Ok(Self {
                input: inputs,
                schema: Arc::clone(&self.schema),
            })
        }

        fn supports_limit_pushdown(&self) -> bool {
            false // Disallow limit push-down by default
        }
    }

    #[test]
    fn user_defined_plan() -> Result<()> {
        let table_scan = test_table_scan()?;

        let custom_plan = LogicalPlan::Extension(Extension {
            node: Arc::new(NoopPlan {
                input: vec![table_scan.clone()],
                schema: Arc::clone(table_scan.schema()),
            }),
        });
        let plan = LogicalPlanBuilder::from(custom_plan)
            .filter(col("a").eq(lit(1i64)))?
            .build()?;

        // Push filter below NoopPlan
        assert_optimized_plan_equal!(
            plan,
            @r"
        NoopPlan
          TableScan: test, full_filters=[test.a = Int64(1)]
        "
        )?;

        let custom_plan = LogicalPlan::Extension(Extension {
            node: Arc::new(NoopPlan {
                input: vec![table_scan.clone()],
                schema: Arc::clone(table_scan.schema()),
            }),
        });
        let plan = LogicalPlanBuilder::from(custom_plan)
            .filter(col("a").eq(lit(1i64)).and(col("c").eq(lit(2i64))))?
            .build()?;

        // Push only predicate on `a` below NoopPlan
        assert_optimized_plan_equal!(
            plan,
            @r"
        Filter: test.c = Int64(2)
          NoopPlan
            TableScan: test, full_filters=[test.a = Int64(1)]
        "
        )?;

        let custom_plan = LogicalPlan::Extension(Extension {
            node: Arc::new(NoopPlan {
                input: vec![table_scan.clone(), table_scan.clone()],
                schema: Arc::clone(table_scan.schema()),
            }),
        });
        let plan = LogicalPlanBuilder::from(custom_plan)
            .filter(col("a").eq(lit(1i64)))?
            .build()?;

        // Push filter below NoopPlan for each child branch
        assert_optimized_plan_equal!(
            plan,
            @r"
        NoopPlan
          TableScan: test, full_filters=[test.a = Int64(1)]
          TableScan: test, full_filters=[test.a = Int64(1)]
        "
        )?;

        let custom_plan = LogicalPlan::Extension(Extension {
            node: Arc::new(NoopPlan {
                input: vec![table_scan.clone(), table_scan.clone()],
                schema: Arc::clone(table_scan.schema()),
            }),
        });
        let plan = LogicalPlanBuilder::from(custom_plan)
            .filter(col("a").eq(lit(1i64)).and(col("c").eq(lit(2i64))))?
            .build()?;

        // Push only predicate on `a` below NoopPlan
        assert_optimized_plan_equal!(
            plan,
            @r"
        Filter: test.c = Int64(2)
          NoopPlan
            TableScan: test, full_filters=[test.a = Int64(1)]
            TableScan: test, full_filters=[test.a = Int64(1)]
        "
        )
    }

    /// verifies that when two filters apply after an aggregation that only allows one to be pushed, one is pushed
    /// and the other not.
    #[test]
    fn multi_filter() -> Result<()> {
        // the aggregation allows one filter to pass (b), and the other one to not pass (sum(c))
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a").alias("b"), col("c")])?
            .aggregate(vec![col("b")], vec![sum(col("c"))])?
            .filter(col("b").gt(lit(10i64)))?
            .filter(col("sum(test.c)").gt(lit(10i64)))?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        Filter: sum(test.c) > Int64(10)
          Filter: b > Int64(10)
            Aggregate: groupBy=[[b]], aggr=[[sum(test.c)]]
              Projection: test.a AS b, test.c
                TableScan: test
        ",
        );
        // filter is before the projections
        assert_optimized_plan_equal!(
            plan,
            @r"
        Filter: sum(test.c) > Int64(10)
          Aggregate: groupBy=[[b]], aggr=[[sum(test.c)]]
            Projection: test.a AS b, test.c
              TableScan: test, full_filters=[test.a > Int64(10)]
        "
        )
    }

    /// verifies that when a filter with two predicates is applied after an aggregation that only allows one to be pushed, one is pushed
    /// and the other not.
    #[test]
    fn split_filter() -> Result<()> {
        // the aggregation allows one filter to pass (b), and the other one to not pass (sum(c))
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a").alias("b"), col("c")])?
            .aggregate(vec![col("b")], vec![sum(col("c"))])?
            .filter(and(
                col("sum(test.c)").gt(lit(10i64)),
                and(col("b").gt(lit(10i64)), col("sum(test.c)").lt(lit(20i64))),
            ))?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        Filter: sum(test.c) > Int64(10) AND b > Int64(10) AND sum(test.c) < Int64(20)
          Aggregate: groupBy=[[b]], aggr=[[sum(test.c)]]
            Projection: test.a AS b, test.c
              TableScan: test
        ",
        );
        // filter is before the projections
        assert_optimized_plan_equal!(
            plan,
            @r"
        Filter: sum(test.c) > Int64(10) AND sum(test.c) < Int64(20)
          Aggregate: groupBy=[[b]], aggr=[[sum(test.c)]]
            Projection: test.a AS b, test.c
              TableScan: test, full_filters=[test.a > Int64(10)]
        "
        )
    }

    /// verifies that when two limits are in place, we jump neither
    #[test]
    fn double_limit() -> Result<()> {
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a"), col("b")])?
            .limit(0, Some(20))?
            .limit(0, Some(10))?
            .project(vec![col("a"), col("b")])?
            .filter(col("a").eq(lit(1i64)))?
            .build()?;
        // filter does not just any of the limits
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a, test.b
          Filter: test.a = Int64(1)
            Limit: skip=0, fetch=10
              Limit: skip=0, fetch=20
                Projection: test.a, test.b
                  TableScan: test
        "
        )
    }

    #[test]
    fn union_all() -> Result<()> {
        let table_scan = test_table_scan()?;
        let table_scan2 = test_table_scan_with_name("test2")?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .union(LogicalPlanBuilder::from(table_scan2).build()?)?
            .filter(col("a").eq(lit(1i64)))?
            .build()?;
        // filter appears below Union
        assert_optimized_plan_equal!(
            plan,
            @r"
        Union
          TableScan: test, full_filters=[test.a = Int64(1)]
          TableScan: test2, full_filters=[test2.a = Int64(1)]
        "
        )
    }

    #[test]
    fn union_all_on_projection() -> Result<()> {
        let table_scan = test_table_scan()?;
        let table = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a").alias("b")])?
            .alias("test2")?;

        let plan = table
            .clone()
            .union(table.build()?)?
            .filter(col("b").eq(lit(1i64)))?
            .build()?;

        // filter appears below Union
        assert_optimized_plan_equal!(
            plan,
            @r"
        Union
          SubqueryAlias: test2
            Projection: test.a AS b
              TableScan: test, full_filters=[test.a = Int64(1)]
          SubqueryAlias: test2
            Projection: test.a AS b
              TableScan: test, full_filters=[test.a = Int64(1)]
        "
        )
    }

    #[test]
    fn test_union_different_schema() -> Result<()> {
        let left = LogicalPlanBuilder::from(test_table_scan()?)
            .project(vec![col("a"), col("b"), col("c")])?
            .build()?;

        let schema = Schema::new(vec![
            Field::new("d", DataType::UInt32, false),
            Field::new("e", DataType::UInt32, false),
            Field::new("f", DataType::UInt32, false),
        ]);
        let right = table_scan(Some("test1"), &schema, None)?
            .project(vec![col("d"), col("e"), col("f")])?
            .build()?;
        let filter = and(col("test.a").eq(lit(1)), col("test1.d").gt(lit(2)));
        let plan = LogicalPlanBuilder::from(left)
            .cross_join(right)?
            .project(vec![col("test.a"), col("test1.d")])?
            .filter(filter)?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a, test1.d
          Cross Join:
            Projection: test.a, test.b, test.c
              TableScan: test, full_filters=[test.a = Int32(1)]
            Projection: test1.d, test1.e, test1.f
              TableScan: test1, full_filters=[test1.d > Int32(2)]
        "
        )
    }

    #[test]
    fn test_project_same_name_different_qualifier() -> Result<()> {
        let table_scan = test_table_scan()?;
        let left = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a"), col("b"), col("c")])?
            .build()?;
        let right_table_scan = test_table_scan_with_name("test1")?;
        let right = LogicalPlanBuilder::from(right_table_scan)
            .project(vec![col("a"), col("b"), col("c")])?
            .build()?;
        let filter = and(col("test.a").eq(lit(1)), col("test1.a").gt(lit(2)));
        let plan = LogicalPlanBuilder::from(left)
            .cross_join(right)?
            .project(vec![col("test.a"), col("test1.a")])?
            .filter(filter)?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a, test1.a
          Cross Join:
            Projection: test.a, test.b, test.c
              TableScan: test, full_filters=[test.a = Int32(1)]
            Projection: test1.a, test1.b, test1.c
              TableScan: test1, full_filters=[test1.a > Int32(2)]
        "
        )
    }

    /// verifies that filters with the same columns are correctly placed
    #[test]
    fn filter_2_breaks_limits() -> Result<()> {
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a")])?
            .filter(col("a").lt_eq(lit(1i64)))?
            .limit(0, Some(1))?
            .project(vec![col("a")])?
            .filter(col("a").gt_eq(lit(1i64)))?
            .build()?;
        // Should be able to move both filters below the projections

        // not part of the test
        assert_snapshot!(plan,
        @r"
        Filter: test.a >= Int64(1)
          Projection: test.a
            Limit: skip=0, fetch=1
              Filter: test.a <= Int64(1)
                Projection: test.a
                  TableScan: test
        ",
        );
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a
          Filter: test.a >= Int64(1)
            Limit: skip=0, fetch=1
              Projection: test.a
                TableScan: test, full_filters=[test.a <= Int64(1)]
        "
        )
    }

    /// verifies that filters to be placed on the same depth are ANDed
    #[test]
    fn two_filters_on_same_depth() -> Result<()> {
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .limit(0, Some(1))?
            .filter(col("a").lt_eq(lit(1i64)))?
            .filter(col("a").gt_eq(lit(1i64)))?
            .project(vec![col("a")])?
            .build()?;

        // not part of the test
        assert_snapshot!(plan,
        @r"
        Projection: test.a
          Filter: test.a >= Int64(1)
            Filter: test.a <= Int64(1)
              Limit: skip=0, fetch=1
                TableScan: test
        ",
        );
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a
          Filter: test.a >= Int64(1) AND test.a <= Int64(1)
            Limit: skip=0, fetch=1
              TableScan: test
        "
        )
    }

    /// verifies that filters on a plan with user nodes are not lost
    /// (ARROW-10547)
    #[test]
    fn filters_user_defined_node() -> Result<()> {
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .filter(col("a").lt_eq(lit(1i64)))?
            .build()?;

        let plan = user_defined::new(plan);

        // not part of the test
        assert_snapshot!(plan,
        @r"
        TestUserDefined
          Filter: test.a <= Int64(1)
            TableScan: test
        ",
        );
        assert_optimized_plan_equal!(
            plan,
            @r"
        TestUserDefined
          TableScan: test, full_filters=[test.a <= Int64(1)]
        "
        )
    }

    /// post-on-join predicates on a column common to both sides is pushed to both sides
    #[test]
    fn filter_on_join_on_common_independent() -> Result<()> {
        let table_scan = test_table_scan()?;
        let left = LogicalPlanBuilder::from(table_scan).build()?;
        let right_table_scan = test_table_scan_with_name("test2")?;
        let right = LogicalPlanBuilder::from(right_table_scan)
            .project(vec![col("a")])?
            .build()?;
        let plan = LogicalPlanBuilder::from(left)
            .join(
                right,
                JoinType::Inner,
                (vec![Column::from_name("a")], vec![Column::from_name("a")]),
                None,
            )?
            .filter(col("test.a").lt_eq(lit(1i64)))?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        Filter: test.a <= Int64(1)
          Inner Join: test.a = test2.a
            TableScan: test
            Projection: test2.a
              TableScan: test2
        ",
        );
        // filter sent to side before the join
        assert_optimized_plan_equal!(
            plan,
            @r"
        Inner Join: test.a = test2.a
          TableScan: test, full_filters=[test.a <= Int64(1)]
          Projection: test2.a
            TableScan: test2, full_filters=[test2.a <= Int64(1)]
        "
        )
    }

    /// post-using-join predicates on a column common to both sides is pushed to both sides
    #[test]
    fn filter_using_join_on_common_independent() -> Result<()> {
        let table_scan = test_table_scan()?;
        let left = LogicalPlanBuilder::from(table_scan).build()?;
        let right_table_scan = test_table_scan_with_name("test2")?;
        let right = LogicalPlanBuilder::from(right_table_scan)
            .project(vec![col("a")])?
            .build()?;
        let plan = LogicalPlanBuilder::from(left)
            .join_using(
                right,
                JoinType::Inner,
                vec![Column::from_name("a".to_string())],
            )?
            .filter(col("a").lt_eq(lit(1i64)))?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        Filter: test.a <= Int64(1)
          Inner Join: Using test.a = test2.a
            TableScan: test
            Projection: test2.a
              TableScan: test2
        ",
        );
        // filter sent to side before the join
        assert_optimized_plan_equal!(
            plan,
            @r"
        Inner Join: Using test.a = test2.a
          TableScan: test, full_filters=[test.a <= Int64(1)]
          Projection: test2.a
            TableScan: test2, full_filters=[test2.a <= Int64(1)]
        "
        )
    }

    /// post-join predicates with columns from both sides are converted to join filters
    #[test]
    fn filter_join_on_common_dependent() -> Result<()> {
        let table_scan = test_table_scan()?;
        let left = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a"), col("c")])?
            .build()?;
        let right_table_scan = test_table_scan_with_name("test2")?;
        let right = LogicalPlanBuilder::from(right_table_scan)
            .project(vec![col("a"), col("b")])?
            .build()?;
        let plan = LogicalPlanBuilder::from(left)
            .join(
                right,
                JoinType::Inner,
                (vec![Column::from_name("a")], vec![Column::from_name("a")]),
                None,
            )?
            .filter(col("c").lt_eq(col("b")))?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        Filter: test.c <= test2.b
          Inner Join: test.a = test2.a
            Projection: test.a, test.c
              TableScan: test
            Projection: test2.a, test2.b
              TableScan: test2
        ",
        );
        // Filter is converted to Join Filter
        assert_optimized_plan_equal!(
            plan,
            @r"
        Inner Join: test.a = test2.a Filter: test.c <= test2.b
          Projection: test.a, test.c
            TableScan: test
          Projection: test2.a, test2.b
            TableScan: test2
        "
        )
    }

    /// post-join predicates with columns from one side of a join are pushed only to that side
    #[test]
    fn filter_join_on_one_side() -> Result<()> {
        let table_scan = test_table_scan()?;
        let left = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a"), col("b")])?
            .build()?;
        let table_scan_right = test_table_scan_with_name("test2")?;
        let right = LogicalPlanBuilder::from(table_scan_right)
            .project(vec![col("a"), col("c")])?
            .build()?;

        let plan = LogicalPlanBuilder::from(left)
            .join(
                right,
                JoinType::Inner,
                (vec![Column::from_name("a")], vec![Column::from_name("a")]),
                None,
            )?
            .filter(col("b").lt_eq(lit(1i64)))?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        Filter: test.b <= Int64(1)
          Inner Join: test.a = test2.a
            Projection: test.a, test.b
              TableScan: test
            Projection: test2.a, test2.c
              TableScan: test2
        ",
        );
        assert_optimized_plan_equal!(
            plan,
            @r"
        Inner Join: test.a = test2.a
          Projection: test.a, test.b
            TableScan: test, full_filters=[test.b <= Int64(1)]
          Projection: test2.a, test2.c
            TableScan: test2
        "
        )
    }

    /// post-join predicates on the right side of a left join are not duplicated
    /// TODO: In this case we can sometimes convert the join to an INNER join
    #[test]
    fn filter_using_left_join() -> Result<()> {
        let table_scan = test_table_scan()?;
        let left = LogicalPlanBuilder::from(table_scan).build()?;
        let right_table_scan = test_table_scan_with_name("test2")?;
        let right = LogicalPlanBuilder::from(right_table_scan)
            .project(vec![col("a")])?
            .build()?;
        let plan = LogicalPlanBuilder::from(left)
            .join_using(
                right,
                JoinType::Left,
                vec![Column::from_name("a".to_string())],
            )?
            .filter(col("test2.a").lt_eq(lit(1i64)))?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        Filter: test2.a <= Int64(1)
          Left Join: Using test.a = test2.a
            TableScan: test
            Projection: test2.a
              TableScan: test2
        ",
        );
        // filter not duplicated nor pushed down - i.e. noop
        assert_optimized_plan_equal!(
            plan,
            @r"
        Filter: test2.a <= Int64(1)
          Left Join: Using test.a = test2.a
            TableScan: test, full_filters=[test.a <= Int64(1)]
            Projection: test2.a
              TableScan: test2
        "
        )
    }

    /// post-join predicates on the left side of a right join are not duplicated
    #[test]
    fn filter_using_right_join() -> Result<()> {
        let table_scan = test_table_scan()?;
        let left = LogicalPlanBuilder::from(table_scan).build()?;
        let right_table_scan = test_table_scan_with_name("test2")?;
        let right = LogicalPlanBuilder::from(right_table_scan)
            .project(vec![col("a")])?
            .build()?;
        let plan = LogicalPlanBuilder::from(left)
            .join_using(
                right,
                JoinType::Right,
                vec![Column::from_name("a".to_string())],
            )?
            .filter(col("test.a").lt_eq(lit(1i64)))?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        Filter: test.a <= Int64(1)
          Right Join: Using test.a = test2.a
            TableScan: test
            Projection: test2.a
              TableScan: test2
        ",
        );
        // filter not duplicated nor pushed down - i.e. noop
        assert_optimized_plan_equal!(
            plan,
            @r"
        Filter: test.a <= Int64(1)
          Right Join: Using test.a = test2.a
            TableScan: test
            Projection: test2.a
              TableScan: test2, full_filters=[test2.a <= Int64(1)]
        "
        )
    }

    /// post-left-join predicate on a column common to both sides is pushed to both sides
    #[test]
    fn filter_using_left_join_on_common() -> Result<()> {
        let table_scan = test_table_scan()?;
        let left = LogicalPlanBuilder::from(table_scan).build()?;
        let right_table_scan = test_table_scan_with_name("test2")?;
        let right = LogicalPlanBuilder::from(right_table_scan)
            .project(vec![col("a")])?
            .build()?;
        let plan = LogicalPlanBuilder::from(left)
            .join_using(
                right,
                JoinType::Left,
                vec![Column::from_name("a".to_string())],
            )?
            .filter(col("a").lt_eq(lit(1i64)))?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        Filter: test.a <= Int64(1)
          Left Join: Using test.a = test2.a
            TableScan: test
            Projection: test2.a
              TableScan: test2
        ",
        );
        // filter sent to left side of the join and to the right
        assert_optimized_plan_equal!(
            plan,
            @r"
        Left Join: Using test.a = test2.a
          TableScan: test, full_filters=[test.a <= Int64(1)]
          Projection: test2.a
            TableScan: test2, full_filters=[test2.a <= Int64(1)]
        "
        )
    }

    /// post-right-join predicate on a column common to both sides is pushed to both sides
    #[test]
    fn filter_using_right_join_on_common() -> Result<()> {
        let table_scan = test_table_scan()?;
        let left = LogicalPlanBuilder::from(table_scan).build()?;
        let right_table_scan = test_table_scan_with_name("test2")?;
        let right = LogicalPlanBuilder::from(right_table_scan)
            .project(vec![col("a")])?
            .build()?;
        let plan = LogicalPlanBuilder::from(left)
            .join_using(
                right,
                JoinType::Right,
                vec![Column::from_name("a".to_string())],
            )?
            .filter(col("test2.a").lt_eq(lit(1i64)))?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        Filter: test2.a <= Int64(1)
          Right Join: Using test.a = test2.a
            TableScan: test
            Projection: test2.a
              TableScan: test2
        ",
        );
        // filter sent to right side of join, sent to the left as well
        assert_optimized_plan_equal!(
            plan,
            @r"
        Right Join: Using test.a = test2.a
          TableScan: test, full_filters=[test.a <= Int64(1)]
          Projection: test2.a
            TableScan: test2, full_filters=[test2.a <= Int64(1)]
        "
        )
    }

    /// single table predicate parts of ON condition should be pushed to both inputs
    #[test]
    fn join_on_with_filter() -> Result<()> {
        let table_scan = test_table_scan()?;
        let left = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a"), col("b"), col("c")])?
            .build()?;
        let right_table_scan = test_table_scan_with_name("test2")?;
        let right = LogicalPlanBuilder::from(right_table_scan)
            .project(vec![col("a"), col("b"), col("c")])?
            .build()?;
        let filter = col("test.c")
            .gt(lit(1u32))
            .and(col("test.b").lt(col("test2.b")))
            .and(col("test2.c").gt(lit(4u32)));
        let plan = LogicalPlanBuilder::from(left)
            .join(
                right,
                JoinType::Inner,
                (vec![Column::from_name("a")], vec![Column::from_name("a")]),
                Some(filter),
            )?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        Inner Join: test.a = test2.a Filter: test.c > UInt32(1) AND test.b < test2.b AND test2.c > UInt32(4)
          Projection: test.a, test.b, test.c
            TableScan: test
          Projection: test2.a, test2.b, test2.c
            TableScan: test2
        ",
        );
        assert_optimized_plan_equal!(
            plan,
            @r"
        Inner Join: test.a = test2.a Filter: test.b < test2.b
          Projection: test.a, test.b, test.c
            TableScan: test, full_filters=[test.c > UInt32(1)]
          Projection: test2.a, test2.b, test2.c
            TableScan: test2, full_filters=[test2.c > UInt32(4)]
        "
        )
    }

    /// join filter should be completely removed after pushdown
    #[test]
    fn join_filter_removed() -> Result<()> {
        let table_scan = test_table_scan()?;
        let left = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a"), col("b"), col("c")])?
            .build()?;
        let right_table_scan = test_table_scan_with_name("test2")?;
        let right = LogicalPlanBuilder::from(right_table_scan)
            .project(vec![col("a"), col("b"), col("c")])?
            .build()?;
        let filter = col("test.b")
            .gt(lit(1u32))
            .and(col("test2.c").gt(lit(4u32)));
        let plan = LogicalPlanBuilder::from(left)
            .join(
                right,
                JoinType::Inner,
                (vec![Column::from_name("a")], vec![Column::from_name("a")]),
                Some(filter),
            )?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        Inner Join: test.a = test2.a Filter: test.b > UInt32(1) AND test2.c > UInt32(4)
          Projection: test.a, test.b, test.c
            TableScan: test
          Projection: test2.a, test2.b, test2.c
            TableScan: test2
        ",
        );
        assert_optimized_plan_equal!(
            plan,
            @r"
        Inner Join: test.a = test2.a
          Projection: test.a, test.b, test.c
            TableScan: test, full_filters=[test.b > UInt32(1)]
          Projection: test2.a, test2.b, test2.c
            TableScan: test2, full_filters=[test2.c > UInt32(4)]
        "
        )
    }

    /// predicate on join key in filter expression should be pushed down to both inputs
    #[test]
    fn join_filter_on_common() -> Result<()> {
        let table_scan = test_table_scan()?;
        let left = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a")])?
            .build()?;
        let right_table_scan = test_table_scan_with_name("test2")?;
        let right = LogicalPlanBuilder::from(right_table_scan)
            .project(vec![col("b")])?
            .build()?;
        let filter = col("test.a").gt(lit(1u32));
        let plan = LogicalPlanBuilder::from(left)
            .join(
                right,
                JoinType::Inner,
                (vec![Column::from_name("a")], vec![Column::from_name("b")]),
                Some(filter),
            )?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        Inner Join: test.a = test2.b Filter: test.a > UInt32(1)
          Projection: test.a
            TableScan: test
          Projection: test2.b
            TableScan: test2
        ",
        );
        assert_optimized_plan_equal!(
            plan,
            @r"
        Inner Join: test.a = test2.b
          Projection: test.a
            TableScan: test, full_filters=[test.a > UInt32(1)]
          Projection: test2.b
            TableScan: test2, full_filters=[test2.b > UInt32(1)]
        "
        )
    }

    /// single table predicate parts of ON condition should be pushed to right input
    #[test]
    fn left_join_on_with_filter() -> Result<()> {
        let table_scan = test_table_scan()?;
        let left = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a"), col("b"), col("c")])?
            .build()?;
        let right_table_scan = test_table_scan_with_name("test2")?;
        let right = LogicalPlanBuilder::from(right_table_scan)
            .project(vec![col("a"), col("b"), col("c")])?
            .build()?;
        let filter = col("test.a")
            .gt(lit(1u32))
            .and(col("test.b").lt(col("test2.b")))
            .and(col("test2.c").gt(lit(4u32)));
        let plan = LogicalPlanBuilder::from(left)
            .join(
                right,
                JoinType::Left,
                (vec![Column::from_name("a")], vec![Column::from_name("a")]),
                Some(filter),
            )?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        Left Join: test.a = test2.a Filter: test.a > UInt32(1) AND test.b < test2.b AND test2.c > UInt32(4)
          Projection: test.a, test.b, test.c
            TableScan: test
          Projection: test2.a, test2.b, test2.c
            TableScan: test2
        ",
        );
        assert_optimized_plan_equal!(
            plan,
            @r"
        Left Join: test.a = test2.a Filter: test.a > UInt32(1) AND test.b < test2.b
          Projection: test.a, test.b, test.c
            TableScan: test
          Projection: test2.a, test2.b, test2.c
            TableScan: test2, full_filters=[test2.a > UInt32(1), test2.c > UInt32(4)]
        "
        )
    }

    /// single table predicate parts of ON condition should be pushed to left input
    #[test]
    fn right_join_on_with_filter() -> Result<()> {
        let table_scan = test_table_scan()?;
        let left = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a"), col("b"), col("c")])?
            .build()?;
        let right_table_scan = test_table_scan_with_name("test2")?;
        let right = LogicalPlanBuilder::from(right_table_scan)
            .project(vec![col("a"), col("b"), col("c")])?
            .build()?;
        let filter = col("test.a")
            .gt(lit(1u32))
            .and(col("test.b").lt(col("test2.b")))
            .and(col("test2.c").gt(lit(4u32)));
        let plan = LogicalPlanBuilder::from(left)
            .join(
                right,
                JoinType::Right,
                (vec![Column::from_name("a")], vec![Column::from_name("a")]),
                Some(filter),
            )?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        Right Join: test.a = test2.a Filter: test.a > UInt32(1) AND test.b < test2.b AND test2.c > UInt32(4)
          Projection: test.a, test.b, test.c
            TableScan: test
          Projection: test2.a, test2.b, test2.c
            TableScan: test2
        ",
        );
        assert_optimized_plan_equal!(
            plan,
            @r"
        Right Join: test.a = test2.a Filter: test.b < test2.b AND test2.c > UInt32(4)
          Projection: test.a, test.b, test.c
            TableScan: test, full_filters=[test.a > UInt32(1)]
          Projection: test2.a, test2.b, test2.c
            TableScan: test2
        "
        )
    }

    /// single table predicate parts of ON condition should not be pushed
    #[test]
    fn full_join_on_with_filter() -> Result<()> {
        let table_scan = test_table_scan()?;
        let left = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a"), col("b"), col("c")])?
            .build()?;
        let right_table_scan = test_table_scan_with_name("test2")?;
        let right = LogicalPlanBuilder::from(right_table_scan)
            .project(vec![col("a"), col("b"), col("c")])?
            .build()?;
        let filter = col("test.a")
            .gt(lit(1u32))
            .and(col("test.b").lt(col("test2.b")))
            .and(col("test2.c").gt(lit(4u32)));
        let plan = LogicalPlanBuilder::from(left)
            .join(
                right,
                JoinType::Full,
                (vec![Column::from_name("a")], vec![Column::from_name("a")]),
                Some(filter),
            )?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        Full Join: test.a = test2.a Filter: test.a > UInt32(1) AND test.b < test2.b AND test2.c > UInt32(4)
          Projection: test.a, test.b, test.c
            TableScan: test
          Projection: test2.a, test2.b, test2.c
            TableScan: test2
        ",
        );
        assert_optimized_plan_equal!(
            plan,
            @r"
        Full Join: test.a = test2.a Filter: test.a > UInt32(1) AND test.b < test2.b AND test2.c > UInt32(4)
          Projection: test.a, test.b, test.c
            TableScan: test
          Projection: test2.a, test2.b, test2.c
            TableScan: test2
        "
        )
    }

    struct PushDownProvider {
        pub filter_support: TableProviderFilterPushDown,
    }

    #[async_trait]
    impl TableSource for PushDownProvider {
        fn schema(&self) -> SchemaRef {
            Arc::new(Schema::new(vec![
                Field::new("a", DataType::Int32, true),
                Field::new("b", DataType::Int32, true),
            ]))
        }

        fn table_type(&self) -> TableType {
            TableType::Base
        }

        fn supports_filters_pushdown(
            &self,
            filters: &[&Expr],
        ) -> Result<Vec<TableProviderFilterPushDown>> {
            Ok((0..filters.len())
                .map(|_| self.filter_support.clone())
                .collect())
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn table_scan_with_pushdown_provider_builder(
        filter_support: TableProviderFilterPushDown,
        filters: Vec<Expr>,
        projection: Option<Vec<usize>>,
    ) -> Result<LogicalPlanBuilder> {
        let test_provider = PushDownProvider { filter_support };

        let table_scan = LogicalPlan::TableScan(TableScan {
            table_name: "test".into(),
            filters,
            projected_schema: Arc::new(DFSchema::try_from(test_provider.schema())?),
            projection,
            source: Arc::new(test_provider),
            fetch: None,
        });

        Ok(LogicalPlanBuilder::from(table_scan))
    }

    fn table_scan_with_pushdown_provider(
        filter_support: TableProviderFilterPushDown,
    ) -> Result<LogicalPlan> {
        table_scan_with_pushdown_provider_builder(filter_support, vec![], None)?
            .filter(col("a").eq(lit(1i64)))?
            .build()
    }

    #[test]
    fn filter_with_table_provider_exact() -> Result<()> {
        let plan = table_scan_with_pushdown_provider(TableProviderFilterPushDown::Exact)?;

        assert_optimized_plan_equal!(
            plan,
            @"TableScan: test, full_filters=[a = Int64(1)]"
        )
    }

    #[test]
    fn filter_with_table_provider_inexact() -> Result<()> {
        let plan =
            table_scan_with_pushdown_provider(TableProviderFilterPushDown::Inexact)?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Filter: a = Int64(1)
          TableScan: test, partial_filters=[a = Int64(1)]
        "
        )
    }

    #[test]
    fn filter_with_table_provider_multiple_invocations() -> Result<()> {
        let plan =
            table_scan_with_pushdown_provider(TableProviderFilterPushDown::Inexact)?;

        let optimized_plan = PushDownFilter::new()
            .rewrite(plan, &OptimizerContext::new())
            .expect("failed to optimize plan")
            .data;

        // Optimizing the same plan multiple times should produce the same plan
        // each time.
        assert_optimized_plan_equal!(
            optimized_plan,
            @r"
        Filter: a = Int64(1)
          TableScan: test, partial_filters=[a = Int64(1)]
        "
        )
    }

    #[test]
    fn filter_with_table_provider_unsupported() -> Result<()> {
        let plan =
            table_scan_with_pushdown_provider(TableProviderFilterPushDown::Unsupported)?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Filter: a = Int64(1)
          TableScan: test
        "
        )
    }

    #[test]
    fn multi_combined_filter() -> Result<()> {
        let plan = table_scan_with_pushdown_provider_builder(
            TableProviderFilterPushDown::Inexact,
            vec![col("a").eq(lit(10i64)), col("b").gt(lit(11i64))],
            Some(vec![0]),
        )?
        .filter(and(col("a").eq(lit(10i64)), col("b").gt(lit(11i64))))?
        .project(vec![col("a"), col("b")])?
        .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: a, b
          Filter: a = Int64(10) AND b > Int64(11)
            TableScan: test projection=[a], partial_filters=[a = Int64(10), b > Int64(11)]
        "
        )
    }

    #[test]
    fn multi_combined_filter_exact() -> Result<()> {
        let plan = table_scan_with_pushdown_provider_builder(
            TableProviderFilterPushDown::Exact,
            vec![],
            Some(vec![0]),
        )?
        .filter(and(col("a").eq(lit(10i64)), col("b").gt(lit(11i64))))?
        .project(vec![col("a"), col("b")])?
        .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: a, b
          TableScan: test projection=[a], full_filters=[a = Int64(10), b > Int64(11)]
        "
        )
    }

    #[test]
    fn test_filter_with_alias() -> Result<()> {
        // in table scan the true col name is 'test.a',
        // but we rename it as 'b', and use col 'b' in filter
        // we need rewrite filter col before push down.
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a").alias("b"), col("c")])?
            .filter(and(col("b").gt(lit(10i64)), col("c").gt(lit(10i64))))?
            .build()?;

        // filter on col b
        assert_snapshot!(plan,
        @r"
        Filter: b > Int64(10) AND test.c > Int64(10)
          Projection: test.a AS b, test.c
            TableScan: test
        ",
        );
        // rewrite filter col b to test.a
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a AS b, test.c
          TableScan: test, full_filters=[test.a > Int64(10), test.c > Int64(10)]
        "
        )
    }

    #[test]
    fn test_filter_with_alias_2() -> Result<()> {
        // in table scan the true col name is 'test.a',
        // but we rename it as 'b', and use col 'b' in filter
        // we need rewrite filter col before push down.
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a").alias("b"), col("c")])?
            .project(vec![col("b"), col("c")])?
            .filter(and(col("b").gt(lit(10i64)), col("c").gt(lit(10i64))))?
            .build()?;

        // filter on col b
        assert_snapshot!(plan,
        @r"
        Filter: b > Int64(10) AND test.c > Int64(10)
          Projection: b, test.c
            Projection: test.a AS b, test.c
              TableScan: test
        ",
        );
        // rewrite filter col b to test.a
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: b, test.c
          Projection: test.a AS b, test.c
            TableScan: test, full_filters=[test.a > Int64(10), test.c > Int64(10)]
        "
        )
    }

    #[test]
    fn test_filter_with_multi_alias() -> Result<()> {
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a").alias("b"), col("c").alias("d")])?
            .filter(and(col("b").gt(lit(10i64)), col("d").gt(lit(10i64))))?
            .build()?;

        // filter on col b and d
        assert_snapshot!(plan,
        @r"
        Filter: b > Int64(10) AND d > Int64(10)
          Projection: test.a AS b, test.c AS d
            TableScan: test
        ",
        );
        // rewrite filter col b to test.a, col d to test.c
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a AS b, test.c AS d
          TableScan: test, full_filters=[test.a > Int64(10), test.c > Int64(10)]
        "
        )
    }

    /// predicate on join key in filter expression should be pushed down to both inputs
    #[test]
    fn join_filter_with_alias() -> Result<()> {
        let table_scan = test_table_scan()?;
        let left = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a").alias("c")])?
            .build()?;
        let right_table_scan = test_table_scan_with_name("test2")?;
        let right = LogicalPlanBuilder::from(right_table_scan)
            .project(vec![col("b").alias("d")])?
            .build()?;
        let filter = col("c").gt(lit(1u32));
        let plan = LogicalPlanBuilder::from(left)
            .join(
                right,
                JoinType::Inner,
                (vec![Column::from_name("c")], vec![Column::from_name("d")]),
                Some(filter),
            )?
            .build()?;

        assert_snapshot!(plan,
        @r"
        Inner Join: c = d Filter: c > UInt32(1)
          Projection: test.a AS c
            TableScan: test
          Projection: test2.b AS d
            TableScan: test2
        ",
        );
        // Change filter on col `c`, 'd' to `test.a`, 'test.b'
        assert_optimized_plan_equal!(
            plan,
            @r"
        Inner Join: c = d
          Projection: test.a AS c
            TableScan: test, full_filters=[test.a > UInt32(1)]
          Projection: test2.b AS d
            TableScan: test2, full_filters=[test2.b > UInt32(1)]
        "
        )
    }

    #[test]
    fn test_in_filter_with_alias() -> Result<()> {
        // in table scan the true col name is 'test.a',
        // but we rename it as 'b', and use col 'b' in filter
        // we need rewrite filter col before push down.
        let table_scan = test_table_scan()?;
        let filter_value = vec![lit(1u32), lit(2u32), lit(3u32), lit(4u32)];
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a").alias("b"), col("c")])?
            .filter(in_list(col("b"), filter_value, false))?
            .build()?;

        // filter on col b
        assert_snapshot!(plan,
        @r"
        Filter: b IN ([UInt32(1), UInt32(2), UInt32(3), UInt32(4)])
          Projection: test.a AS b, test.c
            TableScan: test
        ",
        );
        // rewrite filter col b to test.a
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a AS b, test.c
          TableScan: test, full_filters=[test.a IN ([UInt32(1), UInt32(2), UInt32(3), UInt32(4)])]
        "
        )
    }

    #[test]
    fn test_in_filter_with_alias_2() -> Result<()> {
        // in table scan the true col name is 'test.a',
        // but we rename it as 'b', and use col 'b' in filter
        // we need rewrite filter col before push down.
        let table_scan = test_table_scan()?;
        let filter_value = vec![lit(1u32), lit(2u32), lit(3u32), lit(4u32)];
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a").alias("b"), col("c")])?
            .project(vec![col("b"), col("c")])?
            .filter(in_list(col("b"), filter_value, false))?
            .build()?;

        // filter on col b
        assert_snapshot!(plan,
        @r"
        Filter: b IN ([UInt32(1), UInt32(2), UInt32(3), UInt32(4)])
          Projection: b, test.c
            Projection: test.a AS b, test.c
              TableScan: test
        ",
        );
        // rewrite filter col b to test.a
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: b, test.c
          Projection: test.a AS b, test.c
            TableScan: test, full_filters=[test.a IN ([UInt32(1), UInt32(2), UInt32(3), UInt32(4)])]
        "
        )
    }

    #[test]
    fn test_in_subquery_with_alias() -> Result<()> {
        // in table scan the true col name is 'test.a',
        // but we rename it as 'b', and use col 'b' in subquery filter
        let table_scan = test_table_scan()?;
        let table_scan_sq = test_table_scan_with_name("sq")?;
        let subplan = Arc::new(
            LogicalPlanBuilder::from(table_scan_sq)
                .project(vec![col("c")])?
                .build()?,
        );
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a").alias("b"), col("c")])?
            .filter(in_subquery(col("b"), subplan))?
            .build()?;

        // filter on col b in subquery
        assert_snapshot!(plan,
        @r"
        Filter: b IN (<subquery>)
          Subquery:
            Projection: sq.c
              TableScan: sq
          Projection: test.a AS b, test.c
            TableScan: test
        ",
        );
        // rewrite filter col b to test.a
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a AS b, test.c
          TableScan: test, full_filters=[test.a IN (<subquery>)]
            Subquery:
              Projection: sq.c
                TableScan: sq
        "
        )
    }

    #[test]
    fn test_propagation_of_optimized_inner_filters_with_projections() -> Result<()> {
        // SELECT a FROM (SELECT 1 AS a) b WHERE b.a = 1
        let plan = LogicalPlanBuilder::empty(true)
            .project(vec![lit(0i64).alias("a")])?
            .alias("b")?
            .project(vec![col("b.a")])?
            .alias("b")?
            .filter(col("b.a").eq(lit(1i64)))?
            .project(vec![col("b.a")])?
            .build()?;

        assert_snapshot!(plan,
        @r"
        Projection: b.a
          Filter: b.a = Int64(1)
            SubqueryAlias: b
              Projection: b.a
                SubqueryAlias: b
                  Projection: Int64(0) AS a
                    EmptyRelation: rows=1
        ",
        );
        // Ensure that the predicate without any columns (0 = 1) is
        // still there.
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: b.a
          SubqueryAlias: b
            Projection: b.a
              SubqueryAlias: b
                Projection: Int64(0) AS a
                  Filter: Int64(0) = Int64(1)
                    EmptyRelation: rows=1
        "
        )
    }

    #[test]
    fn test_crossjoin_with_or_clause() -> Result<()> {
        // select * from test,test1 where (test.a = test1.a and test.b > 1) or (test.b = test1.b and test.c < 10);
        let table_scan = test_table_scan()?;
        let left = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a"), col("b"), col("c")])?
            .build()?;
        let right_table_scan = test_table_scan_with_name("test1")?;
        let right = LogicalPlanBuilder::from(right_table_scan)
            .project(vec![col("a").alias("d"), col("a").alias("e")])?
            .build()?;
        let filter = or(
            and(col("a").eq(col("d")), col("b").gt(lit(1u32))),
            and(col("b").eq(col("e")), col("c").lt(lit(10u32))),
        );
        let plan = LogicalPlanBuilder::from(left)
            .cross_join(right)?
            .filter(filter)?
            .build()?;

        assert_optimized_plan_eq_with_rewrite_predicate!(plan.clone(), @r"
        Inner Join:  Filter: test.a = d AND test.b > UInt32(1) OR test.b = e AND test.c < UInt32(10)
          Projection: test.a, test.b, test.c
            TableScan: test, full_filters=[test.b > UInt32(1) OR test.c < UInt32(10)]
          Projection: test1.a AS d, test1.a AS e
            TableScan: test1
        ")?;

        // Originally global state which can help to avoid duplicate Filters been generated and pushed down.
        // Now the global state is removed. Need to double confirm that avoid duplicate Filters.
        let optimized_plan = PushDownFilter::new()
            .rewrite(plan, &OptimizerContext::new())
            .expect("failed to optimize plan")
            .data;
        assert_optimized_plan_equal!(
            optimized_plan,
            @r"
        Inner Join:  Filter: test.a = d AND test.b > UInt32(1) OR test.b = e AND test.c < UInt32(10)
          Projection: test.a, test.b, test.c
            TableScan: test, full_filters=[test.b > UInt32(1) OR test.c < UInt32(10)]
          Projection: test1.a AS d, test1.a AS e
            TableScan: test1
        "
        )
    }

    #[test]
    fn left_semi_join() -> Result<()> {
        let left = test_table_scan_with_name("test1")?;
        let right_table_scan = test_table_scan_with_name("test2")?;
        let right = LogicalPlanBuilder::from(right_table_scan)
            .project(vec![col("a"), col("b")])?
            .build()?;
        let plan = LogicalPlanBuilder::from(left)
            .join(
                right,
                JoinType::LeftSemi,
                (
                    vec![Column::from_qualified_name("test1.a")],
                    vec![Column::from_qualified_name("test2.a")],
                ),
                None,
            )?
            .filter(col("test2.a").lt_eq(lit(1i64)))?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        Filter: test2.a <= Int64(1)
          LeftSemi Join: test1.a = test2.a
            TableScan: test1
            Projection: test2.a, test2.b
              TableScan: test2
        ",
        );
        // Inferred the predicate `test1.a <= Int64(1)` and push it down to the left side.
        assert_optimized_plan_equal!(
            plan,
            @r"
        Filter: test2.a <= Int64(1)
          LeftSemi Join: test1.a = test2.a
            TableScan: test1, full_filters=[test1.a <= Int64(1)]
            Projection: test2.a, test2.b
              TableScan: test2
        "
        )
    }

    #[test]
    fn left_semi_join_with_filters() -> Result<()> {
        let left = test_table_scan_with_name("test1")?;
        let right_table_scan = test_table_scan_with_name("test2")?;
        let right = LogicalPlanBuilder::from(right_table_scan)
            .project(vec![col("a"), col("b")])?
            .build()?;
        let plan = LogicalPlanBuilder::from(left)
            .join(
                right,
                JoinType::LeftSemi,
                (
                    vec![Column::from_qualified_name("test1.a")],
                    vec![Column::from_qualified_name("test2.a")],
                ),
                Some(
                    col("test1.b")
                        .gt(lit(1u32))
                        .and(col("test2.b").gt(lit(2u32))),
                ),
            )?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        LeftSemi Join: test1.a = test2.a Filter: test1.b > UInt32(1) AND test2.b > UInt32(2)
          TableScan: test1
          Projection: test2.a, test2.b
            TableScan: test2
        ",
        );
        // Both side will be pushed down.
        assert_optimized_plan_equal!(
            plan,
            @r"
        LeftSemi Join: test1.a = test2.a
          TableScan: test1, full_filters=[test1.b > UInt32(1)]
          Projection: test2.a, test2.b
            TableScan: test2, full_filters=[test2.b > UInt32(2)]
        "
        )
    }

    #[test]
    fn right_semi_join() -> Result<()> {
        let left = test_table_scan_with_name("test1")?;
        let right_table_scan = test_table_scan_with_name("test2")?;
        let right = LogicalPlanBuilder::from(right_table_scan)
            .project(vec![col("a"), col("b")])?
            .build()?;
        let plan = LogicalPlanBuilder::from(left)
            .join(
                right,
                JoinType::RightSemi,
                (
                    vec![Column::from_qualified_name("test1.a")],
                    vec![Column::from_qualified_name("test2.a")],
                ),
                None,
            )?
            .filter(col("test1.a").lt_eq(lit(1i64)))?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        Filter: test1.a <= Int64(1)
          RightSemi Join: test1.a = test2.a
            TableScan: test1
            Projection: test2.a, test2.b
              TableScan: test2
        ",
        );
        // Inferred the predicate `test2.a <= Int64(1)` and push it down to the right side.
        assert_optimized_plan_equal!(
            plan,
            @r"
        Filter: test1.a <= Int64(1)
          RightSemi Join: test1.a = test2.a
            TableScan: test1
            Projection: test2.a, test2.b
              TableScan: test2, full_filters=[test2.a <= Int64(1)]
        "
        )
    }

    #[test]
    fn right_semi_join_with_filters() -> Result<()> {
        let left = test_table_scan_with_name("test1")?;
        let right_table_scan = test_table_scan_with_name("test2")?;
        let right = LogicalPlanBuilder::from(right_table_scan)
            .project(vec![col("a"), col("b")])?
            .build()?;
        let plan = LogicalPlanBuilder::from(left)
            .join(
                right,
                JoinType::RightSemi,
                (
                    vec![Column::from_qualified_name("test1.a")],
                    vec![Column::from_qualified_name("test2.a")],
                ),
                Some(
                    col("test1.b")
                        .gt(lit(1u32))
                        .and(col("test2.b").gt(lit(2u32))),
                ),
            )?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        RightSemi Join: test1.a = test2.a Filter: test1.b > UInt32(1) AND test2.b > UInt32(2)
          TableScan: test1
          Projection: test2.a, test2.b
            TableScan: test2
        ",
        );
        // Both side will be pushed down.
        assert_optimized_plan_equal!(
            plan,
            @r"
        RightSemi Join: test1.a = test2.a
          TableScan: test1, full_filters=[test1.b > UInt32(1)]
          Projection: test2.a, test2.b
            TableScan: test2, full_filters=[test2.b > UInt32(2)]
        "
        )
    }

    #[test]
    fn left_anti_join() -> Result<()> {
        let table_scan = test_table_scan_with_name("test1")?;
        let left = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a"), col("b")])?
            .build()?;
        let right_table_scan = test_table_scan_with_name("test2")?;
        let right = LogicalPlanBuilder::from(right_table_scan)
            .project(vec![col("a"), col("b")])?
            .build()?;
        let plan = LogicalPlanBuilder::from(left)
            .join(
                right,
                JoinType::LeftAnti,
                (
                    vec![Column::from_qualified_name("test1.a")],
                    vec![Column::from_qualified_name("test2.a")],
                ),
                None,
            )?
            .filter(col("test2.a").gt(lit(2u32)))?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        Filter: test2.a > UInt32(2)
          LeftAnti Join: test1.a = test2.a
            Projection: test1.a, test1.b
              TableScan: test1
            Projection: test2.a, test2.b
              TableScan: test2
        ",
        );
        // For left anti, filter of the right side filter can be pushed down.
        assert_optimized_plan_equal!(
            plan,
            @r"
        Filter: test2.a > UInt32(2)
          LeftAnti Join: test1.a = test2.a
            Projection: test1.a, test1.b
              TableScan: test1, full_filters=[test1.a > UInt32(2)]
            Projection: test2.a, test2.b
              TableScan: test2
        "
        )
    }

    #[test]
    fn left_anti_join_with_filters() -> Result<()> {
        let table_scan = test_table_scan_with_name("test1")?;
        let left = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a"), col("b")])?
            .build()?;
        let right_table_scan = test_table_scan_with_name("test2")?;
        let right = LogicalPlanBuilder::from(right_table_scan)
            .project(vec![col("a"), col("b")])?
            .build()?;
        let plan = LogicalPlanBuilder::from(left)
            .join(
                right,
                JoinType::LeftAnti,
                (
                    vec![Column::from_qualified_name("test1.a")],
                    vec![Column::from_qualified_name("test2.a")],
                ),
                Some(
                    col("test1.b")
                        .gt(lit(1u32))
                        .and(col("test2.b").gt(lit(2u32))),
                ),
            )?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        LeftAnti Join: test1.a = test2.a Filter: test1.b > UInt32(1) AND test2.b > UInt32(2)
          Projection: test1.a, test1.b
            TableScan: test1
          Projection: test2.a, test2.b
            TableScan: test2
        ",
        );
        // For left anti, filter of the right side filter can be pushed down.
        assert_optimized_plan_equal!(
            plan,
            @r"
        LeftAnti Join: test1.a = test2.a Filter: test1.b > UInt32(1)
          Projection: test1.a, test1.b
            TableScan: test1
          Projection: test2.a, test2.b
            TableScan: test2, full_filters=[test2.b > UInt32(2)]
        "
        )
    }

    #[test]
    fn right_anti_join() -> Result<()> {
        let table_scan = test_table_scan_with_name("test1")?;
        let left = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a"), col("b")])?
            .build()?;
        let right_table_scan = test_table_scan_with_name("test2")?;
        let right = LogicalPlanBuilder::from(right_table_scan)
            .project(vec![col("a"), col("b")])?
            .build()?;
        let plan = LogicalPlanBuilder::from(left)
            .join(
                right,
                JoinType::RightAnti,
                (
                    vec![Column::from_qualified_name("test1.a")],
                    vec![Column::from_qualified_name("test2.a")],
                ),
                None,
            )?
            .filter(col("test1.a").gt(lit(2u32)))?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        Filter: test1.a > UInt32(2)
          RightAnti Join: test1.a = test2.a
            Projection: test1.a, test1.b
              TableScan: test1
            Projection: test2.a, test2.b
              TableScan: test2
        ",
        );
        // For right anti, filter of the left side can be pushed down.
        assert_optimized_plan_equal!(
            plan,
            @r"
        Filter: test1.a > UInt32(2)
          RightAnti Join: test1.a = test2.a
            Projection: test1.a, test1.b
              TableScan: test1
            Projection: test2.a, test2.b
              TableScan: test2, full_filters=[test2.a > UInt32(2)]
        "
        )
    }

    #[test]
    fn right_anti_join_with_filters() -> Result<()> {
        let table_scan = test_table_scan_with_name("test1")?;
        let left = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a"), col("b")])?
            .build()?;
        let right_table_scan = test_table_scan_with_name("test2")?;
        let right = LogicalPlanBuilder::from(right_table_scan)
            .project(vec![col("a"), col("b")])?
            .build()?;
        let plan = LogicalPlanBuilder::from(left)
            .join(
                right,
                JoinType::RightAnti,
                (
                    vec![Column::from_qualified_name("test1.a")],
                    vec![Column::from_qualified_name("test2.a")],
                ),
                Some(
                    col("test1.b")
                        .gt(lit(1u32))
                        .and(col("test2.b").gt(lit(2u32))),
                ),
            )?
            .build()?;

        // not part of the test, just good to know:
        assert_snapshot!(plan,
        @r"
        RightAnti Join: test1.a = test2.a Filter: test1.b > UInt32(1) AND test2.b > UInt32(2)
          Projection: test1.a, test1.b
            TableScan: test1
          Projection: test2.a, test2.b
            TableScan: test2
        ",
        );
        // For right anti, filter of the left side can be pushed down.
        assert_optimized_plan_equal!(
            plan,
            @r"
        RightAnti Join: test1.a = test2.a Filter: test2.b > UInt32(2)
          Projection: test1.a, test1.b
            TableScan: test1, full_filters=[test1.b > UInt32(1)]
          Projection: test2.a, test2.b
            TableScan: test2
        "
        )
    }

    #[derive(Debug, PartialEq, Eq, Hash)]
    struct TestScalarUDF {
        signature: Signature,
    }

    impl ScalarUDFImpl for TestScalarUDF {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn name(&self) -> &str {
            "TestScalarUDF"
        }

        fn signature(&self) -> &Signature {
            &self.signature
        }

        fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
            Ok(DataType::Int32)
        }

        fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            Ok(ColumnarValue::Scalar(ScalarValue::from(1)))
        }
    }

    #[test]
    fn test_push_down_volatile_function_in_aggregate() -> Result<()> {
        // SELECT t.a, t.r FROM (SELECT a, sum(b),  TestScalarUDF()+1 AS r FROM test1 GROUP BY a) AS t WHERE t.a > 5 AND t.r > 0.5;
        let table_scan = test_table_scan_with_name("test1")?;
        let fun = ScalarUDF::new_from_impl(TestScalarUDF {
            signature: Signature::exact(vec![], Volatility::Volatile),
        });
        let expr = Expr::ScalarFunction(ScalarFunction::new_udf(Arc::new(fun), vec![]));

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("a")], vec![sum(col("b"))])?
            .project(vec![col("a"), sum(col("b")), add(expr, lit(1)).alias("r")])?
            .alias("t")?
            .filter(col("t.a").gt(lit(5)).and(col("t.r").gt(lit(0.5))))?
            .project(vec![col("t.a"), col("t.r")])?
            .build()?;

        assert_snapshot!(plan,
        @r"
        Projection: t.a, t.r
          Filter: t.a > Int32(5) AND t.r > Float64(0.5)
            SubqueryAlias: t
              Projection: test1.a, sum(test1.b), TestScalarUDF() + Int32(1) AS r
                Aggregate: groupBy=[[test1.a]], aggr=[[sum(test1.b)]]
                  TableScan: test1
        ",
        );
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: t.a, t.r
          SubqueryAlias: t
            Filter: r > Float64(0.5)
              Projection: test1.a, sum(test1.b), TestScalarUDF() + Int32(1) AS r
                Aggregate: groupBy=[[test1.a]], aggr=[[sum(test1.b)]]
                  TableScan: test1, full_filters=[test1.a > Int32(5)]
        "
        )
    }

    #[test]
    fn test_push_down_volatile_function_in_join() -> Result<()> {
        // SELECT t.a, t.r FROM (SELECT test1.a AS a, TestScalarUDF() AS r FROM test1 join test2 ON test1.a = test2.a) AS t WHERE t.r > 0.5;
        let table_scan = test_table_scan_with_name("test1")?;
        let fun = ScalarUDF::new_from_impl(TestScalarUDF {
            signature: Signature::exact(vec![], Volatility::Volatile),
        });
        let expr = Expr::ScalarFunction(ScalarFunction::new_udf(Arc::new(fun), vec![]));
        let left = LogicalPlanBuilder::from(table_scan).build()?;
        let right_table_scan = test_table_scan_with_name("test2")?;
        let right = LogicalPlanBuilder::from(right_table_scan).build()?;
        let plan = LogicalPlanBuilder::from(left)
            .join(
                right,
                JoinType::Inner,
                (
                    vec![Column::from_qualified_name("test1.a")],
                    vec![Column::from_qualified_name("test2.a")],
                ),
                None,
            )?
            .project(vec![col("test1.a").alias("a"), expr.alias("r")])?
            .alias("t")?
            .filter(col("t.r").gt(lit(0.8)))?
            .project(vec![col("t.a"), col("t.r")])?
            .build()?;

        assert_snapshot!(plan,
        @r"
        Projection: t.a, t.r
          Filter: t.r > Float64(0.8)
            SubqueryAlias: t
              Projection: test1.a AS a, TestScalarUDF() AS r
                Inner Join: test1.a = test2.a
                  TableScan: test1
                  TableScan: test2
        ",
        );
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: t.a, t.r
          SubqueryAlias: t
            Filter: r > Float64(0.8)
              Projection: test1.a AS a, TestScalarUDF() AS r
                Inner Join: test1.a = test2.a
                  TableScan: test1
                  TableScan: test2
        "
        )
    }

    #[test]
    fn test_push_down_volatile_table_scan() -> Result<()> {
        // SELECT test.a, test.b FROM test as t WHERE TestScalarUDF() > 0.1;
        let table_scan = test_table_scan()?;
        let fun = ScalarUDF::new_from_impl(TestScalarUDF {
            signature: Signature::exact(vec![], Volatility::Volatile),
        });
        let expr = Expr::ScalarFunction(ScalarFunction::new_udf(Arc::new(fun), vec![]));
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a"), col("b")])?
            .filter(expr.gt(lit(0.1)))?
            .build()?;

        assert_snapshot!(plan,
        @r"
        Filter: TestScalarUDF() > Float64(0.1)
          Projection: test.a, test.b
            TableScan: test
        ",
        );
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a, test.b
          Filter: TestScalarUDF() > Float64(0.1)
            TableScan: test
        "
        )
    }

    #[test]
    fn test_push_down_volatile_mixed_table_scan() -> Result<()> {
        // SELECT test.a, test.b FROM test as t WHERE TestScalarUDF() > 0.1 and test.a > 5 and test.b > 10;
        let table_scan = test_table_scan()?;
        let fun = ScalarUDF::new_from_impl(TestScalarUDF {
            signature: Signature::exact(vec![], Volatility::Volatile),
        });
        let expr = Expr::ScalarFunction(ScalarFunction::new_udf(Arc::new(fun), vec![]));
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a"), col("b")])?
            .filter(
                expr.gt(lit(0.1))
                    .and(col("t.a").gt(lit(5)))
                    .and(col("t.b").gt(lit(10))),
            )?
            .build()?;

        assert_snapshot!(plan,
        @r"
        Filter: TestScalarUDF() > Float64(0.1) AND t.a > Int32(5) AND t.b > Int32(10)
          Projection: test.a, test.b
            TableScan: test
        ",
        );
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a, test.b
          Filter: TestScalarUDF() > Float64(0.1)
            TableScan: test, full_filters=[t.a > Int32(5), t.b > Int32(10)]
        "
        )
    }

    #[test]
    fn test_push_down_volatile_mixed_unsupported_table_scan() -> Result<()> {
        // SELECT test.a, test.b FROM test as t WHERE TestScalarUDF() > 0.1 and test.a > 5 and test.b > 10;
        let fun = ScalarUDF::new_from_impl(TestScalarUDF {
            signature: Signature::exact(vec![], Volatility::Volatile),
        });
        let expr = Expr::ScalarFunction(ScalarFunction::new_udf(Arc::new(fun), vec![]));
        let plan = table_scan_with_pushdown_provider_builder(
            TableProviderFilterPushDown::Unsupported,
            vec![],
            None,
        )?
        .project(vec![col("a"), col("b")])?
        .filter(
            expr.gt(lit(0.1))
                .and(col("t.a").gt(lit(5)))
                .and(col("t.b").gt(lit(10))),
        )?
        .build()?;

        assert_snapshot!(plan,
        @r"
        Filter: TestScalarUDF() > Float64(0.1) AND t.a > Int32(5) AND t.b > Int32(10)
          Projection: a, b
            TableScan: test
        ",
        );
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: a, b
          Filter: t.a > Int32(5) AND t.b > Int32(10) AND TestScalarUDF() > Float64(0.1)
            TableScan: test
        "
        )
    }

    #[test]
    fn test_push_down_filter_to_user_defined_node() -> Result<()> {
        // Define a custom user-defined logical node
        #[derive(Debug, Hash, Eq, PartialEq)]
        struct TestUserNode {
            schema: DFSchemaRef,
        }

        impl PartialOrd for TestUserNode {
            fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
                None
            }
        }

        impl TestUserNode {
            fn new() -> Self {
                let schema = Arc::new(
                    DFSchema::new_with_metadata(
                        vec![(None, Field::new("a", DataType::Int64, false).into())],
                        Default::default(),
                    )
                    .unwrap(),
                );

                Self { schema }
            }
        }

        impl UserDefinedLogicalNodeCore for TestUserNode {
            fn name(&self) -> &str {
                "test_node"
            }

            fn inputs(&self) -> Vec<&LogicalPlan> {
                vec![]
            }

            fn schema(&self) -> &DFSchemaRef {
                &self.schema
            }

            fn expressions(&self) -> Vec<Expr> {
                vec![]
            }

            fn fmt_for_explain(&self, f: &mut Formatter) -> std::fmt::Result {
                write!(f, "TestUserNode")
            }

            fn with_exprs_and_inputs(
                &self,
                exprs: Vec<Expr>,
                inputs: Vec<LogicalPlan>,
            ) -> Result<Self> {
                assert!(exprs.is_empty());
                assert!(inputs.is_empty());
                Ok(Self {
                    schema: Arc::clone(&self.schema),
                })
            }
        }

        // Create a node and build a plan with a filter
        let node = LogicalPlan::Extension(Extension {
            node: Arc::new(TestUserNode::new()),
        });

        let plan = LogicalPlanBuilder::from(node).filter(lit(false))?.build()?;

        // Check the original plan format (not part of the test assertions)
        assert_snapshot!(plan,
        @r"
        Filter: Boolean(false)
          TestUserNode
        ",
        );
        // Check that the filter is pushed down to the user-defined node
        assert_optimized_plan_equal!(
            plan,
            @r"
        Filter: Boolean(false)
          TestUserNode
        "
        )
    }

    /// Test that filters are NOT pushed through MoveTowardsLeafNodes projections.
    /// These are cheap expressions (like get_field) where re-inlining into a filter
    /// has no benefit and causes optimizer instability — ExtractLeafExpressions will
    /// undo the push-down, creating an infinite loop that runs until the iteration
    /// limit is hit.
    #[test]
    fn filter_not_pushed_through_move_towards_leaves_projection() -> Result<()> {
        let table_scan = test_table_scan()?;

        // Create a projection with a MoveTowardsLeafNodes expression
        let proj = LogicalPlanBuilder::from(table_scan)
            .project(vec![
                leaf_udf_expr(col("a")).alias("val"),
                col("b"),
                col("c"),
            ])?
            .build()?;

        // Put a filter on the MoveTowardsLeafNodes column
        let plan = LogicalPlanBuilder::from(proj)
            .filter(col("val").gt(lit(150i64)))?
            .build()?;

        // Filter should NOT be pushed through — val maps to a MoveTowardsLeafNodes expr
        assert_optimized_plan_equal!(
            plan,
            @r"
        Filter: val > Int64(150)
          Projection: leaf_udf(test.a) AS val, test.b, test.c
            TableScan: test
        "
        )
    }

    /// Test mixed predicates: Column predicate pushed, MoveTowardsLeafNodes kept.
    #[test]
    fn filter_mixed_predicates_partial_push() -> Result<()> {
        let table_scan = test_table_scan()?;

        // Create a projection with both MoveTowardsLeafNodes and Column expressions
        let proj = LogicalPlanBuilder::from(table_scan)
            .project(vec![
                leaf_udf_expr(col("a")).alias("val"),
                col("b"),
                col("c"),
            ])?
            .build()?;

        // Filter with both: val > 150 (MoveTowardsLeafNodes) AND b > 5 (Column)
        let plan = LogicalPlanBuilder::from(proj)
            .filter(col("val").gt(lit(150i64)).and(col("b").gt(lit(5i64))))?
            .build()?;

        // val > 150 should be kept above, b > 5 should be pushed through
        assert_optimized_plan_equal!(
            plan,
            @r"
        Filter: val > Int64(150)
          Projection: leaf_udf(test.a) AS val, test.b, test.c
            TableScan: test, full_filters=[test.b > Int64(5)]
        "
        )
    }

    #[test]
    fn filter_not_pushed_down_through_table_scan_with_fetch() -> Result<()> {
        let scan = test_table_scan()?;
        let scan_with_fetch = match scan {
            LogicalPlan::TableScan(scan) => LogicalPlan::TableScan(TableScan {
                fetch: Some(10),
                ..scan
            }),
            _ => unreachable!(),
        };
        let plan = LogicalPlanBuilder::from(scan_with_fetch)
            .filter(col("a").gt(lit(10i64)))?
            .build()?;
        // Filter must NOT be pushed into the table scan when it has a fetch (limit)
        assert_optimized_plan_equal!(
            plan,
            @r"
        Filter: test.a > Int64(10)
          TableScan: test, fetch=10
        "
        )
    }

    #[test]
    fn filter_push_down_through_sort_without_fetch() -> Result<()> {
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .sort(vec![col("a").sort(true, true)])?
            .filter(col("a").gt(lit(10i64)))?
            .build()?;
        // Filter should be pushed below the sort
        assert_optimized_plan_equal!(
            plan,
            @r"
        Sort: test.a ASC NULLS FIRST
          TableScan: test, full_filters=[test.a > Int64(10)]
        "
        )
    }

    #[test]
    fn filter_not_pushed_down_through_sort_with_fetch() -> Result<()> {
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .sort_with_limit(vec![col("a").sort(true, true)], Some(5))?
            .filter(col("a").gt(lit(10i64)))?
            .build()?;
        // Filter must NOT be pushed below the sort when it has a fetch (limit),
        // because the limit should apply before the filter.
        assert_optimized_plan_equal!(
            plan,
            @r"
        Filter: test.a > Int64(10)
          Sort: test.a ASC NULLS FIRST, fetch=5
            TableScan: test
        "
        )
    }

mod extension_predicate_permission {
//! Whole-predicate extension permission preserves zero-column expressions.
use crate::{OptimizerContext, OptimizerRule, push_down_filter::PushDownFilter};
use datafusion_common::{DFSchemaRef, Result};
use datafusion_expr::{
    Expr, Extension, LogicalPlan, LogicalPlanBuilder, ScalarUDF, Signature, SimpleScalarUDF,
    UserDefinedLogicalNodeCore, Volatility,
};
use std::{fmt, sync::Arc};

#[derive(Debug, PartialEq, Eq, PartialOrd, Hash)]
struct Boundary(LogicalPlan);
impl UserDefinedLogicalNodeCore for Boundary {
    fn name(&self) -> &str {
        "Boundary"
    }
    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.0]
    }
    fn schema(&self) -> &DFSchemaRef {
        self.0.schema()
    }
    fn expressions(&self) -> Vec<Expr> {
        Vec::new()
    }
    fn supports_predicate_pushdown(&self, _: &Expr) -> bool {
        false
    }
    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Boundary")
    }
    fn with_exprs_and_inputs(
        &self,
        expressions: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        assert!(expressions.is_empty());
        assert_eq!(inputs.len(), 1);
        Ok(Self(inputs.remove(0)))
    }
}

fn zero_argument_volatile_predicate() -> Expr {
    ScalarUDF::new_from_impl(SimpleScalarUDF::new_with_signature(
        "untrusted_zero_predicate",
        Signature::exact(Vec::new(), Volatility::Volatile),
        arrow::datatypes::DataType::Boolean,
        Arc::new(|_| panic!("the optimizer must not execute this predicate")),
    ))
    .call(Vec::new())
}

#[test]
fn explicit_extension_predicate_permission_keeps_zero_column_predicate_outside() -> Result<()> {
    let input = LogicalPlanBuilder::empty(true).build()?;
    let predicate = zero_argument_volatile_predicate();
    assert!(predicate.column_refs().is_empty());
    let plan = LogicalPlanBuilder::from(LogicalPlan::Extension(Extension {
        node: Arc::new(Boundary(input)),
    }))
    .filter(predicate.clone())?
    .build()?;
    let rewritten = PushDownFilter::new()
        .rewrite(plan, &OptimizerContext::new())?
        .data;
    let LogicalPlan::Filter(filter) = rewritten else {
        panic!("predicate crossed a semantic boundary")
    };
    assert_eq!(filter.predicate, predicate);
    let LogicalPlan::Extension(extension) = filter.input.as_ref() else {
        panic!("same boundary disappeared")
    };
    assert!(matches!(
        extension.node.inputs()[0],
        LogicalPlan::EmptyRelation(_)
    ));
    Ok(())
}

#[test]
fn default_extension_predicate_permission_preserves_previous_zero_column_pushdown() -> Result<()> {
    let input = LogicalPlanBuilder::empty(true).build()?;
    let plan = LogicalPlanBuilder::from(crate::test::user_defined::new(input))
        .filter(zero_argument_volatile_predicate())?
        .build()?;
    let rewritten = PushDownFilter::new()
        .rewrite(plan, &OptimizerContext::new())?
        .data;
    let LogicalPlan::Extension(extension) = rewritten else {
        panic!("default extension no longer preserves its pushdown policy")
    };
    assert!(matches!(extension.node.inputs()[0], LogicalPlan::Filter(_)));
    Ok(())
}
}
