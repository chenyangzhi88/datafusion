use super::super::*;

#[test]
fn wide_rollup_schema_rebuild_keeps_exactly_one_nonnullable_binary_id() -> Result<()> {
    for width in [65usize, 128, 257] {
        let schema = Schema::new(
            (0..width)
                .map(|i| Field::new(format!("k{i}"), DataType::Int32, true))
                .collect::<Vec<_>>(),
        );
        let input = Arc::new(table_scan(TableReference::none(), &schema, None)?.build()?);
        let keys = (0..width).map(|i| col(format!("k{i}"))).collect::<Vec<_>>();
        let grouped = Aggregate::try_new(
            input.clone(),
            vec![Expr::GroupingSet(GroupingSet::Rollup(keys.clone()))],
            vec![count(lit(1))],
        )?;
        assert_eq!(grouped.schema.fields().len(), width + 2);
        assert_eq!(grouped.schema.field(width).data_type(), &DataType::Binary);
        assert!(!grouped.schema.field(width).is_nullable());
        let rebuilt = Aggregate::try_new_with_schema(
            input.clone(),
            grouped.group_expr.clone(),
            grouped.aggr_expr.clone(),
            grouped.schema.clone(),
        )?;
        assert_eq!(rebuilt.schema, grouped.schema);
        let plain = Aggregate::try_new(input, keys, vec![count(lit(1))])?;
        assert_eq!(plain.schema.fields().len(), width + 1);
        assert_eq!(plain.schema.field(width).data_type(), &DataType::Int64);
    }
    Ok(())
}

use crate::builder::LogicalTableSource;
use crate::logical_plan::table_scan;
use crate::select_expr::SelectExpr;
use crate::test::function_stub::{count, count_udaf};
use crate::{
    GroupingSet, binary_expr, col, exists, in_subquery, lit, placeholder, scalar_subquery,
};
use datafusion_common::metadata::ScalarAndMetadata;
use datafusion_common::tree_node::{
    TransformedResult, TreeNodeRewriter, TreeNodeVisitor,
};
use datafusion_common::{Constraint, ScalarValue, not_impl_err};
use insta::{assert_debug_snapshot, assert_snapshot};
use std::hash::DefaultHasher;

fn employee_schema() -> Schema {
    Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("first_name", DataType::Utf8, false),
        Field::new("last_name", DataType::Utf8, false),
        Field::new("state", DataType::Utf8, false),
        Field::new("salary", DataType::Int32, false),
    ])
}

fn display_plan() -> Result<LogicalPlan> {
    let plan1 =
        table_scan(Some("employee_csv"), &employee_schema(), Some(vec![3]))?.build()?;

    table_scan(Some("employee_csv"), &employee_schema(), Some(vec![0, 3]))?
        .filter(in_subquery(col("state"), Arc::new(plan1)))?
        .project(vec![col("id")])?
        .build()
}

#[test]
fn test_display_indent() -> Result<()> {
    let plan = display_plan()?;

    assert_snapshot!(plan.display_indent(), @r"
    Projection: employee_csv.id
      Filter: employee_csv.state IN (<subquery>)
        Subquery:
          TableScan: employee_csv projection=[state]
        TableScan: employee_csv projection=[id, state]
    ");
    Ok(())
}

#[test]
fn test_display_indent_schema() -> Result<()> {
    let plan = display_plan()?;

    assert_snapshot!(plan.display_indent_schema(), @r"
    Projection: employee_csv.id [id:Int32]
      Filter: employee_csv.state IN (<subquery>) [id:Int32, state:Utf8]
        Subquery: [state:Utf8]
          TableScan: employee_csv projection=[state] [state:Utf8]
        TableScan: employee_csv projection=[id, state] [id:Int32, state:Utf8]
    ");
    Ok(())
}

#[test]
fn test_display_subquery_alias() -> Result<()> {
    let plan1 =
        table_scan(Some("employee_csv"), &employee_schema(), Some(vec![3]))?.build()?;
    let plan1 = Arc::new(plan1);

    let plan = table_scan(Some("employee_csv"), &employee_schema(), Some(vec![0, 3]))?
        .project(vec![col("id"), exists(plan1).alias("exists")])?
        .build();

    assert_snapshot!(plan?.display_indent(), @r"
    Projection: employee_csv.id, EXISTS (<subquery>) AS exists
      Subquery:
        TableScan: employee_csv projection=[state]
      TableScan: employee_csv projection=[id, state]
    ");
    Ok(())
}

#[test]
fn test_display_graphviz() -> Result<()> {
    let plan = display_plan()?;

    // just test for a few key lines in the output rather than the
    // whole thing to make test maintenance easier.
    assert_snapshot!(plan.display_graphviz(), @r#"
    // Begin DataFusion GraphViz Plan,
    // display it online here: https://dreampuf.github.io/GraphvizOnline

    digraph {
      subgraph cluster_1
      {
        graph[label="LogicalPlan"]
        2[shape=box label="Projection: employee_csv.id"]
        3[shape=box label="Filter: employee_csv.state IN (<subquery>)"]
        2 -> 3 [arrowhead=none, arrowtail=normal, dir=back]
        4[shape=box label="Subquery:"]
        3 -> 4 [arrowhead=none, arrowtail=normal, dir=back]
        5[shape=box label="TableScan: employee_csv projection=[state]"]
        4 -> 5 [arrowhead=none, arrowtail=normal, dir=back]
        6[shape=box label="TableScan: employee_csv projection=[id, state]"]
        3 -> 6 [arrowhead=none, arrowtail=normal, dir=back]
      }
      subgraph cluster_7
      {
        graph[label="Detailed LogicalPlan"]
        8[shape=box label="Projection: employee_csv.id\nSchema: [id:Int32]"]
        9[shape=box label="Filter: employee_csv.state IN (<subquery>)\nSchema: [id:Int32, state:Utf8]"]
        8 -> 9 [arrowhead=none, arrowtail=normal, dir=back]
        10[shape=box label="Subquery:\nSchema: [state:Utf8]"]
        9 -> 10 [arrowhead=none, arrowtail=normal, dir=back]
        11[shape=box label="TableScan: employee_csv projection=[state]\nSchema: [state:Utf8]"]
        10 -> 11 [arrowhead=none, arrowtail=normal, dir=back]
        12[shape=box label="TableScan: employee_csv projection=[id, state]\nSchema: [id:Int32, state:Utf8]"]
        9 -> 12 [arrowhead=none, arrowtail=normal, dir=back]
      }
    }
    // End DataFusion GraphViz Plan
    "#);
    Ok(())
}

#[test]
fn test_display_pg_json() -> Result<()> {
    let plan = display_plan()?;

    assert_snapshot!(plan.display_pg_json(), @r#"
    [
      {
        "Plan": {
          "Expressions": [
            "employee_csv.id"
          ],
          "Node Type": "Projection",
          "Output": [
            "id"
          ],
          "Plans": [
            {
              "Condition": "employee_csv.state IN (<subquery>)",
              "Node Type": "Filter",
              "Output": [
                "id",
                "state"
              ],
              "Plans": [
                {
                  "Node Type": "Subquery",
                  "Output": [
                    "state"
                  ],
                  "Plans": [
                    {
                      "Node Type": "TableScan",
                      "Output": [
                        "state"
                      ],
                      "Plans": [],
                      "Relation Name": "employee_csv"
                    }
                  ]
                },
                {
                  "Node Type": "TableScan",
                  "Output": [
                    "id",
                    "state"
                  ],
                  "Plans": [],
                  "Relation Name": "employee_csv"
                }
              ]
            }
          ]
        }
      }
    ]
    "#);
    Ok(())
}

/// Tests for the Visitor trait and walking logical plan nodes
#[derive(Debug, Default)]
struct OkVisitor {
    strings: Vec<String>,
}

impl<'n> TreeNodeVisitor<'n> for OkVisitor {
    type Node = LogicalPlan;

    fn f_down(&mut self, plan: &'n LogicalPlan) -> Result<TreeNodeRecursion> {
        let s = match plan {
            LogicalPlan::Projection { .. } => "pre_visit Projection",
            LogicalPlan::Filter { .. } => "pre_visit Filter",
            LogicalPlan::TableScan { .. } => "pre_visit TableScan",
            _ => {
                return not_impl_err!("unknown plan type");
            }
        };

        self.strings.push(s.into());
        Ok(TreeNodeRecursion::Continue)
    }

    fn f_up(&mut self, plan: &'n LogicalPlan) -> Result<TreeNodeRecursion> {
        let s = match plan {
            LogicalPlan::Projection { .. } => "post_visit Projection",
            LogicalPlan::Filter { .. } => "post_visit Filter",
            LogicalPlan::TableScan { .. } => "post_visit TableScan",
            _ => {
                return not_impl_err!("unknown plan type");
            }
        };

        self.strings.push(s.into());
        Ok(TreeNodeRecursion::Continue)
    }
}

#[test]
fn visit_order() {
    let mut visitor = OkVisitor::default();
    let plan = test_plan();
    let res = plan.visit_with_subqueries(&mut visitor);
    assert!(res.is_ok());

    assert_debug_snapshot!(visitor.strings, @r#"
    [
        "pre_visit Projection",
        "pre_visit Filter",
        "pre_visit TableScan",
        "post_visit TableScan",
        "post_visit Filter",
        "post_visit Projection",
    ]
    "#);
}

#[derive(Debug, Default)]
/// Counter than counts to zero and returns true when it gets there
struct OptionalCounter {
    val: Option<usize>,
}

impl OptionalCounter {
    fn new(val: usize) -> Self {
        Self { val: Some(val) }
    }
    // Decrements the counter by 1, if any, returning true if it hits zero
    fn dec(&mut self) -> bool {
        if Some(0) == self.val {
            true
        } else {
            self.val = self.val.take().map(|i| i - 1);
            false
        }
    }
}

#[derive(Debug, Default)]
/// Visitor that returns false after some number of visits
struct StoppingVisitor {
    inner: OkVisitor,
    /// When Some(0) returns false from pre_visit
    return_false_from_pre_in: OptionalCounter,
    /// When Some(0) returns false from post_visit
    return_false_from_post_in: OptionalCounter,
}

impl<'n> TreeNodeVisitor<'n> for StoppingVisitor {
    type Node = LogicalPlan;

    fn f_down(&mut self, plan: &'n LogicalPlan) -> Result<TreeNodeRecursion> {
        if self.return_false_from_pre_in.dec() {
            return Ok(TreeNodeRecursion::Stop);
        }
        self.inner.f_down(plan)?;

        Ok(TreeNodeRecursion::Continue)
    }

    fn f_up(&mut self, plan: &'n LogicalPlan) -> Result<TreeNodeRecursion> {
        if self.return_false_from_post_in.dec() {
            return Ok(TreeNodeRecursion::Stop);
        }

        self.inner.f_up(plan)
    }
}

/// test early stopping in pre-visit
#[test]
fn early_stopping_pre_visit() {
    let mut visitor = StoppingVisitor {
        return_false_from_pre_in: OptionalCounter::new(2),
        ..Default::default()
    };
    let plan = test_plan();
    let res = plan.visit_with_subqueries(&mut visitor);
    assert!(res.is_ok());

    assert_debug_snapshot!(
        visitor.inner.strings,
        @r#"
    [
        "pre_visit Projection",
        "pre_visit Filter",
    ]
    "#
    );
}

#[test]
fn early_stopping_post_visit() {
    let mut visitor = StoppingVisitor {
        return_false_from_post_in: OptionalCounter::new(1),
        ..Default::default()
    };
    let plan = test_plan();
    let res = plan.visit_with_subqueries(&mut visitor);
    assert!(res.is_ok());

    assert_debug_snapshot!(
        visitor.inner.strings,
        @r#"
    [
        "pre_visit Projection",
        "pre_visit Filter",
        "pre_visit TableScan",
        "post_visit TableScan",
    ]
    "#
    );
}

#[derive(Debug, Default)]
/// Visitor that returns an error after some number of visits
struct ErrorVisitor {
    inner: OkVisitor,
    /// When Some(0) returns false from pre_visit
    return_error_from_pre_in: OptionalCounter,
    /// When Some(0) returns false from post_visit
    return_error_from_post_in: OptionalCounter,
}

impl<'n> TreeNodeVisitor<'n> for ErrorVisitor {
    type Node = LogicalPlan;

    fn f_down(&mut self, plan: &'n LogicalPlan) -> Result<TreeNodeRecursion> {
        if self.return_error_from_pre_in.dec() {
            return not_impl_err!("Error in pre_visit");
        }

        self.inner.f_down(plan)
    }

    fn f_up(&mut self, plan: &'n LogicalPlan) -> Result<TreeNodeRecursion> {
        if self.return_error_from_post_in.dec() {
            return not_impl_err!("Error in post_visit");
        }

        self.inner.f_up(plan)
    }
}

#[test]
fn error_pre_visit() {
    let mut visitor = ErrorVisitor {
        return_error_from_pre_in: OptionalCounter::new(2),
        ..Default::default()
    };
    let plan = test_plan();
    let res = plan.visit_with_subqueries(&mut visitor).unwrap_err();
    assert_snapshot!(
        res.strip_backtrace(),
        @"This feature is not implemented: Error in pre_visit"
    );
    assert_debug_snapshot!(
        visitor.inner.strings,
        @r#"
    [
        "pre_visit Projection",
        "pre_visit Filter",
    ]
    "#
    );
}

#[test]
fn error_post_visit() {
    let mut visitor = ErrorVisitor {
        return_error_from_post_in: OptionalCounter::new(1),
        ..Default::default()
    };
    let plan = test_plan();
    let res = plan.visit_with_subqueries(&mut visitor).unwrap_err();
    assert_snapshot!(
        res.strip_backtrace(),
        @"This feature is not implemented: Error in post_visit"
    );
    assert_debug_snapshot!(
        visitor.inner.strings,
        @r#"
    [
        "pre_visit Projection",
        "pre_visit Filter",
        "pre_visit TableScan",
        "post_visit TableScan",
    ]
    "#
    );
}

#[test]
fn test_partial_eq_hash_and_partial_ord() {
    let empty_values = Arc::new(LogicalPlan::EmptyRelation(EmptyRelation {
        produce_one_row: true,
        schema: Arc::new(DFSchema::empty()),
    }));

    let count_window_function = |schema| {
        Window::try_new_with_schema(
            vec![Expr::WindowFunction(Box::new(WindowFunction::new(
                WindowFunctionDefinition::AggregateUDF(count_udaf()),
                vec![],
            )))],
            Arc::clone(&empty_values),
            Arc::new(schema),
        )
        .unwrap()
    };

    let schema_without_metadata = || {
        DFSchema::from_unqualified_fields(
            vec![Field::new("count", DataType::Int64, false)].into(),
            HashMap::new(),
        )
        .unwrap()
    };

    let schema_with_metadata = || {
        DFSchema::from_unqualified_fields(
            vec![Field::new("count", DataType::Int64, false)].into(),
            [("key".to_string(), "value".to_string())].into(),
        )
        .unwrap()
    };

    // A Window
    let f = count_window_function(schema_without_metadata());

    // Same like `f`, different instance
    let f2 = count_window_function(schema_without_metadata());
    assert_eq!(f, f2);
    assert_eq!(hash(&f), hash(&f2));
    assert_eq!(f.partial_cmp(&f2), Some(Ordering::Equal));

    // Same like `f`, except for schema metadata
    let o = count_window_function(schema_with_metadata());
    assert_ne!(f, o);
    assert_ne!(hash(&f), hash(&o)); // hash can collide for different values but does not collide in this test
    assert_eq!(f.partial_cmp(&o), None);
}

fn hash<T: Hash>(value: &T) -> u64 {
    let hasher = &mut DefaultHasher::new();
    value.hash(hasher);
    hasher.finish()
}

#[test]
fn projection_expr_schema_mismatch() -> Result<()> {
    let empty_schema = Arc::new(DFSchema::empty());
    let p = Projection::try_new_with_schema(
        vec![col("a")],
        Arc::new(LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema: Arc::clone(&empty_schema),
        })),
        empty_schema,
    );
    assert_snapshot!(p.unwrap_err().strip_backtrace(), @"Error during planning: Projection has mismatch between number of expressions (1) and number of fields in schema (0)");
    Ok(())
}

fn test_plan() -> LogicalPlan {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("state", DataType::Utf8, false),
    ]);

    table_scan(TableReference::none(), &schema, Some(vec![0, 1]))
        .unwrap()
        .filter(col("state").eq(lit("CO")))
        .unwrap()
        .project(vec![col("id")])
        .unwrap()
        .build()
        .unwrap()
}

#[test]
fn test_replace_invalid_placeholder() {
    // test empty placeholder
    let schema = Schema::new(vec![Field::new("id", DataType::Int32, false)]);

    let plan = table_scan(TableReference::none(), &schema, None)
        .unwrap()
        .filter(col("id").eq(placeholder("")))
        .unwrap()
        .build()
        .unwrap();

    let param_values = vec![ScalarValue::Int32(Some(42))];
    plan.replace_params_with_values(&param_values.clone().into())
        .expect_err("unexpectedly succeeded to replace an invalid placeholder");

    // test $0 placeholder
    let schema = Schema::new(vec![Field::new("id", DataType::Int32, false)]);

    let plan = table_scan(TableReference::none(), &schema, None)
        .unwrap()
        .filter(col("id").eq(placeholder("$0")))
        .unwrap()
        .build()
        .unwrap();

    plan.replace_params_with_values(&param_values.clone().into())
        .expect_err("unexpectedly succeeded to replace an invalid placeholder");

    // test $00 placeholder
    let schema = Schema::new(vec![Field::new("id", DataType::Int32, false)]);

    let plan = table_scan(TableReference::none(), &schema, None)
        .unwrap()
        .filter(col("id").eq(placeholder("$00")))
        .unwrap()
        .build()
        .unwrap();

    plan.replace_params_with_values(&param_values.into())
        .expect_err("unexpectedly succeeded to replace an invalid placeholder");
}

#[test]
fn test_replace_placeholder_mismatched_metadata() {
    let schema = Schema::new(vec![Field::new("id", DataType::Int32, false)]);

    // Create a prepared statement with explicit fields that do not have metadata
    let plan = table_scan(TableReference::none(), &schema, None)
        .unwrap()
        .filter(col("id").eq(placeholder("$1")))
        .unwrap()
        .build()
        .unwrap();
    let prepared_builder = LogicalPlanBuilder::new(plan)
        .prepare(
            "".to_string(),
            vec![Field::new("", DataType::Int32, true).into()],
        )
        .unwrap();

    // Attempt to bind a parameter with metadata
    let mut scalar_meta = HashMap::new();
    scalar_meta.insert("some_key".to_string(), "some_value".to_string());
    let param_values = ParamValues::List(vec![ScalarAndMetadata::new(
        ScalarValue::Int32(Some(42)),
        Some(scalar_meta.into()),
    )]);
    prepared_builder
        .plan()
        .clone()
        .with_param_values(param_values)
        .expect_err("prepared field metadata mismatch unexpectedly succeeded");
}

#[test]
fn test_replace_placeholder_empty_relation_valid_schema() {
    // SELECT $1, $2;
    let plan = LogicalPlanBuilder::empty(false)
        .project(vec![
            SelectExpr::from(placeholder("$1")),
            SelectExpr::from(placeholder("$2")),
        ])
        .unwrap()
        .build()
        .unwrap();

    // original
    assert_snapshot!(plan.display_indent_schema(), @r"
    Projection: $1, $2 [$1:Null;N, $2:Null;N]
      EmptyRelation: rows=0 []
    ");

    let plan = plan
        .with_param_values(vec![ScalarValue::from(1i32), ScalarValue::from("s")])
        .unwrap();

    // replaced
    assert_snapshot!(plan.display_indent_schema(), @r#"
    Projection: Int32(1) AS $1, Utf8("s") AS $2 [$1:Int32, $2:Utf8]
      EmptyRelation: rows=0 []
    "#);
}

#[test]
fn test_nullable_schema_after_grouping_set() {
    let schema = Schema::new(vec![
        Field::new("foo", DataType::Int32, false),
        Field::new("bar", DataType::Int32, false),
    ]);

    let plan = table_scan(TableReference::none(), &schema, None)
        .unwrap()
        .aggregate(
            vec![Expr::GroupingSet(GroupingSet::GroupingSets(vec![
                vec![col("foo")],
                vec![col("bar")],
            ]))],
            vec![count(lit(true))],
        )
        .unwrap()
        .build()
        .unwrap();

    let output_schema = plan.schema();

    assert!(
        output_schema
            .field_with_name(None, "foo")
            .unwrap()
            .is_nullable(),
    );
    assert!(
        output_schema
            .field_with_name(None, "bar")
            .unwrap()
            .is_nullable()
    );
}

#[test]
fn test_filter_is_scalar() {
    // test empty placeholder
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

    let source = Arc::new(LogicalTableSource::new(schema));
    let schema = Arc::new(
        DFSchema::try_from_qualified_schema(
            TableReference::bare("tab"),
            &source.schema(),
        )
        .unwrap(),
    );
    let scan = Arc::new(LogicalPlan::TableScan(TableScan {
        table_name: TableReference::bare("tab"),
        source: Arc::clone(&source) as Arc<dyn TableSource>,
        projection: None,
        projected_schema: Arc::clone(&schema),
        filters: vec![],
        fetch: None,
    }));
    let col = schema.field_names()[0].clone();

    let filter = Filter::try_new(
        Expr::Column(col.into()).eq(Expr::Literal(ScalarValue::Int32(Some(1)), None)),
        scan,
    )
    .unwrap();
    assert!(!filter.is_scalar());
    let unique_schema = Arc::new(
        schema
            .as_ref()
            .clone()
            .with_functional_dependencies(FunctionalDependencies::new_from_constraints(
                Some(&Constraints::new_unverified(vec![Constraint::Unique(
                    vec![0],
                )])),
                1,
            ))
            .unwrap(),
    );
    let scan = Arc::new(LogicalPlan::TableScan(TableScan {
        table_name: TableReference::bare("tab"),
        source,
        projection: None,
        projected_schema: Arc::clone(&unique_schema),
        filters: vec![],
        fetch: None,
    }));
    let col = schema.field_names()[0].clone();

    let filter = Filter::try_new(Expr::Column(col.into()).eq(lit(1i32)), scan).unwrap();
    assert!(filter.is_scalar());
}

#[test]
fn test_transform_explain() {
    let schema = Schema::new(vec![
        Field::new("foo", DataType::Int32, false),
        Field::new("bar", DataType::Int32, false),
    ]);

    let plan = table_scan(TableReference::none(), &schema, None)
        .unwrap()
        .explain(false, false)
        .unwrap()
        .build()
        .unwrap();

    let external_filter = col("foo").eq(lit(true));

    // after transformation, because plan is not the same anymore,
    // the parent plan is built again with call to LogicalPlan::with_new_inputs -> with_new_exprs
    let plan = plan
        .transform(|plan| match plan {
            LogicalPlan::TableScan(table) => {
                let filter = Filter::try_new(
                    external_filter.clone(),
                    Arc::new(LogicalPlan::TableScan(table)),
                )
                .unwrap();
                Ok(Transformed::yes(LogicalPlan::Filter(filter)))
            }
            x => Ok(Transformed::no(x)),
        })
        .data()
        .unwrap();

    let actual = format!("{}", plan.display_indent());
    assert_snapshot!(actual, @r"
    Explain
      Filter: foo = Boolean(true)
        TableScan: ?table?
    ")
}

#[test]
fn test_plan_partial_ord() {
    let empty_relation = LogicalPlan::EmptyRelation(EmptyRelation {
        produce_one_row: false,
        schema: Arc::new(DFSchema::empty()),
    });

    let describe_table = LogicalPlan::DescribeTable(DescribeTable {
        schema: Arc::new(Schema::new(vec![Field::new("foo", DataType::Int32, false)])),
        output_schema: DFSchemaRef::new(DFSchema::empty()),
    });

    let describe_table_clone = LogicalPlan::DescribeTable(DescribeTable {
        schema: Arc::new(Schema::new(vec![Field::new("foo", DataType::Int32, false)])),
        output_schema: DFSchemaRef::new(DFSchema::empty()),
    });

    assert_eq!(
        empty_relation.partial_cmp(&describe_table),
        Some(Ordering::Less)
    );
    assert_eq!(
        describe_table.partial_cmp(&empty_relation),
        Some(Ordering::Greater)
    );
    assert_eq!(describe_table.partial_cmp(&describe_table_clone), None);
}

#[test]
fn test_limit_with_new_children() {
    let input = Arc::new(LogicalPlan::Values(Values {
        schema: Arc::new(DFSchema::empty()),
        values: vec![vec![]],
    }));
    let cases = [
        LogicalPlan::Limit(Limit {
            skip: None,
            fetch: None,
            input: Arc::clone(&input),
        }),
        LogicalPlan::Limit(Limit {
            skip: None,
            fetch: Some(Box::new(Expr::Literal(
                ScalarValue::new_ten(&DataType::UInt32).unwrap(),
                None,
            ))),
            input: Arc::clone(&input),
        }),
        LogicalPlan::Limit(Limit {
            skip: Some(Box::new(Expr::Literal(
                ScalarValue::new_ten(&DataType::UInt32).unwrap(),
                None,
            ))),
            fetch: None,
            input: Arc::clone(&input),
        }),
        LogicalPlan::Limit(Limit {
            skip: Some(Box::new(Expr::Literal(
                ScalarValue::new_one(&DataType::UInt32).unwrap(),
                None,
            ))),
            fetch: Some(Box::new(Expr::Literal(
                ScalarValue::new_ten(&DataType::UInt32).unwrap(),
                None,
            ))),
            input,
        }),
    ];

    for limit in cases {
        let new_limit = limit
            .with_new_exprs(
                limit.expressions(),
                limit.inputs().into_iter().cloned().collect(),
            )
            .unwrap();
        assert_eq!(limit, new_limit);
    }
}

#[test]
fn test_with_subqueries_jump() {
    // The test plan contains a `Project` node above a `Filter` node, and the
    // `Project` node contains a subquery plan with a `Filter` root node, so returning
    // `TreeNodeRecursion::Jump` on `Project` should cause not visiting any of the
    // `Filter`s.
    let subquery_schema = Schema::new(vec![Field::new("sub_id", DataType::Int32, false)]);

    let subquery_plan =
        table_scan(TableReference::none(), &subquery_schema, Some(vec![0]))
            .unwrap()
            .filter(col("sub_id").eq(lit(0)))
            .unwrap()
            .build()
            .unwrap();

    let schema = Schema::new(vec![Field::new("id", DataType::Int32, false)]);

    let plan = table_scan(TableReference::none(), &schema, Some(vec![0]))
        .unwrap()
        .filter(col("id").eq(lit(0)))
        .unwrap()
        .project(vec![col("id"), scalar_subquery(Arc::new(subquery_plan))])
        .unwrap()
        .build()
        .unwrap();

    let mut filter_found = false;
    plan.apply_with_subqueries(|plan| {
        match plan {
            LogicalPlan::Projection(..) => return Ok(TreeNodeRecursion::Jump),
            LogicalPlan::Filter(..) => filter_found = true,
            _ => {}
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    assert!(!filter_found);

    struct ProjectJumpVisitor {
        filter_found: bool,
    }

    impl ProjectJumpVisitor {
        fn new() -> Self {
            Self {
                filter_found: false,
            }
        }
    }

    impl<'n> TreeNodeVisitor<'n> for ProjectJumpVisitor {
        type Node = LogicalPlan;

        fn f_down(&mut self, node: &'n Self::Node) -> Result<TreeNodeRecursion> {
            match node {
                LogicalPlan::Projection(..) => return Ok(TreeNodeRecursion::Jump),
                LogicalPlan::Filter(..) => self.filter_found = true,
                _ => {}
            }
            Ok(TreeNodeRecursion::Continue)
        }
    }

    let mut visitor = ProjectJumpVisitor::new();
    plan.visit_with_subqueries(&mut visitor).unwrap();
    assert!(!visitor.filter_found);

    let mut filter_found = false;
    plan.clone()
        .transform_down_with_subqueries(|plan| {
            match plan {
                LogicalPlan::Projection(..) => {
                    return Ok(Transformed::new(plan, false, TreeNodeRecursion::Jump));
                }
                LogicalPlan::Filter(..) => filter_found = true,
                _ => {}
            }
            Ok(Transformed::no(plan))
        })
        .unwrap();
    assert!(!filter_found);

    let mut filter_found = false;
    plan.clone()
        .transform_down_up_with_subqueries(
            |plan| {
                match plan {
                    LogicalPlan::Projection(..) => {
                        return Ok(Transformed::new(
                            plan,
                            false,
                            TreeNodeRecursion::Jump,
                        ));
                    }
                    LogicalPlan::Filter(..) => filter_found = true,
                    _ => {}
                }
                Ok(Transformed::no(plan))
            },
            |plan| Ok(Transformed::no(plan)),
        )
        .unwrap();
    assert!(!filter_found);

    struct ProjectJumpRewriter {
        filter_found: bool,
    }

    impl ProjectJumpRewriter {
        fn new() -> Self {
            Self {
                filter_found: false,
            }
        }
    }

    impl TreeNodeRewriter for ProjectJumpRewriter {
        type Node = LogicalPlan;

        fn f_down(&mut self, node: Self::Node) -> Result<Transformed<Self::Node>> {
            match node {
                LogicalPlan::Projection(..) => {
                    return Ok(Transformed::new(node, false, TreeNodeRecursion::Jump));
                }
                LogicalPlan::Filter(..) => self.filter_found = true,
                _ => {}
            }
            Ok(Transformed::no(node))
        }
    }

    let mut rewriter = ProjectJumpRewriter::new();
    plan.rewrite_with_subqueries(&mut rewriter).unwrap();
    assert!(!rewriter.filter_found);
}

#[test]
fn test_with_unresolved_placeholders() {
    let field_name = "id";
    let placeholder_value = "$1";
    let schema = Schema::new(vec![Field::new(field_name, DataType::Int32, false)]);

    let plan = table_scan(TableReference::none(), &schema, None)
        .unwrap()
        .filter(col(field_name).eq(placeholder(placeholder_value)))
        .unwrap()
        .build()
        .unwrap();

    // Check that the placeholder parameters have not received a DataType.
    let params = plan.get_parameter_fields().unwrap();
    assert_eq!(params.len(), 1);

    let parameter_type = params.clone().get(placeholder_value).unwrap().clone();
    assert_eq!(parameter_type, None);
}

#[test]
fn test_join_with_new_exprs() -> Result<()> {
    fn create_test_join(
        on: Vec<(Expr, Expr)>,
        filter: Option<Expr>,
    ) -> Result<LogicalPlan> {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]);

        let left_schema = DFSchema::try_from_qualified_schema("t1", &schema)?;
        let right_schema = DFSchema::try_from_qualified_schema("t2", &schema)?;

        Ok(LogicalPlan::Join(Join {
            left: Arc::new(
                table_scan(Some("t1"), left_schema.as_arrow(), None)?.build()?,
            ),
            right: Arc::new(
                table_scan(Some("t2"), right_schema.as_arrow(), None)?.build()?,
            ),
            on,
            filter,
            join_type: JoinType::Inner,
            join_constraint: JoinConstraint::On,
            schema: Arc::new(left_schema.join(&right_schema)?),
            null_equality: NullEquality::NullEqualsNothing,
            null_aware: false,
        }))
    }

    {
        let join = create_test_join(vec![(col("t1.a"), (col("t2.a")))], None)?;
        let LogicalPlan::Join(join) = join.with_new_exprs(
            join.expressions(),
            join.inputs().into_iter().cloned().collect(),
        )?
        else {
            unreachable!()
        };
        assert_eq!(join.on, vec![(col("t1.a"), (col("t2.a")))]);
        assert_eq!(join.filter, None);
    }

    {
        let join = create_test_join(vec![], Some(col("t1.a").gt(col("t2.a"))))?;
        let LogicalPlan::Join(join) = join.with_new_exprs(
            join.expressions(),
            join.inputs().into_iter().cloned().collect(),
        )?
        else {
            unreachable!()
        };
        assert_eq!(join.on, vec![]);
        assert_eq!(join.filter, Some(col("t1.a").gt(col("t2.a"))));
    }

    {
        let join = create_test_join(
            vec![(col("t1.a"), (col("t2.a")))],
            Some(col("t1.b").gt(col("t2.b"))),
        )?;
        let LogicalPlan::Join(join) = join.with_new_exprs(
            join.expressions(),
            join.inputs().into_iter().cloned().collect(),
        )?
        else {
            unreachable!()
        };
        assert_eq!(join.on, vec![(col("t1.a"), (col("t2.a")))]);
        assert_eq!(join.filter, Some(col("t1.b").gt(col("t2.b"))));
    }

    {
        let join = create_test_join(
            vec![(col("t1.a"), (col("t2.a"))), (col("t1.b"), (col("t2.b")))],
            None,
        )?;
        let LogicalPlan::Join(join) = join.with_new_exprs(
            vec![
                binary_expr(col("t1.a"), Operator::Plus, lit(1)),
                binary_expr(col("t2.a"), Operator::Plus, lit(2)),
                col("t1.b"),
                col("t2.b"),
                lit(true),
            ],
            join.inputs().into_iter().cloned().collect(),
        )?
        else {
            unreachable!()
        };
        assert_eq!(
            join.on,
            vec![
                (
                    binary_expr(col("t1.a"), Operator::Plus, lit(1)),
                    binary_expr(col("t2.a"), Operator::Plus, lit(2))
                ),
                (col("t1.b"), (col("t2.b")))
            ]
        );
        assert_eq!(join.filter, Some(lit(true)));
    }

    Ok(())
}

#[test]
fn test_join_try_new() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
    ]);

    let left_scan = table_scan(Some("t1"), &schema, None)?.build()?;

    let right_scan = table_scan(Some("t2"), &schema, None)?.build()?;

    let join_types = vec![
        JoinType::Inner,
        JoinType::Left,
        JoinType::Right,
        JoinType::Full,
        JoinType::LeftSemi,
        JoinType::LeftAnti,
        JoinType::RightSemi,
        JoinType::RightAnti,
        JoinType::LeftMark,
    ];

    for join_type in join_types {
        let join = Join::try_new(
            Arc::new(left_scan.clone()),
            Arc::new(right_scan.clone()),
            vec![(col("t1.a"), col("t2.a"))],
            Some(col("t1.b").gt(col("t2.b"))),
            join_type,
            JoinConstraint::On,
            NullEquality::NullEqualsNothing,
            false,
        )?;

        match join_type {
            JoinType::LeftSemi | JoinType::LeftAnti => {
                assert_eq!(join.schema.fields().len(), 2);

                let fields = join.schema.fields();
                assert_eq!(
                    fields[0].name(),
                    "a",
                    "First field should be 'a' from left table"
                );
                assert_eq!(
                    fields[1].name(),
                    "b",
                    "Second field should be 'b' from left table"
                );
            }
            JoinType::RightSemi | JoinType::RightAnti => {
                assert_eq!(join.schema.fields().len(), 2);

                let fields = join.schema.fields();
                assert_eq!(
                    fields[0].name(),
                    "a",
                    "First field should be 'a' from right table"
                );
                assert_eq!(
                    fields[1].name(),
                    "b",
                    "Second field should be 'b' from right table"
                );
            }
            JoinType::LeftMark => {
                assert_eq!(join.schema.fields().len(), 3);

                let fields = join.schema.fields();
                assert_eq!(
                    fields[0].name(),
                    "a",
                    "First field should be 'a' from left table"
                );
                assert_eq!(
                    fields[1].name(),
                    "b",
                    "Second field should be 'b' from left table"
                );
                assert_eq!(
                    fields[2].name(),
                    "mark",
                    "Third field should be the mark column"
                );

                assert!(!fields[0].is_nullable());
                assert!(!fields[1].is_nullable());
                assert!(!fields[2].is_nullable());
            }
            _ => {
                assert_eq!(join.schema.fields().len(), 4);

                let fields = join.schema.fields();
                assert_eq!(
                    fields[0].name(),
                    "a",
                    "First field should be 'a' from left table"
                );
                assert_eq!(
                    fields[1].name(),
                    "b",
                    "Second field should be 'b' from left table"
                );
                assert_eq!(
                    fields[2].name(),
                    "a",
                    "Third field should be 'a' from right table"
                );
                assert_eq!(
                    fields[3].name(),
                    "b",
                    "Fourth field should be 'b' from right table"
                );

                if join_type == JoinType::Left {
                    // Left side fields (first two) shouldn't be nullable
                    assert!(!fields[0].is_nullable());
                    assert!(!fields[1].is_nullable());
                    // Right side fields (third and fourth) should be nullable
                    assert!(fields[2].is_nullable());
                    assert!(fields[3].is_nullable());
                } else if join_type == JoinType::Right {
                    // Left side fields (first two) should be nullable
                    assert!(fields[0].is_nullable());
                    assert!(fields[1].is_nullable());
                    // Right side fields (third and fourth) shouldn't be nullable
                    assert!(!fields[2].is_nullable());
                    assert!(!fields[3].is_nullable());
                } else if join_type == JoinType::Full {
                    assert!(fields[0].is_nullable());
                    assert!(fields[1].is_nullable());
                    assert!(fields[2].is_nullable());
                    assert!(fields[3].is_nullable());
                }
            }
        }

        assert_eq!(join.on, vec![(col("t1.a"), col("t2.a"))]);
        assert_eq!(join.filter, Some(col("t1.b").gt(col("t2.b"))));
        assert_eq!(join.join_type, join_type);
        assert_eq!(join.join_constraint, JoinConstraint::On);
        assert_eq!(join.null_equality, NullEquality::NullEqualsNothing);
    }

    Ok(())
}

#[test]
fn test_join_try_new_with_using_constraint_and_overlapping_columns() -> Result<()> {
    let left_schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false), // Common column in both tables
        Field::new("name", DataType::Utf8, false), // Unique to left
        Field::new("value", DataType::Int32, false), // Common column, different meaning
    ]);

    let right_schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false), // Common column in both tables
        Field::new("category", DataType::Utf8, false), // Unique to right
        Field::new("value", DataType::Float64, true), // Common column, different meaning
    ]);

    let left_plan = table_scan(Some("t1"), &left_schema, None)?.build()?;

    let right_plan = table_scan(Some("t2"), &right_schema, None)?.build()?;

    // Test 1: USING constraint with a common column
    {
        // In the logical plan, both copies of the `id` column are preserved
        // The USING constraint is handled later during physical execution, where the common column appears once
        let join = Join::try_new(
            Arc::new(left_plan.clone()),
            Arc::new(right_plan.clone()),
            vec![(col("t1.id"), col("t2.id"))],
            None,
            JoinType::Inner,
            JoinConstraint::Using,
            NullEquality::NullEqualsNothing,
            false,
        )?;

        let fields = join.schema.fields();

        assert_eq!(fields.len(), 6);

        assert_eq!(
            fields[0].name(),
            "id",
            "First field should be 'id' from left table"
        );
        assert_eq!(
            fields[1].name(),
            "name",
            "Second field should be 'name' from left table"
        );
        assert_eq!(
            fields[2].name(),
            "value",
            "Third field should be 'value' from left table"
        );
        assert_eq!(
            fields[3].name(),
            "id",
            "Fourth field should be 'id' from right table"
        );
        assert_eq!(
            fields[4].name(),
            "category",
            "Fifth field should be 'category' from right table"
        );
        assert_eq!(
            fields[5].name(),
            "value",
            "Sixth field should be 'value' from right table"
        );

        assert_eq!(join.join_constraint, JoinConstraint::Using);
    }

    // Test 2: Complex join condition with expressions
    {
        // Complex condition: join on id equality AND where left.value < right.value
        let join = Join::try_new(
            Arc::new(left_plan.clone()),
            Arc::new(right_plan.clone()),
            vec![(col("t1.id"), col("t2.id"))], // Equijoin condition
            Some(col("t1.value").lt(col("t2.value"))), // Non-equi filter condition
            JoinType::Inner,
            JoinConstraint::On,
            NullEquality::NullEqualsNothing,
            false,
        )?;

        let fields = join.schema.fields();
        assert_eq!(fields.len(), 6);

        assert_eq!(
            fields[0].name(),
            "id",
            "First field should be 'id' from left table"
        );
        assert_eq!(
            fields[1].name(),
            "name",
            "Second field should be 'name' from left table"
        );
        assert_eq!(
            fields[2].name(),
            "value",
            "Third field should be 'value' from left table"
        );
        assert_eq!(
            fields[3].name(),
            "id",
            "Fourth field should be 'id' from right table"
        );
        assert_eq!(
            fields[4].name(),
            "category",
            "Fifth field should be 'category' from right table"
        );
        assert_eq!(
            fields[5].name(),
            "value",
            "Sixth field should be 'value' from right table"
        );

        assert_eq!(join.filter, Some(col("t1.value").lt(col("t2.value"))));
    }

    // Test 3: Join with null equality behavior set to true
    {
        let join = Join::try_new(
            Arc::new(left_plan.clone()),
            Arc::new(right_plan.clone()),
            vec![(col("t1.id"), col("t2.id"))],
            None,
            JoinType::Inner,
            JoinConstraint::On,
            NullEquality::NullEqualsNull,
            false,
        )?;

        assert_eq!(join.null_equality, NullEquality::NullEqualsNull);
    }

    Ok(())
}

#[test]
fn test_join_try_new_schema_validation() -> Result<()> {
    let left_schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("value", DataType::Float64, true),
    ]);

    let right_schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, true),
        Field::new("code", DataType::Int16, false),
    ]);

    let left_plan = table_scan(Some("t1"), &left_schema, None)?.build()?;

    let right_plan = table_scan(Some("t2"), &right_schema, None)?.build()?;

    let join_types = vec![
        JoinType::Inner,
        JoinType::Left,
        JoinType::Right,
        JoinType::Full,
    ];

    for join_type in join_types {
        let join = Join::try_new(
            Arc::new(left_plan.clone()),
            Arc::new(right_plan.clone()),
            vec![(col("t1.id"), col("t2.id"))],
            Some(col("t1.value").gt(lit(5.0))),
            join_type,
            JoinConstraint::On,
            NullEquality::NullEqualsNothing,
            false,
        )?;

        let fields = join.schema.fields();
        assert_eq!(fields.len(), 6, "Expected 6 fields for {join_type} join");

        for (i, field) in fields.iter().enumerate() {
            let expected_nullable = match (i, &join_type) {
                // Left table fields (indices 0, 1, 2)
                (0, JoinType::Right | JoinType::Full) => true, // id becomes nullable in RIGHT/FULL
                (1, JoinType::Right | JoinType::Full) => true, // name becomes nullable in RIGHT/FULL
                (2, _) => true, // value is already nullable

                // Right table fields (indices 3, 4, 5)
                (3, JoinType::Left | JoinType::Full) => true, // id becomes nullable in LEFT/FULL
                (4, _) => true, // category is already nullable
                (5, JoinType::Left | JoinType::Full) => true, // code becomes nullable in LEFT/FULL

                _ => false,
            };

            assert_eq!(
                field.is_nullable(),
                expected_nullable,
                "Field {} ({}) nullability incorrect for {:?} join",
                i,
                field.name(),
                join_type
            );
        }
    }

    let using_join = Join::try_new(
        Arc::new(left_plan.clone()),
        Arc::new(right_plan.clone()),
        vec![(col("t1.id"), col("t2.id"))],
        None,
        JoinType::Inner,
        JoinConstraint::Using,
        NullEquality::NullEqualsNothing,
        false,
    )?;

    assert_eq!(
        using_join.schema.fields().len(),
        6,
        "USING join should have all fields"
    );
    assert_eq!(using_join.join_constraint, JoinConstraint::Using);

    Ok(())
}

#[test]
fn grouping_id_width_keeps_primitive_boundaries_and_widens_one_key() {
    for (count, expected) in [
        (0, DataType::UInt8),
        (8, DataType::UInt8),
        (9, DataType::UInt16),
        (16, DataType::UInt16),
        (17, DataType::UInt32),
        (32, DataType::UInt32),
        (33, DataType::UInt64),
        (64, DataType::UInt64),
        (65, DataType::Binary),
        (128, DataType::Binary),
        (257, DataType::Binary),
    ] {
        assert_eq!(Aggregate::grouping_id_type(count), expected);
    }
}
