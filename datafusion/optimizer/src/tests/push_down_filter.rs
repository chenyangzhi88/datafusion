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
