use crate::{ExecutionPlan, empty::EmptyExec};
use arrow::datatypes::Schema;
use std::sync::Arc;
#[test]
fn default_physical_plan_has_no_product_explain_metadata() {
    let physical = EmptyExec::new(Arc::new(Schema::empty()));
    assert!(physical.explain_metadata().is_none());
}
