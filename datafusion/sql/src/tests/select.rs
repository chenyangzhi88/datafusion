// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::{parser::DFParser, planner::SqlToRel};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion_common::{Result, TableReference, config::ConfigOptions};
use datafusion_expr::{
    AggregateUDF, Expr, LogicalPlan, ScalarUDF, TableSource, WindowUDF, col, lit,
    planner::ContextProvider,
};
use std::sync::Arc;

#[derive(Default)]
struct WildcardContext {
    options: ConfigOptions,
}

impl ContextProvider for WildcardContext {
    fn get_table_source(&self, _name: TableReference) -> Result<Arc<dyn TableSource>> {
        Ok(datafusion_expr::logical_plan::builder::table_source(
            &Schema::new(vec![
                Field::new("a", DataType::Int64, false),
                Field::new("b", DataType::Int64, false),
                Field::new("Mi\"xed", DataType::Int64, false),
            ]),
        ))
    }

    fn get_function_meta(&self, _name: &str) -> Option<Arc<ScalarUDF>> {
        None
    }

    fn get_aggregate_meta(&self, _name: &str) -> Option<Arc<AggregateUDF>> {
        None
    }

    fn get_window_meta(&self, _name: &str) -> Option<Arc<WindowUDF>> {
        None
    }

    fn get_variable_type(&self, _names: &[String]) -> Option<DataType> {
        None
    }

    fn options(&self) -> &ConfigOptions {
        &self.options
    }

    fn udf_names(&self) -> Vec<String> {
        vec![]
    }

    fn udaf_names(&self) -> Vec<String> {
        vec![]
    }

    fn udwf_names(&self) -> Vec<String> {
        vec![]
    }
}

fn plan(sql: &str) -> Result<LogicalPlan> {
    let context = WildcardContext::default();
    let mut statements = DFParser::parse_sql(sql)?;
    SqlToRel::new(&context).statement_to_plan(statements.pop_front().unwrap())
}

#[test]
fn wildcard_replace_uses_original_planned_expressions_and_column_order() {
    let LogicalPlan::Projection(projection) =
        plan("SELECT t.* REPLACE (a + 1 AS a, 17 AS b) FROM t").unwrap()
    else {
        panic!("actual wildcard projection");
    };
    assert_eq!(projection.expr[0], (col("t.a") + lit(1_i64)).alias("a"));
    assert_eq!(projection.expr[1], lit(17_i64).alias("b"));
    assert_eq!(projection.expr.len(), 3);
    assert!(
        matches!(&projection.expr[2], Expr::Column(column) if column.name == "Mi\"xed")
    );
}

#[test]
fn wildcard_quoted_exclude_keeps_case_and_original_duplicate_error() {
    let LogicalPlan::Projection(projection) =
        plan(r#"SELECT * EXCLUDE ("Mi""xed") FROM t"#).unwrap()
    else {
        panic!("actual wildcard projection");
    };
    assert_eq!(projection.expr, [col("t.a"), col("t.b")]);
    assert_eq!(
        plan("SELECT * EXCLUDE (a, a) FROM t")
            .unwrap_err()
            .to_string(),
        "Error during planning: EXCLUDE or EXCEPT contains duplicate column names"
    );
}

#[test]
fn wildcard_exclude_rejection_stays_after_replacement_planning_and_before_duplicates() {
    let error = plan("SELECT * EXCLUDE (t.a) REPLACE (missing_function(a) AS b) FROM t")
        .unwrap_err()
        .to_string();
    assert!(error.contains("missing_function"), "{error}");
    assert!(!error.contains("Qualified EXCLUDE"), "{error}");
    for sql in [
        "SELECT * EXCLUDE (t.a) FROM t",
        "SELECT * EXCLUDE (a, a, t.b) FROM t",
    ] {
        assert_eq!(
            plan(sql).unwrap_err().to_string(),
            "This feature is not implemented: Qualified EXCLUDE names are not supported"
        );
    }
}
