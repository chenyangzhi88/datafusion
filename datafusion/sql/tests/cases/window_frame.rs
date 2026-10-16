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

use datafusion_common::{DFSchema, Result, ScalarValue, TableReference};
use datafusion_expr::test::function_stub::sum_udaf;
use datafusion_expr::{
    WindowFrame, WindowFrameBound, WindowFrameExclusion, WindowFrameUnits,
};
use datafusion_sql::planner::{ContextProvider, PlannerContext, SqlToRel};
use datafusion_sql::unparser::expr_to_sql;
use sqlparser::ast;
use sqlparser::dialect::PostgreSqlDialect;
use sqlparser::parser::Parser;

use crate::common::{MockContextProvider, MockSessionState};

#[test]
fn window_frame_exclusion_conversion_preserves_bounds_and_causality() -> Result<()> {
    for (syntax, expected) in [
        (None, None),
        (Some(ast::WindowFrameExclusion::NoOthers), None),
        (
            Some(ast::WindowFrameExclusion::CurrentRow),
            Some(WindowFrameExclusion::CurrentRow),
        ),
        (
            Some(ast::WindowFrameExclusion::Group),
            Some(WindowFrameExclusion::Group),
        ),
        (
            Some(ast::WindowFrameExclusion::Ties),
            Some(WindowFrameExclusion::Ties),
        ),
    ] {
        let frame = WindowFrame::try_from(ast::WindowFrame {
            units: ast::WindowFrameUnits::Rows,
            start_bound: ast::WindowFrameBound::Preceding(Some(Box::new(
                ast::Expr::value(ast::Value::Number("2".to_owned(), false)),
            ))),
            end_bound: Some(ast::WindowFrameBound::Preceding(Some(Box::new(
                ast::Expr::value(ast::Value::Number("1".to_owned(), false)),
            )))),
            exclusion: syntax,
        })?;

        assert_eq!(frame.exclusion, expected, "{syntax:?}");
        assert_eq!(frame.units, WindowFrameUnits::Rows);
        assert_eq!(
            frame.start_bound,
            WindowFrameBound::Preceding(ScalarValue::UInt64(Some(2)))
        );
        assert_eq!(
            frame.end_bound,
            WindowFrameBound::Preceding(ScalarValue::UInt64(Some(1)))
        );
        assert!(frame.is_causal());
        let base = "ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING";
        let expected_sql = match expected {
            Some(exclusion) => format!("{base} EXCLUDE {exclusion}"),
            None => base.to_owned(),
        };
        assert_eq!(frame.to_string(), expected_sql);
    }
    Ok(())
}

#[test]
fn window_frame_exclusions_roundtrip_through_sql_planner_and_unparser() -> Result<()> {
    let context = MockContextProvider {
        state: MockSessionState::default().with_aggregate_function(sum_udaf()),
    };
    let schema = context
        .get_table_source(TableReference::bare("person"))?
        .schema();
    let schema = DFSchema::try_from(schema)?;
    let planner = SqlToRel::new(&context);
    let dialect = PostgreSqlDialect {};
    let prefix = "sum(age) OVER (ORDER BY age ASC NULLS LAST ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING";

    for exclusion in ["CURRENT ROW", "GROUP", "TIES", "NO OTHERS"] {
        let sql = format!("{prefix} EXCLUDE {exclusion})");
        let parsed = Parser::new(&dialect).try_with_sql(&sql)?.parse_expr()?;
        let logical = planner.sql_to_expr(parsed, &schema, &mut PlannerContext::new())?;
        let unparsed = expr_to_sql(&logical)?;

        let expected = if exclusion == "NO OTHERS" {
            format!("{prefix})")
        } else {
            sql.clone()
        };
        assert_eq!(unparsed.to_string(), expected, "{sql}");

        // Compare the complete expression, including arguments, sorting and frame metadata.
        let reparsed = Parser::new(&dialect)
            .try_with_sql(&unparsed.to_string())?
            .parse_expr()?;
        let roundtrip =
            planner.sql_to_expr(reparsed, &schema, &mut PlannerContext::new())?;
        assert_eq!(roundtrip, logical, "{sql}");
    }
    Ok(())
}
