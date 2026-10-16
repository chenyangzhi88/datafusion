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

#[cfg(feature = "sql")]
use crate::LogicalPlanBuilder;
use crate::expr::{
    ExcludeSelectItem, ExcludedColumn, PlannedReplaceSelectItem, ReplaceSelectElement,
    WildcardOptions,
};
use crate::{col, lit, select_expr::SelectExpr};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[cfg(not(feature = "sql"))]
use crate::expr::Ident;
#[cfg(feature = "sql")]
use sqlparser::ast::Ident;

fn ident(value: &str) -> Ident {
    #[cfg(feature = "sql")]
    {
        Ident::new(value)
    }
    #[cfg(not(feature = "sql"))]
    {
        Ident {
            value: value.into(),
            quote_style: None,
            span: String::new(),
        }
    }
}

#[test]
fn planned_replacement_clone_hash_and_expression_order_use_the_same_payload() {
    let planned = PlannedReplaceSelectItem {
        items: vec![
            ReplaceSelectElement {
                expr: "a + 1".into(),
                column_name: ident("a"),
                as_keyword: true,
            },
            ReplaceSelectElement {
                expr: "17".into(),
                column_name: ident("b"),
                as_keyword: false,
            },
        ],
        planned_expressions: vec![col("a") + lit(1_i64), lit(17_i64)],
    };
    let copied = planned.clone();
    assert_eq!(planned, copied);
    assert_eq!(planned.expressions(), &[col("a") + lit(1_i64), lit(17_i64)]);
    let mut original_hash = DefaultHasher::new();
    let mut copied_hash = DefaultHasher::new();
    planned.hash(&mut original_hash);
    copied.hash(&mut copied_hash);
    assert_eq!(original_hash.finish(), copied_hash.finish());
    let mut changed = copied;
    changed.planned_expressions.swap(0, 1);
    assert_ne!(planned, changed);
}

#[cfg(feature = "sql")]
fn raw_options(sql: &str) -> sqlparser::ast::WildcardAdditionalOptions {
    use sqlparser::ast::{SelectItem, SetExpr, Statement};
    use sqlparser::{dialect::GenericDialect, parser::Parser};
    let mut statements = Parser::parse_sql(&GenericDialect, sql).unwrap();
    let Statement::Query(query) = statements.remove(0) else {
        panic!("actual SELECT");
    };
    let SetExpr::Select(mut select) = *query.body else {
        panic!("actual SELECT body");
    };
    let SelectItem::Wildcard(options) = select.projection.remove(0) else {
        panic!("actual wildcard");
    };
    options
}

#[cfg(feature = "sql")]
#[test]
fn planned_replacement_preserves_original_display_quotes_and_as_keyword() {
    let raw = raw_options(r#"SELECT * REPLACE (a + 1 AS "Mi""xed", 'x''y' b) FROM t"#)
        .opt_replace
        .unwrap();
    let display = raw.to_string();
    let planned = PlannedReplaceSelectItem {
        items: raw.items.into_iter().map(|item| (*item).into()).collect(),
        planned_expressions: vec![col("a") + lit(1_i64), lit("x'y")],
    };
    assert!(planned.items()[0].as_keyword);
    assert!(!planned.items()[1].as_keyword);
    assert_eq!(planned.items()[0].column_name.value, "Mi\"xed");
    assert_eq!(planned.items()[0].column_name.quote_style, Some('"'));
    assert_eq!(planned.to_string(), display);
    let wildcard = SelectExpr::Wildcard(WildcardOptions {
        replace: Some(planned),
        ..Default::default()
    });
    assert_eq!(
        wildcard.to_string(),
        "* REPLACE (a + 1 AS \"Mi\"\"xed\", 'x''y' b)"
    );
}

#[cfg(feature = "sql")]
#[test]
fn planned_exclude_preserves_original_quoted_identifier_and_display() {
    let raw = raw_options(r#"SELECT * EXCLUDE ("Mi""xed", a) FROM t"#)
        .opt_exclude
        .unwrap();
    let display = raw.to_string();
    let planned = ExcludeSelectItem::from(raw);
    let ExcludeSelectItem::Multiple(columns) = &planned else {
        panic!("two original names");
    };
    let ExcludedColumn::Ident(name) = &columns[0] else {
        panic!("actual column identifier");
    };
    assert_eq!(name.value, "Mi\"xed");
    assert_eq!(name.quote_style, Some('"'));
    assert_eq!(planned.to_string(), display);
}

#[cfg(feature = "sql")]
#[test]
fn excluded_shapes_preserve_display_and_defer_original_errors_until_expansion() {
    use arrow::datatypes::{DataType, Field, Schema};
    use sqlparser::ast::{ObjectName, ObjectNamePart, ObjectNamePartFunction};
    let schema = Schema::new(vec![Field::new("a", DataType::Int64, false)]);
    let plan = LogicalPlanBuilder::scan(
        "t",
        crate::logical_plan::builder::table_source(&schema),
        None,
    )
    .unwrap()
    .build()
    .unwrap();
    for (name, qualified) in [
        (ObjectName(vec![]), true),
        (
            ObjectName::from(vec![Ident::new("t"), Ident::new("a")]),
            true,
        ),
        (
            ObjectName(vec![ObjectNamePart::Function(ObjectNamePartFunction {
                name: Ident::new("identifier"),
                args: vec![],
            })]),
            false,
        ),
        (
            ObjectName(vec![ObjectNamePart::UserVariable(Ident::new("a"))]),
            false,
        ),
    ] {
        let display = name.to_string();
        let projected = ExcludedColumn::from(name);
        assert_eq!(projected.to_string(), display);
        if qualified {
            assert!(matches!(projected, ExcludedColumn::DeferredQualified(_)));
        } else {
            assert!(matches!(
                projected,
                ExcludedColumn::DeferredNonIdentifier(_)
            ));
        }
        let options = WildcardOptions {
            exclude: Some(ExcludeSelectItem::Single(projected)),
            ..Default::default()
        };
        let error = crate::utils::expand_wildcard(plan.schema(), &plan, Some(&options))
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            if qualified {
                "This feature is not implemented: Qualified EXCLUDE names are not supported"
            } else {
                "Error during planning: EXCLUDE requires a column identifier"
            }
        );
    }
}

#[cfg(not(feature = "sql"))]
#[test]
fn no_sql_local_identifier_display_keeps_its_original_bracket_form() {
    let mut name = ident("a");
    name.quote_style = Some('"');
    let exclude = ExcludeSelectItem::Single(name.clone().into());
    let replace = ReplaceSelectElement {
        expr: "c1".into(),
        column_name: name,
        as_keyword: true,
    };
    assert_eq!(exclude.to_string(), "EXCLUDE [a]");
    assert_eq!(replace.to_string(), "c1 AS [a]");
}
