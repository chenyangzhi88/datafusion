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

use datafusion_common::{DFSchema, DataFusionError, Result, TableReference};
use datafusion_expr::Expr;
use datafusion_sql::planner::{ContextProvider, PlannerContext, SqlToRel};
use datafusion_sql::unparser::expr_to_sql;
use sqlparser::ast;
use sqlparser::dialect::{
    Dialect, GenericDialect, MsSqlDialect, MySqlDialect, PostgreSqlDialect,
};
use sqlparser::parser::Parser;

use crate::common::{MockContextProvider, MockSessionState};

#[test]
fn new_ast_statement_semantics_are_rejected_before_planning() -> Result<()> {
    let pg = PostgreSqlDialect {};
    let mysql = MySqlDialect {};
    let mssql = MsSqlDialect {};
    let context = MockContextProvider {
        state: MockSessionState::default(),
    };
    let planner = SqlToRel::new(&context);
    let cases: [(&dyn Dialect, &str, &str); 11] = [
        (
            &pg,
            "INSERT INTO person (id) OVERRIDING SYSTEM VALUE VALUES (1)",
            "INSERT OVERRIDING, OUTPUT or multiple targets are not supported",
        ),
        (
            &mssql,
            "INSERT INTO person (id) OUTPUT INSERTED.id VALUES (1)",
            "INSERT OVERRIDING, OUTPUT or multiple targets are not supported",
        ),
        (
            &mssql,
            "UPDATE person SET age = 1 OUTPUT INSERTED.age",
            "UPDATE inheritance modifiers, OUTPUT or ORDER BY are not supported",
        ),
        (
            &mssql,
            "DELETE FROM person OUTPUT DELETED.id",
            "DELETE inheritance modifiers or OUTPUT are not supported",
        ),
        (
            &pg,
            "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c WHERE n < 2) SEARCH DEPTH FIRST BY n SET ord SELECT * FROM c",
            "CTE SEARCH or CYCLE clauses are not supported",
        ),
        (
            &pg,
            "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c WHERE n < 2) CYCLE n SET mark USING path SELECT * FROM c",
            "CTE SEARCH or CYCLE clauses are not supported",
        ),
        (
            &pg,
            "CREATE FUNCTION f() RETURNS SETOF INTEGER LANGUAGE SQL AS 'SELECT 1'",
            "Set-returning CREATE FUNCTION is not supported",
        ),
        (
            &mysql,
            "SELECT 1 INTO @v",
            "SELECT INTO variables or files is not supported",
        ),
        (
            &mysql,
            "SELECT 1 INTO OUTFILE 'unification.out'",
            "SELECT INTO variables or files is not supported",
        ),
        (
            &mysql,
            "COMMIT RELEASE",
            "Transaction RELEASE is not supported",
        ),
        (
            &mysql,
            "ROLLBACK RELEASE",
            "Transaction RELEASE is not supported",
        ),
    ];

    for (dialect, sql, expected) in cases {
        let mut statements = Parser::parse_sql(dialect, sql)?;
        assert_eq!(statements.len(), 1, "{sql}");
        let error = planner
            .sql_statement_to_plan(statements.pop().unwrap())
            .expect_err(sql);
        let DataFusionError::NotImplemented(message) = error else {
            panic!("expected an unsupported semantic for {sql}, got {error}");
        };
        assert_eq!(message, expected, "{sql}");
    }
    Ok(())
}

#[test]
fn similar_to_and_like_keep_distinct_operators_patterns_and_escapes() -> Result<()> {
    let context = MockContextProvider {
        state: MockSessionState::default(),
    };
    let schema = context
        .get_table_source(TableReference::bare("person"))?
        .schema();
    let schema = DFSchema::try_from(schema)?;
    let planner = SqlToRel::new(&context);
    let dialect = GenericDialect {};
    let pg_dialect = PostgreSqlDialect {};

    for (sql, similar, negated, escape) in [
        ("first_name SIMILAR TO 'A_%'", true, false, None),
        (
            "first_name NOT SIMILAR TO 'A_+%' ESCAPE '!'",
            true,
            true,
            Some('!'),
        ),
        ("first_name LIKE 'A_+%' ESCAPE '!'", false, false, Some('!')),
        ("first_name NOT LIKE 'A_%'", false, true, None),
    ] {
        let syntax = Parser::new(&dialect).try_with_sql(sql)?.parse_expr()?;
        // PostgreSQL ESCAPE binds its native routine and pattern operator.
        // The generic AST retains LIKE/SIMILAR TO and their literal escape.
        let native = Parser::new(&pg_dialect).try_with_sql(sql)?.parse_expr()?;
        if let Some(escape) = escape {
            let (operand, pattern, escape_value) = match &syntax {
                ast::Expr::SimilarTo {
                    expr,
                    pattern,
                    escape_char,
                    ..
                }
                | ast::Expr::Like {
                    expr,
                    pattern,
                    escape_char,
                    ..
                } => (
                    expr,
                    pattern,
                    escape_char
                        .as_ref()
                        .expect("the generic AST retains ESCAPE"),
                ),
                _ => panic!("wrong generic pattern AST for {sql}: {syntax:?}"),
            };
            assert!(
                matches!(escape_value.as_ref(), ast::Expr::Value(value)
                    if value.value == ast::Value::SingleQuotedString(escape.to_string())),
                "{sql}"
            );
            let ast::Expr::BinaryOp { left, op, right } = &native else {
                panic!(
                    "expected PostgreSQL's native pattern operator for {sql}: {native:?}"
                );
            };
            assert_eq!(left, operand, "{sql}");
            let expected_operator = match (similar, negated) {
                (true, true) => "!~",
                (true, false) => "~",
                (false, true) => "!~~",
                (false, false) => "~~",
            };
            assert_eq!(
                op,
                &ast::BinaryOperator::Custom(expected_operator.to_owned()),
                "{sql}"
            );
            let ast::Expr::Function(function) = right.as_ref() else {
                panic!("expected PostgreSQL's escape routine for {sql}: {right:?}");
            };
            assert_eq!(
                function.name,
                ast::ObjectName::from(vec![
                    ast::Ident::new("pg_catalog"),
                    ast::Ident::new(if similar {
                        "similar_to_escape"
                    } else {
                        "like_escape"
                    }),
                ]),
                "{sql}",
            );
            let ast::FunctionArguments::List(arguments) = &function.args else {
                panic!("expected PostgreSQL's exact pattern and escape arguments");
            };
            assert_eq!(arguments.duplicate_treatment, None);
            assert!(arguments.clauses.is_empty());
            assert_eq!(
                arguments.args,
                vec![
                    ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(
                        pattern.as_ref().clone()
                    )),
                    ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(
                        *escape_value.clone(),
                    )),
                ],
                "{sql}",
            );
        } else {
            assert_eq!(
                native, syntax,
                "without ESCAPE the two AST contracts agree: {sql}"
            );
        }
        let logical = planner.sql_to_expr(syntax, &schema, &mut PlannerContext::new())?;
        let predicate = match (&logical, similar) {
            (Expr::SimilarTo(predicate), true) | (Expr::Like(predicate), false) => {
                predicate
            }
            _ => panic!("wrong logical operator for {sql}: {logical:?}"),
        };
        assert_eq!(predicate.negated, negated, "{sql}");
        assert_eq!(predicate.escape_char, escape, "{sql}");

        let output = expr_to_sql(&logical)?;
        match (&output, similar) {
            (
                ast::Expr::SimilarTo {
                    negated: actual, ..
                },
                true,
            )
            | (
                ast::Expr::Like {
                    negated: actual, ..
                },
                false,
            ) => {
                assert_eq!(*actual, negated, "{sql}");
            }
            _ => panic!("wrong unparsed operator for {sql}: {output:?}"),
        }
        assert_eq!(output.to_string(), sql, "{sql}");
        let reparsed = Parser::new(&dialect)
            .try_with_sql(&output.to_string())?
            .parse_expr()?;
        let roundtrip =
            planner.sql_to_expr(reparsed, &schema, &mut PlannerContext::new())?;
        assert_eq!(roundtrip, logical, "{sql}");
    }
    Ok(())
}

#[test]
fn native_mysql_byte_text_is_not_a_generic_set_string() -> Result<()> {
    use sqlparser::ast::{MySqlText, MySqlTextRole, SqlSourceAdmission, SqlSourceOwner};
    use sqlparser::tokenizer::{MySqlLexicalUnit, MySqlSourceAdapter, MySqlSourceError};
    use std::sync::Arc;

    // These fixtures contain ASCII SQL syntax and literal data in captured,
    // client byte domains. No literal-to-system/UTF8 conversion is supplied.
    struct ByteSource {
        charset: &'static str,
    }
    impl MySqlSourceAdapter for ByteSource {
        fn client_charset(&self) -> &str {
            self.charset
        }
        fn client_min_width(&self) -> u8 {
            1
        }
        fn lexical_unit(
            &self,
            input: &[u8],
        ) -> std::result::Result<MySqlLexicalUnit, MySqlSourceError> {
            let byte = *input.first().ok_or("missing fixture byte")?;
            Ok(MySqlLexicalUnit {
                width: 1,
                expected_width: 1,
                state: 0,
                valid: byte.is_ascii() || matches!(self.charset, "latin1" | "binary"),
                ascii: byte.is_ascii(),
                space: byte.is_ascii_whitespace(),
                control: byte.is_ascii_control(),
                identifier: byte.is_ascii_alphanumeric() || byte == b'_',
            })
        }
        fn system_identifier(
            &self,
            bytes: &[u8],
            _: bool,
        ) -> std::result::Result<(String, SqlSourceAdmission), MySqlSourceError> {
            if !bytes.is_ascii() {
                return Err("the fixture only admits ASCII identifiers".into());
            }
            Ok((
                String::from_utf8(bytes.to_vec()).unwrap(),
                self.admit(bytes.len())?,
            ))
        }
        fn system_text(
            &self,
            _: &MySqlText,
        ) -> std::result::Result<(String, SqlSourceAdmission), MySqlSourceError> {
            Err("the fixture supplies no native system-text conversion".into())
        }
        fn admit(
            &self,
            _: usize,
        ) -> std::result::Result<SqlSourceAdmission, MySqlSourceError> {
            Ok(SqlSourceAdmission::new(Arc::new(())))
        }
    }

    let context = MockContextProvider {
        state: MockSessionState::default(),
    };
    let planner = SqlToRel::new(&context);
    let dialect = MySqlDialect {};
    let cases: [(&str, &[u8], &[u8]); 3] = [
        ("utf8mb4", b"SET time_zone = 'UTC'", b"UTC"),
        ("latin1", b"SET time_zone = 'UTC\\0'", b"UTC\0"),
        ("binary", b"SET time_zone = '\xff\\0'", b"\xff\0"),
    ];
    for (charset, sql, expected_bytes) in cases {
        let owner =
            SqlSourceOwner::adopt(sql.to_vec(), SqlSourceAdmission::new(Arc::new(())));
        let adapter = ByteSource { charset };
        let parsed = Parser::new(&dialect)
            .try_with_mysql_source(owner.occurrence(0..sql.len()).unwrap(), &adapter)?
            .parse_next_mysql_source()?
            .expect("one actual byte-source SET statement");
        assert_eq!(parsed.source_occurrence.bytes().unwrap(), sql);
        let ast::Statement::Set(ast::Set::SingleAssignment { values, .. }) =
            &parsed.statement
        else {
            panic!("expected the actual SET assignment AST");
        };
        let [ast::Expr::Value(value)] = values.as_slice() else {
            panic!("expected one parser-issued literal");
        };
        let ast::Value::MySqlText(text) = &value.value else {
            panic!("decoded string input does not cover the native byte carrier");
        };
        assert_eq!(text.role(), MySqlTextRole::Literal);
        assert_eq!(text.charset(), charset);
        assert_eq!(text.bytes().unwrap(), expected_bytes);
        assert!(Arc::ptr_eq(text.source().owner().unwrap(), &owner));
        let expected_error = format!("Unsupported value {:?}", value.value);

        let error = planner
            .sql_statement_to_plan(parsed.statement.clone())
            .unwrap_err();
        let DataFusionError::Plan(message) = error else {
            panic!("expected the generic SET value rejection, got {error}");
        };
        assert_eq!(message, expected_error);
    }

    let mut statements =
        Parser::parse_sql(&PostgreSqlDialect {}, "SET timezone = 'UTC'")?;
    let plan = planner.sql_statement_to_plan(statements.pop().unwrap())?;
    let datafusion_expr::LogicalPlan::Statement(datafusion_expr::Statement::SetVariable(
        variable,
    )) = plan
    else {
        panic!("the ordinary decoded PostgreSQL SET must remain supported");
    };
    assert_eq!(variable.variable, "datafusion.execution.time_zone");
    assert_eq!(variable.value, "UTC");
    Ok(())
}
