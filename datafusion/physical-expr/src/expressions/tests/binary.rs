use std::sync::Arc;

use arrow::array::{Array, BooleanArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion_common::{DFSchema, ScalarValue};
use datafusion_expr::execution_props::ExecutionProps;
use datafusion_expr::{Expr, Operator, col};

use crate::PhysicalExpr;
use crate::expressions::{BinaryExpr, Column, Literal};

#[test]
fn null_safe_comparisons_have_nonnullable_fields_and_real_array_scalar_values() {
    for left_nullable in [false, true] {
        for right_nullable in [false, true] {
            let schema = Arc::new(Schema::new(vec![
                Field::new("left", DataType::Boolean, left_nullable),
                Field::new("right", DataType::Boolean, right_nullable),
            ]));
            let left = vec![
                Some(true),
                Some(false),
                if left_nullable { None } else { Some(true) },
            ];
            let right = vec![
                Some(true),
                Some(true),
                if right_nullable { None } else { Some(false) },
            ];
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(BooleanArray::from(left.clone())),
                    Arc::new(BooleanArray::from(right.clone())),
                ],
            )
            .unwrap();
            for op in [Operator::IsDistinctFrom, Operator::IsNotDistinctFrom] {
                let expression = BinaryExpr::new(
                    Arc::new(Column::new("left", 0)),
                    op,
                    Arc::new(Column::new("right", 1)),
                );
                assert!(!expression.nullable(&schema).unwrap());
                let field = expression.return_field(&schema).unwrap();
                assert_eq!(field.data_type(), &DataType::Boolean);
                assert!(!field.is_nullable());
                let output = expression.evaluate(&batch).unwrap().into_array(3).unwrap();
                let output = output.as_any().downcast_ref::<BooleanArray>().unwrap();
                assert_eq!(output.null_count(), 0);
                for row in 0..3 {
                    assert_eq!(
                        output.value(row),
                        if op == Operator::IsDistinctFrom {
                            left[row] != right[row]
                        } else {
                            left[row] == right[row]
                        }
                    );
                }
                for a in [None, Some(false), Some(true)] {
                    for b in [None, Some(false), Some(true)] {
                        let scalar = BinaryExpr::new(
                            Arc::new(Literal::new(ScalarValue::Boolean(a))),
                            op,
                            Arc::new(Literal::new(ScalarValue::Boolean(b))),
                        );
                        assert!(
                            !scalar.return_field(&Schema::empty()).unwrap().is_nullable()
                        );
                        assert_eq!(
                            scalar
                                .evaluate(&batch)
                                .unwrap()
                                .into_array(3)
                                .unwrap()
                                .as_any()
                                .downcast_ref::<BooleanArray>()
                                .unwrap()
                                .value(0),
                            if op == Operator::IsDistinctFrom {
                                a != b
                            } else {
                                a == b
                            }
                        );
                    }
                }
            }
            let equality = BinaryExpr::new(
                Arc::new(Column::new("left", 0)),
                Operator::Eq,
                Arc::new(Column::new("right", 1)),
            );
            assert_eq!(
                equality.nullable(&schema).unwrap(),
                left_nullable || right_nullable
            );
            assert_eq!(
                equality.return_field(&schema).unwrap().is_nullable(),
                left_nullable || right_nullable
            );
        }
    }
}

#[test]
fn all_six_logical_is_predicates_share_the_nonnullable_physical_factory() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Boolean, true)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(BooleanArray::from(vec![
            Some(true),
            Some(false),
            None,
        ]))],
    )
    .unwrap();
    let input = DFSchema::try_from(schema.as_ref().clone()).unwrap();
    for (expression, expected) in [
        (Expr::IsTrue(Box::new(col("v"))), [true, false, false]),
        (Expr::IsNotTrue(Box::new(col("v"))), [false, true, true]),
        (Expr::IsFalse(Box::new(col("v"))), [false, true, false]),
        (Expr::IsNotFalse(Box::new(col("v"))), [true, false, true]),
        (Expr::IsUnknown(Box::new(col("v"))), [false, false, true]),
        (Expr::IsNotUnknown(Box::new(col("v"))), [true, true, false]),
    ] {
        let physical =
            crate::create_physical_expr(&expression, &input, &ExecutionProps::new())
                .unwrap();
        assert!(physical.as_any().is::<BinaryExpr>());
        assert!(!physical.return_field(&schema).unwrap().is_nullable());
        let output = physical.evaluate(&batch).unwrap().into_array(3).unwrap();
        let output = output.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(output.null_count(), 0);
        assert_eq!(
            output.iter().collect::<Vec<_>>(),
            expected.into_iter().map(Some).collect::<Vec<_>>()
        );
    }
}
