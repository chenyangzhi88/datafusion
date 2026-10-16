// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file distributed
// with this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with
// the License. You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
use arrow::datatypes::{DataType, Field, Schema};
use datafusion_common::{DFSchema, Result, ScalarValue};
use datafusion_expr::{
    Expr, ExprSchemable,
    execution_props::ExecutionProps,
    var_provider::{VarProvider, VarType},
};
use datafusion_physical_expr::create_physical_expr;
use std::{collections::HashMap, sync::Arc};
#[derive(Debug)]
struct Variable {
    value: ScalarValue,
}
impl VarProvider for Variable {
    fn get_value(&self, _: Vec<String>) -> Result<ScalarValue> {
        Ok(self.value.clone())
    }
    fn get_type(&self, _: &[String]) -> Option<DataType> {
        Some(self.value.data_type())
    }
}
#[test]
fn system_and_user_variable_physical_literals_keep_the_selected_field_metadata() {
    let types = [
        DataType::Int32,
        DataType::Utf8,
        DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
        DataType::Struct(
            vec![Arc::new(Field::new("member", DataType::Int32, true))].into(),
        ),
    ];
    for datatype in types {
        for (kind, name) in [
            (VarType::System, "@@runtime"),
            (VarType::UserDefined, "@runtime"),
        ] {
            let metadata = HashMap::from([
                ("type_identity".into(), "statement-selected".into()),
                ("typmod".into(), "12".into()),
                ("collation".into(), "950".into()),
                (
                    "descriptor".into(),
                    "owned-array-or-record-descriptor".into(),
                ),
            ]);
            let field = Arc::new(
                Field::new("selected", datatype.clone(), true)
                    .with_metadata(metadata.clone()),
            );
            let logical = Expr::ScalarVariable(field, vec![name.into()]);
            assert_eq!(
                logical.to_field(&DFSchema::empty()).unwrap().1.metadata(),
                &metadata
            );
            let value = ScalarValue::try_new_null(&datatype).unwrap();
            let mut properties = ExecutionProps::new();
            properties.add_var_provider(kind, Arc::new(Variable { value }));
            let physical =
                create_physical_expr(&logical, &DFSchema::empty(), &properties).unwrap();
            let actual = physical.return_field(&Schema::empty()).unwrap();
            assert_eq!(actual.data_type(), &datatype);
            assert_eq!(actual.metadata(), &metadata);
        }
    }
}
