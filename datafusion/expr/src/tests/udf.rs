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

use crate::{ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature};
use arrow::datatypes::DataType;
use datafusion_common::Result;
use datafusion_expr_common::signature::Volatility;
use std::any::Any;

#[derive(Debug, PartialEq, Eq, Hash)]
struct ArgumentsFirst(Signature);
impl ScalarUDFImpl for ArgumentsFirst {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "arguments_first"
    }
    fn signature(&self) -> &Signature {
        &self.0
    }
    fn return_type(&self, types: &[DataType]) -> Result<DataType> {
        Ok(types[0].clone())
    }
    fn requires_executor_initialization(&self) -> bool {
        true
    }
    fn initializes_arguments_first(&self) -> bool {
        true
    }
    fn invoke_with_args(&self, mut args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        Ok(args.args.remove(0))
    }
}

#[test]
fn aliases_preserve_the_original_executor_initialization_order() {
    let function =
        ScalarUDF::new_from_impl(ArgumentsFirst(Signature::any(1, Volatility::Volatile)));
    let aliased = function
        .clone()
        .with_aliases(["first"])
        .with_aliases(["second"]);
    assert!(function.inner().requires_executor_initialization());
    assert!(aliased.inner().requires_executor_initialization());
    assert!(aliased.inner().initializes_arguments_first());
    assert_eq!(
        aliased.aliases(),
        &["first".to_string(), "second".to_string()]
    );
    assert_eq!(aliased.signature(), function.signature());
}
