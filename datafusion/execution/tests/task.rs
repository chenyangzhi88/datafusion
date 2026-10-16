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

use std::sync::Arc;

use arrow::datatypes::DataType;
use datafusion_common::{Result, internal_err};
use datafusion_execution::{FunctionRegistry, TaskContext};
use datafusion_expr::{
    AggregateUDF, ScalarUDF, Volatility, WindowUDF, create_udaf, create_udf, create_udwf,
};

fn scalar(aliases: &[&'static str]) -> Arc<ScalarUDF> {
    Arc::new(
        create_udf(
            "scalar",
            vec![],
            DataType::Int32,
            Volatility::Immutable,
            Arc::new(|_| internal_err!("registration test does not invoke functions")),
        )
        .with_aliases(aliases.iter().copied()),
    )
}

fn aggregate(aliases: &[&'static str]) -> Arc<AggregateUDF> {
    Arc::new(
        create_udaf(
            "aggregate",
            vec![],
            Arc::new(DataType::Int32),
            Volatility::Immutable,
            Arc::new(|_| internal_err!("registration test does not create accumulators")),
            Arc::new(vec![DataType::Int32]),
        )
        .with_aliases(aliases.iter().copied()),
    )
}

fn window(aliases: &[&'static str]) -> Arc<WindowUDF> {
    Arc::new(
        create_udwf(
            "window",
            DataType::Int32,
            Arc::new(DataType::Int32),
            Volatility::Immutable,
            Arc::new(|| internal_err!("registration test does not create evaluators")),
        )
        .with_aliases(aliases.iter().copied()),
    )
}

#[test]
fn function_registration_keeps_task_clones_isolated() -> Result<()> {
    let mut original = TaskContext::default();
    let old_scalar = scalar(&["scalar_alias"]);
    let old_aggregate = aggregate(&["aggregate_alias"]);
    let old_window = window(&["window_alias"]);
    assert!(original.register_udf(Arc::clone(&old_scalar))?.is_none());
    assert!(
        original
            .register_udaf(Arc::clone(&old_aggregate))?
            .is_none()
    );
    assert!(original.register_udwf(Arc::clone(&old_window))?.is_none());

    let projected = original.clone_with_session_config(original.session_config().clone());
    for mut cloned in [original.clone(), projected] {
        let new_scalar = scalar(&["scalar_alias", "new_scalar_alias"]);
        let new_aggregate = aggregate(&["aggregate_alias", "new_aggregate_alias"]);
        let new_window = window(&["window_alias", "new_window_alias"]);
        assert!(Arc::ptr_eq(&cloned.udf("scalar")?, &old_scalar));
        assert!(Arc::ptr_eq(&cloned.udaf("aggregate")?, &old_aggregate));
        assert!(Arc::ptr_eq(&cloned.udwf("window")?, &old_window));
        assert!(Arc::ptr_eq(
            &cloned.register_udf(Arc::clone(&new_scalar))?.unwrap(),
            &old_scalar
        ));
        assert!(Arc::ptr_eq(
            &cloned.register_udaf(Arc::clone(&new_aggregate))?.unwrap(),
            &old_aggregate
        ));
        assert!(Arc::ptr_eq(
            &cloned.register_udwf(Arc::clone(&new_window))?.unwrap(),
            &old_window
        ));
        for name in ["scalar", "scalar_alias", "new_scalar_alias"] {
            assert!(Arc::ptr_eq(&cloned.udf(name)?, &new_scalar));
        }
        for name in ["aggregate", "aggregate_alias", "new_aggregate_alias"] {
            assert!(Arc::ptr_eq(&cloned.udaf(name)?, &new_aggregate));
        }
        for name in ["window", "window_alias", "new_window_alias"] {
            assert!(Arc::ptr_eq(&cloned.udwf(name)?, &new_window));
        }
        assert!(Arc::ptr_eq(&original.udf("scalar_alias")?, &old_scalar));
        assert!(Arc::ptr_eq(
            &original.udaf("aggregate_alias")?,
            &old_aggregate
        ));
        assert!(Arc::ptr_eq(&original.udwf("window_alias")?, &old_window));
        assert!(original.udf("new_scalar_alias").is_err());
        assert!(original.udaf("new_aggregate_alias").is_err());
        assert!(original.udwf("new_window_alias").is_err());
    }
    Ok(())
}
