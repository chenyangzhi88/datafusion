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

use crate::{AggregateUDF, AggregateUDFImpl};
use arrow::datatypes::{DataType, FieldRef};
use datafusion_common::Result;
use datafusion_expr_common::accumulator::Accumulator;
use datafusion_expr_common::signature::{Signature, Volatility};
use datafusion_functions_aggregate_common::accumulator::{AccumulatorArgs, StateFieldsArgs};
use std::any::Any;
use std::cmp::Ordering;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
#[test]
fn private_alias_rebuild_preserves_ordered_capability_exact_alias_chain_and_foreign_identity() {
    #[derive(Debug, PartialEq, Eq, Hash)]
    struct Ordered(AMeanUdf);
    impl AggregateUDFImpl for Ordered {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn name(&self) -> &str {
            self.0.name()
        }
        fn signature(&self) -> &Signature {
            self.0.signature()
        }
        fn return_type(&self, args: &[DataType]) -> Result<DataType> {
            self.0.return_type(args)
        }
        fn accumulator(&self, args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
            self.0.accumulator(args)
        }
        fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
            self.0.state_fields(args)
        }
        fn requires_ordered_input(&self) -> bool {
            true
        }
    }
    let original = AggregateUDF::from(Ordered(AMeanUdf::new()));
    assert_eq!(original.unaliased_inner_rebuild_bytes().unwrap(), 0);
    let aliases = original
        .with_aliases(["first", "other"])
        .with_aliases(["last"]);
    assert!(aliases.requires_ordered_input());
    let bytes = aliases.unaliased_inner_rebuild_bytes().unwrap();
    assert!(
        bytes
            >= aliases
                .aliases()
                .iter()
                .map(String::capacity)
                .sum::<usize>()
    );
    let replacement = Arc::new(Ordered(AMeanUdf::new())) as Arc<dyn AggregateUDFImpl>;
    let rebuilt = aliases.with_unaliased_inner(Arc::clone(&replacement));
    assert_eq!(rebuilt.aliases(), aliases.aliases());
    assert!(Arc::ptr_eq(rebuilt.unaliased_inner(), &replacement));
    assert!(
        rebuilt.unaliased_inner_rebuild_bytes().unwrap() <= bytes,
        "String/Vec clones may retire unused original capacity, never under-admit"
    );
    assert!(rebuilt.requires_ordered_input());
    let foreign = AggregateUDF::from(BMeanUdf::new()).with_aliases(["first"]);
    assert!(!foreign.requires_ordered_input());
    assert!(foreign.unaliased_inner().as_any().is::<BMeanUdf>());
}

#[test]
fn unaliased_inner_borrows_only_the_existing_private_alias_owner() {
    let original = AggregateUDF::from(AMeanUdf::new());
    let inner = Arc::clone(original.inner());
    let aliased = original
        .clone()
        .with_aliases(["first"])
        .with_aliases(["second"]);
    assert!(Arc::ptr_eq(&inner, aliased.unaliased_inner()));
    assert!(
        aliased
            .inner()
            .as_any()
            .downcast_ref::<AMeanUdf>()
            .is_none()
    );
    assert!(
        aliased
            .unaliased_inner()
            .as_any()
            .downcast_ref::<AMeanUdf>()
            .is_some()
    );
    assert_eq!(aliased.aliases(), ["first".to_owned(), "second".to_owned()]);
    assert!(Arc::ptr_eq(original.inner(), original.unaliased_inner()));
    let other = AggregateUDF::from(BMeanUdf::new()).with_aliases(["a"]);
    assert!(
        other
            .unaliased_inner()
            .as_any()
            .downcast_ref::<BMeanUdf>()
            .is_some()
    );
    assert!(!Arc::ptr_eq(&inner, other.unaliased_inner()));
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct AMeanUdf {
    signature: Signature,
}

impl AMeanUdf {
    fn new() -> Self {
        Self {
            signature: Signature::uniform(1, vec![DataType::Float64], Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for AMeanUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "a"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _args: &[DataType]) -> Result<DataType> {
        unimplemented!()
    }
    fn accumulator(&self, _acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        unimplemented!()
    }
    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        unimplemented!()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BMeanUdf {
    signature: Signature,
}
impl BMeanUdf {
    fn new() -> Self {
        Self {
            signature: Signature::uniform(1, vec![DataType::Float64], Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for BMeanUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "b"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _args: &[DataType]) -> Result<DataType> {
        unimplemented!()
    }
    fn accumulator(&self, _acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        unimplemented!()
    }
    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        unimplemented!()
    }
}

#[test]
fn test_partial_eq() {
    let a1 = AggregateUDF::from(AMeanUdf::new());
    let a2 = AggregateUDF::from(AMeanUdf::new());
    let eq = a1 == a2;
    assert!(eq);
    assert_eq!(a1, a2);
    assert_eq!(hash(a1), hash(a2));
}

#[test]
fn test_partial_ord() {
    // Test validates that partial ord is defined for AggregateUDF using the name and signature,
    // not intended to exhaustively test all possibilities
    let a1 = AggregateUDF::from(AMeanUdf::new());
    let a2 = AggregateUDF::from(AMeanUdf::new());
    assert_eq!(a1.partial_cmp(&a2), Some(Ordering::Equal));

    let b1 = AggregateUDF::from(BMeanUdf::new());
    assert!(a1 < b1);
    assert!(!(a1 == b1));
}

fn hash<T: Hash>(value: T) -> u64 {
    let hasher = &mut DefaultHasher::new();
    value.hash(hasher);
    hasher.finish()
}
