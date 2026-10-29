use crate::expressions::Column;
use crate::{PhysicalExpr, ScalarFunctionExpr};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion_common::config::ConfigOptions;
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, Volatility};
use datafusion_expr::{ScalarUDF, ScalarUDFImpl, Signature};
use datafusion_physical_expr_common::physical_expr::is_volatile;
use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// Test helper to create a mock UDF with a specific volatility
#[derive(Debug, PartialEq, Eq, Hash)]
struct MockScalarUDF {
    signature: Signature,
}

impl ScalarUDFImpl for MockScalarUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "mock_function"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Int32)
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        Ok(ColumnarValue::Scalar(ScalarValue::Int32(Some(42))))
    }
}

#[test]
fn test_scalar_function_volatile_node() {
    // Create a volatile UDF
    let volatile_udf = Arc::new(ScalarUDF::from(MockScalarUDF {
        signature: Signature::uniform(1, vec![DataType::Float32], Volatility::Volatile),
    }));

    // Create a non-volatile UDF
    let stable_udf = Arc::new(ScalarUDF::from(MockScalarUDF {
        signature: Signature::uniform(1, vec![DataType::Float32], Volatility::Stable),
    }));

    let schema = Schema::new(vec![Field::new("a", DataType::Float32, false)]);
    let args = vec![Arc::new(Column::new("a", 0)) as Arc<dyn PhysicalExpr>];
    let config_options = Arc::new(ConfigOptions::new());

    // Test volatile function
    let volatile_expr = ScalarFunctionExpr::try_new(
        volatile_udf,
        args.clone(),
        &schema,
        Arc::clone(&config_options),
    )
    .unwrap();

    assert!(volatile_expr.is_volatile_node());
    let volatile_arc: Arc<dyn PhysicalExpr> = Arc::new(volatile_expr);
    assert!(is_volatile(&volatile_arc));

    // Test non-volatile function
    let stable_expr =
        ScalarFunctionExpr::try_new(stable_udf, args, &schema, config_options).unwrap();

    assert!(!stable_expr.is_volatile_node());
    let stable_arc: Arc<dyn PhysicalExpr> = Arc::new(stable_expr);
    assert!(!is_volatile(&stable_arc));
}

/// This crate does not depend on the execution memory pool. Observe the real
/// factory's final payload release here; compute's companion regression uses
/// the actual MemoryReservation at the same physical boundary.
#[derive(Debug)]
struct FactoryFieldRelease {
    name: &'static str,
    field: Weak<Field>,
    live: Arc<AtomicUsize>,
    releases: Arc<Mutex<Vec<(&'static str, bool)>>>,
}

impl Drop for FactoryFieldRelease {
    fn drop(&mut self) {
        self.releases
            .lock()
            .unwrap()
            .push((self.name, self.field.strong_count() != 0));
        self.live.fetch_sub(1, Ordering::SeqCst);
    }
}

#[derive(Debug)]
struct FieldLoanFactory {
    signature: Signature,
    output: Arc<Field>,
    // Like a real COLLATE declaration, this borrows only the child Field,
    // not its function/Item owner. The actual argument retains that owner.
    _child_field: Option<Arc<Field>>,
    _release: FactoryFieldRelease,
}

impl PartialEq for FieldLoanFactory {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.output, &other.output)
    }
}
impl Eq for FieldLoanFactory {}
impl std::hash::Hash for FieldLoanFactory {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&Arc::as_ptr(&self.output), state);
    }
}
impl ScalarUDFImpl for FieldLoanFactory {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "original_field_loan_factory"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(self.output.data_type().clone())
    }
    fn return_field_from_args(
        &self,
        _args: datafusion_expr::ReturnFieldArgs,
    ) -> Result<Arc<Field>> {
        Ok(Arc::clone(&self.output))
    }
    fn invoke_with_args(&self, _: ScalarFunctionArgs) -> Result<ColumnarValue> {
        panic!("Field ownership and Drop never request VALUE")
    }
}

fn field_loan_node(
    name: &'static str,
    output: Arc<Field>,
    child_field: Option<Arc<Field>>,
    arguments: Vec<Arc<dyn PhysicalExpr>>,
    live: Arc<AtomicUsize>,
    releases: Arc<Mutex<Vec<(&'static str, bool)>>>,
    typed_constructor: bool,
) -> Arc<dyn PhysicalExpr> {
    let weak = Arc::downgrade(&output);
    live.fetch_add(1, Ordering::SeqCst);
    let function = Arc::new(ScalarUDF::from(FieldLoanFactory {
        signature: if arguments.is_empty() {
            Signature::nullary(Volatility::Volatile)
        } else {
            Signature::any(arguments.len(), Volatility::Volatile)
        },
        output: Arc::clone(&output),
        _child_field: child_field,
        _release: FactoryFieldRelease {
            name,
            field: weak,
            live,
            releases,
        },
    }));
    let options = Arc::new(ConfigOptions::default());
    let expression = if typed_constructor {
        ScalarFunctionExpr::try_new(function, arguments, &Schema::empty(), options).unwrap()
    } else {
        ScalarFunctionExpr::new(name, function, arguments, output, options)
    };
    Arc::new(expression)
}

#[test]
fn scalar_function_releases_output_then_parent_field_loans_before_child_owners() {
    for typed_constructor in [false, true] {
        let live = Arc::new(AtomicUsize::new(0));
        let releases = Arc::new(Mutex::new(Vec::new()));
        let child_field = Arc::new(Field::new("child", DataType::Int64, false));
        let child_weak = Arc::downgrade(&child_field);
        let child = field_loan_node(
            "child",
            Arc::clone(&child_field),
            None,
            vec![],
            Arc::clone(&live),
            Arc::clone(&releases),
            typed_constructor,
        );
        let parent_field = Arc::new(Field::new("parent", DataType::Int64, false));
        let parent_weak = Arc::downgrade(&parent_field);
        let parent = field_loan_node(
            "parent",
            Arc::clone(&parent_field),
            Some(child_field),
            vec![child],
            Arc::clone(&live),
            Arc::clone(&releases),
            typed_constructor,
        );
        drop(parent_field);
        let last = Arc::clone(&parent);
        drop(parent);
        assert_eq!(live.load(Ordering::SeqCst), 2);
        assert!(parent_weak.upgrade().is_some());
        assert!(child_weak.upgrade().is_some());
        assert!(releases.lock().unwrap().is_empty());
        drop(last);
        assert_eq!(
            *releases.lock().unwrap(),
            vec![("parent", false), ("child", false)],
            "the parent's own Field drops before its factory; its child Field loan drops before the argument's original owner"
        );
        assert_eq!(live.load(Ordering::SeqCst), 0);
        assert!(parent_weak.upgrade().is_none());
        assert!(child_weak.upgrade().is_none());
    }
}
