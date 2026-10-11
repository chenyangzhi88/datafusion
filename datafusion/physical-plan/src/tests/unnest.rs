use super::super::*;
use arrow::array::{GenericListArray, NullBufferBuilder, OffsetSizeTrait, StringArray};
use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow::datatypes::{Field, Int32Type};
use datafusion_common::test_util::batches_to_string;
use insta::assert_snapshot;

fn bounded_pool(bytes: usize) -> Arc<dyn MemoryPool> {
    Arc::new(datafusion_execution::memory_pool::GreedyMemoryPool::new(
        bytes,
    ))
}

#[test]
fn retention_bound_covers_actual_input_views_and_output_owners_without_input_payload() -> Result<()>
{
    let child = DataType::Struct(vec![Field::new("v", DataType::Int64, true)].into());
    let data_type = DataType::List(Arc::new(Field::new("rows", child, true)));
    let views = allocation_view_bound(&data_type)?;
    for inputs in [0_usize, 1, 1000] {
        let bound = array_retention_bound(&data_type, std::iter::repeat_n(&data_type, inputs))?;
        assert_eq!(
            bound,
            (inputs + 1) * views * 128 + views * ALLOCATION_OWNER_BYTES
        );
    }
    let value = Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
        None::<Vec<Option<i32>>>,
    ]));
    let inputs: Vec<ArrayRef> = (0..1000).map(|_| value.clone() as ArrayRef).collect();
    let pool = bounded_pool(2 << 20);
    let reservation = MemoryConsumer::new("actual retention bound").register(&pool);
    let expected = value.data_type();
    reservation.try_grow(
        array_retention_bound(expected, inputs.iter().map(|input| input.data_type()))?
            + array_construction_bound(expected, 1000, 0)?,
    )?;
    let references = inputs
        .iter()
        .map(|input| input.as_ref())
        .collect::<Vec<_>>();
    let output = arrow::compute::concat(&references)?;
    let output = retain_array_allocations(vec![output], reservation, &inputs)?;
    assert_eq!(output[0].len(), 1000);
    assert_eq!(output[0].null_count(), 1000);
    let slice = output[0].slice(999, 1);
    drop(output);
    assert!(pool.reserved() > 0);
    drop(slice);
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[test]
fn immutable_buffer_owner_survives_slice_nulls_and_last_column_only() -> Result<()> {
    let pool = bounded_pool(1 << 20);
    let reservation = MemoryConsumer::new("test construction").register(&pool);
    reservation.try_grow(65536)?;
    let first: ArrayRef = Arc::new(StringArray::from(vec![Some("left"), None, Some("right")]));
    let second: ArrayRef = Arc::new(Int64Array::from(vec![Some(10), None, Some(30)]));
    let first_pointer = first.to_data().buffers()[1].as_ptr();
    let retained = retain_array_allocations(vec![first, second], reservation, &[])?;
    assert_eq!(retained[0].to_data().buffers()[1].as_ptr(), first_pointer);
    let slice = retained[0].slice(1, 2);
    let before_drop = pool.reserved();
    assert!(before_drop > 0);
    drop(retained);
    assert!(pool.reserved() > 0);
    assert!(
        pool.reserved() < before_drop,
        "independent discarded column must release its own allocations"
    );
    assert_eq!(
        slice.as_string::<i32>().iter().collect::<Vec<_>>(),
        vec![None, Some("right")]
    );
    let clone = Arc::clone(&slice);
    drop(slice);
    assert!(pool.reserved() > 0);
    drop(clone);
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[test]
fn shared_input_dictionary_and_nested_buffers_keep_the_original_owner() -> Result<()> {
    use arrow::array::{DictionaryArray, Int32Array};
    let pool = bounded_pool(1 << 20);
    let source_reservation = MemoryConsumer::new("input construction").register(&pool);
    source_reservation.try_grow(32768)?;
    let source = retain_array_allocation(
        Arc::new(StringArray::from(vec!["one", "two"])),
        source_reservation,
    )?;
    let original_bytes = pool.reserved();
    let keys = Int32Array::from(vec![Some(0), None, Some(1)]);
    let dictionary: ArrayRef = Arc::new(DictionaryArray::<Int32Type>::try_new(
        keys,
        Arc::clone(&source),
    )?);
    let output_reservation = MemoryConsumer::new("dictionary keys").register(&pool);
    output_reservation.try_grow(32768)?;
    let mut output =
        retain_array_allocations(vec![dictionary], output_reservation, &[Arc::clone(&source)])?;
    let field = Arc::new(Field::new(
        "dictionary",
        output[0].data_type().clone(),
        true,
    ));
    let structure = StructArray::new(vec![field].into(), vec![output.remove(0)], None);
    let leaf = Arc::clone(structure.column(0));
    drop(structure);
    drop(source);
    assert!(
        pool.reserved() > original_bytes,
        "new keys and reused input values both remain live"
    );
    assert!(leaf.is_null(1));
    drop(leaf);
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[test]
fn bounded_unnest_admits_before_kernels_and_keeps_padding_and_retained_leaf() -> Result<()> {
    let list: ArrayRef = Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
        Some(vec![Some(1), Some(2)]),
        None,
        Some(vec![Some(3)]),
    ]));
    let other: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("list", list.data_type().clone(), true),
        Field::new("other", DataType::Utf8, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("list", DataType::Int32, true),
        Field::new("other", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(input_schema, vec![Arc::clone(&list), other])?;
    let specs = [ListUnnest {
        index_in_input_schema: 0,
        depth: 1,
    }];
    let options = UnnestOptions {
        preserve_nulls: true,
        recursions: vec![],
    };
    let tiny = bounded_pool(1);
    let error = build_batch(
        &batch,
        &output_schema,
        &specs,
        &HashSet::new(),
        &options,
        &tiny,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        datafusion_common::DataFusionError::ResourcesExhausted(_)
    ));
    assert_eq!(tiny.reserved(), 0);
    assert_eq!(list.len(), 3);
    let pool = bounded_pool(1 << 20);
    let output = build_batch(
        &batch,
        &output_schema,
        &specs,
        &HashSet::new(),
        &options,
        &pool,
    )?
    .unwrap();
    assert_eq!(output.num_rows(), 4);
    assert_eq!(
        output
            .column(0)
            .as_primitive::<Int32Type>()
            .iter()
            .collect::<Vec<_>>(),
        vec![Some(1), Some(2), None, Some(3)]
    );
    let retained_leaf = output.column(1).slice(1, 2);
    drop(output);
    assert!(pool.reserved() > 0);
    assert_eq!(
        retained_leaf.as_string::<i32>().iter().collect::<Vec<_>>(),
        vec![Some("a"), Some("b")]
    );
    drop(retained_leaf);
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[test]
fn construction_overflow_and_empty_arrays_release_without_payload() -> Result<()> {
    assert!(array_construction_bound(&DataType::Int64, usize::MAX, 0).is_err());
    assert!(array_construction_bound(&DataType::Utf8, 1, usize::MAX).is_err());
    let pool = bounded_pool(65536);
    let reservation = MemoryConsumer::new("empty construction").register(&pool);
    reservation.try_grow(8192)?;
    let output =
        retain_array_allocation(Arc::new(StringArray::from(Vec::<&str>::new())), reservation)?;
    assert_eq!(output.len(), 0);
    drop(output);
    assert_eq!(pool.reserved(), 0);
    let reservation = MemoryConsumer::new("empty but allocated construction").register(&pool);
    reservation.try_grow(8192)?;
    let bytes = arrow::buffer::Buffer::from(Vec::<u8>::with_capacity(512));
    assert!(bytes.is_empty());
    assert!(bytes.capacity() >= 512);
    let output =
        arrow::array::BinaryArray::try_new(OffsetBuffer::new(vec![0i32].into()), bytes, None)?;
    let output = retain_array_allocation(Arc::new(output), reservation)?;
    assert_eq!(output.len(), 0);
    assert!(
        pool.reserved() >= 512,
        "empty logical payload still owns its actual allocation capacity"
    );
    drop(output);
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[tokio::test]
async fn pending_stream_drop_keeps_only_the_retained_output_allocation() -> Result<()> {
    use crate::stream::RecordBatchStreamAdapter;
    use futures::FutureExt;
    let pool = bounded_pool(1 << 20);
    let source_owner = MemoryConsumer::new("stream input").register(&pool);
    source_owner.try_grow(32768)?;
    let list: ArrayRef = Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
        Some(vec![Some(11), Some(12)]),
        Some(vec![Some(13)]),
    ]));
    let list = retain_array_allocation(list, source_owner)?;
    let input_schema = Arc::new(Schema::new(vec![Field::new(
        "list",
        list.data_type().clone(),
        true,
    )]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("list", DataType::Int32, true)]));
    let input = RecordBatch::try_new(Arc::clone(&input_schema), vec![list])?;
    let input = futures::stream::iter(vec![Ok(input)]).chain(futures::stream::pending());
    let mut stream = UnnestStream {
        input: Box::pin(RecordBatchStreamAdapter::new(input_schema, input)),
        schema: output_schema,
        list_type_columns: vec![ListUnnest {
            index_in_input_schema: 0,
            depth: 1,
        }],
        struct_column_indices: HashSet::new(),
        options: UnnestOptions {
            preserve_nulls: true,
            recursions: vec![],
        },
        metrics: UnnestMetrics::new(0, &ExecutionPlanMetricsSet::new()),
        pool: Arc::clone(&pool),
    };
    let output = stream.next().await.expect("first actual batch")?;
    assert_eq!(
        output
            .column(0)
            .as_primitive::<Int32Type>()
            .iter()
            .collect::<Vec<_>>(),
        vec![Some(11), Some(12), Some(13)]
    );
    let leaf = output.column(0).slice(1, 2);
    drop(output);
    assert!(
        stream.next().now_or_never().is_none(),
        "the actual input is pending, not EOF"
    );
    drop(stream);
    assert!(
        pool.reserved() > 0,
        "cancelling/dropping the pending stream must not release a retained output column"
    );
    assert_eq!(
        leaf.as_primitive::<Int32Type>().iter().collect::<Vec<_>>(),
        vec![Some(12), Some(13)]
    );
    drop(leaf);
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

// Create a GenericListArray with the following list values:
//  [A, B, C], [], NULL, [D], NULL, [NULL, F]
fn make_generic_array<OffsetSize>() -> GenericListArray<OffsetSize>
where
    OffsetSize: OffsetSizeTrait,
{
    let mut values = vec![];
    let mut offsets: Vec<OffsetSize> = vec![OffsetSize::zero()];
    let mut valid = NullBufferBuilder::new(6);

    // [A, B, C]
    values.extend_from_slice(&[Some("A"), Some("B"), Some("C")]);
    offsets.push(OffsetSize::from_usize(values.len()).unwrap());
    valid.append_non_null();

    // []
    offsets.push(OffsetSize::from_usize(values.len()).unwrap());
    valid.append_non_null();

    // NULL with non-zero value length
    // Issue https://github.com/apache/datafusion/issues/9932
    values.push(Some("?"));
    offsets.push(OffsetSize::from_usize(values.len()).unwrap());
    valid.append_null();

    // [D]
    values.push(Some("D"));
    offsets.push(OffsetSize::from_usize(values.len()).unwrap());
    valid.append_non_null();

    // Another NULL with zero value length
    offsets.push(OffsetSize::from_usize(values.len()).unwrap());
    valid.append_null();

    // [NULL, F]
    values.extend_from_slice(&[None, Some("F")]);
    offsets.push(OffsetSize::from_usize(values.len()).unwrap());
    valid.append_non_null();

    let field = Arc::new(Field::new_list_field(DataType::Utf8, true));
    GenericListArray::<OffsetSize>::new(
        field,
        OffsetBuffer::new(offsets.into()),
        Arc::new(StringArray::from(values)),
        valid.finish(),
    )
}

// Create a FixedSizeListArray with the following list values:
//  [A, B], NULL, [C, D], NULL, [NULL, F], [NULL, NULL]
fn make_fixed_list() -> FixedSizeListArray {
    let values = Arc::new(StringArray::from_iter([
        Some("A"),
        Some("B"),
        None,
        None,
        Some("C"),
        Some("D"),
        None,
        None,
        None,
        Some("F"),
        None,
        None,
    ]));
    let field = Arc::new(Field::new_list_field(DataType::Utf8, true));
    let valid = NullBuffer::from(vec![true, false, true, false, true, true]);
    FixedSizeListArray::new(field, 2, values, Some(valid))
}

fn verify_unnest_list_array(
    list_array: &dyn ListArrayType,
    lengths: Vec<i64>,
    expected: Vec<Option<&str>>,
) -> Result<()> {
    let length_array = Int64Array::from(lengths);
    let unnested_array = unnest_list_array(list_array, &length_array, 3 * 6)?;
    let strs = unnested_array.as_string::<i32>().iter().collect::<Vec<_>>();
    assert_eq!(strs, expected);
    Ok(())
}

#[test]
fn test_build_batch_list_arr_recursive() -> Result<()> {
    // col1                             | col2
    // [[1,2,3],null,[4,5]]             | ['a','b']
    // [[7,8,9,10], null, [11,12,13]]   | ['c','d']
    // null                             | ['e']
    let list_arr1 = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
        Some(vec![Some(1), Some(2), Some(3)]),
        None,
        Some(vec![Some(4), Some(5)]),
        Some(vec![Some(7), Some(8), Some(9), Some(10)]),
        None,
        Some(vec![Some(11), Some(12), Some(13)]),
    ]);

    let list_arr1_ref = Arc::new(list_arr1) as ArrayRef;
    let offsets = OffsetBuffer::from_lengths([3, 3, 0]);
    let mut nulls = NullBufferBuilder::new(3);
    nulls.append_non_null();
    nulls.append_non_null();
    nulls.append_null();
    // list<list<int32>>
    let col1_field = Field::new_list_field(
        DataType::List(Arc::new(Field::new_list_field(
            list_arr1_ref.data_type().to_owned(),
            true,
        ))),
        true,
    );
    let col1 = ListArray::new(
        Arc::new(Field::new_list_field(
            list_arr1_ref.data_type().to_owned(),
            true,
        )),
        offsets,
        list_arr1_ref,
        nulls.finish(),
    );

    let list_arr2 = StringArray::from(vec![Some("a"), Some("b"), Some("c"), Some("d"), Some("e")]);

    let offsets = OffsetBuffer::from_lengths([2, 2, 1]);
    let mut nulls = NullBufferBuilder::new(3);
    nulls.append_n_non_nulls(3);
    let col2_field = Field::new(
        "col2",
        DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, true))),
        true,
    );
    let col2 = GenericListArray::<i32>::new(
        Arc::new(Field::new_list_field(DataType::Utf8, true)),
        OffsetBuffer::new(offsets.into()),
        Arc::new(list_arr2),
        nulls.finish(),
    );
    // convert col1 and col2 to a record batch
    let schema = Arc::new(Schema::new(vec![col1_field, col2_field]));
    let out_schema = Arc::new(Schema::new(vec![
        Field::new(
            "col1_unnest_placeholder_depth_1",
            DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true))),
            true,
        ),
        Field::new("col1_unnest_placeholder_depth_2", DataType::Int32, true),
        Field::new("col2_unnest_placeholder_depth_1", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(col1) as ArrayRef, Arc::new(col2) as ArrayRef],
    )
    .unwrap();
    let list_type_columns = vec![
        ListUnnest {
            index_in_input_schema: 0,
            depth: 1,
        },
        ListUnnest {
            index_in_input_schema: 0,
            depth: 2,
        },
        ListUnnest {
            index_in_input_schema: 1,
            depth: 1,
        },
    ];
    let ret = build_batch(
        &batch,
        &out_schema,
        list_type_columns.as_ref(),
        &HashSet::default(),
        &UnnestOptions {
            preserve_nulls: true,
            recursions: vec![],
        },
        &(Arc::new(datafusion_execution::memory_pool::UnboundedMemoryPool::default())
            as Arc<dyn MemoryPool>),
    )?
    .unwrap();

    assert_snapshot!(batches_to_string(&[ret]),
    @r"
    +---------------------------------+---------------------------------+---------------------------------+
    | col1_unnest_placeholder_depth_1 | col1_unnest_placeholder_depth_2 | col2_unnest_placeholder_depth_1 |
    +---------------------------------+---------------------------------+---------------------------------+
    | [1, 2, 3]                       | 1                               | a                               |
    |                                 | 2                               | b                               |
    | [4, 5]                          | 3                               |                                 |
    | [1, 2, 3]                       |                                 | a                               |
    |                                 |                                 | b                               |
    | [4, 5]                          |                                 |                                 |
    | [1, 2, 3]                       | 4                               | a                               |
    |                                 | 5                               | b                               |
    | [4, 5]                          |                                 |                                 |
    | [7, 8, 9, 10]                   | 7                               | c                               |
    |                                 | 8                               | d                               |
    | [11, 12, 13]                    | 9                               |                                 |
    |                                 | 10                              |                                 |
    | [7, 8, 9, 10]                   |                                 | c                               |
    |                                 |                                 | d                               |
    | [11, 12, 13]                    |                                 |                                 |
    | [7, 8, 9, 10]                   | 11                              | c                               |
    |                                 | 12                              | d                               |
    | [11, 12, 13]                    | 13                              |                                 |
    |                                 |                                 | e                               |
    +---------------------------------+---------------------------------+---------------------------------+
    ");
    Ok(())
}

#[test]
fn test_unnest_list_array() -> Result<()> {
    // [A, B, C], [], NULL, [D], NULL, [NULL, F]
    let list_array = make_generic_array::<i32>();
    verify_unnest_list_array(
        &list_array,
        vec![3, 2, 1, 2, 0, 3],
        vec![
            Some("A"),
            Some("B"),
            Some("C"),
            None,
            None,
            None,
            Some("D"),
            None,
            None,
            Some("F"),
            None,
        ],
    )?;

    // [A, B], NULL, [C, D], NULL, [NULL, F], [NULL, NULL]
    let list_array = make_fixed_list();
    verify_unnest_list_array(
        &list_array,
        vec![3, 1, 2, 0, 2, 3],
        vec![
            Some("A"),
            Some("B"),
            None,
            None,
            Some("C"),
            Some("D"),
            None,
            Some("F"),
            None,
            None,
            None,
        ],
    )?;

    Ok(())
}

fn verify_longest_length(
    list_arrays: &[ArrayRef],
    preserve_nulls: bool,
    expected: Vec<i64>,
) -> Result<()> {
    let options = UnnestOptions {
        preserve_nulls,
        recursions: vec![],
    };
    let longest_length = find_longest_length(list_arrays, &options)?;
    let expected_array = Int64Array::from(expected);
    assert_eq!(
        longest_length
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap(),
        &expected_array
    );
    Ok(())
}

#[test]
fn test_longest_list_length() -> Result<()> {
    // Test with single ListArray
    //  [A, B, C], [], NULL, [D], NULL, [NULL, F]
    let list_array = Arc::new(make_generic_array::<i32>()) as ArrayRef;
    verify_longest_length(&[Arc::clone(&list_array)], false, vec![3, 0, 0, 1, 0, 2])?;
    verify_longest_length(&[Arc::clone(&list_array)], true, vec![3, 0, 1, 1, 1, 2])?;

    // Test with single LargeListArray
    //  [A, B, C], [], NULL, [D], NULL, [NULL, F]
    let list_array = Arc::new(make_generic_array::<i64>()) as ArrayRef;
    verify_longest_length(&[Arc::clone(&list_array)], false, vec![3, 0, 0, 1, 0, 2])?;
    verify_longest_length(&[Arc::clone(&list_array)], true, vec![3, 0, 1, 1, 1, 2])?;

    // Test with single FixedSizeListArray
    //  [A, B], NULL, [C, D], NULL, [NULL, F], [NULL, NULL]
    let list_array = Arc::new(make_fixed_list()) as ArrayRef;
    verify_longest_length(&[Arc::clone(&list_array)], false, vec![2, 0, 2, 0, 2, 2])?;
    verify_longest_length(&[Arc::clone(&list_array)], true, vec![2, 1, 2, 1, 2, 2])?;

    // Test with multiple list arrays
    //  [A, B, C], [], NULL, [D], NULL, [NULL, F]
    //  [A, B], NULL, [C, D], NULL, [NULL, F], [NULL, NULL]
    let list1 = Arc::new(make_generic_array::<i32>()) as ArrayRef;
    let list2 = Arc::new(make_fixed_list()) as ArrayRef;
    let list_arrays = vec![Arc::clone(&list1), Arc::clone(&list2)];
    verify_longest_length(&list_arrays, false, vec![3, 0, 2, 1, 2, 2])?;
    verify_longest_length(&list_arrays, true, vec![3, 1, 2, 1, 2, 2])?;

    Ok(())
}

#[test]
fn test_create_take_indices() -> Result<()> {
    let length_array = Int64Array::from(vec![2, 3, 1]);
    let take_indices = create_take_indices(&length_array, 6);
    let expected = Int64Array::from(vec![0, 0, 1, 1, 1, 2]);
    assert_eq!(take_indices, expected);
    Ok(())
}
