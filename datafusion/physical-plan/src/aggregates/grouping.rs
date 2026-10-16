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

//! Internal grouping bitmap encoding and input-batch scratch admission.
// Arrays are charged before construction under the caller's aggregate stream
// reservation. The caller must keep the scratch reservation until all returned
// grouping arrays have been interned and dropped.

use super::PhysicalGroupBy;
use arrow::array::{
    ArrayRef, BinaryArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow::buffer::{Buffer, OffsetBuffer, ScalarBuffer};
use datafusion_common::{
    DataFusionError, Result, internal_datafusion_err, resources_datafusion_err,
};
use datafusion_execution::memory_pool::MemoryReservation;
use std::sync::Arc;

fn checked_add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right).ok_or_else(|| {
        resources_datafusion_err!("Grouping bitmap allocation size overflow")
    })
}
fn checked_mul(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right).ok_or_else(|| {
        resources_datafusion_err!("Grouping bitmap allocation size overflow")
    })
}

/// Byte and offset buffers created for one mask array. Allocator-private
/// rounding and existing Arrow shared-allocation headers are not inferred here.
fn mask_buffer_bytes(keys: usize, rows: usize) -> Result<usize> {
    if keys <= 64 {
        let width = if keys <= 8 {
            1
        } else if keys <= 16 {
            2
        } else if keys <= 32 {
            4
        } else {
            8
        };
        return checked_mul(rows, width);
    }
    let width = checked_add(keys, 7)? / 8;
    let values = checked_mul(rows, width)?;
    // Arrow Binary uses signed 32-bit offsets. Fail before allocating.
    i32::try_from(values).map_err(|_| {
        resources_datafusion_err!("Grouping bitmap exceeds Arrow Binary offset capacity")
    })?;
    checked_add(
        values,
        checked_mul(checked_add(rows, 1)?, size_of::<i32>())?,
    )
}

/// Validate the actual grouping shape and admit all newly built mask buffers
/// before evaluating expressions or allocating any bitmap.
pub(super) fn admit_grouping_masks(
    group_by: &PhysicalGroupBy,
    rows: usize,
    scratch: Option<&MemoryReservation>,
) -> Result<()> {
    if group_by
        .groups
        .iter()
        .any(|group| group.len() != group_by.expr.len())
    {
        return Err(internal_datafusion_err!(
            "Grouping mask shape does not match actual expressions"
        ));
    }
    if group_by.is_single() {
        return Ok(());
    }
    if group_by.null_expr.len() != group_by.expr.len() {
        return Err(internal_datafusion_err!(
            "Grouping null-expression shape does not match actual expressions"
        ));
    }
    let scratch = scratch.ok_or_else(|| {
        internal_datafusion_err!(
            "Grouping bitmap construction requires the aggregate stream reservation"
        )
    })?;
    let bytes = checked_mul(
        mask_buffer_bytes(group_by.expr.len(), rows)?,
        group_by.groups.len(),
    )?;
    scratch.try_resize(bytes)
}

fn reserved_vec<T>(capacity: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(capacity).map_err(|error| {
        DataFusionError::ResourcesExhausted(format!(
            "Grouping bitmap allocation failed: {error}"
        ))
    })?;
    Ok(values)
}

pub(super) fn group_id_array(group: &[bool], rows: usize) -> Result<ArrayRef> {
    if group.len() <= 64 {
        let id = group
            .iter()
            .fold(0u64, |acc, &is_null| (acc << 1) | u64::from(is_null));
        macro_rules! primitive {
            ($array:ty, $value:expr) => {{
                let mut values = reserved_vec(rows)?;
                values.resize(rows, $value);
                Ok(Arc::new(<$array>::from(values)) as ArrayRef)
            }};
        }
        return if group.len() <= 8 {
            primitive!(UInt8Array, id as u8)
        } else if group.len() <= 16 {
            primitive!(UInt16Array, id as u16)
        } else if group.len() <= 32 {
            primitive!(UInt32Array, id as u32)
        } else {
            primitive!(UInt64Array, id)
        };
    }
    // The preflight is repeated only for checked shape arithmetic, not admission.
    mask_buffer_bytes(group.len(), rows)?;
    let width = checked_add(group.len(), 7)? / 8;
    let data_len = checked_mul(width, rows)?;
    let mut offsets = reserved_vec(checked_add(rows, 1)?)?;
    let mut data = reserved_vec(data_len)?;
    offsets.push(0i32);
    if rows != 0 {
        data.resize(width, 0u8);
        for (index, excluded) in group.iter().enumerate() {
            if *excluded {
                let bit = group.len() - 1 - index;
                data[bit / 8] |= 1u8 << (bit % 8);
            }
        }
        offsets.push(
            i32::try_from(width).map_err(|_| {
                resources_datafusion_err!("Grouping bitmap offset overflow")
            })?,
        );
        for row in 1..rows {
            data.extend_from_within(0..width);
            offsets.push(i32::try_from(checked_mul(row + 1, width)?).map_err(|_| {
                resources_datafusion_err!("Grouping bitmap offset overflow")
            })?);
        }
    }
    Ok(Arc::new(BinaryArray::try_new(
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        Buffer::from(data),
        None,
    )?))
}

#[cfg(test)]
#[path = "tests/mod.rs"]
pub(in crate::aggregates) mod tests;
