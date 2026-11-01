//! Property index structures for fast equality and range lookups.

use std::collections::HashMap;

use bincode::{Decode, Encode};

/// Type-dispatched property index.
#[derive(Encode, Decode, Clone, Debug)]
pub enum PropIndex {
    UInt(UIntIndex),
    Int(IntIndex),
    Bool(BoolIndex),
    String(StringIndex),
    Enum(EnumIndex),
}

/// Sorted-array index for `u32` properties, enabling binary-search lookups.
#[derive(Encode, Decode, Clone, Debug)]
pub struct UIntIndex {
    pub(crate) sorted_values: Vec<u32>,
    pub(crate) sorted_oid: Vec<u32>,
}

/// Sorted-array index for `i32` properties.
#[derive(Encode, Decode, Clone, Debug)]
pub struct IntIndex {
    pub(crate) sorted_values: Vec<i32>,
    pub(crate) sorted_oid: Vec<u32>,
}

/// Two-list index for boolean properties, partitioning object IDs by value.
#[derive(Encode, Decode, Clone, Debug)]
pub struct BoolIndex {
    pub(crate) false_ids: Vec<u32>,
    pub(crate) true_ids: Vec<u32>,
}

/// Hash-map index for string properties, mapping each distinct value to its object IDs.
#[derive(Encode, Decode, Clone, Debug)]
pub struct StringIndex {
    pub(crate) map: HashMap<String, Vec<u32>>,
}

/// Bucketed index for enum properties, one bucket per variant.
#[derive(Encode, Decode, Clone, Debug)]
pub struct EnumIndex {
    pub(crate) buckets: Vec<Vec<u32>>,
}
