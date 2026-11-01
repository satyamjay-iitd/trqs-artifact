//! Property column storage with typed indexes and predicate evaluation.

use anyhow::anyhow;
use bincode::{Decode, Encode};
use trql_types::{
    ObjId,
    prop::{Schema, TypeDesc, Value},
};

use crate::index::{BoolIndex, EnumIndex, IntIndex, PropIndex, StringIndex, UIntIndex};

/// A single property column: values for every object plus an optional index.
#[derive(Encode, Decode, Clone, Debug)]
pub struct Prop {
    name: String,
    values: Vec<Value>,
    ty: TypeDesc,
    index: Option<PropIndex>,
}

impl Prop {
    /// Creates an empty column with the given name and type.
    pub fn empty(name: &str, ty: TypeDesc) -> Prop {
        Prop {
            name: name.to_string(),
            values: vec![],
            ty,
            index: None,
        }
    }

    /// Returns the value at the given object index.
    pub fn get(&self, index: ObjId) -> &Value {
        &self.values[index as usize]
    }

    /// Returns the column name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the column's type descriptor.
    pub fn type_(&self) -> &TypeDesc {
        &self.ty
    }

    /// Builds a type-specific index for fast predicate lookups.
    pub fn build_index(&mut self) {
        let index = match &mut self.ty {
            TypeDesc::UInt => {
                let mut sorted: Vec<(u32, ObjId)> = Vec::new();
                for (i, v) in self.values.iter().enumerate() {
                    match v {
                        Value::UInt(x) => sorted.push((*x, i as ObjId)),
                        _ => unreachable!(),
                    }
                }
                sorted.sort_unstable_by_key(|(x, _)| *x);
                let (values, oids): (Vec<u32>, Vec<ObjId>) = sorted.into_iter().unzip();
                PropIndex::UInt(UIntIndex {
                    sorted_values: values,
                    sorted_oid: oids,
                })
            }
            TypeDesc::Int => {
                let mut sorted: Vec<(i32, ObjId)> = Vec::new();
                for (i, v) in self.values.iter().enumerate() {
                    match v {
                        Value::Int(x) => sorted.push((*x, i as ObjId)),
                        _ => unreachable!(),
                    }
                }
                sorted.sort_unstable_by_key(|(x, _)| *x);
                let (values, oids): (Vec<i32>, Vec<ObjId>) = sorted.into_iter().unzip();
                PropIndex::Int(IntIndex {
                    sorted_values: values,
                    sorted_oid: oids,
                })
            }
            TypeDesc::Bool => {
                let mut f = Vec::new();
                let mut t = Vec::new();
                for (i, v) in self.values.iter().enumerate() {
                    match v {
                        Value::Bool(false) => f.push(i as ObjId),
                        Value::Bool(true) => t.push(i as ObjId),
                        _ => unreachable!(),
                    }
                }
                PropIndex::Bool(BoolIndex {
                    false_ids: f,
                    true_ids: t,
                })
            }
            TypeDesc::Enum(variants) => {
                let mut buckets = vec![Vec::new(); variants.len()];
                for (row_id, v) in self.values.iter().enumerate() {
                    match v {
                        Value::Enum(variant_id) => buckets[*variant_id].push(row_id as ObjId),
                        _ => unreachable!(),
                    }
                }
                PropIndex::Enum(EnumIndex { buckets })
            }
            TypeDesc::String => {
                let mut map: std::collections::HashMap<String, Vec<ObjId>> =
                    std::collections::HashMap::new();
                for (i, v) in self.values.iter().enumerate() {
                    match v {
                        Value::String(s) => map.entry(s.clone()).or_default().push(i as ObjId),
                        _ => unreachable!(),
                    }
                }
                PropIndex::String(StringIndex { map })
            }
        };
        self.index = Some(index);
    }

    /// Appends a value to this column (must match the column type).
    pub fn append(&mut self, v: Value) {
        debug_assert!(self.ty.matches(&v));
        self.values.push(v);
    }

    /// Returns object IDs matching `pred` via the index. Requires [`build_index`](Prop::build_index) first.
    pub fn index_scan<'a>(&'a self, pred: AttrPred) -> anyhow::Result<&'a [ObjId]> {
        let index = self
            .index
            .as_ref()
            .ok_or_else(|| anyhow!("Index doesn't exist on this column, build_index first"))?;
        match (&self.ty, index, pred) {
            (TypeDesc::UInt, PropIndex::UInt(idx), AttrPred::UIntPred(p)) => {
                let s = &idx.sorted_values;
                let oids = &idx.sorted_oid;
                match p {
                    UIntPred::Eq(v) => {
                        let lo = s.partition_point(|x| *x < v);
                        let hi = s.partition_point(|x| *x <= v);
                        Ok(&oids[lo..hi])
                    }
                    UIntPred::Lt(v) => Ok(&oids[..s.partition_point(|x| *x < v)]),
                    UIntPred::Lte(v) => Ok(&oids[..s.partition_point(|x| *x <= v)]),
                    UIntPred::Gt(v) => Ok(&oids[s.partition_point(|x| *x <= v)..]),
                    UIntPred::Gte(v) => Ok(&oids[s.partition_point(|x| *x < v)..]),
                }
            }
            (TypeDesc::Int, PropIndex::Int(idx), AttrPred::IntPred(p)) => {
                let s = &idx.sorted_values;
                let oids = &idx.sorted_oid;
                match p {
                    IntPred::Eq(v) => {
                        let lo = s.partition_point(|x| *x < v);
                        let hi = s.partition_point(|x| *x <= v);
                        Ok(&oids[lo..hi])
                    }
                    IntPred::Lt(v) => Ok(&oids[..s.partition_point(|x| *x < v)]),
                    IntPred::Lte(v) => Ok(&oids[..s.partition_point(|x| *x <= v)]),
                    IntPred::Gt(v) => Ok(&oids[s.partition_point(|x| *x <= v)..]),
                    IntPred::Gte(v) => Ok(&oids[s.partition_point(|x| *x < v)..]),
                }
            }
            (TypeDesc::Bool, PropIndex::Bool(idx), AttrPred::BoolPred(v)) => {
                Ok(if v { &idx.true_ids } else { &idx.false_ids })
            }
            (TypeDesc::Enum(variants), PropIndex::Enum(idx), AttrPred::EnumPred(v)) => {
                if v >= variants.len() {
                    anyhow::bail!("Enum value {} out of range (0..{})", v, variants.len());
                }
                Ok(&idx.buckets[v])
            }
            (TypeDesc::String, PropIndex::String(idx), AttrPred::StringPred(ref s)) => {
                Ok(idx.map.get(s).map(|v| v.as_slice()).unwrap_or(&[]))
            }
            _ => anyhow::bail!("Invalid predicate for this column type"),
        }
    }
}

/// Predicate for filtering objects by a property value.
#[derive(Clone)]
pub enum AttrPred {
    UIntPred(UIntPred),
    IntPred(IntPred),
    BoolPred(bool),
    EnumPred(usize),
    /// Equality predicate for string properties.
    StringPred(String),
}

/// Comparison predicate for `i32` properties.
#[derive(Clone)]
pub enum IntPred {
    Eq(i32),
    Lt(i32),
    Gt(i32),
    Lte(i32),
    Gte(i32),
}

/// Comparison predicate for `u32` properties.
#[derive(Clone)]
pub enum UIntPred {
    Eq(u32),
    Lt(u32),
    Gt(u32),
    Lte(u32),
    Gte(u32),
}

/// Validates that `props` matches the schema in arity and types.
pub fn validate_props(schema: &Schema, props: &[Value]) -> anyhow::Result<()> {
    if props.len() != schema.num_cols() {
        anyhow::bail!("arity mismatch");
    }

    for ((_, ty), val) in schema.col_iter().zip(props) {
        if !ty.matches(val) {
            anyhow::bail!("type mismatch");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_schema() -> Schema {
        Schema::new(vec![
            ("speed".into(), TypeDesc::UInt),
            (
                "label".into(),
                TypeDesc::Enum(vec!["car".into(), "bus".into(), "truck".into()]),
            ),
            ("active".into(), TypeDesc::Bool),
        ])
    }

    // --- validate_props ---

    #[test]
    fn validate_props_ok() {
        let s = test_schema();
        let props = vec![Value::UInt(50), Value::Enum(0), Value::Bool(true)];
        assert!(validate_props(&s, &props).is_ok());
    }

    #[test]
    fn validate_props_arity_mismatch() {
        let s = test_schema();
        assert!(validate_props(&s, &[Value::UInt(1)]).is_err());
    }

    #[test]
    fn validate_props_type_mismatch() {
        let s = test_schema();
        let props = vec![Value::Bool(true), Value::Enum(0), Value::Bool(true)]; // first should be UInt
        assert!(validate_props(&s, &props).is_err());
    }

    // --- Prop: build_index + index_scan ---

    #[test]
    fn uint_index_eq_scan() {
        let mut prop = Prop::empty("speed", TypeDesc::UInt);
        for v in [10, 20, 30, 20, 10] {
            prop.append(Value::UInt(v));
        }
        prop.build_index();

        let result = prop
            .index_scan(AttrPred::UIntPred(UIntPred::Eq(20)))
            .unwrap();
        let mut ids: Vec<ObjId> = result.to_vec();
        ids.sort();
        assert_eq!(ids, vec![1, 3]);
    }

    #[test]
    fn uint_index_gt_scan() {
        let mut prop = Prop::empty("speed", TypeDesc::UInt);
        for v in [10, 20, 30, 40, 50] {
            prop.append(Value::UInt(v));
        }
        prop.build_index();

        let result = prop
            .index_scan(AttrPred::UIntPred(UIntPred::Gt(25)))
            .unwrap();
        let mut ids: Vec<ObjId> = result.to_vec();
        ids.sort();
        assert_eq!(ids, vec![2, 3, 4]);
    }

    #[test]
    fn uint_index_lt_scan() {
        let mut prop = Prop::empty("speed", TypeDesc::UInt);
        for v in [10, 20, 30, 40, 50] {
            prop.append(Value::UInt(v));
        }
        prop.build_index();

        let result = prop
            .index_scan(AttrPred::UIntPred(UIntPred::Lt(25)))
            .unwrap();
        let mut ids: Vec<ObjId> = result.to_vec();
        ids.sort();
        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn int_index_eq_scan() {
        let mut prop = Prop::empty("temp", TypeDesc::Int);
        for v in [-10, 0, 10, 0, -10] {
            prop.append(Value::Int(v));
        }
        prop.build_index();

        let result = prop.index_scan(AttrPred::IntPred(IntPred::Eq(0))).unwrap();
        let mut ids: Vec<ObjId> = result.to_vec();
        ids.sort();
        assert_eq!(ids, vec![1, 3]);
    }

    #[test]
    fn bool_index_scan() {
        let mut prop = Prop::empty("active", TypeDesc::Bool);
        for v in [true, false, true, true, false] {
            prop.append(Value::Bool(v));
        }
        prop.build_index();

        let true_ids = prop.index_scan(AttrPred::BoolPred(true)).unwrap();
        let mut t: Vec<ObjId> = true_ids.to_vec();
        t.sort();
        assert_eq!(t, vec![0, 2, 3]);

        let false_ids = prop.index_scan(AttrPred::BoolPred(false)).unwrap();
        let mut f: Vec<ObjId> = false_ids.to_vec();
        f.sort();
        assert_eq!(f, vec![1, 4]);
    }

    #[test]
    fn enum_index_scan() {
        let variants = vec!["car".into(), "bus".into(), "truck".into()];
        let mut prop = Prop::empty("label", TypeDesc::Enum(variants));
        for v in [0usize, 1, 2, 0, 1] {
            prop.append(Value::Enum(v));
        }
        prop.build_index();

        let car_ids = prop.index_scan(AttrPred::EnumPred(0)).unwrap();
        let mut ids: Vec<ObjId> = car_ids.to_vec();
        ids.sort();
        assert_eq!(ids, vec![0, 3]);

        let truck_ids = prop.index_scan(AttrPred::EnumPred(2)).unwrap();
        assert_eq!(truck_ids, &[2]);
    }

    #[test]
    fn string_index_eq_scan() {
        let mut prop = Prop::empty("name", TypeDesc::String);
        for s in ["alice", "bob", "alice", "carol", "bob"] {
            prop.append(Value::String(s.into()));
        }
        prop.build_index();

        let alice_ids = prop
            .index_scan(AttrPred::StringPred("alice".into()))
            .unwrap();
        let mut ids: Vec<ObjId> = alice_ids.to_vec();
        ids.sort();
        assert_eq!(ids, vec![0, 2]);

        let carol_ids = prop
            .index_scan(AttrPred::StringPred("carol".into()))
            .unwrap();
        assert_eq!(carol_ids, &[3]);

        // Non-existent value returns empty
        let none = prop
            .index_scan(AttrPred::StringPred("dave".into()))
            .unwrap();
        assert!(none.is_empty());
    }

    #[test]
    fn index_scan_without_index_errors() {
        let mut prop = Prop::empty("speed", TypeDesc::UInt);
        prop.append(Value::UInt(10));
        // No build_index call
        assert!(
            prop.index_scan(AttrPred::UIntPred(UIntPred::Eq(10)))
                .is_err()
        );
    }

    // --- Prop accessors ---

    #[test]
    fn prop_name_and_type() {
        let prop = Prop::empty("speed", TypeDesc::UInt);
        assert_eq!(prop.name(), "speed");
        assert_eq!(prop.type_(), &TypeDesc::UInt);
    }

    #[test]
    fn prop_get() {
        let mut prop = Prop::empty("x", TypeDesc::Int);
        prop.append(Value::Int(42));
        prop.append(Value::Int(-7));
        assert_eq!(prop.get(0), &Value::Int(42));
        assert_eq!(prop.get(1), &Value::Int(-7));
    }
}
