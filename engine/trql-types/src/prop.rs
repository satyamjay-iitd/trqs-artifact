//! Property schema and typed value definitions.

use bincode::{Decode, Encode};
use smallvec::SmallVec;

/// A typed property value.
#[derive(Debug, Clone, Encode, Decode, PartialEq, Eq)]
pub enum Value {
    UInt(u32),
    Int(i32),
    Bool(bool),
    String(String),
    /// Index into the enum variant list defined by [`TypeDesc::Enum`].
    Enum(usize),
}

/// Describes the type of a property column.
#[derive(Debug, Clone, Encode, Decode, PartialEq, Eq)]
pub enum TypeDesc {
    UInt,
    Int,
    Bool,
    String,
    /// Closed set of named variants.
    Enum(Vec<String>),
}

impl std::fmt::Display for TypeDesc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeDesc::UInt => write!(f, "u32"),
            TypeDesc::Int => write!(f, "i32"),
            TypeDesc::Bool => write!(f, "bool"),
            TypeDesc::String => write!(f, "str"),
            TypeDesc::Enum(variants) => {
                write!(f, "enum {{ ")?;
                for (i, variant) in variants.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{variant}")?;
                }
                write!(f, " }}")
            }
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::UInt(v) => write!(f, "{v}"),
            Value::Int(v) => write!(f, "{v}"),
            Value::Bool(v) => write!(f, "{v}"),
            Value::String(v) => write!(f, "\"{v}\""),
            Value::Enum(idx) => write!(f, "Variant({idx})"),
        }
    }
}

impl TypeDesc {
    /// Parses a string into a [`Value`] according to this type descriptor.
    pub fn parse_value(&self, input: &str) -> Result<Value, ValConvertError> {
        match self {
            TypeDesc::UInt => {
                input
                    .parse::<u32>()
                    .map(Value::UInt)
                    .map_err(|_| ValConvertError::InvalidValue {
                        expected: "u64",
                        found: input.to_string(),
                    })
            }
            TypeDesc::Int => {
                input
                    .parse::<i32>()
                    .map(Value::Int)
                    .map_err(|_| ValConvertError::InvalidValue {
                        expected: "i64",
                        found: input.to_string(),
                    })
            }
            TypeDesc::Bool => {
                input
                    .parse::<bool>()
                    .map(Value::Bool)
                    .map_err(|_| ValConvertError::InvalidValue {
                        expected: "bool",
                        found: input.to_string(),
                    })
            }
            TypeDesc::String => Ok(Value::String(input.to_owned())),
            TypeDesc::Enum(variants) => variants
                .iter()
                .position(|v| v == input)
                .map(Value::Enum)
                .ok_or_else(|| ValConvertError::InvalidValue {
                    expected: "enum variant",
                    found: input.to_string(),
                }),
        }
    }

    /// Returns `true` if the value's runtime type matches this descriptor.
    pub fn matches(&self, v: &Value) -> bool {
        matches!(
            (self, v),
            (TypeDesc::UInt, Value::UInt(_))
                | (TypeDesc::Int, Value::Int(_))
                | (TypeDesc::Bool, Value::Bool(_))
                | (TypeDesc::String, Value::String(_))
                | (TypeDesc::Enum { .. }, Value::Enum(_))
        )
    }
}

/// Error converting a raw value into a typed [`Value`].
#[derive(Debug, thiserror::Error)]
pub enum ValConvertError {
    #[error("Unknown Properties {props:?}")]
    UnknownProperties { props: Vec<String> },
    #[error("Invalid Value expected: {expected:?} found: {found:?}")]
    InvalidValue {
        expected: &'static str,
        found: String,
    },
    #[error("Duplicate fid found: An object can only be present once in a frame")]
    DupicateFid { duplicate: usize },
    #[error("Required property {prop:?} not found.")]
    RequiredPropNotFound { prop: String },
}

/// Error validating values against a [`Schema`].
#[derive(Debug, thiserror::Error)]
pub enum SchemaErr {
    #[error("Arity Mismatch (expected {expected_cols:?}, got {found_cols:?} cols)")]
    ArityMismatch {
        expected_cols: Vec<String>,
        found_cols: usize,
    },

    #[error("Type Mismatch for col: {col:?} (expected {expected:?}, got {found:?})")]
    TypeMismatch {
        col: String,
        expected: TypeDesc,
        found: Value,
    },
    #[error("Invalid Enum for col: {col:?} (max_value {max:?}, got {value:?})")]
    EnumOutOfRange {
        col: String,
        value: usize,
        max: usize,
    },
    #[error("Empty Schema")]
    Empty,
}

/// Ordered property schema mapping column names to types.
#[derive(Debug, Clone, Encode, Decode)]
pub struct Schema {
    cols: Vec<(String, TypeDesc)>,
    #[bincode(with_serde)]
    name_to_idx: SmallVec<[String; 10]>,
}

impl Schema {
    /// Creates a new schema from `(name, type)` pairs.
    pub fn new(props: Vec<(String, TypeDesc)>) -> Self {
        let name_to_idx = props.iter().map(|(col, _)| col.to_string()).collect();
        Schema {
            cols: props,
            name_to_idx,
        }
    }

    /// Returns the type descriptor for the named column, if it exists.
    pub fn get_type(&self, name: &str) -> Option<&TypeDesc> {
        Some(&self.cols.iter().find(|(col, _)| col == name)?.1)
    }

    /// Iterates over `(name, type)` pairs in column order.
    pub fn col_iter(&self) -> impl Iterator<Item = &(String, TypeDesc)> {
        self.cols.iter()
    }

    /// Returns the variant index for an enum value in the given column.
    pub fn get_enum_id(&self, col: &str, val: &str) -> Option<usize> {
        match self.get_type(col)? {
            TypeDesc::Enum(variants) => variants.iter().position(|x| x == val),
            _ => None,
        }
    }

    /// Returns the positional index of a column by name.
    pub fn get_col_idx(&self, col_name: &str) -> Option<usize> {
        self.name_to_idx.iter().position(|x| x == col_name)
    }

    /// Returns the number of columns.
    pub fn num_cols(&self) -> usize {
        self.cols.len()
    }
}

impl std::fmt::Display for Schema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "DOMAIN {{")?;
        for (name, ty) in &self.cols {
            writeln!(f, "  {}: {},", name, ty)?;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- TypeDesc::parse_value ---

    #[test]
    fn parse_uint() {
        assert_eq!(TypeDesc::UInt.parse_value("42").unwrap(), Value::UInt(42));
    }

    #[test]
    fn parse_uint_invalid() {
        assert!(TypeDesc::UInt.parse_value("abc").is_err());
    }

    #[test]
    fn parse_int() {
        assert_eq!(TypeDesc::Int.parse_value("-7").unwrap(), Value::Int(-7));
    }

    #[test]
    fn parse_bool() {
        assert_eq!(
            TypeDesc::Bool.parse_value("true").unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            TypeDesc::Bool.parse_value("false").unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn parse_bool_invalid() {
        assert!(TypeDesc::Bool.parse_value("yes").is_err());
    }

    #[test]
    fn parse_string() {
        assert_eq!(
            TypeDesc::String.parse_value("hello").unwrap(),
            Value::String("hello".to_string())
        );
    }

    #[test]
    fn parse_enum_valid() {
        let td = TypeDesc::Enum(vec!["car".into(), "bus".into(), "truck".into()]);
        assert_eq!(td.parse_value("bus").unwrap(), Value::Enum(1));
    }

    #[test]
    fn parse_enum_invalid() {
        let td = TypeDesc::Enum(vec!["car".into(), "bus".into()]);
        assert!(td.parse_value("train").is_err());
    }

    // --- TypeDesc::matches ---

    #[test]
    fn matches_correct_types() {
        assert!(TypeDesc::UInt.matches(&Value::UInt(0)));
        assert!(TypeDesc::Int.matches(&Value::Int(0)));
        assert!(TypeDesc::Bool.matches(&Value::Bool(true)));
        assert!(TypeDesc::String.matches(&Value::String("x".into())));
        assert!(TypeDesc::Enum(vec![]).matches(&Value::Enum(0)));
    }

    #[test]
    fn matches_wrong_types() {
        assert!(!TypeDesc::UInt.matches(&Value::Int(0)));
        assert!(!TypeDesc::Bool.matches(&Value::UInt(1)));
        assert!(!TypeDesc::String.matches(&Value::Bool(false)));
    }

    // --- Schema ---

    fn test_schema() -> Schema {
        Schema::new(vec![
            ("speed".into(), TypeDesc::UInt),
            (
                "color".into(),
                TypeDesc::Enum(vec!["red".into(), "blue".into()]),
            ),
            ("active".into(), TypeDesc::Bool),
        ])
    }

    #[test]
    fn schema_num_cols() {
        assert_eq!(test_schema().num_cols(), 3);
    }

    #[test]
    fn schema_get_type() {
        let s = test_schema();
        assert_eq!(s.get_type("speed"), Some(&TypeDesc::UInt));
        assert_eq!(s.get_type("active"), Some(&TypeDesc::Bool));
        assert_eq!(s.get_type("nonexistent"), None);
    }

    #[test]
    fn schema_get_col_idx() {
        let s = test_schema();
        assert_eq!(s.get_col_idx("speed"), Some(0));
        assert_eq!(s.get_col_idx("color"), Some(1));
        assert_eq!(s.get_col_idx("active"), Some(2));
        assert_eq!(s.get_col_idx("missing"), None);
    }

    #[test]
    fn schema_get_enum_id() {
        let s = test_schema();
        assert_eq!(s.get_enum_id("color", "red"), Some(0));
        assert_eq!(s.get_enum_id("color", "blue"), Some(1));
        assert_eq!(s.get_enum_id("color", "green"), None);
        assert_eq!(s.get_enum_id("speed", "red"), None); // not an enum column
    }

    #[test]
    fn schema_col_iter() {
        let s = test_schema();
        let names: Vec<&str> = s.col_iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["speed", "color", "active"]);
    }
}
