//! Per-frame spatial predicates over bounding boxes.
//!
//! [`UnaryCoord`] and [`UnaryCmp`] remain here because they are shared between
//! [`TrajExpr::AtomicAnalyticUnary`] (in `trql-algebra`) and the polynomial
//! root-finding in `trql-poly`. [`SpatialBody`] itself is now limited to the
//! two opaque-closure variants; all structured predicates should be expressed via
//! `trql-bbox-compiler` and compiled to `TrajExpr` analytic nodes.

use std::sync::Arc;

use trql_types::obj::BoundingBox;

/// Opaque per-frame spatial predicate evaluated on bounding boxes.
///
/// Only the two closure variants remain. Structured predicates (coordinate
/// comparisons, region containment, boolean combinations) should be expressed
/// with `trql-bbox-compiler` types and compiled analytically to `TrajExpr`
/// nodes, which avoids per-frame sampling entirely.
#[derive(Clone)]
pub enum SpatialBody {
    /// Opaque unary predicate — evaluated per-frame on a single bounding box.
    UnaryAtomic(Arc<dyn Fn(&BoundingBox) -> bool + Send + Sync>),
    /// Opaque binary predicate — evaluated per-frame on two bounding boxes.
    BinaryAtomic(Arc<dyn Fn(&BoundingBox, &BoundingBox) -> bool + Send + Sync>),
    And(Box<SpatialBody>, Box<SpatialBody>),
    Or(Box<SpatialBody>, Box<SpatialBody>),
    Not(Box<SpatialBody>),
    True,
    False,
}

impl std::fmt::Debug for SpatialBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnaryAtomic(_) => f.debug_tuple("UnaryAtomic").finish(),
            Self::BinaryAtomic(_) => f.debug_tuple("BinaryAtomic").finish(),
            Self::And(arg0, arg1) => f.debug_tuple("And").field(arg0).field(arg1).finish(),
            Self::Or(arg0, arg1) => f.debug_tuple("Or").field(arg0).field(arg1).finish(),
            Self::Not(arg0) => f.debug_tuple("Not").field(arg0).finish(),
            Self::True => write!(f, "True"),
            Self::False => write!(f, "False"),
        }
    }
}

impl SpatialBody {
    /// Evaluates this predicate on a slice of bounding boxes.
    pub fn eval(&self, bbox: &[&BoundingBox]) -> bool {
        match self {
            SpatialBody::UnaryAtomic(fun) => (fun)(bbox[0]),
            SpatialBody::BinaryAtomic(fun) => (fun)(bbox[0], bbox[1]),
            SpatialBody::And(a, b) => a.eval(bbox) && b.eval(bbox),
            SpatialBody::Or(a, b) => a.eval(bbox) || b.eval(bbox),
            SpatialBody::Not(a) => !a.eval(bbox),
            SpatialBody::True => true,
            SpatialBody::False => false,
        }
    }

    /// Returns `true` if this predicate requires two bounding boxes.
    pub fn is_binary(&self) -> bool {
        matches!(self, SpatialBody::BinaryAtomic(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bbox(x1: u32, y1: u32, x2: u32, y2: u32) -> BoundingBox {
        BoundingBox { x1, y1, x2, y2 }
    }

    // --- UnaryCoord ---

    // #[test]
    // fn unary_coord_extracts_values() {
    //     let b = bbox(10, 20, 30, 40);
    //     assert_eq!(UnaryCoord::X1.eval(&b), 10);
    //     assert_eq!(UnaryCoord::Y1.eval(&b), 20);
    //     assert_eq!(UnaryCoord::X2.eval(&b), 30);
    //     assert_eq!(UnaryCoord::Y2.eval(&b), 40);
    //     assert_eq!(UnaryCoord::CenterX.eval(&b), 20);
    //     assert_eq!(UnaryCoord::CenterY.eval(&b), 30);
    // }

    // // --- UnaryCmp ---

    // #[test]
    // fn unary_cmp_gt() {
    //     assert!(UnaryCmp::Gt(10).eval(11));
    //     assert!(!UnaryCmp::Gt(10).eval(10));
    //     assert!(!UnaryCmp::Gt(10).eval(9));
    // }

    // #[test]
    // fn unary_cmp_gte() {
    //     assert!(UnaryCmp::Gte(10).eval(11));
    //     assert!(UnaryCmp::Gte(10).eval(10));
    //     assert!(!UnaryCmp::Gte(10).eval(9));
    // }

    // #[test]
    // fn unary_cmp_lt() {
    //     assert!(UnaryCmp::Lt(10).eval(9));
    //     assert!(!UnaryCmp::Lt(10).eval(10));
    //     assert!(!UnaryCmp::Lt(10).eval(11));
    // }

    // #[test]
    // fn unary_cmp_lte() {
    //     assert!(UnaryCmp::Lte(10).eval(9));
    //     assert!(UnaryCmp::Lte(10).eval(10));
    //     assert!(!UnaryCmp::Lte(10).eval(11));
    // }

    // --- SpatialBody ---

    #[test]
    fn spatial_unary_atomic() {
        let pred = SpatialBody::UnaryAtomic(Arc::new(|b| b.x2 - b.x1 > 50));
        let wide = bbox(0, 0, 100, 10);
        assert!(pred.eval(&[&wide]));
        let narrow = bbox(0, 0, 20, 10);
        assert!(!pred.eval(&[&narrow]));
    }

    #[test]
    fn spatial_binary_atomic() {
        let pred = SpatialBody::BinaryAtomic(Arc::new(|a, b| a.x1 < b.x1));
        let a = bbox(10, 0, 20, 20);
        let b = bbox(30, 0, 40, 40);
        assert!(pred.eval(&[&a, &b]));
        assert!(!pred.eval(&[&b, &a]));
    }

    #[test]
    fn is_binary() {
        assert!(!SpatialBody::UnaryAtomic(Arc::new(|_| true)).is_binary());
        assert!(SpatialBody::BinaryAtomic(Arc::new(|_, _| true)).is_binary());
    }

    // --- New CLEVRER binary predicates ---
    //
    // Layout used in tests:
    //   left_obj:  x1=10, y1=50, x2=30, y2=70  → center (20, 60)
    //   right_obj: x1=60, y1=50, x2=80, y2=70  → center (70, 60)
    //   top_obj:   x1=40, y1=10, x2=60, y2=30  → center (50, 20)
    //   bot_obj:   x1=40, y1=70, x2=60, y2=90  → center (50, 80)
}
