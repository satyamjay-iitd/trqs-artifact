//! Polygonal regions, shared by every trajectory-degree module.
//!
//! Deliberately degree-independent: the geometry here is about the *region*, not about the
//! curve being tested against it, so `linear`, `quadratic`, and `cubic` all use this one
//! copy rather than each carrying their own.
//!
//! # Why convexity is worth detecting
//!
//! A trajectory coordinate is a polynomial in the frame, so any affine combination of
//! `x1, y1, x2, y2` is too — that is exactly what [`BBoxExpr`](crate::BBoxExpr) represents.
//! An edge `p → q` carries the implicit line `n·(P − p) = 0` with `n = (−(q_y − p_y), q_x − p_x)`,
//! so *substituting a trajectory point into an edge produces one more polynomial of the same
//! degree*. For a convex polygon, "inside" is the conjunction of those half-planes, which
//! means containment compiles to predicates the existing evaluator already solves — no new
//! machinery, and the optimizer and SQL backend keep seeing shapes they understand.
//!
//! A non-convex polygon has no such conjunctive form and needs the `InPolygon` node, which
//! collects the sign-change frames of every edge polynomial and settles each resulting
//! sub-range with a single [`Polygon::contains`] test.
//!
//! # Whole-box containment
//!
//! Testing that an entire bounding box is inside a region is corner-testing only while the
//! region is convex. A concave region can hold all four corners of a box whose middle
//! crosses a notch, so containment there needs a second condition: no edge of the region may
//! pass through the box's interior. That one stays affine — see
//! [`corner_extreme_weights`] — so whole-box containment in a concave region is still a
//! boolean combination of coordinate comparisons plus one point test per corner.

/// A closed simple polygon, stored counter-clockwise.
///
/// "Counter-clockwise" means positive signed area in whatever coordinate system the caller
/// is using. Callers need not normalize; [`Polygon::new`] reverses a clockwise vertex list
/// so that the interior is always to the *left* of each directed edge, which is what makes
/// [`Polygon::half_planes`] uniformly `≥ 0` on the inside.
#[derive(Clone, Debug, PartialEq)]
pub struct Polygon {
    verts: Vec<[f64; 2]>,
}

/// Why a vertex list could not become a [`Polygon`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PolygonError {
    /// Fewer than three distinct vertices survived duplicate removal.
    #[error("a polygon needs at least 3 distinct vertices")]
    TooFewVertices,
    /// All vertices are collinear, so the polygon encloses nothing.
    #[error("polygon has zero area — all vertices are collinear")]
    Degenerate,
}

impl Polygon {
    /// Builds a polygon from a vertex ring, normalizing it to counter-clockwise order.
    ///
    /// The ring is implicitly closed: do not repeat the first vertex at the end (one that
    /// is repeated is dropped as a duplicate, so both spellings work). Consecutive
    /// duplicates are removed because they contribute a zero-length edge, whose normal is
    /// `(0, 0)` — a degenerate half-plane that would admit every point.
    pub fn new(verts: impl IntoIterator<Item = [f64; 2]>) -> Result<Self, PolygonError> {
        let mut verts: Vec<[f64; 2]> = verts.into_iter().collect();
        verts.dedup();
        // The ring wraps, so the last vertex is also "adjacent" to the first.
        while verts.len() > 1 && verts[0] == verts[verts.len() - 1] {
            verts.pop();
        }
        if verts.len() < 3 {
            return Err(PolygonError::TooFewVertices);
        }

        let area2 = signed_area2(&verts);
        if area2.abs() < 1e-12 {
            return Err(PolygonError::Degenerate);
        }
        if area2 < 0.0 {
            verts.reverse();
        }
        Ok(Polygon { verts })
    }

    /// The polygon equivalent of an axis-aligned `[x1, y1, x2, y2]` region.
    ///
    /// Used to give `INSIDE` a single code path; the rectangle fast path in the DSL is kept
    /// only because four comparisons beat four half-planes on readability, not correctness.
    pub fn from_rect([x1, y1, x2, y2]: [f64; 4]) -> Result<Self, PolygonError> {
        Self::new([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    }

    pub fn verts(&self) -> &[[f64; 2]] {
        &self.verts
    }

    /// The directed edges of the ring, `(p, q)`, wrapping from the last vertex to the first.
    pub fn edges(&self) -> impl Iterator<Item = ([f64; 2], [f64; 2])> + '_ {
        let n = self.verts.len();
        (0..n).map(move |i| (self.verts[i], self.verts[(i + 1) % n]))
    }

    /// The inward half-plane of each edge as `(n_x, n_y, d)`, meaning `n_x·x + n_y·y + d ≥ 0`
    /// holds exactly on the interior side.
    ///
    /// Only meaningful as a containment test when the polygon [`is_convex`](Self::is_convex);
    /// for a non-convex ring these are still the edge *lines*, which is what the `InPolygon`
    /// evaluator wants for its breakpoints.
    pub fn half_planes(&self) -> impl Iterator<Item = (f64, f64, f64)> + '_ {
        self.edges().map(|(p, q)| {
            let (nx, ny) = (-(q[1] - p[1]), q[0] - p[0]);
            (nx, ny, -(nx * p[0] + ny * p[1]))
        })
    }

    /// Whether every turn around the ring goes the same way.
    ///
    /// Collinear vertices (zero cross product) are tolerated — they do not break the
    /// half-plane conjunction, they just contribute a redundant constraint.
    pub fn is_convex(&self) -> bool {
        let n = self.verts.len();
        // Normalized CCW, so a convex ring never turns right.
        (0..n).all(|i| {
            let (a, b, c) = (
                self.verts[i],
                self.verts[(i + 1) % n],
                self.verts[(i + 2) % n],
            );
            cross(a, b, c) >= -1e-9
        })
    }

    /// Point-in-polygon by ray crossing, valid for convex and non-convex rings alike.
    ///
    /// Casts a ray in `+x` and counts edge crossings. The half-open `(p_y > y) != (q_y > y)`
    /// test counts a vertex exactly once, so a ray passing through one does not flip parity
    /// twice; horizontal edges are skipped by the same test rather than dividing by zero.
    pub fn contains(&self, x: f64, y: f64) -> bool {
        let mut inside = false;
        for (p, q) in self.edges() {
            if (p[1] > y) != (q[1] > y) && x < (q[0] - p[0]) * (y - p[1]) / (q[1] - p[1]) + p[0] {
                inside = !inside;
            }
        }
        inside
    }

    /// Rewrites every vertex through `f`, re-normalizing orientation.
    ///
    /// Re-normalizing is the point: a coordinate-frame change that flips an axis — which is
    /// exactly what the server's world frame does to Y — reverses the winding, and a
    /// silently clockwise ring would invert every half-plane.
    pub fn map(&self, f: impl FnMut([f64; 2]) -> [f64; 2]) -> Result<Self, PolygonError> {
        Self::new(self.verts.iter().copied().map(f))
    }
}

/// The `[x1_w, y1_w, x2_w, y2_w]` weights selecting the box corner that extremizes `n·c`.
///
/// `maximize` picks the largest such corner, otherwise the smallest. The result is meant to
/// be read as an affine expression over the box coordinates: `x1_w·x1 + … + y2_w·y2`.
///
/// The whole point is that no runtime choice is involved. Which corner is extremal along an
/// axis depends only on the *signs* of `n`, because the box invariants `x1 ≤ x2` and
/// `y1 ≤ y2` settle the rest — so a region can bake the choice in when it is compiled. That
/// is what keeps the separating-axis test between a box and a polygon edge affine, and
/// therefore expressible as an ordinary coordinate comparison rather than something the
/// evaluator would have to special-case per frame.
///
/// A zero component contributes no term, which is correct: an axis the normal does not span
/// cannot separate anything along it.
pub fn corner_extreme_weights(nx: f64, ny: f64, maximize: bool) -> [f64; 4] {
    let (x1_w, x2_w) = if (nx > 0.0) == maximize {
        (0.0, nx)
    } else {
        (nx, 0.0)
    };
    let (y1_w, y2_w) = if (ny > 0.0) == maximize {
        (0.0, ny)
    } else {
        (ny, 0.0)
    };
    [x1_w, y1_w, x2_w, y2_w]
}

/// Twice the signed area of the ring (the shoelace sum); positive when counter-clockwise.
fn signed_area2(verts: &[[f64; 2]]) -> f64 {
    let n = verts.len();
    (0..n)
        .map(|i| {
            let (p, q) = (verts[i], verts[(i + 1) % n]);
            p[0] * q[1] - q[0] * p[1]
        })
        .sum()
}

/// Z component of `(b − a) × (c − b)`: positive when the turn at `b` is to the left.
fn cross(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> Polygon {
        Polygon::new([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]).unwrap()
    }

    /// An L, traversed so that the reflex corner at (5,5) turns the other way.
    fn ell() -> Polygon {
        Polygon::new([
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 5.0],
            [5.0, 5.0],
            [5.0, 10.0],
            [0.0, 10.0],
        ])
        .unwrap()
    }

    #[test]
    fn rejects_degenerate_rings() {
        assert_eq!(
            Polygon::new([[0.0, 0.0], [1.0, 1.0]]),
            Err(PolygonError::TooFewVertices)
        );
        // Repeating the closing vertex must not be what makes a triangle "valid".
        assert_eq!(
            Polygon::new([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]),
            Err(PolygonError::TooFewVertices)
        );
        assert_eq!(
            Polygon::new([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
            Err(PolygonError::Degenerate)
        );
    }

    /// A closed ring written with the first vertex repeated at the end is the same polygon,
    /// not a four-vertex one with a zero-length edge whose normal admits everything.
    #[test]
    fn closing_vertex_is_dropped() {
        let open = Polygon::new([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]).unwrap();
        let closed = Polygon::new([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [0.0, 0.0]]).unwrap();
        assert_eq!(open, closed);
    }

    /// Orientation is normalized, so a clockwise vertex list and its reverse describe the
    /// same region — otherwise every half-plane would come out inverted for one of them.
    #[test]
    fn winding_is_normalized() {
        let ccw = Polygon::new([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]).unwrap();
        let cw = Polygon::new([[0.0, 10.0], [10.0, 10.0], [10.0, 0.0], [0.0, 0.0]]).unwrap();
        assert_eq!(ccw, cw);
    }

    #[test]
    fn convexity() {
        assert!(square().is_convex());
        assert!(!ell().is_convex());
        // A collinear midpoint is redundant, not a violation of convexity.
        let with_midpoint = Polygon::new([
            [0.0, 0.0],
            [5.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ])
        .unwrap();
        assert!(with_midpoint.is_convex());
    }

    /// The conjunction of the half-planes must be exactly containment, for *both* windings:
    /// every one holds inside, and at least one fails outside. Only the first half would
    /// still pass if the normals pointed the wrong way for a clockwise ring.
    #[test]
    fn half_planes_conjunction_is_containment() {
        for poly in [
            square(),
            Polygon::new([[0.0, 10.0], [10.0, 10.0], [10.0, 0.0], [0.0, 0.0]]).unwrap(),
        ] {
            for (x, y) in [(5.0, 5.0), (1.0, 9.0), (9.0, 1.0)] {
                assert!(
                    poly.half_planes()
                        .all(|(nx, ny, d)| nx * x + ny * y + d > 0.0),
                    "({x},{y}) should be inside"
                );
            }
            for (x, y) in [
                (-1.0, 5.0),
                (5.0, -1.0),
                (11.0, 5.0),
                (5.0, 11.0),
                (-1.0, -1.0),
            ] {
                assert!(
                    poly.half_planes()
                        .any(|(nx, ny, d)| nx * x + ny * y + d < 0.0),
                    "({x},{y}) should be excluded by some edge"
                );
            }
        }
    }

    #[test]
    fn contains_handles_the_reflex_corner() {
        let l = ell();
        assert!(l.contains(2.0, 2.0));
        assert!(l.contains(8.0, 2.0));
        assert!(l.contains(2.0, 8.0));
        // The notch: inside the bounding box, outside the polygon.
        assert!(!l.contains(8.0, 8.0));
        assert!(!l.contains(-1.0, 5.0));
        assert!(!l.contains(11.0, 5.0));
    }

    /// A ray leaving a vertex must flip parity once, not twice. The `(p_y > y) != (q_y > y)`
    /// half-open test is what buys that; a naive `>=` on both ends double-counts and reports
    /// points inside a diamond as outside.
    #[test]
    fn contains_counts_vertices_once() {
        let diamond = Polygon::new([[0.0, 5.0], [5.0, 0.0], [10.0, 5.0], [5.0, 10.0]]).unwrap();
        assert!(diamond.contains(5.0, 5.0));
        // y = 5 passes exactly through the left and right vertices.
        assert!(diamond.contains(4.0, 5.0));
        assert!(!diamond.contains(-1.0, 5.0));
        assert!(!diamond.contains(11.0, 5.0));
        // y = 0 grazes the bottom vertex; nothing at that height is interior.
        assert!(!diamond.contains(5.0, 0.0));
    }

    /// The selected corner must really be the extreme one, for every sign combination —
    /// a wrong pick would silently weaken the separating-axis test into one that reports
    /// "disjoint" for edges that do cut through the box.
    #[test]
    fn corner_extreme_weights_picks_the_extreme_corner() {
        let boxes = [[0.0, 0.0, 4.0, 3.0], [-5.0, -2.0, -1.0, 7.0]];
        let normals = [
            (1.0, 2.0),
            (-1.0, 2.0),
            (1.0, -2.0),
            (-1.0, -2.0),
            (0.0, 1.0),
            (1.0, 0.0),
        ];
        for [x1, y1, x2, y2] in boxes {
            let corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]];
            for (nx, ny) in normals {
                let projections: Vec<f64> = corners.iter().map(|c| nx * c[0] + ny * c[1]).collect();
                for (maximize, want) in [
                    (true, projections.iter().cloned().fold(f64::MIN, f64::max)),
                    (false, projections.iter().cloned().fold(f64::MAX, f64::min)),
                ] {
                    let [a, b, c, d] = corner_extreme_weights(nx, ny, maximize);
                    let got = a * x1 + b * y1 + c * x2 + d * y2;
                    assert_eq!(got, want, "n=({nx},{ny}) maximize={maximize}");
                }
            }
        }
    }

    /// Mirroring Y — the server's world frame — must not turn the interior inside out.
    #[test]
    fn map_renormalizes_winding_after_a_flip() {
        let flipped = square().map(|[x, y]| [x, -y]).unwrap();
        assert!(flipped.contains(5.0, -5.0));
        assert!(!flipped.contains(5.0, 5.0));
        for (nx, ny, d) in flipped.half_planes() {
            assert!(nx * 5.0 + ny * -5.0 + d > 0.0);
        }
    }
}
