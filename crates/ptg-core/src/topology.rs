//! Pluggable lateral-mesh topologies (§3.1.3 "The Lateral Mesh").
//!
//! The mesh dictates column interconnectivity (Ring, Torus, fully-connected,
//! Small-World). Topologies here are **pure graph** functions over an ordered
//! `N`-column id list; they know nothing about system prompts or inference.
//!
//! # Direction convention
//!
//! Every connection is expressed as `LateralConnection { listener_id,
//! source_id }`: the **listener** receives the **source**'s prediction. This
//! matches `CorticalMesh::establish_lateral_connection(from = listener, to =
//! source)` — `from` listens to `to` — and `lateral_context_for(listener)`
//! reads the sources in `adjacency_list[listener]`. Naming the fields
//! `listener`/`source` (rather than `from`/`to`) makes the data-flow direction
//! unambiguous and prevents the class of bug where an echo screen compares
//! against the wrong end of an edge.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

/// A single directed lateral edge. `listener_id` receives `source_id`'s
/// most recent prediction on every tick after the first (see
/// `CorticalMesh::lateral_context_for`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LateralConnection {
    /// The column that *receives* a prediction (the `from` endpoint of
    /// `establish_lateral_connection`).
    pub listener_id: String,
    /// The column whose prediction is injected (the `to` endpoint).
    pub source_id: String,
}

impl LateralConnection {
    /// Construct a directed edge `listener` ← `source`.
    #[must_use]
    pub fn new(listener_id: &str, source_id: &str) -> Self {
        Self {
            listener_id: listener_id.to_string(),
            source_id: source_id.to_string(),
        }
    }
}

/// Errors raised while materializing a topology against a column-id list.
#[derive(Debug, thiserror::Error)]
pub enum TopologyError {
    /// The id list is empty.
    #[error("topology requires a non-empty column id list")]
    EmptyNodeSet,
    /// An id appeared more than once in the ordered id list.
    #[error("duplicate column id in id list: {0}")]
    DuplicateNodeId(String),
    /// A connection references an id not present in the id list.
    #[error("connection references unknown column id: {0}")]
    UnknownNodeId(String),
    /// A connection would have a column listen to itself.
    #[error("self-edge forbidden (column listens to itself): {0}")]
    SelfEdge(String),
    /// A built-in generator emitted a duplicate edge (internal invariant).
    /// Should never happen for the shipped generators; surfaces loudly rather
    /// than being silently dropped so a future generator bug cannot mask itself.
    #[error("duplicate edge from generator: {listener} <- {from}")]
    DuplicateEdge { listener: String, from: String },
    /// The requested geometry does not match the supplied column count.
    #[error("torus geometry mismatch: {width}x{height} != {n} columns")]
    TorusGeometry {
        width: usize,
        height: usize,
        n: usize,
    },
    /// A torus needs both dimensions >= 3 for a true 4-neighbor wraparound.
    #[error("torus requires width >= 3 and height >= 3 (got {width}x{height})")]
    TorusTooSmall { width: usize, height: usize },
    /// A parameter was out of its valid range.
    #[error("{parameter} out of range: {reason}")]
    OutOfRange {
        parameter: &'static str,
        reason: &'static str,
    },
    /// The node count is too small for the requested topology.
    #[error("{topology} requires at least {min} columns (got {n})")]
    TooFewNodes {
        topology: &'static str,
        min: usize,
        n: usize,
    },
}

/// Declarative description of a lateral mesh topology. Materialized against an
/// ordered column-id list via [`TopologySpec::connections_for`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TopologySpec {
    /// A 1-D cycle. Each column listens to its predecessor; if `bidirectional`
    /// it also listens to its successor. Requires >= 2 columns.
    Ring {
        /// Listen to both neighbors (predecessor + successor), not just one.
        bidirectional: bool,
    },
    /// A 2-D toroidal grid (wraparound on both axes). Each column listens to
    /// its four cardinal neighbors. Requires `width * height` columns and both
    /// dimensions >= 3 so every column genuinely has four distinct neighbors.
    Torus2d {
        /// Grid width (number of columns per row).
        width: usize,
        /// Grid height (number of rows).
        height: usize,
    },
    /// Every column listens to every other column. `n*(n-1)` edges. Requires
    /// >= 2 columns.
    FullyConnected,
    /// A Watts-Strogatz small-world graph built from a directed ring lattice
    /// (each column listens to its `degree` nearest neighbors) with each edge
    /// rewired to a new source with probability `rewire_probability`.
    /// Deterministic given `seed`. Requires an even `degree` with
    /// `0 < degree < n`.
    ///
    /// # Degeneracy warning (benchmark integrity)
    ///
    /// The Watts-Strogatz model is only meaningful when `degree ≪ n`. As `degree`
    /// approaches `n` the free-target pool for rewiring shrinks to nothing and
    /// most rewires silently fall back to the lattice edge, so the result is a
    /// near-complete / ring-lattice graph *labeled* "small-world". In particular
    /// `degree == n - 1` (reachable only for odd `n`, since `degree` must be
    /// even) materializes to the **complete digraph**, identical to
    /// [`TopologySpec::FullyConnected`]. A topology-comparison benchmark MUST
    /// assert that the materialized edge sets of the topologies it compares are
    /// actually different before drawing any contrast.
    SmallWorld {
        /// Ring-lattice out-degree (number of nearest neighbors each column
        /// listens to). Must be even and strictly less than the column count.
        degree: usize,
        /// Probability an edge is rewired to a new source, in `[0, 1]`.
        rewire_probability: f64,
        /// Seed for the deterministic rewiring PRNG (splitmix64).
        seed: u64,
    },
    /// A caller-supplied edge list, validated for self-edges, duplicates, and
    /// unknown ids.
    Custom(Vec<LateralConnection>),
}

/// Deterministic splitmix64 PRNG — zero-dependency, reproducible. Seeded from a
/// `u64`; used only for Watts-Strogatz rewiring.
struct Prng {
    state: u64,
}

impl Prng {
    fn new(seed: u64) -> Self {
        // splitmix64 defines the increment; a zero seed is fine but we still
        // mix, so different seeds diverge from the first draw.
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform index in `[0, n)`. Requires `n > 0`.
    fn range(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform `f64` in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

impl TopologySpec {
    /// Materialize this topology against the ordered column-id list. Edges are
    /// returned as `listener → source` pairs. Output order is deterministic for
    /// every variant (positional for ring/torus/fully-connected;
    /// seeded-iteration-order for small-world).
    ///
    /// # Errors
    /// See [`TopologyError`] for the per-variant validation rules.
    pub fn connections_for(&self, ids: &[String]) -> Result<Vec<LateralConnection>, TopologyError> {
        if ids.is_empty() {
            return Err(TopologyError::EmptyNodeSet);
        }
        let mut seen: HashSet<&str> = HashSet::new();
        for id in ids {
            if !seen.insert(id.as_str()) {
                return Err(TopologyError::DuplicateNodeId(id.clone()));
            }
        }
        let n = ids.len();
        let edges = match self {
            Self::Ring { bidirectional } => ring_edges(n, *bidirectional)?,
            Self::Torus2d { width, height } => torus_edges(n, *width, *height)?,
            Self::FullyConnected => fully_connected_edges(n)?,
            Self::SmallWorld {
                degree,
                rewire_probability,
                seed,
            } => small_world_edges(n, *degree, *rewire_probability, *seed)?,
            Self::Custom(conns) => {
                let index: std::collections::HashMap<&str, usize> = ids
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (s.as_str(), i))
                    .collect();
                let mut out = Vec::with_capacity(conns.len());
                let mut custom_seen: HashSet<(usize, usize)> = HashSet::new();
                for c in conns {
                    let li = *index
                        .get(c.listener_id.as_str())
                        .ok_or_else(|| TopologyError::UnknownNodeId(c.listener_id.clone()))?;
                    let si = *index
                        .get(c.source_id.as_str())
                        .ok_or_else(|| TopologyError::UnknownNodeId(c.source_id.clone()))?;
                    if li == si {
                        return Err(TopologyError::SelfEdge(c.listener_id.clone()));
                    }
                    if !custom_seen.insert((li, si)) {
                        return Err(TopologyError::DuplicateNodeId(format!(
                            "{}<-{}",
                            c.listener_id, c.source_id
                        )));
                    }
                    out.push((li, si));
                }
                out
            }
        };

        // Final guard: no self-edges, no duplicates, across all variants. For
        // built-in generators a duplicate here is an internal invariant violation
        // (every generator is expected to emit distinct edges); surface it loudly
        // rather than silently dropping, so a future generator bug cannot mask
        // itself. Custom deduplicates upstream and also errors.
        let mut out = Vec::with_capacity(edges.len());
        let mut dedup: HashSet<(usize, usize)> = HashSet::new();
        for (li, si) in edges {
            if li == si {
                return Err(TopologyError::SelfEdge(ids[li].clone()));
            }
            if !dedup.insert((li, si)) {
                return Err(TopologyError::DuplicateEdge {
                    listener: ids[li].clone(),
                    from: ids[si].clone(),
                });
            }
            out.push(LateralConnection {
                listener_id: ids[li].clone(),
                source_id: ids[si].clone(),
            });
        }
        Ok(out)
    }
}

/// Directed ring: column `i` listens to `(i-1) mod n`; if bidirectional it also
/// listens to `(i+1) mod n`.
fn ring_edges(n: usize, bidirectional: bool) -> Result<Vec<(usize, usize)>, TopologyError> {
    if n < 2 {
        return Err(TopologyError::TooFewNodes {
            topology: "ring",
            min: 2,
            n,
        });
    }
    // n=2 bidirectional degenerates: predecessor == successor for every node, so
    // the naive loop would emit each edge twice. Emit the two distinct edges
    // directly (this is also == FullyConnected for n=2; see the degeneracy note
    // on `TopologySpec::SmallWorld`).
    if n == 2 && bidirectional {
        return Ok(vec![(0, 1), (1, 0)]);
    }
    let mut out = Vec::with_capacity(if bidirectional { 2 * n } else { n });
    for i in 0..n {
        let pred = (i + n - 1) % n;
        out.push((i, pred));
        if bidirectional {
            let succ = (i + 1) % n;
            out.push((i, succ));
        }
    }
    Ok(out)
}

/// 2-D torus: column at `(r, c)` listens to its four cardinal neighbors with
/// wraparound on both axes.
fn torus_edges(
    n: usize,
    width: usize,
    height: usize,
) -> Result<Vec<(usize, usize)>, TopologyError> {
    if width * height != n {
        return Err(TopologyError::TorusGeometry { width, height, n });
    }
    if width < 3 || height < 3 {
        return Err(TopologyError::TorusTooSmall { width, height });
    }
    let at = |r: usize, c: usize| r * width + c;
    let mut out = Vec::with_capacity(4 * n);
    for r in 0..height {
        for c in 0..width {
            let here = at(r, c);
            let up = at((r + height - 1) % height, c);
            let down = at((r + 1) % height, c);
            let left = at(r, (c + width - 1) % width);
            let right = at(r, (c + 1) % width);
            for src in [up, down, left, right] {
                out.push((here, src));
            }
        }
    }
    Ok(out)
}

/// Fully connected: every column listens to every other column.
fn fully_connected_edges(n: usize) -> Result<Vec<(usize, usize)>, TopologyError> {
    if n < 2 {
        return Err(TopologyError::TooFewNodes {
            topology: "fully_connected",
            min: 2,
            n,
        });
    }
    let mut out = Vec::with_capacity(n * (n - 1));
    for i in 0..n {
        for j in 0..n {
            if i != j {
                out.push((i, j));
            }
        }
    }
    Ok(out)
}

/// Watts-Strogatz directed small-world: ring lattice of `degree` nearest
/// neighbors, each edge rewired with probability `p` to a new source (not self,
/// not a duplicate) using a seeded splitmix64 PRNG. Deterministic given `seed`.
fn small_world_edges(
    n: usize,
    degree: usize,
    rewire_probability: f64,
    seed: u64,
) -> Result<Vec<(usize, usize)>, TopologyError> {
    if n < 2 {
        return Err(TopologyError::TooFewNodes {
            topology: "small_world",
            min: 2,
            n,
        });
    }
    if degree == 0 || degree % 2 != 0 {
        return Err(TopologyError::OutOfRange {
            parameter: "degree",
            reason: "must be a positive even integer",
        });
    }
    if degree >= n {
        return Err(TopologyError::OutOfRange {
            parameter: "degree",
            reason: "must be strictly less than the column count",
        });
    }
    if !rewire_probability.is_finite() || !(0.0..=1.0).contains(&rewire_probability) {
        return Err(TopologyError::OutOfRange {
            parameter: "rewire_probability",
            reason: "must be a finite number in [0, 1]",
        });
    }

    // sources[i] = the set of column indices column i listens to (ring lattice).
    let half = degree / 2;
    let mut sources: Vec<HashSet<usize>> = (0..n)
        .map(|i| {
            let mut s = HashSet::with_capacity(degree);
            for d in 1..=half {
                s.insert((i + d) % n);
                s.insert((i + n - d) % n);
            }
            s
        })
        .collect();

    // Rewire in a fixed iteration order so the result is a pure function of seed.
    let mut rng = Prng::new(seed);
    let max_attempts = 2 * n;
    for (i, sources_i) in sources.iter_mut().enumerate() {
        // Snapshot the lattice neighbors so rewiring one does not depend on the
        // order we visit them within this listener.
        let lattice: Vec<usize> = (1..=half)
            .flat_map(|d| [(i + d) % n, (i + n - d) % n])
            .collect();
        for s in lattice {
            if !sources_i.contains(&s) {
                // Already rewired away from this neighbor by an earlier step.
                continue;
            }
            if rng.next_f64() < rewire_probability {
                // Choose a new source != i, not already a source of i.
                let mut chosen: Option<usize> = None;
                for _ in 0..max_attempts {
                    let candidate = rng.range(n);
                    if candidate != i && !sources_i.contains(&candidate) {
                        chosen = Some(candidate);
                        break;
                    }
                }
                if let Some(new_src) = chosen {
                    sources_i.remove(&s);
                    sources_i.insert(new_src);
                }
                // else: keep the original edge (could not find a free target).
            }
        }
    }

    // Emit in a deterministic order: by listener, then ascending source index.
    let mut out = Vec::with_capacity(n * degree);
    for (i, sources_i) in sources.iter().enumerate() {
        let mut sorted: Vec<usize> = sources_i.iter().copied().collect();
        sorted.sort_unstable();
        for s in sorted {
            out.push((i, s));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    //! Tests use `?` throughout (no unwrap/expect/panic). Helpers build a
    //! canonical id list `["n0", "n1", ...]`.

    use super::*;

    fn ids(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("n{i}")).collect()
    }

    #[test]
    fn empty_id_list_is_rejected() {
        let spec = TopologySpec::FullyConnected;
        assert!(spec.connections_for(&[]).is_err());
    }

    #[test]
    fn duplicate_ids_are_rejected() -> Result<(), Box<dyn std::error::Error>> {
        let dup = vec!["a".to_string(), "a".to_string()];
        assert!(TopologySpec::FullyConnected.connections_for(&dup).is_err());
        Ok(())
    }

    #[test]
    fn ring_unidirectional_is_a_directed_cycle() -> Result<(), Box<dyn std::error::Error>> {
        let edges = TopologySpec::Ring {
            bidirectional: false,
        }
        .connections_for(&ids(5))?;
        // 5 columns, each listens to exactly its predecessor.
        assert_eq!(edges.len(), 5);
        let by_listener: std::collections::HashMap<&str, &str> = edges
            .iter()
            .map(|e| (e.listener_id.as_str(), e.source_id.as_str()))
            .collect();
        assert_eq!(by_listener["n0"], "n4");
        assert_eq!(by_listener["n1"], "n0");
        assert_eq!(by_listener["n2"], "n1");
        assert_eq!(by_listener["n3"], "n2");
        assert_eq!(by_listener["n4"], "n3");
        Ok(())
    }

    #[test]
    fn ring_bidirectional_has_two_sources_each() -> Result<(), Box<dyn std::error::Error>> {
        let edges = TopologySpec::Ring {
            bidirectional: true,
        }
        .connections_for(&ids(4))?;
        assert_eq!(edges.len(), 8);
        let sources_of_n0: Vec<&str> = edges
            .iter()
            .filter(|e| e.listener_id == "n0")
            .map(|e| e.source_id.as_str())
            .collect();
        assert!(sources_of_n0.contains(&"n1"));
        assert!(sources_of_n0.contains(&"n3"));
        Ok(())
    }

    #[test]
    fn torus_has_four_distinct_cardinal_sources_with_wraparound(
    ) -> Result<(), Box<dyn std::error::Error>> {
        // 3x3 = 9 columns, row-major: n0..n8.
        let edges = TopologySpec::Torus2d {
            width: 3,
            height: 3,
        }
        .connections_for(&ids(9))?;
        // 9 listeners * 4 sources = 36 directed edges.
        assert_eq!(edges.len(), 36);
        // Corner n0 (row 0, col 0): up wraps to row 2 (n6), down = row1 (n3),
        // left wraps to col 2 (n2), right = col1 (n1).
        let mut sources: Vec<&str> = edges
            .iter()
            .filter(|e| e.listener_id == "n0")
            .map(|e| e.source_id.as_str())
            .collect();
        sources.sort_unstable();
        assert_eq!(sources, vec!["n1", "n2", "n3", "n6"]);
        Ok(())
    }

    #[test]
    fn torus_geometry_mismatch_is_rejected() -> Result<(), Box<dyn std::error::Error>> {
        let res = TopologySpec::Torus2d {
            width: 3,
            height: 3,
        }
        .connections_for(&ids(8));
        assert!(res.is_err());
        Ok(())
    }

    #[test]
    fn torus_too_small_is_rejected() -> Result<(), Box<dyn std::error::Error>> {
        // 2x2 = 4 columns matches the count but dimensions < 3.
        let res = TopologySpec::Torus2d {
            width: 2,
            height: 2,
        }
        .connections_for(&ids(4));
        assert!(res.is_err());
        Ok(())
    }

    #[test]
    fn fully_connected_edge_count_is_n_times_n_minus_one() -> Result<(), Box<dyn std::error::Error>>
    {
        let edges = TopologySpec::FullyConnected.connections_for(&ids(5))?;
        assert_eq!(edges.len(), 20);
        // No self-edges, no duplicates.
        let mut seen = HashSet::new();
        for e in &edges {
            assert_ne!(e.listener_id, e.source_id);
            assert!(seen.insert((e.listener_id.clone(), e.source_id.clone())));
        }
        Ok(())
    }

    #[test]
    fn small_world_is_deterministic_for_a_fixed_seed() -> Result<(), Box<dyn std::error::Error>> {
        let spec = TopologySpec::SmallWorld {
            degree: 4,
            rewire_probability: 0.3,
            seed: 42,
        };
        let a = spec.connections_for(&ids(20))?;
        let b = spec.connections_for(&ids(20))?;
        assert_eq!(a, b);
        // Out-degree is preserved at `degree` (rewiring never adds/removes).
        assert_eq!(a.len(), 20 * 4);
        Ok(())
    }

    #[test]
    fn small_world_different_seeds_can_diverge() -> Result<(), Box<dyn std::error::Error>> {
        let s1 = TopologySpec::SmallWorld {
            degree: 4,
            rewire_probability: 0.9,
            seed: 1,
        };
        let s2 = TopologySpec::SmallWorld {
            degree: 4,
            rewire_probability: 0.9,
            seed: 2,
        };
        let a = s1.connections_for(&ids(40))?;
        let b = s2.connections_for(&ids(40))?;
        assert_ne!(a, b, "different seeds should diverge at high p");
        Ok(())
    }

    #[test]
    fn small_world_rejects_odd_degree() -> Result<(), Box<dyn std::error::Error>> {
        let res = TopologySpec::SmallWorld {
            degree: 3,
            rewire_probability: 0.2,
            seed: 1,
        }
        .connections_for(&ids(20));
        assert!(res.is_err());
        Ok(())
    }

    #[test]
    fn small_world_rejects_bad_probability() -> Result<(), Box<dyn std::error::Error>> {
        let res = TopologySpec::SmallWorld {
            degree: 4,
            rewire_probability: 1.5,
            seed: 1,
        }
        .connections_for(&ids(20));
        assert!(res.is_err());
        Ok(())
    }

    #[test]
    fn custom_validates_self_and_unknown_and_dup() -> Result<(), Box<dyn std::error::Error>> {
        let id5 = ids(5);
        // Self-edge.
        let self_edge = TopologySpec::Custom(vec![LateralConnection::new("n0", "n0")]);
        assert!(self_edge.connections_for(&id5).is_err());
        // Unknown id.
        let unknown = TopologySpec::Custom(vec![LateralConnection::new("n0", "zzz")]);
        assert!(unknown.connections_for(&id5).is_err());
        // Duplicate edge is rejected (no silent dedup for Custom).
        let dup = TopologySpec::Custom(vec![
            LateralConnection::new("n0", "n1"),
            LateralConnection::new("n0", "n1"),
        ]);
        assert!(dup.connections_for(&id5).is_err());
        Ok(())
    }

    #[test]
    fn zero_seed_does_not_collide_with_nonzero() -> Result<(), Box<dyn std::error::Error>> {
        let a = TopologySpec::SmallWorld {
            degree: 4,
            rewire_probability: 0.5,
            seed: 0,
        }
        .connections_for(&ids(30))?;
        let b = TopologySpec::SmallWorld {
            degree: 4,
            rewire_probability: 0.5,
            seed: 7,
        }
        .connections_for(&ids(30))?;
        assert_ne!(a, b);
        Ok(())
    }

    #[test]
    fn ring_bidirectional_n2_yields_two_distinct_edges() -> Result<(), Box<dyn std::error::Error>> {
        // n=2 bidirectional: predecessor == successor, so without the special
        // case the naive loop emits each edge twice. Expect exactly 2 distinct.
        let edges = TopologySpec::Ring {
            bidirectional: true,
        }
        .connections_for(&ids(2))?;
        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].listener_id, "n0");
        assert_eq!(edges[0].source_id, "n1");
        assert_eq!(edges[1].listener_id, "n1");
        assert_eq!(edges[1].source_id, "n0");
        Ok(())
    }

    /// Records the KNOWN degenerate equivalences (red-team finding). These are
    /// not bugs — the graphs are genuinely identical in these parameter regions —
    /// but a topology-comparison benchmark MUST guard against comparing two specs
    /// that collapse to the same edge set. The test exists to document the hazard
    /// and fail loudly if the equivalences ever change shape.
    #[test]
    fn documented_degenerate_equivalences() -> Result<(), Box<dyn std::error::Error>> {
        let canon = |cs: Vec<LateralConnection>| {
            let mut v: Vec<(String, String)> = cs
                .into_iter()
                .map(|e| (e.listener_id, e.source_id))
                .collect();
            v.sort();
            v
        };

        // n=2: bidirectional ring == unidirectional ring == fully connected.
        let id2 = ids(2);
        assert_eq!(
            canon(
                TopologySpec::Ring {
                    bidirectional: true
                }
                .connections_for(&id2)?
            ),
            canon(TopologySpec::FullyConnected.connections_for(&id2)?)
        );

        // n=3: bidirectional ring == fully connected (out-degree 2 of only 2 others).
        let id3 = ids(3);
        assert_eq!(
            canon(
                TopologySpec::Ring {
                    bidirectional: true
                }
                .connections_for(&id3)?
            ),
            canon(TopologySpec::FullyConnected.connections_for(&id3)?)
        );

        // n=5: small-world degree=4 (== n-1, odd n) == fully connected,
        // regardless of rewire_probability or seed.
        let id5 = ids(5);
        let sw_complete = TopologySpec::SmallWorld {
            degree: 4,
            rewire_probability: 0.0,
            seed: 1,
        };
        let fc5 = TopologySpec::FullyConnected.connections_for(&id5)?;
        assert_eq!(
            canon(sw_complete.connections_for(&id5)?),
            canon(fc5.clone()),
            "degree==n-1 on odd n must collapse to the complete graph"
        );
        // Different p/seed still collapse (no free targets to rewire to).
        assert_eq!(
            canon(
                TopologySpec::SmallWorld {
                    degree: 4,
                    rewire_probability: 1.0,
                    seed: 99,
                }
                .connections_for(&id5)?
            ),
            canon(fc5)
        );

        // Contrast: a sane small-world region (n=20, degree=4) is NOT fully
        // connected and is NOT a pure ring lattice at high p.
        let id20 = ids(20);
        let lattice = TopologySpec::SmallWorld {
            degree: 4,
            rewire_probability: 0.0,
            seed: 1,
        }
        .connections_for(&id20)?;
        let rewired = TopologySpec::SmallWorld {
            degree: 4,
            rewire_probability: 1.0,
            seed: 1,
        }
        .connections_for(&id20)?;
        assert_eq!(lattice.len(), 80);
        assert_eq!(rewired.len(), 80, "out-degree is preserved");
        assert_ne!(
            canon(lattice),
            canon(rewired),
            "degree≪n with high p should actually rewire"
        );
        assert_ne!(
            canon(
                TopologySpec::SmallWorld {
                    degree: 4,
                    rewire_probability: 0.5,
                    seed: 1,
                }
                .connections_for(&id20)?
            ),
            canon(TopologySpec::FullyConnected.connections_for(&id20)?),
            "sane small-world is not fully connected"
        );
        Ok(())
    }
}
