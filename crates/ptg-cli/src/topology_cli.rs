//! Topology resolution shared by the PTG binaries (§3.1.3).
//!
//! A generated topology plan is a fully-materialized list of columns plus a
//! listener->source edge list and a human label. [`topology_plan`] resolves a
//! [`MeshTopologyParams`] into such a plan; `None` means "use the named default
//! reference mesh".
//!
//! This is the single source of truth for topology validation: both the `ptg`
//! CLI and `ptg-bench` route through it so column-count and small-world degree
//! guards cannot drift between the two.

use std::path::PathBuf;

use clap::ValueEnum;
use ptg_core::{replicated_default_columns, CorticalColumn, LateralConnection, TopologySpec};

use crate::column_pack;

/// Selectable lateral topologies for `--topology`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum TopologyKind {
    /// Named 4-column reference graph (§8.4), unchanged.
    Default,
    /// Directed 1-D cycle: each column listens to its predecessor.
    Ring,
    /// Bidirectional ring: each column listens to predecessor + successor.
    /// Requires >= 4 columns (for n<=3 it collapses to the complete graph).
    RingBi,
    /// 2-D wraparound grid: each column listens to its four cardinal neighbors.
    Torus,
    /// Every column listens to every other.
    FullyConnected,
    /// Seeded Watts-Strogatz small-world (deterministic given `--small-world-seed`).
    SmallWorld,
}

impl TopologyKind {
    /// The kebab-case flag value the user would type (for error messages).
    #[must_use]
    pub const fn kebab(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Ring => "ring",
            Self::RingBi => "ring-bi",
            Self::Torus => "torus",
            Self::FullyConnected => "fully-connected",
            Self::SmallWorld => "small-world",
        }
    }
}

/// Resolved topology input, independent of any particular CLI's arg struct.
/// Construct this from `ptg` or `ptg-bench` flags and pass it to
/// [`topology_plan`].
#[derive(Clone, Debug)]
pub struct MeshTopologyParams {
    /// Topology family (`Default` => named 4-column reference graph).
    pub topology: TopologyKind,
    /// Number of columns for a generated topology (ignored for `Default`).
    pub columns: Option<usize>,
    /// Torus grid width (required for `Torus`).
    pub torus_width: Option<usize>,
    /// Torus grid height (required for `Torus`).
    pub torus_height: Option<usize>,
    /// Small-world ring-lattice out-degree (even).
    pub small_world_degree: usize,
    /// Small-world edge rewire probability in `[0, 1]`.
    pub small_world_rewire: f64,
    /// Small-world PRNG seed (same seed => same graph).
    pub small_world_seed: u64,
    /// Optional explicit column pack (id / sphere / system_prompt / level).
    pub column_pack: Option<PathBuf>,
}

/// A fully-materialized topology plan: the columns plus the listener->source
/// edge list, plus a human label. Built by [`topology_plan`]; `None` means "use
/// the default reference mesh".
#[derive(Debug)]
pub struct MeshPlan {
    /// The columns, in positional order.
    pub columns: Vec<CorticalColumn>,
    /// The lateral edges as `LateralConnection`s (listener receives source).
    pub connections: Vec<LateralConnection>,
    /// Optional abstraction-level tag per column (parallel to `columns`),
    /// populated from a column pack's `level` field; empty for replicated
    /// defaults.
    pub levels: Vec<Option<u8>>,
    /// Human-readable topology label (`"ring"`, `"small-world"`, ...).
    pub label: String,
}

/// The four named ids of the reference 4-column mesh, in any order.
fn expected_default_ids() -> &'static [&'static str] {
    &["CC_PHYSICS_01", "CC_MATH_01", "CC_CODE_01", "CC_PSYCH_01"]
}

/// Validate that a pack reproduces the named reference graph (by id, order-
/// independent) so `--topology default --column-pack` is the same wiring as
/// `default_mesh`.
///
/// # Errors
/// `Err(String)` if the pack id set is not exactly the four reference ids.
fn validate_default_pack_ids(columns: &[CorticalColumn]) -> Result<(), String> {
    let mut have: Vec<&str> = columns.iter().map(|c| c.id.as_str()).collect();
    have.sort_unstable();
    let mut want: Vec<&str> = expected_default_ids().to_vec();
    want.sort_unstable();
    if have == want {
        Ok(())
    } else {
        Err(format!(
            "--topology default with --column-pack requires exactly these ids \n(order-independent): {}\n(pack has: {})",
            expected_default_ids().join(", "),
            columns
                .iter()
                .map(|c| c.id.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }
}

/// The reference 4-column lateral edges as `LateralConnection`s.
/// `default_connections()` tuples are `(listener, source)` (the listener
/// receives the source's prediction); the conversion preserves that direction.
fn default_reference_edges() -> Vec<LateralConnection> {
    ptg_core::default_connections()
        .into_iter()
        .map(|(listener, source)| LateralConnection::new(listener, source))
        .collect()
}

/// Resolve the column count from `--columns`, or for torus from the grid dims.
/// Errors with a friendly message naming the topology.
fn resolve_columns(p: &MeshTopologyParams) -> Result<usize, String> {
    if p.topology == TopologyKind::Torus {
        let w = p
            .torus_width
            .ok_or_else(|| "--topology torus requires --torus-width".to_string())?;
        let h = p
            .torus_height
            .ok_or_else(|| "--topology torus requires --torus-height".to_string())?;
        let derived = w
            .checked_mul(h)
            .ok_or_else(|| "--torus-width * --torus-height overflows usize".to_string())?;
        if let Some(requested) = p.columns {
            if requested != derived {
                return Err(format!(
                    "--columns {requested} != --torus-width {w} * --torus-height {h} ({derived})"
                ));
            }
        }
        Ok(derived)
    } else {
        p.columns
            .ok_or_else(|| format!("--topology {} requires --columns", p.topology.kebab()))
    }
}

/// Build a topology plan from resolved params. Returns `Ok(None)` for
/// [`TopologyKind::Default`] without a column pack (the caller uses the default
/// reference mesh); otherwise the columns + edge list + label.
///
/// # Errors
/// Friendly `String` messages for: missing/contradictory column counts, degree
/// too high for small-world, or column-pack id/count mismatches.
pub fn topology_plan(p: &MeshTopologyParams) -> Result<Option<MeshPlan>, String> {
    if p.topology == TopologyKind::Default {
        return match &p.column_pack {
            None => Ok(None),
            Some(path) => {
                // A pack + default topology is only valid if it reproduces the
                // named reference graph (by id, order-independent).
                let cols = column_pack::load_column_pack_with_levels(path)?;
                let (columns, levels): (Vec<_>, Vec<_>) = cols.into_iter().unzip();
                validate_default_pack_ids(&columns)?;
                let connections = default_reference_edges();
                Ok(Some(MeshPlan {
                    columns,
                    connections,
                    levels,
                    label: "default".to_string(),
                }))
            }
        };
    }

    let n = resolve_columns(p)?;

    match p.topology {
        TopologyKind::Default => {}
        TopologyKind::Ring if n < 2 => {
            return Err("--topology ring requires --columns >= 2".to_string());
        }
        TopologyKind::RingBi if n < 4 => {
            return Err(
                "--topology ring-bi requires --columns >= 4 (else it == fully-connected)"
                    .to_string(),
            );
        }
        TopologyKind::FullyConnected if n < 2 => {
            return Err("--topology fully-connected requires --columns >= 2".to_string());
        }
        TopologyKind::SmallWorld if p.small_world_degree * 2 >= n => {
            return Err(format!(
                "--topology small-world requires --small-world-degree * 2 < --columns \n(got degree {}, columns {})",
                p.small_world_degree, n
            ));
        }
        _ => {}
    }

    let columns_levels: Vec<Option<u8>>;
    let columns = match &p.column_pack {
        Some(path) => {
            let pack = column_pack::load_column_pack_with_levels(path)?;
            if pack.len() != n {
                return Err(format!(
                    "--column-pack has {} column(s) but topology expects {n} \n(derive via --columns or --torus-width/--torus-height)",
                    pack.len()
                ));
            }
            let (columns, levels): (Vec<_>, Vec<_>) = pack.into_iter().unzip();
            columns_levels = levels;
            columns
        }
        None => {
            columns_levels = vec![None; n];
            replicated_default_columns(n)
        }
    };
    let ids: Vec<String> = columns.iter().map(|c| c.id.clone()).collect();
    let (spec, label) = match p.topology {
        TopologyKind::Default => {
            // `Default` is dispatched and returned earlier (it never reaches
            // this second match). Surface as an error rather than panic.
            return Err(
                "internal: Default topology should be handled before topology dispatch".to_string(),
            );
        }
        TopologyKind::Ring => (
            TopologySpec::Ring {
                bidirectional: false,
            },
            "ring",
        ),
        TopologyKind::RingBi => (
            TopologySpec::Ring {
                bidirectional: true,
            },
            "ring-bi",
        ),
        TopologyKind::Torus => {
            let w = p.torus_width.unwrap_or(0);
            let h = p.torus_height.unwrap_or(0);
            (
                TopologySpec::Torus2d {
                    width: w,
                    height: h,
                },
                "torus",
            )
        }
        TopologyKind::FullyConnected => (TopologySpec::FullyConnected, "fully-connected"),
        TopologyKind::SmallWorld => (
            TopologySpec::SmallWorld {
                degree: p.small_world_degree,
                rewire_probability: p.small_world_rewire,
                seed: p.small_world_seed,
            },
            "small-world",
        ),
    };
    let connections = spec
        .connections_for(&ids)
        .map_err(|e| format!("{label}: {e}"))?;
    Ok(Some(MeshPlan {
        columns,
        connections,
        levels: columns_levels,
        label: label.to_string(),
    }))
}
