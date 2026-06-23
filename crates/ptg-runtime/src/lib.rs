//! Runtime that orchestrates a population of cortical columns: parallel
//! fan-out of stimulus to columns, lateral context injection, and multi-round
//! consensus over the mesh.

use std::sync::Arc;

use ptg_consensus::{ConvergenceCriteria, VotingState};
use ptg_core::{ColumnSpec, Topology};
use ptg_vllm::VllmEngine;

/// A fully-configured cortical mesh ready to run.
///
/// All columns share a single inference engine instance (`engine`), matching
/// the single-engine topology described in the specification.
pub struct CorticalMesh {
    /// Column specifications (system prompts + reference frames + modalities).
    pub columns: Vec<ColumnSpec>,
    /// Lateral connection graph driving token injection and voting.
    pub topology: Topology,
    /// The shared local inference engine ("thalamus").
    pub engine: Arc<dyn VllmEngine>,
    /// When to stop the multi-round voting loop.
    pub convergence: ConvergenceCriteria,
}

impl CorticalMesh {
    /// Create a new mesh with default convergence criteria.
    #[must_use]
    pub fn new(columns: Vec<ColumnSpec>, topology: Topology, engine: Arc<dyn VllmEngine>) -> Self {
        Self {
            columns,
            topology,
            engine,
            convergence: ConvergenceCriteria::default(),
        }
    }
}

/// Snapshot of one completed consensus pass over the mesh.
#[derive(Debug, Clone)]
pub struct MeshResult {
    pub rounds: u32,
    pub final_state: VotingState,
}
