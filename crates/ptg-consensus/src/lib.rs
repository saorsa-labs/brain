//! Multi-round, asynchronous consensus over a population of column outputs,
//! mirroring the lateral voting paradigm of the Thousand Brains Theory.
//!
//! There is no master coordinator: columns exchange predictions along the
//! topology's lateral edges and re-evaluate until metric-based convergence
//! criteria are satisfied.

use ptg_core::{ColumnId, ColumnOutput, Topology};

/// Snapshot of one consensus round across the mesh.
#[derive(Debug, Clone)]
pub struct ConsensusRound {
    pub round: u32,
    pub outputs: Vec<ColumnOutput>,
}

/// Metric-based convergence criteria that decide when voting has stabilized.
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria {
    /// Hard cap on the number of voting rounds.
    pub max_rounds: u32,
    /// Stop once mean column confidence reaches this threshold.
    pub min_mean_confidence: f32,
    /// Stop once the semantic delta between rounds drops below this value.
    pub max_output_delta: f32,
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self {
            max_rounds: 8,
            min_mean_confidence: 0.8,
            max_output_delta: 0.02,
        }
    }
}

/// State held between rounds of lateral voting.
#[derive(Debug, Clone)]
pub struct VotingState {
    pub current: Vec<ColumnOutput>,
}

impl VotingState {
    /// Create a new voting state from an initial set of column outputs.
    #[must_use]
    pub fn new(initial: Vec<ColumnOutput>) -> Self {
        Self { current: initial }
    }

    /// Mean confidence across all current column outputs.
    pub fn mean_confidence(&self) -> f32 {
        if self.current.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.current.iter().map(|o| o.confidence).sum();
        sum / self.current.len() as f32
    }
}

/// Determine whether the mesh has converged under the given criteria.
///
/// The semantic output-delta metric depends on the structured-prediction
/// comparison strategy and is wired up as the consensus engine is implemented;
/// for now convergence is driven by round count and mean confidence.
pub fn has_converged(state: &VotingState, criteria: &ConvergenceCriteria, round: u32) -> bool {
    round >= criteria.max_rounds || state.mean_confidence() >= criteria.min_mean_confidence
}

/// Build the per-column weight set used to fold neighbor votes, derived from
/// the topology's lateral edge weights.
pub fn neighbor_weights(topology: &Topology, column: &ColumnId) -> Vec<(ColumnId, f32)> {
    topology
        .neighbors(column)
        .into_iter()
        .map(|(id, w)| (id.clone(), w))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_state_has_zero_confidence() {
        let state = VotingState::new(Vec::new());
        assert_eq!(state.mean_confidence(), 0.0);
    }

    #[test]
    fn convergence_triggers_on_round_cap() {
        let state = VotingState::new(Vec::new());
        let criteria = ConvergenceCriteria::default();
        assert!(has_converged(&state, &criteria, criteria.max_rounds));
    }
}
