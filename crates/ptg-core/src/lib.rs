//! Core domain types for Project Thousand-Gemma (PTG).
//!
//! These types model the building blocks of the distributed, prompt-based
//! cortical mesh described in the architectural specification: virtual
//! cortical columns, their reference frames, their outputs, and the
//! topology of lateral connections between them.

use serde::{Deserialize, Serialize};

/// Identifier for a single virtual cortical column within the mesh.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ColumnId(pub u64);

impl ColumnId {
    /// Create a new column identifier.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }
}

impl std::fmt::Display for ColumnId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "col-{}", self.0)
    }
}

/// The sensory / cognitive modality a column is specialized for.
///
/// Mirrors the multi-modal columns referenced in the Thousand Brains mapping
/// (visual, tactile, abstract-mathematical, ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    /// Visual scene analysis.
    Visual,
    /// Tactile / somatosensory signals.
    Tactile,
    /// Auditory signals.
    Auditory,
    /// Abstract symbolic reasoning (e.g. mathematics, logic).
    Abstract,
    /// Proprioceptive / positional signals.
    Proprioceptive,
}

/// A localized coordinate map that constrains a column's perception to a
/// specific spatial or conceptual region.
///
/// In PTG a reference frame is realized as forced structural JSON that bounds
/// the column's output space, so arbitrary spatial / conceptual axes can be
/// expressed without code changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceFrame {
    /// Stable identifier for the frame (e.g. `"left-visual-field"`).
    pub id: String,
    /// Human-readable description of what the frame covers.
    pub description: String,
    /// Structured schema bounding the frame's coordinate space.
    pub schema: serde_json::Value,
}

/// Hyper-targeted system prompt and configuration that turns a generic
/// inference call into a specialized cortical column.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnSpec {
    pub id: ColumnId,
    pub modality: Modality,
    pub reference_frame: ReferenceFrame,
    /// The domain-specific system prompt enforcing this column's cognitive prism.
    pub system_prompt: String,
}

/// A single column's prediction about its slice of the world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnOutput {
    pub column: ColumnId,
    /// The column's prediction / hypothesis, structurally formatted per its frame.
    pub prediction: serde_json::Value,
    /// Self-reported confidence in `[0.0, 1.0]`, used during lateral voting.
    pub confidence: f32,
}

/// A token-bearing message passed along a lateral connection from one column
/// to a neighbor, injecting structural context into the receiver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LateralMessage {
    pub from: ColumnId,
    pub to: ColumnId,
    /// Serialized neighbor context injected into the receiver's prompt.
    pub context: String,
    /// Weight applied to this neighbor's vote.
    pub weight: f32,
}

/// Edge list describing the mesh of lateral connections between columns.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Topology {
    /// Directed weighted edges: `(from, to, weight)`.
    pub edges: Vec<(ColumnId, ColumnId, f32)>,
}

impl Topology {
    /// Create an empty topology.
    #[must_use]
    pub fn new() -> Self {
        Self { edges: Vec::new() }
    }

    /// Connect `from` -> `to` with the given lateral weight.
    pub fn connect(&mut self, from: ColumnId, to: ColumnId, weight: f32) {
        self.edges.push((from, to, weight));
    }

    /// Return the columns that `column` listens to, paired with edge weight.
    pub fn neighbors(&self, column: &ColumnId) -> Vec<(&ColumnId, f32)> {
        self.edges
            .iter()
            .filter(|(_, to, _)| to == column)
            .map(|(from, _, w)| (from, *w))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_id_displays_stable_prefix() {
        assert_eq!(ColumnId::new(7).to_string(), "col-7");
    }

    #[test]
    fn topology_neighbors_respect_direction() {
        let mut topo = Topology::new();
        topo.connect(ColumnId::new(1), ColumnId::new(2), 0.5);
        topo.connect(ColumnId::new(3), ColumnId::new(2), 0.25);
        let neighbors = topo.neighbors(&ColumnId::new(2));
        assert_eq!(neighbors.len(), 2);
    }
}
