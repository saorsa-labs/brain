//! Lateral voting convergence math (§6 Phase 3: "Global Integration").
//!
//! There is no master coordinator: columns exchange predictions along lateral
//! connections and the mesh re-evaluates until metric-based convergence criteria
//! are satisfied. This crate provides those criteria and the vector math used to
//! test them.
//!
//! V1 convergence operates on the per-column confidence vector. Full *semantic*
//! cosine similarity over prediction embeddings is future work (§9) since it
//! requires an embedding backend.

use ndarray::Array1;
use ptg_core::CorticalColumn;

/// Metric-based convergence criteria for the consensus loop.
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria {
    /// Hard cap on the number of voting ticks per epoch.
    pub max_ticks: u32,
    /// Stop once mean column confidence reaches this threshold.
    pub min_mean_confidence: f32,
    /// Stop once mean absolute confidence delta between ticks drops at or below this.
    pub max_confidence_delta: f32,
    /// Stop once cosine similarity of successive confidence vectors reaches this.
    pub min_cosine_similarity: f32,
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self {
            max_ticks: 5,
            min_mean_confidence: 0.8,
            max_confidence_delta: 0.02,
            min_cosine_similarity: 0.999,
        }
    }
}

/// Mean confidence across a set of columns. Returns `0.0` for an empty set.
#[must_use]
pub fn mean_confidence(columns: &[&CorticalColumn]) -> f32 {
    if columns.is_empty() {
        return 0.0;
    }
    let sum: f32 = columns.iter().map(|c| c.last_confidence).sum();
    sum / columns.len() as f32
}

/// Build the confidence vector from a set of columns, in the given order.
#[must_use]
pub fn confidence_vector(columns: &[&CorticalColumn]) -> Array1<f32> {
    Array1::from_iter(columns.iter().map(|c| c.last_confidence))
}

/// Cosine similarity of two vectors. Returns `0.0` for empty or mismatched
/// lengths, or if either vector has zero magnitude.
#[must_use]
pub fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let dot = a.dot(b);
    let na = a.dot(a).sqrt();
    let nb = b.dot(b).sqrt();
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Whether the mesh has reached a *quality* convergence criterion (not merely
/// the tick cap). Used to flag that predictions stabilized.
///
/// Converges if any of:
/// - mean confidence `>= min_mean_confidence`, or
/// - mean absolute confidence delta `<= max_confidence_delta`, or
/// - cosine similarity of successive confidence vectors `>= min_cosine_similarity`.
///
/// The delta/cosine checks require a non-empty `previous` vector of the same
/// length as `current`; on the first tick they are skipped.
#[must_use]
pub fn quality_converged(
    columns: &[&CorticalColumn],
    previous: &Array1<f32>,
    current: &Array1<f32>,
    criteria: &ConvergenceCriteria,
) -> bool {
    if mean_confidence(columns) >= criteria.min_mean_confidence {
        return true;
    }
    if !previous.is_empty() && previous.len() == current.len() && !current.is_empty() {
        let delta = (current - previous).mapv(f32::abs).sum() / current.len() as f32;
        if delta <= criteria.max_confidence_delta {
            return true;
        }
        if cosine_similarity(previous, current) >= criteria.min_cosine_similarity {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptg_core::{CorticalColumn, DomainSphere};

    fn col(id: &str, conf: f32) -> CorticalColumn {
        let mut c = CorticalColumn::with_defaults(id, DomainSphere::Physics);
        c.last_confidence = conf;
        c
    }

    #[test]
    fn mean_confidence_empty_is_zero() {
        assert!((mean_confidence(&[]) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_zero_vector_is_zero() {
        let a = Array1::from_vec(vec![0.0, 0.0]);
        let b = Array1::from_vec(vec![1.0, 1.0]);
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_identical_is_one() {
        let a = Array1::from_vec(vec![0.1, 0.9, 0.5]);
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn quality_converges_on_high_mean_confidence() {
        let cols = [col("a", 0.95), col("b", 0.9)];
        let refs: Vec<&CorticalColumn> = cols.iter().collect();
        let current = confidence_vector(&refs);
        let previous = Array1::zeros(0);
        let criteria = ConvergenceCriteria::default();
        assert!(quality_converged(&refs, &previous, &current, &criteria));
    }

    #[test]
    fn quality_converges_on_small_delta() {
        let cols = [col("a", 0.3), col("b", 0.35)];
        let refs: Vec<&CorticalColumn> = cols.iter().collect();
        let previous = Array1::from_vec(vec![0.3, 0.34]);
        let current = confidence_vector(&refs);
        let criteria = ConvergenceCriteria::default();
        // mean (0.325) < 0.8, but delta is tiny -> converged.
        assert!(quality_converged(&refs, &previous, &current, &criteria));
    }

    #[test]
    fn quality_does_not_converge_on_first_tick_without_high_mean() {
        let cols = [col("a", 0.1), col("b", 0.2)];
        let refs: Vec<&CorticalColumn> = cols.iter().collect();
        let current = confidence_vector(&refs);
        let previous = Array1::zeros(0); // first tick
        let criteria = ConvergenceCriteria::default();
        assert!(!quality_converged(&refs, &previous, &current, &criteria));
    }
}
