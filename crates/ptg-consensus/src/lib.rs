//! Lateral voting convergence math (§6 Phase 3: "Global Integration").
//!
//! There is no master coordinator: columns exchange predictions along lateral
//! connections and the mesh re-evaluates until metric-based convergence criteria
//! are satisfied. This crate provides those criteria and the vector math used to
//! test them.
//!
//! V1 convergence operates on the per-column confidence vector. A cheap,
//! model-independent **prediction-stability** proxy (token-Jaccard similarity of
//! successive per-column predictions) is available via
//! `ConvergenceCriteria.min_prediction_similarity` — this does not rely on the
//! self-reported confidence a model can game. Full *semantic* cosine similarity
//! over prediction embeddings remains future work (§9) since it requires an
//! embedding backend.

use std::collections::BTreeSet;

use ndarray::Array1;
use ptg_core::CorticalColumn;

/// Metric-based convergence criteria for the consensus loop.
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria {
    /// Hard cap on the number of voting ticks per epoch.
    pub max_ticks: u32,
    /// Do not consider convergence before this tick. Guarantees at least this
    /// many ticks of lateral-context exchange regardless of confidence. Default
    /// `1` (preserve legacy behavior). Set higher (e.g. 2) to force the mesh to
    /// actually exercise lateral voting even when an overconfident model would
    /// otherwise "converge" on tick 1.
    pub min_ticks: u32,
    /// Stop once mean column confidence reaches this threshold.
    pub min_mean_confidence: f32,
    /// Stop once mean absolute confidence delta between ticks drops at or below this.
    pub max_confidence_delta: f32,
    /// Stop once cosine similarity of successive confidence vectors reaches this.
    pub min_cosine_similarity: f32,
    /// Stop once mean token-Jaccard similarity of successive per-column
    /// predictions reaches this. A cheap, model-independent string proxy for
    /// semantic stabilization (§9.3, unblocked approximation) that does **not**
    /// rely on the self-reported confidence a model can game. `None` (default)
    /// disables prediction-stability convergence and preserves V1 behavior.
    pub min_prediction_similarity: Option<f32>,
    /// Columns with final confidence below this are excluded from the integrated
    /// global percept (§6 Phase 3).
    pub min_integration_confidence: f32,
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self {
            max_ticks: 5,
            min_ticks: 1,
            min_mean_confidence: 0.8,
            max_confidence_delta: 0.02,
            min_cosine_similarity: 0.999,
            min_prediction_similarity: None,
            min_integration_confidence: 0.5,
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

/// Why the consensus loop stopped on a *quality* criterion (as opposed to
/// merely running out of ticks). Reported on [`crate`] results so callers can
/// attribute which mechanism actually terminated the epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceReason {
    /// Mean column confidence reached `min_mean_confidence`.
    MeanConfidence,
    /// Mean absolute confidence delta between ticks dropped at or below
    /// `max_confidence_delta`.
    ConfidenceDelta,
    /// Cosine similarity of successive confidence vectors reached
    /// `min_cosine_similarity`.
    ConfidenceCosine,
    /// Mean token-Jaccard similarity of successive per-column predictions
    /// reached `min_prediction_similarity`.
    PredictionSimilarity,
}

impl std::fmt::Display for ConvergenceReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::MeanConfidence => "mean confidence threshold met",
            Self::ConfidenceDelta => "confidence delta stabilized",
            Self::ConfidenceCosine => "confidence vector cosine stabilized",
            Self::PredictionSimilarity => "prediction token-similarity stabilized",
        };
        f.write_str(s)
    }
}

/// Which confidence-based convergence criterion (if any) is satisfied.
///
/// Checks, in order: mean confidence, then (given a non-empty `previous` of
/// matching length) mean absolute delta, then cosine similarity. Returns the
/// first criterion that fires.
#[must_use]
pub fn confidence_converged(
    columns: &[&CorticalColumn],
    previous: &Array1<f32>,
    current: &Array1<f32>,
    criteria: &ConvergenceCriteria,
) -> Option<ConvergenceReason> {
    if mean_confidence(columns) >= criteria.min_mean_confidence {
        return Some(ConvergenceReason::MeanConfidence);
    }
    if !previous.is_empty() && previous.len() == current.len() && !current.is_empty() {
        let delta = (current - previous).mapv(f32::abs).sum() / current.len() as f32;
        if delta <= criteria.max_confidence_delta {
            return Some(ConvergenceReason::ConfidenceDelta);
        }
        if cosine_similarity(previous, current) >= criteria.min_cosine_similarity {
            return Some(ConvergenceReason::ConfidenceCosine);
        }
    }
    None
}

/// Whether a prediction-stability criterion (token-Jaccard) is satisfied, if
/// `min_prediction_similarity` is enabled. Requires a non-empty `previous`
/// prediction slice of the same length as `current`; skipped on the first tick.
#[must_use]
pub fn prediction_similarity_converged(
    previous: &[String],
    current: &[String],
    criteria: &ConvergenceCriteria,
) -> Option<ConvergenceReason> {
    let threshold = criteria.min_prediction_similarity?;
    if previous.is_empty() || current.is_empty() || previous.len() != current.len() {
        return None;
    }
    let sim = mean_prediction_similarity(previous, current);
    if sim >= threshold {
        Some(ConvergenceReason::PredictionSimilarity)
    } else {
        None
    }
}

/// Lowercase a string and split it into an unordered set of alphanumeric tokens.
/// A zero-dependency tokenizer shared by the similarity helpers.
fn tokenize(text: &str) -> BTreeSet<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(std::string::ToString::to_string)
        .collect()
}

/// Token-Jaccard similarity of two strings: `|tokens(a) ∩ tokens(b)| /
/// |tokens(a) ∪ tokens(b)|`. Order-independent, case-insensitive. Returns `0.0`
/// when either side has no tokens (including two empty strings), so empty or
/// mismatched inputs never spuriously "converge".
#[must_use]
pub fn token_jaccard_similarity(a: &str, b: &str) -> f32 {
    let ta = tokenize(a);
    let tb = tokenize(b);
    let union = ta.union(&tb).count();
    if union == 0 {
        return 0.0;
    }
    let inter = ta.intersection(&tb).count() as f32;
    inter / union as f32
}

/// Mean token-Jaccard similarity across aligned per-column prediction strings.
/// Returns `0.0` for empty or mismatched-length slices.
#[must_use]
pub fn mean_prediction_similarity(previous: &[String], current: &[String]) -> f32 {
    if previous.is_empty() || previous.len() != current.len() {
        return 0.0;
    }
    let sum: f32 = previous
        .iter()
        .zip(current.iter())
        .map(|(p, c)| token_jaccard_similarity(p, c))
        .sum();
    sum / previous.len() as f32
}

/// Whether the mesh has reached a *quality* convergence criterion (not merely
/// the tick cap). Convenience bool wrapper over [`confidence_converged`] for
/// callers that do not need the reason. Note: this checks confidence criteria
/// only — prediction-stability convergence is reported separately via
/// [`prediction_similarity_converged`].
#[must_use]
pub fn quality_converged(
    columns: &[&CorticalColumn],
    previous: &Array1<f32>,
    current: &Array1<f32>,
    criteria: &ConvergenceCriteria,
) -> bool {
    confidence_converged(columns, previous, current, criteria).is_some()
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

    #[test]
    fn confidence_converged_reports_mean_confidence_reason() {
        let cols = [col("a", 0.95), col("b", 0.9)];
        let refs: Vec<&CorticalColumn> = cols.iter().collect();
        let current = confidence_vector(&refs);
        let reason = confidence_converged(
            &refs,
            &Array1::zeros(0),
            &current,
            &ConvergenceCriteria::default(),
        );
        assert_eq!(reason, Some(ConvergenceReason::MeanConfidence));
    }

    #[test]
    fn confidence_converged_reports_delta_reason() {
        let cols = [col("a", 0.3), col("b", 0.35)];
        let refs: Vec<&CorticalColumn> = cols.iter().collect();
        let current = confidence_vector(&refs);
        let previous = Array1::from_vec(vec![0.3, 0.34]);
        let reason =
            confidence_converged(&refs, &previous, &current, &ConvergenceCriteria::default());
        assert_eq!(reason, Some(ConvergenceReason::ConfidenceDelta));
    }

    #[test]
    fn token_jaccard_identical_strings_is_one() {
        assert!((token_jaccard_similarity("hello world", "Hello, World!") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn token_jaccard_partial_overlap() {
        // tokens a,b,c vs b,c,d -> intersection {b,c}=2, union {a,b,c,d}=4 -> 0.5
        let sim = token_jaccard_similarity("a b c", "b c d");
        assert!((sim - 0.5).abs() < 1e-6);
    }

    #[test]
    fn token_jaccard_disjoint_is_zero() {
        assert!((token_jaccard_similarity("alpha beta", "gamma delta") - 0.0).abs() < 1e-6);
    }

    #[test]
    fn token_jaccard_empty_is_zero() {
        assert!((token_jaccard_similarity("", "") - 0.0).abs() < 1e-6);
        assert!((token_jaccard_similarity("words", "") - 0.0).abs() < 1e-6);
    }

    #[test]
    fn mean_prediction_similarity_mismatched_length_is_zero() {
        let prev = vec!["a".to_string()];
        let cur = vec!["a".to_string(), "b".to_string()];
        assert!((mean_prediction_similarity(&prev, &cur) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn prediction_similarity_converged_disabled_by_default() {
        let prev = vec!["identical".to_string()];
        let cur = vec!["identical".to_string()];
        assert_eq!(
            prediction_similarity_converged(&prev, &cur, &ConvergenceCriteria::default()),
            None,
            "min_prediction_similarity=None must disable the check"
        );
    }

    #[test]
    fn prediction_similarity_converged_above_threshold() {
        let criteria = ConvergenceCriteria {
            min_mean_confidence: 1.0, // force confidence to never fire
            min_prediction_similarity: Some(0.5),
            ..Default::default()
        };
        let prev = vec!["the cat sat".to_string(), "a dog ran".to_string()];
        let cur = vec!["the cat sat".to_string(), "a dog ran".to_string()];
        assert_eq!(
            prediction_similarity_converged(&prev, &cur, &criteria),
            Some(ConvergenceReason::PredictionSimilarity)
        );
    }

    #[test]
    fn prediction_similarity_converged_below_threshold_is_none() {
        let criteria = ConvergenceCriteria {
            min_prediction_similarity: Some(0.9),
            ..Default::default()
        };
        let prev = vec!["alpha beta gamma".to_string()];
        let cur = vec!["delta epsilon zeta".to_string()];
        assert_eq!(
            prediction_similarity_converged(&prev, &cur, &criteria),
            None
        );
    }
}
