//! Cortical mesh orchestration runtime (§3.1.3, §6).
//!
//! [`CorticalMesh`] owns the column population, the lateral connection topology,
//! and a shared inference engine, and drives the three-phase epoch loop:
//!
//! 1. **Parallel feed-forward** — the stimulus is broadcast to every column.
//! 2. **Lateral exchange** — neighbor predictions are injected as context and
//!    columns re-evaluate for up to `max_ticks`, stopping early on convergence.
//! 3. **Global integration** — final per-column outputs are aggregated with mean
//!    confidence and a stabilization flag.
//!
//! Column IDs are processed in sorted order for reproducible consensus, and
//! engine errors fail fast rather than being silently swallowed.

use std::collections::HashMap;
use std::sync::Arc;

use futures::future::join_all;
use ndarray::Array1;
use ptg_consensus::{
    confidence_converged, confidence_vector, mean_confidence, prediction_similarity_converged,
    token_jaccard_similarity, ConvergenceCriteria, ConvergenceReason,
};
use ptg_core::{
    ColumnOutputSchema, CorticalColumn, LateralConnection, Stimulus, TopologyError, TopologySpec,
};
use ptg_vllm::{ColumnEngine, EngineError};

/// Errors raised by mesh construction or execution.
#[derive(Debug, thiserror::Error)]
pub enum MeshError {
    /// A referenced column id does not exist in the mesh.
    #[error("unknown column id: {0}")]
    UnknownColumn(String),
    /// A column with this id was already added.
    #[error("duplicate column id: {0}")]
    DuplicateColumn(String),
    /// An inference engine error occurred while running a column tick.
    #[error("engine error in column {column}: {source}")]
    Engine {
        column: String,
        #[source]
        source: EngineError,
    },
    /// A topology could not be materialized against the supplied columns.
    #[error("topology error: {0}")]
    Topology(#[from] TopologyError),
}

/// How a column selects which neighbors' predictions to inject as lateral
/// context (§9.1 "Dynamic Topology Scaling"). Phase 3B.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum RoutingPolicy {
    /// Inject every neighbor (with a non-empty prediction) equally. The
    /// original V1 behavior and the default.
    #[default]
    All,
    /// Inject only the `k` highest-confidence neighbors (ties broken by source
    /// id ascending). Each selected source's weight is its own confidence.
    ConfidenceTopK { k: usize },
    /// Diversity-preserving (MMR-style) selection: anchor on the highest-
    /// confidence source, then greedily add sources that maximize
    /// `0.5 * confidence + 0.5 * (1 - max token-Jaccard to any already-selected)`.
    /// Up to `k` sources. The hypothesized mitigation for the homogenization
    /// signal: it keeps dissident/niche frames in the context instead of
    /// majority-voting them away. The anchor's weight is its confidence; each
    /// later source's weight is its selection score.
    DiversityPreserving { k: usize },
}

/// How a source's prediction is rendered when injected as lateral context.
/// The structured-lateral disambiguator (see
/// `docs/STRUCTURED_LATERAL_EXPERIMENT.md`) tests whether the *medium* of
/// exchange — not the concept — explains the raw-text echo-leakage.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LateralContextMode {
    /// Inject the neighbor's full free-text prediction verbatim
    /// (`Prediction="<full text>"`). The original V1 behavior and the default.
    #[default]
    Raw,
    /// Inject a bounded "claim excerpt" per source plus an explicit
    /// synthesis directive, instead of the full prediction. The full
    /// prediction is never shared. The excerpt is a substring of the
    /// prediction, so the existing echo screen still catches verbatim
    /// copying unchanged.
    Structured,
}

/// Maximum chars (not bytes) for a structured lateral claim excerpt (~30 words).
const STRUCTURED_CLAIM_MAX_CHARS: usize = 180;

/// One source a listener column hears from on a given tick, with the attention
/// weight the routing policy assigned it.
#[derive(Debug, Clone, PartialEq)]
pub struct RoutedSource {
    pub source_id: String,
    /// Attention weight the policy assigned: `1.0` for `All`, source confidence
    /// for `ConfidenceTopK`, selection score for `DiversityPreserving`.
    pub weight: f32,
    /// The source's own self-reported confidence at the tick it was heard.
    pub confidence: f32,
}

/// A single listener's routing decision for one tick (observability for
/// homogenization experiments; §9.1).
#[derive(Debug, Clone)]
pub struct RouteDecision {
    pub listener_id: String,
    pub sources: Vec<RoutedSource>,
}

/// The lateral context for one listener: the rendered prompt fragment plus the
/// routing decision it was built from.
#[derive(Debug, Clone)]
struct RoutedLateralContext {
    text: String,
    sources: Vec<RoutedSource>,
}

/// The orchestrator tracking column population and topology (§3.1.3, §8.2).
pub struct CorticalMesh {
    /// Columns keyed by id.
    pub columns: HashMap<String, CorticalColumn>,
    /// Directed adjacency: `from_id -> [to_id, ...]`.
    pub adjacency_list: HashMap<String, Vec<String>>,
    /// The shared local inference engine ("thalamus").
    pub engine: Arc<dyn ColumnEngine>,
    /// When to stop the consensus loop.
    pub criteria: ConvergenceCriteria,
    /// How each column selects which neighbors to listen to (§9.1). Default `All`.
    pub routing_policy: RoutingPolicy,
    /// How a source prediction is rendered when injected laterally. Default
    /// `Raw`. See `LateralContextMode` and the structured-lateral experiment.
    pub lateral_context_mode: LateralContextMode,
}

impl CorticalMesh {
    /// Create an empty mesh with default convergence criteria.
    #[must_use]
    pub fn new(engine: Arc<dyn ColumnEngine>) -> Self {
        Self {
            columns: HashMap::new(),
            adjacency_list: HashMap::new(),
            engine,
            criteria: ConvergenceCriteria::default(),
            routing_policy: RoutingPolicy::default(),
            lateral_context_mode: LateralContextMode::default(),
        }
    }

    /// Add a column to the mesh.
    ///
    /// # Errors
    /// [`MeshError::DuplicateColumn`] if the id already exists.
    pub fn add_column(&mut self, col: CorticalColumn) -> Result<(), MeshError> {
        if self.columns.contains_key(&col.id) {
            return Err(MeshError::DuplicateColumn(col.id.clone()));
        }
        self.adjacency_list.entry(col.id.clone()).or_default();
        self.columns.insert(col.id.clone(), col);
        Ok(())
    }

    /// Establish a directed lateral connection `from -> to` (§8.2). Idempotent.
    ///
    /// # Errors
    /// [`MeshError::UnknownColumn`] if either endpoint is unknown.
    pub fn establish_lateral_connection(&mut self, from: &str, to: &str) -> Result<(), MeshError> {
        if !self.columns.contains_key(from) {
            return Err(MeshError::UnknownColumn(from.to_string()));
        }
        if !self.columns.contains_key(to) {
            return Err(MeshError::UnknownColumn(to.to_string()));
        }
        if let Some(neighbors) = self.adjacency_list.get_mut(from) {
            if !neighbors.iter().any(|n| n == to) {
                neighbors.push(to.to_string());
            }
        }
        Ok(())
    }

    /// Return the columns a given column is connected to.
    ///
    /// # Errors
    /// [`MeshError::UnknownColumn`] if the id is unknown.
    pub fn get_neighbors(&self, id: &str) -> Result<Vec<CorticalColumn>, MeshError> {
        let Some(neighbor_ids) = self.adjacency_list.get(id) else {
            return Err(MeshError::UnknownColumn(id.to_string()));
        };
        Ok(neighbor_ids
            .iter()
            .filter_map(|nid| self.columns.get(nid).cloned())
            .collect())
    }

    /// Column ids in deterministic (sorted) order.
    fn column_ids_sorted(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.columns.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Build the injected lateral context for a column from its neighbors'
    /// most recent predictions (§6 Phase 2), selecting/weighting sources per
    /// [`routing_policy`](CorticalMesh::routing_policy) (§9.1). Returns the
    /// rendered prompt fragment and the routing decision. Both are empty until
    /// neighbors have produced a prediction.
    fn lateral_context_for(&self, id: &str) -> RoutedLateralContext {
        let sources = self.select_lateral_sources(id);
        if sources.is_empty() {
            return RoutedLateralContext {
                text: String::new(),
                sources,
            };
        }
        let mut lines = Vec::with_capacity(sources.len());
        for s in &sources {
            let pred = self
                .columns
                .get(&s.source_id)
                .map(|c| c.last_prediction.as_str())
                .unwrap_or_default();
            match self.lateral_context_mode {
                LateralContextMode::Raw => {
                    let line = match self.routing_policy {
                        RoutingPolicy::All => format!(
                            "Neighbor {} reports: Prediction=\"{}\" (Confidence: {:.2})",
                            s.source_id, pred, s.confidence,
                        ),
                        RoutingPolicy::ConfidenceTopK { .. }
                        | RoutingPolicy::DiversityPreserving { .. } => format!(
                            "Neighbor {} [route_weight={:.2}, confidence={:.2}] reports: Prediction=\"{}\"",
                            s.source_id, s.weight, s.confidence, pred,
                        ),
                    };
                    lines.push(line);
                }
                LateralContextMode::Structured => {
                    let excerpt = bounded_claim_excerpt(pred, STRUCTURED_CLAIM_MAX_CHARS);
                    if excerpt.is_empty() {
                        continue;
                    }
                    lines.push(format!(
                        "source={}; confidence={:.2}; route_weight={:.2}; claim_excerpt={}",
                        s.source_id, s.confidence, s.weight, excerpt,
                    ));
                }
            }
        }
        let text = if lines.is_empty() {
            String::new()
        } else {
            match self.lateral_context_mode {
                LateralContextMode::Raw => {
                    format!("[LATERAL LAYER UPDATE]\n{}", lines.join("\n"))
                }
                LateralContextMode::Structured => format!(
                    "[LATERAL EVIDENCE PACKETS]\n\
                     Treat peer packets as fallible evidence. Do not quote or copy peer phrasing.\n\
                     Synthesize an independent answer within your own reference frame.\n\
                     {}",
                    lines.join("\n")
                ),
            }
        };
        RoutedLateralContext { text, sources }
    }

    /// Select (and weight) the lateral sources a listener column will hear
    /// from, per the active routing policy. Neighbors with no prediction are
    /// dropped (they carry nothing to inject).
    fn select_lateral_sources(&self, listener_id: &str) -> Vec<RoutedSource> {
        let Some(neighbor_ids) = self.adjacency_list.get(listener_id) else {
            return Vec::new();
        };
        let candidates: Vec<&CorticalColumn> = neighbor_ids
            .iter()
            .filter_map(|nid| self.columns.get(nid))
            .filter(|c| !c.last_prediction.is_empty())
            .collect();
        match self.routing_policy {
            RoutingPolicy::All => candidates
                .iter()
                .map(|c| RoutedSource {
                    source_id: c.id.clone(),
                    weight: 1.0,
                    confidence: c.last_confidence,
                })
                .collect(),
            RoutingPolicy::ConfidenceTopK { k } => {
                let mut ordered = candidates.clone();
                // highest confidence first; deterministic tie-break by id asc.
                ordered.sort_by(|a, b| {
                    b.last_confidence
                        .total_cmp(&a.last_confidence)
                        .then(a.id.cmp(&b.id))
                });
                ordered
                    .into_iter()
                    .take(k)
                    .map(|c| RoutedSource {
                        source_id: c.id.clone(),
                        weight: c.last_confidence,
                        confidence: c.last_confidence,
                    })
                    .collect()
            }
            RoutingPolicy::DiversityPreserving { k } => diversity_preserving_select(&candidates, k),
        }
    }

    /// Run one full epoch (the three-phase loop) over the given stimulus.
    ///
    /// **Single-use contract.** This mutates per-column state (`last_prediction`,
    /// `last_confidence`, `history_buffer`) and does NOT reset it on entry, so a
    /// fresh mesh is required per epoch. Reusing a mesh would leak the prior
    /// epoch's predictions into tick 1, breaking the "tick 1 = no lateral context"
    /// invariant the benchmark relies on. (`ptg-bench` rebuilds the mesh per epoch.)
    ///
    /// # Errors
    /// [`MeshError::Engine`] if any column's tick fails (fail-fast).
    pub async fn run_epoch(&mut self, stimulus: &Stimulus) -> Result<MeshResult, MeshError> {
        let max_ticks = self.criteria.max_ticks;
        let engine = Arc::clone(&self.engine);
        let mut previous_conf: Array1<f32> = Array1::zeros(0);
        let mut previous_predictions: Vec<String> = Vec::new();
        let mut last_outputs: Vec<(String, ColumnOutputSchema)> = Vec::new();
        let mut tick_outputs: Vec<TickOutputs> = Vec::new();
        let mut ticks_run = 0u32;
        let mut stabilized = false;
        let mut convergence_reason: Option<ConvergenceReason> = None;

        for tick in 1..=max_ticks {
            ticks_run = tick;

            // Phase 1 + 2 prep: gather deterministic per-column context.
            let mut tasks: Vec<(String, CorticalColumn, String)> =
                Vec::with_capacity(self.columns.len());
            let mut tick_routes: Vec<RouteDecision> = Vec::new();
            for id in &self.column_ids_sorted() {
                let Some(col) = self.columns.get(id) else {
                    continue;
                };
                let col = col.clone();
                let routed = self.lateral_context_for(id);
                if !routed.sources.is_empty() {
                    tick_routes.push(RouteDecision {
                        listener_id: id.clone(),
                        sources: routed.sources,
                    });
                }
                tasks.push((id.clone(), col, routed.text));
            }

            // Phase 1 + 2 exec: concurrent column ticks against the shared engine.
            let futures = tasks.into_iter().map(|(id, col, ctx)| {
                let engine = Arc::clone(&engine);
                let stimulus = stimulus.clone();
                async move {
                    let result = engine.execute_column_tick(&col, &stimulus, &ctx).await;
                    (id, result)
                }
            });
            let results = join_all(futures).await;

            // Phase 3 prep: apply results, fail-fast on engine errors.
            last_outputs.clear();
            let mut failure: Option<MeshError> = None;
            for (id, result) in results {
                match result {
                    Ok(schema) => {
                        if let Some(col) = self.columns.get_mut(&id) {
                            col.record_tick(tick, &schema);
                        }
                        last_outputs.push((id, schema));
                    }
                    Err(source) => {
                        if failure.is_none() {
                            failure = Some(MeshError::Engine { column: id, source });
                        }
                    }
                }
            }
            if let Some(err) = failure {
                return Err(err);
            }

            // Capture this tick's per-column outputs BEFORE convergence can
            // short-circuit, so every executed tick is observable downstream.
            // This enables within-run mechanism comparisons (e.g. tick 1 with
            // no lateral context vs tick 2 with lateral context).
            tick_outputs.push(TickOutputs {
                tick,
                outputs: last_outputs.clone(),
                routes: tick_routes,
            });

            // Phase 3: convergence check. Only consider convergence once we've
            // run at least `min_ticks` ticks, so an overconfident model can't
            // short-circuit the lateral-voting mechanism on tick 1. Confidence
            // criteria are checked first, then the model-independent
            // prediction-stability (token-Jaccard) criterion if enabled.
            let refs: Vec<&CorticalColumn> = self.column_ids_sorted_local_refs();
            let current = confidence_vector(&refs);
            let current_predictions: Vec<String> =
                refs.iter().map(|c| c.last_prediction.clone()).collect();
            if tick >= self.criteria.min_ticks {
                let reason = confidence_converged(&refs, &previous_conf, &current, &self.criteria)
                    .or_else(|| {
                        prediction_similarity_converged(
                            &previous_predictions,
                            &current_predictions,
                            &self.criteria,
                        )
                    });
                if let Some(reason) = reason {
                    stabilized = true;
                    convergence_reason = Some(reason);
                    break;
                }
            }
            previous_conf = current;
            previous_predictions = current_predictions;
        }

        let refs: Vec<&CorticalColumn> = self.column_ids_sorted_local_refs();
        let mean = mean_confidence(&refs);
        let threshold = self.criteria.min_integration_confidence;
        let (accepted_outputs, rejected_outputs) = last_outputs
            .iter()
            .cloned()
            .partition::<Vec<_>, _>(|(_, out)| out.confidence >= threshold);
        Ok(MeshResult {
            ticks_run,
            outputs: last_outputs,
            tick_outputs,
            accepted_outputs,
            rejected_outputs,
            mean_confidence: mean,
            stabilized,
            convergence_reason,
        })
    }

    /// Convenience: run an epoch over a text-only stimulus.
    ///
    /// # Errors
    /// [`MeshError::Engine`] if any column's tick fails (fail-fast).
    pub async fn run_text_epoch(&mut self, input: &str) -> Result<MeshResult, MeshError> {
        self.run_epoch(&Stimulus::text(input)).await
    }

    /// Borrow the columns in sorted-id order (helper to satisfy borrow checker).
    fn column_ids_sorted_local_refs(&self) -> Vec<&CorticalColumn> {
        self.column_ids_sorted()
            .into_iter()
            .filter_map(|id| self.columns.get(&id))
            .collect()
    }
}

/// MMR-style diversity-preserving source selection (§9.1). Anchor on the
/// highest-confidence candidate, then greedily add candidates maximizing
/// `0.5 * confidence + 0.5 * (1 - max token-Jaccard to any already-selected)`.
/// Tie-breaks deterministically: score desc, confidence desc, id asc. Returns up
/// to `k` sources (fewer if there are not enough candidates).
fn diversity_preserving_select(candidates: &[&CorticalColumn], k: usize) -> Vec<RoutedSource> {
    if k == 0 || candidates.is_empty() {
        return Vec::new();
    }
    // Owned (id, prediction, confidence) triples for borrow-free selection.
    let mut pool: Vec<(String, String, f32)> = candidates
        .iter()
        .map(|c| (c.id.clone(), c.last_prediction.clone(), c.last_confidence))
        .collect();
    // Anchor priority: confidence desc, id asc.
    pool.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
    let Some((first, rest)) = pool.split_first() else {
        return Vec::new();
    };
    let (aid, apred, aconf) = first.clone();
    let mut selected_preds: Vec<String> = vec![apred];
    let mut result: Vec<RoutedSource> = vec![RoutedSource {
        source_id: aid,
        weight: aconf,
        confidence: aconf,
    }];
    let mut remaining: Vec<(String, String, f32)> = rest.to_vec();
    while result.len() < k && !remaining.is_empty() {
        let mut scored: Vec<(f32, String, String, f32)> = remaining
            .iter()
            .map(|(id, pred, conf)| {
                let max_sim = selected_preds
                    .iter()
                    .map(|sp| token_jaccard_similarity(pred, sp))
                    .fold(0.0f32, f32::max);
                let score = 0.5 * conf + 0.5 * (1.0 - max_sim);
                (score, id.clone(), pred.clone(), *conf)
            })
            .collect();
        // score desc, confidence desc, id asc.
        scored.sort_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then(b.3.total_cmp(&a.3))
                .then(a.1.cmp(&b.1))
        });
        if let Some((score, wid, wpred, wconf)) = scored.into_iter().next() {
            result.push(RoutedSource {
                source_id: wid.clone(),
                weight: score,
                confidence: wconf,
            });
            selected_preds.push(wpred);
            remaining.retain(|(id, _, _)| *id != wid);
        }
    }
    result
}

/// Bound a source prediction to a terse "claim excerpt" for structured
/// lateral injection (`LateralContextMode::Structured`). Whitespace is
/// normalized, then the text is truncated **char-safely** to the first sentence
/// boundary within `max_chars`, or — if no sentence boundary is found — to the
/// last word boundary. An ellipsis marks mid-text truncation; sentence-complete
/// excerpts get none. Never panics on multibyte UTF-8. Empty/whitespace-only
/// input yields an empty string.
#[must_use]
fn bounded_claim_excerpt(prediction: &str, max_chars: usize) -> String {
    let normalized: String = prediction.split_whitespace().collect::<Vec<_>>().join(" ");
    let chars: Vec<char> = normalized.chars().collect();
    if chars.len() <= max_chars {
        return normalized;
    }
    // Within the window, track the last sentence-ending punctuation.
    let mut cut = 0;
    let mut hit_sentence = false;
    for (i, &c) in chars.iter().take(max_chars).enumerate() {
        if c == '.' || c == '!' || c == '?' {
            cut = i + 1;
            hit_sentence = true;
        }
    }
    if !hit_sentence {
        // No sentence boundary in the window: trim back to the last space to
        // avoid a mid-word cut. If there is no space, hard-cut at max_chars.
        cut = chars
            .iter()
            .take(max_chars)
            .rposition(|&c| c == ' ')
            .unwrap_or(max_chars);
    }
    let excerpt: String = chars.iter().take(cut).collect();
    let trimmed = excerpt.trim_end();
    if trimmed.is_empty() {
        return String::new();
    }
    if hit_sentence {
        trimmed.to_string()
    } else {
        format!("{trimmed} …")
    }
}

/// The per-column outputs captured at a single executed tick.
#[derive(Debug, Clone)]
pub struct TickOutputs {
    /// 1-indexed tick number this snapshot is from.
    pub tick: u32,
    /// Per-column outputs at this tick (all columns, sorted-id order).
    pub outputs: Vec<(String, ColumnOutputSchema)>,
    /// Lateral routing decisions used to build THIS tick's prompts (§9.1).
    /// Empty on tick 1 (no prior predictions to route) and for any listener
    /// that selected no sources.
    pub routes: Vec<RouteDecision>,
}

/// Result of one completed epoch.
#[derive(Debug, Clone)]
pub struct MeshResult {
    /// Number of ticks actually executed (may be less than `max_ticks`).
    pub ticks_run: u32,
    /// Final per-column outputs from the last executed tick (all columns).
    pub outputs: Vec<(String, ColumnOutputSchema)>,
    /// Snapshot of per-column outputs at every executed tick (tick 1..ticks_run),
    /// in execution order. Enables within-run comparisons (e.g. tick 1 with no
    /// lateral context vs tick 2 with lateral context).
    pub tick_outputs: Vec<TickOutputs>,
    /// Outputs whose confidence met the integration threshold (§6 Phase 3).
    pub accepted_outputs: Vec<(String, ColumnOutputSchema)>,
    /// Outputs filtered out of the global percept for low confidence.
    pub rejected_outputs: Vec<(String, ColumnOutputSchema)>,
    /// Mean confidence across columns at the end of the epoch.
    pub mean_confidence: f32,
    /// Whether a quality convergence criterion was met (vs. running out of ticks).
    pub stabilized: bool,
    /// Which quality criterion terminated the epoch, if any. `None` when the
    /// epoch ran to `max_ticks` without stabilizing.
    pub convergence_reason: Option<ConvergenceReason>,
}

/// Build the default reference mesh (§8.4) from the shared engine.
///
/// # Errors
/// Propagates [`MeshError`] from column/connection registration (should not
/// occur for the hardcoded defaults).
pub fn default_mesh(engine: Arc<dyn ColumnEngine>) -> Result<CorticalMesh, MeshError> {
    let mut mesh = CorticalMesh::new(engine);
    for col in ptg_core::default_columns() {
        mesh.add_column(col)?;
    }
    for (from, to) in ptg_core::default_connections() {
        mesh.establish_lateral_connection(from, to)?;
    }
    Ok(mesh)
}

/// Build a mesh from an explicit column population and a caller-supplied edge
/// list. Each `LateralConnection` says the `listener_id` receives the
/// `source_id`'s prediction (see `establish_lateral_connection`). Column ids in
/// the edge list must already have been added via the `columns` slice; unknown
/// ids surface as [`MeshError::UnknownColumn`].
///
/// # Errors
/// [`MeshError::UnknownColumn`] if an edge references an id not in `columns`.
pub fn mesh_from_columns(
    engine: Arc<dyn ColumnEngine>,
    columns: Vec<CorticalColumn>,
    connections: impl IntoIterator<Item = LateralConnection>,
) -> Result<CorticalMesh, MeshError> {
    let mut mesh = CorticalMesh::new(engine);
    for col in columns {
        mesh.add_column(col)?;
    }
    for conn in connections {
        mesh.establish_lateral_connection(&conn.listener_id, &conn.source_id)?;
    }
    Ok(mesh)
}

/// Build a mesh from an explicit column population and a declarative topology.
/// The topology is materialized against the ordered column-id list (taken from
/// `columns` in iteration order), so positional variants (ring/torus/small-
/// world) map onto the columns positionally. This is the primary Phase 3 entry
/// point for pluggable topologies (§3.1.3).
///
/// # Errors
/// [`MeshError::Topology`] if the topology does not fit the column count, or
/// [`MeshError::UnknownColumn`] on an internal id mismatch.
pub fn mesh_with_topology(
    engine: Arc<dyn ColumnEngine>,
    columns: Vec<CorticalColumn>,
    topology: &TopologySpec,
) -> Result<CorticalMesh, MeshError> {
    let ids: Vec<String> = columns.iter().map(|c| c.id.clone()).collect();
    let connections = topology.connections_for(&ids)?;
    mesh_from_columns(engine, columns, connections)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use ptg_core::DomainSphere;
    use ptg_vllm::EngineError as VllmEngineError;

    /// Deterministic mock engine whose prediction tracks its sphere + tick count,
    /// enabling convergence tests without a live server.
    struct MockEngine;

    #[async_trait]
    impl ColumnEngine for MockEngine {
        async fn execute_column_tick(
            &self,
            column: &CorticalColumn,
            _stimulus: &ptg_core::Stimulus,
            _lateral: &str,
        ) -> Result<ColumnOutputSchema, VllmEngineError> {
            let prediction = format!("{}-prediction", column.sphere.as_str());
            Ok(ColumnOutputSchema {
                reference_frame_coordinates: format!("coord-{}", column.id),
                prediction,
                confidence: 0.95, // above default min_mean_confidence -> converges tick 1
                domain_fields: std::collections::BTreeMap::new(),
            })
        }
    }

    fn mesh_with(columns: Vec<CorticalColumn>) -> CorticalMesh {
        let mut mesh = CorticalMesh::new(Arc::new(MockEngine));
        for c in columns {
            assert!(mesh.add_column(c).is_ok(), "test setup: add_column");
        }
        mesh
    }

    #[test]
    fn rejects_duplicate_column() {
        let mut mesh = CorticalMesh::new(Arc::new(MockEngine));
        assert!(mesh
            .add_column(CorticalColumn::with_defaults("CC_A", DomainSphere::Physics))
            .is_ok());
        let second = mesh.add_column(CorticalColumn::with_defaults("CC_A", DomainSphere::Physics));
        assert!(matches!(second, Err(MeshError::DuplicateColumn(_))));
    }

    #[test]
    fn rejects_connection_to_unknown_column() {
        let mut mesh = mesh_with(vec![CorticalColumn::with_defaults(
            "CC_A",
            DomainSphere::Physics,
        )]);
        let result = mesh.establish_lateral_connection("CC_A", "CC_GHOST");
        assert!(matches!(result, Err(MeshError::UnknownColumn(_))));
    }

    #[test]
    fn get_neighbors_returns_connected_columns() -> Result<(), Box<dyn std::error::Error>> {
        let mut mesh = mesh_with(vec![
            CorticalColumn::with_defaults("CC_A", DomainSphere::Physics),
            CorticalColumn::with_defaults("CC_B", DomainSphere::Mathematics),
        ]);
        mesh.establish_lateral_connection("CC_A", "CC_B")?;
        let neighbors = mesh.get_neighbors("CC_A")?;
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].id, "CC_B");
        Ok(())
    }

    #[tokio::test]
    async fn run_epoch_converges_in_one_tick_with_mock() -> Result<(), Box<dyn std::error::Error>> {
        let mut mesh = mesh_with(ptg_core::default_columns());
        for (from, to) in ptg_core::default_connections() {
            mesh.establish_lateral_connection(from, to)?;
        }
        let result = mesh.run_text_epoch("stimulus").await?;
        assert_eq!(result.outputs.len(), 4);
        assert!(result.stabilized, "high-confidence mock should converge");
        assert_eq!(result.ticks_run, 1);
        assert!(result.mean_confidence >= 0.8);
        assert_eq!(result.accepted_outputs.len(), 4, "all high-conf accepted");
        assert!(result.rejected_outputs.is_empty());
        assert_eq!(
            result.convergence_reason,
            Some(ConvergenceReason::MeanConfidence)
        );
        Ok(())
    }

    /// Low-confidence constant engine: never converges by mean confidence, but
    /// the confidence vector stops changing, so it converges via the delta
    /// criterion on the second tick — exercising the multi-tick lateral path.
    struct LowConfEngine;

    #[async_trait]
    impl ColumnEngine for LowConfEngine {
        async fn execute_column_tick(
            &self,
            column: &CorticalColumn,
            _stimulus: &ptg_core::Stimulus,
            _lateral: &str,
        ) -> Result<ColumnOutputSchema, VllmEngineError> {
            Ok(ColumnOutputSchema {
                reference_frame_coordinates: format!("coord-{}", column.id),
                prediction: format!("{}-pred", column.sphere.as_str()),
                confidence: 0.3,
                domain_fields: std::collections::BTreeMap::new(),
            })
        }
    }

    #[tokio::test]
    async fn run_epoch_converges_via_delta_on_second_tick() -> Result<(), Box<dyn std::error::Error>>
    {
        let mut mesh = default_mesh(Arc::new(LowConfEngine))?;
        mesh.criteria.max_ticks = 3;
        let result = mesh.run_text_epoch("stimulus").await?;
        assert_eq!(result.outputs.len(), 4);
        assert_eq!(result.ticks_run, 2, "should converge via delta on tick 2");
        assert!(result.stabilized);
        assert_eq!(
            result.accepted_outputs.len(),
            0,
            "0.3 conf below default 0.5 threshold"
        );
        assert_eq!(result.rejected_outputs.len(), 4);
        for (_id, out) in &result.outputs {
            assert!(out.prediction.ends_with("-pred"));
            assert!((0.0..=1.0).contains(&out.confidence));
        }
        assert_eq!(
            result.convergence_reason,
            Some(ConvergenceReason::ConfidenceDelta)
        );
        Ok(())
    }

    /// Constant-prediction, phase-shifted-confidence engine: each column's
    /// confidence rotates through a distinct phase per tick, so the confidence
    /// vector's *direction* keeps changing (no mean/delta/cosine criterion can
    /// fire) while predictions are identical every tick — leaving the
    /// token-Jaccard criterion as the only thing that can converge.
    struct PredictionStableEngine;

    #[async_trait]
    impl ColumnEngine for PredictionStableEngine {
        async fn execute_column_tick(
            &self,
            column: &CorticalColumn,
            _stimulus: &ptg_core::Stimulus,
            _lateral: &str,
        ) -> Result<ColumnOutputSchema, VllmEngineError> {
            // Per-sphere phase so the confidence vector genuinely rotates
            // direction tick to tick (a uniform vector would trivially hit
            // cosine similarity 1.0). Predictions stay constant so the
            // token-Jaccard of successive ticks is 1.0.
            let phase: u32 = match column.sphere.as_str() {
                "Physics" => 0,
                "Mathematics" => 1,
                "Coding" => 2,
                _ => 3,
            };
            let confidence = 0.2 + 0.15 * ((column.history_buffer.len() as u32 + phase) % 4) as f32;
            Ok(ColumnOutputSchema {
                reference_frame_coordinates: format!("coord-{}", column.id),
                prediction: format!("{}-stable-prediction-text", column.sphere.as_str()),
                confidence,
                domain_fields: std::collections::BTreeMap::new(),
            })
        }
    }

    #[tokio::test]
    async fn run_epoch_converges_via_prediction_similarity(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut mesh = default_mesh(Arc::new(PredictionStableEngine))?;
        // Default confidence thresholds never fire: mean stays ~0.43 < 0.8 and
        // the confidence vector keeps rotating direction.
        mesh.criteria.max_ticks = 5;
        // Prediction-stability criterion enabled.
        mesh.criteria.min_prediction_similarity = Some(0.5);
        let result = mesh.run_text_epoch("stimulus").await?;
        assert_eq!(
            result.ticks_run, 2,
            "tick 2 is the first with a previous prediction"
        );
        assert!(result.stabilized);
        assert_eq!(
            result.convergence_reason,
            Some(ConvergenceReason::PredictionSimilarity)
        );
        Ok(())
    }

    #[tokio::test]
    async fn prediction_similarity_disabled_by_default_is_inert(
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Same engine, but min_prediction_similarity stays None (default):
        // nothing converges, the epoch runs to max_ticks without stabilizing.
        let mut mesh = default_mesh(Arc::new(PredictionStableEngine))?;
        mesh.criteria.max_ticks = 2;
        let result = mesh.run_text_epoch("stimulus").await?;
        assert!(!result.stabilized);
        assert_eq!(result.convergence_reason, None);
        Ok(())
    }

    fn source_ids(mesh: &CorticalMesh, id: &str) -> Result<Vec<String>, MeshError> {
        Ok(mesh.get_neighbors(id)?.into_iter().map(|c| c.id).collect())
    }

    #[test]
    fn default_mesh_wiring_is_unchanged() -> Result<(), Box<dyn std::error::Error>> {
        // Regression guard: the named 4-column topology must keep its
        // hand-coded edges (PHYSICS listens to MATH; MATH to PHYSICS+CODE;
        // CODE to PSYCH; PSYCH to none — the graph sink).
        let mesh = default_mesh(Arc::new(LowConfEngine))?;
        let mut phys = source_ids(&mesh, "CC_PHYSICS_01")?;
        phys.sort();
        assert_eq!(phys, vec!["CC_MATH_01"]);
        let mut math = source_ids(&mesh, "CC_MATH_01")?;
        math.sort();
        assert_eq!(math, vec!["CC_CODE_01", "CC_PHYSICS_01"]);
        let code = source_ids(&mesh, "CC_CODE_01")?;
        assert_eq!(code, vec!["CC_PSYCH_01"]);
        let psych = source_ids(&mesh, "CC_PSYCH_01")?;
        assert!(psych.is_empty(), "PSYCH is the graph sink");
        Ok(())
    }

    #[test]
    fn mesh_from_columns_wires_explicit_edges() -> Result<(), Box<dyn std::error::Error>> {
        let cols = ptg_core::replicated_default_columns(3);
        let conns = vec![
            LateralConnection::new("CC_PHYSICS_01", "CC_MATH_01"),
            LateralConnection::new("CC_PHYSICS_01", "CC_CODE_01"),
        ];
        let mesh = mesh_from_columns(Arc::new(LowConfEngine), cols, conns)?;
        let mut s = source_ids(&mesh, "CC_PHYSICS_01")?;
        s.sort();
        assert_eq!(s, vec!["CC_CODE_01", "CC_MATH_01"]);
        assert!(source_ids(&mesh, "CC_MATH_01")?.is_empty());
        Ok(())
    }

    #[test]
    fn mesh_with_topology_ring_wires_predecessor() -> Result<(), Box<dyn std::error::Error>> {
        // 5 columns round-robin; unidirectional ring → each listens to its
        // predecessor in id order.
        let cols = ptg_core::replicated_default_columns(5);
        let ids: Vec<String> = cols.iter().map(|c| c.id.clone()).collect();
        let mesh = mesh_with_topology(
            Arc::new(LowConfEngine),
            cols,
            &TopologySpec::Ring {
                bidirectional: false,
            },
        )?;
        assert_eq!(source_ids(&mesh, &ids[0])?, vec![ids[4].clone()]);
        assert_eq!(source_ids(&mesh, &ids[1])?, vec![ids[0].clone()]);
        Ok(())
    }

    #[test]
    fn mesh_with_topology_torus_gives_four_neighbors() -> Result<(), Box<dyn std::error::Error>> {
        let cols = ptg_core::replicated_default_columns(9);
        let mesh = mesh_with_topology(
            Arc::new(LowConfEngine),
            cols,
            &TopologySpec::Torus2d {
                width: 3,
                height: 3,
            },
        )?;
        for col in mesh.columns.values() {
            assert_eq!(
                source_ids(&mesh, &col.id)?.len(),
                4,
                "every torus node has four cardinal sources"
            );
        }
        Ok(())
    }

    #[test]
    fn mesh_with_topology_fully_connected() -> Result<(), Box<dyn std::error::Error>> {
        let cols = ptg_core::replicated_default_columns(4);
        let mesh =
            mesh_with_topology(Arc::new(LowConfEngine), cols, &TopologySpec::FullyConnected)?;
        for col in mesh.columns.values() {
            assert_eq!(source_ids(&mesh, &col.id)?.len(), 3);
        }
        Ok(())
    }

    #[test]
    fn mesh_with_topology_propagates_geometry_error() {
        // 8 columns cannot form a 3x3 torus.
        let cols = ptg_core::replicated_default_columns(8);
        let res = mesh_with_topology(
            Arc::new(LowConfEngine),
            cols,
            &TopologySpec::Torus2d {
                width: 3,
                height: 3,
            },
        );
        assert!(matches!(res, Err(MeshError::Topology(_))));
    }

    // ---- Phase 3B: lateral routing (§9.1) ----

    /// Build a single-listener mesh whose neighbors carry explicit
    /// (prediction, confidence) state, for routing-selection unit tests.
    fn routed_mesh(
        listener: &str,
        neighbors: &[(&str, &str, f32)],
    ) -> Result<CorticalMesh, Box<dyn std::error::Error>> {
        let mut mesh = CorticalMesh::new(Arc::new(LowConfEngine));
        mesh.add_column(CorticalColumn::with_defaults(
            listener,
            DomainSphere::Physics,
        ))?;
        for (id, _, _) in neighbors {
            mesh.add_column(CorticalColumn::with_defaults(id, DomainSphere::Mathematics))?;
            mesh.establish_lateral_connection(listener, id)?;
        }
        for (id, pred, conf) in neighbors {
            if let Some(c) = mesh.columns.get_mut(*id) {
                c.last_prediction = (*pred).to_string();
                c.last_confidence = *conf;
            }
        }
        Ok(mesh)
    }

    fn routed_source_ids(sources: &[RoutedSource]) -> Vec<String> {
        let mut v: Vec<String> = sources.iter().map(|s| s.source_id.clone()).collect();
        v.sort();
        v
    }

    #[test]
    fn all_routing_selects_all_non_empty_neighbors() -> Result<(), Box<dyn std::error::Error>> {
        let mut mesh = routed_mesh(
            "CC_L",
            &[
                ("CC_A", "pred a", 0.5),
                ("CC_B", "pred b", 0.9),
                ("CC_C", "", 0.99),
            ],
        )?;
        mesh.routing_policy = RoutingPolicy::All;
        let sources = mesh.select_lateral_sources("CC_L");
        // CC_C has an empty prediction -> dropped.
        assert_eq!(
            routed_source_ids(&sources),
            vec!["CC_A".to_string(), "CC_B".to_string()]
        );
        for s in &sources {
            assert!((s.weight - 1.0).abs() < 1e-6, "All weights are 1.0");
        }
        Ok(())
    }

    #[test]
    fn confidence_top_k_selects_highest_confidence_with_deterministic_ties(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut mesh = routed_mesh(
            "CC_L",
            &[
                ("CC_B", "pred b", 0.8),
                ("CC_A", "pred a", 0.8),
                ("CC_C", "pred c", 0.5),
            ],
        )?;
        mesh.routing_policy = RoutingPolicy::ConfidenceTopK { k: 2 };
        let sources = mesh.select_lateral_sources("CC_L");
        // tie at 0.8 broken by id asc -> A, B; C excluded.
        assert_eq!(
            routed_source_ids(&sources),
            vec!["CC_A".to_string(), "CC_B".to_string()]
        );
        assert!(sources
            .iter()
            .all(|s| (s.weight - s.confidence).abs() < 1e-6));
        Ok(())
    }

    #[test]
    fn diversity_preserving_prefers_dissimilar_over_redundant_high_conf(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut mesh = routed_mesh(
            "CC_L",
            &[
                ("CC_ANCHOR", "alpha beta gamma", 0.9),
                ("CC_REDUNDANT", "alpha beta gamma delta", 0.85),
                ("CC_NOVEL", "completely different tokens zeta", 0.6),
            ],
        )?;
        mesh.routing_policy = RoutingPolicy::DiversityPreserving { k: 2 };
        let sources = mesh.select_lateral_sources("CC_L");
        let ids = routed_source_ids(&sources);
        assert_eq!(ids.len(), 2);
        assert!(
            ids.contains(&"CC_ANCHOR".to_string()),
            "highest-conf anchor always selected"
        );
        assert!(
            ids.contains(&"CC_NOVEL".to_string()),
            "dissimilar source preferred"
        );
        assert!(
            !ids.contains(&"CC_REDUNDANT".to_string()),
            "redundant high-conf dropped"
        );
        Ok(())
    }

    #[test]
    fn k_zero_selects_no_sources() -> Result<(), Box<dyn std::error::Error>> {
        let mut mesh = routed_mesh("CC_L", &[("CC_A", "pred a", 0.9)])?;
        mesh.routing_policy = RoutingPolicy::DiversityPreserving { k: 0 };
        assert!(mesh.select_lateral_sources("CC_L").is_empty());
        mesh.routing_policy = RoutingPolicy::ConfidenceTopK { k: 0 };
        assert!(mesh.select_lateral_sources("CC_L").is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn run_epoch_captures_route_decisions_for_tick_2(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut mesh = default_mesh(Arc::new(MockEngine))?;
        mesh.criteria.min_ticks = 2; // force 2 ticks so lateral exchange runs
        mesh.criteria.max_ticks = 2;
        let result = mesh.run_text_epoch("stimulus").await?;
        assert_eq!(result.tick_outputs.len(), 2, "ran 2 ticks");
        assert!(
            result.tick_outputs[0].routes.is_empty(),
            "tick 1 has no prior predictions"
        );
        let tick2 = &result.tick_outputs[1];
        // Default topology has 3 lateral receivers (PHYSICS, MATH, CODE); PSYCH is a sink.
        assert_eq!(tick2.routes.len(), 3, "three receivers route on tick 2");
        for dec in &tick2.routes {
            assert!(!dec.sources.is_empty(), "each receiver selected >=1 source");
        }
        Ok(())
    }

    // --- Structured lateral exchange (STRUCTURED_LATERAL_EXPERIMENT) ----------

    #[test]
    fn bounded_claim_excerpt_short_text_unchanged() {
        assert_eq!(bounded_claim_excerpt("short answer", 180), "short answer");
    }

    #[test]
    fn bounded_claim_excerpt_truncates_at_sentence_boundary() {
        let pred = "First sentence ends here. This is a very long continuation that \
                   should not appear in the excerpt because we already hit a boundary.";
        let out = bounded_claim_excerpt(pred, 40);
        assert_eq!(out, "First sentence ends here.");
    }

    #[test]
    fn bounded_claim_excerpt_truncates_at_word_boundary_with_ellipsis() {
        // No sentence punctuation within the window → word-boundary cut + ellipsis.
        let pred = "alpha beta gamma delta epsilon zeta eta theta iota kappa";
        let out = bounded_claim_excerpt(pred, 30);
        assert!(out.ends_with('…'), "ends with ellipsis: {out}");
        assert!(!out.contains("kappa"), "does not reach the end: {out}");
        // char-safe: never panics, returns something non-empty.
        assert!(!out.is_empty());
    }

    #[test]
    fn bounded_claim_excerpt_is_char_safe_on_multibyte() {
        // Em-dashes and CJK are multibyte in UTF-8; byte slicing would panic.
        let pred = "能量守恒定律指出 — energy is conserved — and cannot be created or \
                   destroyed in an isolated system ever under any circumstances at all.";
        let out = bounded_claim_excerpt(pred, 20);
        // Must not panic and must be <= the char window (plus possible ellipsis).
        assert!(out.chars().count() <= 22, "respect char budget: {out}");
        assert!(!out.is_empty());
    }

    #[test]
    fn bounded_claim_excerpt_empty_input_yields_empty() {
        assert_eq!(bounded_claim_excerpt("   \t\n ", 180), "");
    }

    /// Build a mesh whose source column has a known `last_prediction`, so we can
    /// assert on the rendered lateral context without running a live tick.
    fn mesh_with_seeded_prediction() -> Result<CorticalMesh, MeshError> {
        let mut mesh = default_mesh(Arc::new(MockEngine))?;
        // Seed MATH's prediction so PHYSICS (which listens to MATH) has a source.
        // Must exceed STRUCTURED_CLAIM_MAX_CHARS (180) so truncation engages; the
        // first sentence is the claim, the tail (with "elaboration") is discarded.
        if let Some(math) = mesh.columns.get_mut("CC_MATH_01") {
            math.last_prediction = "The system reaches thermal equilibrium at 300K. \
                This continuation is deliberately very long so that the total character \
                count exceeds one hundred and eighty characters and forces the truncation \
                logic to engage properly, with the word elaboration appearing only in \
                the discarded tail portion of the prediction text."
                .to_string();
            math.last_confidence = 0.82;
        }
        Ok(mesh)
    }

    #[test]
    fn raw_lateral_context_injects_full_prediction() -> Result<(), MeshError> {
        let mesh = mesh_with_seeded_prediction()?;
        let ctx = mesh.lateral_context_for("CC_PHYSICS_01");
        assert!(!ctx.text.is_empty());
        assert!(
            ctx.text.contains("Prediction=\""),
            "raw mode quotes prediction"
        );
        assert!(
            ctx.text.contains("elaboration"),
            "raw mode includes full text"
        );
        Ok(())
    }

    #[test]
    fn structured_lateral_context_never_quotes_full_prediction() -> Result<(), MeshError> {
        let mut mesh = mesh_with_seeded_prediction()?;
        mesh.lateral_context_mode = LateralContextMode::Structured;
        let ctx = mesh.lateral_context_for("CC_PHYSICS_01");
        assert!(!ctx.text.is_empty());
        assert!(
            !ctx.text.contains("Prediction=\""),
            "no raw prediction label"
        );
        // The full tail must NOT appear (it was truncated at the sentence boundary).
        assert!(!ctx.text.contains("elaboration"), "truncates the full text");
        // The claim excerpt (first sentence) IS present, bounded.
        assert!(
            ctx.text.contains("thermal equilibrium"),
            "includes the claim"
        );
        assert!(ctx.text.contains("source=CC_MATH_01"), "tags the source id");
        assert!(ctx.text.contains("confidence=0.82"), "includes confidence");
        assert!(
            ctx.text.contains("Do not quote or copy"),
            "includes synthesis directive"
        );
        Ok(())
    }

    #[test]
    fn default_lateral_mode_is_raw() -> Result<(), MeshError> {
        let mesh = default_mesh(Arc::new(MockEngine))?;
        assert_eq!(mesh.lateral_context_mode, LateralContextMode::Raw);
        Ok(())
    }
}
