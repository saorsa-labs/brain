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
    ConvergenceCriteria, ConvergenceReason,
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
    /// most recent predictions (§6 Phase 2). Empty until neighbors have output.
    fn lateral_context_for(&self, id: &str) -> String {
        let Some(neighbor_ids) = self.adjacency_list.get(id) else {
            return String::new();
        };
        let mut lines = Vec::new();
        for nid in neighbor_ids {
            let Some(n) = self.columns.get(nid) else {
                continue;
            };
            if n.last_prediction.is_empty() {
                continue;
            }
            lines.push(format!(
                "Neighbor {} reports: Prediction=\"{}\" (Confidence: {:.2})",
                n.id, n.last_prediction, n.last_confidence
            ));
        }
        if lines.is_empty() {
            String::new()
        } else {
            format!("[LATERAL LAYER UPDATE]\n{}", lines.join("\n"))
        }
    }

    /// Run one full epoch (the three-phase loop) over the given stimulus.
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
            for id in &self.column_ids_sorted() {
                let Some(col) = self.columns.get(id) else {
                    continue;
                };
                let col = col.clone();
                let ctx = self.lateral_context_for(id);
                tasks.push((id.clone(), col, ctx));
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

/// The per-column outputs captured at a single executed tick.
#[derive(Debug, Clone)]
pub struct TickOutputs {
    /// 1-indexed tick number this snapshot is from.
    pub tick: u32,
    /// Per-column outputs at this tick (all columns, sorted-id order).
    pub outputs: Vec<(String, ColumnOutputSchema)>,
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
}
