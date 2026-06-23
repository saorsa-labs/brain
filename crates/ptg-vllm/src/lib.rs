//! Client bindings to the local inference engine that PTG uses as its shared
//! "thalamus": a single model instance multiplexed across all virtual columns
//! via dynamic system prompts, prefix caching, and attention-mask reuse.
//!
//! The specification names a local `vLLM` engine for this role; the trait
//! below is engine-agnostic so a `mistral.rs` or other backend can be swapped
//! in without changing the mesh runtime.

use async_trait::async_trait;
use ptg_core::{ColumnId, ColumnOutput, ColumnSpec};

/// Parameters for a single inference request against the shared engine.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub column: ColumnId,
    /// User-side stimulus fanned into the column.
    pub stimulus: String,
    /// Injected neighbor context (lateral connections), already serialized.
    pub lateral_context: Vec<String>,
}

/// Errors surfaced by the inference engine layer.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// The underlying HTTP / engine request failed.
    #[error("inference request failed: {0}")]
    Request(String),
    /// The engine produced output that did not match the column's frame schema.
    #[error("engine returned malformed structured output: {0}")]
    Malformed(String),
}

/// Contract for the shared local inference engine.
///
/// One instance of an `VllmEngine` implementor is shared (e.g. behind an
/// `Arc`) by every column in the mesh, reflecting the single-engine topology
/// described in the specification.
#[async_trait]
pub trait VllmEngine: Send + Sync {
    /// Run one column's inference pass, returning its structured prediction.
    async fn infer(
        &self,
        spec: &ColumnSpec,
        request: &InferenceRequest,
    ) -> Result<ColumnOutput, EngineError>;
}
