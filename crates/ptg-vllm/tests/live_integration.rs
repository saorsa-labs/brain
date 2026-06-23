//! Live integration test against a local OpenAI-compatible server.
//!
//! Ignored by default — run with `cargo test -p ptg-vllm --test live_integration
//! -- --ignored`. Only hits the network if `PTG_VLLM_URL` points at a reachable
//! server; otherwise returns early with success.

use std::sync::Arc;

use ptg_runtime::default_mesh;
use ptg_vllm::{list_models, InferenceEngine};

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

#[tokio::test]
#[ignore = "requires a running local inference server (set PTG_VLLM_URL)"]
async fn live_text_epoch() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let url = env_or("PTG_VLLM_URL", "http://127.0.0.1:18135");
    let model = env_or("PTG_MODEL", "gemma-4-e4b");

    // Probe; skip gracefully if no server is running.
    let probe = InferenceEngine::new(&url, &model)?;
    let models = match list_models(probe.http_client(), probe.vllm_url()).await {
        Ok(m) => m,
        Err(e) => {
            eprintln!("skipping live_text_epoch: server unreachable ({e})");
            return Ok(());
        }
    };
    if models.is_empty() {
        eprintln!("skipping live_text_epoch: no models served");
        return Ok(());
    }

    let engine: Arc<dyn ptg_vllm::ColumnEngine> = Arc::new(probe);
    let mut mesh = default_mesh(engine)?;
    let result = mesh
        .run_text_epoch("A ball is dropped from a height. Briefly: what happens to its kinetic energy as it falls?")
        .await?;

    assert_eq!(
        result.outputs.len(),
        4,
        "all four columns should have output"
    );
    for (_id, out) in &result.outputs {
        assert!(
            (0.0..=1.0).contains(&out.confidence),
            "confidence {} out of range for output",
            out.confidence
        );
    }
    eprintln!(
        "live_text_epoch: {} tick(s), stabilized={}, mean_conf={:.3}",
        result.ticks_run, result.stabilized, result.mean_confidence
    );
    Ok(())
}
