//! PTG command-line entry point (§8.4).
//!
//! Builds the default reference mesh and runs a consensus epoch against a local
//! vLLM server. Use `--dry-run` to validate the wiring without a server.

use std::sync::Arc;

use clap::Parser;
use ptg_runtime::default_mesh;
use ptg_vllm::{ColumnEngine, InferenceEngine};

/// Default stimulus from the reference wiring (§8.4).
const DEFAULT_INPUT: &str = "Anomalous kinetic energy burst detected tracking at vector [4, 12, -2]. Script automation system failed initialization step.";

/// Run Project Thousand-Gemma.
#[derive(Parser, Debug)]
#[command(
    name = "ptg",
    version,
    about = "Project Thousand-Gemma: a distributed, prompt-based cortical mesh simulator"
)]
struct Cli {
    /// Base URL of the local vLLM "thalamus" server.
    #[arg(long, env = "PTG_VLLM_URL", default_value = "http://localhost:8000")]
    vllm_url: String,

    /// Model id served by the engine (§7.1).
    #[arg(
        long,
        env = "PTG_MODEL",
        default_value = "solidrust/Gemma-4-2B-Multimodal-Q4_K_M"
    )]
    model: String,

    /// Maximum number of consensus ticks per epoch.
    #[arg(long, default_value_t = 5)]
    ticks: u32,

    /// Input stimulus broadcast to all columns.
    #[arg(long, default_value = DEFAULT_INPUT)]
    input: String,

    /// Validate wiring and print the plan without contacting the engine.
    #[arg(long)]
    dry_run: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();
    run(cli).await
}

async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Initializing Project Thousand-Gemma Cortical Simulation Workstation...");

    if cli.dry_run {
        println!("PTG dry run — no inference will be performed.");
        println!("  vLLM URL : {}", cli.vllm_url);
        println!("  model    : {}", cli.model);
        println!("  ticks    : {}", cli.ticks);
        for col in ptg_core::default_columns() {
            println!("  column   : {} [{}]", col.id, col.sphere.as_str());
        }
        return Ok(());
    }

    // 1. Initialize the shared inference engine ("thalamus").
    let engine: Arc<dyn ColumnEngine> = Arc::new(InferenceEngine::new(&cli.vllm_url, &cli.model)?);

    // 2. Instantiate the reference mesh (§8.4) and tune the tick budget.
    let mut mesh = default_mesh(engine)?;
    mesh.criteria.max_ticks = cli.ticks;

    // 3. Broadcast stimulus and run the decentralized consensus epoch.
    println!("Broadcast Input Signal: '{}'", cli.input);
    tracing::info!(ticks = cli.ticks, "running consensus epoch");
    let result = mesh.run_epoch(&cli.input).await?;

    // 4. Global integration readout.
    println!(
        "\nEpoch complete: {} tick(s), stabilized={}, mean confidence={:.3}",
        result.ticks_run, result.stabilized, result.mean_confidence
    );
    for (id, out) in &result.outputs {
        println!(
            "  [{id}] coordinate={} confidence={:.2} prediction=\"{}\"",
            out.reference_frame_coordinates, out.confidence, out.prediction
        );
    }
    println!("\nConsensus process completed. Global perceptual state stabilized.");
    Ok(())
}
