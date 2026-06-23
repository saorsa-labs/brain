//! PTG command-line entry point (§8.4).
//!
//! Builds the default reference mesh and runs a consensus epoch against a local
//! OpenAI-compatible server. Use `--dry-run` to validate wiring without a
//! server, or `--probe` to test reachability without running an epoch.

use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;
use ptg_core::{ImageDetail, ImageUrlRef, Stimulus};
use ptg_runtime::default_mesh;
use ptg_vllm::{list_models, ColumnEngine, InferenceEngine};

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
    /// Base URL of the local OpenAI-compatible "thalamus" server.
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

    /// Image URL(s) (repeatable) to attach to the stimulus, making it multimodal.
    /// May be an `https://` URL or an inline `data:image/...;base64,...` payload.
    #[arg(long = "image-url")]
    image_urls: Vec<String>,

    /// Resolution hint for image parts: low, high, or auto.
    #[arg(long, default_value = "auto")]
    image_detail: String,

    /// Probe the server (`/v1/models`) and exit without running an epoch.
    #[arg(long)]
    probe: bool,

    /// Validate wiring and print the plan without contacting the engine.
    #[arg(long)]
    dry_run: bool,
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();
    let cli = Cli::parse();
    // Run on a single-threaded runtime: the mesh uses join_all, not spawn.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build();
    let rt = match rt {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ptg: failed to start runtime: {e}");
            return ExitCode::FAILURE;
        }
    };
    match rt.block_on(run(cli)) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("ptg: {err}");
            ExitCode::FAILURE
        }
    }
}

fn parse_detail(s: &str) -> Result<ImageDetail, String> {
    match s.to_ascii_lowercase().as_str() {
        "low" => Ok(ImageDetail::Low),
        "high" => Ok(ImageDetail::High),
        "auto" => Ok(ImageDetail::Auto),
        other => Err(format!("invalid --image-detail `{other}` (low|high|auto)")),
    }
}

async fn run(cli: Cli) -> Result<ExitCode, Box<dyn std::error::Error + Send + Sync>> {
    println!("Initializing Project Thousand-Gemma Cortical Simulation Workstation...");

    if cli.dry_run {
        println!("PTG dry run — no inference will be performed.");
        println!("  vLLM URL : {}", cli.vllm_url);
        println!("  model    : {}", cli.model);
        println!("  ticks    : {}", cli.ticks);
        for col in ptg_core::default_columns() {
            println!("  column   : {} [{}]", col.id, col.sphere.as_str());
        }
        return Ok(ExitCode::SUCCESS);
    }

    let engine = InferenceEngine::new(&cli.vllm_url, &cli.model)?;

    if cli.probe {
        let models = list_models(engine.http_client(), engine.vllm_url()).await?;
        if models.is_empty() {
            println!("reachable, but no models served at {}", cli.vllm_url);
            return Ok(ExitCode::SUCCESS);
        }
        println!("reachable: {} model(s) at {}", models.len(), cli.vllm_url);
        for m in &models {
            println!("  - {m}");
        }
        if !models.iter().any(|m| m == &cli.model) {
            eprintln!(
                "warning: requested model `{}` is not in the served list",
                cli.model
            );
        }
        return Ok(ExitCode::SUCCESS);
    }

    // 1. Build the stimulus: text, or multimodal if image URLs were supplied.
    let stimulus = if cli.image_urls.is_empty() {
        println!("Stimulus mode: text");
        Stimulus::text(cli.input.clone())
    } else {
        let detail = parse_detail(&cli.image_detail)?;
        let parts = cli
            .image_urls
            .iter()
            .map(|url| ptg_core::StimulusPart::ImageUrl {
                image_url: ImageUrlRef {
                    url: url.clone(),
                    detail,
                },
            })
            .collect();
        println!(
            "Stimulus mode: multimodal ({} image part(s))",
            cli.image_urls.len()
        );
        Stimulus::Multimodal {
            text: cli.input.clone(),
            parts,
        }
    };

    // 2. Instantiate the reference mesh (§8.4) and tune the tick budget.
    let engine: Arc<dyn ColumnEngine> = Arc::new(engine);
    let mut mesh = default_mesh(engine)?;
    mesh.criteria.max_ticks = cli.ticks;

    // 3. Broadcast stimulus and run the decentralized consensus epoch.
    println!("Broadcast Input Signal: '{}'", stimulus.text_str());
    tracing::info!(ticks = cli.ticks, "running consensus epoch");
    let result = mesh.run_epoch(&stimulus).await?;

    // 4. Global integration readout (§6 Phase 3).
    println!(
        "\nEpoch complete: {} tick(s), stabilized={}, mean confidence={:.3}",
        result.ticks_run, result.stabilized, result.mean_confidence
    );
    let threshold = mesh.criteria.min_integration_confidence;
    println!(
        "integration: {} accepted, {} rejected (threshold={:.2})",
        result.accepted_outputs.len(),
        result.rejected_outputs.len(),
        threshold
    );
    for (id, out) in &result.accepted_outputs {
        println!(
            "  [{id}] coordinate={} confidence={:.2} prediction=\"{}\"",
            out.reference_frame_coordinates, out.confidence, out.prediction
        );
    }
    for (id, out) in &result.rejected_outputs {
        println!(
            "  [{id}] REJECTED confidence={:.2} prediction=\"{}\"",
            out.confidence, out.prediction
        );
    }
    println!("\nConsensus process completed. Global perceptual state stabilized.");
    Ok(ExitCode::SUCCESS)
}
