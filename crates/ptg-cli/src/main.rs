//! PTG command-line entry point (§8.4).
//!
//! Builds the default reference mesh and runs a consensus epoch against a local
//! OpenAI-compatible server. Use `--dry-run` to validate wiring without a
//! server, or `--probe` to test reachability without running an epoch.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use clap::{Parser, ValueEnum};
use ptg_core::{ImageDetail, ImageUrlRef, Stimulus};
use ptg_runtime::{default_mesh, mesh_from_columns};
use ptg_vllm::{list_models, ColumnEngine, InferenceEngine};

use ptg_cli::setup::{load_config, PhaseCommand};
use ptg_cli::topology_cli::{MeshPlan, MeshTopologyParams, TopologyKind};

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
    /// Optional phase subcommand. With no subcommand, `ptg` runs a mesh against
    /// the configured server (use `ptg setup` / `ptg serve` to prepare it).
    #[command(subcommand)]
    command: Option<PhaseCommand>,
    /// Base URL of the local OpenAI-compatible "thalamus" server. If unset,
    /// falls back to the value written by `ptg setup`, then to
    /// `http://localhost:8000`.
    #[arg(long, env = "PTG_VLLM_URL")]
    vllm_url: Option<String>,

    /// Model id served by the engine (§7.1). If unset, falls back to the alias
    /// written by `ptg setup`, then to the bundled default.
    #[arg(long, env = "PTG_MODEL")]
    model: Option<String>,

    /// Maximum number of consensus ticks per epoch.
    #[arg(long, default_value_t = 5)]
    ticks: u32,

    /// Minimum number of ticks before convergence is considered (forces lateral
    /// exchange to run even when columns are overconfident on tick 1). Default 1;
    /// set >= 2 to actually exercise the lateral mechanism on overconfident models.
    #[arg(long, default_value_t = 1)]
    min_ticks: u32,

    /// Maximum output tokens per column tick (default 1024). Raise if columns
    /// emit JSON that truncates mid-string (common at high column counts where
    /// neighbor context lengthens the prompt).
    #[arg(long, default_value_t = 1024)]
    max_tokens: u32,

    /// Sampling temperature (default 0.0 for deterministic output).
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// Stop the consensus loop once mean token-Jaccard similarity of successive
    /// per-column predictions reaches this value in `[0.0, 1.0]`. A cheap,
    /// model-independent string proxy for semantic stabilization that does not
    /// rely on the self-reported confidence a model can game. Omit to disable
    /// (default), preserving confidence-only convergence.
    #[arg(long, value_name = "0.0..1.0")]
    min_prediction_similarity: Option<f32>,

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

    /// TOML column pack defining explicit columns (id / sphere / system_prompt /
    /// optional level). Replaces the round-robin replicated defaults for a
    /// generated topology. With `--topology default` the pack ids must be exactly
    /// the four named reference ids. For generated topologies the pack count
    /// must equal `--columns` (or width*height for torus).
    #[arg(long = "column-pack", value_name = "PATH")]
    column_pack: Option<PathBuf>,

    /// Lateral mesh topology (§3.1.3). `default` is the named 4-column reference
    /// graph; the others are built over `--columns` replicated domain spheres.
    #[arg(long, value_enum, default_value_t = TopologyKind::Default)]
    topology: TopologyKind,

    /// Number of columns for a generated topology (ignored for `default`).
    /// For `torus`, derived from `--torus-width` * `--torus-height`.
    #[arg(long)]
    columns: Option<usize>,

    /// Torus grid width (required for `--topology torus`).
    #[arg(long)]
    torus_width: Option<usize>,

    /// Torus grid height (required for `--topology torus`).
    #[arg(long)]
    torus_height: Option<usize>,

    /// Small-world ring-lattice out-degree (even; default 4).
    #[arg(long, default_value_t = 4)]
    small_world_degree: usize,

    /// Small-world edge rewire probability in `[0,1]` (default 0.10).
    #[arg(long, default_value_t = 0.10)]
    small_world_rewire: f64,

    /// Small-world PRNG seed (default 42). Same seed -> same graph.
    #[arg(long, default_value_t = 42)]
    small_world_seed: u64,

    /// Lateral routing policy (§9.1): how each column selects which neighbors'
    /// predictions to inject. `all` (default, V1) injects every neighbor;
    /// `confidence-top-k` injects the k highest-confidence; `diversity`
    /// diversity-preserving (MMR) selection that keeps dissimilar frames — the
    /// hypothesized mitigation for lateral homogenization.
    #[arg(long, value_enum, default_value_t = RoutingPolicyKind::All)]
    routing_policy: RoutingPolicyKind,

    /// Source budget for `--routing-policy confidence-top-k` / `diversity`
    /// (default 2). Ignored for `all`.
    #[arg(long, default_value_t = 2)]
    routing_k: usize,
}

/// CLI mirror of [`ptg_runtime::RoutingPolicy`] (§9.1). `All` is the default and
/// preserves V1 behavior; the other two take a budget `k` via `--routing-k`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "kebab-case")]
enum RoutingPolicyKind {
    /// Inject every neighbor equally (V1 behavior).
    All,
    /// Inject the k highest-confidence neighbors.
    ConfidenceTopK,
    /// MMR-style diversity-preserving selection of up to k neighbors.
    Diversity,
}

impl RoutingPolicyKind {
    /// Materialize into the runtime policy with the supplied budget.
    fn to_policy(self, k: usize) -> ptg_runtime::RoutingPolicy {
        match self {
            Self::All => ptg_runtime::RoutingPolicy::All,
            Self::ConfidenceTopK => ptg_runtime::RoutingPolicy::ConfidenceTopK { k },
            Self::Diversity => ptg_runtime::RoutingPolicy::DiversityPreserving { k },
        }
    }

    /// The kebab-case flag value the user would type (for dry-run / errors).
    fn kebab(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::ConfidenceTopK => "confidence-top-k",
            Self::Diversity => "diversity",
        }
    }
}

/// Resolve the topology plan from this CLI's flags. Delegates to the shared
/// [`ptg_cli::topology_cli`] resolver so `ptg` and `ptg-bench` cannot drift on
/// column-count / small-world-degree validation.
fn topology_plan(cli: &Cli) -> Result<Option<MeshPlan>, String> {
    ptg_cli::topology_cli::topology_plan(&MeshTopologyParams {
        topology: cli.topology,
        columns: cli.columns,
        torus_width: cli.torus_width,
        torus_height: cli.torus_height,
        small_world_degree: cli.small_world_degree,
        small_world_rewire: cli.small_world_rewire,
        small_world_seed: cli.small_world_seed,
        column_pack: cli.column_pack.clone(),
    })
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
    // Phase subcommands short-circuit before the mesh path.
    if let Some(phase) = cli.command {
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
        let result = match phase {
            PhaseCommand::Setup(args) => rt.block_on(ptg_cli::setup::run_setup(args)),
            PhaseCommand::Serve(args) => rt.block_on(ptg_cli::setup::run_serve(args)),
        };
        return match result {
            Ok(code) => code,
            Err(e) => {
                eprintln!("ptg: {e}");
                ExitCode::FAILURE
            }
        };
    }
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
    // Resolve server URL / model in priority order: explicit flag > env > the
    // config written by `ptg setup` > bundled defaults. This is what makes
    // "`ptg setup` once, then just `ptg`" work.
    let cfg = load_config();
    let vllm_url = cli
        .vllm_url
        .clone()
        .or_else(|| cfg.as_ref().map(|c| c.server_url()))
        .unwrap_or_else(|| "http://localhost:8000".to_string());
    let model = cli
        .model
        .clone()
        .or_else(|| cfg.as_ref().map(|c| c.model_alias.clone()))
        .unwrap_or_else(|| "solidrust/Gemma-4-2B-Multimodal-Q4_K_M".to_string());
    if let Some(sim) = cli.min_prediction_similarity {
        if !(0.0..=1.0).contains(&sim) {
            return Err(
                format!("--min-prediction-similarity must be in [0.0, 1.0], got {sim}").into(),
            );
        }
    }
    println!("Initializing Project Thousand-Gemma Cortical Simulation Workstation...");

    if cli.dry_run {
        println!("PTG dry run — no inference will be performed.");
        println!("  vLLM URL : {vllm_url}");
        println!("  model    : {model}");
        println!("  ticks    : {}", cli.ticks);
        match topology_plan(&cli)? {
            None => {
                println!("  topology : default (named 4-column reference graph)");
                for col in ptg_core::default_columns() {
                    println!("  column   : {} [{}]", col.id, col.sphere.as_str());
                }
            }
            Some(plan) => {
                println!(
                    "  topology : {} ({} columns, {} lateral edges)",
                    plan.label,
                    plan.columns.len(),
                    plan.connections.len()
                );
                for (col, level) in plan.columns.iter().zip(plan.levels.iter()) {
                    match level {
                        Some(lv) => println!(
                            "  column   : {} [{}] level={}",
                            col.id,
                            col.sphere.as_str(),
                            lv
                        ),
                        None => println!("  column   : {} [{}]", col.id, col.sphere.as_str()),
                    }
                }
                let mut by_listener: std::collections::BTreeMap<&str, Vec<&str>> =
                    std::collections::BTreeMap::new();
                for c in &plan.connections {
                    by_listener
                        .entry(c.listener_id.as_str())
                        .or_default()
                        .push(c.source_id.as_str());
                }
                for (listener, sources) in &by_listener {
                    let mut s = sources.to_vec();
                    s.sort_unstable();
                    println!("  edge     : {listener} <- [{}]", s.join(", "));
                }
            }
        }
        let k_label = match cli.routing_policy {
            RoutingPolicyKind::All => String::new(),
            _ => format!(" k={}", cli.routing_k),
        };
        println!("  routing  : {}{k_label}", cli.routing_policy.kebab());
        return Ok(ExitCode::SUCCESS);
    }

    let engine = InferenceEngine::builder(&vllm_url, &model)
        .max_tokens(cli.max_tokens)
        .temperature(cli.temperature)
        .build()?;

    if cli.probe {
        let models = list_models(engine.http_client(), engine.vllm_url()).await?;
        if models.is_empty() {
            println!("reachable, but no models served at {vllm_url}");
            return Ok(ExitCode::SUCCESS);
        }
        println!("reachable: {} model(s) at {}", models.len(), vllm_url);
        for m in &models {
            println!("  - {m}");
        }
        if !models.iter().any(|m| m == &model) {
            eprintln!("warning: requested model `{model}` is not in the served list");
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

    // 2. Instantiate the mesh: the named reference graph, or a generated
    //    topology over replicated domain spheres (§3.1.3).
    let engine: Arc<dyn ColumnEngine> = Arc::new(engine);
    let mut mesh = match topology_plan(&cli)? {
        Some(plan) => {
            println!(
                "Topology: {} ({} columns, {} lateral edges)",
                plan.label,
                plan.columns.len(),
                plan.connections.len()
            );
            mesh_from_columns(engine, plan.columns, plan.connections)?
        }
        None => default_mesh(engine)?,
    };
    mesh.criteria.max_ticks = cli.ticks;
    mesh.criteria.min_ticks = cli.min_ticks;
    mesh.criteria.min_prediction_similarity = cli.min_prediction_similarity;
    mesh.routing_policy = cli.routing_policy.to_policy(cli.routing_k);

    // 3. Broadcast stimulus and run the decentralized consensus epoch.
    println!("Broadcast Input Signal: '{}'", stimulus.text_str());
    tracing::info!(ticks = cli.ticks, "running consensus epoch");
    let result = mesh.run_epoch(&stimulus).await?;

    // 4. Global integration readout (§6 Phase 3).
    println!(
        "\nEpoch complete: {} tick(s), stabilized={}, mean confidence={:.3}",
        result.ticks_run, result.stabilized, result.mean_confidence
    );
    if let Some(reason) = result.convergence_reason {
        println!("convergence: {reason}");
    }
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

#[cfg(test)]
mod tests {
    //! Validation tests for `topology_plan`. Use `?` throughout (no unwrap/expect/panic).

    use super::*;

    fn test_cli(topology: TopologyKind, columns: Option<usize>) -> Cli {
        Cli {
            command: None,
            vllm_url: None,
            model: None,
            ticks: 5,
            min_ticks: 1,
            max_tokens: 1024,
            temperature: 0.0,
            min_prediction_similarity: None,
            input: String::new(),
            image_urls: Vec::new(),
            image_detail: String::from("auto"),
            probe: false,
            dry_run: false,
            column_pack: None,
            topology,
            columns,
            torus_width: None,
            torus_height: None,
            small_world_degree: 4,
            small_world_rewire: 0.10,
            small_world_seed: 42,
            routing_policy: RoutingPolicyKind::All,
            routing_k: 2,
        }
    }

    #[test]
    fn default_topology_yields_no_plan() -> Result<(), String> {
        assert!(topology_plan(&test_cli(TopologyKind::Default, None))?.is_none());
        Ok(())
    }

    #[test]
    fn ring_5_has_five_columns_and_five_edges() -> Result<(), String> {
        let plan = topology_plan(&test_cli(TopologyKind::Ring, Some(5)))?
            .ok_or("expected a plan for ring")?;
        assert_eq!(plan.columns.len(), 5);
        assert_eq!(
            plan.connections.len(),
            5,
            "directed ring: one edge per column"
        );
        assert_eq!(plan.label, "ring");
        Ok(())
    }

    #[test]
    fn ring_bi_4_has_eight_edges() -> Result<(), String> {
        let plan = topology_plan(&test_cli(TopologyKind::RingBi, Some(4)))?
            .ok_or("expected a plan for ring-bi")?;
        assert_eq!(plan.columns.len(), 4);
        assert_eq!(
            plan.connections.len(),
            8,
            "bidirectional ring: two edges per column"
        );
        Ok(())
    }

    #[test]
    fn ring_bi_3_is_rejected() -> Result<(), String> {
        let res = topology_plan(&test_cli(TopologyKind::RingBi, Some(3)));
        assert!(
            res.is_err(),
            "ring-bi with n<=3 collapses to fully-connected"
        );
        Ok(())
    }

    #[test]
    fn torus_3x3_has_nine_columns_and_thirty_six_edges() -> Result<(), String> {
        let mut cli = test_cli(TopologyKind::Torus, None);
        cli.torus_width = Some(3);
        cli.torus_height = Some(3);
        let plan = topology_plan(&cli)?.ok_or("expected a plan for torus")?;
        assert_eq!(plan.columns.len(), 9);
        assert_eq!(plan.connections.len(), 36, "9 nodes * 4 cardinal neighbors");
        Ok(())
    }

    #[test]
    fn torus_missing_dimension_is_rejected() -> Result<(), String> {
        let mut cli = test_cli(TopologyKind::Torus, None);
        cli.torus_width = Some(3);
        // height missing
        assert!(topology_plan(&cli).is_err());
        Ok(())
    }

    #[test]
    fn fully_connected_4_has_twelve_edges() -> Result<(), String> {
        let plan = topology_plan(&test_cli(TopologyKind::FullyConnected, Some(4)))?
            .ok_or("expected a plan for fully-connected")?;
        assert_eq!(plan.columns.len(), 4);
        assert_eq!(plan.connections.len(), 12, "n*(n-1) = 4*3");
        Ok(())
    }

    #[test]
    fn small_world_20_degree_4_has_eighty_edges() -> Result<(), String> {
        let plan = topology_plan(&test_cli(TopologyKind::SmallWorld, Some(20)))?
            .ok_or("expected a plan for small-world")?;
        assert_eq!(plan.columns.len(), 20);
        assert_eq!(plan.connections.len(), 80, "20 * degree 4");
        Ok(())
    }

    #[test]
    fn small_world_degree_too_high_is_rejected() -> Result<(), String> {
        let mut cli = test_cli(TopologyKind::SmallWorld, Some(6));
        cli.small_world_degree = 4; // 4*2 = 8 >= 6 → under-rewiring hazard
        assert!(topology_plan(&cli).is_err());
        Ok(())
    }

    #[test]
    fn ring_missing_columns_is_rejected() -> Result<(), String> {
        let res = topology_plan(&test_cli(TopologyKind::Ring, None));
        let msg = match res {
            Err(m) => m,
            Ok(plan) => return Err(format!("expected error, got plan: {plan:?}")),
        };
        // Friendly message uses the kebab flag name, not the Debug form.
        assert!(msg.contains("--topology ring"), "got: {msg}");
        Ok(())
    }

    #[test]
    fn small_world_is_deterministic_across_calls() -> Result<(), String> {
        let a = topology_plan(&test_cli(TopologyKind::SmallWorld, Some(30)))?;
        let b = topology_plan(&test_cli(TopologyKind::SmallWorld, Some(30)))?;
        let (a_conns, b_conns) = match (a, b) {
            (Some(pa), Some(pb)) => (pa.connections, pb.connections),
            _ => return Err("expected plans".to_string()),
        };
        assert_eq!(a_conns, b_conns);
        Ok(())
    }

    /// Helper: write a pack to a temp file and return its path.
    fn write_test_pack(body: &str) -> Result<PathBuf, String> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static MAIN_PACK_COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir();
        let id = MAIN_PACK_COUNTER.fetch_add(1, Ordering::SeqCst);
        // PID + counter: nextest runs each test in its own process, so a
        // per-process counter alone collides on a shared temp path.
        let path = dir.join(format!("ptg-main-pack-{}-{id}.toml", std::process::id()));
        std::fs::write(&path, body).map_err(|e| format!("write: {e}"))?;
        Ok(path)
    }

    #[test]
    fn column_pack_default_with_wrong_ids_is_rejected() -> Result<(), String> {
        // Two columns but NOT the four reference ids.
        let body = r#"
[[columns]]
id = "CC_X"
sphere = "Physics"
system_prompt = "x"
[[columns]]
id = "CC_Y"
sphere = "Coding"
system_prompt = "y"
"#;
        let path = write_test_pack(body)?;
        let mut cli = test_cli(TopologyKind::Default, None);
        cli.column_pack = Some(path.clone());
        let res = topology_plan(&cli);
        std::fs::remove_file(&path).ok();
        assert!(
            res.is_err(),
            "pack with non-reference ids + default should error"
        );
        Ok(())
    }

    #[test]
    fn column_pack_generated_count_mismatch_is_rejected() -> Result<(), String> {
        // Pack has 2 columns but ring expects --columns 4.
        let body = r#"
[[columns]]
id = "CC_A"
sphere = "Physics"
system_prompt = "a"
[[columns]]
id = "CC_B"
sphere = "Coding"
system_prompt = "b"
"#;
        let path = write_test_pack(body)?;
        let mut cli = test_cli(TopologyKind::Ring, Some(4));
        cli.column_pack = Some(path.clone());
        let res = topology_plan(&cli);
        std::fs::remove_file(&path).ok();
        assert!(res.is_err(), "pack count != topology columns should error");
        Ok(())
    }

    #[test]
    fn column_pack_generated_matching_count_yields_plan() -> Result<(), String> {
        // Pack has 3 columns matching --columns 3 for a ring.
        let body = r#"
[[columns]]
id = "CC_A"
sphere = "Physics"
system_prompt = "a"
level = 1
[[columns]]
id = "CC_B"
sphere = "Coding"
system_prompt = "b"
level = 2
[[columns]]
id = "CC_C"
sphere = "Mathematics"
system_prompt = "c"
level = 3
"#;
        let path = write_test_pack(body)?;
        let mut cli = test_cli(TopologyKind::Ring, Some(3));
        cli.column_pack = Some(path.clone());
        let plan = topology_plan(&cli)?;
        std::fs::remove_file(&path).ok();
        let plan = match plan {
            Some(p) => p,
            None => return Err("expected a plan".to_string()),
        };
        assert_eq!(plan.columns.len(), 3, "pack columns should be used");
        assert_eq!(plan.connections.len(), 3, "ring over 3 = 3 edges");
        // Levels from the pack should be surfaced.
        assert_eq!(plan.levels, vec![Some(1), Some(2), Some(3)]);
        // The pack's OWN ids, not the replicated defaults.
        assert_eq!(plan.columns[0].id, "CC_A");
        Ok(())
    }
}
