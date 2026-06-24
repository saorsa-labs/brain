//! PTG command-line entry point (§8.4).
//!
//! Builds the default reference mesh and runs a consensus epoch against a local
//! OpenAI-compatible server. Use `--dry-run` to validate wiring without a
//! server, or `--probe` to test reachability without running an epoch.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use clap::{Parser, ValueEnum};
use ptg_core::{
    replicated_default_columns, CorticalColumn, ImageDetail, ImageUrlRef, LateralConnection,
    Stimulus, TopologySpec,
};
use ptg_runtime::{default_mesh, mesh_from_columns};
use ptg_vllm::{list_models, ColumnEngine, InferenceEngine};

mod column_pack;

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

    /// Small-world edge rewire probability in [0,1] (default 0.10).
    #[arg(long, default_value_t = 0.10)]
    small_world_rewire: f64,

    /// Small-world PRNG seed (default 42). Same seed -> same graph.
    #[arg(long, default_value_t = 42)]
    small_world_seed: u64,
}

/// Selectable lateral topologies for `--topology`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "kebab-case")]
enum TopologyKind {
    /// Named 4-column reference graph (§8.4), unchanged.
    Default,
    /// Directed 1-D cycle: each column listens to its predecessor.
    Ring,
    /// Bidirectional ring: each column listens to predecessor + successor.
    /// Requires >= 4 columns (for n<=3 it collapses to the complete graph).
    RingBi,
    /// 2-D wraparound grid: each column listens to its four cardinal neighbors.
    Torus,
    /// Every column listens to every other.
    FullyConnected,
    /// Seeded Watts-Strogatz small-world (deterministic given `--small-world-seed`).
    SmallWorld,
}

/// A fully-materialized topology plan: the columns plus the listener->source edge
/// list, plus a human label. Built by [`topology_plan`]; `None` means "use the
/// default reference mesh".
#[derive(Debug)]
struct MeshPlan {
    columns: Vec<CorticalColumn>,
    connections: Vec<LateralConnection>,
    /// Optional abstraction-level tag per column (parallel to `columns`),
    /// populated from a column pack's `level` field; empty for replicated
    /// defaults. Surfaced in `--dry-run` for experiment attribution.
    levels: Vec<Option<u8>>,
    label: String,
}

impl TopologyKind {
    /// The kebab-case flag value the user would type (for error messages).
    fn kebab(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Ring => "ring",
            Self::RingBi => "ring-bi",
            Self::Torus => "torus",
            Self::FullyConnected => "fully-connected",
            Self::SmallWorld => "small-world",
        }
    }
}

/// The four named ids of the reference 4-column mesh, in any order.
fn expected_default_ids() -> &'static [&'static str] {
    &["CC_PHYSICS_01", "CC_MATH_01", "CC_CODE_01", "CC_PSYCH_01"]
}

/// Validate that a pack reproduces the named reference graph (by id, order-
/// independent) so `--topology default --column-pack` is the same wiring as
/// `default_mesh`.
///
/// # Errors
/// `Err(String)` if the pack id set is not exactly the four reference ids.
fn validate_default_pack_ids(columns: &[CorticalColumn]) -> Result<(), String> {
    let mut have: Vec<&str> = columns.iter().map(|c| c.id.as_str()).collect();
    have.sort_unstable();
    let mut want: Vec<&str> = expected_default_ids().to_vec();
    want.sort_unstable();
    if have == want {
        Ok(())
    } else {
        Err(format!(
            "--topology default with --column-pack requires exactly these ids \n(order-independent): {}\n(pack has: {})",
            expected_default_ids().join(", "),
            columns.iter().map(|c| c.id.as_str()).collect::<Vec<_>>().join(", ")
        ))
    }
}

/// The reference 4-column lateral edges as `LateralConnection`s.
/// `default_connections()` tuples are `(listener, source)` (the listener
/// receives the source's prediction); the conversion preserves that direction.
fn default_reference_edges() -> Vec<LateralConnection> {
    ptg_core::default_connections()
        .into_iter()
        .map(|(listener, source)| LateralConnection::new(listener, source))
        .collect()
}

/// Resolve the column count from `--columns`, or for torus from the grid dims.
/// Errors with a friendly message naming the topology.
fn resolve_columns(cli: &Cli) -> Result<usize, String> {
    if cli.topology == TopologyKind::Torus {
        let w = cli
            .torus_width
            .ok_or_else(|| "--topology torus requires --torus-width".to_string())?;
        let h = cli
            .torus_height
            .ok_or_else(|| "--topology torus requires --torus-height".to_string())?;
        let derived = w * h;
        if let Some(requested) = cli.columns {
            if requested != derived {
                return Err(format!(
                    "--columns {requested} != --torus-width {w} * --torus-height {h} ({derived})"
                ));
            }
        }
        return Ok(derived);
    }
    cli.columns
        .ok_or_else(|| format!("--topology {} requires --columns", cli.topology.kebab()))
}

/// Build a topology plan from the CLI flags. Returns `Ok(None)` for
/// [`TopologyKind::Default`] (the caller uses [`default_mesh`]); returns an
/// error string on any validation failure (degeneracy guardrails are stricter
/// than the library minimums, to keep generated graphs meaningful).
fn topology_plan(cli: &Cli) -> Result<Option<MeshPlan>, String> {
    if cli.topology == TopologyKind::Default {
        return match &cli.column_pack {
            None => Ok(None),
            Some(path) => {
                // A pack + default topology is only valid if it reproduces the
                // named 4-column reference graph exactly (by id; order-independent).
                let cols = column_pack::load_column_pack_with_levels(path)?;
                let (columns, levels): (Vec<_>, Vec<_>) = cols.into_iter().unzip();
                validate_default_pack_ids(&columns)?;
                let connections = default_reference_edges();
                Ok(Some(MeshPlan {
                    columns,
                    connections,
                    levels,
                    label: "default (from pack)".to_string(),
                }))
            }
        };
    }
    let n = resolve_columns(cli)?;

    // CLI-level degeneracy guardrails (stricter than the library minimums).
    match cli.topology {
        TopologyKind::Default => {}
        TopologyKind::Ring if n < 2 => {
            return Err("--topology ring requires --columns >= 2".to_string());
        }
        TopologyKind::RingBi if n < 4 => {
            // For n<=3 a bidirectional ring collapses to the complete graph.
            return Err(
                "--topology ring-bi requires --columns >= 4 (else it == fully-connected)"
                    .to_string(),
            );
        }
        TopologyKind::FullyConnected if n < 2 => {
            return Err("--topology fully-connected requires --columns >= 2".to_string());
        }
        TopologyKind::SmallWorld if cli.small_world_degree * 2 >= n => {
            // degree ≪ n is required for Watts-Strogatz to behave as intended;
            // near n it silently under-rewires (red-team finding).
            return Err(format!(
                "--topology small-world requires --small-world-degree * 2 < --columns \n(got degree {}, columns {})",
                cli.small_world_degree, n
            ));
        }
        _ => {}
    }

    let columns_levels: Vec<Option<u8>>;
    let columns = match &cli.column_pack {
        Some(path) => {
            let pack = column_pack::load_column_pack_with_levels(path)?;
            if pack.len() != n {
                return Err(format!(
                    "--column-pack has {} column(s) but topology expects {n} \n(derive via --columns or --torus-width/--torus-height)",
                    pack.len()
                ));
            }
            let (columns, levels): (Vec<_>, Vec<_>) = pack.into_iter().unzip();
            columns_levels = levels;
            columns
        }
        None => {
            columns_levels = vec![None; n];
            replicated_default_columns(n)
        }
    };
    let ids: Vec<String> = columns.iter().map(|c| c.id.clone()).collect();
    let (spec, label) = match cli.topology {
        TopologyKind::Default => unreachable!("handled above"),
        TopologyKind::Ring => (
            TopologySpec::Ring {
                bidirectional: false,
            },
            "ring",
        ),
        TopologyKind::RingBi => (
            TopologySpec::Ring {
                bidirectional: true,
            },
            "ring-bi",
        ),
        TopologyKind::Torus => {
            let w = cli.torus_width.unwrap_or(0);
            let h = cli.torus_height.unwrap_or(0);
            (
                TopologySpec::Torus2d {
                    width: w,
                    height: h,
                },
                "torus",
            )
        }
        TopologyKind::FullyConnected => (TopologySpec::FullyConnected, "fully-connected"),
        TopologyKind::SmallWorld => (
            TopologySpec::SmallWorld {
                degree: cli.small_world_degree,
                rewire_probability: cli.small_world_rewire,
                seed: cli.small_world_seed,
            },
            "small-world",
        ),
    };
    let connections = spec
        .connections_for(&ids)
        .map_err(|e| format!("{label}: {e}"))?;
    Ok(Some(MeshPlan {
        columns,
        connections,
        levels: columns_levels,
        label: label.to_string(),
    }))
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
        return Ok(ExitCode::SUCCESS);
    }

    let engine = InferenceEngine::builder(&cli.vllm_url, &cli.model)
        .max_tokens(cli.max_tokens)
        .temperature(cli.temperature)
        .build()?;

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

#[cfg(test)]
mod tests {
    //! Validation tests for `topology_plan`. Use `?` throughout (no unwrap/expect/panic).

    use super::*;

    fn test_cli(topology: TopologyKind, columns: Option<usize>) -> Cli {
        Cli {
            vllm_url: String::new(),
            model: String::new(),
            ticks: 5,
            min_ticks: 1,
            max_tokens: 1024,
            temperature: 0.0,
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
        let path = dir.join(format!("ptg-main-pack-{id}.toml"));
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
