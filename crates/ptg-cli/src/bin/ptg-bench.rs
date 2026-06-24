//! `ptg-bench` — benchmark harness comparing the PTG cortical mesh against
//! single-monolithic-context baselines on the same local inference server.
//!
//! Measures **latency** and **token economy** (including prefix-cache effects).
//! Quality (LLM-as-judge) is a SEPARATE follow-up step (see `docs/BENCHMARKING.md`
//! §"Quality"); this harness records raw generator outputs for that pass.
//!
//! # Generators (conditions)
//! - `mesh_adaptive` — the PTG mesh (4 columns + lateral voting + convergence).
//!   Primary artifact = ALL columns (unfiltered); accepted/rejected recorded.
//! - `mono_all_prompts` — 1 call with all 4 sphere prompts + the task.
//! - `mono_x4` — 4 independent monolithic calls (compute-matched to a 1-tick mesh).
//!
//! All conditions share: `temperature: 0.0`, a pinned seed, `response_format:
//! json_object`, the same message-role convention, and a shared metrics sink so
//! every call's `usage`/`cached_tokens`/`finish_reason`/latency is captured.
//!
//! # FAIRNESS
//! This harness is designed per `docs/BENCHMARKING.md`, which documents the four
//! Critical confounds (compute, prompt-budget, prefix-cache/token accounting,
//! survivorship) and their mitigations. Outputs are PILOT-scale: no headline
//! quality claim is made from this harness alone.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use clap::Parser;
use ptg_core::{
    ColumnOutputSchema, Stimulus, PROMPT_CODING, PROMPT_MATHEMATICS, PROMPT_PHYSICS,
    PROMPT_PSYCHOLOGY,
};
use ptg_runtime::default_mesh;
use ptg_vllm::{CollectorSink, EngineCallMetrics, InferenceEngine, MetricsSink};
use serde::Serialize;

/// Benchmark generation settings (fixed for determinism across all conditions).
const BENCH_TEMPERATURE: f32 = 0.0;

#[derive(Debug, Clone, Copy, clap::ValueEnum, Serialize, PartialEq, Eq)]
enum Condition {
    /// The PTG mesh: 4 sphere columns + lateral voting + convergence. Primary
    /// artifact = ALL columns (unfiltered). Stratify results by `ticks_run`.
    MeshAdaptive,
    /// One monolithic call with all 4 sphere prompts + the task (equal
    /// instruction *union*, plus a minimal combining instruction). 1 call.
    MonoAllPrompts,
    /// 4 IDENTICAL monolithic calls (same prompt). At temp 0 these are 4
    /// identical outputs — a degenerate compute-only control (C1) that does NOT
    /// separate prompt-diversity from the mechanism. Relegated to a secondary
    /// control by the methodology.
    MonoX4,
    /// 4 sphere-specialized calls with NO lateral context and NO voting
    /// (implemented as a 1-tick mesh). This is the PRIMARY mechanism control:
    /// `mesh_adaptive` = diverse columns + voting; `sphere_x4_no_lateral` =
    /// diverse columns − voting. The difference isolates the cortical
    /// mechanism from prompt diversity.
    SphereX4NoLateral,
}

impl Condition {
    const fn label(self) -> &'static str {
        match self {
            Self::MeshAdaptive => "mesh_adaptive",
            Self::MonoAllPrompts => "mono_all_prompts",
            Self::MonoX4 => "mono_x4",
            Self::SphereX4NoLateral => "sphere_x4_no_lateral",
        }
    }
}

#[derive(Parser)]
#[command(name = "ptg-bench", about = "PTG mesh-vs-monolithic benchmark harness")]
struct Args {
    /// Engine base URL.
    #[arg(long, default_value = "http://127.0.0.1:18135")]
    vllm_url: String,
    /// Model id.
    #[arg(long, default_value = "gemma-4-e4b")]
    model: String,
    /// Measured repeats per (prompt × condition). Pilot default 3.
    #[arg(long, default_value_t = 3)]
    repeats: u32,
    /// Max voting ticks for the mesh.
    #[arg(long, default_value_t = 2)]
    max_ticks: u32,
    /// Minimum ticks before convergence is considered (`mesh_adaptive` only).
    /// Forces lateral exchange to actually run even when an overconfident model
    /// would otherwise converge at tick 1. Must be <= --max-ticks.
    #[arg(long, default_value_t = 2)]
    min_ticks: u32,
    /// Fixed sampling seed (reproducibility; confound M4).
    #[arg(long, default_value_t = 20260623u64)]
    seed: u64,
    /// Per-column completion cap for the mesh (tokens).
    #[arg(long, default_value_t = 1024u32)]
    max_tokens_col: u32,
    /// Monolithic completion cap override (tokens). If set, used directly;
    /// otherwise = --max-tokens-col × --mono-budget-multiplier.
    #[arg(long)]
    max_tokens_mono: Option<u32>,
    /// Monolithic completion cap multiplier when --max-tokens-mono is unset.
    #[arg(long, default_value_t = 2)]
    mono_budget_multiplier: u32,
    /// Output root (a timestamped subdir is created inside it).
    #[arg(long, default_value = "bench-runs")]
    out_dir: PathBuf,
    /// Run only one condition (default: all three).
    #[arg(long, value_enum)]
    only: Option<Condition>,
    /// Skip the measured runs; just print the plan and prompt set (wiring check).
    #[arg(long)]
    dry_run: bool,
}

/// One answer-producing run.
#[derive(Serialize)]
struct AnswerRunRecord {
    record_type: &'static str,
    schema_version: u32,
    run_id: String,
    timestamp_unix_ms: u64,
    condition: String,
    prompt_id: String,
    repeat_index: u32,
    nonce: u64,
    server_url: String,
    model: String,
    temperature: f32,
    seed: Option<u64>,
    call_count: u32,
    /// Sum of gross `usage.prompt_tokens` across all calls (None if any unknown).
    prompt_tokens_gross: Option<u64>,
    completion_tokens_gross: Option<u64>,
    total_tokens_gross: Option<u64>,
    cached_tokens_gross: Option<u64>,
    /// `prompt_tokens_gross − cached_tokens_gross` (count each cached prefix
    /// once; completion tokens excluded). The prompt-portion cache adjustment.
    prompt_tokens_cache_adjusted: Option<u64>,
    /// `total_tokens_gross − cached_tokens_gross` = `(prompt−cached)+completion`
    /// = true compute-equivalent cost (completion is never cached).
    total_tokens_cache_adjusted: Option<u64>,
    cache_hit_rate: Option<f64>,
    finish_reasons: Vec<Option<String>>,
    /// Truncation flag: any call ended with `finish_reason == "length"`
    /// (robust to multi-choice responses).
    truncated: bool,
    /// End-to-end wall-clock latency of the condition run (ms).
    wall_latency_ms: u64,
    /// Sum of per-call HTTP latencies (compute-equivalent, ms).
    sum_call_latency_ms: u64,
    // Mesh-specific (None for monolithic conditions).
    ticks_run: Option<u32>,
    stabilized: Option<bool>,
    mean_confidence: Option<f32>,
    accepted_count: Option<u32>,
    rejected_count: Option<u32>,
    /// Confidence threshold used for the accepted/rejected split (recorded so
    /// the C4 ablation is self-contained). `None` for non-mesh conditions.
    integration_threshold: Option<f32>,
    parse_ok: bool,
    error: Option<String>,
    per_call: Vec<EngineCallMetrics>,
    /// Canonical representation of the generator's raw outputs (unfiltered).
    /// - mesh: array of `{column_id, sphere, schema}` for ALL columns.
    /// - mono_all_prompts: the raw content string.
    /// - mono_x4: array of the 4 raw content strings.
    outputs: serde_json::Value,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    let args = Args::parse();
    if args.min_ticks > args.max_ticks {
        return Err(format!(
            "--min-ticks ({}) must be <= --max-ticks ({})",
            args.min_ticks, args.max_ticks
        )
        .into());
    }
    let conditions: Vec<Condition> = match args.only {
        Some(c) => vec![c],
        None => vec![
            Condition::MeshAdaptive,
            Condition::SphereX4NoLateral,
            Condition::MonoAllPrompts,
            Condition::MonoX4,
        ],
    };
    let prompts = default_prompts();

    if args.dry_run {
        eprintln!("ptg-bench DRY RUN");
        eprintln!("  server:  {}", args.vllm_url);
        eprintln!("  model:   {}", args.model);
        eprintln!(
            "  conditions: {}",
            conditions
                .iter()
                .map(|c| c.label())
                .collect::<Vec<_>>()
                .join(", ")
        );
        eprintln!(
            "  repeats: {}, ticks: min={} max={}, seed: {}",
            args.repeats, args.min_ticks, args.max_ticks, args.seed
        );
        let mono_disp = args.max_tokens_mono.map_or_else(
            || {
                format!(
                    "{}x (col×{})",
                    args.mono_budget_multiplier, args.mono_budget_multiplier
                )
            },
            |m| m.to_string(),
        );
        eprintln!(
            "  max_tokens_col: {}, max_tokens_mono: {}",
            args.max_tokens_col, mono_disp
        );
        eprintln!("  prompts ({}):", prompts.len());
        for (id, p) in &prompts {
            eprintln!("    {id}: {:.80}", p.replace('\n', " "));
        }
        return Ok(());
    }

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let records = rt.block_on(run_bench(&args, &conditions, &prompts))?;

    let ts = epoch_ms();
    let out_root = args.out_dir.join(format!("{ts}"));
    fs::create_dir_all(&out_root)?;
    let jsonl_path = out_root.join("results.jsonl");
    let summary_path = out_root.join("summary.md");
    write_jsonl(&jsonl_path, &records)?;
    let summary = build_summary(&args, &records, ts);
    fs::write(&summary_path, summary)?;
    eprintln!("\nwrote {} run records", records.len());
    eprintln!("  raw:     {}", jsonl_path.display());
    eprintln!("  summary: {}", summary_path.display());
    Ok(())
}

async fn run_bench(
    args: &Args,
    conditions: &[Condition],
    prompts: &[(String, String)],
) -> Result<Vec<AnswerRunRecord>, Box<dyn std::error::Error>> {
    // Shared sink: both engines record here so ALL calls are captured.
    let sink: Arc<CollectorSink> = Arc::new(CollectorSink::new());

    let mesh_dyn: Arc<dyn MetricsSink> = sink.clone();
    let mono_dyn: Arc<dyn MetricsSink> = sink.clone();
    let mesh_engine = InferenceEngine::builder(&args.vllm_url, &args.model)
        .temperature(BENCH_TEMPERATURE)
        .max_tokens(args.max_tokens_col)
        .seed(Some(args.seed))
        .metrics_sink(mesh_dyn)
        .build()?;
    let mono_max = args.max_tokens_mono.unwrap_or_else(|| {
        args.max_tokens_col
            .saturating_mul(args.mono_budget_multiplier)
    });
    let mono_engine = Arc::new(
        InferenceEngine::builder(&args.vllm_url, &args.model)
            .temperature(BENCH_TEMPERATURE)
            .max_tokens(mono_max)
            .seed(Some(args.seed))
            .metrics_sink(mono_dyn)
            .build()?,
    );
    let mesh_engine: Arc<dyn ptg_vllm::ColumnEngine> = Arc::new(mesh_engine);

    // Warmup: one discarded run per condition on a dummy prompt (cold cache /
    // order effects — confound L2). Discard the metrics too.
    for &cond in conditions {
        let _ = run_one(
            cond,
            "WARMUP",
            0,
            0,
            "warmup probe; ignore",
            sink.as_ref(),
            &mesh_engine,
            &mono_engine,
            args,
        )
        .await;
    }
    if !sink.is_empty() {
        let _ = sink.drain();
    }

    let mut records = Vec::new();
    for repeat in 0..args.repeats {
        for (pidx, (pid, prompt)) in prompts.iter().enumerate() {
            // Deterministic order rotation so no condition is always first
            // (absorbs drift without a rand dependency).
            let mut order: Vec<Condition> = conditions.to_vec();
            let rot = (repeat as usize + pidx) % order.len().max(1);
            order.rotate_left(rot);
            for cond in order {
                let nonce = args
                    .seed
                    .saturating_mul(1_000_000)
                    .saturating_add(repeat as u64 * 1000)
                    .saturating_add(pidx as u64);
                let rec = run_one(
                    cond,
                    pid,
                    repeat,
                    nonce,
                    prompt,
                    sink.as_ref(),
                    &mesh_engine,
                    &mono_engine,
                    args,
                )
                .await;
                eprintln!(
                    "  [repeat {repeat}] {pid} / {}  wall={}ms calls={} tokens(cached)={:?} ok={}",
                    cond.label(),
                    rec.wall_latency_ms,
                    rec.call_count,
                    rec.cached_tokens_gross,
                    rec.parse_ok
                );
                records.push(rec);
            }
        }
    }
    Ok(records)
}

#[allow(clippy::too_many_arguments)]
async fn run_one(
    cond: Condition,
    pid: &str,
    repeat: u32,
    nonce: u64,
    prompt: &str,
    sink: &CollectorSink,
    mesh_engine: &Arc<dyn ptg_vllm::ColumnEngine>,
    mono_engine: &Arc<InferenceEngine>,
    args: &Args,
) -> AnswerRunRecord {
    // Drain any stale records so we capture exactly this run's calls.
    let _ = sink.drain();
    let wall_start = Instant::now();

    let mut rec = AnswerRunRecord {
        record_type: "answer_run",
        schema_version: 1,
        run_id: format!("{}_{}_{}_{}", cond.label(), pid, repeat, nonce),
        timestamp_unix_ms: epoch_ms(),
        condition: cond.label().to_string(),
        prompt_id: pid.to_string(),
        repeat_index: repeat,
        nonce,
        server_url: args.vllm_url.clone(),
        model: args.model.clone(),
        temperature: BENCH_TEMPERATURE,
        seed: Some(args.seed),
        call_count: 0,
        prompt_tokens_gross: None,
        completion_tokens_gross: None,
        total_tokens_gross: None,
        cached_tokens_gross: None,
        prompt_tokens_cache_adjusted: None,
        total_tokens_cache_adjusted: None,
        cache_hit_rate: None,
        finish_reasons: Vec::new(),
        truncated: false,
        wall_latency_ms: 0,
        sum_call_latency_ms: 0,
        ticks_run: None,
        stabilized: None,
        mean_confidence: None,
        accepted_count: None,
        rejected_count: None,
        integration_threshold: None,
        parse_ok: false,
        error: None,
        per_call: Vec::new(),
        outputs: serde_json::Value::Null,
    };

    // The nonce sits AFTER the static prompt prefix and BEFORE the task, so the
    // per-sphere system-prompt prefixes stay cacheable while repeated prompts
    // do not get a free full-prefix cache hit (prefix-cache protocol, §6).
    let stimulus = Stimulus::text(format!(
        "BENCH_RUN_ID: {nonce}; ignore this identifier.\n\nTASK:\n{prompt}"
    ));
    let outcome: Result<(serde_json::Value, MeshExtras), String> = match cond {
        Condition::MeshAdaptive => run_mesh(mesh_engine, &stimulus, args).await,
        Condition::SphereX4NoLateral => run_mesh_one_tick(mesh_engine, &stimulus).await,
        Condition::MonoAllPrompts => run_mono_once(mono_engine, &stimulus).await,
        Condition::MonoX4 => run_mono_x4(mono_engine, &stimulus).await,
    };
    let wall = wall_start.elapsed().as_millis() as u64;
    rec.wall_latency_ms = wall;

    let calls = sink.drain();
    aggregate_metrics(&mut rec, &calls);

    match outcome {
        Ok((outputs, extras)) => {
            rec.outputs = outputs;
            rec.ticks_run = extras.ticks_run;
            rec.stabilized = extras.stabilized;
            rec.mean_confidence = extras.mean_confidence;
            rec.accepted_count = extras.accepted_count;
            rec.rejected_count = extras.rejected_count;
            rec.integration_threshold = extras.integration_threshold;
            rec.parse_ok = true;
        }
        Err(msg) => {
            rec.error = Some(msg);
        }
    }
    rec
}

/// Mesh-specific fields carried out of a successful mesh run.
#[derive(Default)]
struct MeshExtras {
    ticks_run: Option<u32>,
    stabilized: Option<bool>,
    mean_confidence: Option<f32>,
    accepted_count: Option<u32>,
    rejected_count: Option<u32>,
    /// Integration threshold used for the accepted/rejected split (C4 ablation).
    integration_threshold: Option<f32>,
}

async fn run_mesh(
    engine: &Arc<dyn ptg_vllm::ColumnEngine>,
    stimulus: &Stimulus,
    args: &Args,
) -> Result<(serde_json::Value, MeshExtras), String> {
    let mut mesh = default_mesh(engine.clone()).map_err(|e| e.to_string())?;
    mesh.criteria.max_ticks = args.max_ticks;
    mesh.criteria.min_ticks = args.min_ticks;
    let threshold = mesh.criteria.min_integration_confidence;
    let result = mesh.run_epoch(stimulus).await.map_err(|e| e.to_string())?;
    let outputs: Vec<serde_json::Value> = result
        .outputs
        .iter()
        .map(|(id, schema)| canonical_column(id, schema, threshold))
        .collect();
    let extras = MeshExtras {
        ticks_run: Some(result.ticks_run),
        stabilized: Some(result.stabilized),
        mean_confidence: Some(result.mean_confidence),
        accepted_count: Some(result.accepted_outputs.len() as u32),
        rejected_count: Some(result.rejected_outputs.len() as u32),
        integration_threshold: Some(threshold),
    };
    Ok((serde_json::Value::Array(outputs), extras))
}

/// `sphere_x4_no_lateral` — the PRIMARY mechanism control. Runs the default mesh
/// with `max_ticks = 1`: 4 sphere-specialized column calls, empty lateral
/// context (no prior ticks), no voting/convergence. Reuses the same engine/path
/// and per-sphere validation as `mesh_adaptive`, so the ONLY difference vs a
/// 2-tick `mesh_adaptive` is the absence of lateral exchange + integration.
async fn run_mesh_one_tick(
    engine: &Arc<dyn ptg_vllm::ColumnEngine>,
    stimulus: &Stimulus,
) -> Result<(serde_json::Value, MeshExtras), String> {
    let mut mesh = default_mesh(engine.clone()).map_err(|e| e.to_string())?;
    mesh.criteria.max_ticks = 1;
    let threshold = mesh.criteria.min_integration_confidence;
    let result = mesh.run_epoch(stimulus).await.map_err(|e| e.to_string())?;
    let outputs: Vec<serde_json::Value> = result
        .outputs
        .iter()
        .map(|(id, schema)| canonical_column(id, schema, threshold))
        .collect();
    let extras = MeshExtras {
        ticks_run: Some(result.ticks_run),
        stabilized: Some(result.stabilized),
        mean_confidence: Some(result.mean_confidence),
        accepted_count: Some(result.accepted_outputs.len() as u32),
        rejected_count: Some(result.rejected_outputs.len() as u32),
        integration_threshold: Some(threshold),
    };
    Ok((serde_json::Value::Array(outputs), extras))
}

async fn run_mono_once(
    engine: &Arc<InferenceEngine>,
    stimulus: &Stimulus,
) -> Result<(serde_json::Value, MeshExtras), String> {
    let prompt = mono_prompt(stimulus);
    let (content, _m) = engine.complete(&prompt).await.map_err(|e| e.to_string())?;
    Ok((serde_json::Value::String(content), MeshExtras::default()))
}

async fn run_mono_x4(
    engine: &Arc<InferenceEngine>,
    stimulus: &Stimulus,
) -> Result<(serde_json::Value, MeshExtras), String> {
    let prompt = mono_prompt(stimulus);
    let mut out = Vec::with_capacity(4);
    for _ in 0..4 {
        let (content, _m) = engine.complete(&prompt).await.map_err(|e| e.to_string())?;
        out.push(serde_json::Value::String(content));
    }
    Ok((serde_json::Value::Array(out), MeshExtras::default()))
}

/// The monolithic prompt: all 4 sphere prompts (union instruction budget plus
/// a minimal combining instruction — NOT token-equal to the mesh; confound C2)
/// and the nonce-bearing task from the stimulus. `response_format: json_object`
/// is forced by the engine for all conditions.
fn mono_prompt(stimulus: &Stimulus) -> String {
    format!(
        "{phys}\n\n{math}\n\n{code}\n\n{psych}\n\n{task}\n\nReturn ONE JSON object with keys \"empirical\", \"mathematics\", \"coding\", \"psychology\" — each an object containing at least {{\"reference_frame_coordinates\": ..., \"prediction\": ..., \"confidence\": 0.0}} giving that perspective's analysis.",
        phys = PROMPT_PHYSICS,
        math = PROMPT_MATHEMATICS,
        code = PROMPT_CODING,
        psych = PROMPT_PSYCHOLOGY,
        task = stimulus.text_str(),
    )
}

fn canonical_column(id: &str, schema: &ColumnOutputSchema, threshold: f32) -> serde_json::Value {
    serde_json::json!({
        "column_id": id,
        "sphere": sphere_name(schema),
        "accepted": schema.confidence >= threshold,
        "schema": schema,
    })
}

/// Map a column's output to its perspective label, derived from which sphere-
/// specific key is present in the flattened `domain_fields` map.
fn sphere_name(schema: &ColumnOutputSchema) -> &'static str {
    let f = &schema.domain_fields;
    if f.contains_key("empirical_observation") {
        "empirical"
    } else if f.contains_key("deductive_synthesis") {
        "mathematics"
    } else if f.contains_key("algorithmic_analysis") {
        "coding"
    } else if f.contains_key("behavioral_synthesis") {
        "psychology"
    } else {
        "unknown"
    }
}

/// Roll drained per-call metrics into the aggregate fields of an answer run.
fn aggregate_metrics(rec: &mut AnswerRunRecord, calls: &[EngineCallMetrics]) {
    rec.call_count = calls.len() as u32;
    rec.sum_call_latency_ms = calls.iter().map(|c| c.latency_ms).sum();
    rec.finish_reasons = calls.iter().map(|c| c.finish_reason.clone()).collect();
    // Robust to multi-choice responses: flag truncation if ANY call truncated.
    rec.truncated = calls.iter().any(|c| c.truncated);

    rec.prompt_tokens_gross = sum_opt(calls.iter().map(|c| c.prompt_tokens));
    rec.completion_tokens_gross = sum_opt(calls.iter().map(|c| c.completion_tokens));
    rec.total_tokens_gross = sum_opt(calls.iter().map(|c| c.total_tokens));
    rec.cached_tokens_gross = sum_opt(calls.iter().map(|c| c.cached_tokens));
    rec.per_call = calls.to_vec();

    if let (Some(gross), Some(cached)) = (rec.prompt_tokens_gross, rec.cached_tokens_gross) {
        rec.cache_hit_rate = if gross == 0 {
            Some(0.0)
        } else {
            Some(cached as f64 / gross as f64)
        };
        // Prompt-portion cache adjustment: count each cached prefix once.
        rec.prompt_tokens_cache_adjusted = Some(gross.saturating_sub(cached));
    }
    if let (Some(total), Some(cached)) = (rec.total_tokens_gross, rec.cached_tokens_gross) {
        // Compute-equivalent total: (prompt−cached)+completion = total−cached.
        rec.total_tokens_cache_adjusted = Some(total.saturating_sub(cached));
    }
}

/// Sum a sequence of `Option<u32>` into `Option<u64>`; `None` if any is `None`.
fn sum_opt<I>(iter: I) -> Option<u64>
where
    I: IntoIterator<Item = Option<u32>>,
{
    let mut total: u64 = 0;
    for v in iter {
        total = total.saturating_add(v? as u64);
    }
    Some(total)
}

/// The pilot prompt set: diverse, multi-domain tasks (each exercises
/// Physics+Math+Coding+Psychology meaningfully). Used verbatim.
fn default_prompts() -> Vec<(String, String)> {
    vec![
        ("P1".to_string(),
         "At 14:03 a warehouse robot carrying a 12 kg parcel accelerated from 0.4 m/s to 1.6 m/s over 2.0 seconds, its battery temperature rose from 38C to 53C, the path-planning service retried the same command 7 times after a timeout, and a tired operator dismissed two warnings to keep throughput high. What is the most likely failure chain, and what immediate mitigation should be taken?".to_string()),
        ("P2".to_string(),
         "An online math tutor shows a student answers 18 of 20 arithmetic drills correctly when hints are disabled, but only 9 of 20 word problems. After a caching change, response latency rose to 2.4 seconds, and the student began clicking hints rapidly after two failures. Diagnose whether the main issue is conceptual, software-related, motivational, or a combination, and propose the next experiment.".to_string()),
        ("P3".to_string(),
         "A quadcopter rescue drone at 80 meters altitude faces a 12 m/s crosswind, battery charge is 22%, GPS position jumps by plus/minus 15 meters, the obstacle-avoidance loop was changed from O(n) to O(n^2) over 180 detected objects, and the pilot reports tunnel vision under time pressure. Should the mission continue, abort, or switch mode, and why?".to_string()),
        ("P4".to_string(),
         "A smart thermostat overshoots the target by 3C after a firmware update. Outdoor temperature dropped from 8C to 1C overnight, the heat pump draws 1.8 kW during recovery, a PID coefficient was changed in the scheduler, and two family members keep manually overriding the schedule because they feel cold in the morning. Explain the likely cause and the safest fix.".to_string()),
        ("P5".to_string(),
         "A wearable reports a runner's heart rate rising from 120 bpm to 178 bpm over 90 seconds while pace slows from 5:00/km to 6:30/km on a 7% grade. The anomaly detector recently switched to a longer smoothing window, and the runner wants to continue because the workout is part of a public challenge. What is probably happening, what might the software miss, and how should the system respond?".to_string()),
    ]
}

fn write_jsonl(path: &Path, records: &[AnswerRunRecord]) -> Result<(), Box<dyn std::error::Error>> {
    if records.is_empty() {
        fs::write(path, "")?;
        return Ok(());
    }
    let mut buf = String::new();
    for r in records {
        buf.push_str(&serde_json::to_string(r)?);
        buf.push('\n');
    }
    fs::write(path, buf)?;
    Ok(())
}

fn build_summary(args: &Args, records: &[AnswerRunRecord], ts: u64) -> String {
    let mut s = String::new();
    s.push_str("# PTG Benchmark — pilot run\n\n");
    s.push_str(&format!("- Generated (unix ms): {ts}\n"));
    s.push_str(&format!(
        "- Server: `{}`  Model: `{}`\n",
        args.vllm_url, args.model
    ));
    s.push_str(&format!(
        "- temperature: {}, seed: {:?}, ticks(min/max): {}/{}, max_tokens_col: {}, max_tokens_mono: {} ({})\n",
        BENCH_TEMPERATURE,
        Some(args.seed),
        args.min_ticks,
        args.max_ticks,
        args.max_tokens_col,
        args.max_tokens_mono.unwrap_or_else(|| args.max_tokens_col.saturating_mul(args.mono_budget_multiplier)),
        if args.max_tokens_mono.is_some() { "explicit" } else { "col×mult" },
    ));
    s.push_str(&format!(
        "- Conditions: {}  Repeats/prompt: {}\n",
        records
            .iter()
            .map(|r| r.condition.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .join(", "),
        args.repeats
    ));
    s.push_str(
        "\n> ⚠️ **PILOT — harness validation only.** Latency and token-economy numbers only. No headline quality claim.\n",
    );
    s.push_str(
        "> Quality (LLM-as-judge) is a separate follow-up step per `docs/BENCHMARKING.md`.\n\n",
    );
    s.push_str("### How NOT to read these numbers\n\n");
    s.push_str("- **Wall latency is not a speedup claim** — the server has 1 slot, so mesh fan-out serializes; use `sum_call_lat`, not `wall_lat`, for cross-condition latency.\n");
    s.push_str("- **Token economy has no quality meaning yet** — without a judge score, cost-effectiveness is uninterpretable.\n");
    s.push_str("- **Gross / cache-adjusted / cache-hit% must be read together** — no single token column is a fair headline cost.\n");
    s.push_str("- **`mean_confidence` and accept-rate are self-reported mechanism internals, NOT quality evidence.**\n");
    s.push_str("- **Only compare equal call counts** — a 1-tick mesh (4 calls) = `sphere_x4_no_lateral` (IDENTICAL calls, not just same count) = `mono_x4`; a 2-tick mesh (8 calls) is NOT comparable to `mono_x4`.\n");
    s.push_str("- **`mesh_adaptive` vs `sphere_x4_no_lateral` is a QUALITY comparison control, not a latency/token race** — equal call counts make it fair to compare, but a latency/token delta alone does NOT prove lateral voting helps.\n");
    s.push_str("- **A 1-tick mesh is NOT evidence of the lateral mechanism** — it is IDENTICAL to `sphere_x4_no_lateral`; only `ticks_run >= 2` runs exercise lateral exchange.\n");
    s.push_str("- **n is tiny** — pilot deltas are directional only; no headline claim until judge + ≥50-prompt scaled run.\n\n");

    s.push_str("## Per-condition aggregate (all prompts × repeats)\n\n");
    s.push_str("| condition | n | parse_ok | wall_lat ms (med/min/max) | sum_call_lat ms (med) | gross total tok (med) | cache-adj total tok (med) | cached% (med) | calls (med) | truncated |\n");
    s.push_str("|---|---|---|---|---|---|---|---|---|---|\n");
    let mut conds: Vec<&str> = records.iter().map(|r| r.condition.as_str()).collect();
    conds.sort();
    conds.dedup();
    for c in conds {
        let subset: Vec<&AnswerRunRecord> = records.iter().filter(|r| r.condition == c).collect();
        let n = subset.len();
        let ok = subset.iter().filter(|r| r.parse_ok).count();
        let wall: Vec<u64> = subset.iter().map(|r| r.wall_latency_ms).collect();
        let sumcall: Vec<u64> = subset.iter().map(|r| r.sum_call_latency_ms).collect();
        let gross: Vec<u64> = subset.iter().filter_map(|r| r.total_tokens_gross).collect();
        let cadj: Vec<u64> = subset
            .iter()
            .filter_map(|r| r.total_tokens_cache_adjusted)
            .collect();
        let cached_pct: Vec<f64> = subset
            .iter()
            .filter_map(|r| r.cache_hit_rate.map(|x| x * 100.0))
            .collect();
        let calls: Vec<u64> = subset.iter().map(|r| r.call_count as u64).collect();
        let trunc = subset.iter().filter(|r| r.truncated).count();
        s.push_str(&format!(
            "| {} | {n} | {ok} | {} / {} / {} | {} | {} | {} | {} | {} | {trunc} |\n",
            c,
            median(&wall),
            wall.iter().copied().min().unwrap_or(0),
            wall.iter().copied().max().unwrap_or(0),
            median(&sumcall),
            median_opt(&gross),
            median_opt(&cadj),
            median_f(&cached_pct),
            median(&calls),
        ));
    }

    // Mesh-specific detail, stratified by ticks_run (red-team guard: a 1-tick
    // run is NOT evidence of the lateral mechanism).
    for mesh_cond in ["mesh_adaptive", "sphere_x4_no_lateral"] {
        let mesh: Vec<&AnswerRunRecord> = records
            .iter()
            .filter(|r| r.condition == mesh_cond)
            .collect();
        if mesh.is_empty() {
            continue;
        }
        let title = if mesh_cond == "mesh_adaptive" {
            "mesh_adaptive"
        } else {
            "sphere_x4_no_lateral (mechanism control)"
        };
        s.push_str(&format!("\n## {title} detail\n\n"));
        let ticks: Vec<u32> = mesh.iter().filter_map(|r| r.ticks_run).collect();
        let stabilized = mesh.iter().filter(|r| r.stabilized == Some(true)).count();
        let meanconf: Vec<f64> = mesh
            .iter()
            .filter_map(|r| r.mean_confidence.map(|x| x as f64))
            .collect();
        s.push_str(&format!(
            "- ticks_run: min={} max={} | stabilized: {}/{} | mean_confidence (med): {} (self-reported, NOT quality)\n",
            ticks.iter().copied().min().unwrap_or(0),
            ticks.iter().copied().max().unwrap_or(0),
            stabilized,
            mesh.len(),
            median_f(&meanconf),
        ));

        // Stratification table by ticks_run.
        let mut tick_buckets: std::collections::BTreeMap<u32, Vec<&AnswerRunRecord>> =
            std::collections::BTreeMap::new();
        for r in &mesh {
            if let Some(t) = r.ticks_run {
                tick_buckets.entry(t).or_default().push(r);
            }
        }
        if tick_buckets.len() > 1 || !tick_buckets.is_empty() {
            s.push_str("\n| ticks_run | n | calls (med) | gross total tok (med) | total cache-adj tok (med) | wall_lat ms (med) | sum_call_lat ms (med) |\n");
            s.push_str("|---|---|---|---|---|---|---|\n");
            for (t, bucket) in &tick_buckets {
                let calls: Vec<u64> = bucket.iter().map(|r| r.call_count as u64).collect();
                let gross: Vec<u64> = bucket.iter().filter_map(|r| r.total_tokens_gross).collect();
                let cadj: Vec<u64> = bucket
                    .iter()
                    .filter_map(|r| r.total_tokens_cache_adjusted)
                    .collect();
                let wall: Vec<u64> = bucket.iter().map(|r| r.wall_latency_ms).collect();
                let sumc: Vec<u64> = bucket.iter().map(|r| r.sum_call_latency_ms).collect();
                s.push_str(&format!(
                    "| {t} | {} | {} | {} | {} | {} | {} |\n",
                    bucket.len(),
                    median(&calls),
                    median_opt(&gross),
                    median_opt(&cadj),
                    median(&wall),
                    median(&sumc),
                ));
            }
            s.push_str("\n> ⚠️ Only `ticks_run == 1` rows are call-matched to `sphere_x4_no_lateral` / `mono_x4`. Higher-tick mesh rows spend more calls and are NOT directly comparable.\n");
        }
    }

    // Silent-degeneracy guard: if mesh_adaptive never reaches ticks>=2, the
    // lateral mechanism was never exercised (every run == sphere_x4_no_lateral).
    let mesh_adaptive: Vec<&AnswerRunRecord> = records
        .iter()
        .filter(|r| r.condition == "mesh_adaptive")
        .collect();
    let max_tick_seen = mesh_adaptive
        .iter()
        .filter_map(|r| r.ticks_run)
        .max()
        .unwrap_or(0);
    if max_tick_seen <= 1 && !mesh_adaptive.is_empty() {
        s.push_str(&format!(
            "\n> 🛑 **MECHANISM UNMEASURED THIS ROUND.** All `mesh_adaptive` runs converged at tick 1 (max ticks_run = {max_tick_seen}), so they are IDENTICAL to `sphere_x4_no_lateral`. The lateral-voting mechanism was never exercised. To measure the mechanism, force runs that reach ticks >= 2 (e.g. raise `min_mean_confidence` or raise `--max-ticks` and disable early-stop).\n"
        ));
    }

    let parsefails = records.iter().filter(|r| !r.parse_ok).count();
    if parsefails > 0 {
        s.push_str(&format!("\n## Parse/validation failures: {parsefails}\n\nSee `error` fields in `results.jsonl`. Fail-fast on any malformed column means a mesh run can fail even if other columns are fine.\n"));
    }

    s.push_str("\n## Confound accounting (per `docs/BENCHMARKING.md`)\n\n");
    s.push_str("- **C1 compute**: `sphere_x4_no_lateral` (4 diverse sphere calls, no voting) is the PRIMARY mechanism control; `mono_x4` (4 identical calls, degenerate at temp 0) is a secondary compute-only control. Compare at equal call count.\n");
    s.push_str("- **C3 cache**: gross, prompt-cache-adjusted (`prompt−cached`), total-cache-adjusted (`total−cached`), and cache-hit% all reported per record.\n");
    s.push_str("- **C4 survivorship**: mesh `outputs` are ALL columns (unfiltered), each tagged `accepted`; `integration_threshold` recorded. Accepted-only is the ablation.\n");
    s.push_str("- Finish truncation (`finish_reason == length`, robust to multi-choice) flagged per condition.\n");
    s
}

fn median(xs: &[u64]) -> u64 {
    if xs.is_empty() {
        return 0;
    }
    let mut v = xs.to_vec();
    v.sort_unstable();
    v[v.len() / 2]
}

fn median_opt(xs: &[u64]) -> String {
    if xs.is_empty() {
        return "—".to_string();
    }
    median(xs).to_string()
}

fn median_f(xs: &[f64]) -> String {
    if xs.is_empty() {
        return "—".to_string();
    }
    let mut v = xs.to_vec();
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    format!("{:.1}", v[v.len() / 2])
}

fn epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
