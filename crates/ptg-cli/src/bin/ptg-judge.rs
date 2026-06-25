//! `ptg-judge` — the A2 quality pass for the PTG benchmark.
//!
//! Reads a `results.jsonl` produced by `ptg-bench` (does NOT rerun generators)
//! and emits a judge report. Per `docs/BENCHMARKING.md` ("A2 scope: mechanism
//! ACTIVATION, not consensus benefit"), the A2 question is narrow and honest:
//!
//!   *Does the lateral context injected on tick 2 ACTIVATE / PERTURB the
//!    lateral-receiving columns, and (corroborating) is the perturbed output
//!    perceived as better by an external judge?*
//!
//! It is explicitly NOT a thesis-level "the mesh reaches better consensus"
//! verdict (there is no consensus artifact in v0) and "improvement" is bounded
//! by the missing A3 no-lateral-second-tick control.
//!
//! Design (synthesized from a parallel planner + plan-reviewer audit):
//! 1. **PRIMARY signal — programmatic perturbation delta.** For each lateral-
//!    receiving column, a normalized edit distance between the tick-1 and
//!    tick-2 `prediction` strings. Zero judge confound — this is "did the
//!    lateral context make the column move at all?"
//! 2. **Determinism gate.** CC_PSYCH_01 has no incoming lateral edges, so at
//!    temperature 0 its tick-1 and tick-2 outputs must be byte-identical. If
//!    they differ, the whole run is flagged `determinism_failed` and excluded.
//! 3. **Echo / truncation exclusion.** Pairs whose tick-2 echoes a neighbor's
//!    prediction (de-blinding) or whose tick-2 truncated are excluded and
//!    counted, never scored as a loss.
//! 4. **LLM corroborating judge** (optional, external distinct-family model via
//!    OpenAI-compatible API — default Groq llama-3.3-70b). Blind pairwise,
//!    A/B swapped, third adjudication on disagreement. Rubric is the four
//!    CHECKABLE axes only (factual correctness, reasoning quality, reference-
//!    frame grounding, concision) — "integration" and "calibration" are dropped
//!    as circular/uncheckable under A2.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The lateral-receiving columns in the default topology (have neighbors).
const LATERAL_RECEIVERS: &[&str] = &["CC_PHYSICS_01", "CC_MATH_01", "CC_CODE_01"];
/// The graph sink — no incoming lateral edges → free determinism gate.
const DETERMINISM_COLUMN: &str = "CC_PSYCH_01";

/// Default-topology listener→sources map. An edge `(from, to)` in
/// `ptg_core::default_connections` means `from` LISTENS TO `to` (it receives
/// `to`'s prediction), because `establish_lateral_connection` appends `to` to
/// `adjacency_list[from]` and `lateral_context_for(id)` reads `adjacency_list[id]`.
/// These are therefore the columns whose tick-1 predictions get injected into
/// each receiver on tick 2 — exactly what the echo-screen must check against.
fn listener_sources() -> &'static [(&'static str, &'static [&'static str])] {
    &[
        ("CC_PHYSICS_01", &["CC_MATH_01"]),
        ("CC_MATH_01", &["CC_PHYSICS_01", "CC_CODE_01"]),
        ("CC_CODE_01", &["CC_PSYCH_01"]),
    ]
}

/// The sources a given receiver listens to in the default topology, or empty.
fn sources_for(receiver: &str) -> &'static [&'static str] {
    listener_sources()
        .iter()
        .find(|(r, _)| *r == receiver)
        .map(|(_, s)| *s)
        .unwrap_or(&[])
}

#[derive(Parser)]
#[command(
    name = "ptg-judge",
    about = "PTG benchmark A2 quality pass (activation, not consensus)"
)]
struct Args {
    /// `results.jsonl` from a `ptg-bench` run.
    #[arg(long, default_value = "results.jsonl")]
    input: PathBuf,
    /// Output report path (Markdown).
    #[arg(long, default_value = "judge-report.md")]
    out: PathBuf,
    /// JSONL of raw judge-call records (append).
    #[arg(long, default_value = "judge-calls.jsonl")]
    calls_out: PathBuf,
    /// Run the LLM corroborating judge (requires an external provider).
    #[arg(long)]
    judge: bool,
    /// Judge OpenAI-compatible base URL.
    #[arg(
        long,
        env = "PTG_JUDGE_API_URL",
        default_value = "https://api.groq.com/openai/v1"
    )]
    judge_api_url: String,
    /// Judge model id.
    #[arg(
        long,
        env = "PTG_JUDGE_MODEL",
        default_value = "llama-3.3-70b-versatile"
    )]
    judge_model: String,
    /// Environment variable holding the judge API key.
    #[arg(long, env = "PTG_JUDGE_API_KEY_ENV", default_value = "GROQ_API_KEY")]
    judge_api_key_env: String,
    /// Max LLM judge retries on parse failure (never score parse-fail as a loss).
    #[arg(long, default_value_t = 3u32)]
    judge_parse_retries: u32,
}

// ---------------------------------------------------------------------------
// Bench record shapes (the subset we read from results.jsonl)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct AnswerRun {
    condition: String,
    run_id: String,
    prompt_id: String,
    repeat_index: u32,
    parse_ok: bool,
    truncated: bool,
    ticks_run: Option<u32>,
    /// Per-tick snapshots (mesh-style conditions only).
    tick_outputs: Option<Vec<TickSnapshot>>,
}

#[derive(Debug, Deserialize)]
struct TickSnapshot {
    tick: u32,
    outputs: Vec<CanonicalColumn>,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)] // sphere/accepted are deserialized for completeness, read via serde in report
struct CanonicalColumn {
    column_id: String,
    sphere: String,
    accepted: bool,
    lateral_active: bool,
    schema: Value,
}

// ---------------------------------------------------------------------------
// Judge wire types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatRequestMessage>,
    temperature: f32,
    max_tokens: u32,
    response_format: ResponseFormat,
}

#[derive(Debug, Serialize)]
struct ChatRequestMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatRespMessage,
}

#[derive(Debug, Deserialize)]
struct ChatRespMessage {
    content: Option<String>,
}

/// The strict judge verdict schema (per planner design).
#[derive(Debug, Deserialize, Serialize, Clone)]
struct Verdict {
    winner: String,
    scores: BTreeMap<String, BTreeMap<String, i64>>,
    rationale: String,
}

/// One raw LLM judge call record (written to judge-calls.jsonl).
#[derive(Debug, Serialize)]
struct JudgeCallRecord {
    record_type: &'static str,
    judge_call_id: String,
    run_id: String,
    prompt_id: String,
    repeat_index: u32,
    column_id: String,
    tick_a: u32,
    tick_b: u32,
    presentation: String, // "tick2_A_tick1_B" etc.
    adjudication_index: u32,
    judge_model: String,
    latency_ms: u64,
    usage: Option<Value>,
    verdict: Option<Verdict>,
    parse_ok: bool,
    error: Option<String>,
}

// ---------------------------------------------------------------------------
// Per-pair analysis result (programmatic + optional judge)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct PairResult {
    run_id: String,
    prompt_id: String,
    repeat_index: u32,
    column_id: String,
    determinism_ok: bool,
    tick2_echoed_neighbor: bool,
    tick2_truncated: bool,
    excluded: bool,
    exclusion_reason: Option<String>,
    /// Normalized Levenshtein distance between tick-1 and tick-2 `prediction`.
    prediction_edit_distance: Option<f64>,
    /// Fraction of domain-field keys that differ between tick-1 and tick-2.
    domain_field_change_rate: Option<f64>,
    /// Confidence delta (tick2 − tick1). Self-reported → non-evidence, recorded only.
    confidence_delta: Option<f64>,
    /// Optional corroborating LLM verdict (normalized to tick_1/tick_2/tie).
    llm_winner_normalized: Option<String>,
    llm_score_gap: Option<i64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let text = fs::read_to_string(&args.input)?;
    let runs: Vec<AnswerRun> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    eprintln!("loaded {} run records", runs.len());

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let judge_client = if args.judge {
        let key = std::env::var(&args.judge_api_key_env).map_err(|_| {
            format!(
                "judge enabled but env var {} is not set",
                args.judge_api_key_env
            )
        })?;
        Some((
            reqwest::Client::builder()
                .timeout(Duration::from_secs(90))
                .build()?,
            key,
        ))
    } else {
        None
    };

    let mut pairs: Vec<PairResult> = Vec::new();
    let mut judge_calls: Vec<JudgeCallRecord> = Vec::new();

    for run in &runs {
        if run.condition != "mesh_adaptive" || !run.parse_ok || run.truncated {
            continue;
        }
        let Some(ticks) = &run.tick_outputs else {
            continue;
        };
        if run.ticks_run.unwrap_or(0) < 2 || ticks.len() < 2 {
            continue;
        }
        // Determinism gate: PSYCH tick-1 == tick-2 (canonical JSON).
        let determinism_ok = check_determinism(ticks);
        // Build per-column tick-1 / tick-2 lookups.
        let by_tick: BTreeMap<u32, &Vec<CanonicalColumn>> =
            ticks.iter().map(|t| (t.tick, &t.outputs)).collect();
        let tick1 = by_tick.get(&1);
        let tick2 = by_tick.get(&2);
        let (Some(t1), Some(t2)) = (tick1, tick2) else {
            continue;
        };
        let t1_map: BTreeMap<&str, &CanonicalColumn> =
            t1.iter().map(|c| (c.column_id.as_str(), c)).collect();
        let t2_map: BTreeMap<&str, &CanonicalColumn> =
            t2.iter().map(|c| (c.column_id.as_str(), c)).collect();

        for col_id in LATERAL_RECEIVERS {
            let col_id: &str = col_id;
            let (Some(c1), Some(c2)) = (t1_map.get(col_id), t2_map.get(col_id)) else {
                continue;
            };
            // Echo screen against the ACTUAL injected neighbor (source) text, not
            // the receiver's own tick-1. Gather each source's tick-1 prediction.
            let source_preds: Vec<&str> = sources_for(col_id)
                .iter()
                .filter_map(|src| t1_map.get(*src))
                .filter_map(|c| c.schema.get("prediction").and_then(Value::as_str))
                .collect();
            let echoed = c2.lateral_active && echo_screen_against_sources(c2, &source_preds);
            let tick2_trunc = run.truncated;
            let mut excluded = false;
            let mut reason = None;
            if !determinism_ok {
                excluded = true;
                reason = Some("determinism_failed".into());
            } else if tick2_trunc {
                excluded = true;
                reason = Some("tick2_truncated".into());
            } else if echoed {
                excluded = true;
                reason = Some("tick2_echoed_neighbor".into());
            }
            let pred_ed = if !excluded {
                prediction_edit_distance(c1, c2)
            } else {
                None
            };
            let dfc = if !excluded {
                domain_field_change_rate(c1, c2)
            } else {
                None
            };
            let conf_delta = confidence_delta(c1, c2);

            // LLM corroborating judge (only for non-excluded pairs, when enabled).
            let (llm_winner, llm_gap, calls_for_pair) = if !excluded {
                if let Some((client, key)) = judge_client.as_ref() {
                    rt.block_on(run_judge_pair(client, key, &args, run, col_id, c1, c2))?
                } else {
                    (None, None, Vec::new())
                }
            } else {
                (None, None, Vec::new())
            };
            judge_calls.extend(calls_for_pair);

            pairs.push(PairResult {
                run_id: run.run_id.clone(),
                prompt_id: run.prompt_id.clone(),
                repeat_index: run.repeat_index,
                column_id: col_id.to_string(),
                determinism_ok,
                tick2_echoed_neighbor: echoed,
                tick2_truncated: tick2_trunc,
                excluded,
                exclusion_reason: reason,
                prediction_edit_distance: pred_ed,
                domain_field_change_rate: dfc,
                confidence_delta: conf_delta,
                llm_winner_normalized: llm_winner,
                llm_score_gap: llm_gap,
            });
        }
    }

    // Write outputs.
    let report = build_report(&args, &pairs);
    fs::write(&args.out, report)?;
    write_judge_calls(&args.calls_out, &judge_calls)?;
    eprintln!(
        "analyzed {} pairs | wrote {} and {}",
        pairs.len(),
        args.out.display(),
        args.calls_out.display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Determinism gate
// ---------------------------------------------------------------------------

fn check_determinism(ticks: &[TickSnapshot]) -> bool {
    let mut by_tick: BTreeMap<u32, &Vec<CanonicalColumn>> =
        ticks.iter().map(|t| (t.tick, &t.outputs)).collect();
    let t1 = by_tick.remove(&1);
    let t2 = by_tick.remove(&2);
    let (Some(t1), Some(t2)) = (t1, t2) else {
        return false;
    };
    let (Some(p1), Some(p2)) = (find_determinism(t1), find_determinism(t2)) else {
        return false;
    };
    canonical(&p1.schema) == canonical(&p2.schema)
}

/// Find the determinism column (PSYCH) in a tick's output set.
fn find_determinism(cols: &[CanonicalColumn]) -> Option<&CanonicalColumn> {
    cols.iter().find(|c| c.column_id == DETERMINISM_COLUMN)
}

/// Canonical JSON string for equality comparison (sorted keys, compact).
fn canonical(v: &Value) -> String {
    let mut sorted = v.clone();
    sort_object_keys(&mut sorted);
    serde_json::to_string(&sorted).unwrap_or_default()
}

fn sort_object_keys(v: &mut Value) {
    match v {
        Value::Object(map) => {
            // Recurse into values first.
            for (_, val) in map.iter_mut() {
                sort_object_keys(val);
            }
            // BTreeMap sorts keys; rebuild.
            let bt: BTreeMap<String, Value> = map
                .iter()
                .map(|(k, val)| (k.clone(), val.clone()))
                .collect();
            *map = serde_json::Map::from_iter(bt);
        }
        Value::Array(arr) => {
            for val in arr.iter_mut() {
                sort_object_keys(val);
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Programmatic perturbation metrics
// ---------------------------------------------------------------------------

/// Normalized Levenshtein distance between the two `prediction` strings, in
/// [0, 1]. `None` if either schema lacks a string `prediction`.
fn prediction_edit_distance(c1: &CanonicalColumn, c2: &CanonicalColumn) -> Option<f64> {
    let p1 = c1.schema.get("prediction")?.as_str()?;
    let p2 = c2.schema.get("prediction")?.as_str()?;
    let d = levenshtein(p1, p2) as f64;
    let max_len = p1.len().max(p2.len()).max(1) as f64;
    Some(d / max_len)
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr: Vec<usize> = vec![0; b.len() + 1];
    for i in 1..=a.len() {
        curr[0] = i;
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

/// Fraction of domain-field keys whose JSON value differs between tick-1 and
/// tick-2 (0.0 = identical, 1.0 = all changed).
fn domain_field_change_rate(c1: &CanonicalColumn, c2: &CanonicalColumn) -> Option<f64> {
    let d1 = c1.schema.get("domain_fields")?.as_object()?;
    let d2 = c2.schema.get("domain_fields")?.as_object()?;
    if d1.is_empty() && d2.is_empty() {
        return None;
    }
    let keys: std::collections::BTreeSet<&String> = d1.keys().chain(d2.keys()).collect();
    let changed = keys.iter().filter(|k| d1.get(**k) != d2.get(**k)).count();
    Some(changed as f64 / keys.len().max(1) as f64)
}

fn confidence_delta(c1: &CanonicalColumn, c2: &CanonicalColumn) -> Option<f64> {
    let a = c1.schema.get("confidence")?.as_f64()?;
    let b = c2.schema.get("confidence")?.as_f64()?;
    Some(b - a)
}

/// Echo screen: flag if the receiver's tick-2 `prediction` contains a long
/// verbatim run of any of its SOURCE neighbors' tick-1 predictions. The lateral
/// context injected on tick 2 embeds each source's prediction verbatim (see
/// `lateral_context_for`), so a literal echo de-blinds the judge toward tick-2.
/// NOTE: compares against the actual injected NEIGHBOR text (sources), NOT the
/// receiver's own tick-1 (that detects self-repetition, not leakage).
/// Heuristic and intentionally conservative: any 40-char window of a source's
/// tick-1 prediction appearing in the receiver's tick-2 prediction.
fn echo_screen_against_sources(c2: &CanonicalColumn, source_tick1_predictions: &[&str]) -> bool {
    let Some(p2) = c2.schema.get("prediction").and_then(Value::as_str) else {
        return false;
    };
    for src in source_tick1_predictions {
        // Slide a 40-CHAR (not byte) window: model predictions routinely contain
        // multibyte UTF-8 (em dashes, smart quotes, °, µ, CJK), and byte slicing
        // `src[i..i+40]` would panic on a non-char-boundary.
        let chars: Vec<char> = src.chars().collect();
        if chars.len() < 40 {
            continue;
        }
        for window in chars.windows(40) {
            let needle: String = window.iter().collect();
            if p2.contains(&needle) {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// LLM corroborating judge
// ---------------------------------------------------------------------------

async fn run_judge_pair(
    client: &reqwest::Client,
    api_key: &str,
    args: &Args,
    run: &AnswerRun,
    column_id: &str,
    c1: &CanonicalColumn,
    c2: &CanonicalColumn,
) -> Result<(Option<String>, Option<i64>, Vec<JudgeCallRecord>), Box<dyn std::error::Error>> {
    let mut calls = Vec::new();
    // Two judgments with swapped order.
    let order1 = (2u32, 1u32); // tick2 as A, tick1 as B
    let order2 = (1u32, 2u32); // swapped
    let mut verdicts: Vec<(String, i64)> = Vec::new();
    for (idx, (ta, tb)) in [order1, order2].iter().enumerate() {
        let (a, b) = if *ta == 1 { (c1, c2) } else { (c2, c1) };
        let pres = format!("tick{}_A_tick{}_B", ta, tb);
        let call_id = format!("{}_{}_{}", run.run_id, column_id, idx);
        match judge_once(
            client,
            api_key,
            args,
            column_id,
            a,
            b,
            *ta,
            *tb,
            idx as u32 + 1,
            &call_id,
            &run.run_id,
            &run.prompt_id,
            run.repeat_index,
        )
        .await
        {
            Ok((verdict, latency, usage, record)) => {
                let (norm, gap) = normalize_verdict(&verdict, *ta);
                verdicts.push((norm, gap));
                calls.push(record);
                let _ = (latency, usage);
            }
            Err(e) => {
                calls.push(JudgeCallRecord {
                    record_type: "judge_call",
                    judge_call_id: call_id,
                    run_id: run.run_id.clone(),
                    prompt_id: run.prompt_id.clone(),
                    repeat_index: run.repeat_index,
                    column_id: column_id.to_string(),
                    tick_a: *ta,
                    tick_b: *tb,
                    presentation: pres,
                    adjudication_index: idx as u32 + 1,
                    judge_model: args.judge_model.clone(),
                    latency_ms: 0,
                    usage: None,
                    verdict: None,
                    parse_ok: false,
                    error: Some(e),
                });
            }
        }
    }

    // Disagreement → third adjudication (fresh order).
    let disagree = verdicts.len() >= 2
        && (verdicts[0].0 != verdicts[1].0
            || (verdicts[0].1.signum() != verdicts[1].1.signum()
                && verdicts[0].1 != 0
                && verdicts[1].1 != 0));
    if disagree && verdicts.len() == 2 {
        let (ta, tb) = (1u32, 2u32);
        let call_id = format!("{}_{}_adj", run.run_id, column_id);
        match judge_once(
            client,
            api_key,
            args,
            column_id,
            c1,
            c2,
            ta,
            tb,
            3,
            &call_id,
            &run.run_id,
            &run.prompt_id,
            run.repeat_index,
        )
        .await
        {
            Ok((verdict, _, _, record)) => {
                let (norm, gap) = normalize_verdict(&verdict, ta);
                verdicts.push((norm, gap));
                calls.push(record);
            }
            Err(e) => {
                calls.push(JudgeCallRecord {
                    record_type: "judge_call",
                    judge_call_id: call_id,
                    run_id: run.run_id.clone(),
                    prompt_id: run.prompt_id.clone(),
                    repeat_index: run.repeat_index,
                    column_id: column_id.to_string(),
                    tick_a: ta,
                    tick_b: tb,
                    presentation: "tick1_A_tick2_B".into(),
                    adjudication_index: 3,
                    judge_model: args.judge_model.clone(),
                    latency_ms: 0,
                    usage: None,
                    verdict: None,
                    parse_ok: false,
                    error: Some(e),
                });
            }
        }
    }

    // Majority normalized winner (3-way split → tie).
    let winner = if verdicts.is_empty() {
        None
    } else {
        let mut counts: BTreeMap<String, i32> = BTreeMap::new();
        for (w, _) in &verdicts {
            *counts.entry(w.clone()).or_default() += 1;
        }
        let max = counts.values().copied().max().unwrap_or(0);
        let leaders: Vec<&String> = counts
            .iter()
            .filter(|(_, v)| **v == max)
            .map(|(k, _)| k)
            .collect();
        if leaders.len() == 1 {
            Some(leaders[0].clone())
        } else {
            Some("tie".to_string())
        }
    };
    let gap = verdicts.iter().map(|(_, g)| *g).sum::<i64>() / verdicts.len().max(1) as i64;
    Ok((
        winner,
        if verdicts.is_empty() { None } else { Some(gap) },
        calls,
    ))
}

/// Normalize a verdict back to {tick_1, tick_2, tie} given which tick was "A".
fn normalize_verdict(verdict: &Verdict, tick_a: u32) -> (String, i64) {
    let a_sum: i64 = verdict
        .scores
        .get("A")
        .map(|m| m.values().sum())
        .unwrap_or(0);
    let b_sum: i64 = verdict
        .scores
        .get("B")
        .map(|m| m.values().sum())
        .unwrap_or(0);
    let norm = match verdict.winner.as_str() {
        "A" => {
            if tick_a == 1 {
                "tick_1"
            } else {
                "tick_2"
            }
        }
        "B" => {
            if tick_a == 1 {
                "tick_2"
            } else {
                "tick_1"
            }
        }
        _ => "tie",
    };
    // gap = tick_2_score − tick_1_score (positive ⇒ tick_2 scored higher).
    let (s2, s1) = if tick_a == 1 {
        (b_sum, a_sum)
    } else {
        (a_sum, b_sum)
    };
    (norm.to_string(), s2 - s1)
}

#[allow(clippy::too_many_arguments)]
async fn judge_once(
    client: &reqwest::Client,
    api_key: &str,
    args: &Args,
    column_id: &str,
    a: &CanonicalColumn,
    b: &CanonicalColumn,
    tick_a: u32,
    tick_b: u32,
    adjudication_index: u32,
    call_id: &str,
    run_id: &str,
    prompt_id: &str,
    repeat_index: u32,
) -> Result<(Verdict, u64, Option<Value>, JudgeCallRecord), String> {
    let prompt = build_judge_prompt(column_id, a, b);
    let req = ChatRequest {
        model: &args.judge_model,
        messages: vec![ChatRequestMessage {
            role: "user",
            content: prompt,
        }],
        temperature: 0.0,
        max_tokens: 600,
        response_format: ResponseFormat {
            kind: "json_object",
        },
    };
    let endpoint = format!("{}/chat/completions", args.judge_api_url);

    let mut last_err: Option<String> = None;
    for attempt in 0..args.judge_parse_retries.max(1) {
        let start = Instant::now();
        let send = client
            .post(&endpoint)
            .bearer_auth(api_key)
            .json(&req)
            .send()
            .await;
        match send {
            Ok(resp) => {
                let status = resp.status();
                let latency = start.elapsed().as_millis() as u64;
                let body_text = match resp.text().await {
                    Ok(t) => t,
                    Err(e) => {
                        last_err = Some(format!("read body: {e}"));
                        continue;
                    }
                };
                let parsed: Option<ChatResponse> = serde_json::from_str(&body_text).ok();
                let usage = parsed.as_ref().and_then(|p| p.usage.clone());
                if !status.is_success() {
                    last_err = Some(format!("HTTP {status}"));
                    continue;
                }
                let content = parsed
                    .as_ref()
                    .and_then(|p| p.choices.first())
                    .and_then(|c| c.message.content.clone());
                if let Some(ref c) = content {
                    if let Ok(v) = parse_verdict(c) {
                        let record = JudgeCallRecord {
                            record_type: "judge_call",
                            judge_call_id: call_id.to_string(),
                            run_id: run_id.to_string(),
                            prompt_id: prompt_id.to_string(),
                            repeat_index,
                            column_id: column_id.to_string(),
                            tick_a,
                            tick_b,
                            presentation: format!("tick{}_A_tick{}_B", tick_a, tick_b),
                            adjudication_index,
                            judge_model: args.judge_model.clone(),
                            latency_ms: latency,
                            usage: usage.clone(),
                            verdict: Some(v.clone()),
                            parse_ok: true,
                            error: None,
                        };
                        return Ok((v, latency, usage, record));
                    }
                }
                last_err = Some("verdict did not parse".to_string());
            }
            Err(e) => {
                last_err = Some(format!("request: {e}"));
            }
        }
        let _ = attempt;
    }
    Err(last_err.unwrap_or_else(|| "unknown judge error".to_string()))
}

/// Parse the judge's JSON content into a Verdict, tolerant of code-fence wrappers.
fn parse_verdict(content: &str) -> Result<Verdict, serde_json::Error> {
    let s = content.trim();
    let s = s.strip_prefix("```json").unwrap_or(s);
    let s = s.strip_prefix("```").unwrap_or(s);
    let s = s.strip_suffix("```").unwrap_or(s);
    let s = s.trim();
    let start = s.find('{');
    let end = s.rfind('}');
    if let (Some(a), Some(b)) = (start, end) {
        if a <= b {
            return serde_json::from_str(&s[a..=b]);
        }
    }
    serde_json::from_str(s)
}

fn build_judge_prompt(column_id: &str, a: &CanonicalColumn, b: &CanonicalColumn) -> String {
    let (sphere, remit) = sphere_remit(column_id);
    format!(
        "You are an independent benchmark judge evaluating two BLINDED drafts of the same \
reference-frame analysis. The labels \"Answer A\" and \"Answer B\" are arbitrary; do not infer \
which is which or anything from order. These are two drafts of an analysis for the SAME task.\n\n\
Reference frame being judged:\n- column: {column_id}\n- frame: {sphere}\n- remit: {remit}\n\n\
Each draft is a JSON object with reference_frame_coordinates, prediction, confidence, and \
sphere-specific domain fields.\n\n\
Judging rules (STRICT):\n\
- Do NOT reward verbosity, length, repetition, formatting, or JSON labels.\n\
- Do NOT reward an answer for having more words or more fields.\n\
- A concise answer can beat a longer one. Prefer high-signal analytical content.\n\
- Penalize hallucinated facts, unsupported causal claims, contradictions of the task, and filler.\n\
- Judge ONLY the content shown; do not speculate about which draft had extra context.\n\n\
Score each answer 1-5 on these FOUR dimensions (only):\n\
1. factual_correctness: 1 = contradicts/invents important facts; 5 = preserves stated facts/numbers, no unsupported claims.\n\
2. reasoning_quality: 1 = incoherent, restatement-only, or causally invalid; 5 = clear plausible causal/logical chain.\n\
3. reference_frame_grounding: 1 = generic prose outside the assigned frame; 5 = strongly grounded in the {sphere} frame.\n\
4. concision_signal: 1 = verbose filler; 5 = concise, high-signal, no unnecessary verbosity.\n\n\
Answer A:\n{a}\n\nAnswer B:\n{b}\n\n\
Return ONLY a valid JSON object with exactly this shape:\n\
{{\"winner\":\"A\"|\"B\"|\"tie\",\"scores\":{{\"A\":{{\"factual_correctness\":N,\"reasoning_quality\":N,\"reference_frame_grounding\":N,\"concision_signal\":N}},\"B\":{{...same...}}}},\"rationale\":\"1-3 sentences\"}}",
        a = serde_json::to_string_pretty(&a.schema).unwrap_or_default(),
        b = serde_json::to_string_pretty(&b.schema).unwrap_or_default(),
    )
}

fn sphere_remit(column_id: &str) -> (&'static str, &'static str) {
    match column_id {
        "CC_PHYSICS_01" => ("empirical", "forces, energy, thermal/mechanical constraints; isolated variables and empirical observations"),
        "CC_MATH_01" => ("mathematics", "quantities, rates, ratios, complexity, formal relations; axiomatic assertions and deductive synthesis"),
        "CC_CODE_01" => ("coding", "state, retries, latency, control flow, algorithmic/software failure modes; state variables and algorithmic analysis"),
        _ => ("unknown", "the assigned reference frame"),
    }
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

fn build_report(args: &Args, pairs: &[PairResult]) -> String {
    let mut s = String::new();
    s.push_str("# PTG Benchmark — A2 judge report\n\n");
    s.push_str(
        "> **A2 scope: mechanism ACTIVATION ONLY — not consensus, not improvement, not calibration.**\n",
    );
    s.push_str(
        "> This report answers only: *did the lateral context injected on tick 2 make the\n",
    );
    s.push_str(
        "> receiving columns MOVE?* It does NOT show the mesh reaches consensus, that lateral\n",
    );
    s.push_str(
        "> context IMPROVES outputs, or anything about quality. See `docs/BENCHMARKING.md`.\n\n",
    );
    s.push_str("### How NOT to read this report\n\n");
    s.push_str(
        "- **Edit distance is NOT quality.** A 0.75 perturbation means the output CHANGED a lot,\n",
    );
    s.push_str("  not that it got better. A column can change a lot and get worse.\n");
    s.push_str(
        "- **LLM winner counts are NOT the headline and are NOT a quality verdict.** They are\n",
    );
    s.push_str(
        "  directional noise from a corroborating signal; do not cite them as proof of anything.\n",
    );
    s.push_str("- **`determinism_ok` checks only `CC_PSYCH_01` per run** (it has no lateral inputs); it does\n");
    s.push_str("  NOT prove cross-run determinism for the receiving columns.\n");
    s.push_str("- **`confidence_delta` is self-reported → NON-evidence** for quality or calibration; near-zero\n");
    s.push_str("  may reflect overconfidence ceiling effect, not good calibration.\n");
    s.push_str(
        "- **Echo exclusion is a crude 40-char heuristic;** survivors may still be de-blinded.\n\n",
    );
    s.push_str(&format!("- Input: `{}`\n", args.input.display()));
    s.push_str(&format!("- Pairs analyzed: {}\n", pairs.len()));
    let excluded = pairs.iter().filter(|p| p.excluded).count();
    s.push_str(&format!("- Excluded pairs: {}\n", excluded));

    // Exclusion breakdown.
    s.push_str("\n## Exclusion breakdown\n\n");
    let mut exc: BTreeMap<String, i32> = BTreeMap::new();
    for p in pairs {
        if let Some(r) = &p.exclusion_reason {
            *exc.entry(r.clone()).or_default() += 1;
        }
    }
    if exc.is_empty() {
        s.push_str("(none)\n");
    } else {
        for (k, v) in &exc {
            s.push_str(&format!("- {k}: {v}\n"));
        }
        s.push_str(
            "\n> Echo (`tick2_echoed_neighbor`) exclusion is a conservative 40-char-substring heuristic;\n",
        );
        s.push_str(
            "> it catches verbatim neighbor-text leaks only. A nonzero count here signals non-trivial\n",
        );
        s.push_str("> leakage risk, and survivors may still be de-blinded by paraphrase.\n");
    }

    // PRIMARY: programmatic perturbation delta (non-excluded pairs only).
    let scored: Vec<&PairResult> = pairs.iter().filter(|p| !p.excluded).collect();
    s.push_str("\n## Primary: programmatic perturbation delta (non-excluded)\n\n");
    s.push_str("> Zero judge confound. \"Did the lateral context injected on tick 2 change the\n");
    s.push_str(
        "> column's output at all?\" Normalized edit distance on `prediction`; domain-field\n",
    );
    s.push_str(
        "> change rate. A nonzero delta = activation; its size is the perturbation magnitude.\n",
    );
    s.push_str(
        "> **⚠️ Edit distance = perturbation magnitude, NOT quality.** A column can change a lot\n",
    );
    s.push_str(
        "> and get worse. domain-field change counts CHANGED KEYS, not correctness; `—` = no\n",
    );
    s.push_str("> comparable fields.\n\n");
    if scored.is_empty() {
        s.push_str("(no non-excluded pairs)\n");
    } else {
        s.push_str("| column | n | prediction edit dist (med) | domain-field change % (med) | mean conf delta (self-report, NON-evidence) |\n");
        s.push_str("|---|---|---|---|---|\n");
        for col in LATERAL_RECEIVERS {
            let cs: Vec<&PairResult> = scored
                .iter()
                .copied()
                .filter(|p| p.column_id == *col)
                .collect();
            if cs.is_empty() {
                continue;
            }
            let mut ed: Vec<f64> = cs
                .iter()
                .filter_map(|p| p.prediction_edit_distance)
                .collect();
            ed.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mut dfc: Vec<f64> = cs
                .iter()
                .filter_map(|p| p.domain_field_change_rate)
                .collect();
            dfc.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let cd: Vec<f64> = cs.iter().filter_map(|p| p.confidence_delta).collect();
            let med = |v: &[f64]| {
                if v.is_empty() {
                    String::from("—")
                } else {
                    format!("{:.3}", v[v.len() / 2])
                }
            };
            let mean = |v: &[f64]| {
                if v.is_empty() {
                    String::from("—")
                } else {
                    format!("{:+.3}", v.iter().sum::<f64>() / v.len() as f64)
                }
            };
            s.push_str(&format!(
                "| {} | {} | {} | {} | {} |\n",
                col,
                cs.len(),
                med(&ed),
                med(&dfc),
                mean(&cd)
            ));
        }
    }

    // CORROBORATING: LLM judge (if any pairs judged).
    let judged: Vec<&PairResult> = scored
        .iter()
        .copied()
        .filter(|p| p.llm_winner_normalized.is_some())
        .collect();
    s.push_str("\n## Corroborating: LLM judge (if run)\n\n");
    if judged.is_empty() {
        s.push_str("(LLM judge not run — pass `--judge` and set the judge API key env var (default `GROQ_API_KEY`)\n");
    } else {
        s.push_str("**⚠️ NOT the primary result. Directional noise only — do not cite as a quality verdict.**\n\n");
        s.push_str(&format!(
            "- judge model: `{}` via `{}`\n",
            args.judge_model, args.judge_api_url
        ));
        let mut wins: BTreeMap<String, i32> = BTreeMap::new();
        for p in &judged {
            if let Some(w) = &p.llm_winner_normalized {
                *wins.entry(w.clone()).or_default() += 1;
            }
        }
        s.push_str("- normalized winners: ");
        s.push_str(&format!(
            "{{ {} }}\n",
            wins.iter()
                .map(|(k, v)| format!("{k}: {v}"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        s.push_str(
            "\n> Reminder: tick_2 winning does NOT prove lateral *improvement* — it may be\n",
        );
        s.push_str(
            "> 'a second look' (J1) or length/echo bias (J2/J3). Treat as directional only.\n",
        );
    }

    s.push_str("\n## Honest bounds\n\n");
    s.push_str(&format!("- n per column is tiny ({} judged pairs total); any LLM winner split is within swap-disagreement noise — no statistical claim.\n", judged.len()));
    s.push_str("- 3 columns is the generalization ceiling (lateral activates *these domains*, not 'columns' in general).\n");
    s.push_str("- `confidence_delta` is self-reported → NON-evidence for quality or calibration (near-zero may be overconfidence ceiling).\n");
    s.push_str("- `determinism_ok` is per-run, `CC_PSYCH_01`-only; it does NOT establish cross-run determinism for the receiving columns.\n");
    s.push_str("- Echo-exclusion survivors may still be de-blinded by paraphrase; the survivor set may be biased toward columns that ignored lateral context.\n");
    s.push_str("- A clean 'lateral IMPROVES' claim requires the A3 no-lateral-second-tick control (not yet built).\n");
    s
}

fn write_judge_calls(
    path: &Path,
    calls: &[JudgeCallRecord],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut buf = String::new();
    for c in calls {
        buf.push_str(&serde_json::to_string(c)?);
        buf.push('\n');
    }
    fs::write(path, buf)?;
    Ok(())
}
