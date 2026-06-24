# PTG Benchmark Methodology

Status: **committed design — under review.** No benchmark numbers are to be
produced or cited until this document and the engine instrumentation are
reviewed (see "Review gate" below).

## Objective

Answer one question as honestly as the hardware allows: **is the PTG cortical
mechanism (specialized columns + lateral voting + confidence integration) better
than (a) a single monolithic context and (b) an undifferentiated compute-matched
ensemble, on latency, token economy, and answer quality?**

We commit to a result that may say **"the mechanism does not help / is
indistinguishable from more compute."** A benchmark that can only flatter the
mesh is worthless.

## Server facts (measured)

- `llama-server` at `http://127.0.0.1:18135`, model `gemma-4-e4b`, OpenAI-compatible.
- **Parallelism: `-np` absent → 1 slot.** Concurrent mesh column calls **serialize**
  at the server; wall-clock mesh latency ≈ sum of per-call latency, not a speedup.
- **Context window: `-c 4096`.** The monolithic baseline's concatenated prompt
  budget must fit within this.
- **Prefix caching: ACTIVE.** `usage.prompt_tokens_details.cached_tokens` is
  reported per call and is nonzero on repeated prefixes.
- **No embeddings endpoint** (`/v1/embeddings` → HTTP 501). Semantic-agreement
  metrics are **not possible on this server**; no "consensus" claim may be made
  from confidence alone.

## The four Critical confounds and how we neutralize them

These are adapted from a fresh-context confound audit; each independently biases
a naive comparison toward the mesh and must be neutralized.

| ID | Confound | Mitigation in this design |
|----|----------|---------------------------|
| **C1** | "More compute = better": the mesh spends `4 × ticks` calls vs the baseline's 1, so any quality win is confounded with compute. | The **PRIMARY mechanism control is `sphere_x4_no_lateral`** (4 diverse sphere calls, no voting). `mono_x4` (4 identical calls) is a secondary compute-only control that is **degenerate at `temperature: 0`** (4 identical outputs) and does not separate prompt-diversity from the mechanism. Report quality normalized by total inference tokens AND by call count; only compare at **equal call count** (stratify `mesh_adaptive` by `ticks_run`). Never select "best-of-4 by judge" as primary (oracle selection). |
| **C2** | Prompt-verbosity: the mesh carries ~4 specialized system prompts; a short generic baseline rewards verbosity, not mechanism. | The monolithic baseline receives **the union of the four PTG sphere prompts plus a minimal combining instruction** (so it produces a parseable multi-perspective object). This is **not token-equal** to the mesh (the monolith also gets the combining instruction); the delta is small, acknowledged, and the actual `prompt_tokens` of both are recorded. |
| **C3** | Prefix-cache asymmetry + token double-counting: the mesh reuses 4 stable prefixes across ticks; naive summing double-counts cached tokens; naive subtraction makes the mesh look artificially cheap. | **Report gross `prompt_tokens`, `cached_tokens`, and `completion_tokens` separately per condition.** Two cache-adjusted views are emitted, both named explicitly: **`prompt_tokens_cache_adjusted = prompt − cached`** (counts each cached prefix once; completion excluded) and **`total_tokens_cache_adjusted = total − cached = (prompt−cached)+completion`** (true compute-equivalent cost, since completion is never cached). Also report the **cache-hit rate**. **v0 measures WARM only** (cold is not implemented); the first run per side is discarded as warmup. |
| **C4** | Self-selection / survivorship: the mesh rejects sub-threshold columns; confidence is self-reported; comparing mesh-accepted vs baseline-raw is best-of-filtered vs raw. | (1) **Primary mesh artifact = ALL columns (unfiltered)**; accepted-only is a **secondary ablation** that quantifies the filter's contribution, never the headline. (2) Because confidence is self-reported, **all mesh internal metrics (`mean_confidence`, `stabilized`, accept-rate) are treated as non-evidence for quality**; quality is scored externally only. |

## Conditions (generators)

All three generators run on the same model/server with identical generation
settings (see "Fixed settings").

1. **`mesh_adaptive`** — the current PTG mesh: 4 columns (Physics/Math/Coding/
   Psychology), 3-phase epoch, lateral-context injection, confidence-vector
   convergence (adaptive ticks), accepted/rejected split. Evaluated on **all
   columns** (primary) and accepted-only (ablation).

2. **`sphere_x4_no_lateral`** — **the PRIMARY mechanism control.** The default
   mesh run with `max_ticks = 1`: 4 sphere-specialized column calls with empty
   lateral context (no prior ticks) and no voting/convergence, reusing the same
   engine path and per-sphere validation as the mesh. This isolates the cortical
   mechanism: `mesh_adaptive` = diverse columns **+** voting; this condition =
   diverse columns **−** voting. The difference is the mechanism's contribution.

3. **`mono_all_prompts`** — a single chat-completion call whose prompt is **all
   four PTG sphere prompts concatenated** plus the task and a minimal combining
   instruction. Tests the mechanism at equal instruction *union* but 1 call.
   Completion cap = `4 × per-column cap`.

4. **`mono_x4`** — **4 identical** `mono_all_prompts` calls (same prompt, same
   task). At `temperature: 0` these produce 4 identical outputs, so this is a
   **degenerate compute-only control** (C1) that does NOT separate
   prompt-diversity from the mechanism. It is relegated to a secondary control.

There is **no common LLM "integration pass"** applied to the generators' outputs
in v0 — it would change the system under test and add its own compute confound.
Instead each generator's raw outputs are placed in a **deterministic canonical
envelope** and judged as-is (see "Quality").

## Fixed settings

Identical across all conditions:

- `temperature: 0.0` for the benchmark (determinism). The engine currently forces
  `0.2`; the benchmark will parameterize temperature. If the server ignores a
  provided `seed`, that is recorded.
- `top_p: 1.0`, `stream: false`, `response_format: {"type":"json_object"}` for
  all generators (identical format constraint both sides).
- **Same message-role convention** both sides. Note: the mesh currently places the
  system prompt inside a single `user` message (no `system` role); the baseline
  will use the **same convention** so the prefix-cache boundary and attention
  behavior match.
- Per-column completion cap for the mesh (`--max-tokens-col`, default 1024);
  the monolithic cap is `--max-tokens-mono` (default `--max-tokens-col × 2`) and
  must fit the server's `-c 4096` context. Any `finish_reason == "length"`
  truncation is **flagged as a scored failure** on both sides.
- **Forced minimum ticks for the mesh** (`--min-ticks`, default 2). Because an
  overconfident model can self-report `mean_confidence ≥ min_mean_confidence` on
  tick 1 — converging before any lateral exchange — the benchmark forces
  `min_ticks ≥ 2` for `mesh_adaptive` so the lateral-voting mechanism is actually
  exercised. `sphere_x4_no_lateral` stays forced at `max_ticks = 1`.

## Pilot findings (mechanism)

The harness-validation pilot revealed a genuine V1 finding, recorded here for
provenance:

- **Confidence-vector convergence is vulnerable to overconfident self-reports.**
  With `gemma-4-e4b`, the model self-reports `confidence ≈ 1.0` on every column,
  so the default mesh (without `min_ticks`) "converges" at tick 1 every time and
  the lateral-voting mechanism is **never exercised**. This is itself a result
  about the V1 mesh: confidence-based convergence on a non-calibrated model is
  non-functional, and `min_ticks` is a necessary workaround. The benchmark's
  "mechanism unmeasured this round" guard surfaces this automatically.
- **Output truncation dominates early failures.** The math column is verbose and
  overflows a 512-token cap → malformed JSON. Raising the per-column cap (and the
  monolithic cap to match, within `-c 4096`) eliminates this.
- A **true equal-call mechanism ablation** (`sphere_x8_no_lateral` / forced-2-tick
  no-lateral) is deferred to the quality/scaled phase (A3): a naive 8-call
  control at `temperature: 0` would just repeat identical calls.

## Metrics

For every measured run we record, per condition:

- **Latency**: (a) **wall-clock end-to-end** (includes queueing at the 1-slot
  server) and (b) **sum of per-call latency** (compute-equivalent). Both reported.
- **Tokens**: gross `prompt_tokens`, `cached_tokens`, `completion_tokens`,
  `total_tokens`; plus two cache-adjusted views — **`prompt_tokens_cache_adjusted`
  = `prompt − cached`** (counts each cached prefix once; completion excluded) and
  **`total_tokens_cache_adjusted = total − cached`** = `(prompt−cached)+completion`
  (true compute-equivalent cost) — and the cache-hit rate.
- **Call count** and, for the mesh, **`ticks_run`** and stabilized flag. Results
  are stratified by `ticks_run` because a 1-tick mesh is not evidence of the
  lateral mechanism.
- **Survivorship fields** (mesh only): each column's `accepted: bool` tag and
  the per-run `integration_threshold` (so the accepted-only ablation is
  self-contained; confound C4).
- **Quality** (external; see below).

Because the server has 1 slot, `sphere_x4_no_lateral`, `mono_x4`, and a 1-tick
`mesh_adaptive` all have the **same call count** (4) — making the quality-per-call
and quality-per-token comparisons the scientifically central ones.

## Quality (LLM-as-judge, provisional)

No embeddings, so quality uses a same-server **pairwise blind judge**:

- Judge sees two answers in a **canonical envelope** (method labels stripped),
  A/B order **randomized and swapped** across the two judgments per pair.
- Rubric (correctness, empirical grounding, quantitative reasoning, software
  diagnosis, behavioral insight, integration, calibration, concision); the judge
  prompt **forbids rewarding verbosity or format labels**.
- **Disagreement → third adjudication.**
- **Limitation (documented): same-model judge self-correlation.** Mitigations:
  prefer **programmatic factual spot-checks** (e.g. "did the answer state the
  12 kg mass?") wherever a task permits; **human spot-check** on a stratified
  sample; treat judge scores as **provisional** until a distinct stronger judge
  is available.

Comparisons are deliberately separated into three questions:

1. **Diversity/compute (no mechanism):** does having 4 different sphere prompts
   beat a single prompt? `sphere_x4_no_lateral` (4 diverse calls, no voting) vs
   `mono_x4` (4 identical calls) vs `mono_all_prompts` (1 call). These ARE
   equal-call-count or single-call and fair as stated.
2. **Mechanism:** does lateral voting on top of diversity help? The ONLY honest
   comparison is `mesh_adaptive` at `ticks_run ≥ 2` (lateral context is non-empty)
   vs `sphere_x4_no_lateral`. Note: a `mesh_adaptive` run that stops at tick 1 is
   **identical** to `sphere_x4_no_lateral` (same calls, no lateral context) and
   carries NO mechanism signal — so this comparison is only valid for runs that
   reach tick ≥ 2, and it is **inherently compute-asymmetric** (8 calls vs 4):
   report quality normalized by call count, never as a raw win. A future
   `sphere_x8_no_lateral` control (each sphere called twice independently, no
   voting) would make the mechanism comparison equal-call-count; it is on the
   roadmap for the scaled run.
3. All quality comparisons are **blind-judge**, normalized by total inference
   tokens and call count.

## Prompt set (pilot)

Five diverse, deliberately multi-domain prompts (each meaningfully exercises
Physics+Math+Coding+Psychology), used verbatim. They are hardcoded in
`crates/ptg-cli/src/bin/ptg-bench.rs::default_prompts`. Per-domain
stratification is recorded so no single domain dominates (confound L3).

## Run protocol

- **v0 measures WARM only.** Cold-cache measurement (server restart before
  each trial) is **not implemented** in v0; if it matters, note cold as not
  measured. The warmup runs warm the static sphere-prompt prefixes.
- **Pilot first**: 5 prompts × 3 repeats × 4 conditions, **warm**. This run
  **validates the harness and gives a directional read only** — explicitly **no
  headline quality conclusion**.
- Warmup: one discarded run per condition (cold cache / order effects, confound
  L2). Interleave condition order with a fixed seed so thermal/drift is shared.
- A **cold smoke pass** (server restart before one trial) is recorded if practical;
  otherwise cold is noted as not measured this round.
- Deterministic run nonce per trial (after the static prompt prefix, before the
  task) so repeated prompts don't get a free full-prefix cache hit while the
  sphere prompts still cache.
- **Scale gate**: scale to ~50+ prompts × ≥3 repeats **only if** the pilot
  harness is sound and the compute-matched control survives review.

## Statistical reporting

At every stage report `n`, mean/median/min/max for latency and tokens,
cache-hit rate, call count, `ticks_run` distribution (stratify results by
`ticks_run` — a 1-tick run is 4 parallel independent calls with no lateral
exchange), and judge win/tie/loss. Report **paired deltas** and, at scale,
a paired test (Wilcoxon signed-rank) with a bootstrap CI. Pre-register the
primary metric before the scaled run.

## Instrumentation (no trait change)

To preserve the existing `ColumnEngine` trait, instrumentation lives on the
engine + a metrics sink:

- Parse `usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`,
  `prompt_tokens_details.cached_tokens`) and `finish_reason` from each response
  in `crates/ptg-vllm/src/lib.rs`.
- Add an optional `EngineCallMetrics` sink/callback on `InferenceEngine` /
  `EngineBuilder`; `CorticalMesh::run_epoch` is otherwise unchanged.
- The `ptg-bench` binary collects all engine call records between method
  start/end and emits them in the JSONL record.

## Artifacts

- Raw: `bench-runs/<timestamp>/results.jsonl` (gitignored) — one record per
  answer run + one per judge run, with all raw fields (usage, per-call usage,
  latency both ways, `ticks_run`, `finish_reason`, raw output, canonical envelope).
- Summary: `bench-runs/<timestamp>/summary.md` — the human table.
- `bench-runs/` is gitignored.

## Implementation location

The first harness lives in **`crates/ptg-cli/src/bin/ptg-bench.rs`** (not a new
crate). It reuses `default_mesh`, `InferenceEngine`, and the new metrics sink.

## Review gate (must pass before any numbers are produced or cited)

1. Fresh-context `reviewer`: instrumentation correctness (are `usage`/
   `cached_tokens`/`finish_reason` captured accurately? is the sink wired right?).
2. Fresh-context `plan-reviewer`: methodology/fairness audit **against this
   document** (are C1–C4 actually neutralized in the code? is `mono_x4`
   non-oracle? is the mesh evaluated unfiltered?).
3. Fresh-context `red-team`: result-interpretation attack ("can these numbers be
   read as a mechanism win when they are really a compute/cache/filter artifact?").

Only after these three pass do we run the pilot and record numbers.
