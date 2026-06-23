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
| **C1** | "More compute = better": the mesh spends `4 × ticks` calls vs the baseline's 1, so any quality win is confounded with compute. | (1) **Always report quality normalized by total inference tokens AND by call count.** (2) **Add a compute-matched baseline** (`mono_x4`): 4 independent monolithic calls at the same total call budget as a 1-tick mesh. (3) Never select "best-of-4 by judge" as primary — that is oracle selection. If a single answer is needed from `mono_x4`, use a **predefined non-oracle rule** (e.g. highest self-reported confidence) and report it as an ablation; prefer recording all four. |
| **C2** | Prompt-verbosity: the mesh carries ~4 specialized system prompts; a short generic baseline rewards verbosity, not mechanism. | The monolithic baseline receives **all four PTG sphere prompts concatenated** (the union budget), so total instruction context is equal across `mono_*` and mesh. |
| **C3** | Prefix-cache asymmetry + token double-counting: the mesh reuses 4 stable prefixes across ticks; naive summing double-counts cached tokens; naive subtraction makes the mesh look artificially cheap. | **Report gross `prompt_tokens`, `cached_tokens`, and `completion_tokens` separately per condition** — never a single "true cost" number. Also report **cache-adjusted = `prompt_tokens − cached_tokens` per call** with the cache-hit rate. Measure **cold and warm separately**; discard the first run per side as warmup. |
| **C4** | Self-selection / survivorship: the mesh rejects sub-threshold columns; confidence is self-reported; comparing mesh-accepted vs baseline-raw is best-of-filtered vs raw. | (1) **Primary mesh artifact = ALL columns (unfiltered)**; accepted-only is a **secondary ablation** that quantifies the filter's contribution, never the headline. (2) Because confidence is self-reported, **all mesh internal metrics (`mean_confidence`, `stabilized`, accept-rate) are treated as non-evidence for quality**; quality is scored externally only. |

## Conditions (generators)

All three generators run on the same model/server with identical generation
settings (see "Fixed settings").

1. **`mesh_adaptive`** — the current PTG mesh: 4 columns (Physics/Math/Coding/
   Psychology), 3-phase epoch, lateral-context injection, confidence-vector
   convergence (adaptive ticks), accepted/rejected split. Evaluated on **all
   columns** (primary) and accepted-only (ablation).

2. **`mono_all_prompts`** — a single chat-completion call whose prompt is **all
   four PTG sphere prompts concatenated** plus the task. Tests the mechanism at
   equal instruction budget but 1 call. Completion cap = `4 × per-column cap`.

3. **`mono_x4`** — **4 independent** `mono_all_prompts` calls (same prompt, same
   task), matching the call count of a 1-tick mesh. All four raw outputs are
   recorded. Any single-answer reduction uses a **non-oracle rule** (highest
   self-reported confidence), reported as an ablation.

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
- Per-column completion cap for the mesh; `mono_all_prompts` cap = `4 ×` that; any
  `finish_reason == "length"` truncation is **flagged as a scored failure** on both
  sides.

## Metrics

For every measured run we record, per condition:

- **Latency**: (a) **wall-clock end-to-end** (includes queueing at the 1-slot
  server) and (b) **sum of per-call latency** (compute-equivalent). Both reported.
- **Tokens**: gross `prompt_tokens`, `cached_tokens`, `completion_tokens`,
  `total_tokens`; plus cache-adjusted (`prompt − cached`) and cache-hit rate.
- **Call count** and, for the mesh, **`ticks_run`** and stabilized flag.
- **Quality** (external; see below).

Because the server has 1 slot, `mono_x4` and a 1-tick `mesh_adaptive` have the
**same call count** — making the quality-per-call and quality-per-token
comparisons the scientifically central ones.

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

Primary comparison is **blind-judge score at equal call count** (`mesh_adaptive`
1-tick vs `mono_x4`) and **at equal single-call** (`mesh_adaptive` vs
`mono_all_prompts`), each normalized by total inference tokens.

## Prompt set (pilot)

Five diverse, deliberately multi-domain prompts (each meaningfully exercises
Physics+Math+Coding+Psychology), used verbatim. Captured in
`crates/ptg-cli/data/bench-prompts.json`. Per-domain stratification is recorded
so no single domain dominates (confound L3).

## Run protocol

- **Pilot first**: 5 prompts × 3 repeats × 3 conditions, **warm**. This run
  **validates the harness and gives a directional read only** — explicitly **no
  headline quality conclusion**.
- Warmup: discard the first run per condition (cold cache). Interleave condition
  order with a fixed seed so thermal/drift is shared.
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
