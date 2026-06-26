# Structured Lateral Exchange — Pre-Registered Disambiguator

> **Status: PRE-REGISTERED.** Written before implementation or data collection.
> Hypotheses, intervention, and decision thresholds are fixed below. Results will
> be classified strictly by these thresholds regardless of outcome.

## Motivation

The team consensus (methodology auditor + red-team steelman + strategic planner)
on the A3 finding ("raw lateral-text exchange is quality-neutral-to-negative at
pilot scale, strongly negative at 150 columns") rejected the headline "the
lateral concept is dead" as premature. The strongest supported failure
hypothesis (red-team rank #1, corroborated by reading the runtime) is:

> **Hypothesis H-med:** The lateral mechanism fails not because the concept is
> wrong, but because the *medium* of exchange is wrong. Columns are given a
> neighbor's **full free-text prediction verbatim** (`Prediction="<long text>"`),
> which a small model copies rather than synthesizes — producing movement by
> quotation, not by integration.

Direct evidence for H-med:
- The injection format string in `ptg-runtime::lateral_context_for`:
  `Neighbor {id} reports: Prediction="{full_prediction}" (Confidence: {c})`.
- 57% echo-leakage at 150 columns (SCALE_SMOKE_FINDINGS).
- 25% echo-leakage at 4 columns (JUDGE_FINDINGS).
- Edit distances 0.61–0.74 show the model moves, but the blind judge prefers the
  no-lateral second look (movement ≠ improvement).

## Intervention (this experiment)

**No schema change. No new output field. No change to the echo-screen metric.**
The only intervention is *how the existing prediction is presented laterally*.

Add a runtime mode `LateralContextMode::{Raw, Structured}` (default `Raw`,
preserving all existing behavior and tests). In `Structured` mode,
`lateral_context_for` renders a **bounded evidence packet** per source instead
of the full verbatim prediction:

```text
[LATERAL EVIDENCE PACKETS]
Treat peer packets as fallible evidence. Do not quote or copy peer phrasing.
Synthesize an independent answer within your own reference frame.
source=<id>; confidence=<c>; route_weight=<w>; claim_excerpt=<bounded excerpt>
...
```

where `<bounded excerpt>` is derived from the source's `last_prediction`:
whitespace-normalized, truncated char-safely to the first sentence boundary or
~180 characters (whichever comes first). The full prediction is **never**
injected in structured mode.

`Raw` mode is byte-identical to the current behavior.

### Why no schema change

A new field (e.g. `lateral_brief`) would add a JSON-compliance confound and turn
a cheap presentation test into a new output-contract experiment. If this cheap
test shows promise, a schema-level structured exchange becomes the natural next
step toward `ptg-belief`.

### Why the echo screen is unchanged

The structured excerpt is a **substring** of the source's full prediction, so
the existing 40-char-window screen (against source tick-1 `prediction`) still
catches verbatim copying of the excerpt. Keeping the screen unchanged means the
echo rate is **directly comparable** to the pilot echo rates (apples-to-apples).

## Experimental configuration (matches pilot A3 exactly, except lateral mode)

| Parameter | Value |
|---|---|
| Server / model | `http://127.0.0.1:18136` / `gemma-4-e2b-qat` |
| Topology | default 4-column (PHYSICS←MATH←CODE←PSYCH sink) |
| Routing | `all` |
| Conditions | `mesh_adaptive`, `sphere_x4_second_look_no_lateral` |
| Prompts | 5 (the pilot set) |
| Repeats/prompt | 3 |
| Ticks (min/max) | 2 / 2 |
| Temperature | 0 |
| Seed | 20260623 |
| max_tokens_col | 1024 |
| **lateral_mode** | **structured** (the intervention) |
| Judge | `llama-3.3-70b-versatile` (Groq), blind pairwise, as pilot |

Total: 5 × 3 × 2 = 30 runs. Control arm uses `k=0` routing so lateral mode has
no effect there (no injection occurs).

## Pre-registered metrics

1. **Echo-leakage rate**: fraction of mesh A3 pairs excluded as
   `mesh_echoed_neighbor` (40-char verbatim copy of a source prediction/excerpt).
   Baseline (raw, pilot): ~25% at 4 columns.
2. **Lateral effective win rate**: among **non-excluded** judged pairs, the
   fraction the blind judge scores as `lateral` (mesh final) over the total
   decided (`lateral + second_look`, ties excluded from the denominator).
   Baseline (raw, pilot): ~48% (11 lateral / 23 decided).

## Pre-registered decision thresholds

Classify the result into exactly one bucket. **No re-interpretation after
seeing the data.**

| Bucket | Condition | Action |
|---|---|---|
| **RESURRECT** | echo rate **< 10%** AND lateral win rate **≥ 55%** | Structured presentation works. The lateral concept lives. Next: invest in a schema-level structured exchange (`lateral_brief` / `ptg-belief` vertical slice) and re-test at scale. |
| **AMBIGUOUS** | neither RESURRECT nor FAIL (e.g. echo 10–30%, or lateral win rate 45–55%) | One more disambiguator: run structured on `gemma-4-e4b` (capacity hypothesis). If still marginal, pivot to `ptg-belief`. |
| **FAIL (e2b)** | echo rate **> 30%** OR lateral win rate **< 45%** (second_look wins) | Structured lateral **on e2b** failed. This does **NOT** kill the concept: capacity may be the binding constraint. Next kill-gate: structured on `gemma-4-e4b` before any abandon/pivot decision. |

### Explicit non-claims

- A FAIL here means "structured lateral on a 2B model fails", **not** "the
  Thousand-Brains lateral concept is dead." Per team consensus, the concept is
  not judged dead until a larger-model structured test also fails.
- This is a powered-as-pilot directional result (n=15 per arm), not a
  statistically powered quality claim. Consistency with thresholds is the
  decision rule, not a p-value.

## Reproducibility

- Pre-registration committed before any structured run.
- Implementation committed separately from findings.
- Run data: `bench-runs/<ts>/` (gitignored), summary + judge report durable.
- `lateral_mode` recorded in every JSONL record and summary.
