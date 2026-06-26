# Structured Lateral Exchange — Findings (e2b)

> **Classification: FAIL (e2b)** — pre-registered, decided by the fixed thresholds
> in `docs/STRUCTURED_LATERAL_EXPERIMENT.md`. This does NOT kill the concept;
> the next pre-registered gate is structured-on-e4b (capacity hypothesis).

## Run

- Run dir: `bench-runs/1782496025097/`
- Config: e2b, 4-col default, `all` routing, 5 prompts × 3 repeats, 2/2 ticks,
  temp 0, seed 20260623, 1024 token cap, **`--lateral-mode structured`**.
- 30/30 runs `parse_ok` (1 mesh truncation; excluded by judge).

## Pre-registered metrics (vs raw pilot baseline)

| Metric | Raw (pilot) | Structured (this run) | Direction |
|---|---:|---:|---|
| Echo-leakage rate | 13/42 = 31.0% | 11/42 = **26.2%** | ↓ improved (modest) |
| Lateral win rate (decided) | 11/23 = 47.8% | 10/30 = **33.3%** | ↓ **worse** |
| Cache-hit % (mesh, med) | 36.9% | 64.1% | ↑ (shorter lateral text) |

## Decision (pre-registered table)

| Bucket | Condition | Met? |
|---|---|---|
| RESURRECT | echo < 10% **AND** lateral ≥ 55% | No (26.2%, 33.3%) |
| AMBIGUOUS | neither fail nor resurrect | No — fail clause fires |
| **FAIL (e2b)** | echo > 30% **OR** lateral win < 45% | **Yes** (33.3% < 45%) |

**Verdict: structured lateral on e2b fails the quality gate.**

## What this result actually tells us (the important nuance)

This result is more diagnostic than a simple "no." It separates two failure
modes that the raw experiment could not:

1. **The medium matters for LEAKAGE.** Changing from full-verbatim injection to
   bounded claim-excerpts + a synthesis directive reduced echo-leakage from
   31.0% → 26.2%. So red-team's hypothesis (b) — "raw text invites copying" — is
   **partially confirmed**: the format does influence how much a small model
   copies. But the reduction is modest, not the <10% that would indicate the
   leakage problem is solved.

2. **The medium does NOT matter for QUALITY.** Structured exchange did not
   improve quality — it made it **worse** (47.8% → 33.3%). Giving the model
   terse excerpts and an explicit "synthesize, don't quote" directive produced
   *lower*-quality output than raw verbatim injection. This is the opposite of
   what H-med predicted.

This combination points away from "the format is the problem" and toward
**hypothesis (a): the model is too small to integrate peer evidence
productively, regardless of how it's presented.** A 2B model, told to
synthesize from peer claims, does not synthesize better — it produces worse
output, plausibly because the synthesis directive disrupts its output pattern
or because the truncated excerpts discard information it would have used.

## Why structured might be worse, not just neutral

Two candidate explanations (not pre-registered; hypotheses for the e4b test):
- **Truncation loss.** Bounding to ~180 chars discards the neighbor's reasoning
  detail. For a model that relies on surface features, less text = less to work
  with, not more clarity.
- **Directive confusion.** A 2B model may not robustly follow "synthesize an
  independent answer" — the instruction may steer it off its best default
  pattern without improving integration.

Both predict that a larger model (e4b) would respond differently to the same
structured format — which is exactly what the next gate tests.

## Side note: cache economy

Structured lateral improved cache-hit from 36.9% → 64.1% (shorter injected text
keeps the prompt prefix stable). This is a real operational benefit of the
structured format but is **irrelevant to the quality question** and does not
rescue the mechanism.

## Next step (pre-registered)

> An e2b failure triggers structured-on-e4b, NOT abandon.

The next gate: run the identical structured A3 on `gemma-4-e4b` (~9 GB, 4B
params). Decision rule (pre-registered carryover):
- If e4b structured also fails (lateral win < 45% or echo > 30%), the lateral-
  text concept has failed on two model sizes → pivot to `ptg-belief`.
- If e4b structured resurrects (echo < 10% AND lateral ≥ 55%), capacity was the
  binding constraint → the concept lives, and structured exchange + a larger
  model is the path.

This run has NOT been executed yet; it is the recommended next step pending
the e4b server being live.
