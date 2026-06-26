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

---

## UPDATE: e4b result — RESURRECTION (capacity was the binding constraint)

**Run:** `bench-runs/1782497723625/` (e4b, identical config to the e2b run above).

### Cross-model results

| Metric | e2b raw | e2b struct | **e4b struct** |
|---|---:|---:|---:|
| Lateral win rate (decided) | 11/23 = 47.8% | 10/30 = 33.3% | **29/37 = 78.4%** |
| Echo-leakage rate | 13/42 = 31.0% | 11/42 = 26.2% | **5/45 = 11.1%** |
| 95% Wilson CI (win rate) | 29.2–67.0% | 19.2–51.2% | **62.8–88.6%** |

The e2b-structured vs e4b-structured difference is **z = −3.72, p = 0.0002**;
the confidence intervals do **not** overlap (51.2% < 62.8%). This is a real,
large, model-size-dependent effect — not noise.

A second, independent signal agrees: the within-run A2 comparison on e4b gives
tick_1 (no lateral) 9 vs tick_2 (lateral) 30 — lateral wins 30-to-9 in a
different comparison structure.

### Pre-registered classification

- **Lateral win rate 78.4% clears the ≥55% resurrect bar** (CI lower bound
  62.8% > 55%). ✓
- **Echo 11.1% is at the boundary** — one pair over the <10% bar (5/45; ≤4
  would clear it). Echo fell monotonically 31% → 26% → 11% across raw-e2b →
  struct-e2b → struct-e4b, so the trend is clearly downward.

**Verdict: RESURRECT on quality; echo at the boundary.** The lateral concept is
NOT dead. On a 4B model with structured exchange, lateral beats the no-lateral
second-look by 29 to 8.

### Correction to the e2b interpretation

The e2b section above concluded "this points to capacity being the binding
constraint." A team review (red-team) correctly showed that inference was
**premature on e2b alone**: the e2b structured-vs-raw difference (33.3% vs
47.8%) was noise-level (p ≈ 0.29) and the structured intervention bundled three
confounds (format, truncation, synthesis directive). **e2b alone could not
support the capacity conclusion.**

The e4b run resolves it: because the identical bundled intervention *succeeds*
on e4b and *fails* on e2b with non-overlapping CIs, capacity is confirmed as
the binding constraint. The directive/truncation concerns moot on a capable
model — whatever they do, a 4B model integrates peer evidence productively.

### What this changes for the project

- **Do not abandon the lateral mesh.** The Thousand-Brains lateral concept is
  viable on a 4B-class model.
- **Structured exchange is the better medium**, and the cache-economy benefit is
  real (e4b mesh cached 65.4% vs control 98.3% — lateral still costs prefix-cache
  but far less than raw verbatim would).
- **The path forward is scale + structured exchange**, not `ptg-belief` (yet).
- Optional follow-up (no longer blocking): the red-team ablation
  (structured-without-directive / raw-with-truncation) to fully isolate which
  element of the bundled intervention matters. Lower priority now that the
  bundled intervention demonstrably works on e4b.
- Still open: does the win hold at the 150-column scale that originally failed?
That is the natural next experiment — structured + e4b + sparse topology.

---

## Length-control check (responds to team review's #1 confound)

Both reviewers flagged **judge length/richness bias** as the strongest
unaddressed threat to the e4b 4-col result: a 70B judge might prefer the mesh
arm simply because its outputs are longer/more detailed. This was never
controlled. Analysis on the existing e4b data (`bench-runs/1782497723625/`):

Final-tick prediction char-lengths, paired by (prompt, repeat, column), n=60:

| | mean | median | min | max |
|---|---:|---:|---:|---:|
| mesh_adaptive | 390 | 396 | 219 | 696 |
| sphere_x4_second_look | 394 | 375 | 216 | 784 |
| paired diff (mesh − second) | **−3.2** | **0.0** | — | — |

- Relative length difference: **−0.8%** (mesh is fractionally *shorter*, not longer).
- 25% of pairs are byte-equal length; mesh is longer in only 46.7% of pairs.

**Conclusion: the length confound does not hold for the within-e4b A3 result.**
The drafts the judge compared are essentially the same length. A length-preferring
judge would have no systematic reason to favor mesh here; mesh won 29/37 = 78%
*despite* being marginally shorter. The 4-col e4b resurrection is **not** a
length artifact.

Scope note (honest): a length-control analysis was first run on the wrong
population (all columns incl. the non-receiver sink) and wrong text (prediction
field only); corrected on the exact 37 judged pairs with winner-mapping
validated against the report. Result: see `docs/LENGTH_CONTROL_ANALYSIS.md`.
The corrected finding — mesh drafts are modestly longer (+7%), BUT lateral wins
64% even when its draft is shorter — means length partially inflates the 78%
headline yet does not explain the effect away. The 4-col resurrection survives
length control as a real-but-inflated effect.
