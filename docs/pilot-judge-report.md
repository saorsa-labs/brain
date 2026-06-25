# PTG Benchmark — A2 judge report

> **A2 scope: mechanism ACTIVATION ONLY — not consensus, not improvement, not calibration.**
> This report answers only: *did the lateral context injected on tick 2 make the
> receiving columns MOVE?* It does NOT show the mesh reaches consensus, that lateral
> context IMPROVES outputs, or anything about quality. See `docs/BENCHMARKING.md`.

### How NOT to read this report

- **Edit distance is NOT quality.** A 0.75 perturbation means the output CHANGED a lot,
  not that it got better. A column can change a lot and get worse.
- **LLM winner counts are NOT the headline and are NOT a quality verdict.** They are
  directional noise from a corroborating signal; do not cite them as proof of anything.
- **`determinism_ok` checks only `CC_PSYCH_01` per run** (it has no lateral inputs); it does
  NOT prove cross-run determinism for the receiving columns.
- **`confidence_delta` is self-reported → NON-evidence** for quality or calibration; near-zero
  may reflect overconfidence ceiling effect, not good calibration.
- **Echo exclusion is a crude 40-char heuristic;** survivors may still be de-blinded.

- Input: `bench-runs/1782415039909/results.jsonl`
- Pairs analyzed: 42
- Excluded pairs: 14

## Exclusion breakdown

- tick2_echoed_neighbor: 14

> Echo (`tick2_echoed_neighbor`) exclusion is a conservative 40-char-substring heuristic;
> it catches verbatim neighbor-text leaks only. A nonzero count here signals non-trivial
> leakage risk, and survivors may still be de-blinded by paraphrase.

## Primary: programmatic perturbation delta (non-excluded)

> Zero judge confound. "Did the lateral context injected on tick 2 change the
> column's output at all?" Normalized edit distance on `prediction`; domain-field
> change rate. A nonzero delta = activation; its size is the perturbation magnitude.
> **⚠️ Edit distance = perturbation magnitude, NOT quality.** A column can change a lot
> and get worse. domain-field change counts CHANGED KEYS, not correctness; `—` = no
> comparable fields.

| column | n | prediction edit dist (med) | domain-field change % (med) | mean conf delta (self-report, NON-evidence) |
|---|---|---|---|---|
| CC_PHYSICS_01 | 10 | 0.741 | — | +0.017 |
| CC_MATH_01 | 10 | 0.726 | — | +0.063 |
| CC_CODE_01 | 8 | 0.613 | — | +0.006 |

## Corroborating: LLM judge (if run)

**⚠️ NOT the primary result. Directional noise only — do not cite as a quality verdict.**

- judge model: `llama-3.3-70b-versatile` via `https://api.groq.com/openai/v1`
- normalized winners: { tick_1: 14, tick_2: 13, tie: 1 }

> Reminder: tick_2 winning does NOT prove lateral *improvement* — it may be
> 'a second look' (J1) or length/echo bias (J2/J3). Treat as directional only.

## Honest bounds

- n per column is tiny (28 judged pairs total); any LLM winner split is within swap-disagreement noise — no statistical claim.
- 3 columns is the generalization ceiling (lateral activates *these domains*, not 'columns' in general).
- `confidence_delta` is self-reported → NON-evidence for quality or calibration (near-zero may be overconfidence ceiling).
- `determinism_ok` is per-run, `CC_PSYCH_01`-only; it does NOT establish cross-run determinism for the receiving columns.
- Echo-exclusion survivors may still be de-blinded by paraphrase; the survivor set may be biased toward columns that ignored lateral context.
- A clean 'lateral IMPROVES' claim requires the A3 no-lateral-second-tick control (not yet built).
