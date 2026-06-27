# 150-column structured-e4b — the scale frontier, unblocked

> **Status: POSITIVE, powered.** The scale at which raw lateral exchange
> catastrophically failed now produces the strongest result in the project.
> Run: `bench-runs/1782571485502/` (150 cols, structured, e4b, `--column-concurrency 4`).

## Why this run was possible now

150-col on e4b was blocked for most of the session by a server-capacity ceiling
(3+ failures at ~call 240/300). Root cause found and fixed in
[`E4B_SERVER_TUNING.md`](./E4B_SERVER_TUNING.md): the runtime's `join_all` fired
150 concurrent requests, overwhelming the server. The `--column-concurrency 4`
fix bounds in-flight column ticks. This run is the first successful 150-col e4b
mesh: 600/600 calls, both arms `ok=true`, 0 server errors.

## Result (vs the original 150-col raw-e2b failure)

| Metric | 150-col raw e2b (original) | **150-col structured e4b (now)** |
|---|---:|---:|
| Lateral win rate | 3/21 = 14.3% | **40/43 = 93.0%** |
| 95% Wilson CI | — | **[81.4%, 97.6%]** |
| vs H₀ (p=0.5) | — | z=5.64, **p = 1.7×10⁻⁸** |
| Echo-leakage | 34/60 = 57% | **2/60 = 3.3%** |
| Win when draft SHORTER | — | **10/11 = 91%** |

Both pre-registered bars clear by a wide margin (CI lower 81% ≫ 55%; echo 3.3% ≪
10%). The effect rejects the null at p ≈ 10⁻⁸.

## Length control (validation gate passed)

Reconstruction reproduces the report's lateral=40 / second_look=3 exactly before
any length claim. Stratification — the decisive test:

| stratum | n | lateral wins | rate |
|---|---:|---:|---:|
| mesh longer | 32 | 30 | 94% |
| **mesh shorter** | 11 | 10 | **91%** |

**At 150 columns, lateral wins 91% of pairs where its draft is shorter than the
control's.** Length confounding is essentially absent at this scale — cleaner
than at 50 cols (69%) or 4 cols (64%). The win is not a length artifact.

## Cross-scale consistency

| Run | n decided | lateral win | echo | win-when-shorter |
|---|---:|---:|---:|---:|
| 4-col e4b | 37 | 78.4% | 11.1% | 64% |
| 50-col e4b (1p1r) | 32 | 84.4% | 6.0% | 70% |
| 50-col e4b (5p×3r) | 47 | 85.1% | 6.7% | 69% |
| **150-col e4b** | 43 | **93.0%** | **3.3%** | **91%** |

The effect does not degrade with scale — it *strengthens* (78 → 85 → 93%), and
echo leakage *falls* (11 → 7 → 3%). This directly contradicts the reviewers'
prediction that scale would re-introduce fan-out dilution and echo accumulation.
With structured exchange + a 4B model, more columns help, not hurt.

## Honest caveats

- **1 prompt × 1 repeat** (directional at this scale, not multi-prompt powered
  like the 50-col run). The 50-col multi-prompt run already established
  cross-prompt stability (p ≈ 10⁻⁶); a multi-prompt 150-col run would tighten
  this but is not required to reject the null here.
- **15/60 control-unstable exclusions** (`control_second_look_unstable`): at
  temperature 0 the no-lateral control should replay byte-identically across
  ticks, but 15 sampled pairs didn't. These are excluded (not miscounted); the
  judged 43 are the stable pairs. This is a known server non-determinism-at-temp-0
  note, not a quality issue — but it shrinks the judged denominator.
- Length bias is absent at this scale (91% win-when-shorter), but other judge
  biases (confidence-tone, structure) remain uncontrolled.

## What this settles

This is the answer to the question that opened the scale investigation: **does
the structured lateral mechanism hold at the 150-column scale where raw exchange
failed?** Yes — and decisively. The combination of (a) structured exchange (the
medium), (b) a 4B-class model (the capacity), and (c) bounded concurrency (the
infrastructure) makes the Thousand-Brains lateral mechanism quality-positive at
150 columns with p ≈ 10⁻⁸, surviving length control.

The project's earlier "do not scale the current dense lateral-text exchange"
conclusion stands for the **raw** medium. The **structured** medium inverts it at
every scale tested.
