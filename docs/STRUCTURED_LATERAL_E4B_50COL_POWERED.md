# Powered 50-col structured-e4b — statistically validated

> **Status: CONFIRMED. The structured-lateral resurrection is real and powered.**
> First statistically interpretable scale result in the project.
> Run: `bench-runs/1782551183695/` (50 cols, structured, e4b, 5 prompts × 3 repeats).

## What changed from the single-prompt 50-col run

The 50-col result (`STRUCTURED_LATERAL_E4B_50COL.md`) was 1 prompt × 1 repeat —
directional. This run is **5 prompts × 3 repeats** (matching the pilot A3 for
comparability), the first powered scale result. 30 runs; 1 transient mesh-arm
failure (96/100 calls, server blip, recovered; the judge drops that one
prompt-repeat). The judge sampled 60 A3 pairs (stride) from the ~700 route-active
receiver pairs.

## Result (pre-registered decision rule)

| Metric | Value | Bar | Verdict |
|---|---:|---|---|
| Lateral win rate | 40/47 = **85.1%** | — | — |
| 95% Wilson CI | **[72.3%, 92.6%]** | — | — |
| vs H0 (p=0.5) | z=4.81, **p=1.5×10⁻⁶** | — | rejects null |
| CI lower bound | 72.3% | > 55% | **CLEARS** |
| Echo-leakage | 4/60 = **6.7%** | < 10% | **CLEARS** |

Both pre-registered bars clear, with margin. The lower CI bound (72.3%) is far
above the 55% resurrect threshold, and the effect rejects the 50%-null at
p ≈ 10⁻⁶.

## Length-control (validation gate passed)

Reconstruction reproduces the report's lateral=40 / second_look=7 exactly before
any length claim is made (the gate that earlier caught a wrong-population
analysis). Judge-visible draft lengths:

| | mean | median |
|---|---:|---:|
| mesh (lateral) | 1765 | 1666 |
| second_look | 1668 | 1546 |
| paired diff | +97 | — (relative +5.8%) |

Stratification (the decisive test):

| stratum | n | lateral wins | rate |
|---|---:|---:|---:|
| mesh longer | 31 | 29 | 94% |
| **mesh shorter** | 16 | 11 | **69%** |

**Lateral wins 69% even when its draft is shorter.** The effect is present in
the length-disadvantaged stratum. As before, length inflates the headline (94%
when longer) but is not the driver.

## Cross-run consistency (the stability the reviewers asked for)

| Run | n decided | lateral win | echo |
|---|---:|---:|---:|
| 4-col e4b (1p×3r) | 37 | 78.4% | 11.1% |
| 50-col e4b (1p×1r) | 32 | 84.4% | 6.0% |
| **50-col e4b (5p×3r)** | 47 | **85.1%** | **6.7%** |

The win rate is stable across prompt sets and scales (78 → 84 → 85%), and echo
stays low (6-11%). This is no longer a single-draw result; it replicates across
prompts.

## What this settles

- **The Thousand-Brains lateral mechanism is genuinely quality-positive** with
  structured exchange on a 4B-class model at moderate scale. Confirmed across
  prompts with proper statistics (p ≈ 10⁻⁶), surviving length control.
- **Structured exchange + e4b is the validated design direction.** The earlier
  "do not scale raw lateral text" conclusion stands for the *raw* medium; the
  *structured* medium reverses it.

## What remains open

- **150-col on e4b** — still infra-blocked (server capacity ceiling). The next
  scale milestone requires server tuning (KV-cache / slot), not more runs.
- **A4 self-revision control** — still needed to separate lateral exchange from
  genuine "reconsider your answer" revision.
- The 1 transient mesh failure (P5 repeat 1) is a reliability note on the e4b
  server under sustained load, consistent with the 150-col ceiling.
