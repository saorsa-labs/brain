# 150-column structured-e4b — POWERED result (5p × 3r)

> **Status: POSITIVE, powered, statistically decisive.** This is the powered
> confirmation of the 1p1r frontier run. It **corrects the 1p1r's optimistic
> 93%** down to the true rate of ~82%, and confirms the result survives length
> control. Run: `bench-runs/1782601932225/` (150 cols, structured, e4b,
> `--column-concurrency 4`, 5 prompts × 3 repeats).

## Headline (powered)

| Metric | Value |
|---|---:|
| Decided pairs (sampled, excl. 1 tie) | **392** |
| Lateral win | **323 / 392 = 82.4%** |
| 95% Wilson CI | **[78.3%, 85.8%]** |
| vs H₀ (p=0.5) | z = 12.83, **p ≈ 0** |
| Echo leakage | 42/500 = **8.4%** |
| Win when draft SHORTER | **154/205 = 75%** |

Both pre-registered bars clear decisively: CI lower 78% ≫ 55% bar; echo 8.4% ≪ 10%
bar. Length confounding is essentially absent (mesh drafts only +2.4% longer;
lateral wins 75% even when shorter).

## The correction: 1p1r overstated, powered does not

The 1p1r run (`STRUCTURED_LATERAL_E4B_150COL.md`, 43 decided pairs) reported
93.0%. **That was small-sample optimism.** At 392 decided pairs the true rate is
82.4% — still a decisive positive, but the 1p1r's 93% should not be cited as the
150-col figure. This is exactly why powered runs exist.

The corrected cross-scale trend is **roughly flat ~80–85%, not increasing**:

| Run | pairs | lateral win | echo |
|---|---:|---:|---:|
| 4-col e4b | 37 | 78.4% | 11.1% |
| 50-col e4b (powered) | 47 | 85.1% | 6.7% |
| **150-col e4b (powered)** | **392** | **82.4%** | **8.4%** |
| ~~150-col e4b (1p1r)~~ | ~~43~~ | ~~93.0%~~ | ~~3.3%~~ *(small-sample noise)* |

The structured mechanism holds at every scale tested (4 → 150 cols) at roughly
constant ~80–85% quality advantage over the equal-call no-lateral control, with
echo staying under 10%. It does **not** keep improving with scale past ~50 cols.

## Survivorship and exclusions (must-read caveats)

This is a sampled result with real exclusions. The honest denominator:

- **Run survivorship:** 12/15 mesh runs completed. 3 mesh runs (P1-r0, P3-r0,
  P3-r1) hit unrecoverable **HTTP 500 Internal Server Error** in MATH columns
  (CC_MATH_06/16/12) that exhausted the 3-attempt retry budget and fail-fast
  aborted. All 15 control runs completed. The 3 failed mesh runs are excluded
  from A3 judging — **if they would have been low-quality, their exclusion
  inflates the win rate.** This is the most important caveat.
- **Pair sampling:** 500 / 1800 A3 pairs sampled (28%), seed=0.
- **Pair exclusions (of 500):** 107 excluded → 393 decided:
  - 65 control second-look unstable (temperature-0 nondeterminism in the
    control arm — the no-lateral replay should be byte-identical across ticks
    but wasn't for these 65; excluded, not miscounted)
  - 42 mesh_echoed_neighbor (echo-screen leakage)
  - 36 determinism_failed (PSYCH-column determinism gate)

The MATH-column HTTP-500 clustering is an infrastructure finding worth its own
investigation (possibly long math generations hitting a server limit), not a
quality result.

## What this settles

1. **The structured lateral mechanism is quality-positive at 150 columns** (the
   scale where raw exchange catastrophically failed at 14%), decisively so
   (p ≈ 0, CI clears all bars, length-controlled). The "do not scale dense
   lateral-text" conclusion holds for the **raw** medium only.
2. **The effect saturates, not escalates**, at ~80–85% past ~50 columns. More
   columns do not keep improving quality; they add compute and survivorship risk.
3. **Powered measurement corrected a small-sample overstatement** (93% → 82%).
   Every directional claim in this project has been tightened by powered runs and
   review; this is consistent with that discipline.

## What it does NOT settle

- Whether the ~20% mesh-run failure rate (HTTP 500 in MATH cols) biases the win
  rate upward (survivorship). A run with 0 mesh failures would settle this; the
  current judge works on completed runs only.
- Cross-model generalization (only gemma-4-e4b tested at scale).
- The A4 explicit-self-revision control (lateral exchange vs "reconsider your
  answer" at equal call budget) — still deferred.
