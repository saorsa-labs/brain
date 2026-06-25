# PTG Pilot Findings — mesh vs. monolithic + homogenization comparison

> ⚠️ **PILOT — directional only.** n=5 prompts × 3 repeats. No headline claim.
> These numbers exist to validate the harness and surface directional signals,
> not to prove the mechanism helps. See [`BENCHMARKING.md`](./BENCHMARKING.md).

## Run details

| | Value |
|---|---|
| Server | `gemma-4-e2b-qat` (QAT, 2.7 GB RAM) on `127.0.0.1:18136` |
| Temperature | 0.0 (deterministic) |
| Seed | 20260623 |
| Ticks | min=2, max=2 (forces lateral exchange) |
| Max tokens/col | 1024 |
| Prompts | 5 (P1–P5, multi-domain) |
| Repeats | 3 |
| Date | 2026-06-25 |

Raw data: `bench-runs/1782415039909/` (all-routing pilot), `bench-runs/1782415217945/` (diversity comparison). Gitignored (local-only artifacts per methodology).

---

## 1. Latency & token economy (the "does it cost more?" question)

Using **sum_call_latency** (not wall — single-slot server serializes fan-out) and **cache-adjusted total tokens** (true compute-equivalent cost):

| condition | calls | sum_call_lat (med) | cache-adj tok (med) | cache-hit% (med) | parse OK |
|---|---|---|---|---|---|
| `mesh_adaptive` | 8 | 33,051 ms | 4,143 | 43.5% | 14/15 |
| `sphere_x4_no_lateral` | 4 | 16,439 ms | 1,379 | 98.3% | 15/15 |
| `mono_all_prompts` | 1 | 2,861 ms | 915 | 39.0% | 15/15 |
| `mono_x4` | 4 | 11,319 ms | 1,848 | 99.9% | 15/15 |

### What the numbers say

- **The mesh costs 3× its no-lateral control** in compute-equivalent tokens (4,143 vs 1,379). The extra cost is tick-2's lateral context injection — each column's neighbor context is unique, so it can't reuse the prefix cache (43.5% cache hit vs 98.3% for the static-prompt controls).
- **The mesh costs 4.5× the single monolithic call** (4,143 vs 915 tokens). Whether that buys better quality is **unanswered** — it needs the judge.
- **`mono_x4` has 99.9% cache hit** (4 identical calls at temp 0 → byte-identical outputs). This confirms the methodology's C1 confound warning: `mono_x4` is a degenerate compute control, not a diversity control.
- **1/15 mesh runs failed** (6.7%): a JSON parse error on one column aborted the epoch. This is the fail-fast limitation; it's why the pilot exists.

### The cache-hit story

The most architecturally interesting number: **mesh_adaptive has only 43.5% cache hit** while every other condition is 98%+. The lateral context appended on tick 2 is unique per column per tick, so it breaks prefix-cache reuse. This is the fundamental tension: lateral exchange (the core mechanism) is also the thing that defeats the architecture's primary performance optimization (§7.1 prefix caching). The TurboQuant KV-cache path (v0.1.0 scaling tier) is the long-term answer, but this quantifies why it matters.

---

## 2. Homogenization comparison: all-routing vs diversity-routing

Same 5 prompts × 3 repeats, `mesh_adaptive` only, `--routing-policy diversity --routing-k 2` vs `all`.

### Core metric: pairwise token-Jaccard of final-tick predictions

(0 = completely diverse predictions, 1 = identical text. Lower = less homogenized.)

| | all | diversity | delta |
|---|---|---|---|
| Mean pairwise similarity (med) | 0.190 | 0.168 | **−0.022** |
| Min pairwise similarity (med) | 0.120 | 0.127 | +0.007 |
| Tick1→Tick2 drift (med) | 0.601 | 0.588 | −0.013 |
| Confidence spread (med) | 0.130 | 0.070 | **−0.060** |
| Mean confidence (med) | 0.885 | 0.910 | +0.025 |

### Per-prompt pairwise similarity

| prompt | all | diversity | delta |
|---|---|---|---|
| P1 | 0.341 | 0.236 | **−0.105** |
| P2 | 0.164 | 0.161 | −0.003 |
| P3 | 0.231 | 0.174 | −0.057 |
| P4 | 0.167 | 0.165 | −0.001 |
| P5 | 0.162 | 0.168 | +0.006 |

### What the numbers say

**The v0.2.0 single-run homogenization lead does NOT replicate strongly at pilot scale.**

- Diversity routing produces a **directional** reduction in pairwise similarity (−0.022 median), but the effect is **small and inconsistent**: concentrated on P1 (−0.105) and P3 (−0.057), absent on P2/P4/P5.
- The **min-pairwise** metric (the most divergent column pair) shows **no improvement** — diversity routing doesn't help the most-niche frame survive.
- The **most robust effect** is on confidence, not prediction text: diversity routing **narrows the confidence spread** (0.130 → 0.070) and **raises mean confidence** (0.885 → 0.910). Interpretation: hearing fewer/more diverse neighbors reduces conflicting signals, making columns more uniformly confident — the opposite of "preserving uncertainty."
- The **tick1→tick2 drift** is essentially identical (0.601 vs 0.588) — both routing policies cause the same amount of prediction change after lateral exchange.

### Honest assessment

The single-run demonstration from v0.2.0 (psych column retaining its frame under diversity) was **real but not representative**. At pilot scale:
- The homogenization effect is **weaker than the single run suggested** — baseline pairwise similarity is only 0.190 under `all` routing, meaning columns are already quite diverse.
- Diversity routing's effect is **prompt-dependent** and statistically fragile at n=15.
- The **confidence-narrowing** effect is the most consistent signal, but it may be a side effect of reduced context, not evidence of frame preservation.

**No claim should be made about diversity routing reducing homogenization until**:
1. A larger prompt set (≥50) with paired statistics
2. A quality judge scoring whether the preserved frames are actually useful
3. The semantic-similarity backend unblocks (token-Jaccard is a crude proxy)

---

## 3. What the pilot validated (harness integrity)

- ✅ **The 4-condition design works**: all conditions ran, captured metrics, and produced parseable JSON.
- ✅ **The nonce mechanism is load-bearing**: `mesh_adaptive` tick-1 calls ARE byte-identical to `sphere_x4_no_lateral` calls (same system prompt, nonce, task, empty lateral) — confirmed by the near-identical cache-hit patterns.
- ✅ **The `CC_PSYCH_01` determinism gate works**: as a graph sink (no incoming edges), its tick-1 and tick-2 outputs are byte-identical at temp 0. No lateral context = no change.
- ✅ **The routing observability works**: `tick_outputs.routes` correctly captures per-listener source selection and weights.
- ⚠️ **6.7% mesh failure rate** (1/15): fail-fast on malformed JSON. Raising `--max-tokens-col` would help but doesn't eliminate it.
- ⚠️ **Cache-hit collapse under lateral exchange** (43.5%): the architecture's core mechanism fights its primary performance optimization. This is the real bottleneck for scaling.

---

## 4. Next steps (evidence-based)

| Priority | What | Why |
|---|---|---|
| **High** | Run `ptg-judge` on the pilot outputs | The pilot has NO quality signal. Latency/token numbers are meaningless without knowing if the mesh produces BETTER answers. This is the missing half. |
| **High** | Scale to ≥50 prompts with paired statistics | The homogenization signal is too weak at n=5 to call. Need power. |
| **Medium** | Investigate the cache-hit collapse | 43.5% vs 98% is the real scaling bottleneck. The lateral context is unique per column — can we canonicalize/summarize it to preserve prefix-cache hits? |
| **Medium** | Try `--min-prediction-similarity` in the bench | Add it as a convergence criterion and measure if it changes tick counts / quality. |
| **Low** | Unblock embeddings (§9.3) | Token-Jaccard is too crude for the homogenization measurement to be convincing. |
