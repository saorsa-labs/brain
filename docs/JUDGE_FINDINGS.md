# PTG Quality Pass — Judge Findings (A2)

> ⚠️ **PILOT — directional only.** n=28 judged pairs. No headline claim.
> This answers the quality half of the pilot question. See
> [`PILOT_FINDINGS.md`](./PILOT_FINDINGS.md) for the latency/token half and
> [`pilot-judge-report.md`](./pilot-judge-report.md) for the raw judge output.

## The question, and its honest scope

The pilot measured cost (3–4.5× more compute) and left quality unanswered. The
judge (`ptg-judge`) closes that loop — but under a deliberately **narrow** scope
(§BENCHMARKING "A2 scope: mechanism ACTIVATION, not consensus benefit"):

> *Did the lateral context injected on tick 2 ACTIVATE the receiving columns,
> and (corroborating) was the perturbed output perceived as BETTER by an
> external judge?*

This is **not** the thesis-level "the mesh reaches better consensus than
monolithic." That needs the A3 no-lateral-second-tick control (not yet built).
A2 is the necessary precondition: if the mechanism doesn't activate, or
perturbs toward *worse*, the mesh provably isn't helping.

## Run details

| | Value |
|---|---|
| Input | `bench-runs/1782415039909/results.jsonl` (the `all`-routing pilot) |
| Corroborating judge | `llama-3.3-70b-versatile` via Groq (distinct family from the generator) |
| Judge protocol | Blind pairwise, A/B swapped, third adjudication on disagreement |
| Pairs analyzed | 42 (14 echo-excluded) → 28 judged |
| Generator | `gemma-4-e2b-qat` (QAT), temperature 0 |
| Date | 2026-06-25 |

## Result 1 — The mechanism ACTIVATES (strongly)

**Programmatic perturbation delta** (zero judge confound — pure edit distance
between tick-1 and tick-2 `prediction` strings):

| column | n | prediction edit distance (median) | confidence delta |
|---|---|---|---|
| CC_PHYSICS_01 | 10 | **0.741** | +0.017 |
| CC_MATH_01 | 10 | **0.726** | +0.063 |
| CC_CODE_01 | 8 | **0.613** | +0.006 |

The lateral context genuinely changes the outputs — tick-2 predictions are
**60–74% different** from tick-1 (by normalized Levenshtein). The mechanism is
not dormant; it does real work.

## Result 2 — The perturbation does NOT improve perceived quality

**Corroborating LLM judge winners** (blind A/B, normalized):

| winner | count |
|---|---|
| tick_1 (no lateral) | **14** |
| tick_2 (with lateral) | **13** |
| tie | 1 |

**This is a coin flip.** The lateral-perturbed outputs are *not* perceived as
better than the unperturbed outputs by an external judge. At n=28 with swap
noise, this is indistinguishable from "the lateral mechanism has no effect on
quality."

## Result 3 — Leakage is a real confound

**14 of 56 pairs (25%) were echo-excluded**: tick-2 predictions verbatim-copied
≥40 characters of an injected neighbor's text. This is the model literally
echoing the lateral context rather than reasoning over it — a de-blinding risk
the judge conservatively excludes. Survivors may still be de-blinded by
paraphrase.

## What this means — combined with the pilot

| Dimension | Result | Signal |
|---|---|---|
| **Does the mechanism activate?** | Yes — 0.61–0.74 edit distance | ✅ Strong |
| **Does it improve perceived quality?** | No — 14 vs 13 (coin flip) | ❌ None detected |
| **What does it cost?** | 3–4.5× compute, 43.5% cache-hit collapse | ❌ Expensive |
| **Is it clean?** | 25% of pairs echo neighbor text verbatim | ⚠️ Leakage |

**Directional conclusion (pilot-scale, not headline):** At 4 columns, 2 ticks,
`gemma-4-e2b-qat`, the lateral mechanism activates but does *not* demonstrably
improve answer quality — and it costs 3–4.5× more. This is the kind of result
that should pause "scale it up" and prompt "understand *why* it's a coin flip."

## Candidate explanations (NOT proven — hypotheses to test)

1. **The model is too small.** `gemma-4-e2b-qat` (2.7 GB) may lack the capacity
   to *integrate* lateral context productively — it changes its output (high
   edit distance) but not in a direction an external judge scores as better.
   A larger generator (e4b, 9 GB) might integrate better. **Testable directly.**
2. **Echo leakage is diluting the signal.** 25% of pairs were excluded for
   verbatim echo; the coin flip is among the *survivors*. If leakage is the
   model's dominant response to lateral context, the "perturbation" isn't
   reasoning, it's copying. A stronger echo screen + paraphrase detection
   would clarify this.
3. **2 ticks is too few.** One round of lateral exchange may not be enough for
   productive integration. The mesh may need 3–5 ticks to show a quality delta.
4. **The default topology is too sparse.** 3 receivers, each hearing 1–2
   neighbors. The Phase 3 topologies (torus, ring) with `--routing-policy
   diversity` may behave differently — but the homogenization comparison
   already showed those effects are weak at this scale.
5. **The rubric is mismatched.** The judge scores factual correctness / reasoning
   quality / grounding / concision. Lateral voting's hypothesized benefit is
   *coverage* (catching what one column misses) — which a single-column A/B may
   not surface.

## The honest fork this creates

This is a genuine decision point, not a formality:

| | Fork | Implication |
|---|------|-------------|
| **A** | **Scale is premature; investigate the coin flip.** | Test the hypotheses above (larger model, more ticks, stronger echo screen) before investing in 50-prompt statistics or Phase 4. |
| **B** | **The signal is real — the mechanism doesn't help at this scale.** | Reconsider the architecture: is 2B-parameter lateral voting the wrong substrate? Does TBT need bigger columns, or a different integration step (the deferred integration LLM)? |
| **C** | **A pilot is a pilot — scale it and let statistics decide.** | Accept that n=28 is noise and run 50+ prompts. Risk: spending compute to add decimal places to a coin flip. |

The pilot did its job. It surfaced that the core mechanism, as currently built,
activates but doesn't demonstrably help — and it cost real money to learn that.
That's a successful pilot, not a failed project.

## What I would NOT do

- Cite "the mesh improves quality" — the data says coin flip.
- Cite "diversity routing reduces homogenization" — the comparison was weak at n=5.
- Start Phase 4 scaling — the cache-hit collapse + coin-flip quality means
  we'd be optimizing a mechanism that hasn't shown benefit.
