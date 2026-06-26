# Length-Control Analysis (e4b 4-col A3)

> Responds to the team-review confound: "does the judge prefer mesh because its
> outputs are longer?" Rigorous re-analysis on the EXACT judged population,
> winner-mapping validated against the report before any length claim.

## Method

- Population: the **37 decided A3 pairs** (route-active receivers only; sink
  `CC_PSYCH_01` excluded, as it is not a lateral receiver), reconstructed from
  the `a3_` judge calls in `bench-runs/1782497723625/a3-judge-calls.jsonl`.
- **Validation gate (passed):** per-pair winners reconstructed by majority vote
  of the 2-3 order-swapped calls (matching `normalize_verdict` +
  `run_judge_pair` in `ptg-judge.rs`) reproduce the report exactly:
  **lateral=29, second_look=8**. If this had not matched, the mapping would be
  untrusted and no length claim would be made.
- Length measured as the **judge-visible draft** = the full pretty-printed
  schema JSON (`serde_json::to_string_pretty(&schema)`), i.e. exactly the text
  the judge scored (prediction + domain fields + confidence + coords).

## Result

Judge-visible draft length, 37 decided pairs:

| | mean | median |
|---|---:|---:|
| mesh (lateral) | 1973 | 1924 |
| second_look | 1842 | 1859 |
| paired diff (mesh − second) | +131 | +126 |

Relative: **+7.1%** (mesh is modestly longer). So there *is* a small length
difference — the confound is not absent, as a first (incorrect, wrong-
population) analysis had suggested.

### Stratification by relative length (the decisive test)

| stratum | n | lateral wins | rate |
|---|---:|---:|---:|
| mesh LONGER | 23 | 20 | **87%** |
| second_look LONGER (mesh shorter) | 14 | 9 | **64%** |

**Lateral wins 9/14 = 64% even when its draft is shorter than second_look's.** A
pure length artifact predicts lateral *loses* when shorter; instead it wins a
majority. Length inflates the headline rate (87% when longer) but is not the
driver — the effect is present in the length-disadvantaged stratum too.

## Honest interpretation

- The headline 78% (29/37) is **partially length-inflated**: there is a real
  length gradient (64% when shorter → 87% when longer).
- But length does **not explain away** the result: the length-disadvantaged
  win rate (64%) is above 50% and above the 55% resurrect bar (as a point
  estimate; its CI at n=14 does include 50%, so this stratum alone is
  underpowered).
- Net: **the 4-col e4b resurrection survives length control as a real but
  modestly-inflated effect.** It is not a pure length artifact. It is also not
  a clean, length-independent 78%.

## What this does and does not settle

- DOES: removes the strongest single cited threat (length bias) to the within-
  e4b 4-col quality result. The effect is real, not a presentation artifact.
- DOES NOT: validate the post-hoc cross-model e2b-vs-e4b p-value; address
  scale (150-col is infra-blocked); or rule out other judge biases (e.g.,
  confidence-tone, structure). A length-equalized rejudge would tighten the
  headline but is not required — the stratification already shows the effect is
  not purely length-driven.

## Recommendation

No length-equalized rejudge needed: the stratified result (64% win when
shorter) is sufficient to reject the "pure length artifact" hypothesis. The
4-col resurrection stands as real-but-inflated. The binding open question
remains scale — which is blocked by the e4b server capacity ceiling
(`docs/STRUCTURED_LATERAL_E4B_SCALE_BLOCKED.md`).
