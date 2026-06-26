# Scale Smoke Findings — 150-column sparse mesh

Date: 2026-06-25  
Model/server: `gemma-4-e2b-qat` at `http://127.0.0.1:18136`  
Prompt: thermostat firmware overshoot diagnostic task.

## Goal

Test whether PTG can structurally run at ~150 columns without dense O(N²)
lateral text exchange. This is a **scale/robustness smoke test**, not a quality
claim: generated 150-column meshes reuse the four default sphere prompts, so they
are many replicated reference frames rather than 150 independently designed
experts.

## Initial unpatched results

- 150-column dry-run succeeded:
  - `small-world`, 150 columns, degree 4, 600 lateral edges.
- Live 150 with `--max-tokens 256` failed by JSON truncation:
  - `CC_CODE_01`: EOF while parsing a string.
- Live 150 with `--max-tokens 512` and sparse routing failed with transient HTTP
  send errors while the server remained alive:
  - `small-world + diversity k=2`: failed at `CC_MATH_14` after ~238s.
  - `ring-bi + confidence-top-k k=1`: failed at `CC_PSYCH_16` after ~121s.
- A 50-column sparse run succeeded before adding retries:
  - `ring-bi`, `confidence-top-k`, `routing-k=1`, 2 ticks, 512 cap.
  - 50 accepted, 0 rejected, ~81.6s.

Interpretation: 150 columns were structurally valid, but the client/server path
was brittle for long serial runs. The failures were transport-level, not topology
or schema failures.

## Retry hardening

Added bounded retries in `ptg-vllm` for chat-completions requests:

- max attempts: 3
- backoff: 500ms, 1s, 2s
- retries only transport failures and retryable statuses:
  - 408, 429, 500, 502, 503, 504
- does **not** retry JSON/schema/truncation failures.

Validated with strict clippy (`panic/unwrap/expect` denied) and `just check`.

## Successful 150-column smoke tests after retry hardening

### Conservative sparse ring

Command shape:

```bash
cargo run -q -p ptg-cli --bin ptg -- \
  --vllm-url http://127.0.0.1:18136 \
  --model gemma-4-e2b-qat \
  --topology ring \
  --columns 150 \
  --routing-policy confidence-top-k \
  --routing-k 1 \
  --ticks 2 \
  --min-ticks 2 \
  --max-tokens 512 \
  --input "...thermostat prompt..."
```

Result:

- topology: ring, 150 columns, 150 lateral edges
- epoch complete: 2 ticks
- stabilized: true
- mean confidence: 0.915
- integration: 150 accepted, 0 rejected
- runtime: ~269.3s
- log: `/tmp/ptg-150-ring-k1-512-retry.log`

### Small-world + diversity routing

Command shape:

```bash
cargo run -q -p ptg-cli --bin ptg -- \
  --vllm-url http://127.0.0.1:18136 \
  --model gemma-4-e2b-qat \
  --topology small-world \
  --columns 150 \
  --small-world-degree 4 \
  --small-world-rewire 0.10 \
  --small-world-seed 42 \
  --routing-policy diversity \
  --routing-k 2 \
  --ticks 2 \
  --min-ticks 2 \
  --max-tokens 512 \
  --input "...thermostat prompt..."
```

Result:

- topology: small-world, 150 columns, 600 lateral edges
- route budget: diversity-preserving up to 2 sources/listener
- epoch complete: 2 ticks
- stabilized: true
- mean confidence: 0.913
- integration: 150 accepted, 0 rejected
- runtime: ~267.6s
- log: `/tmp/ptg-150-smallworld-diversity-k2-512-retry.log`

## Conclusion

150 columns are **technically viable** with sparse topology, k-limited routing,
512-token caps, and bounded request retries. Do **not** use dense/fully-connected
`all` routing at this scale: it risks O(N²) neighbor text, cache collapse, and
context blowups.

This result is a robustness/scale result only. Current A3 evidence still says
lateral text exchange activates but does not beat the equal-call no-lateral
control at pilot scale. Quality claims at 150 columns require topology-aware
benchmarking and a generalized judge.

## Recommended next engineering steps

1. Add topology-aware large-column support to `ptg-bench` so scale runs write
   structured JSONL/metrics instead of only terminal output.
2. Generalize `ptg-judge` beyond the default 4-column graph and `CC_PSYCH_01`
   determinism gate.
3. Add fail-soft partial-result reporting for long runs, so a late column failure
   does not discard hundreds of successful calls.
4. Consider a terse/scale-specific column pack if repeated default prompts are
   too homogeneous for meaningful 150-column experiments.

## Scale A3 benchmark (default replicated prompts, 150 columns)

After the retry hardening, a full `ptg-bench` A3 run completed at 150 columns:
small-world (degree 4, 600 edges), diversity routing k=2, 2 ticks, 1 prompt,
1 repeat, **default replicated sphere prompts** (not the differentiated pack —
see the schema-compliance caveat below). Run: `bench-runs/1782488218572/`.

Per-condition (both 300 calls = 150 columns × 2 ticks, equal budget):

| condition | cache-adj total tok (med) | cached% (med) |
|---|---:|---:|
| mesh_adaptive | 174101 | 42.5 |
| sphere_x4_second_look_no_lateral | 123540 | 83.8 |

Lateral exchange again roughly halves the prefix-cache hit rate (42.5% vs
83.8%) — the same cache-collapse pattern seen in the 4-column pilot (43.5% vs
98%+).

### A3 judge (generalized, route-derived receivers, 60-pair sample)

The generalized judge derived ~150 lateral receivers from the routes and sampled
60 (stride, `--max-pairs 60`). With the per-column determinism gate:

- A3 pairs analyzed: 60 (sampled from 150)
- excluded: 39
  - `mesh_echoed_neighbor`: 34 (**57% echo-leakage rate** — severe at scale)
  - `control_second_look_unstable`: 5
- judged winners: **lateral 3, second_look 18**

### Interpretation

At 150 columns the result is stronger and more negative than the 4-column
pilot (which was a coin flip, 11 vs 12):

- lateral exchange is **strongly disfavored** by the blind judge (3 vs 18);
- neighbor-text echo leakage is **severe** (57%), because dense lateral text
  routinely leaks verbatim into outputs and gets excluded;
- the cache-collapse penalty persists.

This is directional (1 prompt, 1 repeat, default replicated prompts, same-model
judge), not a powered quality claim. But it reinforces the core conclusion:
**do not scale the current dense lateral-text exchange.** Scaling makes the
mechanism worse on every measured axis (quality, leakage, cache economy).

### Differentiated pack: schema-compliance blocker

`examples/column-packs/scale-diagnostic-150.toml` differentiates columns by lens
axes, but its concise prompts do not yet elicit reliable per-sphere JSON
compliance at `gemma-4-e2b`: some Coding columns omit `state_variables`, failing
`validate_for_sphere` and aborting the fail-fast mesh (even at a 1024-token cap;
the same column's prompt produces valid JSON in an isolated direct call, so the
failure is a compliance-fragility of the concise prompt under the mesh path).
The hand-tuned default sphere prompts comply reliably. A robust differentiated
pack should base each prompt on the default sphere prompt and append only the
lens suffix, or move to a larger model.
