# Scale Benchmarking — large-column cortical mesh

This document covers running PTG at tens-to-hundreds of columns: the flags, the
differentiated column pack, and how the judge stays bounded. It is a companion
to [`SCALE_SMOKE_FINDINGS.md`](./SCALE_SMOKE_FINDINGS.md) (which records the raw
150-column smoke results) and [`BENCHMARKING.md`](./BENCHMARKING.md) (the core
mesh-vs-monolithic methodology).

## Why scale needs sparse routing

The default `all` routing injects **every** neighbor's prediction into each
listener's prompt. At 150 columns that is O(N²) lateral text and will blow
context, collapse the prefix cache, and multiply cost. Scale runs **must** use a
sparse topology (`ring`, `ring-bi`, `torus`, `small-world`) plus a k-limited
routing policy (`confidence-top-k` or `diversity`) with a small `--routing-k`.

The retry hardening in `ptg-vllm` (3 attempts, 500ms/1s/2s backoff, retries
only transport + 408/429/500/502/503/504) is what made long 150-column serial
runs reliable — without it, transient engine errors mid-run abort the whole
fail-fast bench.

## ptg-bench scale flags

`ptg-bench` now accepts the same topology family as `ptg`:

- `--topology {default,ring,ring-bi,torus,fully-connected,small-world}`
- `--columns N` (torus: `--torus-width/--torus-height`)
- `--small-world-degree/--small-world-rewire/--small-world-seed`
- `--column-pack PATH` (differentiated columns; see below)
- `--conditions a,b,c` (subset; overrides `--only`)
- `--prompt-limit N` (run only the first N prompts — fast scale iteration)

For a non-default topology, **monolithic conditions are rejected**: a
many-prompt monolith is not a fair baseline. Select mesh-style conditions
(`mesh_adaptive`, `sphere_x4_second_look_no_lateral`, `sphere_x4_no_lateral`).
Each record carries `topology`, `column_count`, and `edge_count`.

## Differentiated columns (else it is just replication)

A generated 150-column mesh over the 4 default sphere prompts is 37 copies of
each prompt — it measures replication, not specialization.
`scripts/generate-scale-column-pack.py` emits N columns whose prompts are
concise but differentiated along three axes (abstraction level, time horizon,
diagnostic stance), while still instructing the per-sphere JSON keys each column
must emit to pass `validate_for_sphere`. No new `DomainSphere` / schema envelope.

```bash
python3 scripts/generate-scale-column-pack.py 150 \
  > examples/column-packs/scale-diagnostic-150.toml
```

The emitted column count must equal `--columns`.

## Running a scale A3 bench

```bash
cargo run -q -p ptg-cli --bin ptg-bench -- \
  --vllm-url http://127.0.0.1:18136 --model gemma-4-e2b-qat \
  --topology small-world --columns 150 --small-world-degree 4 \
  --column-pack examples/column-packs/scale-diagnostic-150.toml \
  --routing-policy diversity --routing-k 2 \
  --conditions mesh_adaptive,sphere_x4_second_look_no_lateral \
  --prompt-limit 1 --repeats 1 --min-ticks 2 --max-ticks 2 --max-tokens-col 512
```

Cost shape: `columns × ticks × conditions × repeats` calls plus one warmup run
per condition. At 150 columns, 2 ticks, 2 conditions, 1 repeat that is
`300 × 2 = 600` measured calls plus `300 × 2 = 600` warmup calls. Run it detached
(`nohup ... &`) because it exceeds a single interactive timeout.

## Judging at scale (bounded)

A 150-column run can yield ~150 A3 pairs; judging all of them would be hundreds
of LLM calls. `ptg-judge` now:

- derives A3 receivers from the **routes** (columns with `lateral_active` on the
  mesh final tick), not the 3 named default-graph receivers;
- uses a topology-aware determinism gate (named `CC_PSYCH_01` sink when present;
  otherwise the no-lateral control's second-look stability);
- caps judged pairs via `--max-pairs` (default 60) with deterministic stride
  sampling (`--sample-seed`).

```bash
RUN=$(ls -td bench-runs/* | head -1)
cargo run -q -p ptg-cli --bin ptg-judge -- \
  --input "$RUN/results.jsonl" --out "$RUN/a3-judge-report.md" \
  --calls-out "$RUN/a3-judge-calls.jsonl" --judge --max-pairs 60
```

## What this is and is not

This is a **scale / instrumentation** capability plus a directional quality
signal. It is **not** a headline quality claim: the core A3 pilot (4 columns)
already showed lateral text exchange activates but does not beat the equal-call
no-lateral control at that scale, and a single 1-prompt scale run has no
statistical power. Use it to (a) confirm the pipeline runs at scale and (b) look
for large-effect directional signals before investing in a powered scale study.

## Known gap: fail-fast long runs

`ptg-bench` writes `results.jsonl` only after the full run completes. A single
late failure (even after retries) currently discards every earlier call. Adding
fail-soft partial-result reporting for long runs is the next robustness item.
