# A3 Findings — equal-call no-lateral second-look control

Date: 2026-06-25  
Raw run (gitignored): `bench-runs/1782423155348/`  
Judge report: [`docs/a3-judge-report.md`](./a3-judge-report.md)

## Scope

This is a **pilot-scale** A3 run: 5 prompts × 3 repeats × 5 conditions against
`gemma-4-e2b-qat` on `http://127.0.0.1:18136`, forced to `--min-ticks 2
--max-ticks 2` with `--routing-policy all`.

A3 compares:

- `mesh_adaptive`: 4 columns × 2 ticks = 8 calls, tick 2 receives lateral neighbor text.
- `sphere_x4_second_look_no_lateral`: 4 columns × 2 ticks = 8 calls, tick 2 receives no neighbor text.

This controls for **call budget / compute**, not for a genuine self-revision pass.
The no-lateral second tick is expected to be an inert replay at `temperature=0`.
A future A4 would be needed to compare lateral exchange against an explicit
"reconsider your previous answer" revision prompt.

## Benchmark aggregate

| condition | n | parse_ok | wall_lat med/min/max ms | sum_call_lat med ms | gross total tok med | cache-adj total tok med | cached% med | calls med | truncated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mesh_adaptive | 15 | 14 | 9706 / 7162 / 13703 | 33397 | 5570 | 3693 | 63.7 | 8 | 1 |
| mono_all_prompts | 15 | 15 | 3040 / 1972 / 4067 | 3040 | 1295 | 912 | 39.4 | 1 | 0 |
| mono_x4 | 15 | 15 | 12090 / 7803 / 15167 | 12088 | 5180 | 1848 | 99.9 | 4 | 0 |
| sphere_x4_no_lateral | 15 | 15 | 4457 / 2856 / 5217 | 16433 | 2604 | 1379 | 98.4 | 4 | 0 |
| sphere_x4_second_look_no_lateral | 15 | 15 | 8965 / 5803 / 12292 | 33433 | 5208 | 2806 | 98.3 | 8 | 0 |

## A2 replication inside this run

The lateral mechanism still **activates** strongly, but activation is not quality:

- A2 pairs analyzed: 42
- A2 excluded pairs: 13 (`tick2_echoed_neighbor`)
- Non-excluded median prediction edit distance:
  - PHYSICS: 0.741
  - MATH: 0.726
  - CODE: 0.613
- LLM corroborating judge: `tick_1: 16`, `tick_2: 12`, `tie: 1`

So the previous pattern replicated: lateral context moves outputs, but the judge
does not prefer the moved outputs.

## A3 result

A3 asks whether lateral-final outputs beat equal-call no-lateral second-look finals.

- A3 pairs analyzed: 42
- A3 excluded pairs: 19
  - `mesh_echoed_neighbor`: 13
  - `control_second_look_unstable`: 6
- No-lateral second-look instability was isolated to 2 control runs:
  - P3 repeat 0: `CC_PSYCH_01`
  - P4 repeat 0: `CC_PHYSICS_01`
- Non-excluded median prediction edit distance between lateral-final and
  no-lateral-final:
  - PHYSICS: 0.725
  - MATH: 0.726
  - CODE: 0.613
- LLM corroborating judge: `lateral: 11`, `second_look: 12`

## Interpretation

At this pilot scale, **lateral exchange does not beat the equal-call no-lateral
control**. The current dense text-exchange mechanism remains expensive, leaks
neighbor text often enough to require exclusions, and is not quality-positive
under the blind corroborating judge.

This strengthens the current roadmap decision: do **not** scale the current dense
lateral-text design yet. Next sensible branches are:

1. Run the same A3 control on a larger model (`gemma-4-e4b`) to test the "model too
   small to integrate lateral context" hypothesis.
2. Build an A4 explicit self-revision control if we want to separate lateral
   exchange from a real second-pass revision prompt.
3. Start the `ptg-belief` vertical slice: typed hypotheses, evidence,
   provenance, posterior updates, and dependence-aware aggregation instead of
   raw neighbor text + self-reported confidence.

This is still a successful pilot: it found that the implemented mechanism
activates, but does not yet improve quality under the controls we can defend.
