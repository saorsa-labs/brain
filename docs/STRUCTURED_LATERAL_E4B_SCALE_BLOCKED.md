# Structured Lateral e4b at 150 columns — BLOCKED by server reliability ceiling

> **Status: INCONCLUSIVE (infrastructure failure, not quality result).**
> The 150-column structured-e4b scale test could not be completed because the
> e4b inference server fails reproducibly ~80-90% through the mesh arm.

## UPDATE: root cause found and fixed — 150 cols now UNBLOCKED

The ceiling was **not** a server capacity limit. It was **unbounded client
fan-out**: the runtime's `join_all` over all columns fired 150 concurrent
requests, overwhelming the server's connection handling (transport errors +
task-cancellation bursts). Diagnosis and fix are in
[`E4B_SERVER_TUNING.md`](./E4B_SERVER_TUNING.md).

Fix: `--column-concurrency 4` bounds in-flight column ticks. Validated: a fresh
4-slot e4b server ran mesh-only 150 cols at 300/300 with 0 errors. 150-col
quality runs are now possible. (The conclusions below from the failed attempts
remain valid as the pre-fix record.)

## What was attempted

Three attempts to run the A3 comparison (mesh_adaptive vs
sphere_x4_second_look_no_lateral) at 150 columns on `gemma-4-e4b`
(`http://127.0.0.1:18135`), structured lateral mode, matching the original e2b
150-col failure condition:

| Attempt | Topology | Mesh arm result | Control arm result |
|---|---|---|---|
| 1 | small-world d4, diversity k2 | **239/300 calls, HTTP 500, FAIL** | 300/300, ok (run 1) |
| 2 | small-world d4, diversity k2 (retry) | **265/300 calls, transport error, FAIL** | — (mesh-only) |
| 3 | ring k1 (lightest viable) | **268/300 calls, transport error, FAIL** | 300/300, ok (run 3) |

Run dirs: `bench-runs/1782504853793/`, `bench-runs/1782506547436/`,
`bench-runs/1782508764210/`.

## The failure pattern is robust and topology-independent

The mesh arm dies at **240-270 of 300 calls in every attempt**, regardless of
topology density (600-edge small-world or 150-edge ring), then the server
recovers and the control arm completes cleanly. The control arm — whose tick-2
prompts are empty-lateral (short, fast, low-memory) — completes 300/300 every
time. Only the mesh arm (tick-2 prompts carrying structured lateral context,
fanned across ~100+ receivers) fails.

This is a **reproducible server-capacity ceiling**: the 4B-model server (~9 GB)
cannot sustain a ~11-minute, 300-call mesh run with the heavier tick-2 prompts.
It fails at roughly the same point (~80-90% through) each time, consistent with
resource accumulation / memory pressure under sustained load rather than a
transient blip. The existing 3-attempt retry hardening (500ms/1s/2s backoff)
does not rescue it — the overload window exceeds the ~3.5s retry budget.

## Why this is not a quality finding

- No mesh_adaptive run at 150 cols produced a complete, parseable output set.
- Therefore no judge comparison is possible — there is nothing to judge.
- The e2b server (2.7 GB) sustained identical 150-col mesh runs (see
  `SCALE_SMOKE_FINDINGS.md`); only the e4b server (9 GB) hits this ceiling.
- This says nothing about whether structured lateral exchange helps or hurts at
  150 columns on a 4B model. It says the current single-slot e4b server cannot
  produce the data to answer that question.

## What this means for the project

The headline comparison we wanted — "does structured-e4b rescue the 150-col
result that failed on e2b (lateral 3/21, 57% echo)?" — **cannot be made with the
current infrastructure.** Three independent attempts, including the lightest
possible topology, all hit the same wall.

Two paths forward, neither requiring more 150-col hammering:

1. **Address the more important confound first (cheaper, more diagnostic).**
   The team review of the 4-col e4b "resurrection" (lateral 29/37 = 78.4%)
   identified **length/richness bias** as the #1 unaddressed threat: the
   llama-70b judge may prefer e4b lateral outputs simply because they are longer.
   Re-judging the *existing* 4-col e4b data with length control (length-blind
   judge prompt, or length-normalized scoring) attacks this directly, uses data
   we already have, and costs ~minutes. If lateral still wins under length
   control, the 4-col resurrection is far more credible.

2. **Run at a scale the e4b server can sustain.** The e2b scale smoke showed
   50 columns completes reliably. A 50-col structured-e4b run would test
   whether the 4-col effect holds beyond 4 columns without hitting the 150-col
   ceiling.

Both are recommended over further 150-col retries, which have now failed three
times for the same infra reason.
