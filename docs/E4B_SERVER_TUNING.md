# e4b server 150-column ceiling — root cause & fix

> **Status: ROOT CAUSE FOUND AND FIXED.** The 150-col mesh failures were not OOM,
> not context limits, not stale uptime — they were **unbounded client fan-out**
> overwhelming the inference server. Fix: bounded column concurrency.

## The failure (3+ reproductions)

150-column `mesh_adaptive` runs on `gemma-4-e4b` (`:18135`) died at a variable
point (call 49 / 133 / ~250 of 300) with `error sending request for url` (a
transport error, not an HTTP 500). The no-lateral control arm always completed.
Documented in `STRUCTURED_LATERAL_E4B_SCALE_BLOCKED.md`.

## Diagnosis (logs-driven, one variable at a time)

Captured llama-server logs (`-np 1` and auto/4-slot variants) and ruled out:

| Hypothesis | Evidence | Verdict |
|---|---|---|
| Stale-server accumulation (2.5-day uptime) | Fresh restart failed too | ❌ |
| Multi-slot thrashing | `-np 1` failed *worse* (call 49) | ❌ |
| Context/slot limit | each slot `n_ctx = 4096`; prompts ~500–2000 tok fit | ❌ |
| Server crash / OOM | server stays alive, 5.5 GB RSS, 0 errors in log | ❌ |
| fd limit | `ulimit -n` = 1 048 575; server holds ~32 fds | ❌ |
| Request timeout | 120 s timeout; ~27 s to gen 1024 tok at 37 tok/s | ❌ |

The actual signature in every failure: the server log showed **bursts of
`cancel task`** (16–43 tasks at once) — i.e. the **client disconnected/gave up**
on a batch of in-flight requests. Healthy server, client transport errors,
variable point.

## Root cause

`crates/ptg-runtime/src/lib.rs` epoch loop used `join_all(futures).await` over
**all N columns** — so a 150-column mesh fired up to **150 concurrent HTTP
requests** at the server. With 1–4 slots, 146–149 requests pile up in the
connection/task backlog; connections get dropped at the TCP/HTTP layer → client
`error sending request` → fail-fast aborts the whole mesh. The server then
cancels the orphaned tasks (the log bursts).

This explains everything:
- **Why single-slot was worse**: fewer slots drain the backlog slower.
- **Why 50 cols worked but 150 didn't**: 50 is within connection capacity; 150 isn't.
- **Why e2b handled 150**: the 2 B model drains the backlog faster than the 4 B.
- **Why fresh restart didn't help**: structural, not accumulated state.

## Fix

Added `max_concurrent_column_ticks: Option<NonZeroUsize>` on `CorticalMesh`
(default `None` = the original unbounded `join_all` behavior, so all existing
behavior/tests are unchanged). The epoch loop now uses
`stream::iter(futures).buffered(limit).collect()` (order-preserving, unlike
`buffer_unordered`) when a cap is set.

`ptg-bench --column-concurrency N` (default **4**, matching the e4b auto slot
count; `clap range(1..)` rejects 0), applied to all three mesh run paths,
recorded in JSONL + the summary markdown.

## Validation

Mesh-only 150-col repro on a freshly-restarted **4-slot** e4b server with
`--column-concurrency 4`:

- **300/300 calls, `parse_ok: True`, `ok=true`** (previously failed 4×).
- **0 `cancel task` lines, 0 errors** in the server log (previously hundreds of
  cancellation bursts).

The fix holds with the server in its original 4-slot config — no `-np 1`, no
context bump, no retry expansion needed.

## Note on latency

Bounded concurrency raises wall-time (4 in flight instead of 150), but this is
expected and irrelevant to quality: per `BENCHMARKING.md`, use `sum_call_lat`,
not `wall_lat`, for cross-condition comparisons. The token economy and per-call
workload are unchanged.
