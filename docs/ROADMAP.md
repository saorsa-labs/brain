# Roadmap

PTG is built incrementally against the [specification](./SPECIFICATION.md) and
[architecture](./ARCHITECTURE.md).

## Phase 0 — Repository & architecture scaffold ✅

- [x] Fresh repository under `saorsa-labs/brain`.
- [x] Compiling Rust workspace with crate boundaries matching the spec.

## Phase 1 — V1 execution skeleton ✅

- [x] Core domain model: `CorticalColumn`, `DomainSphere`, `ColumnOutputSchema`,
      `HistoryBuffer`, and the four `PROMPT_*` constants (§3.1.2, §5, §8.1).
- [x] Shared inference engine (`InferenceEngine`) with connection-pooled
      reqwest, typed responses, and a `ColumnEngine` trait for testability
      (§3.1.1, §8.3).
- [x] Three-phase epoch loop: parallel feed-forward, lateral context injection,
      and global integration, with fail-fast error handling (§6).
- [x] Metric-based convergence on the confidence vector (mean / delta / cosine)
      via `ndarray` (§6 Phase 3).
- [x] `ptg` CLI with `clap` + `tracing`, `--dry-run` offline mode, and the
      reference wiring (§8.4).
- [x] Panic-free, clippy-clean (`-D warnings`), 30 unit tests (mock engine, no
      live server required).

## Phase 2 — Real inference & multi-modality ✅

- [x] End-to-end run against a **live** OpenAI-compatible server: validated
      against a local `llama.cpp` `llama-server` (`gemma-4-e4b`); a 2-tick text
      epoch converged in 1 tick at mean confidence ~0.94–0.98 with all four
      columns passing strict per-sphere schema validation. (`ptg --probe`,
      `list_models`, and an `#[ignore]` live integration test were added.)
- [x] Reference-frame JSON schema validation per sphere
      (`ColumnOutputSchema::validate_for_sphere`, enforced in the engine).
- [x] Multimodal stimulus (image/audio): `Stimulus`/`StimulusPart` serialize to
      the exact OpenAI content-array shapes; CLI `--image-url`/`--image-detail`.
      Unit-tested, **not** live-validated (no multimodal model running).
- [x] Confidence-aware filtering in global integration (`accepted_outputs` /
      `rejected_outputs` at `min_integration_confidence`, §6 Phase 3).
- [x] Panic-free, clippy-clean (`-D warnings`); 32 unit tests + live epoch run.
- [ ] Deferred security hardening (see ARCHITECTURE "Security posture"): the
      red-team review found **0 critical** issues for the local-trusted-server
      posture, but SSRF via arbitrary `--vllm-url`, unbounded payloads, and
      verbatim output printing are documented to revisit before exposing the
      CLI to untrusted input.

## Benchmark — Mesh vs monolithic baseline (in progress)

Ground the thesis: is the cortical *mechanism* better than a single-monolithic
context (and a compute-matched ensemble) for latency, token economy, and answer
quality? Methodology is committed in [`docs/BENCHMARKING.md`](./BENCHMARKING.md);
pilot first (5 prompts × 3 repeats), scale to ~50+ only if the harness and
compute-matched controls pass review. No headline quality claim until then.

- [ ] Committed methodology doc + fair-baseline design (C1–C4 confounds neutralized).
- [ ] Instrument the engine for per-call `usage` (incl. `cached_tokens`) +
      `finish_reason`, without changing the `ColumnEngine` trait.
- [ ] `ptg-bench` harness: `mesh_adaptive` / `mono_all_prompts` / `mono_x4`;
      JSONL raw records + Markdown summary.
- [ ] Pilot run against `gemma-4-e4b`; record results (pilot-only, no headline).
- [ ] Scale to ~50+ prompts + paired statistics if pilot is sound.

## Phase 3 — Topologies & convergence depth (in progress)

- [~] Pluggable topologies: ring, torus, small-world (§3.1.3). **Library +
      runtime + CLI `--topology` flag landed.** `TopologySpec` in `ptg-core`, `mesh_with_topology` in `ptg-runtime`, `--topology/--columns` in `ptg`. Bench `--topology` flag deferred until a topology-aware benchmark is wired.
- [ ] Weighted / attention-routed lateral connections (§9.1 "Dynamic Topology
      Scaling") — columns choose which neighbors to listen to.
- [ ] Full **semantic** cosine convergence over prediction embeddings (§9.3),
      requiring an embedding backend (blocked: live server returns HTTP 501 on
      `/v1/embeddings`).
- [ ] Early-stop similarity checks between iterations (cheap string proxy as a
      `ConvergenceCriteria` field — unblocked, not yet built).

## Phase 4 — Scale & observability

- [ ] Workstation tuning on unified-memory hardware (§7): ring-buffer memory
      budget audit, HTTP/2 multiplexing under load.
- [ ] Benchmarks: latency, token economy vs. single-monolithic-context
      baseline, convergence quality.
- [ ] Per-round tracing of predictions and confidence for debugging emergence.
- [ ] `dashmap`-backed column store for high-contention scale (dependency to
      be added when used).
