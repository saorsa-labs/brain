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

## Phase 2 — Real inference & multi-modality

- [ ] End-to-end run against a live vLLM server (§7.1) and validate the
      `Gemma-4-2B-Multimodal-Q4_K_M` model id and prefix-caching behavior.
- [ ] Reference-frame JSON schema validation per modality (reject outputs that
      don't conform to the column's frame).
- [ ] True multimodal stimulus (image/audio) through the multimodal engine.
- [ ] Confidence-aware filtering in global integration (drop low-confidence
      columns from the final percept — §6 Phase 3).

## Phase 3 — Topologies & convergence depth

- [ ] Pluggable topologies: ring, torus, small-world (§3.1.3).
- [ ] Weighted / attention-routed lateral connections (§9.1 "Dynamic Topology
      Scaling") — columns choose which neighbors to listen to.
- [ ] Full **semantic** cosine convergence over prediction embeddings (§9.3),
      requiring an embedding backend.
- [ ] Early-stop similarity checks between iterations.

## Phase 4 — Scale & observability

- [ ] Workstation tuning on unified-memory hardware (§7): ring-buffer memory
      budget audit, HTTP/2 multiplexing under load.
- [ ] Benchmarks: latency, token economy vs. single-monolithic-context
      baseline, convergence quality.
- [ ] Per-round tracing of predictions and confidence for debugging emergence.
- [ ] `dashmap`-backed column store for high-contention scale (dependency to
      be added when used).
