# Roadmap

PTG is built incrementally against the [specification](./SPECIFICATION.md) and
[architecture](./ARCHITECTURE.md). Phases 1+ are derived from the specification
captured at init; the source document was truncated after §3, so later-phase
detail will be refined as the design firms up.

## Phase 0 — Architecture scaffold ✅

- [x] Fresh repository under `saorsa-labs/brain`.
- [x] Compiling Rust workspace with crate boundaries matching the spec.
- [x] Core domain model (`ptg-core`).
- [x] Engine + consensus + runtime type skeletons.
- [x] Specification, architecture, and roadmap docs in-repo.

## Phase 1 — Core execution path

- [ ] Concrete `VllmEngine` HTTP client (vLLM `/v1/completions`-style), with
      prefix-caching-friendly prompt layout and structured-JSON enforcement.
- [ ] Parallel fan-out of stimulus to all columns (`ptg-runtime`).
- [ ] Lateral context injection along the `Topology` between rounds.
- [ ] Full voting loop with round-over-round output-delta convergence.
- [ ] A `ColumnSpec` registry / loader (declarative column definitions).

## Phase 2 — Reference frames & multi-modality

- [ ] Reference-frame JSON schemas per modality (visual, tactile, abstract…).
- [ ] Validation that column outputs conform to their frame schema.
- [ ] Multimodal stimulus handling for the `Gemma-4-2B-Multimodal` engine.

## Phase 3 — Topologies & benchmarks

- [ ] Pluggable topologies (grid, small-world, learned weights).
- [ ] Workstation-tuned deployment on unified-memory hardware.
- [ ] Benchmarks: latency, token economy vs. single-monolithic-context baseline,
      and convergence quality.

## Phase 4 — CLI & observability

- [ ] `ptg` CLI for defining meshes, running queries, and inspecting rounds.
- [ ] Tracing of per-round predictions and confidence for debugging emergence.
