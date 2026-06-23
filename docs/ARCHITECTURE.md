# Architecture

This document maps the PTG specification onto the concrete crate structure in
this repository. The authoritative narrative is
[`SPECIFICATION.md`](./SPECIFICATION.md); this file describes how the code is
organized to realize it.

## Design principles

1. **One shared engine, many columns.** A single inference engine instance (the
   "thalamus") is multiplexed across all virtual columns via dynamic system
   prompts and prefix caching — we never load a model per column.
2. **Local reference frames.** Each column's perception is bounded by a
   `ReferenceFrame` expressed as structured JSON, keeping per-column context
   small and focused (countering context dilution).
3. **Lateral, not centralized.** There is no master coordinator. Columns
   exchange predictions along a weighted `Topology` and re-evaluate until
   `ConvergenceCriteria` are met.
4. **Engine-agnostic.** The inference boundary is the `VllmEngine` trait, so the
   spec's `vLLM` backend can be swapped for `mistral.rs` or another engine
   without touching the mesh runtime.

## Crate map

| Crate | Role | Key types |
| --- | --- | --- |
| `ptg-core` | Domain model | `ColumnId`, `Modality`, `ReferenceFrame`, `ColumnSpec`, `ColumnOutput`, `LateralMessage`, `Topology` |
| `ptg-vllm` | Shared inference engine ("thalamus") | `VllmEngine` trait, `InferenceRequest`, `EngineError` |
| `ptg-consensus` | Lateral voting + convergence | `VotingState`, `ConsensusRound`, `ConvergenceCriteria`, `has_converged`, `neighbor_weights` |
| `ptg-runtime` | Mesh orchestration | `CorticalMesh`, `MeshResult` |
| `ptg-cli` | Entry point | `ptg` binary |

## Data flow (target)

```
        stimulus
           │
           ▼
 ┌─────────────────────┐    fan-out (parallel)
 │   CorticalMesh      │ ─────────────────────▶  Column 0 … Column N
 │  (ptg-runtime)      │                           │  (each: system prompt +
 └─────────────────────┘                           │   reference frame)
           ▲                                       ▼
           │                               ┌───────────────┐
           │  lateral token injection      │  VllmEngine    │  ← single shared
           │  (Topology edges)             │  (ptg-vllm)    │     instance
           │                               └───────┬───────┘
           │                                       │ ColumnOutput
           │                                       ▼
           │                          ┌─────────────────────────┐
           └──────────────────────────│  consensus / voting     │
                                      │  (ptg-consensus)        │
                                      └────────────┬────────────┘
                                                   │ converged? no → next round
                                                   ▼
                                          MeshResult (final_state)
```

1. The runtime fans stimulus out to every column in parallel.
2. Each column calls the shared `VllmEngine` with its own system prompt +
   reference frame + injected neighbor context.
3. `ptg-consensus` collects `ColumnOutput`s, folds neighbor votes by topology
   weight, and checks `has_converged`.
4. Until convergence, neighbor outputs are re-injected as lateral context and
   the loop repeats; the final `VotingState` is returned.

## Status of each layer

- `ptg-core`: types defined and tested.
- `ptg-vllm`: trait + error types defined; concrete HTTP client to be
  implemented in Phase 1.
- `ptg-consensus`: convergence + weight helpers defined; full output-delta
  metric and voting loop to be implemented in Phase 1.
- `ptg-runtime`: mesh struct defined; orchestration loop to be implemented in
  Phase 1.

See [`ROADMAP.md`](./ROADMAP.md) for the phase breakdown.
