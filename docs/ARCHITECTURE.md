# Architecture

This document maps the [specification](./SPECIFICATION.md) onto the concrete
crate structure in this repository. The authoritative narrative is
`SPECIFICATION.md`; this file describes how the code realizes it and where the
shipped implementation refines the §8 reference blueprint for production.

## Design principles

1. **One shared engine, many columns (§3).** A single inference engine instance
   (the "thalamus") is shared behind an `Arc<dyn ColumnEngine>` by every column
   — we never load a model per column.
2. **Local reference frames (§2.3, §5).** Each column's perception is bounded by
   a `ReferenceFrame` expressed as structured JSON, keeping per-column context
   small (countering context dilution).
3. **Lateral, not centralized (§6).** There is no master coordinator. Columns
   exchange predictions along a topology and re-evaluate until metric-based
   convergence criteria are met.
4. **Engine-agnostic (§3.1.1).** The inference boundary is the `ColumnEngine`
   trait, so the spec's `vLLM` backend can be swapped for `mistral.rs` or any
   OpenAI-compatible server without touching the mesh runtime — and the runtime
   is unit-testable with a mock engine (no live server in CI).

## Crate map

| Crate | Spec section | Key types |
| --- | --- | --- |
| `ptg-core` | §3.1.2, §5, §8.1 | `DomainSphere`, `CorticalColumn`, `HistoryBuffer`, `ColumnOutputSchema`, `PROMPT_*`, `default_columns`, `default_connections` |
| `ptg-vllm` | §3.1.1, §8.3 | `ColumnEngine` trait, `InferenceEngine`, `EngineBuilder`, `EngineError` |
| `ptg-consensus` | §6 Phase 3 | `ConvergenceCriteria`, `mean_confidence`, `confidence_vector`, `cosine_similarity`, `quality_converged` |
| `ptg-runtime` | §3.1.3, §6, §8.2 | `CorticalMesh`, `MeshResult`, `MeshError`, `default_mesh` |
| `ptg-cli` | §8.4 | `ptg` binary (`clap` args, `--dry-run`) |

## The three-phase epoch loop (§6)

`CorticalMesh::run_epoch` implements the spec's discrete epoch loop:

```
       input payload
            |
            v
  +-----------------------+   Phase 1: parallel feed-forward
  | broadcast to columns  |   (stimulus cloned to every column)
  +-----------+-----------+
              |
              v
  +-----------------------+   Phase 2: lateral exchange (tick loop)
  | for tick in 1..=max:  |   - build neighbor context from last_prediction
  |   fan out column ticks|   - concurrent execute_column_tick (join_all)
  |   fail-fast on errors |   - record_tick updates state + history_buffer
  |   check convergence   |   - stop early if quality_converged
  +-----------+-----------+
              |
              v
  +-----------------------+   Phase 3: global integration
  | MeshResult            |   - final outputs, mean confidence, stabilized flag
  +-----------------------+
```

1. Column ids are processed in **sorted order** for reproducible consensus
   (fixing the non-deterministic `HashMap` iteration in the §8 reference).
2. The fan-out uses `futures::future::join_all` over the shared `Arc`'d engine.
3. Engine errors **fail fast** (`MeshError::Engine`) rather than being silently
   swallowed (unlike the `eprintln`/`None` path in the §8 reference).

## Convergence (§6 Phase 3, "Compute Cosine Matrix Convergence")

`ptg-consensus` provides metric-based convergence on the per-column confidence
vector using `ndarray`:

- mean confidence `>= min_mean_confidence`, **or**
- mean absolute confidence delta between ticks `<= max_confidence_delta`, **or**
- cosine similarity of successive confidence vectors `>= min_cosine_similarity`.

`quality_converged` returns `true` only when a quality criterion is met, so the
`stabilized` flag in `MeshResult` distinguishes genuine stabilization from
simply running out of ticks. **Full semantic cosine over prediction embeddings
is future work** (§9.3): it requires an embedding backend the V1 engine doesn't
expose.

## Refinements over the §8 reference blueprint

These are the production-quality deltas applied on top of the reference code in
§8 (the reference snippets are preserved verbatim in `SPECIFICATION.md` for
provenance):

| §8 reference | Shipped implementation | Why |
| --- | --- | --- |
| `ColumnOutputSchema` hard-codes `empirical_observation` | Common fields (`reference_frame_coordinates`, `prediction`, `confidence`) + `#[serde(flatten)] domain_fields` | The Math/Coding/Psych prompts emit `deductive_synthesis`/`algorithmic_analysis`/`behavioral_synthesis`; a single-sphere struct would fail to parse them |
| `Client::builder().build().unwrap()` | `InferenceEngine::new` / `EngineBuilder::build` return `Result` | Panic-free policy (AGENTS.md) |
| `columns.get(id).unwrap()` | fail-fast `MeshError::Engine` / `?` propagation | Panic-free policy |
| No `history_buffer` | `CorticalColumn.history_buffer: HistoryBuffer` (fixed-capacity ring buffer) | Spec §3.1.2 / §7.2 require it |
| Fixed `max_ticks`, no convergence | `quality_converged` with confidence-vector delta + cosine | Spec §6 Phase 3 requires convergence |
| `serde_json::Value` indexing of the response | typed `ChatCompletionResponse` structs | Robust parsing |
| `eprintln` + continue on engine error | fail-fast `MeshError::Engine` | Errors surface instead of being swallowed |
| Non-deterministic `HashMap` tick order | sorted column ids | Reproducible consensus |

## Adjacency model (V1)

V1 uses an **unweighted** directed adjacency list, faithful to §8.2. Although
the prose mentions "weight voting", §9.1 itself defers dynamic, attention-
weighted routing to future work — so unweighted topology is correct for V1 and
weighted edges stay on the [roadmap](./ROADMAP.md).

## Running it

```bash
cargo check --workspace --all-targets
cargo clippy --all-features --all-targets -- -D warnings
cargo test
cargo run -p ptg-cli -- --help
cargo run -p ptg-cli -- --dry-run          # offline wiring check
cargo run -p ptg-cli -- --vllm-url http://localhost:8000   # needs a live vLLM server (§7.1)
```
