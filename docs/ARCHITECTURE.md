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
| `ptg-core` | §3.1.2, §5, §8.1 | `DomainSphere`, `CorticalColumn`, `HistoryBuffer`, `ColumnOutputSchema` (`validate`, `validate_for_sphere`), `Stimulus`/`StimulusPart`/`ImageDetail`/`AudioFormat`, `PROMPT_*`, `default_columns`, `default_connections` |
| `ptg-vllm` | §3.1.1, §8.3 | `ColumnEngine` trait (`execute_column_tick(&Stimulus)`), `InferenceEngine`, `EngineBuilder`, `EngineError`, `list_models` |
| `ptg-consensus` | §6 Phase 3 | `ConvergenceCriteria` (`min_integration_confidence`), `mean_confidence`, `confidence_vector`, `cosine_similarity`, `quality_converged` |
| `ptg-runtime` | §3.1.3, §6, §8.2 | `CorticalMesh` (`run_epoch(&Stimulus)`, `run_text_epoch`), `MeshResult` (`accepted_outputs`/`rejected_outputs`), `MeshError`, `default_mesh` |
| `ptg-cli` | §8.4 | `ptg` binary (`--image-url`, `--image-detail`, `--probe`, `--dry-run`) |
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
  | MeshResult            |   - outputs, accepted/rejected split (threshold),
  +-----------------------+     mean confidence, stabilized flag
```

1. Column ids are processed in **sorted order** for reproducible consensus
   (fixing the non-deterministic `HashMap` iteration in the §8 reference).
2. The fan-out uses `futures::future::join_all` over the shared `Arc`'d engine.
3. Engine errors **fail fast** (`MeshError::Engine`) rather than being silently
   swallowed (unlike the `eprintln`/`None` path in the §8 reference).
4. **Confidence-aware integration** (Phase 2): final outputs are partitioned
   into `accepted_outputs` (confidence ≥ `min_integration_confidence`, default
   0.5) and `rejected_outputs` (§6 Phase 3 "filters out low-confidence columns").

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

### Known limitation: confidence convergence assumes a calibrated model

Confidence-vector convergence is **driven by the model's self-reported
confidence**, and a non-calibrated model can game it. Concretely: small
instruction-tuned models (e.g. `gemma-4-e4b`) frequently self-report
`confidence ≈ 1.0`, so `quality_converged` fires on tick 1 via the
`mean_confidence >= min_mean_confidence` branch — and the mesh stabilizes
**before any lateral exchange happens**. In that regime the lateral-voting
mechanism (the system's headline feature) is effectively dead code, and the
`accepted`/`rejected` split, `mean_confidence`, and `stabilized` are all
non-informative.

Mitigations:

- `ConvergenceCriteria.min_ticks` (default `1`) forces at least `min_ticks`
  ticks before convergence is considered, guaranteeing lateral exchange runs.
  `min_ticks >= 2` is recommended for mechanism-measurement / benchmark runs.
- A production default change to `min_ticks >= 2` (or a non-self-reported
  convergence signal) is **pending evidence** from the benchmark quality pass.
- Semantic convergence (§9.3) remains the proper long-term fix but needs an
  embedding backend the current local server does not provide.

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
| `execute_column_tick(column, input_data: &str, lateral)` | `execute_column_tick(column, stimulus: &Stimulus, lateral)` — text serializes as a plain string, multimodal as the OpenAI content-part array (text part first, then `image_url`/`input_audio`) | §2.3 multi-modality (Phase 2) |
| No per-sphere output validation | `ColumnOutputSchema::validate_for_sphere` enforces each sphere's required domain fields; called in `execute_column_tick` | §5 reference frames (Phase 2) |
| No confidence filtering | `MeshResult` splits into `accepted_outputs`/`rejected_outputs` at `min_integration_confidence` | §6 Phase 3 (Phase 2) |
| No reachability check | `ptg --probe` / `list_models` | Operational (Phase 2) |

## Adjacency model (V1)

V1 uses an **unweighted** directed adjacency list, faithful to §8.2. Although
the prose mentions "weight voting", §9.1 itself defers dynamic, attention-
weighted routing to future work — so unweighted topology is correct for V1 and
weighted edges stay on the [roadmap](./ROADMAP.md).

## Pluggable topologies (Phase 3)

Topologies are now **pluggable** (§3.1.3). `TopologySpec`
(`crates/ptg-core/src/topology.rs`) declares the graph as a pure function over
an ordered `N`-column id list — `Ring`, `Torus2d`, `FullyConnected`,
`SmallWorld`, or `Custom` — and `connections_for(&ids)` materializes it into a
`Vec<LateralConnection>`. The runtime builds a mesh from columns plus a
topology via `mesh_with_topology`, or from an explicit edge list via
`mesh_from_columns`. The named 4-column reference topology (`default_mesh`) is
**unchanged** and remains the benchmark baseline.

**Direction convention (load-bearing).** Edges are `listener → source`: the
listener receives the source's prediction. This matches
`establish_lateral_connection(from = listener, to = source)` and
`lateral_context_for(listener)`. Field names are `listener_id` / `source_id`
(not `from` / `to`) precisely to prevent an analysis tool from comparing a
column's output against the wrong end of an edge (a class of bug caught in the
A2 judge review).

**Determinism.** `SmallWorld` is reproducible given its `seed` (zero-
dependency splitmix64 PRNG), so benchmark runs over different topologies are
comparable without confounding randomness. Weighted/attention routing (§9.1)
and semantic-embedding convergence (§9.3) remain deferred/blocked.

## Security posture (Phase 2 review)

`ptg` is a single-user local dev CLI talking to a **trusted** local inference
server. The red-team review found **zero critical** issues in that posture.
Documented, accepted limitations (not Phase 2 blockers — tracked if the CLI is
ever exposed to untrusted input):

- **SSRF / data exfil via arbitrary `--vllm-url`** (reqwest follows redirects).
  Acceptable for a trusted local server; revisit before exposing the CLI as a
  service.
- **Unbounded response parsing / payload sizes** — a malicious *server* could
  return huge JSON. Trusted-server assumption.
- **Model/server output printed verbatim** — could in principle inject terminal
  escapes. Low risk for local use.

## Running it

```bash
cargo check --workspace --all-targets
cargo clippy --all-features --all-targets -- -D warnings
cargo test                                      # unit tests (live test is #[ignore])
cargo run -p ptg-cli -- --help
cargo run -p ptg-cli -- --dry-run               # offline wiring check
cargo run -p ptg-cli -- --probe --vllm-url http://127.0.0.1:18135 --model gemma-4-e4b
                                                 # reachability + model listing
cargo run -p ptg-cli -- --vllm-url http://127.0.0.1:18135 --model gemma-4-e4b --ticks 2
                                                 # live text epoch (needs a server)
cargo run -p ptg-cli -- --image-url https://example.com/img.png --input "describe"  # multimodal
```

Validated live against a local `llama.cpp` `llama-server` (`gemma-4-e4b`):
2-tick text epoch converged in 1 tick, mean confidence 0.94, all four columns
passed strict per-sphere schema validation. Multimodal request serialization
is unit-tested but not live-validated (the running server is text-only).
