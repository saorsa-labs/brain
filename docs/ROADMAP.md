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

## Benchmark — grounding the thesis: does the lateral mechanism help?

Ground the thesis: is the cortical *mechanism* better than a single-monolithic
context (and a compute-matched ensemble) for latency, token economy, and answer
quality? Methodology is in [`docs/BENCHMARKING.md`](./BENCHMARKING.md).

### Done

- [x] Methodology doc + fair-baseline design (C1–C4 confounds neutralized).
- [x] Engine instrumentation (per-call `usage` incl. `cached_tokens`, `finish_reason`).
- [x] `ptg-bench` harness: all conditions + JSONL records + Markdown summary;
      `--routing-policy`, per-tick route observability, `--topology/--columns`,
      `--column-pack`, `--lateral-mode`.
- [x] `ptg-judge`: A2 + A3 equal-call control, route-aware echo screen,
      LLM-corroborating judge, topology-aware + per-column determinism gate,
      `--max-pairs`/`--sample-seed` scale sampling.

### Evidence arc (negative → positive)

- [x] **Pilot (e2b, raw lateral): quality-neutral.** 5×3×4. Mesh costs 3–4.5×
      compute; cache-hit collapses under lateral exchange (43.5% vs 98%+).
      [`PILOT_FINDINGS.md`](./PILOT_FINDINGS.md).
- [x] **A2 judge (e2b, raw): coin flip.** Lateral activates (edit dist 0.61–0.74)
      but does not improve quality (tick_1 14 vs tick_2 13); 25% echo leakage.
      [`JUDGE_FINDINGS.md`](./JUDGE_FINDINGS.md).
- [x] **A3 equal-call control (e2b, raw): lateral does NOT beat the no-lateral
      second look.** 11 vs 12. [`A3_FINDINGS.md`](./A3_FINDINGS.md).
- [x] **Scale A3 (e2b, raw, 150 cols): strongly negative.** lateral 3 vs 18,
      57% echo leakage, cache collapse. [`SCALE_SMOKE_FINDINGS.md`](./SCALE_SMOKE_FINDINGS.md).
      → Conclusion at this point: *raw* lateral-text exchange is quality-neutral
        to negative; do not scale it.
- [x] **Structured lateral exchange** (pre-registered disambiguator, team
      consensus): change the *medium* (bounded claim-excerpt + synthesis
      directive instead of full verbatim prediction), not the schema.
      [`STRUCTURED_LATERAL_EXPERIMENT.md`](./STRUCTURED_LATERAL_EXPERIMENT.md).
- [x] **Structured on e2b: FAIL (e2b)** — lateral 33%, but noise-level
      (p≈0.29) and the intervention bundled confounds; could not conclude
      capacity from e2b alone. [`STRUCTURED_LATERAL_FINDINGS.md`](./STRUCTURED_LATERAL_FINDINGS.md).
- [x] **Structured on e4b: RESURRECTION.** lateral 78.4% (29/37) vs second_look 8.
      Length-control validated (win-mapping reproduces the report exactly; lateral
      wins 64% even when its draft is *shorter*). [`LENGTH_CONTROL_ANALYSIS.md`](./LENGTH_CONTROL_ANALYSIS.md).
- [x] **Structured on e4b at 50 cols: GENERALIZES.** lateral 84.4%, echo 6%
      (clean), survives length control. [`STRUCTURED_LATERAL_E4B_50COL.md`](./STRUCTURED_LATERAL_E4B_50COL.md).
      → The 4-col effect is not an artifact; it holds at 12.5× fan-out.

### Current direction (revised)

The **structured lateral exchange + e4b-class model** path is viable and
quality-positive. The thesis is not dead; the binding constraint was *model
capacity + the raw-text medium*, not the concept. The path forward is
structured exchange + larger models + moderate scale — **not** `ptg-belief` (yet).

### Open / next

- [ ] **Powered 50-col run** (5 prompts × 3 repeats, structured, e4b): confirm
      the 50-col result is stable, not a single-prompt draw. Decision rule:
      lateral win-rate CI lower bound > 55% AND echo < 10%.
- [ ] **150-col on e4b: infra-blocked** (server dies ~85% through the mesh arm,
      3 attempts, 2 topologies). [`STRUCTURED_LATERAL_E4B_SCALE_BLOCKED.md`](./STRUCTURED_LATERAL_E4B_SCALE_BLOCKED.md).
      Needs server-capacity fixes (KV-cache budget, slot tuning) before re-attempt.
- [ ] **A4 explicit self-revision control** ("reconsider your answer"): the A3
      no-lateral second-look at temp 0 is an inert replay, not genuine revision.
      Needed to separate lateral exchange from self-revision.
- [ ] Optional: red-team ablation (structured-without-directive /
      raw-with-truncation) to isolate which element of the bundled intervention
      matters. Lower priority now that the bundle works on e4b.

## Phase 3 — Topologies, routing, convergence, & structured exchange ✅

- [x] Pluggable topologies: ring, torus, small-world (§3.1.3). `TopologySpec`
      in `ptg-core`, `mesh_with_topology` in `ptg-runtime`; `--topology/--columns`
      in `ptg` **and** `ptg-bench` (shared `ptg_cli::topology_cli` resolver).
- [x] Early-stop **prediction-stability** convergence: `min_prediction_similarity`
      (token-Jaccard), `ConvergenceReason`, `MeshResult.convergence_reason`.
- [x] Weighted / attention-routed lateral connections (§9.1): `RoutingPolicy`
      (`All` / `ConfidenceTopK{k}` / `DiversityPreserving{k}`) in `ptg-runtime`;
      `--routing-policy` + `--routing-k` in CLI + bench; per-tick observability.
- [x] **Structured lateral exchange** (`LateralContextMode::{Raw, Structured}`):
      bounded char-safe claim-excerpt + synthesis directive instead of verbatim
      neighbor prediction. `--lateral-mode raw|structured` in bench. The change
      that took the lateral mechanism from quality-neutral (raw) to
      quality-positive (structured, on e4b). Tests + pre-registration landed.
- [ ] Full **semantic** cosine convergence over prediction embeddings (§9.3) —
      blocked: the live server returns HTTP 501 on `/v1/embeddings`. The cheap
      token-Jaccard proxy above is the unblocked approximation.

## Phase 4 — Scale, infrastructure, & the next frontier

- [ ] **Powered 50-col structured-e4b confirmation** (running): 5 prompts × 3
      repeats. The first statistically interpretable scale result.
- [ ] **e4b server capacity fixes** to unlock 150 cols: the 4B-model server dies
      ~85% through sustained mesh runs (3 reproductions, 2 topologies). KV-cache
      budget / slot / keepalive tuning before re-attempting 150-col.
- [ ] Workstation tuning on unified-memory hardware (§7): ring-buffer memory
      budget audit, HTTP/2 multiplexing under load.
- [ ] Per-round tracing of predictions and confidence for debugging emergence.
- [ ] `dashmap`-backed column store for high-contention scale (dependency to
      be added when used).

### Future: the belief/evidence frontier (`ptg-belief`)

If structured lateral exchange holds up at scale, the natural next architecture
is to replace self-reported confidence + truncated claim-excerpts with typed
hypotheses, evidence, provenance, posterior state, and dependence-aware
aggregation. **Deferred** — the current structured-text mechanism is working and
should be confirmed before adding a belief layer.
