# Project Thousand-Gemma (PTG)

> A distributed, prompt-based **cortical mesh simulator** written in Rust — implementing the organizational principles of Jeff Hawkins' *Thousand Brains Theory of Intelligence* as a multi-agent system.

[![status](https://img.shields.io/badge/status-phase%202%20done-brightgreen)](docs/ROADMAP.md)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)
[![rust](https://img.shields.io/badge/rust-1.85%2B-orange)](#getting-started)

## Overview

**Project Thousand-Gemma (PTG)** is an open-source, high-performance computing framework written in Rust that emulates the core organizational tenets of Jeff Hawkins' **Thousand Brains Theory of Intelligence**. Rather than relying on a single, massive, monolithic LLM context window to ingest, process, and synthesize complex multi-modal data, PTG decomposes problems across hundreds or thousands of specialized, structurally localized, virtual **cortical columns**.

Each virtual cortical column is instantiated as an independent, asynchronous processing unit bound to a single local, highly optimized LLM inference engine (`Gemma-4-2B-Multimodal`). The architecture enforces strict domain-specific cognitive, empirical, and sensory prisms via hyper-targeted system prompts and localized spatial **reference frames**. Instead of centralized top-down governance, global intelligence and semantic stability emerge bottom-up through a decentralized, multi-round **consensus** mechanism powered by lateral token-passing, neighborhood weight-voting, and structural context injection.

By leveraging unified memory architectures on modern high-end developer workstations, PTG aims to run massive-scale modular cognitive networks without distributed clusters or supercomputing infrastructure.

> 📐 **Source of truth:** the full architectural blueprint lives in [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md).

## Why not one big model?

| Failure mode of monolithic LLMs | PTG's neocortical answer |
| --- | --- |
| **Context dilution** — attention degrades and "lost-in-the-middle" at long contexts | Each column holds a small, focused context bound to one reference frame |
| **Brittle generalization** — catastrophic interference across domains | Columns are domain-specialized; consensus resolves ambiguity laterally |
| **Explosive compute** — quadratic attention $O(C^2)$ on one giant context | Many small contexts share one engine via prefix caching |

## Biological → software mapping

| Biological concept (Hawkins) | Software realization (PTG) |
| --- | --- |
| Cortical column | An isolated `CorticalColumn` instance with a targeted system prompt |
| Sensory input / afferent pathway | Parallel fan-out of stimulus to a shared inference engine |
| Reference frames / coordinates | Forced structural JSON bounding a column's output space |
| Lateral connections | Topology-constrained token injection from neighbor outputs |
| Inter-column voting | Multi-round asynchronous consensus with metric-based convergence |
| The thalamus | A single shared local inference engine with prefix caching |

## Repository layout

```
brain/
├── crates/
│   ├── ptg-core        # CorticalColumn, ColumnOutputSchema (validate_for_sphere), Stimulus/multimodal, PROMPT_*
│   ├── ptg-vllm        # Shared inference engine ("thalamus"): ColumnEngine trait + reqwest InferenceEngine, list_models
│   ├── ptg-consensus   # Convergence math (mean/delta/cosine over confidence vectors, ndarray)
│   ├── ptg-runtime     # CorticalMesh: 3-phase epoch loop (fan-out + lateral injection + integration)
│   └── ptg-cli         # `ptg` binary (--image-url, --image-detail, --probe, --dry-run)
└── docs/               # Specification, architecture, roadmap
```

## Getting started

> **Start here:** follow [`docs/TUTORIAL.md`](docs/TUTORIAL.md) to start the
> verified Gemma 4 QAT server, run your first cortical mesh, and edit column
> packs for abstraction-level experiments.

### Model setup (3 tiers)

| Tier | Model | Memory | Notes |
|------|-------|--------|-------|
| **Default** | `unsloth/gemma-4-E2B-it-qat-GGUF` (QAT) | ~2.7 GB | 3× less memory, drop-in GGUF. Start here. |
| Fallback | `ggml-org/gemma-4-E2B-it-GGUF:Q4_K_M` | ~3.5 GB | Balanced default if QAT unavailable. |
| Scaling | TurboQuant KV-cache ([fork](https://github.com/TheTom/llama-cpp-turboquant)) | 6× KV cache | Past the memory wall; not drop-in. |

```bash
scripts/start-gemma4-qat.sh          # start the QAT model server (port 18136)
cargo run -p ptg-cli --bin ptg -- --probe \
    --vllm-url http://127.0.0.1:18136 --model gemma-4-e2b-qat

cargo run -p ptg-cli --bin ptg -- \   # your first mesh run
    --vllm-url http://127.0.0.1:18136 --model gemma-4-e2b-qat \
    --topology ring --columns 4 --min-ticks 2 --max-tokens 1024 --temperature 0
```

### Column packs (abstraction-level experiments)

Swap column system prompts via a TOML pack to test how a column's abstraction
level (high-level physics vs mid-level shapes vs low-level sequences) changes
what the mesh converges to:

```bash
# 9 columns at 3 abstraction levels on a 3×3 torus
cargo run -p ptg-cli --bin ptg -- --dry-run \
    --column-pack examples/column-packs/abstraction-ladder-9.toml \
    --topology torus --torus-width 3 --torus-height 3 --columns 9
```

See [`examples/column-packs/`](examples/column-packs/) and the
[tutorial](docs/TUTORIAL.md) §8–9 for experiment recipes.

### Quick checks

```bash
cargo check --workspace                        # type-check all crates
cargo fmt --all                                # format
cargo clippy --workspace --all-targets -- -D warnings
cargo test                                     # 78 tests
```

### Pluggable topologies (Phase 3)

The `--topology` flag selects the lateral mesh layout over `--columns`
replicated domain spheres. `--dry-run` prints the full wiring (columns +
listener->source edges) without any inference:

```bash
# Named 4-column reference graph (default, unchanged)
cargo run -p ptg-cli --bin ptg -- --dry-run

# Directed ring over 8 columns
cargo run -p ptg-cli --bin ptg -- --dry-run --topology ring --columns 8

# 3x3 torus (9 columns, 4 neighbors each)
cargo run -p ptg-cli --bin ptg -- --dry-run --topology torus --torus-width 3 --torus-height 3

# Seeded small-world (deterministic given --small-world-seed)
cargo run -p ptg-cli --bin ptg -- --dry-run --topology small-world \
    --columns 20 --small-world-degree 4 --small-world-rewire 0.2
```

Degeneracy guardrails reject parameters where distinct topologies collapse to
the same graph (e.g. `ring-bi` with < 4 columns == `fully-connected`; `small-
world` with `degree*2 >= columns` silently under-rewires).

## Early research signal: lateral consensus can homogenize frames

> ⚠️ **This is a pilot observation from a single live QAT run (9-column torus,
> `abstraction-ladder-9.toml`), not a benchmarked result.** Treat it as a
> research direction worth investigating, not a confirmed claim. Full
> methodology and confound analysis live in [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md).

Three end-to-end runs against the live Gemma 4 QAT server surfaced a shared,
unexpected theme: **lateral consensus is a homogenizing force.** The mesh tends
to converge toward the dominant interpretation rather than preserving
minority or niche frames.

| Observation | What happened | Status |
|-------------|---------------|--------|
| **Confidence stratifies by abstraction level** | On a causal prompt, high-level (whole-system) columns reported mean confidence ~0.92 while low-level (token/sequence) columns reported ~0.68 — and produced more coherent causal narratives. | Confirmed on this run |
| **Low-level drift, but no divergence** | On deliberately ambiguous token-sequence input, low-level columns latched onto literal token prediction ("the next token is likely…"), but lateral exchange pulled them back toward the high-level physics framing instead of letting the mesh fragment. | Nuanced |
| **Topology changes propagation speed** | The niche "context" column's framing propagated rapidly across a 4-neighbor torus (0.98 conf, system-failure language adopted by neighbors) but stayed isolated in its own frame on a 1-neighbor ring (0.85 conf). | Confirmed on this run |

### Open research questions

This signal points at a central tension in the Thousand-Brains model that is
worth digging into:

- **When does lateral consensus improve perception vs. erase useful minority
  frames?** Homogenization is great when the dominant frame is correct; it is a
  failure mode when the dissenting/niche view is the one that matters.
- **How do topology, confidence thresholds, and `--min-ticks` modulate
  homogenization?** Ring vs torus already shows a large effect; degree and
  rewiring probability are untested.
- **Can weighted/attention-based routing (§9.1, deferred) preserve dissenting
  useful frames** instead of majority-voting them away?
- **Is the confidence stratification by abstraction level a calibration artifact**
  (high-level columns may simply self-report higher confidence) or a real
  cognitive signal? The judge harness (`ptg-judge`) is designed to separate
  these, but the scaled run (A3) has not been executed.

See the [roadmap](docs/ROADMAP.md) for the planned A3 scaled benchmark that
would turn this pilot signal into evidence.

> **First mitigation shipped (Phase 3A):** the convergence loop now also
> supports a model-independent **prediction-stability** signal
> (`--min-prediction-similarity`, token-Jaccard of successive predictions) that
> does not rely on the self-reported confidence a model can game — the same
> `tick_outputs` / `convergence_reason` plumbing gives us the within-run
> measurement needed to study homogenization directly.

## Status

Phases 0–2 are complete and panic-free. The workspace implements the domain
model, a shared `InferenceEngine` client, the three-phase epoch loop with
lateral context injection and metric-based convergence, and a `ptg` CLI.

**Phase 2** added: a `Stimulus` model (text + multimodal image/audio serializing
to the OpenAI content-array shapes), per-sphere reference-frame schema
validation, confidence-aware global integration (`accepted`/`rejected_outputs`),
and a live-inference harness (`--probe` + an `#[ignore]` integration test). It
was **validated end-to-end against a live `llama.cpp` server** (`gemma-4-e4b`):
a 2-tick text epoch converged in 1 tick at mean confidence 0.94 with all four
columns passing strict per-sphere validation. 32 unit tests; clippy-clean
(`-D warnings`).

Next phases — weighted/attention routing, full semantic convergence, true
multimodal *live* validation, and benchmarks — are tracked in the
[roadmap](docs/ROADMAP.md).

## Documentation

- [Tutorial](docs/TUTORIAL.md) — **start here**: server setup, first run, column packs, experiment recipes
- [Specification](docs/SPECIFICATION.md) — full architectural blueprint (source of truth)
- [Architecture](docs/ARCHITECTURE.md) — crate-level design and data flow
- [Roadmap](docs/ROADMAP.md) — implementation phases

## License

Dual-licensed under **MIT** or **Apache-2.0**, at your option ([`LICENSE-MIT`](LICENSE-MIT), [`LICENSE-APACHE`](LICENSE-APACHE)). © 2026 Saorsa Labs.
