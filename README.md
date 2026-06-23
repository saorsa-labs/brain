# Project Thousand-Gemma (PTG)

> A distributed, prompt-based **cortical mesh simulator** written in Rust — implementing the organizational principles of Jeff Hawkins' *Thousand Brains Theory of Intelligence* as a multi-agent system.

[![status](https://img.shields.io/badge/status-v1%20execution%20skeleton-brightgreen)](docs/ROADMAP.md)
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
│   ├── ptg-core        # CorticalColumn, DomainSphere, ColumnOutputSchema, HistoryBuffer, PROMPT_*
│   ├── ptg-vllm        # Shared inference engine ("thalamus"): ColumnEngine trait + reqwest InferenceEngine
│   ├── ptg-consensus   # Convergence math (mean/delta/cosine over confidence vectors, ndarray)
│   ├── ptg-runtime     # CorticalMesh: 3-phase epoch loop (fan-out + lateral injection + integration)
│   └── ptg-cli         # `ptg` binary (clap + tracing, --dry-run)
└── docs/               # Specification, architecture, roadmap
```

## Getting started

```bash
cargo check --workspace                        # type-check all crates
cargo fmt --all                                # format
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p ptg-cli
```

## Status

The **V1 execution skeleton** is implemented and panic-free: a compiling Rust
workspace encoding the domain model, a shared `InferenceEngine` client, the
three-phase epoch loop with lateral context injection, metric-based convergence,
and a `ptg` CLI (with `--dry-run` for offline validation). It is unit-tested
with a mock engine — no live vLLM server is required to build or test. Running
real inference needs a local vLLM server (see [§7.1](docs/SPECIFICATION.md)).

Next phases — weighted/attention routing, full semantic convergence, true
multimodality, and benchmarks — are tracked in the [roadmap](docs/ROADMAP.md).

## Documentation

- [Specification](docs/SPECIFICATION.md) — full architectural blueprint (source of truth)
- [Architecture](docs/ARCHITECTURE.md) — crate-level design and data flow
- [Roadmap](docs/ROADMAP.md) — implementation phases

## License

Dual-licensed under **MIT** or **Apache-2.0**, at your option ([`LICENSE-MIT`](LICENSE-MIT), [`LICENSE-APACHE`](LICENSE-APACHE)). © 2026 Saorsa Labs.
