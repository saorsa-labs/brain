# Project Thousand-Gemma (PTG)

> A distributed, prompt-based **cortical mesh simulator** written in Rust — implementing the organizational principles of Jeff Hawkins' *Thousand Brains Theory of Intelligence* as a multi-agent system.

[![status](https://img.shields.io/badge/status-architecture%20scaffold-orange)](docs/ROADMAP.md)
[![license](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue)](#license)
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
│   ├── ptg-core        # Domain types: columns, reference frames, topology, outputs
│   ├── ptg-vllm        # Shared local inference engine ("thalamus") client
│   ├── ptg-consensus   # Multi-round voting, convergence criteria
│   ├── ptg-runtime     # Mesh orchestration: fan-out + lateral injection + consensus
│   └── ptg-cli         # Command-line entry point
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

This repository currently contains the **architecture scaffold** for PTG: a compiling Rust workspace encoding the core domain model and crate boundaries defined in the specification. Implementation of the inference fan-out, lateral token injection, and consensus engine is tracked in the [roadmap](docs/ROADMAP.md).

## Documentation

- [Specification](docs/SPECIFICATION.md) — full architectural blueprint (source of truth)
- [Architecture](docs/ARCHITECTURE.md) — crate-level design and data flow
- [Roadmap](docs/ROADMAP.md) — implementation phases

## License

Dual **AGPL-3.0 / Commercial** © Saorsa Labs. (`AGPL-3.0-or-later` SPDX in manifests; alternative commercial licensing available on request — david@saorsalabs.com)
