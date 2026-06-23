# Project Specification & Architectural Blueprint
## Project Thousand-Gemma (PTG): A Distributed, Prompt-Based Cortical Mesh Simulator
**A Multi-Agent System Framework Implementing Jeff Hawkins’ Thousand Brains Theory in Rust**

---

## 1. Executive Summary

**Project Thousand-Gemma (PTG)** is an open-source, high-performance computing framework written in Rust designed to emulate the core organizational tenets of Jeff Hawkins’ **Thousand Brains Theory of Intelligence**. Rather than relying on a single, massive, monolithic Large Language Model (LLM) context window to ingest, process, and synthesize complex multi-modal data, PTG decomposes problems across hundreds or thousands of specialized, structurally localized, virtual "Cortical Columns."

Each virtual Cortical Column is instantiated as an independent, asynchronous processing unit bound to a single local, highly optimized LLM inference engine (`Gemma-4-2B-Multimodal`). This architecture enforces strict domain-specific cognitive, empirical, and sensory prisms via hyper-targeted system prompts and localized spatial reference frames. Instead of centralized top-down governance, global intelligence and semantic stability emerge bottom-up through a decentralized, multi-round consensus mechanism powered by lateral token-passing, neighborhood weight-voting, and structural context injection. 

By taking advantage of unified memory architectures found in modern high-end developer workstations, PTG presents an enterprise-ready engineering specification to test, validate, and run massive-scale modular cognitive networks without the need for distributed server clusters or supercomputing infrastructure.

---

## 2. Structural & Philosophical Foundations

### 2.1 The Monolithic Bottleneck of Modern AI
Conventional Deep Learning frameworks rely extensively on increasing parameter scale ($N$) and expanding context windows ($C$) to capture complex world behaviors. This paradigm suffers from critical failure modes:
1. **Context Dilution:** As context lengths grow, models suffer from performance degradation, attention distraction, and "lost-in-the-middle" anomalies where crucial operational parameters are ignored.
2. **Brittle Generalization:** A single network trying to learn physics, coding, psychology, and logic simultaneously suffers from catastrophic interference and representation blurring. It handles nuanced multidisciplinary issues by smoothing over edge cases.
3. **Explosive Compute Costs:** Quadratic attention mechanisms ($O(C^2)$) ensure that expanding context windows becomes prohibitively expensive, leading to memory thrashing even on large-scale infrastructure.

### 2.2 The Hawkins Alternative: Neocortical Modularity
In *A Thousand Brains: A New Theory of Intelligence*, Jeff Hawkins demonstrates that the human neocortex is not a single generalized processing machine. Instead, it is composed of roughly 150,000 highly repetitive, functionally independent structural units called **cortical columns**.
* **Independent Modeling:** Each column learns a complete, self-contained model of a portion of the world. It maps sensory inputs to specific **reference frames** (localized coordinate maps) to predict changes and identify objects.
* **The Voting Paradigm:** There is no master coordinator in the human brain. Instead, columns possess extensive lateral (horizontal) connections. They transmit their internal predictions and confidence scores to adjacent columns. Through continuous, multi-round voting, a stable, global perception emerges.
* **Robust Multi-Modality:** A visual column, a tactile column, and an abstract mathematical column view the same external phenomenon from different angles, resolving ambiguity via lateral consensus rather than central merging.

### 2.3 The PTG Solution
Project Thousand-Gemma translates these exact biological principles into a software engineering architecture:

| Biological Concept (Hawkins) | Software Architecture Realization (PTG) |
| :--- | :--- |
| **Cortical Column** | An isolated `CorticalColumn` instance with a targeted system prompt. |
| **Sensory Input / Afferent Pathway** | Parallel HTTP fan-out of raw data strings to a shared vLLM engine. |
| **Reference Frames / Coordinates** | Forced structural JSON formatting enforcing spatial/conceptual bounds. |
| **Lateral Connections** | Topology-constrained token injection from neighbor outputs into current context. |
| **Inter-Column Voting** | Multi-round asynchronous consensus loop with metric-based convergence criteria. |
| **The Thalamus** | A local `vLLM` server handling prefix caching and memory multiplexing. |

---

## 3. Core Architecture & System Topology

The system topology minimizes its physical footprint by sharing foundational model weights across all virtual columns. Instead of instantiating 1,000 independent model instances in VRAM, PTG uses a single local **vLLM engine instance**, dynamically shifting system prompts, prefix caches, and attention masks to simulate a massive network of concurrent cortical columns.

---

> **Note:** This specification was imported at repository initialization. Sections beyond §3 were not included in the source document and are to be elaborated during implementation. The crate-level realization of the above is described in [`ARCHITECTURE.md`](./ARCHITECTURE.md), and implementation work is tracked in [`ROADMAP.md`](./ROADMAP.md).
