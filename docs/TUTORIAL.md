# PTG Tutorial — From Zero to First Experiment

> **Start here.** This walkthrough takes you from a fresh clone to running a
> cortical mesh and experimenting with column abstraction levels. Every command
> is copy-paste.

---

## 1. What is PTG?

Project Thousand-Gemma (PTG) is a **cortical mesh simulator**: instead of one
big LLM context, it fans a stimulus out to many small, specialized **cortical
columns**, each bound to a single reference frame via a hyper-targeted system
prompt. Columns exchange predictions laterally (neighbor → listener) over a
topology, re-evaluate for several ticks, and converge. Global intelligence is
meant to *emerge* from this decentralized voting — the core thesis of Jeff
Hawkins' *Thousand Brains Theory*.

The central experiment axis is **abstraction level**: a column can reason about
whole-system causality (high-level), geometry and coordinates (mid-level), or
raw signals and token sequences (low-level) — and its level is set *entirely*
by its system prompt. The `sphere` field only selects the JSON validation
envelope; the abstraction level lives in the prompt text.

---

## 2. Prerequisites

- **Rust 1.85+** (`rustup default stable`)
- **A built `llama-server`** (from [llama.cpp](https://github.com/ggml-org/llama.cpp)):
  ```bash
  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
  cmake -B build && cmake --build build --config Release
  # The binary is at build/bin/llama-server
  ```
- **A Hugging Face account** with the Gemma license accepted (Gemma is gated).
  Go to https://huggingface.co/google/gemma-4-E2B-it, accept the license, then:
  ```bash
  pip install huggingface-hub && hf login   # paste your HF token
  ```

No `cargo install` is needed — everything runs via `cargo run -p ptg-cli`, or
via the prebuilt `ptg` binary from the
[releases](https://github.com/saorsa-labs/brain/releases).

> **Faster path:** once the prerequisites above are met, `ptg setup --yes`
> then `ptg serve` does steps 3–4 for you (model download + server launch) and
> writes a config so later `ptg` commands need no flags. The manual steps
> below remain for reference and for choosing non-default model tiers.

---

## 3. Model setup (three tiers)

| Tier | Model | When | Why |
|------|-------|------|-----|
| **Default** | `unsloth/gemma-4-E2B-it-qat-GGUF` (file: `gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf`) | Always start here | Google's QAT variant: **3× less memory** (~2.7 GB), drop-in GGUF, near-original accuracy. Verified working end-to-end with PTG. |
| **Fallback** | `ggml-org/gemma-4-E2B-it-GGUF:Q4_K_M` | If QAT download fails | The documented balanced default. Slightly larger, same serving path. |
| **Scaling path** | TurboQuant KV-cache via [`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant) | When pushing column count / context length | 6× KV-cache compression at zero accuracy loss (Google ICLR 2026). **Not drop-in** — requires building the fork. This is the path past the memory wall. |

> **Why not the "QAT Mobile Format"?** Google ships Gemma 4 QAT in a Mobile
> Format (`.task`/LiteRT for on-device Android). That is the *wrong format* for
> a server pipeline — we serve via the OpenAI-compatible `llama-server`, which
> needs GGUF. The QAT GGUF checkpoints above give you the QAT memory savings in
> the right format.

---

## 4. Start the server

**Recommended (CLI setup phase):**

```bash
ptg setup --yes   # detect server, download model, write config
ptg serve         # foreground; leave running
```

`ptg setup` detects your `llama-server` binary, downloads the QAT model if
needed (first run: ~2.7 GB), and writes `~/.config/ptg/config.toml`. `ptg
serve` launches the server on `http://127.0.0.1:18136`. Leave it running in one
terminal. (`ptg setup`/`ptg serve` accept `--dry-run` to preview.)

**Legacy (bash script, same effect):**

```bash
scripts/start-gemma4-qat.sh
```

This detects your `llama-server` binary, downloads the QAT model if needed
(first run: ~2.6 GB), and starts it on `http://127.0.0.1:18136`. Leave it
running in one terminal.

In another terminal, probe it:

```bash
ptg --probe        # if you ran `ptg setup` — config is remembered
# or explicitly:
cargo run -p ptg-cli --bin ptg -- \
    --probe \
    --vllm-url http://127.0.0.1:18136 \
    --model gemma-4-e2b-qat
```

You should see `reachable: 1 model(s)`.

---

## 5. First dry-run (no inference)

Validate the mesh wiring without touching the server:

```bash
cargo run -p ptg-cli --bin ptg -- \
    --dry-run \
    --topology ring --columns 4
```

This prints every column, its sphere, and the listener→source edge list.

---

## 6. First live run

```bash
cargo run -p ptg-cli --bin ptg -- \
    --vllm-url http://127.0.0.1:18136 \
    --model gemma-4-e2b-qat \
    --topology ring --columns 4 \
    --min-ticks 2 --ticks 3 \
    --max-tokens 1024 --temperature 0 \
    --input "A 2kg block slides down a frictionless ramp angled 30 degrees. Predict the acceleration."
```

Key flags:
- `--min-ticks 2` — forces lateral exchange to actually run (columns are
  overconfident on tick 1 and would converge immediately otherwise).
- `--max-tokens 1024` — prevents JSON truncation when neighbor context
  lengthens the prompt.
- `--temperature 0` — deterministic output for reproducible experiments.
- `--min-prediction-similarity 0.85` — (optional) stop the loop once
  predictions stop changing in *word overlap*, a model-independent signal that
  doesn't rely on the self-reported confidence a model can game. The CLI prints
  which criterion stopped the epoch, e.g. `convergence: prediction
  token-similarity stabilized`.

```bash
cargo run -p ptg-cli --bin ptg -- \
    --vllm-url http://127.0.0.1:18136 --model gemma-4-e2b-qat \
    --topology ring --columns 4 \
    --min-ticks 2 --ticks 4 --max-tokens 1024 --temperature 0 \
    --min-prediction-similarity 0.85 \
    --input "Describe the forces on a falling object."
```

---

## 7. Topologies

```bash
# 8-column directed ring
cargo run -p ptg-cli --bin ptg -- --dry-run --topology ring --columns 8

# 3×3 torus (9 columns, 4 neighbors each)
cargo run -p ptg-cli --bin ptg -- --dry-run --topology torus --torus-width 3 --torus-height 3

# Seeded small-world (deterministic given --small-world-seed)
cargo run -p ptg-cli --bin ptg -- --dry-run --topology small-world \
    --columns 20 --small-world-degree 4 --small-world-rewire 0.2
```

### Routing policies (lateral attention)

By default every column hears from **all** its neighbors (`--routing-policy
all`). That can homogenize the mesh: high-confidence / high-level frames tend to
overwrite niche ones. Two alternatives let a column listen selectively:

```bash
# Hear only the 2 highest-confidence neighbors
--routing-policy confidence-top-k --routing-k 2

# MMR-style diversity-preserving: keep up to 2 dissimilar neighbors
# (the hypothesized mitigation for lateral homogenization)
--routing-policy diversity --routing-k 2
```

Diversity routing anchors on the highest-confidence source, then greedily adds
sources that are most *different* (by token overlap) from what it already heard
— so dissident/niche frames survive. Every routing decision is captured per-tick
in `tick_outputs.routes` (`route_weight` + `confidence` per source), so you can
measure how much each column attended to each neighbor.

---

## 8. Column packs and the abstraction-level experiment

A **column pack** is a TOML file defining explicit columns with custom system
prompts. This is how you set a column's abstraction level. See
[`examples/column-packs/abstraction-ladder-9.toml`](../examples/column-packs/abstraction-ladder-9.toml)
— nine columns at three abstraction levels (high/mid/low) over a 3×3 torus.

Run it:

```bash
cargo run -p ptg-cli --bin ptg -- \
    --dry-run \
    --column-pack examples/column-packs/abstraction-ladder-9.toml \
    --topology torus --torus-width 3 --torus-height 3 --columns 9
```

Note the `level=N` tags in the output — those are the abstraction-level hints
from the pack, surfaced so you can attribute results to level.

---

## 9. Experiment recipes (hypotheses)

### H1 — Higher-level columns produce more confident consensus on causal prompts

```bash
cargo run -p ptg-cli --bin ptg -- \
    --vllm-url http://127.0.0.1:18136 --model gemma-4-e2b-qat \
    --column-pack examples/column-packs/abstraction-ladder-9.toml \
    --topology torus --torus-width 3 --torus-height 3 --columns 9 \
    --min-ticks 2 --ticks 3 --max-tokens 2048 --temperature 0 \
    --input "A satellite in decaying orbit shows rising thermal load, decreasing altitude, and intermittent guidance resets. What will happen next?"
```

**Prediction:** the `level=3` (high-level) columns should report higher
confidence and more coherent causal predictions than `level=1` columns.

### H2 — Low-level columns drift on ambiguous token/sequence input

```bash
cargo run -p ptg-cli --bin ptg -- \
    --vllm-url http://127.0.0.1:18136 --model gemma-4-e2b-qat \
    --column-pack examples/column-packs/abstraction-ladder-9.toml \
    --topology torus --torus-width 3 --torus-height 3 --columns 9 \
    --min-ticks 2 --ticks 4 --max-tokens 2048 --temperature 0 \
    --input "spring bank charge vector patch trace shift loop fall phase"
```

**Prediction:** `level=1` columns latch onto the token-sequence aspect and
predict a continuation; `level=3` columns struggle to find causal meaning and
report lower confidence.

### H3 — Ring vs torus changes how a niche column's view propagates

Run the same input on a ring, then a torus:

```bash
# Ring (each column hears only 1 neighbor)
cargo run -p ptg-cli --bin ptg -- \
    --vllm-url http://127.0.0.1:18136 --model gemma-4-e2b-qat \
    --column-pack examples/column-packs/abstraction-ladder-9.toml \
    --topology ring --columns 9 \
    --min-ticks 2 --ticks 3 --max-tokens 2048 --temperature 0 \
    --input "An operator dismisses an automation fault as a harmless glitch while telemetry shows a real kinetic-energy anomaly."

# Torus (each column hears 4 neighbors)
cargo run -p ptg-cli --bin ptg -- \
    --vllm-url http://127.0.0.1:18136 --model gemma-4-e2b-qat \
    --column-pack examples/column-packs/abstraction-ladder-9.toml \
    --topology torus --torus-width 3 --torus-height 3 --columns 9 \
    --min-ticks 2 --ticks 3 --max-tokens 2048 --temperature 0 \
    --input "An operator dismisses an automation fault as a harmless glitch while telemetry shows a real kinetic-energy anomaly."
```

**Prediction:** on the torus, the context column's ("this is operator bias")
view reaches all columns in one hop; on the ring it propagates slowly along the
chain. Compare the final-tick predictions.

---

## 10. De-risk ladder (if the 9-column run fails)

The mesh is **fail-fast**: one column emitting truncated or malformed JSON
aborts the entire epoch (no partial output). If a 9-column run fails, de-risk:

```bash
# Step 1: 4 columns, generous tokens, fully-connected (simplest)
cargo run -p ptg-cli --bin ptg -- \
    --vllm-url http://127.0.0.1:18136 --model gemma-4-e2b-qat \
    --topology fully-connected --columns 4 \
    --min-ticks 2 --ticks 3 --max-tokens 2048 --temperature 0 \
    --input "Describe the forces on a falling object."

# Step 2: same, but a ring (adds lateral directionality)
cargo run -p ptg-cli --bin ptg -- \
    --vllm-url http://127.0.0.1:18136 --model gemma-4-e2b-qat \
    --topology ring --columns 4 \
    --min-ticks 2 --ticks 3 --max-tokens 2048 --temperature 0 \
    --input "Describe the forces on a falling object."

# Step 3: scale to 9 with the pack
# (use the H1 command above)
```

If a step fails with `engine response was not valid JSON`, raise `--max-tokens`
to 4096 or simplify the input.

---

## 11. Common failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| `engine response was not valid JSON: EOF while parsing` | Output truncated at token limit | Raise `--max-tokens` (1024→2048→4096) |
| Epoch converges in 1 tick, no lateral exchange | Columns overconfident on tick 1 | Add `--min-ticks 2` |
| `error: requested model not served` | `--model` id doesn't match server `--alias` | `--probe` to see the served id, match it |
| `--topology ring-bi requires --columns >= 4` | Degeneracy guardrail: ring-bi with ≤3 columns == fully-connected | Use ≥4 columns or switch topology |
| `--column-pack has N column(s) but topology expects M` | Pack count ≠ topology size | Match `--columns` to pack length, or adjust pack |
| `duplicate column id` | Pack has two columns with the same id | Fix the pack |
| `--min-prediction-similarity must be in [0.0, 1.0]` | Flag value outside the allowed range | Pass a value in `[0, 1]` (e.g. `0.85`), or omit to disable |
| 501 on embeddings / semantic convergence | Server doesn't serve embeddings | Semantic convergence is deferred; use confidence-based convergence |

---

## 12. Known limits

- **No integration LLM yet.** The mesh produces per-column outputs; there is no
  separate model that synthesizes them into one answer. "Voting" is
  confidence-threshold filtering only.
- **Embeddings blocked.** The live server returns HTTP 501 on `/v1/embeddings`,
  so full semantic-cosine convergence (§9.3) is deferred.
- **Confidence is self-reported.** The model scores its own confidence; an
  overconfident model converges prematurely. `--min-ticks` is the workaround.
- **Fail-fast.** One bad column response aborts the epoch. See the de-risk ladder.
- **High column counts serialize.** The mesh uses `join_all` on a single-slot
  server; N columns run as N sequential calls. TurboQuant KV-cache (scaling
  path) is the long-term answer.

---

## Further reading

- [Specification](SPECIFICATION.md) — full architectural blueprint
- [Architecture](ARCHITECTURE.md) — crate-level design, topology convention
- [Roadmap](ROADMAP.md) — what's done, what's next
