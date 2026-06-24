#!/usr/bin/env bash
# Start the Gemma 4 E2B QAT model server for PTG.
#
# Usage: scripts/start-gemma4-qat.sh
# Override: PTG_LLAMA_SERVER, PTG_GGUF_DIR, PTG_PORT, PTG_HOST, PTG_MODEL_ALIAS
#
# This is the DEFAULT encoding (verified working: 2.7 GB RAM, 0.9s load).
# See docs/TUTORIAL.md §"Model setup" for the fallback and scaling-path options.

set -euo pipefail

# ---- Configuration (overridable via env) ----------------------------------
HOST="${PTG_HOST:-127.0.0.1}"
PORT="${PTG_PORT:-18136}"
ALIAS="${PTG_MODEL_ALIAS:-gemma-4-e2b-qat}"
GGUF_DIR="${PTG_GGUF_DIR:-$HOME/.cache/ptg/gguf}"
HF_REPO="unsloth/gemma-4-E2B-it-qat-GGUF"
GGUF_FILE="gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"

# ---- Detect llama-server (order: env, ~/llama-spike, PATH) ----------------
detect_server() {
    for candidate in \
        "${PTG_LLAMA_SERVER:-}" \
        "${LLAMA_SERVER:-}" \
        "$HOME/llama-spike/llama.cpp/build/bin/llama-server" \
        "$(command -v llama-server 2>/dev/null || true)"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

SERVER="$(detect_server || true)"
if [ -z "$SERVER" ]; then
    cat >&2 <<'EOF'
ERROR: could not find llama-server. Set it via one of:
  export PTG_LLAMA_SERVER=/path/to/llama-server
  # or build from source:  https://github.com/ggml-org/llama.cpp
  #   cmake -B build && cmake --build build --config Release
EOF
    exit 1
fi

# ---- Already running? -----------------------------------------------------
if curl -sf "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
    echo "Server already running at http://$HOST:$PORT — model alias: $ALIAS"
    echo "Next: cargo run -p ptg-cli --bin ptg -- --vllm-url http://$HOST:$PORT --model $ALIAS --probe"
    exit 0
fi

# ---- Model present? -------------------------------------------------------
GGUF_PATH="$GGUF_DIR/$GGUF_FILE"
if [ ! -f "$GGUF_PATH" ]; then
    echo "Model not found at $GGUF_PATH"
    # Try to download if we have the hf CLI or huggingface-cli
    if command -v huggingface-cli >/dev/null 2>&1 || command -v hf >/dev/null 2>&1; then
        echo "Downloading $HF_REPO/$GGUF_FILE to $GGUF_DIR ..."
        mkdir -p "$GGUF_DIR"
        # Prefer the modern `hf` CLI, fall back to huggingface-cli
        if command -v hf >/dev/null 2>&1; then
            hf download "$HF_REPO" "$GGUF_FILE" --local-dir "$GGUF_DIR"
        else
            huggingface-cli download "$HF_REPO" "$GGUF_FILE" --local-dir "$GGUF_DIR"
        fi
    else
        cat >&2 <<EOF
Cannot download: neither hf nor huggingface-cli is installed.
Install one of:
  pip install huggingface-hub    # then: hf login (accept Gemma license at huggingface.co)
Then re-run this script.
Or download manually from https://huggingface.co/$HF_REPO
(NOTE: Gemma is gated — you must accept the license and authenticate with HF_TOKEN.)
EOF
        exit 1
    fi
fi

# ---- Launch ---------------------------------------------------------------
echo "Starting: $SERVER"
echo "  model : $GGUF_PATH"
echo "  host  : $HOST:$PORT"
echo "  alias : $ALIAS"
echo ""
echo "Next: cargo run -p ptg-cli --bin ptg -- \\"
echo "    --vllm-url http://$HOST:$PORT --model $ALIAS --probe"
echo ""

exec "$SERVER" \
    -m "$GGUF_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    -c 4096 \
    -ngl 99 \
    -fa on \
    --jinja \
    --alias "$ALIAS" \
    --reasoning off \
    --reasoning-format none
