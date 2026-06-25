# Project Thousand-Gemma (PTG) development recipes.
# Use `just --list` to see available commands.

set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# Show available recipes
default:
    just --list

# Full validation: format check, lint, build, test, doc (PR/CI gate)
check: fmt-check lint build test doc

# Quick validation: format check, lint, test
quick-check: fmt-check lint test

# Format all Rust code
fmt:
    cargo fmt --all

# Check formatting without modifying files
fmt-check:
    cargo fmt --all -- --check

# Clippy with the repository's zero-warning policy
lint:
    cargo clippy --all-features --all-targets -- -D warnings

# Run tests (fast, parallel) via nextest
test:
    cargo nextest run --all-features

# Run tests with captured output shown
test-verbose:
    cargo nextest run --all-features --no-capture

# Debug build
build:
    cargo build --all-features

# Release build
build-release:
    cargo build --release --all-features

# Build documentation
doc:
    cargo doc --all-features --no-deps

# Remove build artifacts
clean:
    cargo clean
