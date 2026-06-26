#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate a scale-specific PTG column pack.

The default mesh has 4 hand-tuned sphere prompts. At 150 columns those just
repeat 37 times, so a scale run measures replication, not differentiation. This
generator emits N columns whose system prompts are CONCISE but differentiated
along three orthogonal axes (abstraction level, time horizon, diagnostic
stance) while still instructing the per-sphere JSON schema each column must emit
to pass `validate_for_sphere`.

No new DomainSphere / schema envelope is introduced — the sphere selects the
validation keys; the prompt carries the specialization.

Usage:
    python3 scripts/generate-scale-column-pack.py 150 \
        > examples/column-packs/scale-diagnostic-150.toml

The emitted column count MUST equal the `--columns` you pass to `ptg`/`ptg-bench`.
"""
from __future__ import annotations

import sys

# Per-sphere required domain_fields keys (must match ptg_core::validate_for_sphere).
SPHERE_KEYS = {
    "Physics": ["isolated_variables", "empirical_observation"],
    "Mathematics": ["axiomatic_assertions", "deductive_synthesis"],
    "Coding": ["state_variables", "algorithmic_analysis"],
    "Psychology": ["cognitive_biases", "behavioral_synthesis"],
}
SPHERE_GUIDANCE = {
    "Physics": "Isolate the physically relevant variables and ground each claim in an empirical observation (forces, energy, thermal/mechanical constraints).",
    "Mathematics": "State the axiomatic/quantitative relations and synthesize deductively (rates, ratios, complexity, formal structure).",
    "Coding": "Name the state variables / control-flow state and analyze the algorithmic or software failure mode.",
    "Psychology": "Surface cognitive biases / affective drivers and synthesize the likely behavioral intent or response.",
}
SPHERES = ["Physics", "Mathematics", "Coding", "Psychology"]
SHORT = {"Physics": "PHYS", "Mathematics": "MATH", "Coding": "CODE", "Psychology": "PSYCH"}

LEVELS = [
    (0, "concrete/mechanistic — focus on the immediate mechanism and components"),
    (1, "system/relational — focus on interactions and feedback between parts"),
    (2, "strategic/abstract — focus on goals, principles, and trade-offs"),
]
HORIZONS = [
    "immediate (the next event / single step)",
    "short-horizon (minutes to hours)",
    "long-horizon (days or more)",
]
STANCES = [
    "empirical-anchored — prefer observable, measurable evidence",
    "quantitative-anchored — prefer ratios, rates, and formal relations",
    "skeptic/failure-first — actively seek what could be wrong or break",
    "integrative — synthesize across perspectives into one coherent frame",
]


def prompt_for(i: int) -> tuple[str, str, int]:
    sphere = SPHERES[i % len(SPHERES)]
    level_idx = (i // len(SPHERES)) % len(LEVELS)
    horizon_idx = (i // (len(SPHERES) * len(LEVELS))) % len(HORIZONS)
    stance_idx = (i // (len(SPHERES) * len(LEVELS) * len(HORIZONS))) % len(STANCES)
    level, level_desc = LEVELS[level_idx]
    horizon = HORIZONS[horizon_idx]
    stance = STANCES[stance_idx]
    required = SPHERE_KEYS[sphere]
    cid = f"CC_{SHORT[sphere]}_{i:03d}"
    # A concrete example object with the sphere's REQUIRED keys filled in. The
    # concise prose alone did not elicit reliable schema compliance at small
    # model sizes (columns omitted required domain_fields); the example anchors
    # the exact shape the validator expects.
    example_fields = ", ".join(f'"{k}": "..."' for k in required)
    body = (
        f"You are cortical column {cid}: a {sphere} specialist. "
        f"{SPHERE_GUIDANCE[sphere]}\n\n"
        f"Emit ONE flat JSON object (no prose, no code fence) with EXACTLY this shape:\n"
        f'{{"reference_frame_coordinates": "...", "prediction": "...", '
        f'"confidence": 0.0, "domain_fields": {{{example_fields}}}}}\n'
        f'The domain_fields MUST contain: {", ".join(required)}. Fill every field '
        f"with substantive content grounded in the task.\n\n"
        f"Operating lens for THIS column (let it shape your analysis):\n"
        f"- abstraction level {level}: {level_desc}\n"
        f"- time horizon: {horizon}\n"
        f"- diagnostic stance: {stance}\n\n"
        f"Be concise and high-signal. Ground every claim in the task data; "
        f"do not invent facts."
    )
    return cid, body, level


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        sys.stderr.write(__doc__)
        return 2
    try:
        n = int(argv[1])
    except ValueError:
        sys.stderr.write(f"count must be an integer, got {argv[1]!r}\n")
        return 2
    if n < 2:
        sys.stderr.write("count must be >= 2\n")
        return 2

    print('# Auto-generated PTG scale column pack.')
    print(f'# {n} columns, differentiated by abstraction level / time horizon /')
    print('# diagnostic stance. Regenerate via scripts/generate-scale-column-pack.py.')
    print('#')
    print('# CAVEAT: these CONCISE generated prompts are differentiated but do not yet')
    print('# elicit reliable per-sphere JSON schema compliance at small model sizes')
    print('# (e.g. gemma-4-e2b): some Coding columns omit `state_variables`, failing')
    print('# validate_for_sphere and aborting the fail-fast mesh. The hand-tuned default')
    print('# sphere prompts DO comply. For a reliable differentiated pack, base each')
    print('# prompt on the default sphere prompt and append only the lens suffix, or')
    print('# raise the model size. See docs/SCALE_BENCHMARKING.md.')
    print('description = "scale diagnostic pack: differentiated sphere+lens columns"')
    print()
    for i in range(n):
        cid, body, level = prompt_for(i)
        sphere = SPHERES[i % len(SPHERES)]
        # TOML basic multi-line string (""" ... """) keeps the prompt readable and
        # avoids manual escaping of quotes/newlines.
        escaped = body.replace('"""', '\\"""')
        print("[[columns]]")
        print(f'id = "{cid}"')
        print(f'sphere = "{sphere}"')
        print(f'level = {level}')
        print('system_prompt = """')
        print(escaped)
        print('"""')
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
