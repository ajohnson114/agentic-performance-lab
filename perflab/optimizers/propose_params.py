from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class KnobPatch:
    description: str
    new_knobs: dict

def load_knobs(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def save_knobs(path: Path, knobs: dict) -> None:
    path.write_text(yaml.safe_dump(knobs, sort_keys=False), encoding="utf-8")


def generate_sweep_candidates(current_knobs: dict) -> list[KnobPatch]:
    """Generate grid search candidates from a ``sweep`` section in tuning.yaml.

    The ``sweep`` key maps knob names to lists of values to try::

        sweep:
          threadsPerBlock: [16, 32, 64, 128, 256]
          BLOCK_SIZE_M: [32, 64, 128]

    Returns one KnobPatch per combination (Cartesian product).  The ``sweep``
    key is stripped from each candidate's ``new_knobs`` so the benchmark
    harness only sees flat knob values.
    """
    sweep = current_knobs.get("sweep")
    if not sweep or not isinstance(sweep, dict):
        return []

    knob_names = list(sweep.keys())
    value_lists = [sweep[k] for k in knob_names]

    # Validate: each value list must be a list
    for _name, vals in zip(knob_names, value_lists, strict=True):
        if not isinstance(vals, list) or len(vals) == 0:
            return []

    candidates: list[KnobPatch] = []
    for combo in itertools.product(*value_lists):
        new_knobs = {k: v for k, v in current_knobs.items() if k != "sweep"}
        desc_parts = []
        for name, val in zip(knob_names, combo, strict=True):
            new_knobs[name] = val
            desc_parts.append(f"{name}={val}")

        candidates.append(KnobPatch(
            description=", ".join(desc_parts),
            new_knobs=new_knobs,
        ))

    return candidates


def sample_candidates(
    candidates: list[KnobPatch],
    max_trials: int,
    seed: int = 42,
) -> list[KnobPatch]:
    """Deterministically subsample candidates when the grid is too large."""
    if len(candidates) <= max_trials:
        return candidates
    return random.Random(seed).sample(candidates, max_trials)


def propose_knob_sweep(current: dict) -> list[KnobPatch]:
    """Propose knob candidates for ``perflab optimize``.

    If ``current`` contains a ``sweep`` section, generates a grid search over
    those values.  Otherwise falls back to the legacy hardcoded sweep over
    ``torch_compile`` and ``batch`` toggles.
    """
    # Data-driven grid search takes priority
    sweep_candidates = generate_sweep_candidates(current)
    if sweep_candidates:
        return sweep_candidates

    # Legacy fallback: hardcoded torch_compile / batch sweep
    candidates = []
    compile_opts = [False, True]
    for tc in compile_opts:
        if tc == current.get("torch_compile", False):
            continue
        knobs = dict(current)
        knobs["torch_compile"] = tc
        candidates.append(KnobPatch(description=f"Set torch_compile={tc}", new_knobs=knobs))

    for bs in [1, 4, 16]:
        if bs == current.get("batch", 1):
            continue
        knobs = dict(current)
        knobs["batch"] = bs
        candidates.append(KnobPatch(description=f"Set batch={bs}", new_knobs=knobs))

    return candidates
