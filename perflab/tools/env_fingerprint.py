"""Environment fingerprint projection and cross-run comparison gating.

``collect_system_info()`` (``perflab.tools.sysinfo``) gathers a broad snapshot
of the host -- clocks, load average, throttle state, ISA features, all useful
for diagnosing *one* run. Most of that is not machine *identity*: load average
and clock state vary run to run on the same box, so folding them into a
fingerprint would flag a busy afternoon as different hardware. This module
extracts the subset that IS identity -- fields that only change when the run
moved to a different machine or toolchain -- and uses it to answer one
question: are these two runs' numbers even comparable?

Two tiers, because not every mismatch is equally dangerous:

* ``BLOCKING_FIELDS`` -- different silicon or OS entirely (GPU/TPU model,
  CPU model, OS/arch). A 40x "speedup" between an H100 run and a laptop run
  is not a speedup, it is two unrelated numbers, so a difference here makes a
  comparison ``"incomparable"``.
* ``ADVISORY_FIELDS`` -- same silicon, different toolchain (driver,
  CUDA/torch/jax version, compiler). These can move the numbers too, but a
  same-hardware driver bump is often the very thing being compared -- so a
  mismatch here is ``"advisory"``, not blocking.

This mirrors the repo's existing "report unverified rather than silently
downgrade" precedent (see ``perflab.analyzers.decision``, where a verdict
degrades to ``verified=False`` rather than quietly passing a candidate): a run
with no captured environment -- it predates ``system_info.json``, or the probe
failed -- is not assumed compatible. It is reported ``"unverified"``, and
callers decide how to show that rather than the comparison silently proceeding
as if the hardware matched.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

#: A difference here means the two runs were not measured on the same
#: machine -- refuse the comparison (see `compare_fingerprints`).
BLOCKING_FIELDS = (
    "system", "machine", "cpu_model", "gpu_name", "gpu_count",
    "tpu_chip", "tpu_count",
)

#: A difference here is the same hardware on a different toolchain -- worth
#: flagging, not worth refusing.
ADVISORY_FIELDS = (
    "cpu_count", "gpu_driver", "cuda_version", "torch_version",
    "jax_version", "triton_version", "cpp_compiler", "python_version",
)

_ALL_FIELDS = (*BLOCKING_FIELDS, *ADVISORY_FIELDS)


def fingerprint_from_sysinfo(info: dict) -> dict:
    """Project the comparison-relevant subset of ``collect_system_info()``.

    Every field is optional: a key is included only when the source data has
    it, never as ``None`` -- an absent field means "not known", and storing
    it as ``None`` would make it indistinguishable from an actually-null
    value. Not a new probe: purely a projection of data ``collect_system_info``
    already returns.
    """
    fp: dict[str, Any] = {}
    for key in (
        "system", "machine", "python_version", "cpu_model", "cpu_count",
        "tpu_chip", "tpu_count", "cuda_version", "torch_version",
        "jax_version", "triton_version", "cpp_compiler",
    ):
        value = info.get(key)
        if value is not None:
            fp[key] = value

    nvidia_gpus = info.get("nvidia_gpus")
    if nvidia_gpus:
        fp["gpu_count"] = len(nvidia_gpus)
        name = nvidia_gpus[0].get("name")
        if name is not None:
            fp["gpu_name"] = name
        driver = nvidia_gpus[0].get("driver_version")
        if driver is not None:
            fp["gpu_driver"] = driver

    return fp


@dataclass
class FingerprintComparison:
    """The result of comparing two environment fingerprints.

    ``verdict`` is the gate: ``"comparable"`` (proceed normally),
    ``"advisory"`` (proceed, but say what differs), ``"incomparable"``
    (refuse without ``--force``), or ``"unverified"`` (one or both sides have
    no fingerprint at all, so compatibility is simply not known).
    """
    matched: list[str] = field(default_factory=list)
    differing: list[tuple[str, str, str]] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    verdict: str = "unverified"

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "matched": list(self.matched),
            "differing": [[f, a, b] for f, a, b in self.differing],
            "unknown": list(self.unknown),
        }

    @classmethod
    def from_dict(cls, d: dict) -> FingerprintComparison:
        """Inverse of `to_dict` -- reconstructs a comparison already reduced
        to its wire/storage shape (e.g. a value read back out of
        ``RunStore.compare_runs``), so formatting helpers below can be shared
        rather than re-implemented against the plain dict."""
        return cls(
            matched=list(d.get("matched", [])),
            differing=[(f, a, b) for f, a, b in d.get("differing", [])],
            unknown=list(d.get("unknown", [])),
            verdict=d.get("verdict", "unverified"),
        )


def compare_fingerprints(a: dict | None, b: dict | None) -> FingerprintComparison:
    """Compare two environment fingerprints and return the compatibility verdict.

    Checked in order:

    1. no fingerprint at all on either side -> ``"unverified"``. An empty
       dict counts as "no fingerprint" too -- ``fingerprint_from_sysinfo``
       never has a reason to return one otherwise.
    2. any ``BLOCKING_FIELDS`` entry differs -> ``"incomparable"``.
    3. any ``ADVISORY_FIELDS`` entry differs -> ``"advisory"``.
    4. otherwise -> ``"comparable"``.

    A field present on only one side lands in ``unknown``, not ``differing``,
    and never affects the verdict -- a CPU-only run and a GPU run legitimately
    disagree on whether ``gpu_name`` exists at all, which is a different
    situation from two runs that both report a GPU and disagree on which one.
    """
    if not a or not b:
        return FingerprintComparison(verdict="unverified")

    matched: list[str] = []
    differing: list[tuple[str, str, str]] = []
    unknown: list[str] = []
    for key in _ALL_FIELDS:
        in_a, in_b = key in a, key in b
        if in_a and in_b:
            if a[key] == b[key]:
                matched.append(key)
            else:
                differing.append((key, str(a[key]), str(b[key])))
        elif in_a or in_b:
            unknown.append(key)

    differing_fields = {f for f, _, _ in differing}
    if differing_fields & set(BLOCKING_FIELDS):
        verdict = "incomparable"
    elif differing_fields & set(ADVISORY_FIELDS):
        verdict = "advisory"
    else:
        verdict = "comparable"

    return FingerprintComparison(matched=matched, differing=differing, unknown=unknown, verdict=verdict)


def blocking_differences(
    differing: Sequence[tuple[str, str, str] | list[str]],
) -> list[tuple[str, str, str]]:
    """The subset of a ``differing`` list whose field is a `BLOCKING_FIELDS` entry.

    Separated out because an ``"incomparable"`` message should show only the
    fields that actually blocked the comparison -- an advisory field that
    happens to differ too is noise in that message, even though it is still
    present in the full ``differing`` list. Accepts either the dataclass's
    tuples or the JSON-shaped lists from `FingerprintComparison.to_dict`, so
    callers can use it on either representation.
    """
    return [(d[0], d[1], d[2]) for d in differing if d[0] in BLOCKING_FIELDS]


def format_comparison(cmp: FingerprintComparison) -> list[str]:
    """Human-readable lines describing ``cmp``.

    Shared by the ``compare`` CLI and ``ci-check`` output so both surfaces
    describe a mismatch identically. Does not know which side is "run A" /
    "baseline" -- callers that need to name sides prepend their own line.
    """
    if cmp.verdict == "unverified":
        return ["environment fingerprint unavailable for one or both sides — compatibility is not known"]
    return [f"{field_name}: {va} != {vb}" for field_name, va, vb in cmp.differing]
