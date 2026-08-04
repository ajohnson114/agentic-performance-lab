"""Block-interleaved (paired) A/B benchmarking.

The problem this exists to remove
=================================
Everywhere else in PerfLab the incumbent and the candidate are measured *at
different times*: ``phases/baseline.py`` measures the incumbent near the start
of an iteration, and ``phases/evaluate.py`` measures each candidate minutes
later, after LLM latency, patch application, a build, and a correctness run.
That gap is not neutral. A laptop's fan curve, a datacenter GPU's thermal
envelope, a CPU's opportunistic boost budget and a noisy-neighbour VM all drift
on exactly the timescale of those minutes, and the drift lands entirely on one
arm of the comparison. It is a *systematic offset between the arms*, not noise:

    measured(arm, t) = true(arm) + drift(t) + eps

With the two arms measured at t_a and t_b, the estimate of the effect is
``true(B) - true(A) + [drift(t_b) - drift(t_a)]``. The bracket does not shrink
when you add repeats, tighten confidence intervals, or lengthen the run --
those all attack ``eps``. No amount of interval machinery on top of a biased
design makes it unbiased, and the drift check in ``evaluate.py`` is a smoke
alarm after the fact, not a control.

Interleaving is the control. If the two arms are measured *alternately*, over
the same stretch of wall-clock, then ``drift`` is common-mode within each pair
and cancels in the per-pair difference ``d_i = b_i - a_i``.

The second, less obvious payoff: sensitivity
--------------------------------------------
Unpaired, the dispersion the test must overcome is ``var(A) + var(B)``, and on
a real machine most of that is slow, shared, between-run wander -- the same
wander that biases the comparison. Paired, the test runs on ``d_i``, whose
variance is ``var(A) + var(B) - 2*cov(A, B)``; the common-mode component sits
in ``cov`` and subtracts out. On a machine whose per-run CV is 8% but whose
*within-pair* spread is 1-2%, that is what makes the project's default 2%
``regression_tolerance`` physically resolvable instead of a number that
silently requires a 15% win to clear.

Why interleave at the process level
-----------------------------------
The textbook way to pair two implementations is to host both in one process and
alternate calls. That would mean rewriting every task harness, and for the
compiled tasks (C++/CUDA) it means linking two versions of the same symbol into
one binary -- brutal, and a source of measurement artifacts of its own
(different code layout, shared caches warmed by the other arm).

None of that is necessary, because PerfLab already has the two ingredients:

  * every candidate is evaluated in a disposable copy of the workspace, so
    standing up a second copy holding the *incumbent* source is one more
    ``copytree``; and
  * every ``bench.py`` already honours ``PERFLAB_BENCH_REPEATS``, so a process
    can be told to run a short *block* of k repeats instead of the full count.

So we alternate **spawns**, each running a block of k repeats, and pair the
per-block statistics. Zero harness-protocol change, and it works identically
for Python and for compiled tasks. The unit of pairing is the block, not the
repeat -- which is the right granularity anyway, since the drift being
cancelled is slow relative to a single repeat.

ABBA, not ABAB
--------------
The naive interleave is strict alternation, ABAB. It does not work, and the
reason is worth spelling out because it is the whole point of the design.

Number the measured spawns 0, 1, 2, ... and model drift as linear in position,
``m_p = mu_arm + c*p``. The unpaired arm-mean difference picks up
``c * (mean position of B - mean position of A)``. With ABAB over 4 pairs, A
sits at 0,2,4,6 (mean 3) and B at 1,3,5,7 (mean 4): B is *always* measured one
slot later, so the estimate inherits a bias of exactly one drift-step ``c``.
Alternation removed none of the bias, it merely shrank the gap from minutes to
seconds.

Counterbalancing fixes it. Flip the within-pair order on alternate pairs --
A B | B A | A B | B A -- and A sits at 0,3,4,7 (mean 3.5) while B sits at
1,2,5,6 (mean 3.5). The position means are equal, so a linear drift cancels
*exactly*, not approximately. :func:`position_imbalance` computes that
difference of position means directly; it is the coefficient multiplying any
linear drift term, and it is 0.0 for ABBA with an even number of pairs.

An odd number of pairs cannot be perfectly counterbalanced by this scheme (the
last pair has no partner to mirror it) and leaves a residual imbalance of
``1/n_pairs`` slots -- still an order of magnitude better than ABAB's 1.0, but
this is why the default block count is even.

What counterbalancing costs, stated honestly
--------------------------------------------
ABBA does not make each individual difference drift-free, and it is worth being
precise about that rather than overclaiming. Within an "AB" pair the two spawns
are one slot apart, so ``d_i`` carries ``+c``; within the mirrored "BA" pair it
carries ``-c``. What ABBA guarantees is that those residuals arrive in equal
numbers and opposite signs, so the *distribution* of ``d`` is centered on the
true effect: mean and median are both unbiased, and the signed-rank null holds.
ABAB, by contrast, puts ``+c`` on every single pair -- a shift of the whole
distribution, which no amount of averaging removes.

So counterbalancing converts a *bias* into symmetric *spread*, which is the
right trade: bias does not shrink as you add pairs, this spread does, and an
inflated spread can only make the gate harder to pass (it fails safe). The
alternative -- treating a whole ABBA super-block as one pair, ``mean(A at 0,3)``
vs ``mean(B at 1,2)``, which cancels drift exactly *within* each pair -- halves
the pair count, and at these sample sizes the exact signed-rank test needs pairs
far more than it needs each one to be individually pristine (at 3 pairs the best
attainable p is 0.125, i.e. no rejection is possible at all). Pairs win.

Sizing the design
-----------------
``blocks`` (= number of pairs) is bounded below by the statistics, not by
taste. The accompanying decision rule uses an *exact* Wilcoxon signed-rank
test, whose smallest attainable one-sided p-value at n pairs is ``2**-n``:

    n = 4  ->  0.0625   can never reach p < 0.05.   Useless.
    n = 5  ->  0.0312   attainable, but odd (imbalance 0.2 slots).
    n = 6  ->  0.0156   attainable, and even (imbalance 0.0).

Hence :data:`MIN_BLOCKS` = 5 and :data:`DEFAULT_BLOCKS` = 6.

``repeats_per_block`` (k) trades warmup amortisation against wall clock. Each
spawn pays process startup, imports and ``PERFLAB_BENCH_WARMUP`` again, so a
k of 1 spends most of its time not measuring. We take
``k = max(MIN_REPEATS_PER_BLOCK, ceil(total_repeats / blocks))``: when the task
already asks for enough repeats to spread across the blocks, the *measured*
work per arm is unchanged from the unpaired path and the whole design costs
~2x (one extra arm) plus the extra warmups. When the task asks for very few
repeats (the matmul demo asks for 3), the block floor dominates and the run
costs more than 2x -- :class:`BlockPlan` reports exactly that ratio rather than
letting the caller assume the happy case.

Cold start
----------
The first spawn of a run measures a colder machine than the rest: page cache
for the freshly-copied workspace, first-touch page faults, CPU still at idle
clocks. Under ABBA the first slot always belongs to arm A, so an uncorrected
cold-start penalty would land entirely on the incumbent and bias the comparison
*toward accepting* -- the dangerous direction. ``lead_in`` therefore spends one
discarded spawn on each arm, in A-then-B order, before slot 0. Two spawns is a
small price for removing an asymmetry that no amount of pairing would cancel,
and :func:`first_block_penalty` exists so the assumption can be re-checked on
any machine instead of taken on faith.
"""
from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from perflab.analyzers.bench_stats import extract_repeated_values
from perflab.runners.benchmark import metric_value, run_benchmark
from perflab.tools.isolation import IsolationPolicy

#: Arm labels. "A" is the incumbent (the thing being defended), "B" the
#: candidate (the thing being proposed), matching the ABBA notation above.
INCUMBENT = "A"
CANDIDATE = "B"

#: Pairs per interleaved run. Even, so counterbalancing is exact; 6 rather than
#: 4 because an exact Wilcoxon signed-rank test cannot reach p < 0.05 below 5
#: pairs. See "Sizing the design" above.
DEFAULT_BLOCKS = 6

#: Below this the paired test is arithmetically incapable of rejecting.
MIN_BLOCKS = 5

#: A block of 1 repeat spends most of its wall clock on startup and warmup.
MIN_REPEATS_PER_BLOCK = 2


# --- Ordering ---------------------------------------------------------------


def interleave_order(n_pairs: int, *, counterbalanced: bool = True) -> list[str]:
    """Spawn order for ``n_pairs`` interleaved blocks.

    ``counterbalanced=True`` gives ABBA ABBA ... (the within-pair order flips on
    odd-numbered pairs); ``False`` gives the naive ABAB alternation, which is
    kept only so tests can demonstrate the bias it carries -- see the module
    docstring and :func:`position_imbalance`.
    """
    order: list[str] = []
    for i in range(max(n_pairs, 0)):
        if counterbalanced and i % 2 == 1:
            order.extend((CANDIDATE, INCUMBENT))
        else:
            order.extend((INCUMBENT, CANDIDATE))
    return order


def position_imbalance(order: Sequence[str]) -> float:
    """``mean(position of B) - mean(position of A)``, in spawn slots.

    This is not a diagnostic curiosity, it is the exact coefficient on a linear
    drift term: under ``m_p = mu_arm + c*p``, the difference of the arm means
    equals ``true effect + c * position_imbalance(order)``. A design with
    imbalance 0.0 cancels linear drift exactly; ABAB has imbalance 1.0 and so
    inherits one full drift-step of bias no matter how many pairs are run.

    Returns 0.0 for an empty or single-armed order (nothing to be imbalanced).
    """
    a_pos = [i for i, arm in enumerate(order) if arm == INCUMBENT]
    b_pos = [i for i, arm in enumerate(order) if arm == CANDIDATE]
    if not a_pos or not b_pos:
        return 0.0
    return statistics.fmean(b_pos) - statistics.fmean(a_pos)


# --- Sizing -----------------------------------------------------------------


@dataclass(frozen=True)
class BlockPlan:
    """How an interleaved run is sized, and what it costs relative to unpaired.

    ``measured_repeats_per_arm`` is ``blocks * repeats_per_block``; the unpaired
    authoritative path measures ``total_repeats`` for the candidate only, so
    ``cost_ratio`` is the honest multiplier on *measured* work. It deliberately
    excludes per-spawn startup and warmup, which :class:`PairedRun` reports as
    real wall clock instead of estimating.
    """

    blocks: int
    repeats_per_block: int
    total_repeats: int
    lead_in_spawns: int
    counterbalanced: bool

    @property
    def measured_repeats_per_arm(self) -> int:
        return self.blocks * self.repeats_per_block

    @property
    def spawns(self) -> int:
        return 2 * self.blocks + self.lead_in_spawns

    @property
    def cost_ratio(self) -> float:
        """Measured repeats across both arms, over the unpaired path's count."""
        if self.total_repeats <= 0:
            return math.inf
        return 2.0 * self.measured_repeats_per_arm / self.total_repeats

    @property
    def imbalance(self) -> float:
        return position_imbalance(interleave_order(
            self.blocks, counterbalanced=self.counterbalanced,
        ))


def plan_blocks(
    total_repeats: int,
    *,
    blocks: int = DEFAULT_BLOCKS,
    repeats_per_block: int | None = None,
    lead_in: bool = True,
    counterbalanced: bool = True,
) -> BlockPlan:
    """Size an interleaved run, refusing block counts the test cannot use.

    Raises ``ValueError`` below :data:`MIN_BLOCKS` rather than running a design
    whose best possible p-value is above any usable alpha -- failing loudly at
    setup beats spending the wall clock and then reporting "not significant" for
    a reason that had nothing to do with the candidate.
    """
    if blocks < MIN_BLOCKS:
        raise ValueError(
            f"blocks={blocks} is below MIN_BLOCKS={MIN_BLOCKS}: an exact "
            f"signed-rank test on {blocks} pairs cannot produce a p-value below "
            f"{0.5 ** blocks:.4f}, so the design could never reject"
        )
    if repeats_per_block is None:
        repeats_per_block = max(
            MIN_REPEATS_PER_BLOCK, math.ceil(max(total_repeats, 1) / blocks),
        )
    if repeats_per_block < 1:
        raise ValueError(f"repeats_per_block={repeats_per_block} must be >= 1")
    return BlockPlan(
        blocks=blocks,
        repeats_per_block=repeats_per_block,
        total_repeats=max(total_repeats, 0),
        lead_in_spawns=2 if lead_in else 0,
        counterbalanced=counterbalanced,
    )


# --- Results ----------------------------------------------------------------


@dataclass(frozen=True)
class SpawnMeasurement:
    """What one benchmark process reported.

    ``samples`` are the per-repeat values inside that one process (see
    ``bench_stats.extract_repeated_values``); ``bench`` is the raw bench.json so
    the caller can still run contract validation and anti-gaming checks on
    every spawn, not just on an aggregate.
    """

    value: float
    samples: list[float] = field(default_factory=list)
    bench: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Spawn:
    """One measured spawn, tagged with where it sat in the sequence."""

    arm: str
    pair: int
    position: int
    measurement: SpawnMeasurement
    wall_s: float


@dataclass(frozen=True)
class PairedRun:
    """The outcome of a block-interleaved A/B run.

    ``pairs`` is in ``(candidate, incumbent)`` order, matching
    ``analyzers.decision.Comparison.pairs``, so it can be handed straight to a
    paired decision rule.

    ``candidate_value``/``incumbent_value`` are the *medians* of the per-block
    values, not the means. Block values are themselves aggregates (typically a
    median over k repeats) whose distribution has a one-sided tail from
    interference events; the median matches the robustness of the signed-rank
    test applied to the differences. The consequence -- that
    ``median(B) - median(A)`` need not equal ``median(d)`` -- is deliberate and
    fail-safe: the decision rule requires both the point-estimate materiality
    floor and the paired significance test, so the two disagreeing means
    rejection.
    """

    pairs: list[tuple[float, float]]
    candidate_value: float
    incumbent_value: float
    candidate_blocks: list[float]
    incumbent_blocks: list[float]
    candidate_samples: list[float]
    incumbent_samples: list[float]
    candidate_bench: dict
    incumbent_bench: dict
    order: list[str]
    spawns: list[Spawn]
    lead_in: list[Spawn]
    plan: BlockPlan
    wall_s: float

    @property
    def diffs(self) -> list[float]:
        """Raw ``b_i - a_i`` per pair. Sign is in the metric's own direction;
        mode-aware normalisation belongs to ``Comparison.paired_diffs``."""
        return [b - a for b, a in self.pairs]

    @property
    def paired_cv(self) -> float | None:
        """``stdev(d) / |median(incumbent)|`` -- the dispersion the paired test
        must actually overcome, and the number to compare against the per-arm
        CV to see whether pairing bought anything on this machine."""
        if len(self.pairs) < 2 or self.incumbent_value == 0:
            return None
        return statistics.stdev(self.diffs) / abs(self.incumbent_value)

    def arm_cv(self, arm: str) -> float | None:
        """CV of one arm's block values -- the unpaired dispersion, for contrast."""
        blocks = self.candidate_blocks if arm == CANDIDATE else self.incumbent_blocks
        if len(blocks) < 2:
            return None
        mean = statistics.fmean(blocks)
        if mean == 0:
            return None
        return statistics.stdev(blocks) / abs(mean)


def first_block_penalty(run: PairedRun) -> float | None:
    """How much slower/faster pair 0 measured than the rest, as a fraction.

    Computed on the *paired* level (the mean of each pair) so it reflects a
    cold-start effect common to both arms rather than an arm difference.
    Positive means pair 0 sat above the later pairs. Exists so the ``lead_in``
    assumption can be audited on a given machine: if this stays near zero with
    ``lead_in=False``, the two priming spawns are wasted wall clock there.

    Returns None with fewer than 2 pairs.
    """
    if len(run.pairs) < 2:
        return None
    levels = [statistics.fmean(p) for p in run.pairs]
    rest = statistics.fmean(levels[1:])
    if rest == 0:
        return None
    return (levels[0] - rest) / abs(rest)


# --- Driver -----------------------------------------------------------------

#: A spawn is anything that, given an arm label, produces one measurement.
#: Keeping the driver behind this callable is what makes the ordering logic
#: testable against a synthetic drifting machine with no subprocesses at all.
SpawnFn = Callable[[str], SpawnMeasurement]


def run_interleaved(spawn: SpawnFn, plan: BlockPlan) -> PairedRun:
    """Execute ``plan``'s spawn sequence and pair the per-block statistics.

    Exceptions from ``spawn`` propagate: a partially-completed ABBA sequence is
    no longer counterbalanced, so there is nothing safe to salvage. The caller
    is expected to fall back to the unpaired path (and to say so), which is the
    fail-closed direction.
    """
    started = time.monotonic()

    lead_in: list[Spawn] = []
    for i in range(plan.lead_in_spawns):
        # A then B, so neither arm is the one that eats the cold machine.
        arm = INCUMBENT if i % 2 == 0 else CANDIDATE
        t0 = time.monotonic()
        measurement = spawn(arm)
        lead_in.append(Spawn(
            arm=arm, pair=-1, position=-(plan.lead_in_spawns - i),
            measurement=measurement, wall_s=time.monotonic() - t0,
        ))

    order = interleave_order(plan.blocks, counterbalanced=plan.counterbalanced)
    spawns: list[Spawn] = []
    for position, arm in enumerate(order):
        t0 = time.monotonic()
        measurement = spawn(arm)
        spawns.append(Spawn(
            arm=arm, pair=position // 2, position=position,
            measurement=measurement, wall_s=time.monotonic() - t0,
        ))

    by_pair: dict[int, dict[str, Spawn]] = {}
    for s in spawns:
        by_pair.setdefault(s.pair, {})[s.arm] = s

    pairs: list[tuple[float, float]] = []
    cand_blocks: list[float] = []
    inc_blocks: list[float] = []
    for pair_idx in sorted(by_pair):
        slot = by_pair[pair_idx]
        if INCUMBENT not in slot or CANDIDATE not in slot:
            continue
        cand = slot[CANDIDATE].measurement.value
        inc = slot[INCUMBENT].measurement.value
        pairs.append((cand, inc))
        cand_blocks.append(cand)
        inc_blocks.append(inc)

    if not pairs:
        raise RuntimeError("interleaved run produced no complete pairs")

    cand_spawns = [s for s in spawns if s.arm == CANDIDATE]
    inc_spawns = [s for s in spawns if s.arm == INCUMBENT]

    return PairedRun(
        pairs=pairs,
        candidate_value=statistics.median(cand_blocks),
        incumbent_value=statistics.median(inc_blocks),
        candidate_blocks=cand_blocks,
        incumbent_blocks=inc_blocks,
        # Flattened per-repeat samples, in spawn order. These carry the full
        # between-block spread, which is what the *unpaired* fallback wants;
        # the paired test never looks at them.
        candidate_samples=[v for s in cand_spawns for v in s.measurement.samples],
        incumbent_samples=[v for s in inc_spawns for v in s.measurement.samples],
        candidate_bench=cand_spawns[-1].measurement.bench if cand_spawns else {},
        incumbent_bench=inc_spawns[-1].measurement.bench if inc_spawns else {},
        order=order,
        spawns=spawns,
        lead_in=lead_in,
        plan=plan,
        wall_s=time.monotonic() - started,
    )


def run_paired_benchmark(
    cmd: str,
    *,
    incumbent_cwd: Path,
    candidate_cwd: Path,
    metric_name: str,
    total_repeats: int,
    warmup: int | None = None,
    blocks: int = DEFAULT_BLOCKS,
    repeats_per_block: int | None = None,
    lead_in: bool = True,
    counterbalanced: bool = True,
    program_type: str | None = None,
    rlimit_as_gb: float | None = None,
    env_passthrough: list[str] | None = None,
    isolation: IsolationPolicy | None = None,
    on_spawn: Callable[[str, int, int], None] | None = None,
) -> PairedRun:
    """Benchmark two workspaces against each other, block-interleaved.

    ``incumbent_cwd`` and ``candidate_cwd`` must be *equivalent* disposable
    workspace copies -- same creation method, same filesystem, differing only
    in the patch applied. Benchmarking the incumbent in the real workspace
    while the candidate runs from a temp copy would reintroduce exactly the
    kind of systematic between-arm difference this module exists to remove
    (different page-cache state, possibly a different device), on top of the
    usual reason candidate code never runs in the real workspace.

    Every spawn goes through :func:`runners.benchmark.run_benchmark`, so it
    keeps the rlimits, env allowlist, CPU pinning, isolation policy and
    bench.json anti-tampering checks the unpaired path has. Only the repeat
    count differs, via ``PERFLAB_BENCH_REPEATS``.

    ``on_spawn(arm, pair_index, position)`` is called before each measured
    spawn for progress reporting.
    """
    plan = plan_blocks(
        total_repeats, blocks=blocks, repeats_per_block=repeats_per_block,
        lead_in=lead_in, counterbalanced=counterbalanced,
    )
    cwds = {INCUMBENT: incumbent_cwd, CANDIDATE: candidate_cwd}
    seen = {INCUMBENT: 0, CANDIDATE: 0}

    def _spawn(arm: str) -> SpawnMeasurement:
        if on_spawn is not None:
            on_spawn(arm, seen[arm], seen[INCUMBENT] + seen[CANDIDATE])
        seen[arm] += 1
        _, bench = run_benchmark(
            cmd, cwd=cwds[arm], fast_mode=False, program_type=program_type,
            rlimit_as_gb=rlimit_as_gb, env_passthrough=env_passthrough,
            isolation=isolation,
            warmup=warmup,
            # The block size is the one thing that differs from an ordinary
            # authoritative run, and it is not negotiable: passed through
            # `env` rather than `repeats=` because run_benchmark lets a
            # session-level PERFLAB_BENCH_REPEATS outrank its `repeats`
            # argument. That precedence is right for a normal run and wrong
            # here -- a stray env var would silently collapse the blocks back
            # to full-length runs and multiply the wall clock by the block
            # count. `env` entries are seeded before every setdefault in
            # run_benchmark, so this wins outright. Identical for both arms.
            env={"PERFLAB_BENCH_REPEATS": str(plan.repeats_per_block)},
            repeats=plan.repeats_per_block,
        )
        return SpawnMeasurement(
            value=metric_value(bench, metric_name),
            samples=extract_repeated_values(bench, metric_name),
            bench=bench,
        )

    return run_interleaved(_spawn, plan)
