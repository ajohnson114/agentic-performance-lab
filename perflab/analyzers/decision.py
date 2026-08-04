"""The accept/reject decision: *is candidate B better than incumbent A?*

This module owns that question end to end. It used to be answered in three
places with three different behaviors -- the knob-search path in
``orchestrator.py`` (a bare ratio), the agent beam search in
``optimizers/phases/evaluate.py`` (statistically gated), and the regression
check in ``ci.py`` (gated, with the mode flipped) -- and the gate that was
added to two of them never reached the third. Every one of those call sites now
depends on this module, so there is exactly one rule to change and one place to
test it.

Why a strategy object rather than flags
---------------------------------------
"Better" is a *statistical* question with more than one defensible answer, and
which answer is right depends on how the measurement was taken:

* a 2-repeat screening pass is a ranking step, so it wants the cheap
  directional test and nothing more (:class:`ToleranceOnly`);
* a full benchmark of two independently-measured programs wants the
  non-overlapping-interval test (:class:`NonOverlappingCI`);
* a block-interleaved A/B run, where measurements alternate so that drift hits
  both arms equally, wants a *paired* test on the per-pair differences -- which
  is strictly more powerful, because the between-block variance cancels.

Those are different rules, not different parameter values, so they are
different objects behind :class:`DecisionRule`. A boolean like the original
``noise_gate=`` could only ever express two of them, and a third would have
turned it into an if/else chain in the one function everything depends on.

The paired rule
---------------
:class:`Comparison` carries ``pairs`` -- the ``(candidate_i, incumbent_i)``
measurements from a block-interleaved run -- and exposes ``paired_diffs``
(``d_i = b_i - a_i``, sign-normalized so positive always means "better").
:class:`PairedDifference` consumes them, and it fit the protocol exactly as
predicted: a new class with a ``decide`` method plus one :func:`register_rule`
call, no change to the protocol, to :class:`ImprovementVerdict` (which already
carried ``paired`` and ``p_value``), or to any of the three call sites.
``perflab.runners.paired`` produces the pairs; rules that predate it ignore the
field, so nothing changes for a task that does not select the paired rule.

Configuration
-------------
``constraints.decision_rule`` names the rule (default
``"non_overlapping_ci"``). ``constraints.noise_gate: false`` remains supported
and means exactly ``decision_rule: tolerance_only``; it takes precedence, since
it is the explicit "turn the gate off" knob.

Selecting ``decision_rule: paired_difference`` also changes how the
*measurement* is taken -- ``phases/evaluate.py`` reads the resolved rule and
switches the authoritative re-benchmark to the interleaved runner -- which is
why the rule name doubles as the switch and no separate config field exists.

No third-party statistics
-------------------------
Every test here is stdlib-only, deliberately. An earlier
``is_statistically_significant()`` helper was deleted because it depended on
scipy and returned "significant" when scipy was missing: it failed OPEN, which
is the one failure mode a gate must never have. The exact signed-rank
distribution below is a dozen lines of dynamic programming, and when a test
cannot run at all the rule degrades to a *stricter-or-equal* unpaired rule with
``verified=False`` rather than waving the candidate through.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from perflab.analyzers.bench_stats import compute_bench_stats, relative_ci_margin

Mode = Literal["maximize", "minimize"]


# --- Inputs -----------------------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    """Everything measured about one candidate-vs-incumbent comparison.

    ``candidate``/``incumbent`` are the point estimates being compared (the
    metric values). ``tolerance`` is ``constraints.regression_tolerance``, the
    materiality floor every rule applies.

    Three sample shapes are supported, in increasing order of statistical
    power; a rule uses whichever it understands and ignores the rest:

    * nothing -- only the point estimates are known;
    * ``candidate_samples`` / ``incumbent_samples`` -- the per-repeat arrays
      behind each point estimate, measured independently (see
      ``bench_stats.extract_repeated_values``). Either side may be absent;
    * ``pairs`` -- ``(candidate_i, incumbent_i)`` from a block-interleaved A/B
      run, where the two arms were measured under the same conditions at the
      same moment. Not produced yet; see the module docstring.

    Samples need not be in the metric's own units: every rule here uses them
    for *relative* dispersion only, which is why a ``times_ms`` array can back
    a ``tflops.median`` metric (see ``extract_repeated_values``).
    """

    candidate: float
    incumbent: float
    mode: Mode
    tolerance: float
    candidate_samples: Sequence[float] | None = None
    incumbent_samples: Sequence[float] | None = None
    pairs: Sequence[tuple[float, float]] | None = None

    @property
    def observed(self) -> float:
        """Improvement over the incumbent as a fraction (>0 means better)."""
        return relative_improvement(self.candidate, self.incumbent, self.mode)

    @property
    def paired_diffs(self) -> list[float]:
        """``d_i`` per pair, sign-normalized so positive always means better.

        Empty when the comparison carries no pairs. Provided here rather than
        in the (future) paired rule so that the sign convention -- positive is
        better, in both modes -- is defined once, next to `observed`.
        """
        if not self.pairs:
            return []
        sign = 1.0 if self.mode == "maximize" else -1.0
        return [sign * (cand - inc) for cand, inc in self.pairs]

    @property
    def is_paired(self) -> bool:
        """True when there are enough pairs for a paired test to be meaningful."""
        return self.pairs is not None and len(self.pairs) >= 2


@dataclass(frozen=True)
class ImprovementVerdict:
    """Why a candidate was (or was not) accepted over the incumbent.

    `improved` is the accept decision. Everything else exists so the caller can
    say *which* bar the candidate failed -- "did not beat the incumbent" and
    "beat it, but by less than this machine can measure" are completely
    different messages for the user.
    """

    improved: bool
    beats_tolerance: bool       # cleared the materiality floor (the old gate)
    observed: float             # relative improvement over the incumbent (fraction)
    required: float             # relative improvement actually needed = max(tolerance, noise_required)
    tolerance: float            # regression_tolerance, the materiality floor
    # Noise-derived *magnitude* requirement. None when untested, and also None
    # for rules whose second bar is a significance level rather than a
    # magnitude (see PairedDifference) -- there is no honest number to put here
    # in that case, and inventing one would misreport `required`.
    noise_required: float | None
    # The dispersion that set the resolution: the noisier arm's CV for the
    # interval rule, the CV of the paired differences for the paired rule --
    # in each case the spread the test actually had to overcome.
    cv: float | None            # None when untested
    n: int | None               # samples on the thinner side (pairs, if paired); None when untested
    verified: bool              # True iff the statistical test actually ran
    reason: str                 # "" when improved
    # Which strategy produced this verdict -- worth recording, since the answer
    # depends on it (the fast screen deliberately uses a weaker rule).
    rule: str = ""
    # Set by rules that consume `Comparison.pairs`; a paired rule reports
    # `paired=True` and, for Wilcoxon/paired-t, the achieved significance.
    # Interval rules leave both alone.
    paired: bool = False
    p_value: float | None = None


# --- The strategy protocol --------------------------------------------------


class DecisionRule(Protocol):
    """A rule for turning a :class:`Comparison` into an accept/reject verdict.

    Implementations must be stateless and safe to share (the module-level
    singletons below are reused across every call site and thread).

    Contract, which the callers rely on and every rule must honor:

    * the materiality floor is never *weakened*: a rule may demand more than
      ``comparison.tolerance``, never less;
    * a verdict reports ``verified=True`` only if its statistical test actually
      ran; degrading to the point-estimate comparison because samples were
      missing must be visible in the verdict, never silent;
    * ``reason`` is non-empty whenever ``improved`` is False, and explains
      which bar was missed.
    """

    name: str

    def decide(self, comparison: Comparison) -> ImprovementVerdict:
        """Return the verdict for ``comparison``."""
        ...


# --- Shared arithmetic ------------------------------------------------------


def relative_improvement(new: float, best: float, mode: Mode) -> float:
    """Improvement of ``new`` over ``best`` as a fraction of ``best`` (>0 = better)."""
    if best == 0:
        return math.inf if (new > 0 if mode == "maximize" else new < 0) else -math.inf
    if mode == "maximize":
        return (new - best) / abs(best)
    return (best - new) / abs(best)


def _beats_tolerance(comparison: Comparison) -> bool:
    """The materiality floor: the point estimate must clear ``tolerance``."""
    if comparison.mode == "maximize":
        return comparison.candidate > comparison.incumbent * (1.0 + comparison.tolerance)
    return comparison.candidate < comparison.incumbent * (1.0 - comparison.tolerance)


def _tolerance_reason(observed: float, tol: float) -> str:
    return (
        f"did not beat the incumbent by the {tol:.1%} regression tolerance "
        f"(observed {observed:+.2%})"
    )


def _point_estimate_verdict(
    comparison: Comparison, rule: str, *, beats_tolerance: bool,
) -> ImprovementVerdict:
    """Verdict from the point estimates alone -- the historical ratio test.

    Used both by :class:`ToleranceOnly` (which is only ever this) and by
    :class:`NonOverlappingCI` when there is nothing to run its test on. Always
    ``verified=False``: no variance information was consulted.
    """
    observed = comparison.observed
    return ImprovementVerdict(
        improved=beats_tolerance,
        beats_tolerance=beats_tolerance,
        observed=observed,
        required=comparison.tolerance,
        tolerance=comparison.tolerance,
        noise_required=None,
        cv=None,
        n=None,
        verified=False,
        reason="" if beats_tolerance else _tolerance_reason(observed, comparison.tolerance),
        rule=rule,
    )


# --- Rules ------------------------------------------------------------------


class ToleranceOnly:
    """Materiality floor only -- the historical bare ratio test.

    ``maximize: new > best * (1 + tol)``; ``minimize: new < best * (1 - tol)``.
    No variance is consulted even when samples are present, so verdicts always
    report ``verified=False``.

    This is not a legacy shim, it is the right rule in two situations:

    * the agent's fast screen, which benchmarks with ``repeats=2`` purely to
      rank candidates. Vetoing there on variance would throw away genuine wins
      before they are ever measured properly -- the full re-bench that follows
      is the authoritative gate, and it uses the real rule;
    * a deterministic metric (instruction counts, bytes moved, allocation
      counts), where run-to-run spread is zero and an interval test only adds
      the risk of dividing by it. This is what ``noise_gate: false`` selects.
    """

    name = "tolerance_only"

    def decide(self, comparison: Comparison) -> ImprovementVerdict:
        return _point_estimate_verdict(
            comparison, self.name, beats_tolerance=_beats_tolerance(comparison),
        )


class NonOverlappingCI:
    """Materiality floor AND non-overlapping 95% confidence intervals.

    THE RULE (two bars, both must clear):

      1. Materiality -- the point estimate must beat the incumbent by
         ``tolerance`` (``regression_tolerance``). Unchanged from the original
         bare ratio: a 0.5% win is not worth a patch even if it is real.
      2. Separation -- the candidate's 95% confidence interval must not
         overlap the incumbent's, in the improving direction:

             maximize:  new * (1 - m_new)  >  best * (1 + m_best)
             minimize:  new * (1 + m_new)  <  best * (1 - m_best)

         where ``m = t * CV / sqrt(n)`` is the 95% CI half-width expressed as
         a *fraction* of the measurement (`bench_stats.relative_ci_margin`).

    Why non-overlapping CIs rather than ``improvement > k * CV``: the k*CV form
    ignores the sample count, so it punishes a task that pays for 50 repeats
    exactly as hard as one that ran 3, and it gives the user no lever other
    than "buy a quieter machine". The interval form is n-aware, so more repeats
    genuinely buy the ability to resolve smaller wins -- the correct incentive,
    and the one that makes a 2% gate an honest promise instead of a wish.

    Why non-overlap rather than a Welch t-test at p<0.05: non-overlapping 95%
    CIs correspond to roughly p<0.01, and the extra strictness is deliberate.
    A beam search runs this test dozens of times per run (n_candidates x
    iterations) and keeps only the winners -- a textbook multiple-comparisons
    machine. At p<0.05 per test, a 30-comparison run is more likely than not to
    promote at least one pure noise event, and the beam then chases it.

    Working in relative terms (rather than comparing raw CI bounds) is what
    lets the samples come from a different-but-proportional quantity, e.g.
    ``times_ms`` alongside a ``tflops.median`` metric, or a per-repeat array
    against a metric reported as a median/p95. See `extract_repeated_values`.

    FALLBACK, explicit and never silent: if either side has fewer than 2 usable
    samples (or the values are non-positive), bar 2 cannot be evaluated. The
    verdict then contains exactly the old bare-ratio decision with
    ``verified=False``, so callers can log "accepted without variance
    verification" rather than pretending the check ran. The gate is never
    *weakened* by the absence of samples -- bar 1 still applies -- and a
    one-sided comparison is supported too: with samples on only one side, the
    other side is treated as a point estimate (zero-width interval), which is
    strictly the old behavior plus one real constraint.
    """

    name = "non_overlapping_ci"

    def decide(self, comparison: Comparison) -> ImprovementVerdict:
        beats_tolerance = _beats_tolerance(comparison)
        observed = comparison.observed
        tol = comparison.tolerance

        stats_new = (
            compute_bench_stats(list(comparison.candidate_samples))
            if comparison.candidate_samples else None
        )
        stats_best = (
            compute_bench_stats(list(comparison.incumbent_samples))
            if comparison.incumbent_samples else None
        )
        testable = (
            (stats_new is not None or stats_best is not None)
            and comparison.incumbent > 0
            and comparison.candidate > 0
        )
        if not testable:
            return _point_estimate_verdict(
                comparison, self.name, beats_tolerance=beats_tolerance,
            )

        m_new = relative_ci_margin(stats_new) if stats_new is not None else 0.0
        m_best = relative_ci_margin(stats_best) if stats_best is not None else 0.0

        # Smallest relative improvement whose interval clears the incumbent's.
        if comparison.mode == "maximize":
            noise_required = (
                math.inf if m_new >= 1.0 else (1.0 + m_best) / (1.0 - m_new) - 1.0
            )
        else:
            noise_required = 1.0 - (1.0 - m_best) / (1.0 + m_new)

        required = max(tol, noise_required)
        improved = beats_tolerance and observed > noise_required

        cvs = [s.cv for s in (stats_new, stats_best) if s is not None]
        ns = [s.n for s in (stats_new, stats_best) if s is not None]
        cv = max(cvs)   # the noisier side sets the resolution
        n = min(ns)

        reason = ""
        if not beats_tolerance:
            reason = _tolerance_reason(observed, tol)
        elif not improved:
            need = (
                "unresolvable at this variance" if math.isinf(required)
                else f"{required:+.2%}"
            )
            reason = (
                f"improvement within noise (CV={cv:.1%} at n={n}; "
                f"observed {observed:+.2%}, need {need})"
            )

        return ImprovementVerdict(
            improved=improved,
            beats_tolerance=beats_tolerance,
            observed=observed,
            required=required,
            tolerance=tol,
            noise_required=noise_required,
            cv=cv,
            n=n,
            verified=True,
            reason=reason,
            rule=self.name,
        )


# --- Exact signed-rank test (stdlib only; see the module docstring) ---------

#: Above this many nonzero pairs the exact enumeration is replaced by the
#: normal approximation. The DP below is O(n^3), so 50 pairs is ~10^5
#: operations -- far past anything a benchmark would actually run, and the
#: cutoff exists only so a pathological caller cannot make the gate slow.
_EXACT_SIGNED_RANK_MAX_N = 50


def _doubled_midranks(magnitudes: Sequence[float]) -> list[int]:
    """Mid-ranks of ``magnitudes``, doubled so tied half-ranks stay integral.

    Ties are given the average of the ranks they span, which is what keeps the
    enumeration below an *exact conditional* test rather than an approximation:
    conditioning on the observed multiset of |d| values, the null distribution
    is generated by the 2^n sign assignments, ties and all.
    """
    order = sorted(range(len(magnitudes)), key=lambda i: magnitudes[i])
    ranks = [0] * len(magnitudes)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and magnitudes[order[j + 1]] == magnitudes[order[i]]:
            j += 1
        doubled = (i + 1) + (j + 1)  # 2 * mean of the 1-based ranks i+1..j+1
        for k in range(i, j + 1):
            ranks[order[k]] = doubled
        i = j + 1
    return ranks


def signed_rank_p_greater(diffs: Sequence[float]) -> tuple[float, int] | None:
    """One-sided Wilcoxon signed-rank p-value for ``H1: the diffs run positive``.

    Returns ``(p_value, n_nonzero)``, or ``None`` when the test cannot be run
    at all (a non-finite difference) -- which callers must treat as "no test",
    never as "not significant" and never as "significant".

    Exact for ``n <= _EXACT_SIGNED_RANK_MAX_N``: ``counts`` is the full null
    distribution of ``W+`` over all 2^n sign assignments, built by subset-sum
    DP over the doubled ranks, so ``p = P(W+ >= observed)`` is a ratio of exact
    integer counts. Above that, the standard normal approximation with a tie
    correction and a continuity correction.

    Zeros are dropped (the classical Wilcoxon treatment) and the test runs on
    what remains; ``n_nonzero`` is reported so the caller can see how much
    evidence was actually available. All-zero differences give ``p = 1.0``:
    identical measurements are the absence of evidence, not evidence of
    improvement.
    """
    values = list(diffs)
    if any(not math.isfinite(d) for d in values):
        return None
    nonzero = [d for d in values if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return 1.0, 0

    ranks = _doubled_midranks([abs(d) for d in nonzero])
    w_plus = sum(r for r, d in zip(ranks, nonzero, strict=True) if d > 0)
    total = sum(ranks)

    if n <= _EXACT_SIGNED_RANK_MAX_N:
        counts = [0] * (total + 1)
        counts[0] = 1
        for r in ranks:
            for s in range(total, r - 1, -1):
                counts[s] += counts[s - r]
        tail = sum(counts[w_plus:])
        return tail / float(1 << n), n

    mean = total / 2.0
    variance = sum(r * r for r in ranks) / 4.0
    if variance <= 0:
        return 1.0, n
    # Continuity correction of 1 doubled unit = 0.5 in rank units.
    z = (w_plus - 1.0 - mean) / math.sqrt(variance)
    return 0.5 * math.erfc(z / math.sqrt(2.0)), n


def min_attainable_p(n_pairs: int) -> float:
    """Smallest one-sided signed-rank p-value reachable with ``n_pairs`` pairs.

    ``2**-n``: every difference would have to point the same way, and even then
    that single sign assignment is one of 2^n equally likely null outcomes.
    Worth exposing, because it is the reason the paired rule needs at least 5
    pairs -- at 4 the best possible p is 0.0625 and the gate could never open.
    """
    return 0.5 ** max(n_pairs, 0)


# --- Rules (paired) ---------------------------------------------------------


class PairedDifference:
    """Materiality floor AND an exact signed-rank test on the paired differences.

    Applies to a *block-interleaved* measurement (``perflab.runners.paired``),
    where the incumbent and the candidate were spawned alternately in a
    counterbalanced ABBA order rather than minutes apart. That design is what
    makes this rule legitimate: the pairs are only exchangeable if both arms
    saw the same machine conditions, and it is what makes it powerful, because
    slow thermal/clock drift is common-mode within a pair and cancels in
    ``d_i``.

    THE RULE (two bars, both must clear):

      1. Materiality -- identical to every other rule here: the point estimate
         must beat the incumbent by ``tolerance``. Shared code
         (:func:`_beats_tolerance`), so the floor cannot drift between rules.
      2. Significance -- an exact one-sided Wilcoxon signed-rank test on
         ``comparison.paired_diffs`` must reach ``p < alpha`` (default 0.05).

    Why signed-rank rather than a paired t-test: a paired t assumes the
    differences are near-normal, and benchmark blocks are not -- an interrupt
    storm or a scheduler migration lands one block far out in a one-sided tail,
    and the t-statistic will happily follow it in either direction. The
    signed-rank statistic uses only the ordering of ``|d_i|``, so a single wild
    block can contribute at most the top rank. At the sample sizes involved
    (5-10 pairs) the exact permutation distribution is also cheap and *exact*,
    while a t-test at n=6 is leaning hard on an assumption nobody checked.

    Why alpha stays at 0.05 when :class:`NonOverlappingCI` deliberately runs
    tighter (~p<0.01): with n pairs the smallest attainable p is ``2**-n``, so
    an alpha of 0.01 would be unreachable below 7 pairs and would turn the gate
    into a permanent "no" for any affordable design. The multiple-comparisons
    pressure the CI rule guards against is answered here by the other bar
    instead: a noise event has to clear the materiality floor on the point
    estimate *and* produce a consistently-signed difference across
    independently-spawned interleaved blocks, and the second is exactly what a
    transient does not do.

    What the verdict reports, and why it is not the interval rule's shape:
    ``noise_required`` is None and ``required`` is just ``tolerance``, because
    this rule's second bar is a significance level, not a magnitude -- there is
    no "you needed +4.1%" number to state, and fabricating one would make
    ``required`` a lie. ``cv`` carries the CV of the paired *differences*
    (``stdev(d) / |incumbent|``), which is the dispersion this design actually
    had to overcome and the number worth comparing against a per-arm CV to see
    what pairing bought.

    FAIL CLOSED, never open. If the comparison carries no usable pairs, or the
    signed-rank test cannot run, this rule does not degrade to "accept": it
    delegates to ``fallback`` (:data:`NON_OVERLAPPING_CI` by default, i.e. the
    stricter unpaired gate the project already shipped) and stamps the result
    ``verified=False`` with ``paired=False``, so the caller can say the paired
    test did not happen. There is direct history behind that emphasis -- see
    the module docstring's note on the scipy-dependent helper that failed open.
    """

    name = "paired_difference"

    #: Enough pairs that ``2**-n`` can fall under alpha. Below this the test is
    #: arithmetically incapable of rejecting, so running it would produce a
    #: "not significant" verdict that says nothing about the candidate.
    min_pairs = 5

    def __init__(
        self, alpha: float = 0.05, fallback: DecisionRule | None = None,
    ) -> None:
        self.alpha = alpha
        self._fallback = fallback

    @property
    def fallback(self) -> DecisionRule:
        # Resolved lazily: NON_OVERLAPPING_CI is defined below this class.
        return self._fallback if self._fallback is not None else NON_OVERLAPPING_CI

    def _unpaired(self, comparison: Comparison, note: str) -> ImprovementVerdict:
        """The fail-closed path: the stricter unpaired gate, honestly labelled."""
        verdict = self.fallback.decide(comparison)
        reason = verdict.reason
        if not verdict.improved:
            reason = f"{reason} [{note}]" if reason else note
        return dataclasses.replace(
            verdict,
            # The *paired* test did not run, whatever the fallback managed.
            verified=False,
            paired=False,
            p_value=None,
            rule=self.name,
            reason=reason,
        )

    def decide(self, comparison: Comparison) -> ImprovementVerdict:
        pairs = comparison.pairs or ()
        if len(pairs) < self.min_pairs:
            return self._unpaired(comparison, (
                f"paired test not run: {len(pairs)} pair(s), needs "
                f"{self.min_pairs} (min attainable p at {len(pairs)} pairs is "
                f"{min_attainable_p(len(pairs)):.3f})"
            ))

        diffs = comparison.paired_diffs  # sign-normalized: positive == better
        result = signed_rank_p_greater(diffs)
        if result is None:
            return self._unpaired(
                comparison, "paired test not run: non-finite paired difference",
            )
        p_value, n_nonzero = result

        beats_tolerance = _beats_tolerance(comparison)
        significant = p_value < self.alpha
        improved = beats_tolerance and significant

        cv: float | None = None
        if len(diffs) >= 2 and comparison.incumbent != 0:
            spread = math.sqrt(
                sum((d - sum(diffs) / len(diffs)) ** 2 for d in diffs)
                / (len(diffs) - 1)
            )
            cv = spread / abs(comparison.incumbent)

        reason = ""
        if not beats_tolerance:
            reason = _tolerance_reason(comparison.observed, comparison.tolerance)
        elif not significant:
            floor = min_attainable_p(n_nonzero)
            detail = (
                f"paired signed-rank p={p_value:.3f} >= alpha={self.alpha:g} "
                f"over {len(diffs)} interleaved pair(s)"
            )
            if floor >= self.alpha:
                detail += (
                    f"; {n_nonzero} non-tied pair(s) cannot reach it "
                    f"(floor p={floor:.3f}) -- run more blocks"
                )
            elif cv is not None:
                detail += f" (paired difference CV={cv:.1%})"
            reason = f"improvement not significant under pairing ({detail})"

        return ImprovementVerdict(
            improved=improved,
            beats_tolerance=beats_tolerance,
            observed=comparison.observed,
            required=comparison.tolerance,
            tolerance=comparison.tolerance,
            noise_required=None,  # see the class docstring: a p-value, not a magnitude
            cv=cv,
            n=len(diffs),
            verified=True,
            reason=reason,
            rule=self.name,
            paired=True,
            p_value=p_value,
        )


# --- Registry and selection -------------------------------------------------

#: Shared, stateless singletons. Import these rather than constructing rules.
TOLERANCE_ONLY: DecisionRule = ToleranceOnly()
NON_OVERLAPPING_CI: DecisionRule = NonOverlappingCI()
#: Opt-in: selecting it also switches the authoritative measurement in
#: phases/evaluate.py to the block-interleaved runner. Stateless like the rest
#: -- `alpha` and `fallback` are fixed at construction and never mutated.
PAIRED_DIFFERENCE: DecisionRule = PairedDifference()

#: What a task gets when it says nothing: the full statistical gate. NOT the
#: paired rule -- pairing changes how the measurement is taken (roughly double
#: the authoritative benchmark's wall clock), so it stays something a task asks
#: for rather than something it discovers by surprise.
DEFAULT_RULE: DecisionRule = NON_OVERLAPPING_CI

#: The rule the agent's fast screen uses. Named so the exception is legible at
#: the call site: screening ranks candidates, it does not decide them.
SCREENING_RULE: DecisionRule = TOLERANCE_ONLY

_RULES: dict[str, DecisionRule] = {
    NON_OVERLAPPING_CI.name: NON_OVERLAPPING_CI,
    PAIRED_DIFFERENCE.name: PAIRED_DIFFERENCE,
    TOLERANCE_ONLY.name: TOLERANCE_ONLY,
}


def is_paired_rule(rule: DecisionRule) -> bool:
    """True when ``rule`` needs a block-interleaved measurement to do its job.

    The measurement side (``phases/evaluate.py``) has to know whether to pay
    for interleaving *before* it benchmarks anything, and asking the rule is
    better than hard-coding a name there: a future paired rule registers itself
    and the measurement follows, with no second place to update.
    """
    return isinstance(rule, PairedDifference)


def register_rule(rule: DecisionRule) -> None:
    """Make ``rule`` selectable by name from ``constraints.decision_rule``."""
    _RULES[rule.name] = rule


def rule_names() -> list[str]:
    """Names accepted by :func:`get_rule` / ``constraints.decision_rule``."""
    return sorted(_RULES)


def get_rule(name: str) -> DecisionRule:
    """Look up a rule by name, raising ``ValueError`` on an unknown one."""
    try:
        return _RULES[name]
    except KeyError:
        raise ValueError(
            f"unknown decision_rule {name!r}; valid options: {', '.join(rule_names())}"
        ) from None


def rule_for_constraints(constraints: Any) -> DecisionRule:
    """Resolve the rule a task's ``constraints`` select.

    ``noise_gate: false`` wins over ``decision_rule`` -- it is the explicit
    "turn the gate off" switch, and silently keeping a statistical rule after
    the user disabled the gate would be the worse surprise.

    Attributes are read defensively so duck-typed constraint objects (test
    doubles, configs predating a field) keep working, and a ``decision_rule``
    that is not a non-empty string falls back to the default rather than
    raising. Typos in real configs are still caught, and caught earlier:
    ``TaskSpec.load`` runs the name through :func:`get_rule` at parse time.
    """
    if not bool(getattr(constraints, "noise_gate", True)):
        return TOLERANCE_ONLY
    name = getattr(constraints, "decision_rule", None)
    if isinstance(name, str) and name:
        return get_rule(name)
    return DEFAULT_RULE


# --- Entry point ------------------------------------------------------------


def decide(
    comparison: Comparison, rule: DecisionRule | None = None,
) -> ImprovementVerdict:
    """Apply ``rule`` (default :data:`DEFAULT_RULE`) to ``comparison``."""
    return (rule or DEFAULT_RULE).decide(comparison)
