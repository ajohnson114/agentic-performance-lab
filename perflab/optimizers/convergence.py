"""Convergence detection for the agent optimization loop.

Tracks consecutive failures and diminishing improvements to decide
when further iterations are unlikely to produce meaningful gains.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConvergenceDetector:
    """Detects when the agent loop should stop early."""

    max_consecutive_failures: int = 5
    min_relative_improvement: float = 0.03  # 3%

    _consecutive_failures: int = field(default=0, init=False)
    _recent_improvements: list[float] = field(default_factory=list, init=False)

    def record_improvement(self, relative_delta: float) -> None:
        """Record that a candidate was accepted with the given relative improvement."""
        self._consecutive_failures = 0
        self._recent_improvements.append(relative_delta)

    def record_failure(self) -> None:
        """Record that no candidate was accepted this iteration."""
        self._consecutive_failures += 1

    def should_stop(self) -> tuple[bool, str]:
        """Check if the loop should stop early.

        Returns (should_stop, human-readable reason).
        """
        # Condition 1: too many consecutive failures
        if self._consecutive_failures >= self.max_consecutive_failures:
            return True, (
                f"Stopping: {self._consecutive_failures} consecutive iterations "
                f"with no improvement found"
            )

        # Condition 2: last 2 improvements were both tiny
        if len(self._recent_improvements) >= 2:
            last_two = self._recent_improvements[-2:]
            if all(d < self.min_relative_improvement for d in last_two):
                pcts = [f"{d:.1%}" for d in last_two]
                return True, (
                    f"Stopping: improvements converging "
                    f"(last two gains were {pcts[0]} and {pcts[1]})"
                )

        return False, ""
