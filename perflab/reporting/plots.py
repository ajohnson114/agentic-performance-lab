from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_metric_history(
    out_png: Path,
    iters: list[int],
    values: list[float],
    metric_name: str,
    baseline_val: float | None = None,
) -> None:
    plt.figure()
    plt.plot(iters, values, marker="o")
    if baseline_val is not None:
        plt.axhline(y=baseline_val, color="gray", linestyle="--", linewidth=1, label="baseline")
        plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} over iterations")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()


def compute_pareto_frontier(
    points: list[dict],
    primary_mode: str = "maximize",
    secondary_mode: str = "minimize",
) -> list[dict]:
    """Compute the Pareto-optimal frontier from a list of points.

    Each point dict must have 'primary' and 'secondary' keys.
    Returns the subset of points on the Pareto frontier.
    """
    if not points:
        return []

    # For Pareto: a point dominates another if it's better or equal on both metrics
    # and strictly better on at least one.
    def dominates(a: dict, b: dict) -> bool:
        a_p, a_s = a["primary"], a["secondary"]
        b_p, b_s = b["primary"], b["secondary"]

        if primary_mode == "maximize":
            p_ge = a_p >= b_p
            p_gt = a_p > b_p
        else:
            p_ge = a_p <= b_p
            p_gt = a_p < b_p

        if secondary_mode == "maximize":
            s_ge = a_s >= b_s
            s_gt = a_s > b_s
        else:
            s_ge = a_s <= b_s
            s_gt = a_s < b_s

        return p_ge and s_ge and (p_gt or s_gt)

    frontier = []
    for p in points:
        if not any(dominates(other, p) for other in points if other is not p):
            frontier.append(p)

    # Sort frontier by primary metric
    reverse = primary_mode == "maximize"
    frontier.sort(key=lambda p: p["primary"], reverse=reverse)
    return frontier


def plot_pareto_frontier(
    out_png: Path,
    points: list[dict],
    frontier: list[dict],
    primary_name: str,
    secondary_name: str,
    primary_mode: str = "maximize",
    secondary_mode: str = "minimize",
) -> None:
    """Plot all points with the Pareto frontier highlighted.

    Each point dict must have 'primary', 'secondary', and optionally 'label'.
    """
    if not points:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot all points
    all_x = [p["primary"] for p in points]
    all_y = [p["secondary"] for p in points]
    ax.scatter(all_x, all_y, c="#888888", alpha=0.5, s=40, label="All iterations", zorder=2)

    # Plot frontier points
    if frontier:
        front_x = [p["primary"] for p in frontier]
        front_y = [p["secondary"] for p in frontier]
        ax.scatter(front_x, front_y, c="#e74c3c", s=80, marker="D", label="Pareto frontier", zorder=3)

        # Draw frontier line
        if len(frontier) > 1:
            ax.plot(front_x, front_y, c="#e74c3c", linewidth=1.5, linestyle="--", alpha=0.7, zorder=2)

    # Labels for frontier points
    for p in frontier:
        label = p.get("label", "")
        if label:
            ax.annotate(label, (p["primary"], p["secondary"]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=8, color="#e74c3c")

    ax.set_xlabel(f"{primary_name} ({'↑' if primary_mode == 'maximize' else '↓'})")
    ax.set_ylabel(f"{secondary_name} ({'↑' if secondary_mode == 'maximize' else '↓'})")
    ax.set_title("Pareto Frontier: Multi-Objective Optimization")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
