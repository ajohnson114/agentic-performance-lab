"""Differential flame graph generator.

Compares two py-spy or perf flame graph sample sets and produces a
red/blue differential SVG where:
  - Red = functions that got hotter (increased CPU share)
  - Blue = functions that got cooler (decreased CPU share)
  - Width = absolute sample count in the "after" profile
"""
from __future__ import annotations

import re
from pathlib import Path


def compute_diff_stacks(
    before_summary: dict,
    after_summary: dict,
) -> list[dict]:
    """Compute per-function CPU share deltas from profiler summaries.

    Accepts pyspy or linux_perf summary dicts with 'hotspots' lists.
    Returns sorted list of {function, before_pct, after_pct, delta, direction}.
    """
    before_funcs = _extract_func_pcts(before_summary)
    after_funcs = _extract_func_pcts(after_summary)

    all_funcs = set(before_funcs) | set(after_funcs)
    diffs: list[dict] = []

    for func in all_funcs:
        bpct = before_funcs.get(func, 0.0)
        apct = after_funcs.get(func, 0.0)
        delta = apct - bpct

        if abs(delta) < 0.5:
            continue

        direction = "hotter" if delta > 0 else "cooler"
        diffs.append({
            "function": func,
            "before_pct": round(bpct, 1),
            "after_pct": round(apct, 1),
            "delta": round(delta, 1),
            "direction": direction,
        })

    diffs.sort(key=lambda d: abs(d["delta"]), reverse=True)
    return diffs


def generate_diff_svg(
    diffs: list[dict],
    output_path: Path,
    title: str = "Differential Flame Graph",
) -> Path | None:
    """Generate a simple differential flame graph SVG.

    Each function is rendered as a horizontal bar, colored red (hotter)
    or blue (cooler), sized by the absolute delta.
    """
    if not diffs:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    bar_height = 24
    padding = 4
    max_bars = 30
    top_diffs = diffs[:max_bars]
    chart_width = 800
    margin_left = 10
    margin_top = 40

    svg_height = margin_top + len(top_diffs) * (bar_height + padding) + 20
    max_delta = max(abs(d["delta"]) for d in top_diffs) if top_diffs else 1.0

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{chart_width}" height="{svg_height}">',
        f'<rect width="{chart_width}" height="{svg_height}" fill="#1a1a2e"/>',
        f'<text x="{chart_width // 2}" y="24" text-anchor="middle" fill="#eee" '
        f'font-family="monospace" font-size="14" font-weight="bold">{_escape(title)}</text>',
    ]

    for i, d in enumerate(top_diffs):
        y = margin_top + i * (bar_height + padding)
        bar_width = max(20, int((abs(d["delta"]) / max_delta) * (chart_width - margin_left * 2 - 200)))

        if d["direction"] == "hotter":
            # Red gradient intensity based on delta
            intensity = min(255, int(100 + 155 * abs(d["delta"]) / max_delta))
            fill = f"rgb({intensity}, {max(0, intensity - 150)}, {max(0, intensity - 180)})"
        else:
            intensity = min(255, int(100 + 155 * abs(d["delta"]) / max_delta))
            fill = f"rgb({max(0, intensity - 180)}, {max(0, intensity - 150)}, {intensity})"

        # Bar
        lines.append(
            f'<rect x="{margin_left}" y="{y}" width="{bar_width}" '
            f'height="{bar_height}" fill="{fill}" rx="3"/>'
        )

        # Label
        sign = "+" if d["delta"] > 0 else ""
        label = f'{d["function"]}  {sign}{d["delta"]:.1f}pp'
        text_x = margin_left + bar_width + 8
        text_y = y + bar_height // 2 + 5
        lines.append(
            f'<text x="{text_x}" y="{text_y}" fill="#ddd" '
            f'font-family="monospace" font-size="12">{_escape(label)}</text>'
        )

    # Legend
    legend_y = svg_height - 14
    lines.append(
        f'<text x="{margin_left}" y="{legend_y}" fill="#888" '
        f'font-family="monospace" font-size="10">'
        f'Red = hotter (increased CPU%)  |  Blue = cooler (decreased CPU%)</text>'
    )

    lines.append("</svg>")
    svg_content = "\n".join(lines)
    output_path.write_text(svg_content, encoding="utf-8")
    return output_path


def _extract_func_pcts(summary: dict) -> dict[str, float]:
    """Extract function -> CPU percentage from profiler summary."""
    funcs: dict[str, float] = {}

    # Try pyspy format
    for profiler_key in ("pyspy", "linux_perf"):
        data = summary.get(profiler_key, summary)
        hotspots = data.get("hotspots", [])
        for h in hotspots:
            func = h.get("function", "")
            pct = h.get("pct", 0.0)
            if func and pct > 0:
                funcs[func] = funcs.get(func, 0.0) + pct

    return funcs


def _escape(text: str) -> str:
    """Escape text for SVG XML."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
