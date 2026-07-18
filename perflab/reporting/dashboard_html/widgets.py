"""Small shared rendering helpers: bar charts, metric pills, formatters."""
from __future__ import annotations

import html as _html
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeGuard


def _render_bar_chart(
    parts: list[str],
    items: list[dict],
    esc,
    color: str = "#3b82f6",
) -> None:
    """Render a bar chart from items: [{name, pct, total_ms?, count?}]."""
    if not items:
        return
    max_pct = max(it.get("pct", 0) for it in items) or 1
    for it in items:
        pct = it.get("pct", 0)
        bar_width = pct / max_pct * 100
        name = it["name"]
        if len(name) > 45:
            name = "..." + name[-42:]
        meta_parts = []
        if "total_ms" in it:
            meta_parts.append(f"{it['total_ms']:.1f}ms")
        if "count" in it:
            meta_parts.append(f"x{it['count']}")
        meta = " ".join(meta_parts)
        parts.append(
            f'<div class="op-bar-container">'
            f'<span class="op-bar-label" title="{esc(it["name"])}">{esc(name)}</span>'
            f'<div class="op-bar-track"><div class="op-bar-fill" style="width: {bar_width:.1f}%; background: {color}"></div></div>'
            f'<span class="op-bar-pct">{pct:.1f}%</span>'
            f'<span class="op-bar-meta">{esc(meta)}</span>'
            f'</div>'
        )


def _metric_pill(
    parts: list[str],
    label: str,
    val_str: str,
    baseline_val: float | None,
    current_val: float | None,
    lower_is_better: bool = True,
) -> None:
    """Render a metric pill badge with optional delta from baseline."""
    delta_html = ""
    if baseline_val is not None and current_val is not None and baseline_val != 0:
        diff = current_val - baseline_val
        pct = diff / abs(baseline_val) * 100
        if abs(pct) >= 0.5:
            sign = "+" if pct > 0 else ""
            if lower_is_better:
                css_class = "good" if pct < 0 else "bad"
            else:
                css_class = "good" if pct > 0 else "bad"
            delta_html = f' <span class="delta {css_class}">{sign}{pct:.0f}%</span>'
    parts.append(
        f'<span class="metric-pill">'
        f'<span class="pill-label">{_html.escape(str(label))}</span>'
        f'<span class="pill-value">{_html.escape(str(val_str))}</span>'
        f'{delta_html}'
        f'</span>'
    )


def _fmt_ms(val: float | None) -> str:
    if val is None:
        return "n/a"
    return f"{val:.1f}ms"


def _fmt_pct(val: float | None) -> str:
    if val is None:
        return "n/a"
    return f"{val:.1f}%"


def _summary_ok(s: dict | None) -> TypeGuard[dict]:
    """Check if a summary dict is present and successful."""
    return s is not None and s.get("returncode", 0) == 0


def _render_speedscope_link(parts: list[str], json_path: Path, esc, label: str = "Open in Speedscope") -> None:
    """Render a button linking to speedscope.dev with the local file path shown."""
    abs_path = esc(str(json_path.resolve()))
    parts.append(
        f'<div style="margin:8px 0;">'
        f'<a href="https://www.speedscope.app" target="_blank" rel="noopener" '
        f'style="display:inline-block;padding:8px 16px;background:#2563eb;color:#fff;'
        f'border-radius:6px;text-decoration:none;font-weight:600;font-size:0.9em;">'
        f'{esc(label)} &#x2197;</a>'
        f'<span style="margin-left:12px;font-size:0.82em;color:#666;">'
        f'Drag &amp; drop <code>{abs_path}</code> into the page</span>'
        f'</div>'
    )
