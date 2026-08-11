"""Small shared rendering helpers: bar charts, metric pills, formatters."""
from __future__ import annotations

import html as _html
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeGuard


_MD_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
_MD_CODE = re.compile(r"`([^`]+)`")
_MD_HEADING = re.compile(r"(#{1,6})\s+(.*)$")
_MD_BULLET = re.compile(r"[-*+]\s+(.*)$")


def _md_inline(s: str) -> str:
    """Inline markdown (bold, code) on ALREADY-ESCAPED text."""
    s = _MD_CODE.sub(r"<code>\1</code>", s)
    return _MD_BOLD.sub(r"<strong>\1</strong>", s)


def _render_markdown(parts: list[str], text: str, esc) -> None:
    """Render the small markdown subset LLM prose actually uses.

    The optimization summary is model-authored markdown (## headings, **bold**,
    bullets). It used to be esc()'d straight into a <p>, so the dashboard showed
    literal '##' and '**'. There is no markdown dependency and the dashboard is
    a self-contained file, so this handles the subset the summary prompt asks
    for and ignores the rest.

    Escapes FIRST and applies transforms to the escaped text -- the markdown
    markers (#, *, `) are untouched by escaping, so the output stays safe no
    matter what the model wrote.
    """
    in_list = False
    for raw in esc(text).split("\n"):
        line = raw.strip()
        if not line:
            if in_list:
                parts.append("</ul>")
                in_list = False
            continue

        if m := _MD_HEADING.match(line):
            if in_list:
                parts.append("</ul>")
                in_list = False
            # Offset below the surrounding section headings (h2/h3/h4) so the
            # summary's own '##' does not compete with real section titles.
            level = min(len(m.group(1)) + 3, 6)
            parts.append(f"<h{level}>{_md_inline(m.group(2))}</h{level}>")
            continue

        if m := _MD_BULLET.match(line):
            if not in_list:
                parts.append("<ul>")
                in_list = True
            parts.append(f"<li>{_md_inline(m.group(1))}</li>")
            continue

        if in_list:
            parts.append("</ul>")
            in_list = False
        parts.append(f'<p class="explanation">{_md_inline(line)}</p>')

    if in_list:
        parts.append("</ul>")


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
    """Render a button to speedscope.app plus the path to load there.

    Points at speedscope's Browse button rather than a one-click
    ``#profileURL=`` link at a local server, because Chrome gates a public
    origin fetching loopback behind a Local Network Access *permission* and
    such a link renders "Something went wrong" instead of a flame graph.
    Browse works everywhere: the click is a user gesture, so the browser gets
    a read grant for that one file, with no drag and no server involved.
    """
    abs_path = esc(str(json_path.resolve()))
    parts.append(
        f'<div style="margin:8px 0;">'
        f'<a href="https://www.speedscope.app" target="_blank" rel="noopener" '
        f'style="display:inline-block;padding:8px 16px;background:#2563eb;color:#fff;'
        f'border-radius:6px;text-decoration:none;font-weight:600;font-size:0.9em;">'
        f'{esc(label)} &#x2197;</a>'
        f'<span style="margin-left:12px;font-size:0.82em;color:#666;">'
        f'then click <strong>Browse</strong> and pick <code>{abs_path}</code></span>'
        f'</div>'
    )
