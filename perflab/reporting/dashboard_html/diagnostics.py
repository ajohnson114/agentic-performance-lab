"""Diagnostics, outcome-analysis, environment, and user-action cards."""
from __future__ import annotations

from .data import GlanceData
from .widgets import _render_bar_chart


def _render_user_actions(parts: list[str], esc, user_actions: list[dict]) -> None:
    """Render a prominent 'User Action Required' card for build suggestions."""
    parts.append(
        '<div class="card" style="background:#fff8e1; border-left:4px solid #f57f17;">'
        '<h2 style="color:#e65100;">&#9888; User Action Required</h2>'
        '<p>The optimizer suggested build/compilation changes that it could not apply '
        'automatically (task.yaml is a protected file). Update your '
        '<code>build.cmd</code> in <code>task.yaml</code> and re-run to unlock '
        'further improvements.</p>'
        '<table class="iter-table">'
        '<tr><th>Flag</th><th>Suggestion</th><th>Iteration</th></tr>'
    )
    for a in user_actions:
        parts.append(
            f'<tr><td><code>{esc(a["flag"])}</code></td>'
            f'<td>{esc(a["suggestion"])}</td>'
            f'<td>{esc(str(a.get("iteration", "—")))}</td></tr>'
        )
    parts.append('</table></div>')


def _render_diagnostics(
    parts: list[str],
    esc,
    *,
    bottleneck_diagnoses: list[dict] | None = None,
    gpu_attribution: list[dict] | None = None,
    profile_diff: list[dict] | None = None,
    build_flag_recs: list[dict] | None = None,
    hotspot_diff: list[dict] | None = None,
    tma_data: dict | None = None,
    tma_level2_data: dict | None = None,
    power_data: dict | None = None,
    vectorization: list[dict] | None = None,
    gpu_memory: dict | None = None,
    thread_sched: dict | None = None,
    ebpf_data: dict | None = None,
    lock_contention_data: dict | None = None,
    hlo_attribution: list[dict] | None = None,
) -> None:
    """Render the Diagnostics card with collapsible sections."""
    if not any([bottleneck_diagnoses, gpu_attribution, profile_diff, build_flag_recs,
                hotspot_diff, tma_data, tma_level2_data, power_data, vectorization, gpu_memory,
                thread_sched, ebpf_data, lock_contention_data, hlo_attribution]):
        return

    parts.append('<div class="card"><h2>Diagnostics</h2>')

    # (a) Bottleneck diagnosis
    if bottleneck_diagnoses:
        parts.append('<details><summary>Bottleneck diagnosis</summary>')
        parts.append('<table class="iter-table">')
        parts.append('<tr><th>Rank</th><th>Bottleneck</th><th>Root Cause</th>'
                     '<th>Confidence</th><th>Suggested Actions</th></tr>')
        for d in bottleneck_diagnoses:
            conf = d.get("confidence", "")
            badge_cls = conf if conf in ("high", "medium", "low") else ""
            actions = d.get("suggested_actions", [])
            actions_str = "; ".join(actions) if isinstance(actions, list) else str(actions)
            parts.append(
                f'<tr>'
                f'<td>{esc(str(d.get("rank", "")))}</td>'
                f'<td>{esc(str(d.get("bottleneck", "")))}</td>'
                f'<td>{esc(str(d.get("root_cause", "")))}</td>'
                f'<td><span class="badge {badge_cls}">{esc(conf)}</span></td>'
                f'<td>{esc(actions_str)}</td>'
                f'</tr>'
            )
        parts.append('</table></details>')

    # (b) GPU attribution
    if gpu_attribution:
        parts.append('<details><summary>GPU attribution</summary>')
        bar_items = [
            {"name": a["name"], "pct": a.get("gpu_pct", 0), "total_ms": a.get("gpu_time_ms", 0)}
            for a in gpu_attribution
        ]
        _render_bar_chart(parts, bar_items, esc, color="#34d399")
        # Per-kernel diagnosis text
        for a in gpu_attribution:
            diag = a.get("diagnosis", "")
            sugg = a.get("suggestions", [])
            if diag or sugg:
                parts.append(f'<p style="font-size:0.85em;margin:2px 0 6px 4px;">'
                             f'<strong>{esc(a["name"][:50])}</strong>: {esc(diag)}')
                if sugg:
                    parts.append(f'<br/><em>{esc("; ".join(sugg))}</em>')
                parts.append('</p>')
        parts.append('</details>')

    # (b2) HLO attribution (JAX/TPU)
    if hlo_attribution:
        parts.append('<details><summary>XLA/HLO op attribution</summary>')
        bar_items = [
            {
                "name": f'{a["op"]} ({a.get("category", "?")})',
                "pct": a.get("estimated_device_pct", 0),
                "total_ms": a.get("count", 0),
            }
            for a in hlo_attribution[:10]
        ]
        _render_bar_chart(parts, bar_items, esc, color="#f59e0b")
        for a in hlo_attribution[:10]:
            diag = a.get("diagnosis", "")
            sugg = a.get("suggestions", [])
            if diag or sugg:
                parts.append(
                    f'<p style="font-size:0.85em;margin:2px 0 6px 4px;">'
                    f'<strong>{esc(str(a["op"]))}</strong> '
                    f'({a.get("count", 0)} ops, ~{a.get("estimated_device_pct", 0):.0f}% device time): '
                    f'{esc(diag)}'
                )
                if sugg:
                    parts.append(f'<br/><em>{esc("; ".join(sugg))}</em>')
                parts.append('</p>')
        parts.append('</details>')

    # (c) Profile diff
    if profile_diff:
        parts.append('<details><summary>Profile diff: baseline vs optimized</summary>')
        parts.append('<table class="iter-table">')
        parts.append('<tr><th>Metric</th><th>Before</th><th>After</th>'
                     '<th>Change%</th><th>Direction</th></tr>')
        for d in profile_diff:
            direction = d.get("direction", "unchanged")
            if direction == "improved":
                dir_badge = '<span class="badge yes">improved</span>'
            elif direction == "regressed":
                dir_badge = '<span class="badge no">regressed</span>'
            else:
                dir_badge = f'<span class="badge">{esc(direction)}</span>'
            delta_pct = d.get("delta_pct", 0)
            sign = "+" if delta_pct > 0 else ""
            parts.append(
                f'<tr>'
                f'<td><code>{esc(str(d.get("metric", "")))}</code></td>'
                f'<td>{d.get("before", 0):.4g}</td>'
                f'<td>{d.get("after", 0):.4g}</td>'
                f'<td>{sign}{delta_pct:.1f}%</td>'
                f'<td>{dir_badge}</td>'
                f'</tr>'
            )
        parts.append('</table></details>')

    # (d) Hotspot diff
    if hotspot_diff:
        parts.append('<details><summary>CPU hotspot shifts: baseline vs optimized</summary>')
        parts.append('<table class="iter-table">')
        parts.append('<tr><th>Function</th><th>Before</th><th>After</th>'
                     '<th>Change</th><th>Status</th></tr>')
        for s in hotspot_diff:
            status = s.get("status", "unchanged")
            before = s.get("before_pct", 0)
            after = s.get("after_pct", 0)
            delta = s.get("delta_pct", 0)
            func = s.get("function", "?")
            if len(func) > 50:
                func = func[:47] + "..."
            if status == "new":
                status_badge = '<span class="badge no">new</span>'
            elif status == "removed":
                status_badge = '<span class="badge yes">gone</span>'
            elif status == "decreased":
                status_badge = '<span class="badge yes">decreased</span>'
            elif status == "increased":
                status_badge = '<span class="badge no">increased</span>'
            else:
                status_badge = f'<span class="badge">{esc(status)}</span>'
            sign = "+" if delta > 0 else ""
            parts.append(
                f'<tr>'
                f'<td><code>{esc(func)}</code></td>'
                f'<td>{before:.1f}%</td>'
                f'<td>{after:.1f}%</td>'
                f'<td>{sign}{delta:.1f}pp</td>'
                f'<td>{status_badge}</td>'
                f'</tr>'
            )
        parts.append('</table></details>')

    # (e) Build flag recommendations
    if build_flag_recs:
        parts.append('<details open><summary>Recommended compile flags</summary>')
        parts.append('<p style="color:#999;font-size:0.85em;">Based on ISA detection + profiler analysis. '
                     'Apply these to your build command for better performance.</p>')
        parts.append('<table class="iter-table">')
        parts.append('<tr><th>Flag</th><th>Reason</th><th>Impact</th><th>Category</th></tr>')
        for r in build_flag_recs:
            impact = r.get("impact", "")
            impact_cls = impact if impact in ("high", "medium", "low") else ""
            parts.append(
                f'<tr>'
                f'<td><code>{esc(str(r.get("flag", "")))}</code></td>'
                f'<td>{esc(str(r.get("reason", "")))}</td>'
                f'<td><span class="badge {impact_cls}">{esc(impact)}</span></td>'
                f'<td>{esc(str(r.get("category", "")))}</td>'
                f'</tr>'
            )
        parts.append('</table>')
        parts.append('<p style="color:#999;font-size:0.85em;margin-top:8px;">'
                     '<strong>Production build:</strong> After optimization converges, compile with: '
                     '<code>-O3 -march=native -mtune=native -flto -DNDEBUG</code>. '
                     'For +10-20%, use PGO: <code>-fprofile-generate</code> → run → '
                     '<code>-fprofile-use -flto</code>.</p>')
        parts.append('</details>')

    # (f) Top-Down Microarchitecture Analysis
    if tma_data:
        parts.append('<details><summary>Top-Down Microarchitecture Analysis (TMA)</summary>')
        parts.append('<div style="display:flex;gap:16px;flex-wrap:wrap;margin:8px 0;">')
        for label, key, color in [
            ("Frontend Bound", "frontend_bound_pct", "#e74c3c"),
            ("Backend Bound", "backend_bound_pct", "#e67e22"),
            ("Bad Speculation", "bad_speculation_pct", "#9b59b6"),
            ("Retiring", "retiring_pct", "#27ae60"),
        ]:
            val = tma_data.get(key, 0)
            parts.append(
                f'<div style="text-align:center;min-width:120px;">'
                f'<div style="font-size:24px;font-weight:bold;color:{color}">{val:.1f}%</div>'
                f'<div style="font-size:12px;color:#666">{label}</div>'
                f'</div>'
            )
        parts.append('</div>')
        dominant = tma_data.get("dominant_bottleneck", "").replace("_", " ").title()
        if dominant:
            parts.append(f'<p>Dominant bottleneck: <strong>{esc(dominant)}</strong></p>')

        # TMA Level 2/3 (from toplev or AMD perf events)
        l2 = tma_level2_data
        if l2:
            parts.append('<h4 style="margin-top:12px;">Level 2/3 Breakdown</h4>')
            parts.append('<div style="display:flex;gap:16px;flex-wrap:wrap;margin:8px 0;">')
            for label, key, color in [
                ("Memory Bound", "memory_bound_pct", "#e67e22"),
                ("Core Bound", "core_bound_pct", "#3498db"),
                ("Fetch Latency", "fetch_latency_pct", "#e74c3c"),
                ("Fetch BW", "fetch_bandwidth_pct", "#c0392b"),
            ]:
                val = l2.get(key)
                if val is not None:
                    parts.append(
                        f'<div style="text-align:center;min-width:110px;">'
                        f'<div style="font-size:20px;font-weight:bold;color:{color}">{val:.1f}%</div>'
                        f'<div style="font-size:11px;color:#666">{label}</div>'
                        f'</div>'
                    )
            parts.append('</div>')

            # Memory hierarchy breakdown
            mem_levels = [
                ("L1", "l1_bound_pct"), ("L2", "l2_bound_pct"),
                ("L3", "l3_bound_pct"), ("DRAM", "dram_bound_pct"),
                ("Store", "store_bound_pct"),
            ]
            active = [(n, l2[k]) for n, k in mem_levels if l2.get(k) is not None]
            if active:
                parts.append('<p style="margin-top:4px;">Memory hierarchy: ')
                parts.append(' → '.join(f'<strong>{n}</strong> {v:.1f}%' for n, v in active))
                dom = l2.get("dominant_memory_level")
                if dom:
                    parts.append(f' (bottleneck: <strong>{esc(dom)}</strong>)')
                parts.append('</p>')

            source = l2.get("source", "")
            if source:
                parts.append(f'<p style="color:#999;font-size:0.8em;">Source: {esc(source)}</p>')

        parts.append('</details>')

    # (g) Power/Energy profiling
    if power_data:
        parts.append('<details><summary>Power &amp; energy profiling</summary>')
        rapl = power_data.get("rapl", {})
        gpu_power = power_data.get("gpu_power", {})
        if rapl:
            parts.append('<h4>CPU (RAPL)</h4><table class="iter-table">')
            if "package_joules" in rapl:
                parts.append(f'<tr><td>Package energy</td><td>{rapl["package_joules"]:.2f} J</td></tr>')
            if "cores_joules" in rapl:
                parts.append(f'<tr><td>Cores energy</td><td>{rapl["cores_joules"]:.2f} J</td></tr>')
            if "avg_package_watts" in rapl:
                parts.append(f'<tr><td>Avg package power</td><td>{rapl["avg_package_watts"]:.1f} W</td></tr>')
            parts.append('</table>')
        if gpu_power:
            parts.append('<h4>GPU (nvidia-smi)</h4><table class="iter-table">')
            if "avg_watts" in gpu_power:
                parts.append(f'<tr><td>Avg power draw</td><td>{gpu_power["avg_watts"]:.1f} W</td></tr>')
            if "max_watts" in gpu_power:
                parts.append(f'<tr><td>Peak power draw</td><td>{gpu_power["max_watts"]:.1f} W</td></tr>')
            if "sample_count" in gpu_power:
                parts.append(f'<tr><td>Samples</td><td>{gpu_power["sample_count"]}</td></tr>')
            parts.append('</table>')
        parts.append('</details>')

    # (h) Vectorization analysis
    if vectorization:
        parts.append('<details><summary>Vectorization analysis</summary>')
        parts.append('<table class="iter-table">')
        parts.append('<tr><th>Function</th><th>SIMD</th><th>ISA</th><th>CPU %</th></tr>')
        for v in vectorization:
            has = v.get("has_simd", False)
            badge = '<span class="badge yes">yes</span>' if has else '<span class="badge no">no</span>'
            parts.append(
                f'<tr>'
                f'<td><code>{esc(str(v.get("function", "")))}</code></td>'
                f'<td>{badge}</td>'
                f'<td>{esc(str(v.get("simd_isa", "none")))}</td>'
                f'<td>{v.get("hot_pct", 0):.1f}%</td>'
                f'</tr>'
            )
        parts.append('</table></details>')

    # (i) GPU memory utilization
    if gpu_memory:
        parts.append('<details><summary>GPU memory utilization</summary>')
        parts.append('<table class="iter-table">')
        if "total_mib" in gpu_memory:
            parts.append(f'<tr><td>Total VRAM</td><td>{gpu_memory["total_mib"]:.0f} MiB</td></tr>')
        if "max_used_mib" in gpu_memory:
            parts.append(f'<tr><td>Peak used</td><td>{gpu_memory["max_used_mib"]:.0f} MiB</td></tr>')
        if "avg_used_mib" in gpu_memory:
            parts.append(f'<tr><td>Avg used</td><td>{gpu_memory["avg_used_mib"]:.0f} MiB</td></tr>')
        if "utilization_pct" in gpu_memory:
            pct = gpu_memory["utilization_pct"]
            color = "#b91c1c" if pct > 90 else "#16a34a"
            parts.append(f'<tr><td>Utilization</td><td style="color:{color};font-weight:700">{pct:.1f}%</td></tr>')
        parts.append('</table></details>')

    # (j) Thread scheduling
    if thread_sched:
        latency = thread_sched.get("latency", [])
        timehist = thread_sched.get("timehist", {})
        if latency or timehist:
            parts.append('<details><summary>Thread scheduling (perf sched)</summary>')
            if latency:
                parts.append('<table class="iter-table">')
                parts.append('<tr><th>Thread</th><th>Runtime</th><th>Switches</th>'
                             '<th>Avg Delay</th><th>Max Delay</th></tr>')
                for entry in latency[:10]:
                    parts.append(
                        f'<tr>'
                        f'<td><code>{esc(str(entry.get("task", "")))}</code></td>'
                        f'<td>{entry.get("runtime_ms", 0):.1f}ms</td>'
                        f'<td>{entry.get("switches", 0)}</td>'
                        f'<td>{entry.get("avg_delay_ms", 0):.3f}ms</td>'
                        f'<td>{entry.get("max_delay_ms", 0):.3f}ms</td>'
                        f'</tr>'
                    )
                parts.append('</table>')
            if timehist.get("migrations"):
                parts.append(f'<p>Thread migrations: <strong>{timehist["migrations"]}</strong></p>')
            parts.append('</details>')

    # (k) eBPF I/O tracing
    if ebpf_data:
        has_io_data = ebpf_data.get("read_syscalls") or ebpf_data.get("write_syscalls")
        if has_io_data:
            parts.append('<details><summary>I/O syscall tracing (eBPF)</summary>')
            parts.append('<table class="iter-table">')
            if ebpf_data.get("read_syscalls") is not None:
                parts.append(f'<tr><td>Read syscalls</td><td>{ebpf_data["read_syscalls"]:,}</td></tr>')
            if ebpf_data.get("write_syscalls") is not None:
                parts.append(f'<tr><td>Write syscalls</td><td>{ebpf_data["write_syscalls"]:,}</td></tr>')
            if ebpf_data.get("read_bytes") is not None:
                mb = ebpf_data["read_bytes"] / (1024 * 1024)
                parts.append(f'<tr><td>Read bytes</td><td>{mb:.1f} MB</td></tr>')
            if ebpf_data.get("write_bytes") is not None:
                mb = ebpf_data["write_bytes"] / (1024 * 1024)
                parts.append(f'<tr><td>Write bytes</td><td>{mb:.1f} MB</td></tr>')
            read_lat = ebpf_data.get("read_latency", {})
            if read_lat.get("p50_ns") is not None:
                parts.append(f'<tr><td>Read latency p50</td><td>{read_lat["p50_ns"] / 1000:.1f} µs</td></tr>')
            if read_lat.get("p99_ns") is not None:
                p99_us = read_lat["p99_ns"] / 1000
                color = "#b91c1c" if p99_us > 10000 else "#16a34a"
                parts.append(f'<tr><td>Read latency p99</td><td style="color:{color};font-weight:700">{p99_us:.0f} µs</td></tr>')
            write_lat = ebpf_data.get("write_latency", {})
            if write_lat.get("p50_ns") is not None:
                parts.append(f'<tr><td>Write latency p50</td><td>{write_lat["p50_ns"] / 1000:.1f} µs</td></tr>')
            if write_lat.get("p99_ns") is not None:
                p99_us = write_lat["p99_ns"] / 1000
                color = "#b91c1c" if p99_us > 10000 else "#16a34a"
                parts.append(f'<tr><td>Write latency p99</td><td style="color:{color};font-weight:700">{p99_us:.0f} µs</td></tr>')
            parts.append('</table></details>')

    # (l) Lock contention
    if lock_contention_data:
        lock_stats = lock_contention_data.get("lock_stats", {})
        c2c_stats = lock_contention_data.get("c2c_stats", {})
        if lock_stats.get("locks") or c2c_stats.get("total_hitm"):
            parts.append('<details><summary>Lock contention (perf lock / c2c)</summary>')
            locks = lock_stats.get("locks", [])
            if locks:
                parts.append('<h4>Lock statistics</h4>')
                parts.append('<table class="iter-table">')
                parts.append('<tr><th>Lock</th><th>Acquired</th><th>Contended</th><th>Contention %</th><th>Total Wait</th></tr>')
                for lock in locks[:10]:
                    acq = lock.get("acquired", 0)
                    cont = lock.get("contended", 0)
                    cont_pct = (cont / acq * 100) if acq > 0 else 0
                    wait_ms = lock.get("total_wait_ns", 0) / 1e6
                    color = "#b91c1c" if cont_pct > 10 else "#16a34a"
                    parts.append(
                        f'<tr>'
                        f'<td><code>{esc(str(lock.get("name", "?")))}</code></td>'
                        f'<td>{acq:,}</td>'
                        f'<td>{cont:,}</td>'
                        f'<td style="color:{color};font-weight:700">{cont_pct:.1f}%</td>'
                        f'<td>{wait_ms:.1f} ms</td>'
                        f'</tr>'
                    )
                parts.append('</table>')
            if c2c_stats.get("total_hitm", 0) > 0:
                parts.append('<h4>False sharing (perf c2c)</h4>')
                parts.append('<table class="iter-table">')
                parts.append(f'<tr><td>Total HITM events</td><td>{c2c_stats["total_hitm"]:,}</td></tr>')
                if c2c_stats.get("total_store"):
                    parts.append(f'<tr><td>Total stores</td><td>{c2c_stats["total_store"]:,}</td></tr>')
                parts.append('</table>')
                sharing_lines = c2c_stats.get("false_sharing_lines", [])
                if sharing_lines:
                    parts.append('<p>Top cache-line conflicts:</p>')
                    parts.append('<table class="iter-table">')
                    parts.append('<tr><th>Address</th><th>HITM</th><th>Stores</th></tr>')
                    for cl in sharing_lines[:5]:
                        parts.append(
                            f'<tr><td><code>{esc(str(cl.get("address", "")))}</code></td>'
                            f'<td>{cl.get("hitm", 0):,}</td>'
                            f'<td>{cl.get("store", 0):,}</td></tr>'
                        )
                    parts.append('</table>')
            parts.append('</details>')

    parts.append('</div>')


def _render_outcome_analysis(
    parts: list[str],
    esc,
    *,
    optimization_summary: str | None = None,
    history: list[dict] | None = None,
    glance: GlanceData | None = None,
) -> None:
    """Render the What Worked / What Didn't analysis card."""
    rows = history or (glance.rows if glance else None) or []
    # finalize.py's maybe_early_stop() appends a synthetic second history
    # entry for the same iteration ("early stop: ...", accepted=False).
    # glance.rows is already filtered the same way in reporting/generate.py,
    # but the raw ``history`` list passed here is not -- drop it here too so
    # the "what didn't work" table doesn't show a duplicate row.
    rows = [r for r in rows if not str(r.get("description", "")).startswith("early stop:")]
    accepted_rows = [r for r in rows if r.get("accepted") and r.get("iter", r.get("iteration", 0)) > 0]
    rejected_rows = [r for r in rows if not r.get("accepted") and r.get("iter", r.get("iteration", 0)) > 0]

    has_content = optimization_summary or accepted_rows or rejected_rows
    if not has_content:
        return

    parts.append('<div class="card">')
    parts.append('<h2>Optimization analysis</h2>')

    # --- What worked ---
    if accepted_rows:
        parts.append('<details open><summary style="color:#16a34a;font-weight:700">What worked</summary>')
        parts.append('<table class="iter-table">')
        parts.append('<tr><th>Iter</th><th>Speedup</th><th>Description</th></tr>')
        for r in accepted_rows:
            it = r.get("iter", r.get("iteration", "?"))
            spd = r.get("speedup", 1.0)
            notes = r.get("notes", r.get("description", ""))
            parts.append(
                f'<tr class="accepted">'
                f'<td>{esc(str(it))}</td>'
                f'<td>{spd:.2f}x</td>'
                f'<td>{esc(str(notes))}</td>'
                f'</tr>'
            )
        parts.append('</table>')
        parts.append('</details>')

    # --- What didn't work ---
    if rejected_rows:
        parts.append('<details><summary style="color:#b91c1c;font-weight:700">What didn\'t work</summary>')
        parts.append('<table class="iter-table">')
        parts.append('<tr><th>Iter</th><th>Description</th></tr>')
        for r in rejected_rows:
            it = r.get("iter", r.get("iteration", "?"))
            notes = r.get("notes", r.get("description", ""))
            parts.append(
                f'<tr>'
                f'<td>{esc(str(it))}</td>'
                f'<td>{esc(str(notes))}</td>'
                f'</tr>'
            )
        parts.append('</table>')
        parts.append('</details>')

    # --- LLM-generated explanation ---
    if optimization_summary:
        parts.append('<details open><summary style="font-weight:700">Why it worked</summary>')
        parts.append(f'<p class="explanation">{esc(optimization_summary)}</p>')
        parts.append('</details>')

    parts.append('</div>')


def _render_environment(parts: list[str], system_info: dict, esc) -> None:
    """Render an environment details/summary dropdown from system_info dict."""
    # Map system_info keys to display labels (ordered)
    field_map = [
        ("nvidia_gpus", "GPU"),
        ("driver_version", "Driver"),
        ("cuda_version", "CUDA"),
        ("torch_version", "PyTorch"),
        ("torch_cuda_version", "PyTorch CUDA"),
        ("tpu_chip", "TPU"),
        ("tpu_count", "TPU Chips"),
        ("jax_version", "JAX"),
        ("triton_version", "Triton"),
        ("cpp_compiler", "C++ Compiler"),
        ("openmp_version", "OpenMP"),
        ("cpu_model", "CPU"),
        ("cpu_count", "CPU Count"),
        ("python_version", "Python"),
        ("platform", "Platform"),
    ]
    # Build rows, skipping missing keys
    rows: list[tuple[str, str]] = []
    for key, label in field_map:
        val = system_info.get(key)
        if val is None:
            continue
        if key == "nvidia_gpus" and isinstance(val, list):
            for i, gpu in enumerate(val):
                name = gpu.get("name", "?")
                mem = gpu.get("memory_mib", "?")
                drv = gpu.get("driver_version", "")
                suffix = f" ({mem} MiB)" if mem != "?" else ""
                prefix = f"GPU {i}" if len(val) > 1 else "GPU"
                rows.append((prefix, f"{name}{suffix}"))
                if drv and not any(r[0] == "Driver" for r in rows):
                    rows.append(("Driver", drv))
        elif key == "driver_version":
            # Skip if already added from nvidia_gpus
            if not any(r[0] == "Driver" for r in rows):
                rows.append((label, str(val)))
        else:
            rows.append((label, str(val)))

    if not rows:
        return

    parts.append('<details class="card" style="cursor:pointer">')
    parts.append('<summary><h2 style="display:inline">Environment</h2></summary>')
    parts.append('<table class="iter-table" style="margin-top:8px">')
    for label, value in rows:
        parts.append(
            f'<tr><td style="font-weight:600;width:140px">{esc(label)}</td>'
            f'<td>{esc(value)}</td></tr>'
        )
    parts.append('</table>')
    parts.append('</details>')
