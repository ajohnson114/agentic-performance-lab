"""Accelerator profiler sections: Metal, nsys, ncu, JAX/XLA, and TPU."""
from __future__ import annotations

from .data import ProfilerData
from .widgets import _fmt_ms, _fmt_pct, _metric_pill, _render_bar_chart, _summary_ok

# ---------------------------------------------------------------------------
# Metal section
# ---------------------------------------------------------------------------

def _render_metal_section(parts: list[str], prof: ProfilerData, esc) -> None:
    ms = prof.metal_summary
    bms = prof.baseline_metal_summary
    has_opt = _summary_ok(ms)
    has_base = _summary_ok(bms)
    # Reads below are guarded by has_opt/has_base; rebind non-Optional for the type checker.
    ms = ms or {}
    bms = bms or {}

    parts.append('<h3>Metal GPU profiler</h3>')

    if has_opt:
        # Metric pills
        parts.append('<div class="metric-pills">')
        gpu_time = ms.get("gpu_time_total_ms")
        base_gpu_time = bms.get("gpu_time_total_ms") if has_base else None
        _metric_pill(parts, "GPU Time", _fmt_ms(gpu_time), base_gpu_time, gpu_time, lower_is_better=True)

        submissions = ms.get("gpu_submissions")
        base_submissions = bms.get("gpu_submissions") if has_base else None
        sub_str = str(submissions) if submissions is not None else "n/a"
        _metric_pill(parts, "Submissions", sub_str,
                     float(base_submissions) if base_submissions is not None else None,
                     float(submissions) if submissions is not None else None,
                     lower_is_better=True)

        gpu_idle = ms.get("gpu_idle_pct")
        base_idle = bms.get("gpu_idle_pct") if has_base else None
        _metric_pill(parts, "GPU Idle", _fmt_pct(gpu_idle), base_idle, gpu_idle, lower_is_better=True)
        parts.append('</div>')

        # Top submissions bar chart
        top_subs = ms.get("top_submissions", [])
        if top_subs:
            items = [
                {
                    "name": s.get("label", s.get("encoder_type", "?")),
                    "pct": s.get("gpu_time_ms", 0) / (gpu_time or 1) * 100 if gpu_time else 0,
                    "total_ms": s.get("gpu_time_ms", 0),
                }
                for s in top_subs[:10]
            ]
            parts.append('<h4>Top submissions</h4>')
            _render_bar_chart(parts, items, esc, color="#a78bfa")

        # GPU counters
        counters = ms.get("gpu_counters", {})
        if counters:
            parts.append('<div class="metric-pills" style="margin-top:8px">')
            base_counters = bms.get("gpu_counters", {}) if has_base else {}
            for key, label in [("alu_utilization", "ALU Util"), ("memory_bandwidth", "Mem BW"),
                               ("occupancy", "Occupancy"), ("gpu_active", "GPU Active")]:
                val = counters.get(key)
                bval = base_counters.get(key)
                if val is not None:
                    _metric_pill(parts, label, _fmt_pct(val), bval, val, lower_is_better=False)
            parts.append('</div>')

    # --- Comparison dropdown ---
    if has_opt and has_base:
        parts.append('<details><summary>Compare with baseline (Metal)</summary>')
        parts.append('<div class="compare-row">')
        parts.append('<div class="compare-col baseline"><h4>Baseline</h4>')
        base_subs = bms.get("top_submissions", [])
        base_gpu_time_val = bms.get("gpu_time_total_ms") or 1
        if base_subs:
            items = [
                {
                    "name": s.get("label", s.get("encoder_type", "?")),
                    "pct": s.get("gpu_time_ms", 0) / base_gpu_time_val * 100,
                    "total_ms": s.get("gpu_time_ms", 0),
                }
                for s in base_subs[:10]
            ]
            _render_bar_chart(parts, items, esc, color="#9ca3af")
        else:
            parts.append('<p style="color:#888">No submissions data</p>')
        parts.append('</div>')
        parts.append('<div class="compare-col optimized"><h4>Optimized</h4>')
        opt_subs = ms.get("top_submissions", [])
        opt_gpu_time_val = ms.get("gpu_time_total_ms") or 1
        if opt_subs:
            items = [
                {
                    "name": s.get("label", s.get("encoder_type", "?")),
                    "pct": s.get("gpu_time_ms", 0) / opt_gpu_time_val * 100,
                    "total_ms": s.get("gpu_time_ms", 0),
                }
                for s in opt_subs[:10]
            ]
            _render_bar_chart(parts, items, esc, color="#a78bfa")
        else:
            parts.append('<p style="color:#888">No submissions data</p>')
        parts.append('</div>')
        parts.append('</div>')
        parts.append('</details>')


# ---------------------------------------------------------------------------
# Nsys section
# ---------------------------------------------------------------------------

def _render_nsys_section(parts: list[str], prof: ProfilerData, esc) -> None:
    ns = prof.nsys_summary
    bns = prof.baseline_nsys_summary
    has_opt = _summary_ok(ns)
    has_base = _summary_ok(bns)
    # Reads below are guarded by has_opt/has_base; rebind non-Optional for the type checker.
    ns = ns or {}
    bns = bns or {}

    parts.append('<h3>Nsight Systems (nsys)</h3>')

    if has_opt:
        # Metric pills
        parts.append('<div class="metric-pills">')
        kernel_time = ns.get("cuda_kernel_time_ms")
        base_kernel_time = bns.get("cuda_kernel_time_ms") if has_base else None
        _metric_pill(parts, "GPU Kernel Time", _fmt_ms(kernel_time), base_kernel_time, kernel_time, lower_is_better=True)

        memcpy_time = ns.get("memcpy_time_ms")
        base_memcpy_time = bns.get("memcpy_time_ms") if has_base else None
        _metric_pill(parts, "Memcpy Time", _fmt_ms(memcpy_time), base_memcpy_time, memcpy_time, lower_is_better=True)

        gpu_active = ns.get("gpu_active_pct")
        base_gpu_active = bns.get("gpu_active_pct") if has_base else None
        _metric_pill(parts, "GPU Active", _fmt_pct(gpu_active), base_gpu_active, gpu_active, lower_is_better=False)
        parts.append('</div>')

        # Top GPU kernels bar chart
        top_kernels = ns.get("top_kernels", [])
        if top_kernels:
            items = [
                {
                    "name": k.get("name", "?"),
                    "pct": k.get("pct", 0),
                    "total_ms": k.get("total_ms", 0),
                    "count": k.get("count", 0),
                }
                for k in top_kernels[:10]
            ]
            parts.append('<h4>Top GPU kernels</h4>')
            _render_bar_chart(parts, items, esc, color="#34d399")

        # Memory transfers
        memcpy_list = ns.get("memcpy", [])
        if memcpy_list:
            parts.append('<h4>Memory transfers</h4>')
            for mc in memcpy_list:
                direction = mc.get("direction", "?")
                count = mc.get("count", 0)
                total_bytes = mc.get("total_bytes", 0)
                total_ms = mc.get("total_ms", 0)
                size_str = f"{total_bytes / 1024 / 1024:.1f}MB" if total_bytes > 1024 * 1024 else f"{total_bytes / 1024:.1f}KB"
                parts.append(
                    f'<p style="font-size:0.85em; color:#666">'
                    f'{esc(direction)}: {count} transfers, {size_str}, {total_ms:.1f}ms</p>'
                )

    # --- Comparison dropdown ---
    if has_opt and has_base:
        parts.append('<details><summary>Compare with baseline (nsys)</summary>')
        parts.append('<div class="compare-row">')
        parts.append('<div class="compare-col baseline"><h4>Baseline</h4>')
        base_kernels = bns.get("top_kernels", [])
        if base_kernels:
            items = [
                {"name": k.get("name", "?"), "pct": k.get("pct", 0), "total_ms": k.get("total_ms", 0), "count": k.get("count", 0)}
                for k in base_kernels[:10]
            ]
            _render_bar_chart(parts, items, esc, color="#9ca3af")
        else:
            parts.append('<p style="color:#888">No kernel data</p>')
        parts.append('</div>')
        parts.append('<div class="compare-col optimized"><h4>Optimized</h4>')
        opt_kernels = ns.get("top_kernels", [])
        if opt_kernels:
            items = [
                {"name": k.get("name", "?"), "pct": k.get("pct", 0), "total_ms": k.get("total_ms", 0), "count": k.get("count", 0)}
                for k in opt_kernels[:10]
            ]
            _render_bar_chart(parts, items, esc, color="#34d399")
        else:
            parts.append('<p style="color:#888">No kernel data</p>')
        parts.append('</div>')
        parts.append('</div>')
        parts.append('</details>')


# ---------------------------------------------------------------------------
# Ncu section
# ---------------------------------------------------------------------------

def _render_ncu_section(parts: list[str], prof: ProfilerData, esc) -> None:
    nc = prof.ncu_summary
    bnc = prof.baseline_ncu_summary
    has_opt = _summary_ok(nc)
    has_base = _summary_ok(bnc)
    # Reads below are guarded by has_opt/has_base; rebind non-Optional for the type checker.
    nc = nc or {}
    bnc = bnc or {}

    parts.append('<h3>Nsight Compute (ncu)</h3>')

    if has_opt:
        # Metric pills
        parts.append('<div class="metric-pills">')
        sm_util = nc.get("sm_utilization_pct")
        base_sm = bnc.get("sm_utilization_pct") if has_base else None
        _metric_pill(parts, "SM Util", _fmt_pct(sm_util), base_sm, sm_util, lower_is_better=False)

        occupancy = nc.get("achieved_occupancy_pct")
        base_occ = bnc.get("achieved_occupancy_pct") if has_base else None
        _metric_pill(parts, "Occupancy", _fmt_pct(occupancy), base_occ, occupancy, lower_is_better=False)

        mem_tp = nc.get("memory_throughput_pct")
        base_mem = bnc.get("memory_throughput_pct") if has_base else None
        _metric_pill(parts, "Mem Throughput", _fmt_pct(mem_tp), base_mem, mem_tp, lower_is_better=False)

        achieved_bw = nc.get("achieved_bw_gbs")
        base_bw = bnc.get("achieved_bw_gbs") if has_base else None
        if achieved_bw is not None:
            _metric_pill(parts, "Achieved BW", f"{achieved_bw:.1f} GB/s", base_bw, achieved_bw, lower_is_better=False)

        branch_eff = nc.get("branch_efficiency_pct")
        base_branch_eff = bnc.get("branch_efficiency_pct") if has_base else None
        if branch_eff is not None:
            _metric_pill(parts, "Branch Eff", _fmt_pct(branch_eff), base_branch_eff, branch_eff, lower_is_better=False)

        tc_util = nc.get("tensor_core_utilization_pct")
        base_tc = bnc.get("tensor_core_utilization_pct") if has_base else None
        if tc_util is not None:
            _metric_pill(parts, "TC Util", _fmt_pct(tc_util), base_tc, tc_util, lower_is_better=False)

        l1_hr = nc.get("l1_hit_rate")
        base_l1 = bnc.get("l1_hit_rate") if has_base else None
        if l1_hr is not None:
            _metric_pill(parts, "L1 Hit Rate", _fmt_pct(l1_hr), base_l1, l1_hr, lower_is_better=False)

        l2_hr = nc.get("l2_hit_rate")
        base_l2 = bnc.get("l2_hit_rate") if has_base else None
        if l2_hr is not None:
            _metric_pill(parts, "L2 Hit Rate", _fmt_pct(l2_hr), base_l2, l2_hr, lower_is_better=False)

        parts.append('</div>')

        # GPU cache hierarchy diagnosis
        if l1_hr is not None and l2_hr is not None:
            mem_tp_val = nc.get("memory_throughput_pct", 0)
            if l1_hr < 50:
                cache_diag = f'<span style="color:#e74c3c;font-weight:bold">L1 bottleneck</span> (L1 hit {l1_hr:.0f}% → tiles too large for shared mem)'
            elif l2_hr < 50 and mem_tp_val > 40:
                cache_diag = f'<span style="color:#e67e22;font-weight:bold">L2 bottleneck</span> (L1 {l1_hr:.0f}% OK, L2 hit {l2_hr:.0f}% → working set exceeds L2)'
            elif mem_tp_val > 70:
                cache_diag = '<span style="color:#f39c12;font-weight:bold">DRAM saturated</span> (caches OK, bandwidth wall)'
            else:
                cache_diag = f'<span style="color:#27ae60">Healthy</span> (L1 {l1_hr:.0f}%, L2 {l2_hr:.0f}%)'
            parts.append(f'<p style="margin-top:8px;">GPU cache hierarchy: {cache_diag}</p>')

        # DRAM traffic breakdown
        dram_read = nc.get("dram_bytes_read_total")
        dram_write = nc.get("dram_bytes_written_total")
        if dram_read is not None or dram_write is not None:
            read_mb = (dram_read or 0) / 1024 / 1024
            write_mb = (dram_write or 0) / 1024 / 1024
            parts.append(
                f'<p style="font-size:0.85em; color:#666; margin-top:4px">'
                f'DRAM traffic: {read_mb:.1f} MB read, {write_mb:.1f} MB written</p>'
            )

        # Per-kernel table (top 8)
        kernels = nc.get("kernels", [])[:8]
        if kernels:
            parts.append('<h4>Top kernels</h4>')
            has_bw_col = any(k.get("achieved_bw_gbs") is not None for k in kernels)
            parts.append('<table class="iter-table">')
            bw_th = "<th>BW (GB/s)</th>" if has_bw_col else ""
            parts.append(f'<tr><th>Kernel</th><th>Invocations</th><th>SM%</th><th>Occupancy%</th><th>Mem%</th>{bw_th}</tr>')
            for k in kernels:
                name = k.get("name", "?")
                if len(name) > 40:
                    name = "..." + name[-37:]
                bw_td = f"<td>{k['achieved_bw_gbs']:.1f}</td>" if has_bw_col and k.get("achieved_bw_gbs") is not None else ("<td>-</td>" if has_bw_col else "")
                parts.append(
                    f'<tr>'
                    f'<td title="{esc(k.get("name", ""))}" style="font-family:monospace;font-size:0.85em">{esc(name)}</td>'
                    f'<td>{k.get("invocations", 0)}</td>'
                    f'<td>{k.get("sm_utilization_pct", 0):.1f}%</td>'
                    f'<td>{k.get("achieved_occupancy_pct", 0):.1f}%</td>'
                    f'<td>{k.get("memory_throughput_pct", 0):.1f}%</td>'
                    f'{bw_td}'
                    f'</tr>'
                )
            parts.append('</table>')

    # --- Comparison dropdown ---
    if has_opt and has_base:
        parts.append('<details><summary>Compare with baseline (ncu)</summary>')
        parts.append('<div class="compare-row">')

        # Baseline kernel table
        parts.append('<div class="compare-col baseline"><h4>Baseline</h4>')
        base_kernels = bnc.get("kernels", [])[:8]
        if base_kernels:
            parts.append('<table class="iter-table">')
            parts.append('<tr><th>Kernel</th><th>Inv</th><th>SM%</th><th>Occ%</th><th>Mem%</th></tr>')
            for k in base_kernels:
                name = k.get("name", "?")
                if len(name) > 30:
                    name = "..." + name[-27:]
                parts.append(
                    f'<tr>'
                    f'<td style="font-family:monospace;font-size:0.85em">{esc(name)}</td>'
                    f'<td>{k.get("invocations", 0)}</td>'
                    f'<td>{k.get("sm_utilization_pct", 0):.1f}%</td>'
                    f'<td>{k.get("achieved_occupancy_pct", 0):.1f}%</td>'
                    f'<td>{k.get("memory_throughput_pct", 0):.1f}%</td>'
                    f'</tr>'
                )
            parts.append('</table>')
        else:
            parts.append('<p style="color:#888">No kernel data</p>')
        parts.append('</div>')

        # Optimized kernel table
        parts.append('<div class="compare-col optimized"><h4>Optimized</h4>')
        opt_kernels = nc.get("kernels", [])[:8]
        if opt_kernels:
            parts.append('<table class="iter-table">')
            parts.append('<tr><th>Kernel</th><th>Inv</th><th>SM%</th><th>Occ%</th><th>Mem%</th></tr>')
            for k in opt_kernels:
                name = k.get("name", "?")
                if len(name) > 30:
                    name = "..." + name[-27:]
                parts.append(
                    f'<tr>'
                    f'<td style="font-family:monospace;font-size:0.85em">{esc(name)}</td>'
                    f'<td>{k.get("invocations", 0)}</td>'
                    f'<td>{k.get("sm_utilization_pct", 0):.1f}%</td>'
                    f'<td>{k.get("achieved_occupancy_pct", 0):.1f}%</td>'
                    f'<td>{k.get("memory_throughput_pct", 0):.1f}%</td>'
                    f'</tr>'
                )
            parts.append('</table>')
        else:
            parts.append('<p style="color:#888">No kernel data</p>')
        parts.append('</div>')

        parts.append('</div>')
        parts.append('</details>')


# ---------------------------------------------------------------------------
# JAX section
# ---------------------------------------------------------------------------

def _render_jax_section(parts: list[str], prof: ProfilerData, esc) -> None:
    js = prof.jax_summary
    bjs = prof.baseline_jax_summary
    has_opt = _summary_ok(js)
    has_base = _summary_ok(bjs)
    # Reads below are guarded by has_opt/has_base; rebind non-Optional for the type checker.
    js = js or {}
    bjs = bjs or {}

    parts.append('<h3>JAX / XLA profiler</h3>')

    if has_opt:
        # Metric pills
        parts.append('<div class="metric-pills">')
        compilations = js.get("xla_compilations")
        base_compilations = bjs.get("xla_compilations") if has_base else None
        comp_str = str(compilations) if compilations is not None else "n/a"
        _metric_pill(parts, "XLA Compilations", comp_str,
                     float(base_compilations) if base_compilations is not None else None,
                     float(compilations) if compilations is not None else None,
                     lower_is_better=True)

        compile_time = js.get("xla_compilation_time_ms")
        base_compile_time = bjs.get("xla_compilation_time_ms") if has_base else None
        _metric_pill(parts, "Compile Time", _fmt_ms(compile_time), base_compile_time, compile_time, lower_is_better=True)

        recomps = js.get("xla_recompilations")
        base_recomps = bjs.get("xla_recompilations") if has_base else None
        recomp_str = str(recomps) if recomps is not None else "n/a"
        _metric_pill(parts, "Recompilations", recomp_str,
                     float(base_recomps) if base_recomps is not None else None,
                     float(recomps) if recomps is not None else None,
                     lower_is_better=True)
        parts.append('</div>')

        # HLO module count
        hlo_count = js.get("hlo_module_count")
        if hlo_count is not None:
            parts.append(
                f'<p style="font-size:0.85em; color:#666; margin-top:4px">'
                f'HLO modules: {hlo_count}</p>'
            )

        # HLO ops table
        hlo_ops = js.get("hlo_ops", [])
        if hlo_ops:
            parts.append('<h4>Top HLO operations</h4>')
            parts.append('<table class="iter-table">')
            parts.append('<tr><th>Operation</th><th>Count</th></tr>')
            for op in hlo_ops[:10]:
                parts.append(
                    f'<tr>'
                    f'<td style="font-family:monospace;font-size:0.85em">{esc(str(op.get("op", "?")))}</td>'
                    f'<td>{op.get("count", 0)}</td>'
                    f'</tr>'
                )
            parts.append('</table>')


# ---------------------------------------------------------------------------
# TPU section (rendered within JAX when TPU data is present)
# ---------------------------------------------------------------------------

def _render_tpu_section(parts: list[str], prof: ProfilerData, esc) -> None:
    js = prof.jax_summary
    if not _summary_ok(js):
        return
    # Only render if TPU-specific data is present
    tpu_chip = js.get("tpu_chip")
    has_tpu = (
        tpu_chip is not None
        or js.get("mxu_utilization_pct") is not None
        or js.get("device_time_us") is not None
    )
    if not has_tpu:
        return

    parts.append('<h3>TPU device metrics</h3>')

    # Device info header
    tpu_count = js.get("tpu_count", 1)
    if tpu_chip:
        parts.append(
            f'<p style="font-size:0.85em; color:#666;">'
            f'{esc(str(tpu_chip))} &mdash; {tpu_count} chip{"s" if tpu_count > 1 else ""}'
            f'</p>'
        )

    # Metric pills
    parts.append('<div class="metric-pills">')

    mxu_util = js.get("mxu_utilization_pct")
    if mxu_util is not None:
        _metric_pill(parts, "MXU Utilization", f"{mxu_util:.1f}%",
                     None, mxu_util, lower_is_better=False)

    device_frac = js.get("device_fraction")
    if device_frac is not None:
        _metric_pill(parts, "Device Active", f"{device_frac * 100:.1f}%",
                     None, device_frac * 100, lower_is_better=False)

    infeed = js.get("infeed_stall_pct")
    if infeed is not None:
        _metric_pill(parts, "Infeed Stall", f"{infeed:.1f}%",
                     None, infeed, lower_is_better=True)

    parts.append('</div>')

    # Host-device time breakdown
    host_us = js.get("host_time_us")
    dev_us = js.get("device_time_us")
    if host_us is not None and dev_us is not None and (host_us + dev_us) > 0:
        total = host_us + dev_us
        h_pct = host_us / total * 100
        d_pct = dev_us / total * 100
        parts.append('<h4>Host vs Device time</h4>')
        parts.append(
            '<div style="display:flex;height:24px;border-radius:6px;overflow:hidden;'
            'margin:4px 0 8px 0;">'
        )
        if h_pct > 0:
            parts.append(
                f'<div style="width:{h_pct:.1f}%;background:#60a5fa;" '
                f'title="Host: {host_us/1e3:.1f}ms ({h_pct:.1f}%)"></div>'
            )
        if d_pct > 0:
            parts.append(
                f'<div style="width:{d_pct:.1f}%;background:#34d399;" '
                f'title="Device: {dev_us/1e3:.1f}ms ({d_pct:.1f}%)"></div>'
            )
        parts.append('</div>')
        parts.append(
            '<p style="font-size:0.8em;color:#888;">'
            f'<span style="color:#60a5fa">&#9632;</span> Host {h_pct:.1f}% '
            f'<span style="color:#34d399">&#9632;</span> Device {d_pct:.1f}%'
            '</p>'
        )

    # HLO cost metrics (FLOPS and bytes from XLA cost annotations)
    hlo_tflops = js.get("hlo_cost_tflops")
    hlo_bytes = js.get("hlo_cost_bytes_accessed")
    if hlo_tflops is not None or hlo_bytes is not None:
        parts.append('<h4>XLA Cost Estimate</h4>')
        cost_parts = []
        if hlo_tflops is not None:
            cost_parts.append(f'<strong>{hlo_tflops:.4f} TFLOPS</strong> (estimated from HLO)')
        if hlo_bytes is not None:
            bytes_mb = hlo_bytes / (1024 * 1024)
            cost_parts.append(f'{bytes_mb:.1f} MB bytes accessed')
        if hlo_tflops is not None and hlo_bytes is not None and hlo_bytes > 0:
            hlo_flops = js.get("hlo_cost_flops", hlo_tflops * 1e12)
            ai = hlo_flops / hlo_bytes
            cost_parts.append(f'AI = {ai:.1f} FLOP/byte')
        parts.append(f'<p>{" | ".join(cost_parts)}</p>')
