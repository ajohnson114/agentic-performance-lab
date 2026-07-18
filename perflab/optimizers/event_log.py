"""Structured event logging for agent optimization runs.

Writes JSON Lines to agent_events.jsonl in the run directory. Each event has
a timestamp, event_type, iteration, and type-specific data. LLM response text
is saved to separate files under llm_responses/.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentEventLog:
    run_dir: Path
    _path: Path = field(init=False)
    _responses_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self._path = self.run_dir / "agent_events.jsonl"
        self._responses_dir = self.run_dir / "llm_responses"
        self._responses_dir.mkdir(parents=True, exist_ok=True)

    def _write(self, event_type: str, iteration: int | None, data: dict[str, Any]) -> None:
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "iteration": iteration,
            **data,
        }
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def baseline_complete(
        self, value: float, bench_data: dict, profiler_summaries_available: list[str],
    ) -> None:
        self._write("baseline_complete", 0, {
            "value": value,
            "bench_data_keys": list(bench_data.keys()),
            "profiler_summaries_available": profiler_summaries_available,
        })

    def llm_request(
        self, iteration: int, prompt_length_chars: int,
        n_candidates_requested: int, model: str, provider: str,
        prompt_token_budget: int = 0,
    ) -> None:
        data = {
            "prompt_length_chars": prompt_length_chars,
            "n_candidates_requested": n_candidates_requested,
            "model": model,
            "provider": provider,
        }
        if prompt_token_budget > 0:
            data["prompt_token_budget"] = prompt_token_budget
        self._write("llm_request", iteration, data)

    def llm_response(
        self, iteration: int, response_text: str,
        usage: Any, n_candidates_parsed: int,
    ) -> None:
        # Save full response to file
        resp_path = self._responses_dir / f"iter{iteration}_response.txt"
        resp_path.write_text(response_text, encoding="utf-8")
        self._write("llm_response", iteration, {
            "response_length_chars": len(response_text),
            "usage_tokens": str(usage),
            "n_candidates_parsed": n_candidates_parsed,
            "raw_response_path": str(resp_path.relative_to(self.run_dir)),
        })

    def candidate_validation(
        self, iteration: int, candidate_index: int,
        valid: bool, errors: list[str],
    ) -> None:
        self._write("candidate_validation", iteration, {
            "candidate_index": candidate_index,
            "valid": valid,
            "errors": errors,
        })

    def candidate_patch(
        self, iteration: int, candidate_index: int,
        blocks: list[dict],
    ) -> None:
        block_summaries = []
        for b in blocks:
            block_summaries.append({
                "file_path": b.get("file_path", ""),
                "search_preview": b.get("search", "")[:100],
                "replace_preview": b.get("replace", "")[:100],
            })
        self._write("candidate_patch", iteration, {
            "candidate_index": candidate_index,
            "blocks": block_summaries,
        })

    def patch_fuzzy_correction(
        self, iteration: int, candidate_index: int, notices: list[str],
    ) -> None:
        self._write("patch_fuzzy_correction", iteration, {
            "candidate_index": candidate_index,
            "notices": notices,
        })

    def candidate_correctness(
        self, iteration: int, candidate_index: int,
        passed: bool, returncode: int, stderr_preview: str,
    ) -> None:
        self._write("candidate_correctness", iteration, {
            "candidate_index": candidate_index,
            "passed": passed,
            "returncode": returncode,
            "stderr_preview": stderr_preview[:500],
        })

    def candidate_benchmark(
        self, iteration: int, candidate_index: int,
        value: float, metric_name: str,
    ) -> None:
        self._write("candidate_benchmark", iteration, {
            "candidate_index": candidate_index,
            "value": value,
            "metric_name": metric_name,
        })

    def candidate_accepted(
        self, iteration: int, candidate_index: int,
        value: float, delta: float, speedup: float, description: str,
    ) -> None:
        self._write("candidate_accepted", iteration, {
            "candidate_index": candidate_index,
            "value": value,
            "delta": delta,
            "speedup": speedup,
            "description": description,
        })

    def iteration_complete(
        self, iteration: int, best_value: float, accepted_any: bool,
    ) -> None:
        self._write("iteration_complete", iteration, {
            "best_value": best_value,
            "accepted_any": accepted_any,
        })

    def drift_check(
        self, iteration: int, clean_value: float, last_accepted_value: float,
        drift_pct: float,
    ) -> None:
        self._write("drift_check", iteration, {
            "clean_value": clean_value,
            "last_accepted_value": last_accepted_value,
            "drift_pct": drift_pct,
        })

    def anti_gaming_warning(
        self, iteration: int, check_type: str, details: str,
        candidate_index: int | None = None,
    ) -> None:
        self._write("anti_gaming_warning", iteration, {
            "check_type": check_type,
            "details": details,
            "candidate_index": candidate_index,
        })

    def rlimit_warning(
        self, iteration: int, details: str,
        candidate_index: int | None = None,
    ) -> None:
        self._write("rlimit_warning", iteration, {
            "details": details,
            "candidate_index": candidate_index,
        })

    def early_stop(self, iteration: int, reason: str) -> None:
        self._write("early_stop", iteration, {"reason": reason})

    def run_complete(
        self, best_value: float, best_iter: int,
        baseline_value: float, total_iterations: int, total_llm_calls: int,
    ) -> None:
        self._write("run_complete", None, {
            "best_value": best_value,
            "best_iter": best_iter,
            "baseline_value": baseline_value,
            "total_iterations": total_iterations,
            "total_llm_calls": total_llm_calls,
        })

    def error_feedback(
        self, iteration: int, errors: list[dict],
    ) -> None:
        self._write("error_feedback", iteration, {
            "error_count": len(errors),
            "errors": [
                {"type": e.get("type", ""), "description": e.get("description", "")}
                for e in errors
            ],
        })

    def build_flags_state(
        self, iteration: int, build_cmd: str,
        flags_recommended: list[dict],
        flags_source: str = "static+profiler",
    ) -> None:
        self._write("build_flags_state", iteration, {
            "build_cmd": build_cmd,
            "flags_recommended": flags_recommended,
            "flags_source": flags_source,
        })

    def auto_tune_sweep(
        self, iteration: int, candidates_tried: int,
        best_value: float, best_knobs: dict, improvement: float,
    ) -> None:
        self._write("auto_tune_sweep", iteration, {
            "candidates_tried": candidates_tried,
            "best_value": best_value,
            "best_knobs": best_knobs,
            "improvement": improvement,
        })

    def roofline_detected(
        self, peak_tflops: float, peak_mem_bw_gbs: float,
        source: str, device: str,
    ) -> None:
        self._write("roofline_detected", None, {
            "peak_tflops": peak_tflops,
            "peak_mem_bw_gbs": peak_mem_bw_gbs,
            "source": source,
            "device": device,
        })


def replay_events(run_dir: Path) -> str:
    """Read agent_events.jsonl and produce a human-readable replay summary."""
    events_path = run_dir / "agent_events.jsonl"
    if not events_path.exists():
        return f"No agent events found in {run_dir}"

    lines: list[str] = []
    events = []
    for raw in events_path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            events.append(json.loads(raw))

    lines.append(f"=== Agent Run Replay ({run_dir.name}) ===\n")

    for ev in events:
        et = ev.get("event_type", "?")
        it = ev.get("iteration")
        iter_str = f"[iter {it}] " if it is not None else ""

        if et == "baseline_complete":
            lines.append(f"  Baseline: value={ev['value']}, profilers={ev.get('profiler_summaries_available', [])}")

        elif et == "llm_request":
            lines.append(f"\n{iter_str}LLM request: {ev['n_candidates_requested']} candidates via {ev['provider']}:{ev['model']} ({ev['prompt_length_chars']} chars)")

        elif et == "llm_response":
            lines.append(f"{iter_str}LLM response: {ev['response_length_chars']} chars, {ev['n_candidates_parsed']} candidates parsed, usage={ev.get('usage_tokens','?')}")
            lines.append(f"  Full response: {ev.get('raw_response_path', '?')}")

        elif et == "candidate_validation":
            status = "VALID" if ev["valid"] else f"INVALID: {ev['errors'][0]}" if ev.get("errors") else "INVALID"
            lines.append(f"{iter_str}  Candidate {ev['candidate_index']}: {status}")

        elif et == "candidate_patch":
            blocks = ev.get("blocks", [])
            for b in blocks:
                lines.append(f"{iter_str}    Patch: {b['file_path']} (search: {b['search_preview']!r}...)")

        elif et == "patch_fuzzy_correction":
            for note in ev.get("notices", []):
                lines.append(f"{iter_str}  ⚠ FUZZY PATCH CORRECTION (candidate {ev.get('candidate_index', '?')}): {note}")

        elif et == "candidate_correctness":
            status = "PASSED" if ev["passed"] else f"FAILED (rc={ev['returncode']})"
            lines.append(f"{iter_str}  Correctness: {status}")
            if not ev["passed"] and ev.get("stderr_preview"):
                lines.append(f"    stderr: {ev['stderr_preview'][:200]}")

        elif et == "candidate_benchmark":
            lines.append(f"{iter_str}  Benchmark: {ev['metric_name']}={ev['value']:.6g}")

        elif et == "candidate_accepted":
            lines.append(f"{iter_str}  >>> ACCEPTED candidate {ev['candidate_index']}: value={ev['value']:.6g}, delta={ev['delta']:+.6g}, speedup={ev['speedup']:.2f}x")

        elif et == "auto_tune_sweep":
            lines.append(
                f"{iter_str}  Auto-tune sweep: {ev.get('candidates_tried', 0)} candidates tried, "
                f"best={ev.get('best_value', 0):.6g} (improvement {ev.get('improvement', 0):+.6g})"
            )
            knobs = ev.get("best_knobs") or {}
            if knobs:
                lines.append(f"    Best knobs: {', '.join(f'{k}={v}' for k, v in knobs.items())}")

        elif et == "error_feedback":
            n = ev.get("error_count", 0)
            if n > 0:
                lines.append(f"{iter_str}Error feedback: {n} error(s) from previous iteration fed to LLM")
                for err in ev.get("errors", []):
                    lines.append(f"  - [{err.get('type', '?')}] {err.get('description', '')}")

        elif et == "drift_check":
            drift = ev.get("drift_pct", 0)
            warn = " WARNING" if drift > 5 else ""
            lines.append(f"{iter_str}  Drift check:{warn} clean={ev.get('clean_value', 0):.6g}, last_accepted={ev.get('last_accepted_value', 0):.6g}, drift={drift:.1f}%")

        elif et == "iteration_complete":
            if not ev["accepted_any"]:
                lines.append(f"{iter_str}No improvement this iteration (best={ev['best_value']:.6g})")

        elif et == "anti_gaming_warning":
            lines.append(f"{iter_str}  ⚠ ANTI-GAMING [{ev.get('check_type', '?')}]: {ev.get('details', '')}")

        elif et == "rlimit_warning":
            lines.append(f"{iter_str}  ⚠ RLIMIT WARNING: {ev.get('details', '')}")

        elif et == "early_stop":
            lines.append(f"\n{iter_str}EARLY STOP: {ev['reason']}")

        elif et == "run_complete":
            lines.append("\n=== Run Complete ===")
            lines.append(f"  Baseline: {ev['baseline_value']:.6g}")
            lines.append(f"  Best: {ev['best_value']:.6g} (iter {ev['best_iter']})")
            lines.append(f"  Iterations: {ev['total_iterations']}, LLM calls: {ev['total_llm_calls']}")
            if ev["baseline_value"] and ev["baseline_value"] != 0:
                speedup = ev["best_value"] / ev["baseline_value"]
                lines.append(f"  Overall speedup: {speedup:.2f}x")

        elif et == "roofline_detected":
            lines.append(f"  Roofline: {ev.get('peak_tflops', 0):.3f} TFLOPS, {ev.get('peak_mem_bw_gbs', 0):.1f} GB/s ({ev.get('source', '?')}: {ev.get('device', '?')})")

    return "\n".join(lines)
