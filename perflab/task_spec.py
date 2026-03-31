from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import yaml

from perflab.analyzers.bottleneck_analyzer import AnalysisThresholds

ProgramType = Literal["python", "pytorch", "jax", "triton", "cpp", "cuda"]

@dataclass
class CommandSpec:
    cmd: str
    expected_exit: int = 0

@dataclass
class MetricSpec:
    name: str
    mode: Literal["maximize", "minimize"] = "maximize"

@dataclass
class SecondaryMetricSpec:
    """Optional secondary metric for multi-objective Pareto optimization."""
    name: str
    mode: Literal["maximize", "minimize"] = "minimize"

@dataclass
class BenchmarkSpec:
    cmd: str
    metric: MetricSpec
    warmup: int = 3
    repeats: int = 20
    secondary_metric: SecondaryMetricSpec | None = None

@dataclass
class ProfilePlan:
    always: list[str] = field(default_factory=list)
    optional: list[str] = field(default_factory=list)

@dataclass
class Constraints:
    max_iters: int = 10
    regression_tolerance: float = 0.02
    rlimit_as_gb: float | None = None  # None = auto (4 GB for CPU, disabled for GPU)
    prompt_token_budget: int = 0  # 0 means unlimited
    top_n: int = 3  # number of bottleneck diagnoses to report
    max_history: int = 3  # number of recent iterations to include in LLM prompt
    allow_fast_math: bool = False  # allow -ffast-math, --use_fast_math (breaks IEEE compliance)
    accuracy_tolerance: str | None = None  # "exact", "1e-3", "1e-1" — how much error is acceptable

@dataclass
class RooflineSpec:
    peak_tflops: float
    peak_mem_bw_gbs: float
    title: str | None = None
    peak_fp16_tflops: float | None = None
    dtype_peaks: dict[str, float] | None = None  # per-dtype peaks (e.g. peak_tflops_fp16, peak_tflops_tf32)

@dataclass
class EditPolicy:
    allowed_paths: list[str] = field(default_factory=list)

@dataclass
class ContractSpec:
    fixed_params: dict[str, int | float] = field(default_factory=dict)
    min_repeats: int = 1
    min_warmup: int = 0
    required_bench_fields: list[str] = field(default_factory=lambda: ["ok"])

    def validate(self) -> list[str]:
        """Validate contract structure. Returns list of error strings (empty = valid)."""
        errors: list[str] = []
        for fp in self.required_bench_fields:
            if not fp or not isinstance(fp, str):
                errors.append(f"required_bench_fields contains invalid entry: {fp!r}")
            elif ".." in fp or fp.startswith(".") or fp.endswith("."):
                errors.append(f"required_bench_fields has malformed dotted path: {fp!r}")
        for key, val in self.fixed_params.items():
            if not key or not isinstance(key, str):
                errors.append(f"fixed_params contains invalid key: {key!r}")
            if not isinstance(val, (int, float)):
                errors.append(f"fixed_params[{key!r}] must be int or float, got {type(val).__name__}")
        if self.min_repeats < 0:
            errors.append(f"min_repeats must be >= 0, got {self.min_repeats}")
        if self.min_warmup < 0:
            errors.append(f"min_warmup must be >= 0, got {self.min_warmup}")
        return errors

@dataclass
class AntiGamingSpec:
    """Configuration for reward-hack mitigations.

    These checks defend against LLM-generated code that games benchmarks
    rather than genuinely optimizing performance.

    All checks are enabled by default and can be disabled per-task if they
    produce false positives for specific workloads.
    """
    # Framework-level checks (run automatically by the agent)
    bench_variance_check: bool = True      # Detect zero-variance timing arrays (caching)
    determinism_rerun: bool = True          # Run correctness twice with different seed
    gaming_speedup_threshold: float = 100.0  # Warn if single-iteration speedup exceeds this
    # Thread monitoring (via bench.json protocol)
    thread_count_check: bool = False        # Check bench.json thread_delta field (opt-in)
    max_thread_delta: int = 0               # Allowed new threads during kernel execution


@dataclass
class DataHints:
    """Hints about the data characteristics to guide algorithmic optimization.

    These don't affect profiling — they help the LLM make better algorithmic
    suggestions (e.g., "data is 95% sparse → consider sparse format").
    """
    sparsity: float | None = None         # fraction of zeros (0.0-1.0), e.g., 0.95
    value_range: list[float] | None = None  # [min, max] of typical values
    access_pattern: str | None = None     # "sequential", "random", "strided", "blocked"
    batch_size_range: list[int] | None = None  # [min, max] production batch sizes
    dtype_safety: str | None = None       # "fp16_safe", "bf16_safe", "int8_safe" — precision reduction won't hurt accuracy
    sequence_lengths: str | None = None   # "fixed", "variable_32_2048", etc.
    custom: list[str] | None = None       # free-form hints: ["data is symmetric", "output is sparse"]


@dataclass
class AgentSpec:
    n_candidates: int = 6
    top_k: int = 2
    max_iters: int = 12

@dataclass
class TaskSpec:
    name: str
    workspace: Path
    program_type: ProgramType

    build: CommandSpec | None
    correctness: CommandSpec
    benchmark: BenchmarkSpec
    target_hardware: str | None = None
    profile_plan: ProfilePlan = field(default_factory=ProfilePlan)
    constraints: Constraints = field(default_factory=Constraints)
    roofline: RooflineSpec | None = None
    edit_policy: EditPolicy = field(default_factory=EditPolicy)
    contract: ContractSpec = field(default_factory=ContractSpec)
    anti_gaming: AntiGamingSpec = field(default_factory=AntiGamingSpec)
    agent: AgentSpec = field(default_factory=AgentSpec)
    analysis_thresholds: AnalysisThresholds = field(default_factory=AnalysisThresholds)
    data_hints: DataHints = field(default_factory=DataHints)

    out_dir: Path = Path("out")  # relative to repo root by default

    @staticmethod
    def load(path: str | Path) -> "TaskSpec":
        p = Path(path)
        data = yaml.safe_load(p.read_text(encoding="utf-8"))

        def cmd_or_none(x: Any) -> CommandSpec | None:
            if x is None:
                return None
            return CommandSpec(cmd=str(x.get("cmd")), expected_exit=int(x.get("expected_exit", 0)))

        # workspace field is for documentation; the YAML file lives in the
        # workspace directory, so p.parent is the canonical workspace path.
        ws = p.parent.resolve()
        build = cmd_or_none(data.get("build"))

        correctness = CommandSpec(
            cmd=str(data["correctness"]["cmd"]),
            expected_exit=int(data["correctness"].get("expected_exit", 0)),
        )

        metric = MetricSpec(
            name=str(data["benchmark"]["metric"]["name"]),
            mode=str(data["benchmark"]["metric"].get("mode", "maximize")),
        )
        secondary_metric = None
        sec_data = data["benchmark"].get("secondary_metric")
        if sec_data:
            secondary_metric = SecondaryMetricSpec(
                name=str(sec_data["name"]),
                mode=str(sec_data.get("mode", "minimize")),
            )

        benchmark = BenchmarkSpec(
            cmd=str(data["benchmark"]["cmd"]),
            metric=metric,
            warmup=int(data["benchmark"].get("warmup", 3)),
            repeats=int(data["benchmark"].get("repeats", 20)),
            secondary_metric=secondary_metric,
        )

        profile_plan = ProfilePlan(
            always=list(data.get("profile_plan", {}).get("always", [])),
            optional=list(data.get("profile_plan", {}).get("optional", [])),
        )

        constraints_data = data.get("constraints", {}) or {}
        rlimit_raw = constraints_data.get("rlimit_as_gb")

        # Global config defaults for prompt settings (task.yaml overrides these)
        _cfg_max_history = 3
        _cfg_token_budget = 0
        try:
            from perflab.config import load_config
            _ga = load_config().agent
            _cfg_max_history = _ga.max_history
            _cfg_token_budget = _ga.prompt_token_budget
        except Exception:
            pass

        constraints = Constraints(
            max_iters=int(constraints_data.get("max_iters", 10)),
            regression_tolerance=float(constraints_data.get("regression_tolerance", 0.02)),
            rlimit_as_gb=float(rlimit_raw) if rlimit_raw is not None else None,
            prompt_token_budget=int(constraints_data.get("prompt_token_budget", _cfg_token_budget)),
            top_n=int(constraints_data.get("top_n", 3)),
            max_history=int(constraints_data.get("max_history", _cfg_max_history)),
        )

        edit_policy = EditPolicy(
            allowed_paths=list(data.get("edit_policy", {}).get("allowed_paths", []))
        )

        roofline = None
        if "roofline" in data and data["roofline"] is not None:
            raw_fp16 = data["roofline"].get("peak_fp16_tflops")
            raw_dtype_peaks = data["roofline"].get("dtype_peaks")
            dtype_peaks = None
            if raw_dtype_peaks and isinstance(raw_dtype_peaks, dict):
                dtype_peaks = {str(k): float(v) for k, v in raw_dtype_peaks.items()}
            roofline = RooflineSpec(
                peak_tflops=float(data["roofline"]["peak_tflops"]),
                peak_mem_bw_gbs=float(data["roofline"]["peak_mem_bw_gbs"]),
                title=data["roofline"].get("title"),
                peak_fp16_tflops=float(raw_fp16) if raw_fp16 is not None else None,
                dtype_peaks=dtype_peaks,
            )

        contract_data = data.get("contract", {}) or {}
        contract = ContractSpec(
            fixed_params=dict(contract_data.get("fixed_params", {})),
            min_repeats=int(contract_data.get("min_repeats", 1)),
            min_warmup=int(contract_data.get("min_warmup", 0)),
            required_bench_fields=list(contract_data.get("required_bench_fields", ["ok"])),
        )

        ag_data = data.get("anti_gaming", {}) or {}
        anti_gaming = AntiGamingSpec(
            bench_variance_check=bool(ag_data.get("bench_variance_check", True)),
            determinism_rerun=bool(ag_data.get("determinism_rerun", True)),
            gaming_speedup_threshold=float(ag_data.get("gaming_speedup_threshold", 100.0)),
            thread_count_check=bool(ag_data.get("thread_count_check", False)),
            max_thread_delta=int(ag_data.get("max_thread_delta", 0)),
        )

        agent_data = data.get("agent", {}) or {}
        agent = AgentSpec(
            n_candidates=int(agent_data.get("n_candidates", AgentSpec.n_candidates)),
            top_k=int(agent_data.get("top_k", AgentSpec.top_k)),
            max_iters=int(agent_data.get("max_iters", AgentSpec.max_iters)),
        )

        # Merge threshold overrides: global config → task.yaml (task wins)
        merged_thresh: dict = {}
        try:
            from perflab.config import load_config
            merged_thresh.update(load_config().analysis_thresholds)
        except Exception:
            pass  # Config loading is optional; defaults still apply
        task_thresh = data.get("analysis_thresholds", {}) or {}
        merged_thresh.update(task_thresh)  # task.yaml overrides config

        thresh_kwargs = {}
        for f in dataclasses.fields(AnalysisThresholds):
            if f.name in merged_thresh:
                # f.type is a string due to __future__ annotations; cast via default's type
                thresh_kwargs[f.name] = type(f.default)(merged_thresh[f.name])
        analysis_thresholds = AnalysisThresholds(**thresh_kwargs)

        # Data hints (optional)
        dh_data = data.get("data_hints", {}) or {}
        data_hints = DataHints(
            sparsity=float(dh_data["sparsity"]) if "sparsity" in dh_data else None,
            value_range=list(dh_data["value_range"]) if "value_range" in dh_data else None,
            access_pattern=str(dh_data["access_pattern"]) if "access_pattern" in dh_data else None,
            batch_size_range=list(dh_data["batch_size_range"]) if "batch_size_range" in dh_data else None,
            dtype_safety=str(dh_data["dtype_safety"]) if "dtype_safety" in dh_data else None,
            sequence_lengths=str(dh_data["sequence_lengths"]) if "sequence_lengths" in dh_data else None,
            custom=list(dh_data["custom"]) if "custom" in dh_data else None,
        )

        out_dir = (ws / Path(data.get("out_dir", "out"))).resolve()

        return TaskSpec(
            name=str(data["name"]),
            workspace=ws,
            program_type=str(data["program_type"]),
            target_hardware=data.get("target_hardware"),
            build=build,
            correctness=correctness,
            benchmark=benchmark,
            profile_plan=profile_plan,
            constraints=constraints,
            roofline=roofline,
            edit_policy=edit_policy,
            contract=contract,
            anti_gaming=anti_gaming,
            agent=agent,
            analysis_thresholds=analysis_thresholds,
            data_hints=data_hints,
            out_dir=out_dir,
        )
