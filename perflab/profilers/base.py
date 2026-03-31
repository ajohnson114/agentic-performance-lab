from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

@dataclass
class ProfileResult:
    name: str
    artifacts: dict[str, str]  # logical_name -> relative path
    summary: dict

class Profiler(Protocol):
    name: str
    def is_available(self) -> bool: ...
    def run(self, bench_cmd: str, cwd: Path, artifacts_dir: Path) -> ProfileResult: ...
