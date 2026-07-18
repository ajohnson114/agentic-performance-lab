"""Shared symbol-name helpers: kernel base-name extraction and C++ demangling.

Consolidates logic previously duplicated across gpu_attribution,
compiler_diagnostics, linux_perf, and ncu_profiler. Comparison semantics
(case handling, containment direction) stay at the call sites; only the
name extraction and demangling live here.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from functools import lru_cache

# Itanium-mangled symbol: _Z<len><identifier>... group(1) captures the
# identifier plus trailing parameter codes — enough for substring matching.
_MANGLED_NAME_RE = re.compile(r"_Z\d+(\w+)")


def kernel_base_name(name: str, *, strip_params: bool = True) -> str:
    """Extract the base symbol name from a kernel/function name.

    Strips namespaces (``ns::name``), template arguments (``name<T>``) and,
    unless *strip_params* is false, the parameter list (``name(int, ...)``).
    """
    base = name.split("::")[-1].split("<")[0]
    if strip_params:
        base = base.split("(")[0]
    return base


def mangled_base_name(name: str) -> str | None:
    """Best-effort identifier extraction from a mangled C++ symbol.

    Returns the identifier (plus trailing parameter codes) embedded in an
    Itanium ``_Z<len><identifier>`` symbol, or None if *name* does not look
    mangled. No c++filt involved.
    """
    m = _MANGLED_NAME_RE.match(name)
    return m.group(1) if m else None


@lru_cache(maxsize=1)
def cxxfilt_available() -> bool:
    """Return True if c++filt is on PATH."""
    return shutil.which("c++filt") is not None


@lru_cache(maxsize=512)
def demangle(name: str, *, base_name_fallback: bool = False) -> str:
    """Demangle a C++ symbol using c++filt if available.

    Names without the ``_Z`` mangling prefix are returned unchanged without
    spawning a subprocess. When c++filt is missing or fails, falls back to
    ``mangled_base_name`` if *base_name_fallback* is true, otherwise returns
    the original name. Results are cached so repeated calls for the same
    symbol avoid spawning a subprocess.
    """
    if not name.startswith("_Z"):
        return name
    if cxxfilt_available():
        try:
            result = subprocess.run(
                ["c++filt", name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            demangled = result.stdout.strip()
            if demangled:
                return demangled
        except (OSError, subprocess.TimeoutExpired):
            pass
    if base_name_fallback:
        return mangled_base_name(name) or name
    return name
