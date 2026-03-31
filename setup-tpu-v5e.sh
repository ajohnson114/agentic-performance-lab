#!/usr/bin/env bash
# =============================================================================
# PerfLab TPU v5e Setup Script
# =============================================================================
# One-shot setup for Google Cloud TPU VMs (v5e, v4, v5p, v6e).
#
# Run:   chmod +x setup-tpu-v5e.sh && ./setup-tpu-v5e.sh
#
# Designed for:
#   - Google Cloud TPU VMs (gcloud compute tpus tpu-vm create ...)
#   - Single-host TPU pods (v5e-4, v5e-8, etc.)
#
# What it does:
#   1. Verifies TPU is accessible via JAX
#   2. Installs system profiler tools (perf, bpftrace)
#   3. Installs Python profiler tools (py-spy, memray, pmu-tools)
#   4. Installs JAX with TPU support if needed
#   5. Installs PerfLab with JAX task dependencies
#   6. Configures perf permissions for non-root profiling
#   7. Configures TPU environment for stable benchmarks
#   8. Runs perflab doctor to verify everything
# =============================================================================

set -euo pipefail

# --- Colors ---------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

ok()   { echo -e "  [${GREEN} OK ${NC}] $1"; }
skip() { echo -e "  [${BLUE}SKIP${NC}] $1 (already installed)"; }
warn() { echo -e "  [${YELLOW}WARN${NC}] $1"; }
fail() { echo -e "  [${RED}FAIL${NC}] $1"; }
info() { echo -e "  [${BLUE}INFO${NC}] $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "=========================================="
echo " PerfLab TPU Setup"
echo "=========================================="
echo ""

# --- Check we're on Linux --------------------------------------------------
if [[ "$(uname)" != "Linux" ]]; then
    fail "TPU VMs run Linux. This script is for Google Cloud TPU VMs."
    exit 1
fi

# --- Check Python -----------------------------------------------------------
echo "Checking Python..."
if command -v python3 &>/dev/null; then
    PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
    if [[ "$PY_MAJOR" -ge 3 && "$PY_MINOR" -ge 10 ]]; then
        ok "Python $PY_VER"
    else
        fail "Python $PY_VER found but PerfLab requires >= 3.10"
        exit 1
    fi
else
    fail "python3 not found"
    exit 1
fi

# --- Check/Install JAX with TPU support ------------------------------------
echo ""
echo "Checking JAX + TPU..."

JAX_OK=false
if python3 -c "import jax; d = jax.devices(); tpus = [x for x in d if x.platform == 'tpu']; assert len(tpus) > 0" 2>/dev/null; then
    TPU_INFO=$(python3 -c "
import jax
d = [x for x in jax.devices() if x.platform == 'tpu']
print(f'{d[0].device_kind} — {len(d)} chip(s)')
print(f'JAX version: {jax.__version__}')
" 2>/dev/null)
    ok "JAX sees TPU devices"
    info "$TPU_INFO"
    JAX_OK=true
else
    warn "JAX cannot see TPU devices — installing jax[tpu]"
    info "Installing JAX with TPU support..."
    python3 -m pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html 2>&1 | tail -3

    # Verify
    if python3 -c "import jax; d = jax.devices(); tpus = [x for x in d if x.platform == 'tpu']; assert len(tpus) > 0" 2>/dev/null; then
        ok "JAX + TPU installed successfully"
        JAX_OK=true
    else
        fail "JAX installed but cannot see TPU — check TPU VM configuration"
        info "Ensure you created a TPU VM (not just a GCE VM):"
        info "  gcloud compute tpus tpu-vm create my-tpu --zone=us-central2-b --accelerator-type=v5litepod-4 --version=tpu-ubuntu2204-base"
        info ""
        info "If using an existing VM, check that libtpu is accessible:"
        info "  ls /usr/share/tpu/tpu-env-setup.sh && source /usr/share/tpu/tpu-env-setup.sh"
    fi
fi

# --- System packages --------------------------------------------------------
echo ""
echo "Installing system profiler tools..."

# Detect package manager
PKG_MGR=""
if command -v apt-get &>/dev/null; then
    PKG_MGR="apt"
elif command -v dnf &>/dev/null; then
    PKG_MGR="dnf"
elif command -v yum &>/dev/null; then
    PKG_MGR="yum"
fi

install_pkg() {
    local pkg="$1"
    local desc="$2"
    if command -v "$pkg" &>/dev/null; then
        skip "$pkg — $desc"
        return 0
    fi
    info "Installing $pkg — $desc"
    case "$PKG_MGR" in
        apt)
            sudo apt-get install -y -qq "$@" 2>/dev/null && ok "$pkg installed" || warn "Failed to install $pkg (non-fatal)"
            ;;
        dnf)
            sudo dnf install -y -q "$@" 2>/dev/null && ok "$pkg installed" || warn "Failed to install $pkg (non-fatal)"
            ;;
        yum)
            sudo yum install -y -q "$@" 2>/dev/null && ok "$pkg installed" || warn "Failed to install $pkg (non-fatal)"
            ;;
        *)
            warn "No supported package manager found — install $pkg manually"
            ;;
    esac
}

# Update package lists (apt only, and only once)
if [[ "$PKG_MGR" == "apt" ]]; then
    if ! command -v perf &>/dev/null || ! command -v bpftrace &>/dev/null; then
        info "Updating package lists..."
        sudo apt-get update -qq 2>/dev/null
    fi
fi

# perf — hardware counter profiler (useful for host-side CPU analysis)
if command -v perf &>/dev/null; then
    skip "perf — hardware counter profiler (host-side CPU analysis)"
else
    KERNEL_VER=$(uname -r)
    case "$PKG_MGR" in
        apt)
            info "Installing perf — hardware counter profiler (host-side CPU analysis)"
            sudo apt-get install -y -qq linux-tools-common "linux-tools-${KERNEL_VER}" 2>/dev/null \
                && ok "perf installed" \
                || {
                    # Some cloud images don't have kernel-matched tools; try generic
                    sudo apt-get install -y -qq linux-tools-common linux-tools-generic 2>/dev/null \
                        && ok "perf installed (generic)" \
                        || warn "Failed to install perf — not critical for TPU workloads"
                }
            ;;
        dnf|yum)
            info "Installing perf — hardware counter profiler (host-side CPU analysis)"
            sudo "$PKG_MGR" install -y -q perf 2>/dev/null \
                && ok "perf installed" \
                || warn "Failed to install perf"
            ;;
        *)
            warn "Install perf manually for hardware counter profiling"
            ;;
    esac
fi

# bpftrace — eBPF I/O and syscall tracer (useful for infeed stall diagnosis)
install_pkg bpftrace "eBPF I/O and syscall tracer (infeed stall diagnosis)"

# c++filt — C++ name demangling (useful for XLA kernel name resolution)
if command -v c++filt &>/dev/null; then
    skip "c++filt — C++ name demangling for kernel name resolution"
else
    info "Installing binutils for c++filt (C++ name demangling)"
    case "$PKG_MGR" in
        apt) sudo apt-get install -y -qq binutils 2>/dev/null && ok "binutils installed" || true ;;
        dnf|yum) sudo "$PKG_MGR" install -y -q binutils 2>/dev/null && ok "binutils installed" || true ;;
        *) warn "Install binutils manually for c++filt" ;;
    esac
fi

# --- Python profiler tools --------------------------------------------------
echo ""
echo "Installing Python profiler tools..."

# py-spy — CPU flame graph profiler for Python
if command -v py-spy &>/dev/null; then
    skip "py-spy — CPU flame graph profiler for Python"
else
    info "Installing py-spy — CPU flame graph profiler for Python"
    python3 -m pip install py-spy 2>/dev/null && ok "py-spy installed" || warn "Failed to install py-spy"
fi

# memray — Python memory allocation profiler
if python3 -c "import memray" 2>/dev/null; then
    skip "memray — Python memory allocation profiler"
else
    info "Installing memray — Python memory allocation profiler"
    python3 -m pip install memray 2>/dev/null && ok "memray installed" || warn "Failed to install memray"
fi

# pmu-tools / toplev — Intel TMA Level 2/3 analysis (TPU VMs have Intel host CPUs)
if command -v toplev &>/dev/null || python3 -c "import pmu" 2>/dev/null; then
    skip "pmu-tools/toplev — Intel TMA Level 2/3 analysis"
else
    info "Installing pmu-tools — Intel TMA Level 2/3 analysis (host-side CPU)"
    python3 -m pip install pmu-tools 2>/dev/null && ok "pmu-tools installed" || warn "Failed to install pmu-tools (non-fatal — TMA Level 1 still works)"
fi

# --- Configure perf permissions ---------------------------------------------
echo ""
echo "Configuring profiler permissions..."

# Allow perf for non-root users (common requirement on cloud instances)
PERF_PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "unknown")
if [[ "$PERF_PARANOID" != "unknown" && "$PERF_PARANOID" -gt 1 ]]; then
    info "Setting perf_event_paranoid to 1 (allows non-root perf stat/record)"
    sudo sysctl -w kernel.perf_event_paranoid=1 2>/dev/null \
        && ok "perf_event_paranoid set to 1" \
        || warn "Could not set perf_event_paranoid — run perf with sudo"
else
    skip "perf_event_paranoid already <= 1"
fi

# --- TPU environment tuning ------------------------------------------------
echo ""
echo "Configuring TPU environment for stable benchmarks..."

# XLA compilation cache — avoids recompilation across runs
if [[ -z "${XLA_FLAGS:-}" ]] || [[ "$XLA_FLAGS" != *"xla_jit_compilation_cache"* ]]; then
    CACHE_DIR="${HOME}/.cache/xla_compilation_cache"
    mkdir -p "$CACHE_DIR"
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_jit_compilation_cache_dir=$CACHE_DIR"
    info "XLA compilation cache: $CACHE_DIR"
    # Persist for future shells
    if ! grep -q "xla_jit_compilation_cache" ~/.bashrc 2>/dev/null; then
        echo "export XLA_FLAGS=\"\${XLA_FLAGS:-} --xla_jit_compilation_cache_dir=$CACHE_DIR\"" >> ~/.bashrc
        ok "XLA compilation cache added to ~/.bashrc"
    else
        skip "XLA compilation cache already in ~/.bashrc"
    fi
else
    skip "XLA compilation cache already configured"
fi

# Persistent cache min compile time — cache only non-trivial compilations
# Reduces noise from small recompilations during benchmarking
if [[ -z "${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-}" ]]; then
    export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
    if ! grep -q "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" ~/.bashrc 2>/dev/null; then
        echo 'export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1' >> ~/.bashrc
        ok "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 added to ~/.bashrc"
    fi
else
    skip "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS already set: $JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"
fi

# Default to bf16 matmul precision for TPU (2x throughput)
if [[ -z "${JAX_DEFAULT_MATMUL_PRECISION:-}" ]]; then
    export JAX_DEFAULT_MATMUL_PRECISION="bfloat16"
    if ! grep -q "JAX_DEFAULT_MATMUL_PRECISION" ~/.bashrc 2>/dev/null; then
        echo 'export JAX_DEFAULT_MATMUL_PRECISION="bfloat16"' >> ~/.bashrc
        ok "JAX_DEFAULT_MATMUL_PRECISION=bfloat16 added to ~/.bashrc"
    fi
else
    skip "JAX_DEFAULT_MATMUL_PRECISION already set: $JAX_DEFAULT_MATMUL_PRECISION"
fi

# Report TPU chip topology and thermal state
if $JAX_OK; then
    echo ""
    echo "Checking TPU health..."
    TPU_HEALTH=$(python3 -c "
import jax
d = [x for x in jax.devices() if x.platform == 'tpu']
chip = d[0].device_kind if d else 'unknown'
n = len(d)
topology = f'{n}-chip pod slice' if n > 1 else 'single chip'
print(f'{chip} — {topology}')
if n > 1:
    print(f'Chips: {n} (benchmarks will use chip 0 by default)')
" 2>/dev/null) && info "$TPU_HEALTH" || true
fi

# --- Install PerfLab --------------------------------------------------------
echo ""
echo "Installing PerfLab..."

cd "$SCRIPT_DIR"

if python3 -m pip show perflab &>/dev/null; then
    info "PerfLab already installed — reinstalling to pick up latest changes"
fi

python3 -m pip install -e ".[all,tasks-jax,profiling]" 2>&1 | tail -3
ok "PerfLab installed with JAX dependencies and profiling tools"

# Also install optax for training tasks
python3 -m pip install optax 2>/dev/null && ok "optax installed" || warn "optax install failed (needed for training tasks)"

# --- Run perflab doctor -----------------------------------------------------
echo ""
echo "=========================================="
echo " Running perflab doctor"
echo "=========================================="
echo ""

perflab doctor --all || true

# --- TPU info summary -------------------------------------------------------
echo ""
echo "=========================================="
echo " Setup complete"
echo "=========================================="
echo ""

if $JAX_OK; then
    TPU_SUMMARY=$(python3 -c "
import jax
d = [x for x in jax.devices() if x.platform == 'tpu']
chip = d[0].device_kind if d else 'unknown'
print(f'  TPU: {chip} ({len(d)} chips)')
print(f'  JAX: {jax.__version__}')
print(f'  Backend: {jax.default_backend()}')
" 2>/dev/null || echo "  (could not query TPU)")
    echo "$TPU_SUMMARY"
    echo ""
fi

echo "Next steps:"
echo ""
echo "  1. Configure your LLM provider:"
echo "       perflab init"
echo ""
echo "  2. Run the TPU attention demo (great for demos!):"
echo "       perflab agent tasks/attention/jax_tpu/task.yaml"
echo ""
echo "  3. Run JAX transformer training optimization:"
echo "       perflab agent tasks/transformer_train/jax/task.yaml"
echo ""
echo "  4. Run JAX matmul optimization:"
echo "       perflab agent tasks/matmul/jax/task.yaml"
echo ""
echo "  5. Review results:"
echo "       perflab list-runs"
echo "       open out/runs/<run_id>/dashboard.html"
echo ""
echo "  6. When done benchmarking, clear XLA compilation cache:"
echo "       rm -rf ~/.cache/xla_compilation_cache"
echo ""
echo "  Tip: The attention demo starts with naive fp32, no-jit attention."
echo "  The agent should discover @jax.jit, bf16, and vectorized heads"
echo "  for a 10-50x speedup on TPU."
echo ""
