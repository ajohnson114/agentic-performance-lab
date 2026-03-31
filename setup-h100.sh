#!/usr/bin/env bash
# =============================================================================
# PerfLab H100 / Linux GPU Cluster Setup
# =============================================================================
# One-shot setup script for rented GPU clusters (Lambda, RunPod, Vast.ai, etc.)
# with NVIDIA H100/A100/L40S GPUs.
#
# Run:   chmod +x setup-h100.sh && ./setup-h100.sh
#
# What it does:
#   1. Checks what's already installed and skips it
#   2. Installs system profiler tools (perf, bpftrace)
#   3. Installs Python profiler tools (py-spy, memray)
#   4. Checks for NVIDIA tools (nsys, ncu, nvidia-smi)
#   5. Installs PerfLab with all task dependencies
#   6. Configures perf permissions for non-root profiling
#   7. Sets NVIDIA GPU persistence mode for stable benchmarks
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
echo " PerfLab H100 Cluster Setup"
echo "=========================================="
echo ""

# --- Check we're on Linux --------------------------------------------------
if [[ "$(uname)" != "Linux" ]]; then
    fail "This script is for Linux GPU clusters. On macOS, use: pip install -e '.[all,tasks-all]' && pip install py-spy memray"
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
        timeout 30 sudo apt-get update -qq 2>/dev/null || warn "apt-get update timed out or failed (non-fatal)"
    fi
fi

# perf — hardware counter profiler (IPC, cache misses, branch stats)
if command -v perf &>/dev/null; then
    skip "perf — hardware counter profiler (IPC, cache misses, branch stats)"
else
    KERNEL_VER=$(uname -r)
    case "$PKG_MGR" in
        apt)
            info "Installing perf — hardware counter profiler (IPC, cache misses, branch stats)"
            sudo apt-get install -y -qq linux-tools-common "linux-tools-${KERNEL_VER}" 2>/dev/null \
                && ok "perf installed" \
                || {
                    # Some cloud images don't have kernel-matched tools; try generic
                    sudo apt-get install -y -qq linux-tools-common linux-tools-generic 2>/dev/null \
                        && ok "perf installed (generic)" \
                        || warn "Failed to install perf — hardware counters won't be available"
                }
            ;;
        dnf|yum)
            info "Installing perf — hardware counter profiler (IPC, cache misses, branch stats)"
            sudo "$PKG_MGR" install -y -q perf 2>/dev/null \
                && ok "perf installed" \
                || warn "Failed to install perf"
            ;;
        *)
            warn "Install perf manually for hardware counter profiling"
            ;;
    esac
fi

# bpftrace — eBPF I/O and syscall tracer
install_pkg bpftrace "eBPF I/O and syscall tracer"

# g++ — needed for C++ tasks
if command -v g++ &>/dev/null; then
    skip "g++ — C++ compiler for cpp tasks"
else
    case "$PKG_MGR" in
        apt) install_pkg g++ "C++ compiler for cpp tasks" ;;
        dnf|yum) install_pkg gcc-c++ "C++ compiler for cpp tasks" ;;
        *) warn "Install g++ manually for C++ tasks" ;;
    esac
fi

# --- NVIDIA tools -----------------------------------------------------------
echo ""
echo "Checking NVIDIA tools..."

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    ok "nvidia-smi — detected: $GPU_NAME"
else
    warn "nvidia-smi not found — NVIDIA drivers may not be installed"
fi

if command -v nvcc &>/dev/null; then
    NVCC_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    ok "nvcc — CUDA compiler $NVCC_VER"
else
    warn "nvcc not found — install CUDA toolkit for CUDA tasks"
fi

if command -v nsys &>/dev/null; then
    skip "nsys — NVIDIA GPU timeline profiler (kernel launches, memory transfers)"
else
    warn "nsys not found — install NVIDIA Nsight Systems for GPU timeline profiling"
    info "Download from: https://developer.nvidia.com/nsight-systems"
    info "Or: sudo apt install nsight-systems (if NVIDIA apt repo is configured)"
    # Try to install from NVIDIA apt repo if available
    if [[ "$PKG_MGR" == "apt" ]]; then
        sudo apt-get install -y -qq nsight-systems 2>/dev/null && ok "nsys installed" || true
    fi
fi

if command -v ncu &>/dev/null; then
    skip "ncu — NVIDIA GPU kernel profiler (SM utilization, memory throughput)"
else
    warn "ncu not found — install NVIDIA Nsight Compute for GPU kernel profiling"
    info "Download from: https://developer.nvidia.com/nsight-compute"
    info "Or: sudo apt install nsight-compute (if NVIDIA apt repo is configured)"
    if [[ "$PKG_MGR" == "apt" ]]; then
        sudo apt-get install -y -qq nsight-compute 2>/dev/null && ok "ncu installed" || true
    fi
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

# pmu-tools / toplev — Intel TMA Level 2/3 analysis (L1/L2/L3/DRAM Bound breakdown)
if command -v toplev &>/dev/null || python3 -c "import pmu" 2>/dev/null; then
    skip "pmu-tools/toplev — Intel TMA Level 2/3 analysis"
else
    info "Installing pmu-tools — Intel TMA Level 2/3 analysis (toplev)"
    python3 -m pip install pmu-tools 2>/dev/null && ok "pmu-tools installed" || warn "Failed to install pmu-tools (non-fatal — TMA Level 1 still works)"
fi

# cuobjdump — CUDA SASS disassembly (usually bundled with CUDA toolkit)
if command -v cuobjdump &>/dev/null; then
    skip "cuobjdump — CUDA SASS disassembly for kernel instruction analysis"
else
    warn "cuobjdump not found — SASS disassembly won't be available (install CUDA toolkit)"
fi

# c++filt — C++ name demangling (usually bundled with binutils)
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

# --- Install PerfLab --------------------------------------------------------
echo ""
echo "Installing PerfLab with all dependencies..."

cd "$SCRIPT_DIR"

if python3 -m pip show perflab &>/dev/null; then
    info "PerfLab already installed — reinstalling to pick up latest changes"
fi

python3 -m pip install -e ".[all,tasks-all]" 2>&1 | tail -3
ok "PerfLab installed with all LLM providers and task dependencies"

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

# --- NVIDIA GPU tuning for stable benchmarks --------------------------------
echo ""
echo "Configuring GPU for stable benchmarks..."

if command -v nvidia-smi &>/dev/null; then
    # Persistence mode — keeps GPU initialized between runs, avoids cold-start jitter
    PERSIST=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null | head -1)
    if [[ "$PERSIST" == "Enabled" ]]; then
        skip "GPU persistence mode already enabled"
    else
        info "Enabling GPU persistence mode (avoids cold-start jitter between runs)"
        sudo nvidia-smi -pm 1 2>/dev/null \
            && ok "GPU persistence mode enabled" \
            || warn "Could not enable persistence mode — benchmarks may have cold-start variance"
    fi

    # Check for ECC mode (informational)
    ECC=$(nvidia-smi --query-gpu=ecc.mode.current --format=csv,noheader 2>/dev/null | head -1)
    if [[ "$ECC" == "Enabled" ]]; then
        info "ECC memory is enabled — this is normal for H100/A100 datacenter GPUs"
    fi

    # Lock GPU clocks at max frequency for stable benchmarks
    # This prevents boost/throttle variance (~22% on H100) between runs
    MAX_SM=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [[ -n "$MAX_SM" && "$MAX_SM" != "[N/A]" ]]; then
        info "Locking GPU SM clocks to ${MAX_SM} MHz for stable benchmarks"
        sudo nvidia-smi -lgc "$MAX_SM","$MAX_SM" 2>/dev/null \
            && ok "GPU clocks locked at ${MAX_SM} MHz" \
            || warn "Could not lock GPU clocks — benchmark variance may be higher"
    else
        warn "Could not query max SM clock — skipping clock lock"
    fi

    # Pin to first GPU on multi-GPU nodes to avoid interference
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    if [[ "$GPU_COUNT" -gt 1 ]]; then
        info "Multi-GPU node detected ($GPU_COUNT GPUs)"
        info "Setting CUDA_VISIBLE_DEVICES=0 in /etc/environment for benchmark isolation"
        if ! grep -q "CUDA_VISIBLE_DEVICES" /etc/environment 2>/dev/null; then
            echo "CUDA_VISIBLE_DEVICES=0" | sudo tee -a /etc/environment >/dev/null 2>&1 \
                && ok "CUDA_VISIBLE_DEVICES=0 written to /etc/environment" \
                || warn "Could not write /etc/environment — set CUDA_VISIBLE_DEVICES=0 manually"
        else
            skip "CUDA_VISIBLE_DEVICES already set in /etc/environment"
        fi
        export CUDA_VISIBLE_DEVICES=0
        ok "CUDA_VISIBLE_DEVICES=0 for this session"
    fi

    # Report current clocks
    GPU_CLOCK=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader 2>/dev/null | head -1)
    MEM_CLOCK=$(nvidia-smi --query-gpu=clocks.mem --format=csv,noheader 2>/dev/null | head -1)
    info "Current GPU clocks: SM=$GPU_CLOCK, Memory=$MEM_CLOCK"

    # Check GPU thermal state
    GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    GPU_POWER=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [[ -n "$GPU_TEMP" ]]; then
        info "GPU temperature: ${GPU_TEMP}°C, power draw: ${GPU_POWER}W"
        if [[ "$GPU_TEMP" -gt 80 ]]; then
            warn "GPU temperature is ${GPU_TEMP}°C — thermal throttling likely. Let GPU cool before benchmarking."
        fi
    fi
else
    skip "No NVIDIA GPU detected — skipping GPU configuration"
fi

# --- Run perflab doctor -----------------------------------------------------
echo ""
echo "=========================================="
echo " Running perflab doctor"
echo "=========================================="
echo ""

perflab doctor --all || true

# --- Summary ----------------------------------------------------------------
echo ""
echo "=========================================="
echo " Setup complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Configure your LLM provider:"
echo "       perflab init"
echo ""
echo "  2. Run your first optimization:"
echo "       perflab agent tasks/matmul/cuda/task.yaml"
echo ""
echo "  3. For H100-specific tasks:"
echo "       perflab agent tasks/matmul/cuda_h100/task.yaml"
echo ""
echo "  4. Review results:"
echo "       perflab list-runs"
echo "       perflab replay out/runs/<run_id>/"
echo ""
echo "  5. When done benchmarking, unlock GPU clocks:"
echo "       sudo nvidia-smi -rgc"
echo ""
