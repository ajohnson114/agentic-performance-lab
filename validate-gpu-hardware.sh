#!/usr/bin/env bash
# =============================================================================
# PerfLab Real-Hardware Validation
# =============================================================================
# docs/ENGINEERING_RATIONALE.md's "Validation Coverage" section flags a real
# gap: no GPU or TPU has ever been in the loop for an automated test of this
# repository. The ~3800 lines of GPU/TPU profiler-parsing code are validated
# only against hand-written fixtures approximating a format observed once --
# a profiler renaming a column, or a metric that no longer exists, is
# invisible to that suite by construction.
#
# This script is the occasional/manual real-hardware pass that closes the
# gap: it runs the full pytest suite plus perflab's actual profiling
# pipeline against whatever GPU/TPU hardware is visible, then feeds the real
# output through perflab's own parsing code so a human can eyeball whether
# it still looks sane. It is a diagnostic pass, not a CI gate.
#
# Run this after ./setup-h100.sh (or setup-tpu-v5e.sh) on a rented box --
# ideally one with 2+ GPUs, so the multi-GPU/NCCL path gets exercised too:
#
#   chmod +x validate-gpu-hardware.sh && ./validate-gpu-hardware.sh
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ok()   { echo -e "  [${GREEN} OK ${NC}] $1"; }
warn() { echo -e "  [${YELLOW}WARN${NC}] $1"; }
fail() { echo -e "  [${RED}FAIL${NC}] $1"; }
info() { echo -e "  [${BLUE}INFO${NC}] $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "=========================================="
echo " PerfLab Real-Hardware Validation"
echo "=========================================="
echo ""

if ! command -v perflab &>/dev/null; then
    fail "perflab not found on PATH. Run ./setup-h100.sh (or setup-tpu-v5e.sh) first."
    exit 1
fi
ok "perflab found: $(command -v perflab)"

echo ""
echo "Environment:"
command -v nvidia-smi &>/dev/null && nvidia-smi -L | sed 's/^/  /' || info "nvidia-smi not found (no NVIDIA GPU, or TPU host)"
command -v nsys &>/dev/null && info "nsys: $(nsys --version 2>&1 | head -1)" || warn "nsys not found"
command -v ncu &>/dev/null && info "ncu: $(ncu --version 2>&1 | head -1)" || warn "ncu not found"

echo ""
echo "=========================================="
echo " perflab doctor"
echo "=========================================="
perflab doctor --all || true

echo ""
echo "=========================================="
echo " Real-hardware validation (pytest + real profiling runs)"
echo "=========================================="
echo ""
python3 scripts/validate_gpu_hardware.py "$@"

echo ""
echo "=========================================="
echo " Done"
echo "=========================================="
echo ""
echo "This was a diagnostic pass, not a pass/fail gate -- review the output"
echo "above for anything marked MISSING / NOT RESOLVED / EMPTY."
echo ""
