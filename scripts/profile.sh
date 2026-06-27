# fast checkout
# ncu --set full --target-processes all -f -o epic_profile/test ./build/epic_driver -b tpccfull -d epic -w 1 -a 0.0 -r true -c 32 -e 5 -s 100000 -f true -m false -n 10000000 -x gpu

#!/bin/bash
# ============================================================================
# EPIC Profiling Script — supports ncu (Nsight Compute) and nsys (Nsight Systems)
# ============================================================================
# Usage:
#   ./scripts/profile.sh ncu   [epic_args...]
#   ./scripts/profile.sh nsys  [epic_args...]
#   ./scripts/profile.sh stats <report.nsys-rep>
#   ./scripts/profile.sh export <report.nsys-rep>
#   ./scripts/profile.sh ui    <report.nsys-rep>
#
# Examples:
#   ./scripts/profile.sh ncu  -b ycsbf -e 200 -n 1000000 -a 0.5
#   ./scripts/profile.sh nsys -b ycsbf -e 200 -n 1000000 -a 0.5
#   ./scripts/profile.sh stats epic_profile/myrun.nsys-rep
#   ./scripts/profile.sh ui    epic_profile/myrun.nsys-rep
# ============================================================================

set -e

PROFILE_DIR="epic_profile"
mkdir -p "$PROFILE_DIR"

EPIC_BIN="${EPIC_BIN:-./build/epic_driver}"

# Default EPIC args — override via CLI or environment
DEFAULT_ARGS=(
    -b tpccfull
    -d epic
    -w 1
    -a 0.0
    -r true
    -c 32
    -e 5
    -s 100000
    -f true
    -m false
    -n 10000000
    -x gpu
)

show_help() {
    cat <<EOF
Usage: $0 <mode> [arguments...]

Modes:
  ncu             Run Nsight Compute (per-kernel deep-dive profiling)
  nsys            Run Nsight Systems (whole-program timeline profiling)
  stats           Print nsys stats summary from a .nsys-rep file
  export          Export .nsys-rep to SQLite for scripted analysis
  ui              Open .nsys-rep in the Nsight Systems GUI

EPIC args (after mode, space-separated key-value pairs):
  -b, --benchmark      Benchmark: ycsba|ycsbb|ycsbc|ycsbf|tpccn|tpccfull
  -d, --database       Database engine: epic|gacco
  -w, --warehouses     TPC-C warehouse count
  -a, --skew           YCSB Zipfian skew factor (0.0–0.99)
  -r, --fullread       Full record read: true|false
  -c, --cpu_threads    CPU executor threads
  -e, --epochs         Number of epochs
  -s, --txns           Transactions per epoch
  -f, --split_fields   Split fields: true|false
  -m, --commutative    Commutative ops: true|false
  -n, --records        Total YCSB records
  -x, --exec_device    Execution device: cpu|gpu

Examples:
  $0 ncu  -b ycsbf -e 200 -n 1000000 -a 0.5
  $0 nsys -b ycsbf -e 200 -n 1000000 -a 0.5
  $0 stats epic_profile/myrun.nsys-rep
  $0 export epic_profile/myrun.nsys-rep
  $0 ui    epic_profile/myrun.nsys-rep

Environment:
  EPIC_BIN            Path to epic_driver (default: ./build/epic_driver)
  NCU_EXTRA_FLAGS     Extra flags for ncu (e.g. "--kernel-name gpuExec")
  NSYS_EXTRA_FLAGS    Extra flags for nsys (e.g. "--trace=cuda,osrt")
EOF
}

# ---------------------------------------------------------------------------
# Parse EPIC args from remaining positional arguments
# Reads key-value pairs: -b ycsbf -e 200 ...
# Merges with DEFAULT_ARGS (CLI overrides defaults)
# Uses shift to consume pairs — avoids indirect expansion bugs.
# ---------------------------------------------------------------------------
parse_epic_args() {
    declare -gA EPIC_ARGS
    # Load defaults first
    local i=0
    while [[ $i -lt ${#DEFAULT_ARGS[@]} ]]; do
        EPIC_ARGS["${DEFAULT_ARGS[$i]}"]="${DEFAULT_ARGS[$((i+1))]}"
        ((i+=2))
    done
    # Override with CLI (consumes $@ via shift)
    while [[ $# -ge 2 ]]; do
        EPIC_ARGS["$1"]="$2"
        shift 2
    done
}

# Build epic_driver command line from EPIC_ARGS associative array
build_epic_cmdline() {
    local cmdline=()
    for k in "${!EPIC_ARGS[@]}"; do
        cmdline+=("$k" "${EPIC_ARGS[$k]}")
    done
    echo "${cmdline[@]}"
}

# ---------------------------------------------------------------------------
# ncu — Nsight Compute (per-kernel deep dive)
# ---------------------------------------------------------------------------
run_ncu() {
    parse_epic_args "$@"
    local epic_cmdline
    epic_cmdline=$(build_epic_cmdline)
    local output="$PROFILE_DIR/ncu_profile"

    echo "=== Nsight Compute (ncu) ==="
    echo "Output: ${output}.ncu-rep"
    echo "EPIC args: $epic_cmdline"
    echo ""

    ncu \
        --set full \
        --target-processes all \
        -f \
        -o "$output" \
        ${NCU_EXTRA_FLAGS:-} \
        $EPIC_BIN $epic_cmdline

    echo ""
    echo "Done. Open with: ncu-ui ${output}.ncu-rep &"
}

# ---------------------------------------------------------------------------
# nsys — Nsight Systems (timeline profiling)
# ---------------------------------------------------------------------------
run_nsys() {
    parse_epic_args "$@"
    local epic_cmdline
    epic_cmdline=$(build_epic_cmdline)
    local output="$PROFILE_DIR/nsys_profile"

    echo "=== Nsight Systems (nsys) ==="
    echo "Output: ${output}.nsys-rep"
    echo "EPIC args: $epic_cmdline"
    echo ""

    nsys profile \
        --trace=cuda \
        --force-overwrite=true \
        --output="$output" \
        ${NSYS_EXTRA_FLAGS:-} \
        $EPIC_BIN $epic_cmdline

    echo ""
    echo "Done."
    echo "  Quick stats:  nsys stats --report cuda_gpu_kern_sum ${output}.nsys-rep"
    echo "  Export to DB: nsys export --type sqlite --output ${output}.sqlite ${output}.nsys-rep"
    echo "  Open GUI:     nsys-ui ${output}.nsys-rep &"
}

# ---------------------------------------------------------------------------
# nsys stats — print summary from a .nsys-rep file
# ---------------------------------------------------------------------------
run_stats() {
    local rep="$1"
    if [[ -z "$rep" || ! -f "$rep" ]]; then
        echo "ERROR: Provide a valid .nsys-rep file."
        echo "Usage: $0 stats <report.nsys-rep>"
        exit 1
    fi

    echo "=== GPU Kernel Summary ==="
    nsys stats --report cuda_gpu_kern_sum "$rep"
    echo ""
    echo "=== GPU Mem Time Summary ==="
    nsys stats --report cuda_gpu_mem_time_sum "$rep"
    echo ""
    echo "=== GPU Mem Size Summary ==="
    nsys stats --report cuda_gpu_mem_size_sum "$rep"
    echo ""
    echo "=== CUDA API Summary ==="
    nsys stats --report cuda_api_sum "$rep"
}

# ---------------------------------------------------------------------------
# nsys export — export .nsys-rep to SQLite
# ---------------------------------------------------------------------------
run_export() {
    local rep="$1"
    if [[ -z "$rep" || ! -f "$rep" ]]; then
        echo "ERROR: Provide a valid .nsys-rep file."
        echo "Usage: $0 export <report.nsys-rep>"
        exit 1
    fi
    local sqlite="${rep%.nsys-rep}.sqlite"

    echo "Exporting $rep → $sqlite ..."
    nsys export --type sqlite --force-overwrite true --output "$sqlite" "$rep"
    echo "Done. Query with: sqlite3 $sqlite"
}

# ---------------------------------------------------------------------------
# nsys-ui — open GUI
# ---------------------------------------------------------------------------
run_ui() {
    local rep="$1"
    if [[ -z "$rep" || ! -f "$rep" ]]; then
        echo "ERROR: Provide a valid .nsys-rep file."
        echo "Usage: $0 ui <report.nsys-rep>"
        exit 1
    fi
    nsys-ui "$rep" &
}

# ===========================================================================
# Main dispatch
# ===========================================================================
case "${1:-}" in
    ncu)
        shift
        run_ncu "$@"
        ;;
    nsys)
        shift
        run_nsys "$@"
        ;;
    stats)
        shift
        run_stats "$1"
        ;;
    export)
        shift
        run_export "$1"
        ;;
    ui)
        shift
        run_ui "$1"
        ;;
    -h|--help|help|"")
        show_help
        ;;
    *)
        echo "ERROR: Unknown mode '$1'"
        show_help
        exit 1
        ;;
esac