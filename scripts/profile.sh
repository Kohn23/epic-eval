#!/bin/bash
# ============================================================================
# EPIC Profiling Script — ncu (Nsight Compute)
# ============================================================================
# Usage:
#   ./scripts/profile.sh show
#   ./scripts/profile.sh ncu [ncu_args] epic [epic_args]
#
# Examples:
#   ./scripts/profile.sh ncu epic
#   ./scripts/profile.sh ncu -k regex:.*gpuExec.* epic -b tpccn -w 16
#   ./scripts/profile.sh ncu --set analysis epic -b ycsbf -e 200
#   ./scripts/profile.sh show
# ============================================================================

set -e
PROFILE_DIR="epic_profile"
mkdir -p "$PROFILE_DIR"
EPIC_BIN="${EPIC_BIN:-./build/epic_driver}"

# ======== Defaults (no -k in ncu defaults) ========
EPIC_DEFAULTS=( -b tpccfull 
                -d epic 
                -w 16 
                -a 0.0 
                -r true 
                -c 32 
                -e 3
                -s 100000 
                -f true 
                -m false 
                -n 10000000 
                -x gpu )

NCU_DEFAULTS=(  --set full 
                --target-processes all 
                -f )

# ======== Help ========
show_help() {
    cat <<EOF
Usage: $0 <command>

Commands:
  show                     Print current default command line
  import                   Import all .ncu-rep files from epic_profile/ to docs/log/
  ncu [ncu_args] epic [epic_args]
                           Run Nsight Compute with the given args.
                           "epic" separates ncu args (left) from epic_driver args (right).

Examples:
  # All defaults
  $0 ncu epic

  # Override epic args
  $0 ncu epic -b ycsbf -e 200 -n 1000000

  # Set kernel filter + change benchmark
  $0 ncu -k regex:.*gpuExec.* epic -b tpccn

  # Show default command
  $0 show

  # Import all .ncu-rep files
  $0 import

Env:
  EPIC_BIN              epic_driver path (default: ./build/epic_driver)
  NCU_EXTRA_FLAGS       extra ncu flags appended after defaults+user args

Output naming: ncu__{bench}_w{wh}_s{txn}[__{kernel}]__{set}.ncu-rep
  kernel part is omitted when no -k is given.
  Regex syntax (regex: ^ \$ . *) is stripped from filename.
EOF
}

# ======== Split args at "epic" delimiter ========
split_at_epic() {
    ncu_user_args=()
    epic_user_args=()
    local found=false
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == "epic" && "$found" == false ]]; then
            found=true
            shift
        elif [[ "$found" == true ]]; then
            epic_user_args+=("$1")
            shift
        else
            ncu_user_args+=("$1")
            shift
        fi
    done
}

# ======== Merge epic args (defaults + user, user wins) ========
# Outputs key-value pairs on stdout, one per line: "key\nvalue"
merge_epic_args() {
    local user=("$@")
    declare -A map
    local i
    for ((i=0; i<${#EPIC_DEFAULTS[@]}; i+=2)); do
        map["${EPIC_DEFAULTS[$i]}"]="${EPIC_DEFAULTS[$((i+1))]}"
    done
    for ((i=0; i<${#user[@]}; i+=2)); do
        map["${user[$i]}"]="${user[$((i+1))]}"
    done
    for k in "${!map[@]}"; do
        printf '%s\n%s\n' "$k" "${map[$k]}"
    done
}

# ======== Merge ncu args (defaults + user, user wins for -k/--set) ========
# Outputs final ncu arg array as null-delimited stdout
merge_ncu_args() {
    local user=("$@")
    local user_has_k=false user_has_set=false
    local i
    for a in "${user[@]}"; do
        [[ "$a" == "-k" ]] && user_has_k=true
        [[ "$a" == "--set" ]] && user_has_set=true
    done

    local merged=()
    for ((i=0; i<${#NCU_DEFAULTS[@]}; i++)); do
        case "${NCU_DEFAULTS[$i]}" in
            -k)    $user_has_k   && { ((i++)); continue; } ;;
            --set) $user_has_set && { ((i++)); continue; } ;;
        esac
        merged+=("${NCU_DEFAULTS[$i]}")
    done
    merged+=("${user[@]}")
    # Append NCU_EXTRA_FLAGS
    if [[ -n "${NCU_EXTRA_FLAGS:-}" ]]; then
        merged+=(${NCU_EXTRA_FLAGS})
    fi
    printf '%s\0' "${merged[@]}"
}

# ======== Extract flag value from arg array ========
extract_flag() {
    local flag="$1"  # e.g. -k, --set, -b
    shift
    local arr=("$@")
    local i
    for ((i=0; i<${#arr[@]}; i++)); do
        if [[ "${arr[$i]}" == "$flag" && $((i+1)) -lt ${#arr[@]} ]]; then
            echo "${arr[$((i+1))]}"
            return
        fi
    done
}

# ======== Clean kernel name for filename ========
clean_kernel_label() {
    local s="$1"
    s="${s#regex:}"     # strip regex: prefix
    s="${s//\*/}"       # strip *
    s="${s//./}"        # strip .
    s="${s//^/}"        # strip ^
    s="${s//$/}"        # strip $
    s="${s//\\/}"       # strip backslash
    [[ -z "$s" ]] && s="all"
    echo "$s"
}

# ======== show ========
run_show() {
    echo "=== Default command ==="
    echo -n "ncu"
    for a in "${NCU_DEFAULTS[@]}"; do echo -n " $a"; done
    echo -n " $EPIC_BIN"
    for a in "${EPIC_DEFAULTS[@]}"; do echo -n " $a"; done
    echo ""
}

# ======== ncu ========
run_ncu() {
    split_at_epic "$@"

    # Merge epic args (defaults + user) into an array
    local epic_merged_lines epic_merged_arr
    mapfile -t epic_merged_lines < <(merge_epic_args "${epic_user_args[@]}")
    # Convert to flat associative for lookup
    declare -A epic_map
    local i
    for ((i=0; i<${#epic_merged_lines[@]}; i+=2)); do
        epic_map["${epic_merged_lines[$i]}"]="${epic_merged_lines[$((i+1))]}"
    done

    # Build epic command line (skip empty values for boolean flags like -g)
    local epic_cmdline=()
    for k in "${!epic_map[@]}"; do
        epic_cmdline+=("$k")
        [[ -n "${epic_map[$k]}" ]] && epic_cmdline+=("${epic_map[$k]}")
    done

    # Merge ncu args (defaults + user + extra), collect into array
    local ncu_merged_arr=()
    while IFS= read -r -d '' arg; do
        ncu_merged_arr+=("$arg")
    done < <(merge_ncu_args "${ncu_user_args[@]}")

    # Extract values for filename
    local bench="${epic_map[-b]:-unknown}"
    local wh="${epic_map[-w]:-0}"
    local txn="${epic_map[-s]:-0}"
    local g_flag=""
    for ((i=0; i<${#epic_merged_lines[@]}; i++)); do
        if [[ "${epic_merged_lines[$i]}" == "-g" ]]; then
            g_flag="${epic_merged_lines[$((i+1))]}"
        fi
    done

    local ncu_kernel ncu_set
    ncu_kernel=$(extract_flag "-k" "${ncu_merged_arr[@]}")
    ncu_set=$(extract_flag "--set" "${ncu_merged_arr[@]}")

    # Build filename: ncu__{bench}_w{wh}_s{txn}[_g][__{kernel}]_{set}
    local fname="ncu__${bench}_w${wh}_s${txn}_"
    [[ -n "$g_flag" ]] && fname+="${g_flag}_"
    if [[ -n "$ncu_kernel" ]]; then
        fname+="_$(clean_kernel_label "$ncu_kernel")"
    fi
    fname+="_${ncu_set:-full}"
    local output="$PROFILE_DIR/$fname"

    echo "=== Nsight Compute (ncu) ==="
    echo "Output: ${output}.ncu-rep"
    echo "Command: ncu ${ncu_merged_arr[*]} $EPIC_BIN ${epic_cmdline[*]}"
    echo ""

    ncu \
        "${ncu_merged_arr[@]}" \
        -o "$output" \
        $EPIC_BIN "${epic_cmdline[@]}"

    echo ""
    echo "Done. Open with: ncu-ui ${output}.ncu-rep &"
}

# ======== import ========
run_import() {
    local LOG_DIR="docs/log"
    local count=0
    local failed=0

    if [[ ! -d "$PROFILE_DIR" ]]; then
        echo "ERROR: Profile directory '$PROFILE_DIR' does not exist."
        exit 1
    fi

    shopt -s nullglob
    local files=("$PROFILE_DIR"/*.ncu-rep)
    shopt -u nullglob

    if [[ ${#files[@]} -eq 0 ]]; then
        echo "No .ncu-rep files found in '$PROFILE_DIR'."
        exit 0
    fi

    mkdir -p "$LOG_DIR"

    echo "=== Importing ${#files[@]} .ncu-rep file(s) ==="
    for f in "${files[@]}"; do
        local basename
        basename=$(basename "$f" .ncu-rep)
        local output="$LOG_DIR/$basename"
        echo -n "[$((++count))/${#files[@]}] $basename ... "
        if ncu --import "$f" > "$output" 2>&1; then
            echo "OK -> $output"
        else
            echo "FAILED"
            ((failed++))
        fi
    done

    echo ""
    echo "Done. Imported $((count - failed))/${count} file(s)."
    [[ $failed -gt 0 ]] && echo "WARNING: $failed import(s) failed."
}

# ======== Dispatch ========
case "${1:-}" in
    show)
        run_show
        ;;
    import)
        run_import
        ;;
    ncu)
        shift
        run_ncu "$@"
        ;;
    -h|--help|help|"")
        show_help
        ;;
    *)
        echo "ERROR: Unknown command '$1'"
        show_help
        exit 1
        ;;
esac