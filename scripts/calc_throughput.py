#!/usr/bin/env python3
"""Calculate throughput from epic_driver stdout, with per-stage breakdown.

Usage:
    ./build/epic_driver -b tpccfull -d epic -w 64 ... 2>&1 | python3 scripts/calc_throughput.py
    python3 scripts/calc_throughput.py epic_output/output__b...txt
"""

import sys
import re

# Patterns for each timed stage (in epoch execution order)
PATTERNS_ORDERED = [
    ("index_tsf", "index xfer",    re.compile(r"Epoch (\d+) index_transfer time: (\d+) us")),
    ("aux1",      "gpu aux idx",   re.compile(r"Epoch (\d+) gpu aux index time: (\d+) us")),
    ("aux2",      "aux idx pt2",   re.compile(r"Epoch (\d+) gpu aux index part2 time: (\d+) us")),
    ("index",     "index (cuco)",  re.compile(r"Epoch (\d+) indexing time: (\d+) us")),
    ("init_tsf",  "init xfer",     re.compile(r"Epoch (\d+) init_transfer time: (\d+) us")),
    ("sub",       "submission",    re.compile(r"Epoch (\d+) submission time: (\d+) us")),
    ("init",      "init (MVCC)",   re.compile(r"Epoch (\d+) initialization time: (\d+) us")),
    ("exec_tsf",  "exec xfer",     re.compile(r"Epoch (\d+) exec_transfer time: (\d+) us")),
    ("exec",      "execution",     re.compile(r"Epoch (\d+) execution time: (\d+) us")),
]

STAGE_KEYS = [k for k, _, _ in PATTERNS_ORDERED]
STAGE_LABELS = [l for _, l, _ in PATTERNS_ORDERED]

# Pattern to extract the full command line from the log
CMD_PAT = re.compile(r"Command: (.+)$")


def parse_cmdline_from_log(lines):
    """Extract -b and -g values from the 'Command: ...' log line emitted by epic_driver."""
    bench_type = ""
    use_gpu = False
    for line in lines:
        m = CMD_PAT.search(line)
        if m:
            cmd = m.group(1)
            # Split respecting quotes (simple split is enough for these args)
            tokens = cmd.split()
            for i, tok in enumerate(tokens):
                if tok in ("-b", "--bench") and i + 1 < len(tokens):
                    bench_type = tokens[i + 1]
                elif tok == "-g":
                    use_gpu = True
                elif tok.startswith("-g"):  # handle fused -gXXX (unlikely but safe)
                    use_gpu = True
            break
    return bench_type, use_gpu


def parse_output(lines):
    """Parse timing data from epic_driver log lines, return dict of epoch -> stage -> us."""
    epochs = {}
    for line in lines:
        for key, _, pat in PATTERNS_ORDERED:
            m = pat.search(line)
            if m:
                eid = int(m.group(1))
                us = int(m.group(2))
                epochs.setdefault(eid, {})[key] = us
    return epochs


def bar(frac, width=30):
    """Return a simple ASCII bar for a fraction (0.0-1.0)."""
    filled = int(round(frac * width))
    return "\u2588" * filled + "\u2591" * (width - filled)


def fmt_us(us):
    """Format microseconds to a human-readable string."""
    if us >= 1_000_000:
        return f"{us / 1_000_000:.2f} s"
    elif us >= 1_000:
        return f"{us / 1_000:.2f} ms"
    else:
        return f"{us:,} \u00b5s"


def print_breakdown(stages, num_txns, title="Stage Breakdown"):
    """Print a detailed per-stage breakdown table."""
    e2e_us = sum(stages.values())

    print(f"\n-- {title} --")
    header = f"  {'Stage':<18} {'Time':>12} {'%':>7}  {'':30s}"
    print(header)
    print(f"  {'-'*18} {'-'*12} {'-'*7}  {'-'*30}")

    for key, label in zip(STAGE_KEYS, STAGE_LABELS):
        us = stages.get(key, 0)
        pct = us / e2e_us * 100 if e2e_us else 0
        print(f"  {label:<18} {fmt_us(us):>12} {pct:>6.1f}%  {bar(pct / 100)}")

    print(f"  {'-'*18} {'-'*12} {'-'*7}  {'-'*30}")
    print(f"  {'TOTAL':<18} {fmt_us(e2e_us):>12} {'100.0%':>7}")

    # Grouped summary
    xfer = (
        stages.get("index_tsf", 0) +
        stages.get("init_tsf", 0) +
        stages.get("exec_tsf", 0)
    )
    gpu_kernels = e2e_us - xfer
    print(f"\n  -- Grouped --")
    pct_gpu = gpu_kernels / e2e_us * 100 if e2e_us else 0
    pct_xfer = xfer / e2e_us * 100 if e2e_us else 0
    print(f"  {'  GPU kernels':<18} {fmt_us(gpu_kernels):>12} {pct_gpu:>6.1f}%")
    print(f"  {'  H2D transfers':<18} {fmt_us(xfer):>12} {pct_xfer:>6.1f}%")

    tput = num_txns / (e2e_us / 1_000_000) / 1_000_000
    print(f"\n  Throughput: {tput:.2f} M txn/s  ({num_txns:,} txns / {fmt_us(e2e_us)})")


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    # Support both stdin pipe and file argument
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    epochs = parse_output(lines)

    if not epochs:
        print("ERROR: No epoch timing data found in input.", file=sys.stderr)
        sys.exit(1)

    # Extract num_txns from log
    num_txns = 0
    for line in lines:
        m = re.search(r"num txns: (\d+)", line)
        if m:
            num_txns = int(m.group(1))
            break
    if not num_txns:
        print("ERROR: Could not determine num_txns.", file=sys.stderr)
        sys.exit(1)

    # Auto-detect -b and -g from the 'Command:' log line
    bench_type, use_group = parse_cmdline_from_log(lines)

    print(f"{'='*72}")
    print(f"{'EPIC Throughput Report':^72}")
    print(f"{'='*72}")
    if bench_type:
        print(f"  Benchmark      : {bench_type}")
    print(f"  Grouped        : {'Yes' if use_group else 'No'}")
    print(f"  Txns per epoch : {num_txns:,}")
    print(f"  Total epochs   : {len(epochs)}")
    print(f"  Total txns     : {num_txns * len(epochs):,}")

    # Per-epoch summary
    print(f"\n-- Per-Epoch Summary --")
    print(f"  {'Epoch':<8} {'Total':>12} {'Exec':>12} {'Throughput':>14}")
    print(f"  {'-'*48}")

    total_e2e_us = 0
    for eid in sorted(epochs):
        stages = epochs[eid]
        e2e_us = sum(stages.values())
        exec_us = stages.get("exec", 0)
        total_e2e_us += e2e_us
        tput = num_txns / (e2e_us / 1_000_000) / 1_000_000
        print(f"  {eid:<8} {fmt_us(e2e_us):>12} {fmt_us(exec_us):>12} {tput:>12.2f} M txn/s")

    avg_e2e = total_e2e_us / len(epochs)
    avg_tput = (num_txns * len(epochs)) / (total_e2e_us / 1_000_000) / 1_000_000
    print(f"  {'-'*48}")
    print(f"  {'Avg':<8} {fmt_us(avg_e2e):>12}  {'':>12} {avg_tput:>12.2f} M txn/s")

    # Detailed breakdown of last epoch (steady-state)
    last_eid = max(epochs)
    print_breakdown(epochs[last_eid], num_txns, f"Steady-State Epoch {last_eid} Breakdown")

    # Aggregate breakdown (all epochs summed)
    all_stages = {k: 0 for k in STAGE_KEYS}
    for eid in epochs:
        for k, v in epochs[eid].items():
            all_stages[k] += v
    total_txns = num_txns * len(epochs)
    print_breakdown(all_stages, total_txns, f"All {len(epochs)} Epochs Aggregate Breakdown")

    print(f"\n{'='*72}")


if __name__ == "__main__":
    main()