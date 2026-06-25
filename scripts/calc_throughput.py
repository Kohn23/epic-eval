#!/usr/bin/env python3
"""Calculate throughput from epic_driver stdout.

Usage:
    ./build/epic_driver -b tpccfull -d epic -w 64 ... | python3 scripts/calc_throughput.py
"""

import sys
import re

# Patterns for each timed stage
PATTERNS = {
    "index_tsf": re.compile(r"Epoch (\d+) index_transfer time: (\d+) us"),
    "aux1":      re.compile(r"Epoch (\d+) gpu aux index time: (\d+) us"),
    "aux2":      re.compile(r"Epoch (\d+) gpu aux index part2 time: (\d+) us"),
    "index":     re.compile(r"Epoch (\d+) indexing time: (\d+) us"),
    "init_tsf":  re.compile(r"Epoch (\d+) init_transfer time: (\d+) us"),
    "sub":       re.compile(r"Epoch (\d+) submission time: (\d+) us"),
    "init":      re.compile(r"Epoch (\d+) initialization time: (\d+) us"),
    "exec_tsf":  re.compile(r"Epoch (\d+) exec_transfer time: (\d+) us"),
    "exec":      re.compile(r"Epoch (\d+) execution time: (\d+) us"),
}


def parse_output(lines):
    """Parse timing data from epic_driver log lines, return dict of epoch -> stage -> us."""
    epochs = {}
    for line in lines:
        for stage, pat in PATTERNS.items():
            m = pat.search(line)
            if m:
                eid = int(m.group(1))
                us = int(m.group(2))
                epochs.setdefault(eid, {})[stage] = us
    return epochs


def main():
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

    print(f"{'='*60}")
    print(f"{'Epic Throughput Report':^60}")
    print(f"{'='*60}")
    print(f"  Txns per epoch : {num_txns:,}")
    print(f"  Total epochs   : {len(epochs)}")
    print(f"  Total txns     : {num_txns * len(epochs):,}")
    print(f"{'='*60}")
    print(f"  {'Epoch':<8} {'E2E(us)':>10} {'Exec(us)':>10} {'E2E(M txn/s)':>15} {'Exec(M txn/s)':>15}")
    print(f"  {'-'*56}")

    total_e2e_us = 0
    total_exec_us = 0

    for eid in sorted(epochs):
        stages = epochs[eid]
        e2e_us = sum(stages.values())
        exec_us = stages.get("exec", 0)
        total_e2e_us += e2e_us
        total_exec_us += exec_us
        e2e_tput = num_txns / (e2e_us / 1_000_000) / 1_000_000
        exec_tput = num_txns / (exec_us / 1_000_000) / 1_000_000 if exec_us else 0
        print(f"  {eid:<8} {e2e_us:>10,} {exec_us:>10,} {e2e_tput:>15.2f} {exec_tput:>15.2f}")

    print(f"  {'-'*56}")
    avg_e2e = (num_txns * len(epochs)) / (total_e2e_us / 1_000_000) / 1_000_000
    avg_exec = (num_txns * len(epochs)) / (total_exec_us / 1_000_000) / 1_000_000 if total_exec_us else 0
    print(f"  {'Overall':<8} {total_e2e_us:>10,} {total_exec_us:>10,} {avg_e2e:>15.2f} {avg_exec:>15.2f}")

    # Steady-state (last epoch only)
    last_eid = max(epochs)
    last_e2e = sum(epochs[last_eid].values())
    last_exec = epochs[last_eid].get("exec", 0)
    steady_e2e = num_txns / (last_e2e / 1_000_000) / 1_000_000
    steady_exec = num_txns / (last_exec / 1_000_000) / 1_000_000 if last_exec else 0
    print(f"{'='*60}")
    print(f"  Steady-state (epoch {last_eid}):")
    print(f"    E2E        : {steady_e2e:.2f} M txn/s")
    print(f"    GPU Exec   : {steady_exec:.2f} M txn/s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
