#!/usr/bin/env python
"""CPU gate: marginal RpGAN misses the pad; paired-error RpGAN plus the recipe critic penalty recovers it.

Each arm trains on the shared toy runner; observations print as JSON lines, then the gate report::

    python -u examples/yue2_particle_2d.py 2>&1 | tee runs/toy-refactor/example_yue2_particle_2d.log
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.yue2_particle_toy import format_report, print_row, run_gate


def main():
    result = run_gate(log=print_row)
    print(format_report(result), flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
