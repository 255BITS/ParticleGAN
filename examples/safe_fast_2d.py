#!/usr/bin/env python
"""CPU gate: paired-error RpGAN lands slowly; the same GAN plus a safe-fast cost lands sooner."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.safe_fast_landing import format_collapse_report, format_report, run_collapse_gate, run_gate


def main():
    result = run_gate()
    print(format_report(result), flush=True)
    collapse = run_collapse_gate()
    print(format_collapse_report(collapse), flush=True)
    raise SystemExit(0 if result["passed"] and collapse["passed"] else 1)


if __name__ == "__main__":
    main()
