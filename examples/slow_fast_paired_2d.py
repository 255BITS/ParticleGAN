#!/usr/bin/env python
"""CPU gate: paired-error RpGAN finetunes a slow pad landing into a fast one.

Lunar Lander is not run. This process must exit 0 (GATE PASS) before a Lunar
speed claim. The gym commands live in docs/gym-slow-fast.md.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.slow_fast_paired import board_markdown, format_report, run_gate

BOARD = ROOT / "reports" / "slow_fast_paired" / "README.md"


def main():
    result = run_gate()
    print(format_report(result), flush=True)
    BOARD.parent.mkdir(parents=True, exist_ok=True)
    BOARD.write_text(board_markdown(result))
    print(f"[slow-fast] wrote {BOARD}", flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
