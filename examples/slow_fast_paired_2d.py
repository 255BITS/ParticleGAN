#!/usr/bin/env python
"""CPU gate: stranger nearest pairs fail; a same-start retime returns.

Lunar Lander is not run. This process must exit 0 (GATE PASS) before Lunar
collect is reworked. The gym note is docs/gym-slow-fast.md.
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
