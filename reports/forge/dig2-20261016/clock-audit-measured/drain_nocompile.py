"""Forge CLI with the per-completion board compile disabled (publication only, not science).

The full leaderboard compile takes 20+ minutes per completion on this tree and holds a shared GPU lease
idle. Attempts, grading and receipts are unchanged; boards can be compiled once afterwards.
"""
import sys

import experiments.forge.knowledge as knowledge

knowledge.compile_memory = lambda root, **kw: {"skipped": "dig2 clock-audit-measured: compile deferred"}
from experiments.forge.__main__ import main  # noqa: E402

sys.exit(main(sys.argv[1:]))
