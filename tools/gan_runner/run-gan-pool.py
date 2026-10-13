#!/usr/bin/env python3
"""Finite independent studies with shared budgets/leases. Default: plan only."""
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get('PARTICLEGAN_RUNNER_ROOT', Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(ROOT))
from experiments.forge.codex_runner import main

if __name__ == '__main__':
    raise SystemExit(main(['--root', str(ROOT), 'pool', *sys.argv[1:]]))
