#!/usr/bin/env python
"""Frozen native100 verdict for one evidence directory (run in its own process).

Imports the UNCHANGED `benchmarks.toy100.gate.score_run` (coverage gate) and
`benchmarks.toy100.accuracy_gate.score_run` (final-five accuracy checks + 100k holdout) from the frozen
repo, after checking their bytes against tasks/native100_fixture.json. A separate process is needed because
those modules import the frozen repo's own `particlegan` (via train.py), which must not mix with a
candidate package.

  native100_score.py RUN_DIR PROBLEM  -> prints one JSON line {coverage, accuracy, sources}
"""
from pathlib import Path
import hashlib
import json
import sys

sys.dont_write_bytecode = True
FIXTURE = json.loads((Path(__file__).resolve().parent / 'tasks' / 'native100_fixture.json').read_text())
ROOT = Path(FIXTURE['frozen_repo'])


def main():
    run_dir, problem = Path(sys.argv[1]), sys.argv[2]
    checked = {}
    for name, want in FIXTURE['host_source_sha256'].items():
        got = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f'frozen host source changed: {name} {got} != {want}')
        checked[name] = got
    sys.path.insert(0, str(ROOT))
    from benchmarks.toy100 import gate, accuracy_gate
    for module in (gate, accuracy_gate):
        assert Path(module.__file__).resolve().is_relative_to(ROOT), module.__file__
    coverage = gate.score_run(run_dir, problem)
    accuracy = accuracy_gate.score_run(run_dir, problem, coverage)
    print(json.dumps(dict(coverage=coverage, accuracy=accuracy, sources=checked), default=str))


if __name__ == '__main__':
    main()
