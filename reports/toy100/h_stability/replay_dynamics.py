"""Replay an archived warm candidate implementation against verified pinned H.

Only filesystem roots are rebound. Training modules come from the candidate's
three copied sources and the checksum-verified H source tree in this checkout.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    args = parser.parse_args()
    source = args.archive.resolve()
    declaration = json.loads((source/'declaration.json').read_text())
    sys.path[:0] = [str(source), str(ROOT), str(ROOT/'reports/toy100')]
    def load(name, filename):
        # The old runners prepend a root inferred from __file__. Explicit
        # module loading prevents that path from selecting today's dynamics.py.
        spec = importlib.util.spec_from_file_location(name, source/filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        assert Path(module.__file__).resolve() == source/filename
        return module

    control = load('stability_runner', 'stability_runner.py')
    update = load('dynamics', 'dynamics.py')
    runner = load('archived_dynamics_runner', 'dynamics_runner.py')
    for name, digest in declaration['actual_candidate_sources'].items():
        assert runner.sha(source/name) == digest
    runner.REPO = runner.SOURCE = ROOT
    runner.EVIDENCE = ROOT/'reports/toy100/critic_signal_attempt'
    runner.CANDIDATE = runner.EVIDENCE/'batch-h/h_n05r06_mixup_c0p01_lr15'
    # PREP_ROOT intentionally remains the archived source directory.
    result = runner.run(declaration['variant'], args.output, declaration['steps'])
    with args.ledger.open('a') as ledger:
        ledger.write(json.dumps(dict(candidate=declaration['variant'],
            gate='warm_probe_200_archive_replay', status=result['status'],
            seconds=result['elapsed_seconds'], metrics=result['window'],
            artifact=str(args.output.resolve()), replay_of=str(source)))+'\n')


if __name__ == '__main__':
    main()
