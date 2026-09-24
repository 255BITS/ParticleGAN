"""Reproduce the bounded critic-confidence dynamics comparison on CPU."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, nullcontext
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.toy100.continuous_probe import run_probe
from benchmarks.toy100.warm_equilibrium_probe import run_warm_variants, constant_rate_context
from benchmarks.transfer_suite import suite
from reports.toy100.confidence_dynamics_scratch import confidence_dynamics, warm_confidence


@contextmanager
def warm_variant(state, prefix, threshold=1., observe_only=False):
    with constant_rate_context(state, lr=.00425) as rate, warm_confidence(
            state, threshold=threshold, observe_only=observe_only) as receipt:
        receipt['fixed_rate'] = rate
        yield receipt


def cold_worker(row, config, output):
    options = {key: value for key, value in row.items() if key != 'tag'}
    with confidence_dynamics(**options) as receipt:
        evidence = run_probe(config, mode='constant', steps=1200, diagnostic_every=1)
    path = Path(output) / (row['tag'] + '.json.gz')
    path.write_bytes(gzip.compress(json.dumps(dict(evidence=evidence, receipt=receipt),
                                             allow_nan=False).encode(), mtime=0))
    result = dict(row, stationary=evidence['stationary'], final=evidence['final'],
                  seconds=evidence['seconds'])
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('warm', 'cold'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / 'configs/toy100/constraints_simple_regularization.json').read_text())
    rows = [dict(tag='threshold_025', threshold=.25), dict(tag='threshold_100', threshold=1.),
            dict(tag='ordinary_observed', threshold=1., observe_only=True)]
    archive = suite.snapshot(args.output)
    sources = {}
    for name in ('reports/toy100/confidence_dynamics_scratch.py',
                 'reports/toy100/confidence_probe.py', 'benchmarks/toy100/models.py'):
        target = args.output / Path(name).name
        shutil.copyfile(ROOT / name, target)
        sources[name] = dict(file=target.name, sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (args.output / 'declaration.json').write_text(json.dumps(dict(rows=rows, config=config,
        mode=args.mode, shared_gate_eligible=False, source_archive=archive,
        supplemental_sources=sources), indent=2) + '\n')
    if args.mode == 'cold':
        with ProcessPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(cold_worker, row, config, str(args.output)) for row in rows]
            result = [future.result() for future in futures]
    else:
        variants = {'identity': lambda state, prefix: nullcontext()}
        for row in rows:
            options = {key: value for key, value in row.items() if key != 'tag'}
            variants[row['tag']] = lambda state, prefix, opts=options: warm_variant(state, prefix, **opts)
        result = run_warm_variants(config, variants, output_dir=args.output / 'forks')
    (args.output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
