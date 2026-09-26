"""Run a pinned native driver at one of the twelve declared qualification cells.

Only use after the candidate qualifies for native seed verification. This wrapper
does not select candidates, resume runs, shorten budgets, or grade the matrix.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


SEEDS = (1234, 1235, 1236, 1237)
TASKS = ('grid100', 'rotated100', 'staggered100')


def source_manifest(candidate):
    files = {str(p.relative_to(candidate)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(candidate.rglob('*'))
             if p.is_file() and p.suffix in ('.py', '.json')
             and '__pycache__' not in p.parts}
    required = {'config.json', 'mechanism.py', 'latent.py', 'response.py', 'native100.py'}
    if not required <= files.keys():
        raise ValueError('Candidate is missing required source files')
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    return dict(files=files, sha256=hashlib.sha256(encoded).hexdigest())


def instrument(source, seed, observer):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError('Seed is outside the fixed qualification matrix')
    load = "    config = json.loads((a.candidate / 'config.json').read_text())\n"
    resolve = "    cfg = resolve_problem_config(config, a.task, device='cuda:0')\n"
    if source.count(load) != 1 or source.count(resolve) != 1:
        raise ValueError('Unknown driver configuration setup; review before adapting')
    source = source.replace(load, load + f"    config['seed'] = {seed}\n    record['seed'] = {seed}\n")
    source = source.replace(resolve, resolve +
        f"    if cfg['steps'] != 7000 or cfg['seed'] != {seed}:\n"
        "        raise RuntimeError('Native qualification requires the declared seed and 7000 updates')\n")
    if observer:
        lines = source.splitlines(keepends=True)
        locations = [i for i, line in enumerate(lines) if line.startswith('import mechanism')]
        if len(locations) != 1:
            raise ValueError('Unknown driver learner imports; cannot install rate observer')
        lines.insert(locations[0] + 1, '_install_rate_observer()\n')
        source = ''.join(lines)
    return source


def argument(argv, key):
    if argv.count(key) != 1 or argv.index(key) + 1 >= len(argv):
        raise ValueError(f'Expected one explicit {key} argument')
    return argv[argv.index(key) + 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--source-sha256')
    parser.add_argument('--observer', action='store_true',
                        help='Use audited same-update observer; requires a pure multiplier query')
    parser.add_argument('--describe', type=Path, help='Print candidate source pin without running training')
    parser.add_argument('driver', type=Path, nargs='?')
    parser.add_argument('driver_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.describe is not None:
        if args.driver is not None or args.seed is not None:
            parser.error('--describe cannot be combined with a training invocation')
        print(json.dumps(source_manifest(args.describe.resolve()), indent=2))
        return
    if args.driver is None or args.seed is None or not args.source_sha256:
        parser.error('Training requires --seed, --source-sha256 and a driver')
    driver = args.driver.resolve()
    candidate = Path(argument(args.driver_args, '--candidate')).resolve()
    output = Path(argument(args.driver_args, '--output')).resolve()
    task = argument(args.driver_args, '--task')
    if driver != candidate / 'native100.py' or task not in TASKS:
        parser.error('Expected the candidate native100.py and one declared native layout')
    if output.exists() or output.is_relative_to(candidate):
        parser.error('Output must be fresh and outside the frozen candidate bundle')
    manifest = source_manifest(candidate)
    if manifest['sha256'] != args.source_sha256:
        parser.error('Candidate source changed from the supplied pin')
    original = driver.read_text()
    executed = instrument(original, args.seed, args.observer)
    observer = Path(__file__).resolve().parents[1] / 'continuous-round-3/rp1-audit/observation_adapter.py'
    observer_sha = hashlib.sha256(observer.read_bytes()).hexdigest() if args.observer else None
    if args.observer and observer_sha != '3b60a1d9e5ab679376ccfd9efe6559c80e5eef23e4813f776349c709577cdf81':
        parser.error('Audited rate observer changed')

    def install_observer():
        spec = importlib.util.spec_from_file_location('_native_qualification_observer', observer)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.install()

    sys.argv = [str(driver), *args.driver_args]
    sys.path.insert(0, str(candidate))
    provenance = dict(scope='Fixed native qualification seed; no formulation or budget changes',
        task=task, seed=args.seed, source=manifest, driver=str(driver),
        driver_sha256=hashlib.sha256(original.encode()).hexdigest(),
        executed_source_sha256=hashlib.sha256(executed.encode()).hexdigest(),
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        observation_adapter_sha256=observer_sha, argv=sys.argv)
    try:
        exec(compile(executed, str(driver), 'exec'),
             {'__name__': '__main__', '__file__': str(driver),
              '_install_rate_observer': install_observer})
    finally:
        provenance['source_unchanged_after'] = source_manifest(candidate) == manifest
        if output.is_dir():
            (output / 'native-seed-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        if not provenance['source_unchanged_after']:
            raise RuntimeError('Candidate source changed during qualification')


if __name__ == '__main__':
    main()
