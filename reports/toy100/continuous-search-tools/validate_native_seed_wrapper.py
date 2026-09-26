"""CPU configuration checks only: no model training or seed experiments."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import native_seed_wrapper as wrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime.resolve()))
    import torch
    from benchmarks.toy100.config import resolve_problem_config

    before_rng = torch.random.get_rng_state().clone()
    original_manifest = wrapper.source_manifest(args.candidate)
    config = json.loads((args.candidate / 'config.json').read_text())
    fixture = '''def configure():
    config = json.loads((a.candidate / 'config.json').read_text())
    cfg = resolve_problem_config(config, a.task, device='cuda:0')
    return cfg
'''
    cells = []
    for task in wrapper.TASKS:
        base = resolve_problem_config(config, task, device='cuda:0', validate_runtime=False)
        for seed in wrapper.SEEDS:
            # Exercise the exact injected configuration statements with the real resolver.
            scope = dict(a=SimpleNamespace(candidate=args.candidate, task=task),
                         record={}, json=json, resolve_problem_config=resolve_problem_config)
            exec(wrapper.instrument(fixture, seed, False), scope)
            actual = scope['configure']()
            expected = deepcopy(base)
            expected['seed'] = seed
            assert actual == expected and actual['steps'] == 7000
            assert scope['record']['seed'] == seed
            cells.append(dict(task=task, seed=seed, only_seed_changed=True, steps=7000))
    for seed in (True, 1233, 1238, None):
        try:
            wrapper.instrument(fixture, seed, False)
        except ValueError:
            pass
        else:
            raise AssertionError('Accepted seed outside the declared matrix')
    for source in ('', fixture + fixture):
        try:
            wrapper.instrument(source, 1234, False)
        except ValueError:
            pass
        else:
            raise AssertionError('Accepted missing or ambiguous driver setup')
    scope = dict(a=SimpleNamespace(candidate=args.candidate, task='grid100'), record={},
                 json=json, resolve_problem_config=lambda *a, **k: dict(steps=6000, seed=1234))
    exec(wrapper.instrument(fixture, 1234, False), scope)
    try:
        scope['configure']()
    except RuntimeError:
        pass
    else:
        raise AssertionError('Accepted a shortened budget')
    for observe in (False, True):
        source = (args.candidate / 'native100.py').read_text()
        compile(wrapper.instrument(source, 1234, observe), 'native-driver-validation', 'exec')
    with tempfile.TemporaryDirectory(prefix='native-source-pin-') as temporary:
        root = Path(temporary)
        for name in ('config.json', 'mechanism.py', 'latent.py', 'response.py', 'native100.py'):
            (root / name).write_text('{}' if name.endswith('.json') else '# original\n')
        first = wrapper.source_manifest(root)
        (root / 'mechanism.py').write_text('# changed\n')
        assert wrapper.source_manifest(root)['sha256'] != first['sha256']
    assert wrapper.source_manifest(args.candidate) == original_manifest
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    result = dict(status='PASS', scope=__doc__, matrix=cells,
        invalid_seed_rejected=True, ambiguous_driver_rejected=True,
        shortened_budget_rejected=True, source_change_detected=True,
        observer_and_plain_driver_compile=True, cpu_rng_unchanged=True,
        candidate_source_unchanged=True, candidate_source=original_manifest,
        wrapper_sha256=hashlib.sha256(Path(wrapper.__file__).read_bytes()).hexdigest(),
        argv=sys.argv, training_updates=0,
        remaining='A qualifying candidate still needs actual runs and installation auditing; this is not a gate pass.')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', configuration_cells=len(cells), training_updates=0)))


if __name__ == '__main__':
    main()
