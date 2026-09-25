"""Reproduce the fixed-recipe linear-skip discriminator on behavioral data toys."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from . import suite, vector_tasks
from .linear_skip_refinement_research import ARCHITECTURES, constructor
from .protocol import test_verdict


def main():
    tasks = {s['name']: s for s in vector_tasks.TASKS if s['tier'] == 'ranking'}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New result directory; never overwrite an existing run.')
    parser.add_argument('--tasks', nargs='+', choices=tasks, default=list(tasks))
    from benchmarks.toy100.device import add_device_argument, apply_device_policy
    add_device_argument(parser)
    args = parser.parse_args()
    apply_device_policy(args.device, log=True)
    if len(set(args.tasks)) != len(args.tasks):
        parser.error('Each task may appear only once.')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'episodes').mkdir()
    card = deepcopy(next(c for c in ARCHITECTURES if c['name'] == 'linear_skip_d96_beta5'))
    policy = vector_tasks.fixed_policy('cosine')
    protocol = suite.snapshot(args.output)
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (args.output/'plan.json').write_text(json.dumps(dict(architecture=card, tasks=args.tasks, policy=policy), indent=2)+'\n')
    records = []
    for name in args.tasks:
        suite.verify_source(protocol)
        original = deepcopy(tasks[name])
        spec = dict(original, d_hidden=96, d_layers=2, research_discriminator=card)
        print(f'START {name}', flush=True)
        with patch.object(vector_tasks, 'SimpleMLPDiscriminator', constructor(card)):
            result = vector_tasks.run_episode(spec, policy, fixed=True)
        verdict = test_verdict(spec, result)
        value = dict(candidate=dict(name=card['name'], architecture=card), original_spec=original,
                     spec=spec, policy=policy, result=result, verdict=verdict,
                     source_sha256=protocol['source_sha256'])
        raw = (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()
        path = f'episodes/{card["name"]}__{name}.json.gz'
        (args.output/path).write_bytes(gzip.compress(raw, mtime=0))
        records.append(dict(task=name, artifact=path, uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
                            verdict=verdict, live=result['live'], ema=result['ema']))
        (args.output/'index.json').write_text(json.dumps(dict(records=records), indent=2)+'\n')
        convergence = verdict.get('convergence', {})
        print(f"DONE {name}: {verdict['status']}; final passing checks={convergence.get('passing_suffix')}; "
              f"confirmed step={convergence.get('confirmed_step')}", flush=True)
        if result.get('error'):
            raise RuntimeError(result['error'])
    suite.verify_source(protocol)


if __name__ == '__main__':
    main()
