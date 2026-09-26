"""Independently replay one declared shared-c6 discriminator witness.

The reference is an archived episode, not a source of training metrics or labels.
Only its declared architecture is used; the host and recipe come from the runner.
"""
import argparse
from copy import deepcopy
import gzip
import hashlib
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
from . import shared_discriminator_search as runner, suite
from .compare_defaults import plan, write
from benchmarks.gan_v3 import legacy_dict

REPLAY_KEYS = ('recipe', 'candidate', 'original_spec', 'spec', 'architecture',
               'discriminator_variant', 'applied', 'result', 'verdict', 'ema_verdict')


def numerical(value):
    """Timing is the only expected difference in an exact CPU replay."""
    ignored = {'seconds', 'controller_seconds', 'confirmed_seconds',
               'stable_from_seconds', 'created_at'}
    if isinstance(value, dict):
        return {key: numerical(item) for key, item in value.items() if key not in ignored}
    if isinstance(value, list):
        return [numerical(item) for item in value]
    return value


def replay(reference, implementation, output):
    expected = json.loads(gzip.decompress(reference.read_bytes()))
    selected = expected['discriminator_variant']
    card = selected['overrides']['research_discriminator']
    job = next(job for job in plan() if job['spec']['name'] == expected['spec']['name'])
    assert job['spec']['runner'] == 'vector'
    assert expected['original_spec'] == job['spec']
    assert expected['recipe'] == json.loads(json.dumps(legacy_dict(runner.recipe())))
    module = importlib.import_module(implementation)
    assert module.variant(card) == selected
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    protocol = suite.snapshot(output)
    protocol.update(reference=str(reference.resolve()),
                    reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                    implementation=implementation, purpose='Exact validation replay; no selection points')
    write(output/'protocol.json', protocol)
    with patch.object(runner, 'constructor', module.constructor), \
         patch.object(runner, 'variant', module.variant):
        actual = runner.episode(job, deepcopy(card))
    suite.verify_source(protocol)
    actual['source_sha256'] = protocol['source_sha256']
    raw = (json.dumps(actual, sort_keys=True, allow_nan=False)+'\n').encode()
    (output/'episode.json.gz').write_bytes(gzip.compress(raw, mtime=0))
    actual = json.loads(raw)
    checks = {key: numerical(actual[key]) == numerical(expected[key]) for key in REPLAY_KEYS}
    write(output/'checks.json', dict(checks=checks, architecture=actual['architecture'],
          task=actual['spec']['name'], uncompressed_sha256=hashlib.sha256(raw).hexdigest()))
    print(json.dumps(dict(checks=checks, verdict=actual['verdict'])), flush=True)
    assert all(checks.values()), 'Replay differs; retained both evidence and failed checks'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--implementation', default='benchmarks.transfer_suite.shared_critic_research')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    replay(args.reference, args.implementation, args.output)
