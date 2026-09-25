"""Reuse the audited shared-D runner with a distinct local-density card registry.

python -u -m benchmarks.transfer_suite.shared_local_density_search --plan PLAN --output OUTPUT
Only the discriminator constructor/registry is replaced, in a serial context;
recipe, numerical host, scoring, source snapshots and optimizer receipts are the
existing shared-discriminator implementation without edits.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch

from . import shared_discriminator_search as runner
from .shared_local_density_research import ARCHITECTURES, constructor, variant


@contextmanager
def registry():
    with patch.object(runner, 'ARCHITECTURES', ARCHITECTURES), \
         patch.object(runner, 'constructor', constructor), patch.object(runner, 'variant', variant):
        yield


def episode(job, card):
    with registry():
        return runner.episode(job, card)


def run(declaration, output):
    with registry():
        return runner.run(declaration, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    from benchmarks.toy100.device import add_device_argument, apply_device_policy
    add_device_argument(parser)
    args = parser.parse_args()
    apply_device_policy(args.device, log=True)
    run(json.loads(args.plan.read_text()), args.output)
