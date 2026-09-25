"""Run initially-zero geometry pathways with the canonical shared-D runner."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch

from . import shared_discriminator_search as runner
from .shared_geometry_residual_research import ARCHITECTURES, constructor, variant


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
    args = parser.parse_args()
    run(json.loads(args.plan.read_text()), args.output)
