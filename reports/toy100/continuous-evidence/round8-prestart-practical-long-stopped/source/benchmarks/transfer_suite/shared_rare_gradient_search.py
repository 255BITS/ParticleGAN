"""Run a frozen screen of critic-gradient architecture cards under shared-c6.

python -u -m benchmarks.transfer_suite.shared_rare_gradient_search --plan PLAN --output OUTPUT
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from . import shared_discriminator_search as base
from .shared_rare_gradient_research import ARCHITECTURES, constructor, variant


def run(declaration, output):
    with patch.object(base, 'ARCHITECTURES', ARCHITECTURES), \
         patch.object(base, 'constructor', constructor), \
         patch.object(base, 'variant', variant):
        base.run(declaration, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.plan.read_text()), args.output)
