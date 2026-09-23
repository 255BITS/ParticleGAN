"""Run the fixed width-research catalog through the canonical shared-cap6 host.

python -u -m benchmarks.transfer_suite.shared_width_search --plan PLAN --output OUTPUT
Only the discriminator catalog differs from shared_discriminator_search.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
from . import shared_discriminator_search as canonical
from .shared_width_catalog import ARCHITECTURES


def run(declaration, output):
    with patch.object(canonical, 'ARCHITECTURES', ARCHITECTURES):
        return canonical.run(declaration, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(json.loads(args.plan.read_text()), args.output)
