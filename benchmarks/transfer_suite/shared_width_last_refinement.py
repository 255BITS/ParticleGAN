"""Final six D-only width refinements under the unchanged shared_c6 recipe.

This is an explicitly authorized adaptive round after 28 negative width trials.
It uses the existing pointwise shared_critic_v1 implementation unchanged.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
from . import shared_discriminator_search as canonical
from .shared_critic_research import _card

ARCHITECTURES = [
    _card('width_last_softplus3_128_l3', hidden=128, beta=3.),
    _card('width_last_softplus4_128_l3', hidden=128, beta=4.),
    _card('width_last_softplus6_128_l3', hidden=128, beta=6.),
    _card('width_last_softplus8_128_l3', hidden=128, beta=8.),
    _card('width_last_silu160_l2_linear_skip_head05', hidden=160, layers=2,
          activation='silu', raw_linear_skip=True, output_scale=.5),
    _card('width_last_silu160_l2_linear_skip_head2', hidden=160, layers=2,
          activation='silu', raw_linear_skip=True, output_scale=2.),
]


def run(declaration, output):
    with patch.object(canonical, 'ARCHITECTURES', ARCHITECTURES):
        return canonical.run(declaration, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    from benchmarks.toy100.device import add_device_argument, apply_device_policy
    add_device_argument(parser)
    args = parser.parse_args()
    apply_device_policy(args.device, log=True)
    run(json.loads(args.plan.read_text()), args.output)
