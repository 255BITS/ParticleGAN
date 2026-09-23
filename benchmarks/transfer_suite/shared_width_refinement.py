"""Adaptive head/skip parameterizations after the frozen 16-card width screen.

No optimizer changes; cards reuse the native pointwise shared_critic_v1 forward.
python -u -m benchmarks.transfer_suite.shared_width_refinement --plan PLAN --output OUTPUT
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
from . import shared_discriminator_search as canonical
from .shared_critic_research import _card

ARCHITECTURES = [
    _card('width_ref_softplus128_l3_head025', hidden=128, output_scale=.25),
    _card('width_ref_softplus128_l3_head05', hidden=128, output_scale=.5),
    _card('width_ref_softplus128_l3_head2', hidden=128, output_scale=2.),
    _card('width_ref_softplus1_128_l3', hidden=128, beta=1.),
    _card('width_ref_softplus10_128_l3', hidden=128, beta=10.),
    _card('width_ref_softplus128_l3_linear_skip', hidden=128, raw_linear_skip=True),
    _card('width_ref_softplus128_l3_residual', hidden=128, residual=True),
    _card('width_ref_silu160_l2_head025', hidden=160, layers=2, activation='silu', output_scale=.25),
    _card('width_ref_silu160_l2_head05', hidden=160, layers=2, activation='silu', output_scale=.5),
    _card('width_ref_silu160_l2_head2', hidden=160, layers=2, activation='silu', output_scale=2.),
    _card('width_ref_silu160_l2_linear_skip', hidden=160, layers=2, activation='silu', raw_linear_skip=True),
    _card('width_ref_silu160_l2_residual', hidden=160, layers=2, activation='silu', residual=True),
]


def run(declaration, output):
    with patch.object(canonical, 'ARCHITECTURES', ARCHITECTURES):
        return canonical.run(declaration, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(json.loads(args.plan.read_text()), args.output)
