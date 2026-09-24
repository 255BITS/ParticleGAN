"""Frozen raw-input smooth-critic catalog for unequal-width stability research.

All cards reuse shared_critic_v1. No data-derived features or training changes.
"""
from .shared_critic_research import _card

ARCHITECTURES = [
    _card('width_raw_silu64_l3', hidden=64, activation='silu'),
    _card('width_raw_silu96_l3', hidden=96, activation='silu'),
    _card('width_raw_silu160_l3', hidden=160, activation='silu'),
    _card('width_raw_silu192_l3', hidden=192, activation='silu'),
    _card('width_raw_silu128_l2', hidden=128, layers=2, activation='silu'),
    _card('width_raw_silu128_l4', hidden=128, layers=4, activation='silu'),
    _card('width_raw_silu96_l2', hidden=96, layers=2, activation='silu'),
    _card('width_raw_silu96_l4', hidden=96, layers=4, activation='silu'),
    _card('width_raw_silu160_l2', hidden=160, layers=2, activation='silu'),
    _card('width_raw_silu160_l4', hidden=160, layers=4, activation='silu'),
    _card('width_raw_silu192_l2', hidden=192, layers=2, activation='silu'),
    _card('width_raw_silu128_l3_linear_skip', hidden=128, activation='silu', raw_linear_skip=True),
    _card('width_residual_raw_silu128_l3', hidden=128, activation='silu', residual=True),
    _card('width_raw_softplus128_l3', hidden=128),
    _card('width_raw_softplus2_128_l3', hidden=128, beta=2.),
    _card('width_raw_softplus128_l4', hidden=128, layers=4),
]
