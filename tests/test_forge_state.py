"""Checkpoint identity supports diagnostic sentinels without changing finite hashes."""
from io import BytesIO
import struct

import torch

from experiments.forge.state import state_digest
from experiments.forge.policy_adapters import finite_policy_state


def test_finite_checkpoint_hashes_remain_byte_compatible():
    value = {'a': 1.25, 'b': [True, None, 'x'], 't': torch.tensor([1., -2.])}
    assert state_digest(value) == '6b95af469474ba9890c211e13ac38a4bb337a714e0719034f976ee7844568724'


def test_nonfinite_float_identity_survives_checkpoint_roundtrip_and_keeps_ieee_bits():
    nan = struct.unpack('!d', struct.pack('!Q', 0x7ff8000000000001))[0]
    other = struct.unpack('!d', struct.pack('!Q', 0x7ff8000000000002))[0]
    value = {'diagnostics': [nan, float('inf'), -float('inf')], 'weights': torch.tensor([1., 2.])}
    stream = BytesIO()
    torch.save(value, stream)
    stream.seek(0)
    assert state_digest(value) == state_digest(torch.load(stream, weights_only=True))
    assert len({state_digest(x) for x in (nan, other, float('inf'), -float('inf'), 'nan', None)}) == 6


def test_hashability_does_not_make_unhealthy_state_finite():
    for value in ({'unexpected_control': float('nan')}, {'weights': torch.tensor([float('nan')])}):
        assert len(state_digest(value)) == 64
        assert not finite_policy_state(value)
