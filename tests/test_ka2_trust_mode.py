"""KA2 game_trust damping: legacy compounding vs the declared ``trust_mode="scale"`` (review finding #20)."""
import copy
from types import SimpleNamespace

import pytest
import torch

from particlegan.ka2 import WARMUP_CALLS, KA2CriticAdam, KA2GradientPenalty


def _alpha_after(trust, mode, calls=1500):
    torch.manual_seed(0)
    critic = torch.nn.Linear(2, 1)
    optimizer = KA2CriticAdam(list(critic.parameters()), critic=critic, ema_critic=copy.deepcopy(critic),
                              **({} if mode is None else {"trust_mode": mode}))
    record = optimizer.record
    penalty = KA2GradientPenalty(record=record)
    penalty.anchor = optimizer.anchor
    penalty.continuous_controller = SimpleNamespace(variant="dv12", game_trust=trust, data_drive=1.0)
    record.calls, record.observed_steps, record.anchor_started = WARMUP_CALLS, 1, True
    record.sur_hist, record.sur_base = [1.0] * 30, 1.0
    x_real, x_fake = torch.randn(4, 2), torch.randn(4, 2)
    for _ in range(calls):
        record.last_sur = 2.5  # surprise ratio 2.5: controller target alpha .75
        penalty._k3p_penalty(critic, x_real, x_fake, 10_000, 1.0, False, None)
    return optimizer, record.alpha


def test_legacy_trust_compounds_across_calls():
    _, alpha = _alpha_after(.9, None)
    assert alpha < .2  # trust .9 removes ~87% of the .75 target, not 10%


@pytest.mark.parametrize("trust", [1.0, .9, .5])
def test_scale_mode_applies_trust_once(trust):
    _, alpha = _alpha_after(trust, "scale")
    assert alpha == pytest.approx(.75 * trust, abs=1e-6)


def test_trust_one_is_identical_across_modes():
    assert _alpha_after(1.0, None)[1] == _alpha_after(1.0, "scale")[1]


def test_trust_mode_checkpoint_contract():
    legacy, _ = _alpha_after(.9, None, calls=3)
    assert "trust_mode" not in legacy.record.state_dict() and "alpha_raw" not in legacy.record.state_dict()
    scaled, _ = _alpha_after(.9, "scale", calls=3)
    state = scaled.record.state_dict()
    fresh, _ = _alpha_after(.9, "scale", calls=0)
    fresh.record.load_state_dict(state)
    assert fresh.record.alpha_raw == scaled.record.alpha_raw and fresh.record.alpha == scaled.record.alpha
    with pytest.raises(ValueError, match="keys"):
        legacy.record.load_state_dict(state)
    with pytest.raises(ValueError):
        KA2CriticAdam([torch.nn.Parameter(torch.zeros(1))], critic=torch.nn.Linear(1, 1), trust_mode="bogus")
