"""Declared trainer options reach their candidates through the public API (review findings #11, #20).

The recipe schema is frozen (tests/test_swappable_priors.py), so both candidates are trainer
options: ``GANTrainer(row_evidence_statistic=...)`` and ``optimizer_options={"ka2_trust_mode": ...}``.
Defaults keep the legacy laws.
"""
import pytest
import torch
from torch import nn

from particlegan import GANTrainer, ParticlePrior, get_recipe


def _trainer(recipe, **options):
    torch.manual_seed(0)
    G = nn.Sequential(nn.Linear(recipe.z_dim, 8), nn.Tanh(), nn.Linear(8, 2))
    D = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
    prior = ParticlePrior(recipe.num_particles, recipe.z_dim, generator=torch.Generator().manual_seed(1))
    return GANTrainer(recipe, G, D, prior=prior, seed=0, **options)


def _e22():
    return get_recipe("e22", num_particles=32, z_dim=2, batch_size=8)


def test_row_evidence_statistic_reaches_the_gate_and_its_checkpoint():
    legacy = _trainer(_e22())
    assert legacy.policy.row_evidence.statistic == "sphericity"
    candidate = _trainer(_e22(), row_evidence_statistic="hotelling")
    assert candidate.policy.row_evidence.statistic == "hotelling"
    with pytest.raises(ValueError, match="row-evidence"):
        legacy.load_state_dict(candidate.state_dict())


@pytest.mark.parametrize("bad, match", [("bogus", "sphericity or hotelling"), ("hotelling", "row_evidence_gate")])
def test_row_evidence_statistic_is_validated(bad, match):
    recipe = _e22() if bad == "bogus" else get_recipe("gan", num_particles=32, z_dim=2, batch_size=8)
    with pytest.raises(ValueError, match=match):
        _trainer(recipe, row_evidence_statistic=bad)


def test_ka2_trust_mode_reaches_the_critic_only():
    recipe = get_recipe("ka2", num_particles=32, z_dim=2, batch_size=8)
    assert recipe.optimizer_family == "formulation" and recipe.critic_formulation == "ka2"
    assert _trainer(recipe).opt_d.record.trust_mode == "compound"
    trainer = _trainer(recipe, optimizer_options={"ka2_trust_mode": "scale"})
    assert trainer.opt_d.record.trust_mode == "scale"
    assert trainer.state_dict()["optimizer_options"] == {"ka2_trust_mode": "scale"}
    with pytest.raises(ValueError, match="optimizer_options"):
        _trainer(recipe).load_state_dict(trainer.state_dict())
    with pytest.raises(ValueError, match="compound or scale"):
        recipe.make_critic_optimizer(nn.Linear(2, 1), ema_critic=nn.Linear(2, 1), ka2_trust_mode="bogus")
    with pytest.raises(ValueError, match="KA2 critic formulation"):
        get_recipe("bcap").make_critic_optimizer(nn.Linear(2, 1), ka2_trust_mode="scale")
