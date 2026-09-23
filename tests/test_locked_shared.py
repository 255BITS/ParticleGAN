"""Pins for the locked_shared stamp. No gym, no training loop."""

import pytest
import torch
from torch.nn import functional as F

from particlegan import GANLoss, GradientPenalty, get_recipe
from particlegan.grad_regularizers import GradRegularizer
from particlegan.locked_shared import (
    LOCKED_SHARED,
    NAMED_DRIFTS,
    drift,
    locked_adv_defaults,
    make_b_cap,
    make_gan_loss,
)


PINS = {
    "loss_type": "logistic",
    "gan_mode": "rp",
    "reg_arm": "b_cap",
    "reg_coeff": 1.0,
    "reg_kappa": 1.0,
    "reg_norm": "l2",
    "lazy_k": 1,
    "reg_method": "autograd",
    "target_anneal": "none",
    "fm_weight": 0.0,
    "cover_weight": 1.5,
    "cover_posture": "demo",
    "n_particles": 12,
    "particle_l2": 0.02,
    "z_dim": 2,
    "pairing": "live",
    "critic": "host",
    "reg_impl": "grad_regularizer",
}


def test_stamp_fields_match_expected_pins():
    assert locked_adv_defaults() == PINS
    assert LOCKED_SHARED.to_dict() == PINS
    assert LOCKED_SHARED.critic == "host"
    assert LOCKED_SHARED.cover_posture == "demo"


def test_locked_adv_defaults_is_frozen():
    stamp = locked_adv_defaults()
    with pytest.raises(TypeError):
        stamp["fm_weight"] = 0.1
    with pytest.raises(TypeError):
        stamp["cover_weight"] = 1.0


def test_builder_returns_rp_logistic_and_b_cap_kappa_1():
    real = torch.tensor([2.0, -1.0])
    fake = torch.tensor([-0.5, 0.5])
    loss = make_gan_loss()
    assert type(loss) is GANLoss
    assert loss.loss_type == "logistic" and loss.mode == "rp"
    assert loss.label_smoothing == 0.0 and loss.label_flip_prob == 0.0
    torch.testing.assert_close(loss.d_loss(real, fake), F.softplus(fake - real).mean())
    torch.testing.assert_close(loss.g_loss(fake, real), F.softplus(real - fake).mean())

    reg = make_b_cap()
    assert type(reg) is GradientPenalty
    assert type(reg) is GradRegularizer
    assert (reg.arm, reg.coeff, reg.kappa, reg.norm) == ("b_cap", 1.0, 1.0, "l2")
    assert reg.lazy_k == 1 and reg.method == "autograd" and reg.target_anneal == "none"
    assert reg.center(0) == LOCKED_SHARED.reg_kappa == 1.0


def test_b_cap_center_tracks_kappa_unlike_a_thinned_cap():
    """A thinned cap stores kappa and penalizes a hardcoded center. This one does not."""
    faithful = GradientPenalty(arm="b_cap", coeff=1.0, kappa=0.2, norm="l2", lazy_k=1)
    assert faithful.center(0) == 0.2
    assert make_b_cap().center(0) == make_b_cap().kappa
    assert drift("thinned_kappa").reg_impl != LOCKED_SHARED.reg_impl


def test_music_cover_1_is_not_the_stamp():
    music = drift("music_cover_1")
    assert music.cover_weight == 1.0 and music.cover_posture == "music"
    assert music != LOCKED_SHARED
    assert music.to_dict() != dict(locked_adv_defaults())


def test_fm_on_is_not_the_stamp():
    fm = drift("fm_on")
    assert fm.fm_weight == 0.1
    assert fm.fm_weight != LOCKED_SHARED.fm_weight
    assert fm != LOCKED_SHARED
    assert fm.to_dict() != dict(locked_adv_defaults())


def test_named_drifts_are_not_locked_shared_and_builders_refuse_them():
    assert set(NAMED_DRIFTS) == {"music_cover_1", "hub128", "fm_on", "stranger", "thinned_kappa"}
    for name in NAMED_DRIFTS:
        stamped = drift(name)
        assert stamped != LOCKED_SHARED
        with pytest.raises(ValueError, match="LOCKED_SHARED"):
            make_gan_loss(stamped)
        with pytest.raises(ValueError, match="LOCKED_SHARED"):
            make_b_cap(stamped)


def test_music_mlp_alias_is_not_the_host_critic():
    alias = LOCKED_SHARED.to_dict()
    alias["critic"] = "mlp"
    assert alias != dict(locked_adv_defaults())
    assert LOCKED_SHARED.critic == "host"


def test_study_recipe_and_yue2_lazy_cap_are_not_the_stamp():
    recipe = get_recipe()
    assert recipe.num_particles == 20_000
    assert recipe.num_particles != LOCKED_SHARED.n_particles
    assert recipe.prior_reg == 0.05
    assert recipe.prior_reg != LOCKED_SHARED.particle_l2
    # YuE2 gym controller: b_cap every fourth step. Documented, not this stamp.
    assert LOCKED_SHARED.lazy_k == 1
    yue2 = LOCKED_SHARED.to_dict()
    yue2["lazy_k"] = 4
    assert yue2 != dict(locked_adv_defaults())
