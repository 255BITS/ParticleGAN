"""New public arm support must preserve the independently pinned legacy catalog."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from benchmarks.legacy.grad_regularizers import GradientPenalty as ArchivedPenalty
from benchmarks.legacy.recipe import LegacyRecipe
from particlegan import GANTrainer, Recipe


@pytest.mark.parametrize("arm", ArchivedPenalty.ARMS)
def test_archived_arms_keep_their_factory_and_recorded_selector(arm):
    recipe = LegacyRecipe(reg_arm=arm, reg_coeff=2., reg_kappa=.7, reg_every=3)
    penalty = recipe.make_gradient_penalty()
    assert isinstance(penalty, ArchivedPenalty)
    assert (penalty.arm, penalty.coeff, penalty.kappa, penalty.lazy_k) == (arm, 2., .7, 3)
    assert recipe.to_dict()["reg_arm"] == arm
    assert LegacyRecipe(**recipe.to_dict()) == recipe
    if arm not in ("k3p", "a_r1r2", "b_cap"):
        with pytest.raises(ValueError, match="arm"):
            Recipe(reg_arm=arm)


def test_archived_method_validation_still_uses_its_pinned_kernel():
    recipe = LegacyRecipe(reg_arm="b_cap", reg_method="finite_difference")
    assert recipe.make_gradient_penalty().method == "finite_difference"
    with pytest.raises(ValueError, match="finite differences"):
        LegacyRecipe(reg_arm="a_r1r2", reg_method="finite_difference")
    with pytest.raises(ValueError, match="grad regularizer arm"):
        LegacyRecipe(reg_arm="unknown")


def _assert_checkpoint_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_checkpoint_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_checkpoint_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("arm", ["k3p", "a_r1r2", "b_cap", "f_none"])
def test_archived_checkpoint_recipe_is_preserved_and_rejects_changed_formulation(arm):
    recipe = LegacyRecipe(z_dim=2, num_particles=8, batch_size=4, total_steps=2,
                          reg_arm=arm)

    def build():
        return GANTrainer(recipe, nn.Linear(2, 2), nn.Linear(2, 1))

    trainer = build()
    trainer.step(torch.arange(8, dtype=torch.float32).reshape(4, 2) / 8)
    checkpoint = trainer.state_dict()
    assert checkpoint["recipe"] == recipe.to_dict()
    assert all(field in checkpoint["recipe"] for field in
               ("loss_type", "gan_mode", "reg_method"))
    restored = build()
    restored.load_state_dict(checkpoint)
    _assert_checkpoint_equal(checkpoint, restored.state_dict())

    for field, other in (("loss_type", "hinge"), ("gan_mode", "ns"),
                         ("reg_method", "finite_difference"),
                         ("reg_arm", "b_cap" if arm != "b_cap" else "a_r1r2")):
        bad = deepcopy(checkpoint)
        bad["recipe"][field] = other
        with pytest.raises(ValueError, match="recipe"):
            restored.load_state_dict(bad)
        _assert_checkpoint_equal(checkpoint, restored.state_dict())
