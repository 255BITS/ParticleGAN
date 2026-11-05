"""Milestone 4 Iteration 1 Challenger 2 Adversarial Stress and Fail-Closed Test Suite.

Adversarially challenges:
1. GradientCoherence edge cases:
   - Zero-gradient inputs (initial, consecutive, recovery)
   - Non-finite gradient inputs (NaN, +Inf, -Inf)
   - Empty gradient lists
   - Orthogonal and opposing gradient sequences (EMA cancellation and coherence clamping)
   - Identical gradient sequences (coherence=1.0, gain=1.0)
   - Parameter layout mutation rejections (count and shape mismatches)
   - State serialization (state_dict / load_state_dict) roundtrips with 0 error and continuation parity
   - Deserialization corrupted state_dict rejection (schema and mode mismatches)
2. Fail-Closed recipe and checkpoint validation:
   - Negative, NaN, Inf reg_coeff rejection
   - Unsupported generator_gradient_response mode rejection
   - GANTrainer checkpoint schema, recipe, step, and execution mode mismatch rejection
3. Physical mechanism of DualNorm unanchored latent drift Z in 1D Gaussian data.
"""
from copy import deepcopy
import math
import pytest
import torch
import torch.nn as nn

from particlegan.gradient_response import GradientCoherence
from particlegan.recipes import Recipe, get_recipe
from particlegan.training import GANTrainer


# ==============================================================================
# 1. GradientCoherence Adversarial Unit Tests
# ==============================================================================

def test_gradient_coherence_zero_gradients_initial_and_consecutive():
    """Verify that all-zero gradient inputs clamp to min_gain without division by zero."""
    gc = GradientCoherence(min_gain=0.015, power=1.0)
    g0 = [torch.zeros(4, dtype=torch.float32), torch.zeros(2, 3, dtype=torch.float32)]
    
    # Initial zero gradient step
    gain = gc.observe(g0)
    assert gain == 0.015
    assert gc.packet["last"]["coherence"] == 0.0
    assert gc.packet["last"]["second_moment"] == 0.0
    assert gc.packet["last"]["mean_square"] == 0.0
    
    # Consecutive zero gradient steps
    for _ in range(10):
        gain = gc.observe(g0)
        assert gain == 0.015
        assert gc.packet["last"]["coherence"] == 0.0

    # Graceful recovery when non-zero gradients arrive
    g_active = [torch.ones(4, dtype=torch.float32), torch.ones(2, 3, dtype=torch.float32)]
    for _ in range(20):
        gain = gc.observe(g_active)
    assert gain > 0.8
    assert gc.packet["last"]["coherence"] > 0.8


@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
def test_gradient_coherence_non_finite_rejection(bad_val):
    """Verify that NaN and Inf gradients fail closed immediately with ValueError."""
    gc = GradientCoherence()
    good_g = torch.randn(4)
    bad_g = torch.tensor([1.0, bad_val, 0.0])
    with pytest.raises(ValueError, match="finite training gradients required"):
        gc.observe([good_g, bad_g])


def test_gradient_coherence_empty_list_rejection():
    """Verify that an empty gradient list fails closed with ValueError."""
    gc = GradientCoherence()
    with pytest.raises(ValueError, match="finite training gradients required"):
        gc.observe([])


def test_gradient_coherence_identical_gradient_sequence():
    """Verify that consistent directional gradients drive coherence and gain to 1.0."""
    gc = GradientCoherence()
    g = [torch.tensor([1.0, 2.0, -1.0]), torch.tensor([[0.5, -0.5]])]
    for _ in range(40):
        gain = gc.observe(g)
    assert gain == 1.0
    assert math.isclose(gc.packet["last"]["coherence"], 1.0, abs_tol=1e-6)


def test_gradient_coherence_opposing_gradients_limit_cycle_damping():
    """Verify that alternating opposing gradients cancel in EMA, driving coherence to 0."""
    gc = GradientCoherence(min_gain=0.015)
    g_pos = [torch.tensor([2.0, 0.0])]
    g_neg = [torch.tensor([-2.0, 0.0])]
    for i in range(60):
        g = g_pos if (i % 2 == 0) else g_neg
        gain = gc.observe(g)
    assert gain == 0.015
    assert gc.packet["last"]["coherence"] < 0.01


def test_gradient_coherence_orthogonal_gradients():
    """Verify that orthogonal alternating gradients settle near expected theoretical coherence."""
    gc = GradientCoherence()
    g_x = [torch.tensor([1.0, 0.0])]
    g_y = [torch.tensor([0.0, 1.0])]
    for i in range(80):
        g = g_x if (i % 2 == 0) else g_y
        gain = gc.observe(g)
    # Theoretical coherence for 2 alternating orthogonal unit vectors under EMA beta=7/8 is ~0.50
    assert 0.48 < gain < 0.53
    assert 0.48 < gc.packet["last"]["coherence"] < 0.53


def test_gradient_coherence_layout_mismatch_rejection():
    """Verify that changes in parameter count or parameter shapes raise ValueError."""
    gc = GradientCoherence()
    gc.observe([torch.zeros(4)])
    
    # Parameter count change
    with pytest.raises(ValueError, match="gradient parameter layout changed"):
        gc.observe([torch.zeros(4), torch.zeros(2)])
        
    # Parameter shape change
    with pytest.raises(ValueError, match="gradient parameter layout changed"):
        gc.observe([torch.zeros(5)])


def test_gradient_coherence_state_dict_roundtrip_bit_exact():
    """Verify bit-exact state_dict / load_state_dict roundtrip and continuation parity."""
    # Test on unobserved instance
    gc_fresh = GradientCoherence(min_gain=0.02, power=1.5)
    sd_fresh = gc_fresh.state_dict()
    gc_restored_fresh = GradientCoherence()
    p_dummy = [nn.Parameter(torch.zeros(3))]
    gc_restored_fresh.load_state_dict(sd_fresh, p_dummy)
    assert gc_restored_fresh.packet is None
    assert gc_restored_fresh.min_gain == 0.02
    assert gc_restored_fresh.power == 1.5

    # Test on active instance after observation history
    gc = GradientCoherence(min_gain=0.015, power=1.0)
    p = [nn.Parameter(torch.randn(3, 4)), nn.Parameter(torch.randn(2))]
    for step in range(15):
        g = [torch.randn(3, 4), torch.randn(2)]
        gc.observe(g)

    sd = gc.state_dict()
    gc_loaded = GradientCoherence()
    gc_loaded.load_state_dict(sd, p)

    # 1. Exact equality of all scalar configuration and state
    assert gc_loaded.beta == gc.beta
    assert gc_loaded.min_gain == gc.min_gain
    assert gc_loaded.power == gc.power
    assert gc_loaded.response == gc.response
    assert gc_loaded.packet["count"] == gc.packet["count"]
    assert gc_loaded.packet["weight"] == gc.packet["weight"]
    assert gc_loaded.packet["weight2"] == gc.packet["weight2"]
    assert gc_loaded.packet["second"] == gc.packet["second"]
    assert gc_loaded.packet["last"] == gc.packet["last"]
    for m1, m2 in zip(gc.packet["means"], gc_loaded.packet["means"]):
        assert torch.equal(m1, m2)

    # 2. Exact continuation parity
    next_g = [torch.randn(3, 4), torch.randn(2)]
    gain1 = gc.observe(next_g)
    gain2 = gc_loaded.observe(next_g)
    assert gain1 == gain2
    assert gc.packet["second"] == gc_loaded.packet["second"]
    for m1, m2 in zip(gc.packet["means"], gc_loaded.packet["means"]):
        assert torch.equal(m1, m2)


def test_gradient_coherence_corrupted_checkpoint_rejection():
    """Verify that corrupt schema or mode in state_dict fails closed."""
    gc = GradientCoherence()
    p = [nn.Parameter(torch.randn(2))]
    gc.observe([torch.randn(2)])
    sd = gc.state_dict()

    # Corrupted schema
    bad_schema = deepcopy(sd)
    bad_schema["schema"] = 2
    with pytest.raises(ValueError, match="gradient response checkpoint identity mismatch"):
        GradientCoherence().load_state_dict(bad_schema, p)

    # Corrupted mode
    bad_mode = deepcopy(sd)
    bad_mode["mode"] = "unsupported_coherence"
    with pytest.raises(ValueError, match="gradient response checkpoint identity mismatch"):
        GradientCoherence().load_state_dict(bad_mode, p)


# ==============================================================================
# 2. Fail-Closed Recipe & GANTrainer Validation Tests
# ==============================================================================

@pytest.mark.parametrize("bad_reg_coeff", [-1.0, -0.001, float("nan"), float("inf")])
def test_recipe_rejects_invalid_reg_coeff(bad_reg_coeff):
    """Verify that negative or non-finite reg_coeff fails closed at recipe creation."""
    with pytest.raises(ValueError, match="reg_coeff must be finite and nonnegative"):
        Recipe(reg_coeff=bad_reg_coeff)


@pytest.mark.parametrize("bad_mode", ["unsupported", "coherent_v3", 123, ""])
def test_recipe_rejects_unsupported_generator_gradient_response(bad_mode):
    """Verify that unknown generator_gradient_response mode fails closed."""
    with pytest.raises(ValueError, match="unknown generator gradient response mode"):
        Recipe(generator_gradient_response=bad_mode)


def test_gantrainer_checkpoint_fail_closed_rejections():
    """Verify that corrupted checkpoint states fail closed in GANTrainer.load_state_dict."""
    trainer = GANTrainer(Recipe(), nn.Linear(2, 2), nn.Linear(2, 1), seed=0)
    sd = trainer.state_dict()

    # 1. Older formulation schema rejection
    bad_schema = deepcopy(sd)
    bad_schema["schema"] = 3
    with pytest.raises(ValueError, match="schema-3 GANTrainer checkpoints come from an older formulation"):
        trainer.load_state_dict(bad_schema)

    # 2. Missing required models key
    missing_key = deepcopy(sd)
    del missing_key["models"]
    with pytest.raises(ValueError, match="invalid GANTrainer checkpoint schema"):
        trainer.load_state_dict(missing_key)

    # 3. Negative completed_steps
    bad_steps = deepcopy(sd)
    bad_steps["completed_steps"] = -1
    with pytest.raises(ValueError, match="invalid checkpoint step count"):
        trainer.load_state_dict(bad_steps)

    # 4. Recipe mismatch
    mismatched_recipe = deepcopy(sd)
    mismatched_recipe["recipe"]["lr"] = 999.0
    with pytest.raises(ValueError, match="checkpoint recipe does not match trainer"):
        trainer.load_state_dict(mismatched_recipe)

    # 5. Execution mode (serial_backward) mismatch
    mismatched_mode = deepcopy(sd)
    mismatched_mode["serial_backward"] = not sd.get("serial_backward", True)
    with pytest.raises(ValueError, match="checkpoint serial_backward execution mode does not match trainer"):
        trainer.load_state_dict(mismatched_mode)


# ==============================================================================
# 3. Physical Mechanism of DualNorm Unanchored Latent Drift Z
# ==============================================================================

def test_dualnorm_prior_rownorm_unanchored_drift():
    """Empirically verify the mathematical mechanism of unanchored latent drift in DualNorm.
    
    In DualNorm:
    - Prior latent particles Z (role='prior') are updated via rownorm:
      Delta Z_i = -rate * (g_i / ||g_i||_2).
    - Step length for each active row is ||Delta Z_i|| = rate = lr * prior_lr_mult.
    - Generator gradient coherence selectively damps generator parameters (role='generator')
      but leaves role='prior' unscaled, allowing Z to drift unconstrained.
    """
    torch.manual_seed(0)
    base = get_recipe("bcap").replace(num_particles=64, z_dim=2, total_steps=50)
    recipe_damped = base.replace(generator_gradient_response="coherent_G_damped_v2")

    class SimpleG(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
        def forward(self, z):
            return self.net(z)

    class SimpleD(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))
        def forward(self, x):
            return self.net(x)

    g = SimpleG()
    d = SimpleD()
    prior = recipe_damped.make_prior()
    z0 = prior.z.detach().clone()
    trainer = GANTrainer(recipe_damped, g, d, prior=prior, seed=0, max_steps=50)

    # Verify that the generator gradient response is instantiated
    assert trainer._gradient_response is not None

    # Step on 1D Gaussian data
    for _ in range(30):
        real = torch.randn(64, 1) * 0.5 + 2.0
        trainer.step(real)

    # Prior drift must be non-zero and substantial
    z_drift = (trainer.prior.z - z0).norm().item()
    assert z_drift > 1.0, f"Expected latent drift > 1.0, got {z_drift}"

    # Verify that prior optimizer group was NOT scaled by coherence gain
    prior_group = [grp for grp in trainer.opt_g.param_groups if grp.get("role") == "prior"][0]
    expected_prior_lr = base.lr * base.prior_lr_mult
    assert math.isclose(prior_group["lr"], expected_prior_lr, rel_tol=1e-5), (
        f"Prior group lr should remain unscaled at {expected_prior_lr}, got {prior_group['lr']}"
    )
