"""Empirical challenge suite for LossSurprise noise bounds sensitivity and GANTrainer state persistence.

This adversarial suite stress-tests:
1. Shift detection sensitivity across noise bounds:
   - Cold-start stability envelope: Verify zero false positives during cold start for
     noise std regimes sigma in [1e-4, 1e-3, 0.01, 0.05, 0.1] (where sigma^2 <= initial var_calm=0.01).
   - Calibrated equilibrium stability: Once var_calm adapts to the regime (sigma in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]),
     sub-threshold shifts (Delta <= 1.5 * sigma) yield exactly 0 false positives.
   - Supra-threshold abrupt shift detection: For any noise regime, an abrupt jump of Delta = 8.0 * sigma
     guarantees streak=1 on step 1, and fire=True on step 2.
   - Variance floor behavior: After 800 constant loss steps, var_calm decays to MIN_VAR (5e-6) without underflow.
   - Negative shift rejection: Abrupt drops (Delta = -10.0 * sigma) never trigger false fires.
   - Robustness to non-finite inputs: NaN, +Inf, -Inf are safely discarded without state corruption.
2. GANTrainer state persistence & continuation parity:
   - Multi-step continuation parity: Checkpointed trainer produces bit-exact weights,
     moments, and LossSurprise state compared to uninterrupted run.
   - Mid-shift checkpointing: Checkpointing at streak=1 faithfully restores streak=1 and fires
     on the very next step in exact lockstep.
   - Cooldown state persistence: Disarmed cooldown state is preserved across serialization.
"""
from copy import deepcopy
import math
import pytest
import torch
import torch.nn as nn

from particlegan import GANTrainer, Recipe, get_recipe
from particlegan.continuous import LossSurprise
from particlegan.init import deterministic_orthogonal_
from particlegan.priors import unwrap_prior


# ==============================================================================
# Suite 1: LossSurprise Sensitivity Across Noise Bounds
# ==============================================================================

@pytest.mark.parametrize("sigma", [1e-4, 1e-3, 0.01, 0.05, 0.1])
def test_loss_surprise_cold_start_stability_within_operating_envelope(sigma: float):
    """Verify zero false positives during cold start for noise regimes within the initial variance envelope (sigma <= 0.1)."""
    torch.manual_seed(12345)
    det = LossSurprise()
    base_loss = 2.0

    # 300 steps of stationary loss
    losses = base_loss + sigma * torch.randn(300)
    for step, l in enumerate(losses.tolist()):
        det.observe(l)
        fired = det.decide(step)
        assert not fired, f"Fired during cold-start at step {step} for sigma={sigma}"

    assert det.fires == 0
    assert det.armed is True


@pytest.mark.parametrize("sigma", [0.01, 0.05, 0.1, 0.5, 1.0, 2.0])
def test_loss_surprise_calibrated_subthreshold_shift_zero_false_positives(sigma: float):
    """Stress-test false alarm rejection across noise regimes once var_calm is calibrated to the true variance."""
    torch.manual_seed(42)
    det = LossSurprise()
    base_loss = 2.0

    # Pre-calibrate var_calm to match true noise variance
    det.var_calm = max(det.MIN_VAR, sigma ** 2)
    det.calm = base_loss
    det.fast = base_loss

    # 100 steps of stationary training
    stationary = base_loss + sigma * torch.randn(100)
    for step, l in enumerate(stationary.tolist()):
        det.observe(l)
        fired = det.decide(step)
        assert not fired, f"Fired during stationary phase at step {step} for sigma={sigma}"

    # 100 steps with sub-threshold mean shift of 1.5 * sigma
    sub_shift = (base_loss + 1.5 * sigma) + sigma * torch.randn(100)
    for step, l in enumerate(sub_shift.tolist(), start=100):
        det.observe(l)
        fired = det.decide(step)
        assert not fired, f"False positive at step {step} for sigma={sigma} under sub-threshold shift"

    assert det.fires == 0


@pytest.mark.parametrize("sigma", [1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0])
def test_loss_surprise_suprathreshold_abrupt_shift_detection(sigma: float):
    """Stress-test detection of abrupt supra-threshold shifts across 8 noise regimes.

    For any noise regime, an abrupt shift of Delta = 8.0 * max(sigma, sqrt(MIN_VAR))
    must trigger streak=1 on step 1 (since 0.5 * 8.0 = 4.0 > 3.0), and fire=True on step 2.
    """
    torch.manual_seed(54321)
    det = LossSurprise()
    base_loss = 2.0

    # Set calibrated baseline
    effective_sigma = max(math.sqrt(det.MIN_VAR), sigma)
    det.calm = base_loss
    det.fast = base_loss
    det.var_calm = effective_sigma ** 2

    shift_magnitude = 8.0 * effective_sigma
    shifted_loss = base_loss + shift_magnitude

    # Step 1 of shift
    det.observe(shifted_loss)
    fired_1 = det.decide(1)
    assert not fired_1, f"Premature fire on step 1 for sigma={sigma}"
    assert det.streak == 1, f"Expected streak=1, got {det.streak} for sigma={sigma}"
    assert det.last_z > det.Z_THRESH

    # Step 2 of shift: MUST fire
    det.observe(shifted_loss)
    fired_2 = det.decide(2)
    assert fired_2 is True, f"Failed to fire on step 2 for sigma={sigma}"
    assert det.fires == 1
    assert det.armed is False
    assert det.streak == 0
    assert det.since_fire == 0


def test_loss_surprise_zero_variance_floor_protection():
    """Verify that when loss is completely deterministic (var -> 0), MIN_VAR prevents division by zero."""
    det = LossSurprise()
    # Decay var_calm towards MIN_VAR over 800 steps
    for step in range(800):
        det.observe(1.0)
        det.decide(step)

    # After 800 steps of 0.99 decay, 0.01 * 0.99^800 = 3.2e-6 <= MIN_VAR (5e-6)
    assert det.var_calm == det.MIN_VAR
    assert det.calm == 1.0
    assert det.fast == 1.0
    assert det.last_z == 0.0

    # A jump of 8 * sqrt(MIN_VAR) = 8 * 0.002236 ~ 0.0179 must be detected cleanly
    effective_sigma = math.sqrt(det.MIN_VAR)
    det.observe(1.0 + 8.0 * effective_sigma)
    assert not det.decide(800)
    assert det.streak == 1

    det.observe(1.0 + 8.0 * effective_sigma)
    assert det.decide(801) is True
    assert det.fires == 1


def test_loss_surprise_negative_loss_drop_rejection():
    """Verify that an abrupt negative drop in loss (e.g. loss drops sharply) never fires."""
    det = LossSurprise()
    for step in range(100):
        det.observe(5.0 + 0.1 * math.sin(step))
        det.decide(step)

    # Abrupt drop from 5.0 to 0.1
    for step in range(100, 150):
        det.observe(0.1)
        fired = det.decide(step)
        assert not fired, f"Fired on negative loss drop at step {step}"
        assert det.last_z <= 0.0
        assert det.streak == 0

    assert det.fires == 0


def test_loss_surprise_non_finite_inputs_discarded():
    """Verify that NaN, +Inf, -Inf do not corrupt internal state or trigger surprise."""
    det = LossSurprise()
    for step in range(50):
        det.observe(0.5)
        det.decide(step)

    calm_before = det.calm
    var_before = det.var_calm

    for non_finite in [float("nan"), float("inf"), float("-inf")]:
        det.observe(non_finite)
        fired = det.decide(99)
        assert not fired
        assert det.calm == calm_before
        assert det.var_calm == var_before
        assert det.pending_loss is None


# ==============================================================================
# Suite 2: GANTrainer Checkpoint State Persistence & Continuation Parity
# ==============================================================================

def _make_setup(seed=42):
    torch.manual_seed(seed)
    recipe = get_recipe("atlas", z_dim=2, num_particles=32, batch_size=16, lr=0.005)
    G = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2))
    D = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
    deterministic_orthogonal_(G, seed=0)
    deterministic_orthogonal_(D, seed=1)
    prior = recipe.make_prior(sigma=0.005)
    deterministic_orthogonal_(unwrap_prior(prior), seed=2)
    trainer = GANTrainer(recipe, G, D, prior=prior, seed=seed, serial_backward=True)
    return trainer


def test_gantrainer_checkpoint_continuation_parity():
    """Verify bit-exact continuation parity between uninterrupted and checkpoint-resumed GANTrainer."""
    t_uninterrupted = _make_setup(seed=100)
    t_resumed = _make_setup(seed=100)

    # 30 initial shared steps
    data_gen = torch.Generator().manual_seed(777)
    batches = [torch.randn(16, 2, generator=data_gen) for _ in range(60)]

    for b in batches[:30]:
        t_uninterrupted.step(b)

    # Save checkpoint at step 30
    ckpt = t_uninterrupted.state_dict()

    # Load checkpoint into second trainer
    t_resumed.load_state_dict(ckpt)

    # Step both for 30 more steps with identical batches
    for b in batches[30:]:
        t_uninterrupted.step(b)
        t_resumed.step(b)

    # Compare G parameters
    for p1, p2 in zip(t_uninterrupted.G.parameters(), t_resumed.G.parameters()):
        assert torch.equal(p1, p2), "Generator parameter mismatch after resumption"

    # Compare D parameters
    for p1, p2 in zip(t_uninterrupted.D.parameters(), t_resumed.D.parameters()):
        assert torch.equal(p1, p2), "Discriminator parameter mismatch after resumption"

    # Compare Prior table z
    p1_prior = unwrap_prior(t_uninterrupted.prior).z
    p2_prior = unwrap_prior(t_resumed.prior).z
    assert torch.equal(p1_prior, p2_prior), "Prior table mismatch after resumption"

    # Compare optimizer states
    for (k1, v1), (k2, v2) in zip(t_uninterrupted.opt_g.state_dict()["state"].items(),
                                  t_resumed.opt_g.state_dict()["state"].items()):
        for opt_key in ("exp_avg", "exp_avg_sq"):
            if opt_key in v1:
                assert torch.equal(v1[opt_key], v2[opt_key]), f"Optimizer G {opt_key} mismatch"

    # Compare LossSurprise state dict
    ls1 = t_uninterrupted.policy.loss_surprise.state_dict()
    ls2 = t_resumed.policy.loss_surprise.state_dict()
    assert ls1 == ls2, f"LossSurprise state mismatch: {ls1} vs {ls2}"


def test_gantrainer_mid_shift_checkpoint_fires_in_lockstep():
    """Verify that checkpointing at streak=1 preserves streak and fires on next step identically."""
    trainer = _make_setup(seed=200)

    # Train for 20 steps
    data_gen = torch.Generator().manual_seed(888)
    for _ in range(20):
        trainer.step(torch.randn(16, 2, generator=data_gen))

    # Force step 1 of shift on trainer's LossSurprise
    ls = trainer.policy.loss_surprise
    effective_sigma = math.sqrt(max(ls.MIN_VAR, ls.var_calm))
    shift_loss = ls.calm + 8.0 * effective_sigma
    ls.observe(shift_loss)
    fired_1 = ls.decide()
    assert not fired_1
    assert ls.streak == 1

    # Checkpoint while in mid-shift excursion (streak=1)
    ckpt = trainer.state_dict()

    # Create fresh trainer and restore
    trainer_restored = _make_setup(seed=999)
    trainer_restored.load_state_dict(ckpt)

    ls_restored = trainer_restored.policy.loss_surprise
    assert ls_restored.streak == 1
    assert ls_restored.is_surprised is True

    # Deliver step 2 of shift to both
    ls.observe(shift_loss)
    ls_restored.observe(shift_loss)

    fired_orig = ls.decide()
    fired_restored = ls_restored.decide()

    assert fired_orig is True, "Original detector failed to fire on step 2"
    assert fired_restored is True, "Restored detector failed to fire on step 2"
    assert ls.state_dict() == ls_restored.state_dict()


def test_gantrainer_cooldown_checkpoint_persistence():
    """Verify that a checkpoint saved during cooldown preserves disarmed state and re-arms at exactly step 16."""
    trainer = _make_setup(seed=300)

    # Train for 20 steps
    data_gen = torch.Generator().manual_seed(111)
    for _ in range(20):
        trainer.step(torch.randn(16, 2, generator=data_gen))

    ls = trainer.policy.loss_surprise
    effective_sigma = math.sqrt(max(ls.MIN_VAR, ls.var_calm))
    shift_loss = ls.calm + 8.0 * effective_sigma

    # Fire the detector
    ls.observe(shift_loss)
    ls.decide()
    ls.observe(shift_loss)
    fired = ls.decide()
    assert fired is True
    assert ls.armed is False
    assert ls.since_fire == 0

    # Advance 7 steps into cooldown
    calm_loss = ls.calm
    for _ in range(7):
        ls.observe(calm_loss)
        ls.decide()
        assert ls.armed is False
    assert ls.since_fire == 7

    # Checkpoint at since_fire = 7
    ckpt = trainer.state_dict()
    assert ckpt["loss_surprise"]["since_fire"] == 7
    assert ckpt["loss_surprise"]["armed"] is False

    # Restore into fresh trainer
    trainer_restored = _make_setup(seed=555)
    trainer_restored.load_state_dict(ckpt)
    ls_restored = trainer_restored.policy.loss_surprise
    assert ls_restored.since_fire == 7
    assert ls_restored.armed is False

    # Step both through remaining 8 cooldown steps (to since_fire = 15)
    for _ in range(8):
        ls.observe(calm_loss)
        ls.decide()
        ls_restored.observe(calm_loss)
        ls_restored.decide()
        assert ls.armed is False
        assert ls_restored.armed is False

    assert ls.since_fire == 15
    assert ls_restored.since_fire == 15

    # Step 16: since_fire reaches 16 -> both re-arm simultaneously!
    ls.observe(calm_loss)
    ls.decide()
    ls_restored.observe(calm_loss)
    ls_restored.decide()

    assert ls.since_fire == 16
    assert ls_restored.since_fire == 16
    assert ls.armed is True, "Original detector failed to rearm at step 16"
    assert ls_restored.armed is True, "Restored detector failed to rearm at step 16"
    assert ls.state_dict() == ls_restored.state_dict()
