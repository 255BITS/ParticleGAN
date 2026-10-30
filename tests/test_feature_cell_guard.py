"""Unit tests verifying the small-table guard on feature-cell birth-death."""
import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe
from particlegan.birth_death import ParticleBirthDeath, ParticleRows


def _make_table(n_rows, z_dim=2):
    t = torch.arange(n_rows * z_dim, dtype=torch.float64).reshape(n_rows, z_dim)
    return nn.Parameter(torch.sin(t * 0.71) + t * 0.03)


def _make_rows(table):
    return ParticleRows(
        table=table,
        optimizer=torch.optim.Adam([table], lr=0.01, amsgrad=True),
        averaged_table=table.detach().clone(),
        generate=lambda z: z,
    )


# ---------------------------------------------------------------------------
# Test 1: Standalone ParticleBirthDeath Guard Bypass
# ---------------------------------------------------------------------------
def test_particle_birth_death_small_table_bypasses_turnover():
    """Verify that ParticleBirthDeath with cells=64 and N=32 skips turnover."""
    table = _make_table(n_rows=32)
    rows = _make_rows(table)
    # Set cells=64: table size 32 <= 64 triggers the guard
    bd = ParticleBirthDeath(rows, seed=42, cells=64)
    assert bd.cells == 64

    # Populate reservoir and set up strong synthetic discovery evidence
    real_sample = table.detach().clone() + 0.1
    bd.observe_real(real_sample)
    bd.S.fill_(1000.0)
    bd.W.fill_(1000.0)
    bd.n.fill_(2)

    table_before = table.detach().clone()
    event = bd.maybe_apply(sigma_out=0.0)

    # Assert turnover was completely bypassed
    assert event is None
    assert bd.moved_rows is None
    assert bd.counters["moves"] == 0
    assert torch.equal(table, table_before)


# ---------------------------------------------------------------------------
# Test 2: Standalone ParticleBirthDeath Normal Turnover on Large Tables
# ---------------------------------------------------------------------------
def test_particle_birth_death_large_table_executes_turnover():
    """Verify that ParticleBirthDeath with cells=64 and N=128 executes turnover."""
    table = _make_table(n_rows=128)
    rows = _make_rows(table)
    # Set cells=64: table size 128 > 64 does NOT trigger the guard
    bd = ParticleBirthDeath(rows, seed=42, cells=64)
    assert bd.cells == 64

    # Prime discovery evidence
    bd.observe_real(table.detach().clone() + 0.05)
    bd.anchor = table.detach().clone()
    bd.radius = torch.full((128,), 100.0, dtype=torch.float64)
    signs = torch.ones(128, dtype=torch.float64)
    signs[64:] = -1.0
    bd.S.copy_(signs * 1000.0)
    bd.W.copy_(signs * 1000.0)
    bd.n.fill_(2)

    table_before = table.detach().clone()
    event = bd.maybe_apply(sigma_out=0.0)

    # Assert turnover executed normally
    assert event is not None
    assert bd.counters["moves"] > 0
    assert bd.moved_rows is not None
    assert not torch.equal(table, table_before)


# ---------------------------------------------------------------------------
# Test 3: Policy Integration on mode_hold Scale (N=12 <= 128)
# ---------------------------------------------------------------------------
def test_policy_small_table_guard_mode_hold_scale():
    """Verify UpdatePolicy with atlas preset on N=12 skips birth-death."""
    recipe = get_recipe("atlas", num_particles=12, z_dim=4, batch_size=16)
    assert recipe.birth_death_cells == 128
    assert recipe.birth_death_backend == "auto"

    G = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 2)).double()
    D = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1)).double()
    prior = recipe.make_prior().double()

    trainer = GANTrainer(recipe, G, D, prior=prior, seed=123)
    assert trainer.birth_death is None
    assert trainer.policy.birth_death is None

    table_before = trainer.prior.z.detach().clone()
    for step in range(20):
        real_batch = torch.randn(16, 2, dtype=torch.float64)
        trainer.step(real_batch)

    # Verify birth-death remains None and table changed only via gradient optimization
    assert trainer.birth_death is None
    assert not torch.equal(trainer.prior.z, table_before)


# ---------------------------------------------------------------------------
# Test 4: Policy Integration on img_intensity2 Scale (N=32 <= 128)
# ---------------------------------------------------------------------------
def test_policy_small_table_guard_img_intensity2_scale():
    """Verify UpdatePolicy with atlas preset on N=32 skips birth-death."""
    recipe = get_recipe("atlas", num_particles=32, z_dim=8, batch_size=32)
    assert recipe.birth_death_cells == 128

    G = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 2)).double()
    D = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1)).double()
    prior = recipe.make_prior().double()

    trainer = GANTrainer(recipe, G, D, prior=prior, seed=456)
    assert trainer.birth_death is None
    assert trainer.policy.birth_death is None

    table_before = trainer.prior.z.detach().clone()
    for step in range(40):
        real_batch = torch.randn(32, 2, dtype=torch.float64)
        trainer.step(real_batch)

    assert trainer.birth_death is None
    assert not torch.equal(trainer.prior.z, table_before)


# ---------------------------------------------------------------------------
# Test 5: Policy Integration on Large Table Scale (N=1024 > 128)
# ---------------------------------------------------------------------------
def test_policy_large_table_allows_feature_cells():
    """Verify UpdatePolicy with atlas preset on N=1024 routes to feature-cells."""
    recipe = get_recipe("atlas", num_particles=1024, z_dim=2, batch_size=128)
    G = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2)).double()
    D = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1)).double()
    prior = recipe.make_prior().double()

    trainer = GANTrainer(recipe, G, D, prior=prior, seed=789)
    # Shape is observed on the first real batch, transitioning from pending to feature_cells
    trainer.step(torch.randn(128, 2, dtype=torch.float64))
    assert trainer.policy._feature_selection.state["actual_backend"] == "feature_cells"


# ---------------------------------------------------------------------------
# Test 6: Legacy kNN Conformance Protection (e22 with N=32)
# ---------------------------------------------------------------------------
def test_legacy_knn_does_not_activate_small_table_guard():
    """Verify that e22 with birth_death_backend='knn' does NOT bind cells."""
    recipe = get_recipe("e22", num_particles=32, z_dim=2, batch_size=16)
    assert recipe.birth_death_backend == "knn"

    G = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2)).double()
    D = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1)).double()
    prior = recipe.make_prior().double()

    trainer = GANTrainer(recipe, G, D, prior=prior, seed=101)
    # Under pure knn backend, cells is None, preserving legacy moves
    assert trainer.policy.birth_death.cells is None


# ---------------------------------------------------------------------------
# Test 7: Suite A - Constructor-Level Guard Parametrization N in [6..128]
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("N", [6, 8, 10, 12, 16, 32, 64, 128])
def test_policy_small_table_guard_constructor_clean(N: int):
    """Verify that UpdatePolicy with atlas preset constructs cleanly without ValueError
    for all N <= 128 (including N=6, 8, 10 which previously crashed on leave-one-out isolation).
    """
    recipe = get_recipe("atlas", num_particles=N, z_dim=4, batch_size=16, birth_death_cells=128)
    assert recipe.birth_death_isolation is True
    assert recipe.birth_death_cells == 128

    G = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 2)).double()
    D = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1)).double()
    prior = recipe.make_prior().double()

    trainer = GANTrainer(recipe, G, D, prior=prior, seed=42)
    assert trainer.birth_death is None
    assert trainer.policy.birth_death is None

    # Step once to confirm runtime execution safety
    trainer.step(torch.randn(16, 2, dtype=torch.float64))
    assert trainer.birth_death is None


# ---------------------------------------------------------------------------
# Test 8: Suite B - Lifecycle Multi-Step and Checkpoint Restore Stability
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("N", [8, 10, 12, 32, 128])
def test_policy_small_table_guard_multistep_and_checkpoint(N: int):
    """Verify multi-step training and checkpoint save/restore for small tables."""
    recipe = get_recipe("atlas", num_particles=N, z_dim=4, batch_size=16, birth_death_cells=128)
    G = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 2)).double()
    D = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1)).double()
    prior = recipe.make_prior().double()

    trainer = GANTrainer(recipe, G, D, prior=prior, seed=42)
    table_before = trainer.prior.z.detach().clone()

    for _ in range(10):
        trainer.step(torch.randn(16, 2, dtype=torch.float64))

    assert trainer.birth_death is None
    assert not torch.equal(trainer.prior.z, table_before)

    # Checkpoint save & restore roundtrip
    state = trainer.state_dict()
    assert "birth_death" not in state
    trainer.load_state_dict(state)
    assert trainer.birth_death is None


# ---------------------------------------------------------------------------
# Test 9: Boundary Guard Behavior N=128 (bypassed) vs N=129 (active)
# ---------------------------------------------------------------------------
def test_policy_table_boundary_cells_guard():
    """Verify exact boundary: N=128 deactivates birth_death, N=129 instantiates ParticleBirthDeath."""
    recipe_128 = get_recipe("atlas", num_particles=128, z_dim=4, batch_size=16, birth_death_cells=128)
    recipe_129 = get_recipe("atlas", num_particles=129, z_dim=4, batch_size=16, birth_death_cells=128)

    G = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 2)).double()
    D = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1)).double()

    trainer_128 = GANTrainer(recipe_128, G, D, prior=recipe_128.make_prior().double(), seed=42)
    trainer_129 = GANTrainer(recipe_129, G, D, prior=recipe_129.make_prior().double(), seed=42)

    assert trainer_128.birth_death is None
    assert trainer_129.birth_death is not None


# ---------------------------------------------------------------------------
# Test 10: Checkpoint Roundtrip Under Feature-Cells When Birth-Death is Deactivated vs Active
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("birth_death_cells,expect_birth_death", [
    (1024, False),  # Case 1: N=1024 <= cells=1024 -> birth_death deactivated (None in state)
    (128, True),    # Case 2: N=1024 > cells=128 -> birth_death active (present in state)
])
def test_feature_cells_checkpoint_roundtrip_when_birth_death_deactivated(birth_death_cells: int, expect_birth_death: bool):
    """Verify checkpoint save and restore roundtrip for feature_cells backend with and without birth_death.

    Regression test for KeyError: 'birth_death' in FeatureSelection.saved_served_source().
    When N=1024 and birth_death_cells=1024, the small-table guard sets birth_death to None,
    omitting 'birth_death' from trainer.state_dict(). When restoring into a fresh trainer,
    saved_served_source() must not assume 'birth_death' key exists in state.
    """
    recipe = get_recipe("atlas", num_particles=1024, z_dim=4, batch_size=16, birth_death_cells=birth_death_cells)
    G = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 2)).double()
    D = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1)).double()
    prior = recipe.make_prior().double()

    trainer = GANTrainer(recipe, G, D, prior=prior, seed=42)

    # Execute 1 step with real batch of shape (16, 2) to trigger backend selection transition
    real_batch = torch.randn(16, 2, dtype=torch.float64)
    trainer.step(real_batch)

    # Verify backend resolved to feature_cells
    assert trainer.policy._feature_selection.state["actual_backend"] == "feature_cells"
    if expect_birth_death:
        assert trainer.birth_death is not None
        assert trainer.policy.birth_death is not None
    else:
        assert trainer.birth_death is None
        assert trainer.policy.birth_death is None

    # Capture checkpoint
    state = trainer.state_dict()
    assert ("birth_death" in state) is expect_birth_death

    # Create fresh trainer with distinct seed to verify parameter overwriting
    G_fresh = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 2)).double()
    D_fresh = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1)).double()
    prior_fresh = recipe.make_prior().double()
    trainer_fresh = GANTrainer(recipe, G_fresh, D_fresh, prior=prior_fresh, seed=999)

    # Verify fresh trainer has distinct parameters before restoration
    assert not torch.equal(trainer.prior.z, trainer_fresh.prior.z)

    # Restore checkpoint into fresh trainer (must not raise KeyError: 'birth_death')
    trainer_fresh.load_state_dict(state)

    # Verify post-restore birth_death state and feature_cells backend
    if expect_birth_death:
        assert trainer_fresh.birth_death is not None
        assert trainer_fresh.policy.birth_death is not None
    else:
        assert trainer_fresh.birth_death is None
        assert trainer_fresh.policy.birth_death is None
    assert trainer_fresh.policy._feature_selection.state["actual_backend"] == "feature_cells"

    # Verify bitwise equality of parameters
    for p_orig, p_restored in zip(trainer.G.parameters(), trainer_fresh.G.parameters()):
        assert torch.equal(p_orig, p_restored)
    for p_orig, p_restored in zip(trainer.D.parameters(), trainer_fresh.D.parameters()):
        assert torch.equal(p_orig, p_restored)
    assert torch.equal(trainer.prior.z, trainer_fresh.prior.z)
    if hasattr(trainer, "ema_G") and trainer.ema_G is not None:
        for p_orig, p_restored in zip(trainer.ema_G.parameters(), trainer_fresh.ema_G.parameters()):
            assert torch.equal(p_orig, p_restored)
    if hasattr(trainer, "ema_prior") and trainer.ema_prior is not None:
        assert torch.equal(trainer.ema_prior.z, trainer_fresh.ema_prior.z)

    # Verify bitwise equality of sampling output
    sample_orig = trainer.sample(16)
    sample_restored = trainer_fresh.sample(16)
    assert torch.equal(sample_orig, sample_restored)

