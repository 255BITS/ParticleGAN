"""Unit tests verifying the small-table guard on feature-cell birth-death."""
import pytest
import torch
from torch import nn

from particlegan import GANTrainer, Recipe, get_recipe
from particlegan.birth_death import ParticleBirthDeath, ParticleRows
from particlegan.policy import UpdatePolicy


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
    assert trainer.policy.birth_death is not None
    assert trainer.policy.birth_death.cells == 128

    table_before = trainer.prior.z.detach().clone()
    for step in range(20):
        real_batch = torch.randn(16, 2, dtype=torch.float64)
        trainer.step(real_batch)

    # Verify no particles were killed or moved
    assert trainer.birth_death.counters["moves"] == 0
    # Prior table changed ONLY via gradient optimization, never via discrete row overwrites
    assert trainer.birth_death.moved_rows is None


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
    assert trainer.policy.birth_death.cells == 128

    for step in range(40):
        real_batch = torch.randn(32, 2, dtype=torch.float64)
        trainer.step(real_batch)

    assert trainer.birth_death.counters["moves"] == 0
    assert trainer.birth_death.moved_rows is None


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
