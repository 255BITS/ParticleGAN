"""Behaviour pins for the feature-cell module split and unified transport planner.

The golden rows were produced by the pre-split ``feature_cells.py`` (three
separate planners) on this exact synthetic, well-separated fixture; the
unified planner must reproduce every action row, allocation and RNG draw.
"""
import importlib

import pytest
import torch

import particlegan.birth_death as birth_death
import particlegan.feature_cells as feature_cells
import particlegan.feature_reference as feature_reference
from particlegan.cell_statistics import certified_capacity, move_budget, q_order_statistic
from particlegan.feature_cells import FeatureCellSnapshot

# Every public name the pre-split module defined at top level.
LEGACY_PUBLIC = (
    "LLOYD_PASSES", "POWER_PASSES", "SCALE_PRIOR", "PARENT_RESERVOIR", "REAL_ANCHORS",
    "PARENT_RANK_CAP", "JITTER_STD", "JITTER_CAP", "Q", "population_policy",
    "SmallPopulationReferenceBirthDeath", "make_feature_cell_birth_death", "LatentLineage",
    "BoundedLatentGeometry", "bounded_jitter", "conditional_count_pvalues",
    "CELL_RESOLUTION_POLICY", "FeatureCellSnapshot", "paired_average_geometry",
    "FeatureCellBirthDeath", "ParticleBirthDeath", "prepare_mean_witness", "run_mean_phase",
    "initial_mean_diagnostics", "check_mean_diagnostics", "MEAN_POLICY", "PLANNING_POLICY",
    "INSIDE_POLICY", "PROJECTION_POLICY", "FIT_CHUNK", "plan_real_anchor_births",
    "learned_latent_features", "plan_residual_global_copies", "plan_residual_isolation",
    "apply_anchor_births", "novel_birth_diagnostics",
    "_features_ok", "_integer_allocate", "_group_integer_allocate", "_fit_cell_count")

OWNERS = {"LatentLineage": "latent_lineage", "BoundedLatentGeometry": "latent_geometry",
          "bounded_jitter": "latent_geometry", "conditional_count_pvalues": "cell_statistics",
          "population_policy": "cell_statistics", "FeatureCellSnapshot": "cell_snapshot",
          "paired_average_geometry": "cell_snapshot", "Q": "cell_statistics"}

EXPECTED = {
    "ordinary": dict(
        child=[8, 11, 4, 12, 2, 7, 5, 10, 1, 13, 3, 0, 9, 14, 6],
        parent=[294, 196, 368, 350, 322, 393, 507, 304, 356, 477, 139, 454, 242, 16, 118],
        kinds=[1]*15, moves=(0, 15, 0),
        support_deaths=[0, 0, 0, 0, 0, 0, 15, 0], support_births=[0, 7, 0, 2, 3, 0, 3, 0]),
    # (children, parents, death_policy or certified budget, within, between moves)
    "mass_clean": ([170, 256, 82, 14], [382, 116, 317, 501], "supported_surplus", 0, 4),
    "mass_guard": ([14, 12, 32, 1, 35, 13, 8, 26, 11, 6, 24, 9, 4, 7, 56, 0, 3, 5, 10, 38, 2],
                   [391, 356, 287, 132, 373, 485, 220, 461, 393, 507, 194, 445, 304, 477, 372,
                    206, 195, 161, 453, 186, 118], "count_certified_flagged_only", 0, 21),
    "global": ([8, 0, 7, 11, 6, 14, 10, 3, 4], [480, 156, 278, 410, 159, 406, 175, 493, 144],
               9, 9, 0),
}


def test_every_legacy_public_name_is_still_importable_from_feature_cells():
    missing = [name for name in LEGACY_PUBLIC if not hasattr(feature_cells, name)]
    assert not missing
    for name, owner in OWNERS.items():
        module = importlib.import_module(f"particlegan.{owner}")
        assert getattr(feature_cells, name) is getattr(module, name)


def test_feature_reference_is_a_named_frozen_fork_not_the_current_law():
    frozen = feature_reference.FrozenFeatureReferenceBirthDeath
    assert feature_reference.ParticleBirthDeath is frozen
    assert feature_cells.ParticleBirthDeath is frozen
    assert frozen is not birth_death.ParticleBirthDeath
    assert issubclass(feature_cells.FeatureCellBirthDeath, frozen)
    assert issubclass(feature_cells.SmallPopulationReferenceBirthDeath, frozen)
    assert "frozen" in frozen.__doc__.lower()
    # The shared numerical helpers are the current law's own functions.
    for name in ("_knn", "_levina_bickel", "_normal_score"):
        assert getattr(feature_reference, name) is getattr(birth_death, name)


def test_shared_budget_certificate_and_order_statistic_helpers():
    assert move_budget(512) == 25 and move_budget(512, 7) == 7 and move_budget(512, 99) == 25
    assert move_budget(512, -3) == -3
    difference = torch.tensor([.1, -.2, .0049], dtype=torch.float64)
    certified = torch.tensor([True, True, True])
    assert certified_capacity(100, difference, certified).tolist() == [10, 0, 0]
    assert certified_capacity(100, -difference, certified).tolist() == [0, 20, 0]
    assert certified_capacity(100, difference, ~certified).tolist() == [0, 0, 0]
    # ceil((1-Q)*rows) for Q=1/20.
    assert [q_order_statistic(n) for n in (1, 20, 21, 256)] == [1, 19, 20, 244]


def _fixture():
    g = torch.Generator().manual_seed(2026)
    f64 = dict(dtype=torch.float64, generator=g)
    centers = torch.tensor([[0., 0.], [6., 0.], [0., 6.], [6., 6.]], dtype=torch.float64)
    real = centers[torch.randint(4, (512,), generator=g)] + .5 * torch.randn(512, 2, **f64)
    weights = torch.tensor([.55, .25, .15, .05])
    pick = torch.multinomial(weights, 512, replacement=True, generator=g)
    query = centers[pick] + .5 * torch.randn(512, 2, **f64)
    query[:15] = torch.tensor([20., -20.], dtype=torch.float64) + torch.randn(15, 2, **f64)
    fake = centers[torch.multinomial(weights, 512, replacement=True, generator=g)]
    fake = fake + .5 * torch.randn(512, 2, **f64)
    fake[:150] = torch.tensor([20., -20.], dtype=torch.float64) + torch.randn(150, 2, **f64)
    snapshot = FeatureCellSnapshot.fit(real, generator=torch.Generator().manual_seed(0),
                                       cells=8, rank=2, chunk=128)
    flags, _, _ = snapshot.support(query)
    flags[:15] = True
    comparison = snapshot.cell_comparison(fake)
    snapshot.cache_queries(query)
    return snapshot, query, flags, comparison


def test_unified_planner_reproduces_the_pre_split_joint_transport_rows():
    snapshot, query, flags, comparison = _fixture()
    generator = torch.Generator().manual_seed(5)
    child, parent, detail = snapshot.ordinary_transport(query, flags, comparison,
                                                        generator=generator)
    expected = EXPECTED["ordinary"]
    assert child.tolist() == expected["child"]
    assert parent.tolist() == expected["parent"]
    assert detail["action_kinds"].tolist() == expected["kinds"]
    assert (detail["mass_moves"], detail["support_moves"], detail["global_moves"]) == expected["moves"]
    assert detail["support_phase"]["death_allocation"].tolist() == expected["support_deaths"]
    assert detail["support_phase"]["birth_allocation"].tolist() == expected["support_births"]


@pytest.mark.parametrize("name", ["mass_clean", "mass_guard"])
def test_unified_planner_reproduces_both_pre_split_mass_death_policies(name):
    snapshot, query, flags, comparison = _fixture()
    flags = torch.zeros_like(flags)
    if name == "mass_guard":
        flags[:60] = True  # More than Q*N flags: the isolation guard rejects.
    child, parent, detail = snapshot._ordinary_mass_transport(
        query, flags, comparison["mass"], generator=torch.Generator().manual_seed(7))
    observed = (child.tolist(), parent.tolist(), detail["death_policy"],
                detail["within_group_moves"], detail["between_group_moves"])
    assert observed == EXPECTED[name]


def test_unified_planner_reproduces_the_pre_split_global_certified_budget():
    snapshot, query, flags, comparison = _fixture()
    child, parent, detail = snapshot._ordinary_global_transport(
        query, flags, comparison["global_support"], generator=torch.Generator().manual_seed(6),
        max_moves=9)
    observed = (child.tolist(), parent.tolist(), detail["certified_budget"],
                detail["within_group_moves"], detail["between_group_moves"])
    assert observed == EXPECTED["global"]


def test_mass_planner_now_validates_flags_like_the_other_phases():
    snapshot, query, flags, comparison = _fixture()
    with pytest.raises(ValueError, match="invalid isolation flags"):
        snapshot._ordinary_mass_transport(query, flags[:-1], comparison["mass"],
                                          generator=torch.Generator().manual_seed(0))
