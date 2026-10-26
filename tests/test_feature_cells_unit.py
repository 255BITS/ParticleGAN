"""Direct unit tests for the public helpers in ``particlegan.feature_cells``.

They pin the current behaviour of the stand-alone pieces (population rule,
lineage graph, bounded latent geometry, jitter and exact count tests) so the
planned split of this module can be checked as behaviour preserving. Trainer
level feature-cell behaviour stays covered by test_feature_auto.py.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from particlegan.feature_cells import (
    JITTER_CAP, JITTER_STD, BoundedLatentGeometry, FeatureCellBirthDeath, LatentLineage,
    SmallPopulationReferenceBirthDeath, bounded_jitter, conditional_count_pvalues,
    make_feature_cell_birth_death, population_policy,
)

ROOT = Path(__file__).resolve().parents[1]


# population_policy -----------------------------------------------------------

def test_population_policy_switches_backend_exactly_at_the_finite_resolution_bound():
    below, at = population_policy(799), population_policy(800)
    assert (below["minimum_bh_flags"], below["maximum_guard_flags"]) == (40, 39)
    assert below["finite_resolution_feasible"] is False
    assert (below["actual_backend"], below["matching_sampler"]) == ("knn", "controller_reference")
    assert (at["minimum_bh_flags"], at["maximum_guard_flags"], at["calibration_rows"]) == (40, 40, 400)
    assert at["finite_resolution_feasible"] is True
    assert (at["actual_backend"], at["matching_sampler"]) == ("feature_cells", "feature_cells")
    feasible = [n for n in range(1, 1200) if population_policy(n)["finite_resolution_feasible"]]
    assert feasible == list(range(800, 1200))


@pytest.mark.parametrize("n", [0, -1, 2.0, True, "800"])
def test_population_policy_rejects_non_positive_integers(n):
    with pytest.raises(ValueError, match="positive integer"):
        population_policy(n)


def _trainer(n, backend):
    from particlegan.particle_prior import ParticlePrior
    from particlegan.recipes import Recipe
    from particlegan.training import GANTrainer

    base = json.loads((ROOT / "tests/fixtures/feature-auto-base.json").read_text())
    config = dict(base, z_dim=2, num_particles=n, batch_size=32, birth_death_backend=backend)
    if backend == "knn":
        for key in ("birth_death_cells", "birth_death_metric_rank", "birth_death_chunk", "birth_death_parent_policy"):
            config.pop(key)
    torch.manual_seed(5)
    G = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
    D = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    prior = ParticlePrior(n, 2, generator=torch.Generator().manual_seed(6))
    return GANTrainer(Recipe(**config), G, D, prior=prior, seed=5, serial_backward=True)


def test_factory_routes_small_populations_to_the_reference_law_with_metadata():
    trainer = _trainer(64, "knn")
    birth_death = make_feature_cell_birth_death(trainer, 11)
    assert type(birth_death) is SmallPopulationReferenceBirthDeath
    assert birth_death.diagnostics()["backend"] == "knn"
    state = birth_death.state_dict()
    assert state["population_policy"] == population_policy(64)
    birth_death.check_state(state)
    with pytest.raises(ValueError, match="small-population backend"):
        birth_death.check_state({**state, "population_policy": population_policy(65)})
    with pytest.raises(ValueError, match="impossible feature-cell resolution"):
        SmallPopulationReferenceBirthDeath(_trainer(800, "knn"), 11)


def test_factory_routes_feasible_populations_to_feature_cells():
    assert type(make_feature_cell_birth_death(_trainer(800, "feature_cells"), 11)) is FeatureCellBirthDeath


# LatentLineage ----------------------------------------------------------------

def rows(*values):
    return torch.tensor(values, dtype=torch.long)


def edges(lineage):
    pairs = set()
    for row, partners in enumerate(lineage.neighbors.tolist()):
        pairs.update(frozenset((row, partner)) for partner in partners if partner >= 0)
    return pairs


def test_lineage_registers_symmetric_newest_first_links():
    lineage = LatentLineage(6, 2, "cpu")
    lineage.register_copies(rows(1), rows(0))
    assert lineage.neighbors.tolist()[:2] == [[1, -1], [0, -1]]
    lineage.register_copies(rows(2), rows(0))
    assert lineage.neighbors[0].tolist() == [2, 1]
    lineage.validate(lineage.neighbors)
    assert edges(lineage) == {frozenset((0, 1)), frozenset((0, 2))}
    assert lineage.work["updates"] == 2 and lineage.work["copied_rows"] == 2


def test_full_parent_evicts_its_oldest_edge_on_both_endpoints():
    lineage = LatentLineage(6, 2, "cpu")
    for child in (1, 2, 3):
        lineage.register_copies(rows(child), rows(0))
    assert lineage.neighbors[0].tolist() == [3, 2]
    assert lineage.neighbors[1].tolist() == [-1, -1]
    assert edges(lineage) == {frozenset((0, 2)), frozenset((0, 3))}
    lineage.validate(lineage.neighbors)


def test_overwritten_row_loses_every_old_edge():
    lineage = LatentLineage(6, 3, "cpu")
    lineage.register_copies(rows(1, 2), rows(0, 0))
    lineage.register_copies(rows(0), rows(4))
    assert edges(lineage) == {frozenset((0, 4))}
    lineage.validate(lineage.neighbors)


def test_parent_overwritten_in_the_same_batch_registers_no_edge_to_its_new_incarnation():
    lineage = LatentLineage(6, 2, "cpu")
    lineage.register_copies(rows(1, 2), rows(2, 3))
    assert edges(lineage) == {frozenset((2, 3))}


@pytest.mark.parametrize("n,degree", [(4, 0), (4, 4), (0, 1), (4, 1.0)])
def test_lineage_rejects_invalid_size_or_degree(n, degree):
    with pytest.raises(ValueError, match="invalid latent lineage size/degree"):
        LatentLineage(n, degree, "cpu")


@pytest.mark.parametrize("children,parents", [
    (rows(1, 1), rows(0, 2)), (rows(9), rows(0)), (rows(1), rows(-1)), (rows(1, 2), rows(0)),
])
def test_lineage_rejects_invalid_copy_rows(children, parents):
    with pytest.raises(ValueError):
        LatentLineage(6, 2, "cpu").register_copies(children, parents)


@pytest.mark.parametrize("table,match", [
    ([[0, -1], [-1, -1], [-1, -1]], "self edges"),
    ([[1, -1], [-1, -1], [-1, -1]], "symmetric"),
    ([[1, 1], [0, -1], [-1, -1]], "repeated neighbors"),
    ([[3, -1], [-1, -1], [-1, -1]], "shape/indices"),
])
def test_lineage_checkpoint_validation_rejects_malformed_graphs(table, match):
    with pytest.raises(ValueError, match=match):
        LatentLineage(3, 2, "cpu").validate(torch.tensor(table, dtype=torch.long))


# BoundedLatentGeometry and jitter ---------------------------------------------

def _points():
    return torch.tensor([[0., 0.], [1., 0.], [3., 0.], [3., 2.5], [-2., 1.]], dtype=torch.float64)


def test_radius_is_half_the_nearest_other_distance_when_candidates_cover_the_table():
    points = _points()
    prior = SimpleNamespace(z=points)
    radius = BoundedLatentGeometry(rank=2, neighbors=64).radius(points, prior)
    distance = torch.cdist(points, points)
    distance.fill_diagonal_(float("inf"))
    torch.testing.assert_close(radius, .5 * distance.min(1).values)


def test_displacement_never_exceeds_the_local_radius_and_records_diagnostics():
    points = _points()
    prior = SimpleNamespace(z=points)
    geometry = BoundedLatentGeometry(rank=2, neighbors=64)
    noise = torch.tensor([[10., 0.], [0., 10.], [.01, .01], [-3., 4.], [0., 0.]], dtype=torch.float64)
    record = []
    delta = geometry.displacement(points, prior, torch.tensor(1., dtype=torch.float64), noise, record=record)
    radius = geometry.radius(points, prior)
    assert bool((delta.norm(dim=1) <= radius + 1e-12).all())
    assert delta[4].abs().sum() == 0
    assert record[-1]["kernel"] == "bounded_local_dv12" and 0 <= record[-1]["clipped_fraction"] <= 1
    assert geometry.work["builds"] == 1  # the sort cache is reused for an unchanged table


@pytest.mark.parametrize("name", ["rank", "neighbors", "chunk"])
def test_geometry_rejects_non_positive_sizes(name):
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        BoundedLatentGeometry(**{name: 0})


def test_bounded_jitter_scales_small_noise_and_caps_large_noise():
    small = torch.tensor([[.1, -.2]])
    torch.testing.assert_close(bounded_jitter(None, small), JITTER_STD * small)
    large = torch.tensor([[30., 40.]])
    capped = bounded_jitter(None, large)
    torch.testing.assert_close(capped.norm(dim=1), torch.tensor([JITTER_CAP]))
    torch.testing.assert_close(capped / capped.norm(), large / large.norm())


# conditional_count_pvalues ----------------------------------------------------

def test_count_pvalues_equal_fisher_exact_per_cell():
    stats = pytest.importorskip("scipy.stats")
    real, fake = torch.tensor([7, 3, 10]), torch.tensor([2, 9, 9])
    work = {"count_test_terms": 0}
    actual = conditional_count_pvalues(real, fake, 20, 20, work)
    expected = [stats.fisher_exact([[int(r), 20 - int(r)], [int(f), 20 - int(f)]])[1]
                for r, f in zip(real, fake)]
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64), rtol=1e-9, atol=1e-12)
    assert work["count_test_terms"] > 0


def test_count_pvalues_are_one_for_identical_counts():
    counts = torch.tensor([4, 4, 2])
    torch.testing.assert_close(conditional_count_pvalues(counts, counts.clone(), 10, 10),
                               torch.ones(3, dtype=torch.float64))


@pytest.mark.parametrize("real,fake,real_rows,fake_rows", [
    (torch.tensor([1, 2]), torch.tensor([1, 2]), 4, 3),        # sums disagree with rows
    (torch.tensor([-1, 4]), torch.tensor([1, 2]), 3, 3),       # negative count
    (torch.tensor([1., 2.]), torch.tensor([1, 2]), 3, 3),      # floating counts
    (torch.tensor([1, 2]), torch.tensor([1, 2, 0]), 3, 3),     # shape mismatch
])
def test_count_pvalues_reject_invalid_tables(real, fake, real_rows, fake_rows):
    with pytest.raises(ValueError, match="invalid categorical counts"):
        conditional_count_pvalues(real, fake, real_rows, fake_rows)
