"""Direct unit tests for ``particlegan.birth_phase`` and ``particlegan.anchor_birth``.

A small geometric snapshot stands in for a fitted feature-cell snapshot: cells
are real representatives in a two-dimensional feature space and a point is
"inside" its nearest cell within radius 0.1. The latent-to-feature map is
linear with one null latent direction that carries the prior spread (and so
the solver's trust radius), so the Newton solver's behaviour is predictable. These pin
current behaviour as a safety net for the planned feature_cells split.
"""
from types import SimpleNamespace

import pytest
import torch

from particlegan.anchor_birth import latent_to_anchor, propose_inaccessible_anchors
from particlegan.birth_phase import (
    allocate_anchor_births, apply_anchor_births, global_certificate_residual, invalidate_birth_rows,
    novel_birth_diagnostics, plan_real_anchor_births, plan_residual_global_copies,
)
from particlegan.feature_cells import LatentLineage

INSIDE = .1


class Snapshot:
    """Cells are real representatives; categories are 2*cell (+1 when outside)."""
    valid_metric = True

    def __init__(self, representatives):
        self.real_representatives = torch.as_tensor(representatives, dtype=torch.float64)
        self.cells = len(self.real_representatives)
        width = self.real_representatives.shape[1]
        self.mean, self.scale = torch.zeros(width, dtype=torch.float64), torch.ones(width, dtype=torch.float64)
        self.basis = torch.eye(width, dtype=torch.float64)
        self.reference_counts = torch.ones(self.cells, dtype=torch.long)

    def transform(self, features):
        return ((features.double() - self.mean) / self.scale) @ self.basis

    def assign(self, features):
        distance = torch.cdist(self.transform(features), self.real_representatives)
        return distance.argmin(1), distance.min(1).values

    def count_categories(self, features):
        ids, distance = self.assign(features)
        return 2 * ids + (distance > INSIDE).long()

    def support(self, features):
        _, distance = self.assign(features)
        outside = distance > INSIDE
        return outside, torch.where(outside, .01, .95).double(), distance

    def _mass_targets(self, n):
        return torch.full((self.cells,), n // self.cells, dtype=torch.long)

    def _mass_topology(self):
        return torch.arange(self.cells)

    def _group_counts(self, counts):
        return counts.clone()


MAP = torch.tensor([[1., .5, 0.], [0., 1., 0.]], dtype=torch.float64)


def feature_of_latent(latent):
    return latent.double() @ MAP.T


def comparison(snapshot, n, outside_fraction):
    k = snapshot.cells
    return {"multiplicity": 3 * k + 3, "cutoff": .05 / (3 * k + 3), "family_sizes": (k, 2 * k, 2, 1),
            "global_support": {"difference": torch.tensor([-outside_fraction, outside_fraction], dtype=torch.float64),
                               "excess": torch.tensor([False, True]), "deficit": torch.tensor([True, False])}}


def table(n=80, outside=(0, 1, 2, 3)):
    """Every row sits inside cell 0 except the listed rows, which are outside it."""
    snapshot = Snapshot([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    latents = torch.zeros(n, 3, dtype=torch.float64)
    latents[:, 2] = torch.linspace(-1.7, 1.7, n)  # null direction: features ignore it
    for row in outside:
        latents[row, :2] = torch.tensor([.4, .4])  # features (.6, .4): outside every cell
    features = feature_of_latent(latents)
    flags, pvalues, _ = snapshot.support(features)
    return snapshot, latents, features, flags, pvalues, comparison(snapshot, n, len(outside) / n)


# anchor_birth.latent_to_anchor -------------------------------------------------

def test_newton_solver_reaches_the_anchor_within_the_trust_bound_without_rng():
    snapshot, latents, *_ = table()
    rng = torch.get_rng_state()
    result = latent_to_anchor(snapshot, feature_of_latent, latents[10], 3, latents)
    assert torch.equal(rng, torch.get_rng_state())
    assert result["accepted"] and result["category"] == 6 and result["support_pvalue"] > .05
    torch.testing.assert_close(feature_of_latent(result["latent"][None])[0], snapshot.real_representatives[3])
    assert result["total_latent_displacement"] <= result["linearizations"] * result["trust_radius"] + 1e-12
    assert result["progress"][0]["accepted"] is False and result["progress"][-1]["accepted"] is True


def test_newton_solver_stops_unaccepted_when_the_pass_budget_is_too_small():
    snapshot, latents, *_ = table()
    result = latent_to_anchor(snapshot, feature_of_latent, latents[10], 3, latents, passes=1)
    assert result["accepted"] is False and result["linearizations"] == 1
    assert result["total_latent_displacement"] == pytest.approx(result["trust_radius"])


@pytest.mark.parametrize("cell,passes,match", [
    (1, 0, "invalid real-anchor"), (1, 5, "invalid real-anchor"),
    (4, 4, "invalid anchor cell"), (-1, 4, "invalid anchor cell"),
])
def test_newton_solver_rejects_invalid_inputs(cell, passes, match):
    snapshot, latents, *_ = table()
    with pytest.raises(ValueError, match=match):
        latent_to_anchor(snapshot, feature_of_latent, latents[10], cell, latents, passes=passes)


def test_inaccessible_cells_get_proposals_from_their_nearest_unreserved_rows():
    snapshot, latents, features, flags, pvalues, _ = table()
    attempts = propose_inaccessible_anchors(snapshot, features, flags, pvalues, latents, feature_of_latent,
                                            reserved_rows=torch.tensor([0, 1]), candidate_limit=4)
    assert [attempt["cell"] for attempt in attempts] == [1, 2, 3]
    assert len({attempt["seed_row"] for attempt in attempts}) == 3
    assert not {attempt["seed_row"] for attempt in attempts} & {0, 1}
    assert all(attempt["accepted"] and attempt["average"] is None for attempt in attempts)
    assert propose_inaccessible_anchors(snapshot, features, flags, pvalues, latents, feature_of_latent,
                                        candidate_limit=0) == []


def test_proposals_reject_unbounded_or_unpaired_requests():
    snapshot, latents, features, flags, pvalues, _ = table()
    with pytest.raises(ValueError, match="four-anchor bound"):
        propose_inaccessible_anchors(snapshot, features, flags, pvalues, latents, feature_of_latent, candidate_limit=5)
    with pytest.raises(ValueError, match="paired EMA"):
        propose_inaccessible_anchors(snapshot, features, flags, pvalues, latents, feature_of_latent,
                                     ema_latents=latents)


# birth_phase ------------------------------------------------------------------

def test_global_certificate_counts_raw_and_spent_capacity():
    snapshot, _, features, _, _, compare = table()
    certificate = global_certificate_residual(snapshot, features, compare)
    assert (int(certificate["raw_death"]), int(certificate["raw_birth"])) == (4, 4)
    assert (int(certificate["residual_death"]), int(certificate["residual_birth"])) == (4, 4)
    assert certificate["outside_certified"] and certificate["inside_certified"]
    spent = global_certificate_residual(snapshot, features, compare, previous_children=torch.tensor([0, 9]),
                                        previous_birth_categories=torch.tensor([0, 2]))
    assert (int(spent["spent_death"]), int(spent["spent_birth"])) == (1, 2)
    assert (int(spent["residual_death"]), int(spent["residual_birth"])) == (3, 2)


@pytest.mark.parametrize("change,match", [
    (dict(multiplicity=7), "unchanged common 3K\\+3 count family"),
    (dict(family_sizes=(4, 8, 2, 2)), "unchanged common 3K\\+3 count family"),
])
def test_global_certificate_requires_the_common_count_family(change, match):
    snapshot, _, features, _, _, compare = table()
    with pytest.raises(ValueError, match=match):
        global_certificate_residual(snapshot, features, {**compare, **change})


@pytest.mark.parametrize("children,categories", [
    (torch.tensor([3, 3]), torch.tensor([0, 0])), (torch.tensor([80]), torch.tensor([0])),
    (torch.tensor([3]), torch.tensor([8])), (torch.tensor([3]), torch.tensor([0, 2])),
])
def test_global_certificate_rejects_invalid_earlier_actions(children, categories):
    snapshot, _, features, _, _, compare = table()
    with pytest.raises(ValueError):
        global_certificate_residual(snapshot, features, compare, previous_children=children,
                                    previous_birth_categories=categories)


def test_solver_seeds_are_never_reused_as_death_donors():
    # The outside rows are also the nearest seeds for every empty cell, so
    # only one unprotected donor remains for three accepted proposals.
    snapshot, latents, features, flags, pvalues, compare = table()
    plan = plan_real_anchor_births(snapshot, features, flags, pvalues, compare, latents, feature_of_latent,
                                   ema_latents=latents.clone(), ema_feature_of_latent=feature_of_latent)
    assert plan["attempted_cells"] == 3 and plan["moves"] == 1
    assert plan["source_seed_rows"].tolist() == [0] and plan["children"].tolist() == [3]


def test_production_plan_allocates_paired_births_to_outside_donors():
    snapshot, latents, features, flags, pvalues, compare = table(n=160, outside=range(8))
    plan = plan_real_anchor_births(snapshot, features, flags, pvalues, compare, latents, feature_of_latent,
                                   ema_latents=latents.clone(), ema_feature_of_latent=feature_of_latent)
    assert plan["moves"] == 3 and plan["budget"] == 8
    assert plan["target_cell_ids"].tolist() == [1, 2, 3]
    assert plan["destination_category_ids"].tolist() == [2, 4, 6]
    assert set(plan["children"].tolist()) <= set(range(8))
    assert not set(plan["children"].tolist()) & set(plan["source_seed_rows"].tolist())
    assert int(plan["certificates_after"]["residual_birth"]) == 5
    assert plan["planned_supported_counts"].tolist() == [152, 1, 1, 1]
    diagnostics = novel_birth_diagnostics(plan)
    assert diagnostics["moves"] == 3 and diagnostics["new_latent_destination_categories"] == [2, 4, 6]
    assert diagnostics["certificates_after"]["residual_birth"] == 5
    assert [entry["cell"] for entry in diagnostics["acceptance"]] == [1, 2, 3]


def test_allocation_honours_the_external_move_cap_and_rejects_bad_ledgers():
    snapshot, latents, features, flags, pvalues, compare = table(n=160, outside=range(8))
    attempts = propose_inaccessible_anchors(snapshot, features, flags, pvalues, latents, feature_of_latent,
                                            ema_latents=latents.clone(), ema_feature_of_latent=feature_of_latent)
    capped = allocate_anchor_births(snapshot, features, flags, pvalues, compare, attempts, max_moves=1)
    assert capped["moves"] == 1 and capped["budget"] == 1
    with pytest.raises(ValueError, match="exact supported ledger"):
        allocate_anchor_births(snapshot, features, flags, pvalues, compare, attempts,
                               supported_counts=torch.tensor([151, 0, 0, 0]))
    with pytest.raises(ValueError, match="four-source work bound"):
        allocate_anchor_births(snapshot, features, flags, pvalues, compare, attempts + attempts[:2])


def test_residual_global_copies_skip_without_remaining_certified_capacity():
    snapshot, latents, features, flags, pvalues, compare = table()
    plan = plan_real_anchor_births(snapshot, features, flags, pvalues, compare, latents, feature_of_latent,
                                   ema_latents=latents.clone(), ema_feature_of_latent=feature_of_latent,
                                   max_moves=0)
    empty = torch.empty(0, dtype=torch.long)
    child, parent, detail = plan_residual_global_copies(snapshot, features, flags, pvalues, compare, plan,
                                                        generator=None, previous_children=empty,
                                                        previous_copy_parents=empty)
    assert child.numel() == parent.numel() == 0
    assert detail["ran"] is False and detail["moves"] == 0 and detail["budget"] == 0


def test_invalidate_birth_rows_removes_overwritten_incarnations_from_the_lineage():
    lineage = LatentLineage(6, 2, "cpu")
    lineage.register_copies(torch.tensor([1, 2]), torch.tensor([0, 3]))
    invalidate_birth_rows(lineage, torch.tensor([1]))
    assert lineage.neighbors[1].tolist() == [-1, -1] and 1 not in lineage.neighbors[0].tolist()
    lineage.validate(lineage.neighbors)
    with pytest.raises(ValueError, match="distinct"):
        invalidate_birth_rows(lineage, torch.tensor([2, 2]))


def _commit_fixture():
    z = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(6, 2))
    prior, ema = SimpleNamespace(z=z), SimpleNamespace(z=z.detach().clone() + 100)
    optimizer = SimpleNamespace(state={z: {"step": torch.tensor(5.), "exp_avg": torch.ones(6, 2)}},
                                latent_history=torch.ones(6, 2))
    lineage = LatentLineage(6, 2, "cpu")
    lineage.register_copies(torch.tensor([1]), torch.tensor([0]))
    birth_death = SimpleNamespace(lineage=lineage, anchor=z.detach().clone(),
                                  **{key: torch.ones(6) for key in ("S", "W", "n", "pending", "radius")})
    return SimpleNamespace(prior=prior, ema_prior=ema, opt_g=optimizer), birth_death


def test_apply_anchor_births_commits_paired_rows_and_clears_row_state():
    trainer, birth_death = _commit_fixture()
    before = trainer.prior.z.detach().clone()
    plan = dict(children=torch.tensor([1, 4]), source_seed_rows=torch.tensor([2, 3]),
                new_latents=torch.tensor([[-1., -2.], [-3., -4.]]),
                paired_ema_latents=torch.tensor([[-5., -6.], [-7., -8.]]))
    apply_anchor_births(trainer, birth_death, plan)
    assert torch.equal(trainer.prior.z.detach()[[1, 4]], plan["new_latents"])
    assert torch.equal(trainer.ema_prior.z[[1, 4]], plan["paired_ema_latents"])
    untouched = [0, 2, 3, 5]
    assert torch.equal(trainer.prior.z.detach()[untouched], before[untouched])
    state = trainer.opt_g.state[trainer.prior.z]
    assert state["step"] == 5 and state["exp_avg"][[1, 4]].abs().sum() == 0 and state["exp_avg"][0].sum() == 2
    assert trainer.opt_g.latent_history[[1, 4]].abs().sum() == 0
    assert all(getattr(birth_death, key)[[1, 4]].abs().sum() == 0 for key in ("S", "W", "n", "pending", "radius"))
    assert torch.equal(birth_death.anchor[[1, 4]], plan["new_latents"])
    assert birth_death.lineage.neighbors[1].tolist() == [-1, -1]


@pytest.mark.parametrize("change,match", [
    (dict(source_seed_rows=torch.tensor([1, 3])), "distinct paired"),
    (dict(new_latents=torch.tensor([[float("nan"), 0.], [0., 0.]])), "invalid paired new-latent"),
    (dict(paired_ema_latents=torch.zeros(1, 2)), "invalid paired new-latent"),
    (dict(children=torch.tensor([1, 1])), "invalid reserved birth rows"),
])
def test_apply_anchor_births_rejects_invalid_plans_before_mutation(change, match):
    trainer, birth_death = _commit_fixture()
    before = trainer.prior.z.detach().clone()
    plan = dict(children=torch.tensor([1, 4]), source_seed_rows=torch.tensor([2, 3]),
                new_latents=torch.zeros(2, 2), paired_ema_latents=torch.zeros(2, 2))
    plan.update(change)
    with pytest.raises(ValueError, match=match):
        apply_anchor_births(trainer, birth_death, plan)
    assert torch.equal(trainer.prior.z.detach(), before)
