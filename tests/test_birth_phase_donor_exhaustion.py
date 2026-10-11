"""CPU function regressions for finite anchor-birth donor exhaustion.

These tests construct tensors and protocol stub state only. They require no
model, forward pass, optimizer, random draw, CUDA, solver or scientific fixture.
"""
from copy import deepcopy
import itertools
import unittest

import torch

from particlegan.birth_phase import allocate_anchor_births


class CountFixture:
    """Source protocol fixture for five cells with unoccupied target cells0..3.

    Four outside rows certify four deaths. Two of those are protected solver
    seeds, leaving only donor rows0 and1 even though capacity remains four.
    """
    cells = 5
    valid_metric = True

    def __init__(self):
        self.reference_counts = torch.ones(self.cells, dtype=torch.long)
        self.support_calls = 0

    def assign(self, features):
        return features[:, 0].long(), torch.zeros(len(features), dtype=torch.float64)

    def count_categories(self, features):
        return 2 * features[:, 0].long() + features[:, 1].long()

    def support(self, features):
        self.support_calls += 1
        outside = features[:, 1] != 0
        return outside, torch.where(outside, 0.01, 0.95), torch.zeros(len(features))

    def _mass_targets(self, n):
        return torch.full((self.cells,), n // self.cells, dtype=torch.long)

    def _mass_topology(self):
        return torch.arange(self.cells, dtype=torch.long)

    def _group_counts(self, counts):
        return counts.clone()


def problem(outside=(0, 1, 2, 3), *, n=80, cells=(0, 1, 2, 3), accepted=(True, True, True, True),
            prior_children=None, prior_parents=None, max_moves=None):
    snapshot = CountFixture()
    features = torch.zeros((n, 2), dtype=torch.float64)
    features[:, 0] = 4
    if outside:
        features[torch.tensor(outside, dtype=torch.long), 1] = 1
    flags = features[:, 1].bool()
    pvalues = torch.where(flags, 0.01, 0.95)
    proportion = len(outside) / n
    comparison = {'multiplicity': 3 * snapshot.cells + 3,
                  'cutoff': 0.05 / (3 * snapshot.cells + 3),
                  'family_sizes': (snapshot.cells, 2 * snapshot.cells, 2, 1),
                  'global_support': {'difference': torch.tensor([-proportion, proportion], dtype=torch.float64),
                                     'excess': torch.tensor([False, True]), 'deficit': torch.tensor([True, False])}}
    attempts = []
    for cell, source, ok in zip(cells, (2, 3, 4, 5), accepted, strict=True):
        current = {'accepted': True, 'features': torch.tensor([[cell, 0.]], dtype=torch.float64),
                   'latent': torch.tensor([cell + 0.25, cell + 0.75], dtype=torch.float64)}
        average = {'accepted': True, 'features': torch.tensor([[cell, 0.]], dtype=torch.float64),
                   'latent': torch.tensor([cell + 0.5, cell + 1.], dtype=torch.float64)}
        attempts.append({'cell': cell, 'seed_row': source, 'accepted': ok, 'current': current, 'average': average})
    kwargs = {'max_moves': max_moves}
    if prior_children is not None:
        kwargs.update(previous_children=torch.tensor(prior_children, dtype=torch.long),
                      previous_copy_parents=torch.tensor(prior_parents, dtype=torch.long))
    return snapshot, features, flags, pvalues, comparison, attempts, kwargs


def run(data):
    snapshot, features, flags, pvalues, comparison, attempts, kwargs = data
    return allocate_anchor_births(snapshot, features, flags, pvalues, comparison, attempts, **kwargs)


def same(actual, expected):
    if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape and torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual: same(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True): same(a, b)
    else:
        assert actual == expected


class DonorExhaustionTests(unittest.TestCase):
    def test_certified_capacity_above_two_donors_safely_stops(self):
        snapshot, *rest = data = problem()
        result = run(data)
        self.assertEqual(result['children'].tolist(), [0, 1])
        self.assertEqual(result['source_seed_rows'].tolist(), [2, 3])
        self.assertEqual(result['target_cell_ids'].tolist(), [0, 1])
        self.assertEqual(result['moves'], 2)
        self.assertEqual(result['attempted_cells'], 4)
        self.assertEqual(result['budget'], 4)
        self.assertEqual(result['remaining_budget_after'], 2)
        self.assertEqual(int(result['certificates_before']['residual_death']), 4)
        self.assertEqual(int(result['certificates_after']['residual_death']), 2)
        self.assertEqual(int(result['certificates_after']['residual_birth']), 2)
        self.assertEqual(int(result['certificates_after']['spent_death']), 2)
        self.assertEqual(int(result['certificates_after']['spent_birth']), 2)
        self.assertEqual(len(torch.unique(result['children'])), result['moves'])
        self.assertFalse(bool(torch.isin(result['children'], torch.tensor([2, 3, 4, 5])).any()))
        self.assertEqual(snapshot.support_calls, 4)  # two paired successful proposals; no post-exhaustion work
        same(result['new_latents'], torch.stack([data[5][i]['current']['latent'] for i in (0, 1)]))
        same(result['paired_ema_latents'], torch.stack([data[5][i]['average']['latent'] for i in (0, 1)]))
        self.assertEqual(result['work_bound'], {'cells': 4, 'linearizations_per_model': 4, 'paired_models': 2})

    def test_rejected_and_paired_rejected_proposals_do_not_consume_donors(self):
        data = problem(accepted=(False, True, True, True))
        data[5][2]['average']['accepted'] = False
        result = run(data)
        self.assertEqual(result['children'].tolist(), [0, 1])
        self.assertEqual(result['source_seed_rows'].tolist(), [3, 5])
        self.assertEqual(result['target_cell_ids'].tolist(), [1, 3])
        self.assertEqual(result['accepted_attempts'], [data[5][1], data[5][3]])

    def test_duplicate_target_does_not_consume_a_donor(self):
        result = run(problem(cells=(0, 0, 1, 2)))
        self.assertEqual(result['children'].tolist(), [0, 1])
        self.assertEqual(result['source_seed_rows'].tolist(), [2, 4])
        self.assertEqual(result['target_cell_ids'].tolist(), [0, 1])

    def test_exact_exhaustion_is_unchanged(self):
        data = problem(accepted=(True, True, False, False))
        result = run(data)
        self.assertEqual(result['children'].tolist(), [0, 1])
        self.assertEqual(result['moves'], 2)
        self.assertEqual(result['source_seed_rows'].tolist(), [2, 3])
        self.assertEqual(int(result['certificates_after']['spent_birth']), 2)
        self.assertEqual(int(result['certificates_after']['spent_death']), 2)

    def test_empty_protected_donor_pool_is_unchanged_and_does_no_acceptance_work(self):
        data = problem(outside=(2, 3))
        result = run(data)
        self.assertEqual(result['moves'], 0)
        self.assertEqual(data[0].support_calls, 0)
        self.assertIsNone(result['new_latents']); self.assertIsNone(result['paired_ema_latents'])

    def test_capacity_bound_still_precedes_other_proposals(self):
        data = problem(max_moves=1)
        result = run(data)
        self.assertEqual(result['moves'], 1); self.assertEqual(result['budget'], 1)
        self.assertEqual(data[0].support_calls, 2)

    def test_four_donors_and_unexhausted_behavior_are_unchanged(self):
        data = problem(outside=(0, 1, 6, 7))
        result = run(data)
        self.assertEqual(result['children'].tolist(), [0, 1, 6, 7]); self.assertEqual(result['moves'], 4)
        self.assertEqual(int(result['certificates_after']['residual_death']), 0)
        self.assertEqual(int(result['certificates_after']['residual_birth']), 0)

    def test_prior_actions_and_solver_sources_keep_their_reservations(self):
        data = problem(outside=(0, 1, 2, 3, 6), n=100, prior_children=(6,), prior_parents=(8,))
        result = run(data)
        self.assertEqual(result['children'].tolist(), [0, 1])
        self.assertEqual(result['earlier_moves'], 1)
        self.assertEqual(result['remaining_budget'], 4); self.assertEqual(result['remaining_budget_after'], 2)
        self.assertEqual(result['reserved_rows_after'].tolist(), [6, 8, 0, 1, 2, 3])
        self.assertEqual(result['previous_children_after'].tolist(), [6, 0, 1])
        self.assertEqual(result['previous_birth_categories_after'].tolist(), [8, 0, 2])
        self.assertEqual(int(result['certificates_before']['spent_death']), 1)
        self.assertEqual(int(result['certificates_after']['spent_death']), 3)
        self.assertEqual(int(result['certificates_after']['spent_birth']), 3)
        self.assertEqual(int(result['certificates_after']['residual_death']), 2)
        self.assertEqual(int(result['certificates_after']['residual_birth']), 2)

    def test_exhausted_plan_does_not_modify_inputs_and_owns_paired_latent_copies(self):
        data = problem(); inputs = deepcopy(data[1:])
        result = run(data)
        same(data[1:], inputs)
        before_current = result['new_latents'].clone(); before_average = result['paired_ema_latents'].clone()
        data[5][0]['current']['latent'].fill_(99); data[5][0]['average']['latent'].fill_(88)
        same(result['new_latents'], before_current); same(result['paired_ema_latents'], before_average)

    def test_all_rejection_patterns_preserve_success_order_and_donor_uniqueness(self):
        for pattern in itertools.product((False, True), repeat=4):
            with self.subTest(accepted=pattern):
                data = problem(accepted=pattern); result = run(data)
                accepted_indices = [i for i, ok in enumerate(pattern) if ok][:2]
                self.assertEqual(result['children'].tolist(), list(range(len(accepted_indices))))
                self.assertEqual(result['source_seed_rows'].tolist(), [2 + i for i in accepted_indices])
                self.assertEqual(result['target_cell_ids'].tolist(), accepted_indices)
                self.assertEqual(result['moves'], min(2, sum(pattern)))
                self.assertEqual(int(result['certificates_after']['spent_death']), result['moves'])
                self.assertEqual(result['remaining_budget_after'], result['budget'] - result['moves'])


if __name__ == '__main__':
    unittest.main()
