"""Small deterministic geometry checks for the offline sampled-anchor witness."""

import unittest

import torch

from reports.toy100.sample_group_anchor import (
    controlled_fields, field, mst_groups, output_mm_step,
)


class SampleGroupAnchorTest(unittest.TestCase):
    def test_mst_gap_discovers_groups_without_configured_count(self):
        offsets = torch.tensor([[-0.03, 0.0], [0.0, 0.02],
                                [0.03, -0.01], [0.01, -0.03]], dtype=torch.float64)
        three_centers = torch.tensor([[0.0, 0.0], [3.0, 0.0],
                                      [0.0, 3.0]], dtype=torch.float64)
        real = torch.cat([center + offsets for center in three_centers])
        centers, receipt = mst_groups(real)
        self.assertEqual(receipt['n_groups'], 3)
        self.assertEqual(receipt['member_sizes'], [4, 4, 4])
        self.assertGreater(receipt['largest_additive_gap'], 10 * receipt['second_largest_gap'])
        self.assertLess(torch.cdist(centers, three_centers).min(dim=1).values.max(), 0.02)

        two_centers, two_receipt = mst_groups(real[:8])
        self.assertEqual(two_receipt['n_groups'], 2)
        self.assertEqual(len(two_centers), 2)

    def test_wrong_subset_gets_distinct_nonlocal_donors_and_good_support_rests(self):
        centers = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]], dtype=torch.float64)
        wrong_support = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                      [3.0, 0.0], [3.0, 0.0]], dtype=torch.float64)
        receipt = controlled_fields(wrong_support, centers)
        self.assertEqual(receipt['actual_missing_groups'], [2])
        donor = receipt['wrong_nonlocal_donors'][0]
        self.assertEqual(donor['group'], 2)
        self.assertGreater(donor['distance'], 0)
        self.assertGreater(receipt['wrong_centered']['gradient_l2'], 0)
        step = output_mm_step(wrong_support, centers)
        self.assertLess(step['after'], step['before'])
        self.assertGreater(len(step['target_occupied_groups']), 2)
        self.assertEqual(receipt['good']['total'], 0)
        self.assertEqual(receipt['good']['gradient_l2'], 0)
        self.assertLess(receipt['good_perturbed']['restoring_component'], 0)

    def test_injective_cover_can_leave_extra_particles_at_any_center(self):
        centers = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
        support = torch.tensor([[0.0, 0.0], [2.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
        rested = field(support, centers)
        self.assertEqual(rested['total'], 0)
        self.assertEqual(rested['gradient_l2'], 0)
        self.assertEqual(len(set(row['particle'] for row in rested['assignments'])), 2)
        with self.assertRaisesRegex(ValueError, 'more observed groups'):
            field(support[:1], centers)


if __name__ == '__main__':
    unittest.main()
