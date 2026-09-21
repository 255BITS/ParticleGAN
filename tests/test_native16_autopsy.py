"""The PR #16 action-MSE gate accepts controllers that miss a state-feedback pad."""
import unittest

from experiments.toy_native16_autopsy import (FIXED_MAX, prev_plant_dead_controller,
    state_plant_counterexamples)


class Native16AutopsyTests(unittest.TestCase):
    def test_previous_only_target_makes_a_dead_action_look_closed_loop_accurate(self):
        row = prev_plant_dead_controller()
        self.assertGreater(row["zero_tf"], 0.4)
        self.assertLess(row["zero_closed"], row["zero_tf"] * 0.3)
        self.assertEqual(row["perfect_tf"], 0.)
        self.assertEqual(row["perfect_closed"], 0.)

    def test_pass_band_misses_the_pad_and_expert_previous_hides_the_leak(self):
        rows = state_plant_counterexamples()
        self.assertTrue(rows["ok"], rows)
        bias, leak, expert = rows["bias"], rows["prev_leak"], rows["expert"]
        self.assertGreaterEqual(bias["tf_mse"], 0.07)
        self.assertLessEqual(bias["tf_mse"], FIXED_MAX)
        self.assertTrue(bias["old_pass"])
        self.assertEqual(bias["land_rate"], 0.)
        self.assertFalse(bias["honest_pass"])
        self.assertEqual(expert["land_rate"], 1.)
        self.assertTrue(expert["honest_pass"])
        self.assertTrue(leak["old_pass"])
        self.assertFalse(leak["honest_pass"])
        self.assertGreater(leak["on_policy_mse"], leak["tf_mse"])


if __name__ == "__main__":
    unittest.main()
