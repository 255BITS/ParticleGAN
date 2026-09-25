"""CPU gate: combined safe-fast RpGAN beats GAN-only on rate and steps. One seed."""
import unittest

import torch

from lib.safe_fast_landing import (CRASH_SINK, CRASH_WEIGHT, HORIZON, PAD, QUICK_SINK, SLOW,
    SUCCESS_BONUS, TIME_WEIGHT, V_LIMIT, evaluate_policy, expert_action, initial_states,
    kinematic_step, pd_action, reference_policies, require_live_adversary, rollout_cost,
    run_gate, train_arm)


def _sink_cost(states, sink, speed_limit=V_LIMIT):
    def transition(state):
        return kinematic_step(state, pd_action(state, sink))

    return float(rollout_cost(states, transition, HORIZON, TIME_WEIGHT, CRASH_WEIGHT,
                              SUCCESS_BONUS, speed_limit, PAD))


class SafeFastToyTests(unittest.TestCase):
    def test_score_ranks_quick_soft_landing_over_hover_and_crash(self):
        refs = reference_policies()
        self.assertEqual(refs["hover"]["landings"], 0)
        self.assertEqual(refs["crash_sink"]["landings"], 0)
        self.assertGreaterEqual(refs["quick_soft"]["landings"], 0.95)
        self.assertLess(refs["hover"]["score"], refs["quick_soft"]["score"])
        self.assertLess(refs["crash_sink"]["score"], refs["quick_soft"]["score"])
        self.assertLess(refs["quick_soft"]["mean_steps"], float(HORIZON))

    def test_soft_cost_prefers_a_quick_safe_sink(self):
        states = initial_states(64, 1000)
        quick = _sink_cost(states, QUICK_SINK)
        slow = _sink_cost(states, SLOW)
        crash = _sink_cost(states, CRASH_SINK)
        self.assertLess(quick, slow)
        self.assertLess(quick, crash)
        self.assertGreater(_sink_cost(states, QUICK_SINK, speed_limit=0.05), quick)

    def test_zero_adv_weight_is_rejected_on_the_gan_arms(self):
        with self.assertRaises(ValueError) as caught:
            train_arm("combined", steps=1, adv_weight=0)
        self.assertIn("adv_weight=0 leaves RpGAN and b_cap configured but not applied",
                      str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            require_live_adversary(0.25)
        self.assertIn("adv_weight stays 1", str(caught.exception))

    def test_gate_combined_beats_baseline_with_the_gan_left_on(self):
        result = run_gate()
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["baseline"]["adv_weight"], 1)
        self.assertEqual(result["baseline"]["safe_fast_weight"], 0)
        self.assertGreater(result["baseline"]["gan_grad_abs"], 1)
        self.assertGreater(result["baseline"]["late_gan_grad"], 0.01)
        self.assertGreater(result["baseline"]["b_cap_applications"], 0)
        self.assertEqual(result["combined"]["adv_weight"], 1)
        self.assertEqual(result["combined"]["safe_fast_weight"], 1)
        self.assertGreater(result["combined"]["gan_grad_abs"], 1)
        self.assertGreater(result["combined"]["late_gan_grad"], 0.2)
        self.assertGreater(result["combined"]["safe_fast_grad_abs"], 1)
        self.assertGreater(result["combined"]["b_cap_applications"], 0)
        self.assertGreaterEqual(result["combined"]["landings"], 0.95)
        self.assertLessEqual(result["baseline"]["landings"], 0.25)
        self.assertGreaterEqual(result["combined"]["landings"], result["baseline"]["landings"] + 0.50)
        self.assertLessEqual(result["combined"]["mean_steps"], 28)
        self.assertGreaterEqual(result["baseline"]["mean_steps"], 40)
        self.assertLessEqual(result["combined"]["mean_steps"], result["baseline"]["mean_steps"] - 12)
        self.assertEqual(result["supervised"]["adv_weight"], 0)
        self.assertFalse(result["supervised"]["accepted"])
        self.assertGreaterEqual(result["supervised"]["landings"], 0.95)
        self.assertLess(result["combined"]["sink"], V_LIMIT)
        # The expert action is the slow law the baseline is pulled toward.
        state = initial_states(4, 11)
        torch.testing.assert_close(expert_action(state), pd_action(state, SLOW))


if __name__ == "__main__":
    unittest.main()
