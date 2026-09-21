"""Closed-loop gate: the shipped #21 shaping fails, the throttle-up revision lands."""
import unittest

import torch

from lib.safe_fast_landing import (CRASH_SINK, SLOW, SOFT_SPEED_LIMIT, V21_SPEED_LIMIT,
    closed_loop_step, evaluate_policy, expert_action, initial_states, kinematic_step,
    require_live_adversary, run_gate, throttle_layout, train_arm)


class SafeFastToyTests(unittest.TestCase):
    def test_throttle_layout_is_not_the_bipolar_unpack(self):
        action = torch.tensor([[-1., 0.4], [1., -0.2]])
        laid = throttle_layout(action)
        torch.testing.assert_close(laid[0], torch.tensor([0.4, 0.]))
        torch.testing.assert_close(laid[1], torch.tensor([-0.2, 1.]))
        state = initial_states(2, 5)
        torch.testing.assert_close(closed_loop_step(state, action), kinematic_step(state, laid))
        swapped = kinematic_step(state, action)
        self.assertGreater(float((swapped - closed_loop_step(state, action)).abs().sum()), 0)

    def test_score_ranks_quick_soft_landing_over_hover_and_crash(self):
        states = initial_states(80, 1000)
        hover = evaluate_policy(lambda state: expert_action(state, 0.), states)
        crash = evaluate_policy(lambda state: expert_action(state, CRASH_SINK), states)
        quick = evaluate_policy(lambda state: expert_action(state, 0.16), states)
        self.assertEqual(hover["landings"], 0)
        self.assertEqual(crash["landings"], 0)
        self.assertGreaterEqual(quick["landings"], 0.95)
        self.assertLess(hover["score"], quick["score"])
        self.assertLess(crash["score"], quick["score"])

    def test_side_engine_bias_is_a_closed_loop_crash(self):
        states = initial_states(80, 1000)
        held = evaluate_policy(lambda state: expert_action(state, SLOW, 0.), states)
        yanked = evaluate_policy(lambda state: expert_action(state, SLOW, -0.9), states)
        self.assertGreater(held["landings"], yanked["landings"])
        self.assertEqual(yanked["landings"], 0)
        self.assertLess(SOFT_SPEED_LIMIT, V21_SPEED_LIMIT)

    def test_zero_adv_weight_is_rejected_on_the_gan_arms(self):
        with self.assertRaises(ValueError) as caught:
            train_arm("fixed", steps=1, adv_weight=0)
        self.assertIn("adv_weight=0 leaves RpGAN and b_cap configured but not applied",
                      str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            require_live_adversary(0.25)
        self.assertIn("adv_weight stays 1", str(caught.exception))

    def test_gate_fails_v21_and_passes_throttle_up(self):
        result = run_gate()
        self.assertTrue(result["passed"], result)
        self.assertLessEqual(result["v21"]["landings"], 0.05)
        self.assertGreaterEqual(result["v21"]["crash_rate"], 0.40)
        self.assertEqual(result["v21"]["adv_weight"], 1)
        self.assertFalse(result["v21"]["accepted"])
        self.assertLessEqual(result["loose"]["landings"], 0.05)
        self.assertEqual(result["loose"]["adv_weight"], 1)
        self.assertGreaterEqual(result["baseline"]["landings"], 0.60)
        self.assertEqual(result["baseline"]["safe_fast_weight"], 0)
        self.assertEqual(result["baseline"]["adv_weight"], 1)
        self.assertGreaterEqual(result["fixed"]["landings"], 0.95)
        self.assertGreaterEqual(result["fixed"]["landings"], result["baseline"]["landings"] + 0.15)
        self.assertLessEqual(result["fixed"]["mean_steps"], result["baseline"]["mean_steps"] - 5)
        self.assertLessEqual(result["fixed"]["crash_rate"], 0.02)
        self.assertEqual(result["fixed"]["adv_weight"], 1)
        self.assertEqual(result["fixed"]["safe_fast_weight"], 1)
        self.assertGreater(result["fixed"]["gan_grad_abs"], 1)
        self.assertGreater(result["fixed"]["late_gan_grad"], 0.01)
        self.assertGreater(result["fixed"]["b_cap_applications"], 0)
        self.assertLessEqual(result["fixed"]["sink"], SOFT_SPEED_LIMIT)
        self.assertEqual(result["supervised"]["adv_weight"], 0)
        self.assertFalse(result["supervised"]["accepted"])
        self.assertGreaterEqual(result["supervised"]["landings"], 0.95)


if __name__ == "__main__":
    unittest.main()
