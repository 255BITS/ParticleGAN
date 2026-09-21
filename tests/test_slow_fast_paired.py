"""CPU gate: matched slow→fast RpGAN beats crash, unpaired, and MSE. One seed."""
import unittest

import torch

from lib.slow_fast_paired import (FAST_W, SLOW_W, SlowPolicy, collect_pairs, crash_action,
    evaluate_policy, fast_action, pair_is_kept, rank_key, require_live_adversary,
    run_gate, slow_action, train_arm)


class SlowFastPairedTests(unittest.TestCase):
    def test_fast_set_drops_crashes_and_pairs_share_the_start(self):
        self.assertFalse(pair_is_kept(True, False, 40, 12))
        self.assertFalse(pair_is_kept(False, True, 40, 12))
        self.assertFalse(pair_is_kept(True, True, 20, 20))
        self.assertTrue(pair_is_kept(True, True, 40, 12))
        table = collect_pairs()
        self.assertGreater(table["kept_starts"], 0)
        self.assertEqual(table["fast_failures_excluded"], 0)
        self.assertTrue(torch.all(table["fast_steps"] < table["slow_steps"]))
        self.assertTrue(torch.all(table["start_id"] != table["unpaired_start_id"]))
        self.assertGreater(float((table["crash_target"] - table["target"]).abs().mean()), 0.05)
        # The stored fast label is the fast law at the slow trajectory's state.
        torch.testing.assert_close(table["target"], fast_action(table["state"]))
        torch.testing.assert_close(table["neutral"], slow_action(table["state"]))
        torch.testing.assert_close(table["crash_target"], crash_action(table["state"]))

    def test_student_starts_at_the_slow_law(self):
        policy = SlowPolicy()
        state = torch.randn(8, 4)
        torch.testing.assert_close(policy.w, SLOW_W)
        self.assertFalse(torch.allclose(policy.w, FAST_W))
        torch.testing.assert_close(policy(state), slow_action(state))

    def test_zero_adv_weight_is_rejected_on_the_gan_arm(self):
        with self.assertRaises(ValueError) as caught:
            train_arm("paired", steps=1, adv_weight=0)
        self.assertIn("adv_weight=0 leaves RpGAN and b_cap configured but not applied",
                      str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            require_live_adversary(0.25)
        self.assertIn("adv_weight stays 1", str(caught.exception))

    def test_rank_key_drops_mse_and_a_frozen_init(self):
        fast_success = dict(adv_weight=1., b_cap_applications=10, score=0.8)
        crash = dict(adv_weight=1., b_cap_applications=10, score=-0.5)
        mse = dict(adv_weight=0., b_cap_applications=0, score=0.95)
        frozen = dict(adv_weight=None, b_cap_applications=0, score=0.7)
        self.assertGreater(rank_key(fast_success), rank_key(crash))
        self.assertLess(rank_key(mse), 0)
        self.assertLess(rank_key(frozen), 0)

    def test_crash_oracle_is_not_a_success(self):
        crashed = evaluate_policy(crash_action)
        self.assertEqual(crashed["landings"], 0)
        self.assertGreater(crashed["crash_rate"], 0.95)
        self.assertLess(crashed["mean_contact_steps"], 30)

    def test_gate_paired_arm_wins_and_mse_is_rejected(self):
        result = run_gate()
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["winner"], "paired")
        self.assertEqual(result["paired"]["adv_weight"], 1)
        self.assertGreater(result["paired"]["b_cap_applications"], 0)
        self.assertGreater(result["paired"]["gan_grad_abs"], 0)
        self.assertGreaterEqual(result["paired"]["landings"], 0.90)
        self.assertLessEqual(result["paired"]["mean_steps"], 18)
        self.assertLessEqual(result["paired"]["mean_steps"], result["zero"]["mean_steps"] - 12)
        self.assertGreaterEqual(result["zero"]["landings"], 0.98)
        self.assertGreaterEqual(result["slow_only"]["landings"], 0.95)
        self.assertGreater(result["slow_only"]["mean_steps"], result["paired"]["mean_steps"])
        self.assertLessEqual(result["crash_fast"]["landings"], 0.20)
        self.assertLessEqual(result["unpaired"]["landings"], 0.20)
        self.assertEqual(result["supervised"]["adv_weight"], 0)
        self.assertFalse(result["supervised"]["accepted"])
        self.assertLess(result["supervised"]["rank_key"], 0)
        self.assertGreater(result["paired"]["rank_key"], result["slow_only"]["rank_key"])
        self.assertGreater(result["paired"]["rank_key"], result["crash_fast"]["rank_key"])
        self.assertGreater(result["paired"]["rank_key"], result["unpaired"]["rank_key"])


if __name__ == "__main__":
    unittest.main()
