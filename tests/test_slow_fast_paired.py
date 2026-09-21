"""CPU gate: a frozen lander plus a bounded residual beats the Lunar overwrite."""
import unittest

import torch

from lib.slow_fast_paired import (ANCHORED_SCALE, FAST_W, NEAREST_DIST, SLOW_W,
    AnchoredPolicy, SlowPolicy, collect_pairs, crash_action, evaluate_policy,
    fast_action, pair_is_kept, rank_key, require_live_adversary, run_gate,
    slow_action, train_arm)


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
        self.assertGreater(table["nearest_rows"], 0)
        self.assertLessEqual(float(table["nearest_distance"].max()), NEAREST_DIST)
        # Nearest-state targets are a different episode's action, not the same-state law.
        gap = (table["nearest_target"] - fast_action(table["nearest_state"])).abs().mean()
        self.assertGreater(float(gap), 0.02)

    def test_student_starts_at_the_slow_law(self):
        policy = SlowPolicy()
        state = torch.randn(8, 4)
        torch.testing.assert_close(policy.w, SLOW_W)
        self.assertFalse(torch.allclose(policy.w, FAST_W))
        torch.testing.assert_close(policy(state), slow_action(state))
        anchored = AnchoredPolicy()
        torch.testing.assert_close(anchored(state), slow_action(state))
        with torch.no_grad():
            edit = ANCHORED_SCALE * anchored.delta(state).tanh()
        self.assertLessEqual(float(edit.abs().max()), ANCHORED_SCALE)

    def test_zero_adv_weight_is_rejected_on_the_gan_arm(self):
        with self.assertRaises(ValueError) as caught:
            train_arm("paired", steps=1, adv_weight=0)
        self.assertIn("adv_weight=0 leaves RpGAN and b_cap configured but not applied",
                      str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            train_arm("anchored", steps=1, adv_weight=0)
        self.assertIn("adv_weight=0 leaves RpGAN and b_cap configured but not applied",
                      str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            require_live_adversary(0.25)
        self.assertIn("adv_weight stays 1", str(caught.exception))

    def test_rank_key_drops_mse_crashes_and_a_landing_dip(self):
        fast_success = dict(adv_weight=1., b_cap_applications=10, landings=1.0,
                            crash_rate=0.0, mean_steps=20.0, min_landings=1.0)
        slower = dict(adv_weight=1., b_cap_applications=10, landings=1.0,
                      crash_rate=0.0, mean_steps=34.0, min_landings=1.0)
        dipped = dict(adv_weight=1., b_cap_applications=10, landings=1.0,
                      crash_rate=0.0, mean_steps=12.0, min_landings=0.2)
        crash = dict(adv_weight=1., b_cap_applications=10, landings=0.0,
                     crash_rate=1.0, mean_steps=10.0, min_landings=0.0)
        mse = dict(adv_weight=0., b_cap_applications=0, landings=1.0,
                   crash_rate=0.0, mean_steps=12.0, min_landings=1.0)
        frozen = dict(adv_weight=None, b_cap_applications=0, landings=1.0,
                      crash_rate=0.0, mean_steps=37.0, min_landings=1.0)
        self.assertGreater(rank_key(fast_success), rank_key(slower))
        self.assertEqual(rank_key(mse)[0], -1.)
        self.assertEqual(rank_key(frozen)[0], -1.)
        self.assertEqual(rank_key(crash)[0], -1.)
        self.assertEqual(rank_key(dipped)[0], -1.)

    def test_crash_oracle_is_not_a_success(self):
        crashed = evaluate_policy(crash_action)
        self.assertEqual(crashed["landings"], 0)
        self.assertGreater(crashed["crash_rate"], 0.95)
        self.assertLess(crashed["mean_contact_steps"], 30)

    def test_gate_anchored_wins_and_overwrite_loses_the_pad(self):
        result = run_gate()
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["winner"], "anchored")
        self.assertEqual(result["anchored"]["adv_weight"], 1)
        self.assertGreater(result["anchored"]["b_cap_applications"], 0)
        self.assertGreater(result["anchored"]["gan_grad_abs"], 0)
        self.assertGreaterEqual(result["anchored"]["landings"], 0.98)
        self.assertGreaterEqual(result["anchored"]["min_landings"], 0.95)
        self.assertLessEqual(result["anchored"]["mean_steps"], 26)
        self.assertLessEqual(result["anchored"]["mean_steps"], result["zero"]["mean_steps"] - 8)
        self.assertGreaterEqual(result["zero"]["landings"], 0.98)
        self.assertGreaterEqual(result["slow_only"]["landings"], 0.95)
        self.assertGreater(result["slow_only"]["mean_steps"], result["anchored"]["mean_steps"])
        self.assertLessEqual(result["crash_fast"]["landings"], 0.20)
        self.assertLessEqual(result["unpaired"]["landings"], 0.20)
        self.assertEqual(result["supervised"]["adv_weight"], 0)
        self.assertFalse(result["supervised"]["accepted"])
        self.assertEqual(result["supervised"]["rank_key"][0], -1.)
        self.assertLessEqual(result["overwrite"]["landings"], 0.25)
        self.assertEqual(result["overwrite"]["adv_weight"], 1.)
        self.assertEqual(result["overwrite"]["rank_key"][0], -1.)
        self.assertGreater(result["anchored"]["rank_key"], result["slow_only"]["rank_key"])


if __name__ == "__main__":
    unittest.main()
