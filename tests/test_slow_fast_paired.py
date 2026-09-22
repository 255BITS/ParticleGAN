"""CPU gate: same-seed progress pairs beat stranger nearest and overspeed."""
import unittest

import torch

from lib.slow_fast_paired import (FAST_W, HELD_TEACHER_STEPS, NEAREST_DIST, SLOW_W,
    SlowPolicy, collect_pairs, crash_action, evaluate_policy, pair_is_kept, rank_key,
    require_live_adversary, run_gate, slow_action, train_arm)


class SlowFastPairedTests(unittest.TestCase):
    def test_fast_set_keeps_both_land_and_aligns_by_progress(self):
        self.assertFalse(pair_is_kept(True, False, 40, 12))
        self.assertFalse(pair_is_kept(False, True, 40, 12))
        self.assertFalse(pair_is_kept(True, True, 20, 20))
        self.assertTrue(pair_is_kept(True, True, 40, 12))
        table = collect_pairs()
        self.assertGreater(table["kept_starts"], 0)
        self.assertEqual(table["fast_failures_excluded"], 0)
        self.assertGreater(table["crash_episodes"], 0)
        self.assertTrue(torch.all(table["fast_steps"] < table["slow_steps"]))
        self.assertTrue(torch.all(table["start_id"] != table["unpaired_start_id"]))
        self.assertEqual(len(table["connected_target"]), len(table["state"]))
        self.assertEqual(len(table["progress"]), len(table["state"]))
        self.assertTrue(torch.all((table["progress"] >= 0) & (table["progress"] <= 1)))
        # The break update is crashy. The fast set does not require a 20/20 teacher.
        self.assertGreater(table["held_teacher_landings"], 0)
        self.assertLess(table["break_teacher_landings"], 0.90)
        self.assertGreater(table["teacher_curve"][-1]["crash_rate"], 0.40)
        self.assertLess(table["overspeed_teacher_steps"], table["held_teacher_steps"])
        self.assertEqual(table["teacher_curve"][HELD_TEACHER_STEPS - 1]["step"], HELD_TEACHER_STEPS)
        # Progress alignment is t/T on the same landing, and the edit is not zero.
        self.assertGreater(table["progress_edit"], 0.02)
        self.assertGreater(table["nearest_rows"], 1000)
        self.assertLessEqual(float(table["nearest_distance"].max()), NEAREST_DIST)
        # Crash rows are a different set from the both-land progress rows.
        self.assertGreater(len(table["crash_state"]), 100)
        self.assertNotEqual(len(table["crash_state"]), len(table["state"]))

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
            train_arm("connected", steps=1, adv_weight=0)
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

    def test_gate_connected_holds_and_faster_speed_does_not(self):
        result = run_gate()
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["winner"], "connected")
        self.assertEqual(result["connected"]["adv_weight"], 1)
        self.assertGreater(result["connected"]["b_cap_applications"], 0)
        self.assertGreater(result["connected"]["gan_grad_abs"], 0)
        self.assertGreaterEqual(result["connected"]["landings"], 0.98)
        self.assertGreaterEqual(result["connected"]["min_landings"], 0.95)
        self.assertLessEqual(result["connected"]["mean_steps"], 32)
        self.assertLessEqual(result["connected"]["mean_steps"], result["zero"]["mean_steps"] - 6)
        self.assertGreaterEqual(result["zero"]["landings"], 0.98)
        self.assertGreaterEqual(result["slow_only"]["landings"], 0.95)
        self.assertGreater(result["slow_only"]["mean_steps"], result["connected"]["mean_steps"])
        self.assertLess(result["crash_fast"]["landings"], 0.90)
        self.assertGreater(result["crash_fast"]["crash_rate"], 0.10)
        self.assertLessEqual(result["unpaired"]["landings"], 0.20)
        self.assertEqual(result["supervised"]["adv_weight"], 0)
        self.assertFalse(result["supervised"]["accepted"])
        self.assertEqual(result["supervised"]["rank_key"][0], -1.)
        self.assertLessEqual(result["stranger"]["landings"], 0.50)
        self.assertGreaterEqual(result["stranger"]["crash_rate"], 0.40)
        self.assertEqual(result["stranger"]["adv_weight"], 1.)
        self.assertEqual(result["stranger"]["rank_key"][0], -1.)
        self.assertLessEqual(result["overspeed"]["landings"], 0.90)
        self.assertEqual(result["overspeed"]["rank_key"][0], -1.)
        self.assertFalse(result["overspeed"]["accepted"])
        self.assertGreater(result["connected"]["rank_key"], result["slow_only"]["rank_key"])


if __name__ == "__main__":
    unittest.main()
