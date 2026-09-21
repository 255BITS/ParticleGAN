"""CPU gate for the 2D particle-collapse toy. No Lunar checkpoint and no GPU.

The gate asserts that pure RpGAN + sample b_cap collapses. It does not assert
a repair. Adversarial weight stays 1; weight 0 is not this arm.

    python -u examples/particle_control_2d.py
    python -m unittest tests.test_particle_control_2d
"""
import unittest

from examples.particle_control_2d import BASELINE_ADV_WEIGHT, run_gate


class ParticleControl2DTests(unittest.TestCase):
    def test_baseline_collapses(self):
        result = run_gate()
        self.assertEqual(result["verdict"], "COLLAPSE", result)
        self.assertGreaterEqual(result["baseline"]["action_mse"], 0.30)
        self.assertLessEqual(result["baseline"]["landings"], 0.25)
        self.assertEqual(result["adv_weight"], 1.0)
        self.assertEqual(BASELINE_ADV_WEIGHT, 1.0)
        self.assertNotEqual(result["adv_weight"], 0.0)
        self.assertEqual(result["gan_mode"], "rp")
        self.assertEqual(result["reg_arm"], "b_cap")
        self.assertEqual(result["prior_lr_mult"], 100.0)
        self.assertLess(result["parent"]["action_mse"], result["baseline"]["action_mse"])


if __name__ == "__main__":
    unittest.main()
