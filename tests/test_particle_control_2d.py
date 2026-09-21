"""CPU gate for the 2D particle-control toy. No Lunar checkpoint and no GPU."""
import unittest

from examples.particle_control_2d import run_gate


class ParticleControl2DTests(unittest.TestCase):
    def test_gate(self):
        result = run_gate()
        self.assertEqual(result["verdict"], "PASS", result)
        self.assertGreaterEqual(result["baseline"]["action_mse"], 0.30)
        self.assertLessEqual(result["baseline"]["landings"], 0.25)
        self.assertLessEqual(result["fix"]["action_mse"], 0.12)
        self.assertGreaterEqual(result["fix"]["landings"], 0.80)
        self.assertEqual(result["adv_weight"], 0.0)
        self.assertEqual(result["anchor_weight"], 0.1)
        self.assertEqual(result["gan_mode"], "rp")
        self.assertEqual(result["reg_arm"], "b_cap")


if __name__ == "__main__":
    unittest.main()
