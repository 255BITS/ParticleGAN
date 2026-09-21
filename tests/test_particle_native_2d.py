"""The 2D particle gate must collapse on the old recipe and pass on the latent-joint recipe."""
import unittest

from experiments.toy_particle_native_2d import run_gate


class ParticleNative2DTests(unittest.TestCase):
    def test_observation_recipe_collapses_and_latent_joint_passes(self):
        result = run_gate()
        self.assertGreaterEqual(result["current"]["ema"], 1.0)
        self.assertLessEqual(result["fixed"]["ema"], 0.18)
        self.assertLess(result["fixed"]["ema"], result["current"]["ema"] * 0.25)
        self.assertEqual(result["adversarial_weight"], 1.)
        self.assertEqual(result["l2_weight"], 0.)
        self.assertEqual(result["b_cap_coeff"], 1.)
        self.assertFalse(result["supervised_only"])
        self.assertTrue(result["ok"], result)


if __name__ == "__main__":
    unittest.main()
