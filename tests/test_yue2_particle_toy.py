"""CPU gate for the YuE2 paired-error controller. One seed, no Lunar weights."""
import unittest

from lib.yue2_particle_toy import run_gate


class Yue2ParticleToyTests(unittest.TestCase):
    def test_gate_rejects_supervised_only_and_passes_paired_gan(self):
        result = run_gate()
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["supervised"]["adv_weight"], 0)
        self.assertFalse(result["supervised"]["accepted"])
        self.assertGreaterEqual(result["supervised"]["landings"], 0.95)
        self.assertEqual(result["paired"]["adv_weight"], 1)
        self.assertGreater(result["paired"]["b_cap_applications"], 0)
        self.assertGreater(result["paired"]["gan_grad_abs"], 0)
        self.assertLess(result["collapsed"]["alpha"], 0)
        self.assertGreater(result["collapsed"]["b_cap_applications"], 0)


if __name__ == "__main__":
    unittest.main()
