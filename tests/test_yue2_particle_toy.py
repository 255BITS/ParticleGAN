"""CPU gate for the YuE2 paired-error controller on the shared toy runner. One seed, no Lunar weights."""
import unittest

from lib.yue2_particle_toy import ARMS, FIXED_LAND_MIN, STEPS, run_gate
from particlegan import get_recipe


class Yue2ParticleToyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_gate()

    def test_arms_use_the_shipped_recipe_at_the_gate_shape(self):
        for arm in ARMS:
            self.assertEqual(arm().recipe().to_dict(), get_recipe(batch_size=64, total_steps=STEPS).to_dict())

    def test_roles(self):
        result = self.result
        self.assertEqual(result["supervised"]["adv_weight"], 0)
        self.assertFalse(result["supervised"]["accepted"])
        self.assertEqual(result["paired"]["adv_weight"], 1)
        self.assertTrue(result["paired"]["accepted"])

    def test_marginal_gan_keeps_the_flipped_sign(self):
        collapsed = self.result["collapsed"]
        self.assertEqual(collapsed["verdict"], "FAIL")
        self.assertLess(collapsed["alpha"], 0)

    def test_supervised_lands_yet_is_rejected(self):
        supervised = self.result["supervised"]
        self.assertGreaterEqual(supervised["landings"], FIXED_LAND_MIN)
        self.assertFalse(supervised["accepted"])

    def test_paired_gan_lands_with_the_expert_sign(self):
        paired = self.result["paired"]
        self.assertEqual(paired["verdict"], "PASS")
        self.assertGreaterEqual(paired["landings"], FIXED_LAND_MIN)
        self.assertGreater(paired["alpha"], 0.5)

    def test_gate_passes(self):
        self.assertTrue(self.result["passed"], self.result)


if __name__ == "__main__":
    unittest.main()
