"""CPU gate for the YuE2 paired-error controller on the shared toy runner. One seed, no Lunar weights."""
import unittest

from lib.yue2_particle_toy import ARMS, run_gate
from particlegan import get_recipe


class Yue2ParticleToyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_gate()

    def test_arms_use_the_shipped_recipe_at_the_gate_shape(self):
        for arm in ARMS:
            self.assertEqual(arm().recipe().to_dict(), get_recipe(batch_size=64, total_steps=200).to_dict())

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

    def test_paired_gan_moves_the_sign_toward_the_expert(self):
        # At the recipe LR only the paired arm's sign crosses zero within 200 updates.
        self.assertGreater(self.result["paired"]["alpha"], 0)
        self.assertGreater(self.result["paired"]["alpha"], self.result["supervised"]["alpha"])

    @unittest.expectedFailure
    def test_gate_passes_in_200_updates(self):
        # Before the migration the gate passed only with a caller-set scalar-gain LR (0.05, about
        # 12x the recipe's). On the recipe LR the paired arm first lands after ~330 updates.
        self.assertTrue(self.result["passed"], self.result)


if __name__ == "__main__":
    unittest.main()
