"""Deterministic CPU checks for the read-only affine motion sidecar."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native100_diagnostics import affine_motion_diagnostics  # noqa: E402


class AffineMotionTest(unittest.TestCase):
    def test_exact_decomposition_and_row_lag(self):
        z0 = np.array([[0., 0.], [1., 2.], [-2., 1.], [3., -1.]])
        z1 = z0 + np.array([[.25, -.5], [.5, 0.], [0., .125], [-.25, .75]])
        w0 = np.eye(2)
        w1 = np.array([[.5, .25], [-.125, 1.5]])
        b0 = np.array([.25, -.5])
        b1 = np.array([-.125, .75])
        old, new = (z0, w0, b0), (z1, w1, b1)
        prior = (z1 - z0) @ ((w0 + w1) / 2).T
        generator = ((z0 + z1) / 2) @ (w1 - w0).T + b1 - b0
        displacement = (z1 @ w1.T + b1) - (z0 @ w0.T + b0)
        np.testing.assert_allclose(prior + generator, displacement, rtol=0, atol=1e-14)

        initial, initial_delta = affine_motion_diagnostics(None, old, None, "grid100")
        self.assertIsNone(initial_delta)
        self.assertIsNone(initial["interval"])
        result, delta = affine_motion_diagnostics(old, new, displacement, "grid100")
        np.testing.assert_allclose(delta, displacement, rtol=0, atol=1e-14)
        sigma = .03
        rms = lambda x: np.sqrt(np.mean(np.sum(x * x, axis=1))) / sigma
        self.assertAlmostEqual(result["rms_prior_sigma"], rms(prior))
        self.assertAlmostEqual(result["rms_generator_sigma"], rms(generator))
        self.assertAlmostEqual(result["rms_total_sigma"], rms(displacement))
        self.assertAlmostEqual(result["rms_total_sigma"] ** 2,
                               result["rms_prior_sigma"] ** 2
                               + result["rms_generator_sigma"] ** 2
                               + 2 * result["mean_cross_dot_sigma2"])
        self.assertAlmostEqual(result["rms_total_sigma"] ** 2,
                               result["rms_mode_mean_sigma"] ** 2
                               + result["rms_within_mode_sigma"] ** 2)
        self.assertAlmostEqual(result["lag1_row_motion_cosine"], 1.)
        self.assertAlmostEqual(result["lag1_mean_row_cosine"], 1.)
        self.assertEqual(result["lag1_valid_row_fraction"], 1.)

        reversed_result, _ = affine_motion_diagnostics(old, new, -displacement, "grid100")
        self.assertAlmostEqual(reversed_result["lag1_row_motion_cosine"], -1.)
        no_lineage, _ = affine_motion_diagnostics(old, new, displacement, "grid100",
                                                  lag_lineage_valid=False)
        self.assertIsNone(no_lineage["lag1_row_motion_cosine"])
        self.assertIsNone(no_lineage["lag1_mean_row_cosine"])


if __name__ == "__main__":
    unittest.main()
