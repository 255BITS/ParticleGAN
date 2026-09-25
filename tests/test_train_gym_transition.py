"""Finite-data trainer correctness and checkpoint replay, without Box2D dependency."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.train_gym_transition import (DEFAULTS, train, load_checkpoint,
    predict, load_split, build_models, parameter_count)
from lib.gym_transition import GymTransitionScaler


class GymTrainerTest(unittest.TestCase):
    @staticmethod
    def dataset(path):
        rng = np.random.default_rng(811)
        path.mkdir()
        for split in ("train", "validation", "test"):
            states = rng.normal(size=(24, 8)).astype(np.float32)
            states[:, 6:] = rng.integers(0, 2, size=(24, 2))
            actions = rng.uniform(-1, 1, size=(24, 2)).astype(np.float32)
            next_states = states.copy()
            next_states[:, :2] += .1 * actions
            terrain = rng.uniform(-.8, -.2, size=(24, 11)).astype(np.float32)
            np.savez(path / f"{split}.npz", states=states, actions=actions,
                     next_states=next_states, terrain=terrain)

    def test_all_arms_train_and_replay_ema_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.dataset(root / "data")
            for arm in ("direct", "reconstruction", "adversarial"):
                with self.subTest(arm=arm):
                    cfg = {**DEFAULTS, "arm": arm, "width": 8, "encoder_width": 8,
                           "d_width": 8, "marginal_width": 8, "z_dim": 4,
                           "num_particles": 8, "steps": 2, "checkpoints": [1, 2],
                           "batch_size": 4, "device": "cpu", "log_interval": 1,
                           "data_dir": str(root / "data"), "out_dir": str(root / arm),
                           "live_log": str(root / "live.log")}
                    summary = train(cfg)
                    bundle = load_checkpoint(root / arm / "best.pt")
                    real, terrain = load_split(root / "data", "validation")
                    prediction = predict(bundle, real[:, :8], real[:, 8:10], terrain)
                    reloaded = load_checkpoint(root / arm / f"checkpoint_{summary['best_step']}.pt")
                    replayed = predict(reloaded, real[:, :8], real[:, 8:10], terrain)
                    torch.testing.assert_close(prediction, replayed, rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(prediction).all())
                    self.assertTrue(((prediction[:, 6:] >= 0) & (prediction[:, 6:] <= 1)).all())
                    scaler = bundle["scaler"]
                    mse = float(((prediction[:, :6]-real[:, 10:16])/scaler.state_scale).square().mean())
                    self.assertAlmostEqual(mse, summary["best_validation_mse"], places=6)
                    self.assertEqual(summary["real_draws"], 16 if arm == "adversarial" else 8)
                    self.assertEqual(summary["unique_training_triples"], 24)
                    rows = [json.loads(line) for line in (root / arm / "metrics.jsonl").read_text().splitlines()]
                    self.assertEqual([r["step"] for r in rows], [1, 2])
                    with self.assertRaises(FileExistsError):
                        train(cfg)

    def test_normalization_uses_only_training_and_capacity_is_matched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.dataset(root / "data")
            real, _ = load_split(root / "data", "train")
            scaler = GymTransitionScaler.fit(real)
            expected = torch.cat([real[:, :6], real[:, 10:16]]).mean(0)
            torch.testing.assert_close(scaler.state_mean, expected)
            models = [build_models({**DEFAULTS, "arm": arm}, scaler, "cpu")
                      for arm in ("reconstruction", "adversarial", "direct")]
            for key in ("G", "E", "prior"):
                self.assertEqual(parameter_count(models[0][key]), parameter_count(models[1][key]))
                for a, b in zip(models[0][key].parameters(), models[1][key].parameters()):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            direct = parameter_count(models[2]["direct"])
            self.assertLess(abs(direct-models[0]["inference_target"])/models[0]["inference_target"], .01)


if __name__ == "__main__":
    unittest.main()
