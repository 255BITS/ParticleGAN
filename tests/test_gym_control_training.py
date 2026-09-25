"""Check supervision alignment, matched initialization, frozen modules, and replay."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.train_gym_transition import DEFAULTS as WORLD_DEFAULTS, build_models, sha256
from experiments.train_gym_control import DEFAULTS, train
from lib.gym_control import (build_expert_records, initialize_control, load_control_checkpoint,
    control_action, predict_control)
from lib.gym_transition import GymTransitionScaler


class ControlTrainingTests(unittest.TestCase):
    def fixture(self, root):
        rng = np.random.default_rng(81)
        s = rng.normal(size=(8, 8)).astype(np.float32)
        s[:, 6:] = rng.integers(0, 2, size=(8, 2))
        a = rng.uniform(-1, 1, size=(8, 2)).astype(np.float32)
        next_s = s.copy()
        next_s[:, :2] += a * .1
        episode = dict(split="train", behavior="heuristic", episode_id=4,
            states=s.tolist(), actions=a.tolist(), next_states=next_s.tolist(), terrain=[-.5]*11)
        ignored = {**episode, "split": "test", "episode_id": 55}
        (root / "episodes.json").write_text(json.dumps([episode, ignored]))
        scaler = GymTransitionScaler.fit(np.concatenate([s, a, next_s], axis=1))
        cfg = {**WORLD_DEFAULTS, "width": 8, "encoder_width": 8, "d_width": 8,
            "marginal_width": 8, "z_dim": 4, "num_particles": 8, "device": "cpu"}
        bundle = build_models(cfg, scaler, "cpu")
        checkpoint = dict(config=cfg, scaler=scaler.state_dict(), step=1000, validation={},
            provenance={"dataset": {"episodes.json": sha256(root / "episodes.json")}})
        checkpoint.update({key: None if bundle[key] is None else bundle[key].state_dict()
                           for key in ("G", "E", "prior", "D", "direct")})
        torch.save(checkpoint, root / "initial.pt")
        return s, a

    def test_previous_action_alignment_and_split(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, actions = self.fixture(root)
            data = build_expert_records(root / "episodes.json")
            self.assertEqual(len(data["states"]), 8)
            np.testing.assert_array_equal(data["previous_actions"][0], [-1., 0.])
            np.testing.assert_array_equal(data["previous_actions"][1:], actions[:-1])
            np.testing.assert_array_equal(data["actions"], actions)
            np.testing.assert_array_equal(data["episode_ids"], [4]*8)

    def test_matched_start_and_both_update_scopes_with_checkpoint_replay(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            states, _ = self.fixture(root)
            start = [initialize_control(root / "initial.pt", arm) for arm in ("imitation", "joint")]
            for key in ("G", "E", "prior", "D", "E_control"):
                for a, b in zip(start[0][key].parameters(), start[1][key].parameters()):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertIsNot(start[0]["E"], start[0]["E_control"])
            for arm in ("imitation", "joint"):
                cfg = {**DEFAULTS, "arm": arm, "steps": 2, "checkpoints": [1, 2],
                    "batch_size": 4, "log_interval": 1, "device": "cpu",
                    "checkpoint": str(root / "initial.pt"), "episodes": str(root / "episodes.json"),
                    "out_dir": str(root / arm), "live_log": str(root / "live.log")}
                summary = train(cfg)
                bundle = load_control_checkpoint(root / arm / "final.pt")
                self.assertEqual(summary["simulator_calls"], 0)
                self.assertEqual(summary["generator_optimizer_record_draws"], 8)
                self.assertEqual(summary["discriminator_optimizer_record_draws"], 8 if arm == "joint" else 0)
                self.assertFalse((root / arm / "best.pt").exists())
                initial = start[0]
                for key in ("E", "prior", "D"):
                    changed = any(not torch.equal(a, b) for a,b in zip(bundle[key].parameters(), initial[key].parameters()))
                    self.assertEqual(changed, arm == "joint", key)
                for i in range(3):
                    changed = any(not torch.equal(a,b) for a,b in zip(bundle["G"].branches[i].parameters(), initial["G"].branches[i].parameters()))
                    self.assertEqual(changed, arm == "joint" or i == 1)
                self.assertTrue(any(not torch.equal(a,b) for a,b in zip(bundle["E_control"].parameters(), initial["E_control"].parameters())))
                for key, tensor in bundle["scaler"].state_dict().items():
                    torch.testing.assert_close(tensor, initial["scaler"].state_dict()[key], rtol=0, atol=0)
                act, route = control_action(bundle, states[0], [-1.,0.], [-.5]*11)
                reload = load_control_checkpoint(root / arm / "checkpoint_2.pt")
                replay, replay_route = control_action(reload, states[0], [-1.,0.], [-.5]*11)
                np.testing.assert_array_equal(act, replay)
                self.assertEqual(route, replay_route)
                self.assertTrue(np.isfinite(act).all())
                self.assertTrue((np.abs(act) <= 1).all())
                with self.assertRaises(FileExistsError):
                    train(cfg)


if __name__ == "__main__":
    unittest.main()
