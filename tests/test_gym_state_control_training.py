"""Gradient boundaries, target exclusion, matched scratch starts and checkpoint replay."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.train_gym_state_control import DEFAULTS, train
from experiments.train_gym_transition import sha256
from lib.gym_state_control import (build_state_models, parameter_hashes, predict_state_control,
    task_losses, state_control_action, load_state_control_checkpoint)
from lib.gym_transition import GymTransitionScaler


class StateControlTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(19)
        self.s = rng.normal(size=(8, 8)).astype(np.float32)
        self.s[:, 6:] = rng.integers(0, 2, size=(8, 2))
        self.a = rng.uniform(-1, 1, size=(8, 2)).astype(np.float32)
        self.n = self.s.copy()
        self.n[:, :2] += self.a * .1
        self.c = np.full((8, 11), -.5, dtype=np.float32)
        self.scaler = GymTransitionScaler.fit(np.concatenate([self.s, self.a, self.n], 1))
        self.cfg = {**DEFAULTS, "z_dim": 4, "num_particles": 8, "width": 8,
                    "encoder_width": 8, "device": "cpu", "batch_size": 4}
        self.batch = {k: torch.from_numpy(v) for k,v in
                      dict(states=self.s, actions=self.a, next_states=self.n, terrain=self.c).items()}

    def test_matched_initialization_and_input_target_separation(self):
        bundles = [build_state_models({**self.cfg, "arm": arm}, self.scaler)
                   for arm in ("probes", "auxiliary")]
        self.assertEqual(parameter_hashes(bundles[0]), parameter_hashes(bundles[1]))
        first, encoding = predict_state_control(bundles[0], self.s, self.c)
        _, changed_terms, changed_encoding = task_losses(bundles[0], **{
            **self.batch, "actions": -self.batch["actions"], "next_states": self.batch["states"]})
        torch.testing.assert_close(encoding.codes, changed_encoding.codes, atol=0, rtol=0)
        second, _ = predict_state_control(bundles[0], self.s, self.c)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        self.assertEqual(bundles[0]["E"].features[0].in_features, 19)
        with self.assertRaises(ValueError):
            bundles[0]["E"](torch.cat([self.batch["states"], self.batch["actions"]], 1),
                            self.batch["terrain"], bundles[0]["prior"])

    def test_gradient_scopes(self):
        def receives_gradient(module):
            return any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters())
        for arm in ("probes", "auxiliary"):
            bundle = build_state_models({**self.cfg, "arm": arm}, self.scaler)
            modules = dict(E=bundle["E"], prior=bundle["prior"],
                           **{f"G{i+1}": b for i,b in enumerate(bundle["G"].branches)})
            _, terms, _ = task_losses(bundle, **self.batch)
            terms["action_loss"].backward()
            self.assertEqual({k for k,m in modules.items() if receives_gradient(m)}, {"E", "prior", "G2"})
            for module in modules.values():
                module.zero_grad(set_to_none=True)
            _, terms, _ = task_losses(bundle, **self.batch)
            (terms["state_loss"] + terms["next_loss"]).backward()
            expected = {"G1", "G3"} | ({"E", "prior"} if arm == "auxiliary" else set())
            self.assertEqual({k for k,m in modules.items() if receives_gradient(m)}, expected)

    def test_matched_training_draws_and_checkpoint_replay(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            episode = dict(split="train", behavior="heuristic", episode_id=4,
                states=self.s.tolist(), actions=self.a.tolist(), next_states=self.n.tolist(), terrain=self.c[0].tolist())
            (root / "episodes.json").write_text(json.dumps([episode, {**episode, "split": "test", "episode_id": 55}]))
            # There are deliberately NO pretrained model weights in this file.
            torch.save(dict(scaler=self.scaler.state_dict(), provenance={"dataset": {
                "episodes.json": sha256(root / "episodes.json")}}), root / "scaler.pt")
            summaries = []
            for arm in ("probes", "auxiliary"):
                cfg = {**self.cfg, "arm": arm, "steps": 2, "checkpoints": [1, 2], "log_interval": 1,
                    "preprocessing_checkpoint": str(root / "scaler.pt"), "episodes": str(root / "episodes.json"),
                    "out_dir": str(root / arm), "live_log": str(root / "live.log")}
                summary = train(cfg)
                summaries.append(summary)
                self.assertEqual(summary["simulator_calls"], 0)
                self.assertEqual(summary["unique_training_records"], 8)
                self.assertEqual(summary["real_draws"], 8)
                self.assertEqual(summary["recipe"]["prior_lr_mult"], 2.)
                self.assertEqual(summary["auxiliary_gradient_scope"], "G1/G3 only" if arm == "probes" else "G1/G3 plus E/prior")
                self.assertNotIn("previous_actions", np.load(root / arm / "expert_records.npz").files)
                bundle = load_state_control_checkpoint(root / arm / "final.pt")
                start = build_state_models(cfg, self.scaler)
                self.assertNotEqual(parameter_hashes(bundle)["prior"], parameter_hashes(start)["prior"])
                act, route = state_control_action(bundle, self.s[0], self.c[0])
                reloaded = load_state_control_checkpoint(root / arm / "checkpoint_2.pt")
                replay, replay_route = state_control_action(reloaded, self.s[0], self.c[0])
                np.testing.assert_array_equal(act, replay)
                self.assertEqual(route, replay_route)
                self.assertTrue(np.isfinite(act).all() and (np.abs(act) <= 1).all())
                # Live action ignores even invalid diagnostic head parameters.
                with torch.no_grad():
                    for i in (0, 2):
                        for p in bundle["G"].branches[i].parameters():
                            p.fill_(float("nan"))
                np.testing.assert_array_equal(act, state_control_action(bundle, self.s[0], self.c[0])[0])
                with self.assertRaises(FileExistsError):
                    train(cfg)
            self.assertEqual(summaries[0]["provenance"]["initial_parameters"], summaries[1]["provenance"]["initial_parameters"])
            self.assertEqual(summaries[0]["data_draw_sha256"], summaries[1]["data_draw_sha256"])
            self.assertEqual(summaries[0]["provenance"]["expert_data"]["arrays"], summaries[1]["provenance"]["expert_data"]["arrays"])


if __name__ == "__main__":
    unittest.main()
