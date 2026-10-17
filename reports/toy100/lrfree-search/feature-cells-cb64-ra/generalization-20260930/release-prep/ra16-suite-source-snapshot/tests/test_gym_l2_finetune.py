"""L2 finetune stays on action MSE and matches the imitation objective."""
import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.config import read_config
from experiments.train_gym_control import DEFAULTS as CONTROL_DEFAULTS, train as train_imitation
from experiments.evaluate_gym_l2_finetune import verify_checkpoint
from experiments.train_gym_l2_finetune import DEFAULTS, train, validate
from experiments.train_gym_transition import DEFAULTS as WORLD_DEFAULTS, build_models, sha256
from lib.gym_control import initialize_control, load_control_checkpoint, predict_control
from lib.gym_l2_finetune import LOCKED_SUPERVISION, assert_l2_supervision, l2_objective
from lib.gym_transition import GymTransitionScaler

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("discriminator_loss", "generator_loss", "adversarial_loss", "make_loss",
             "make_gradient_penalty", "make_prior_regularizer")


def call_names(path):
    tree = ast.parse(path.read_text())
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            found.add(func.id)
        elif isinstance(func, ast.Attribute):
            found.add(func.attr)
    return found


class L2FinetuneTests(unittest.TestCase):
    def fixture(self, root):
        rng = np.random.default_rng(81)
        states = rng.normal(size=(8, 8)).astype(np.float32)
        states[:, 6:] = rng.integers(0, 2, size=(8, 2))
        actions = rng.uniform(-1, 1, size=(8, 2)).astype(np.float32)
        next_states = states.copy()
        next_states[:, :2] += actions * .1
        episode = dict(split="train", behavior="heuristic", episode_id=4,
            states=states.tolist(), actions=actions.tolist(), next_states=next_states.tolist(),
            terrain=[-.5] * 11)
        (root / "episodes.json").write_text(json.dumps([episode, {**episode, "split": "test", "episode_id": 55}]))
        scaler = GymTransitionScaler.fit(np.concatenate([states, actions, next_states], axis=1))
        cfg = {**WORLD_DEFAULTS, "width": 8, "encoder_width": 8, "d_width": 8,
            "marginal_width": 8, "z_dim": 4, "num_particles": 8, "device": "cpu"}
        bundle = build_models(cfg, scaler, "cpu")
        checkpoint = dict(config=cfg, scaler=scaler.state_dict(), step=1000, validation={},
            provenance={"dataset": {"episodes.json": sha256(root / "episodes.json")}})
        checkpoint.update({key: None if bundle[key] is None else bundle[key].state_dict()
                           for key in ("G", "E", "prior", "D", "direct")})
        torch.save(checkpoint, root / "initial.pt")
        return states

    def cfg(self, root):
        return {**DEFAULTS, "steps": 2, "checkpoints": [1, 2], "batch_size": 4, "log_interval": 1,
            "device": "cpu", "checkpoint": str(root / "initial.pt"), "episodes": str(root / "episodes.json"),
            "out_dir": str(root / "l2"), "live_log": str(root / "live.log")}

    def test_source_has_no_adversarial_calls(self):
        for relative in ("experiments/train_gym_l2_finetune.py", "lib/gym_l2_finetune.py",
                         "experiments/evaluate_gym_l2_finetune.py"):
            names = call_names(ROOT / relative)
            if relative.endswith("train_gym_l2_finetune.py"):
                self.assertIn("l2_objective", names)
            self.assertTrue(set(FORBIDDEN).isdisjoint(names), relative)

    def test_config_rejects_adversarial_keys(self):
        validate({**DEFAULTS})
        loaded = {**DEFAULTS, **read_config(ROOT / "configs/gym/lunar_lander_finetune/l2.yaml")}
        validate(loaded)
        with self.assertRaises(ValueError):
            validate({**DEFAULTS, "adversarial_weight": 1.})
        with self.assertRaises(ValueError):
            validate({**DEFAULTS, "arm": "joint"})
        with self.assertRaises(ValueError):
            assert_l2_supervision({"arm": "l2", "supervision": {**LOCKED_SUPERVISION, "adversarial": True}},
                                  {"loss_terms": ["standardized_action_mse"], "adversarial_updates": 0,
                                   "discriminator_optimizer_record_draws": 0, "simulator_calls": 0})

    def test_objective_graph_ignores_frozen_modules(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            states = self.fixture(root)
            bundle = initialize_control(root / "initial.pt", "imitation", "cpu")
            for key in ("G", "E", "prior", "D", "E_control"):
                bundle[key].requires_grad_(True)
            physical = torch.tensor(np.concatenate([
                states, np.zeros((8, 2), np.float32), states], 1))
            loss, terms = l2_objective(bundle["scaler"], predict_control(
                bundle, physical[:4, :8], torch.zeros(4, 2), torch.full((4, 11), -.5))[0],
                bundle["scaler"].action(torch.zeros(4, 2)))
            loss.backward()
            self.assertEqual(set(terms), {"standardized_action_mse"})
            # The encoder reads the frozen prior, so prior can receive gradient if it is
            # trainable. Discriminators, the paired encoder, and G1/G3 must not.
            for module in (bundle["D"], bundle["E"], bundle["G"].branches[0], bundle["G"].branches[2]):
                self.assertTrue(all(p.grad is None for p in module.parameters()), module)
            self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in bundle["prior"].parameters()))
            self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in bundle["E_control"].parameters()))
            self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in bundle["G"].branches[1].parameters()))

    def test_training_matches_imitation_and_freezes_the_rest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            start = initialize_control(root / "initial.pt", "imitation", "cpu")
            summary = train(self.cfg(root))
            imitation_cfg = {**CONTROL_DEFAULTS, "arm": "imitation", "steps": 2, "checkpoints": [1, 2],
                "batch_size": 4, "log_interval": 1, "device": "cpu",
                "checkpoint": str(root / "initial.pt"), "episodes": str(root / "episodes.json"),
                "out_dir": str(root / "imitation"), "live_log": str(root / "imitation.log")}
            train_imitation(imitation_cfg)
            l2_rows = [json.loads(line) for line in (root / "l2" / "metrics.jsonl").read_text().splitlines()]
            imitation_rows = [json.loads(line) for line in (root / "imitation" / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["step"] for row in l2_rows], [1, 2])
            for l2_row, imitation_row in zip(l2_rows, imitation_rows):
                self.assertEqual(l2_row["loss"], l2_row["standardized_action_mse"])
                self.assertEqual(l2_row["adversarial_updates"], 0)
                self.assertNotIn("d_loss", l2_row)
                self.assertNotIn("g_loss", l2_row)
                self.assertAlmostEqual(l2_row["standardized_action_mse"], imitation_row["imitation_loss"], places=6)
            assert_l2_supervision(summary["config"], summary)
            bundle = load_control_checkpoint(root / "l2" / "final.pt")
            imitation = load_control_checkpoint(root / "imitation" / "final.pt")
            self.assertEqual(summary["simulator_calls"], 0)
            self.assertEqual(summary["discriminator_optimizer_record_draws"], 0)
            self.assertFalse((root / "l2" / "best.pt").exists())
            for key in ("E", "prior", "D"):
                for left, right in zip(bundle[key].parameters(), start[key].parameters()):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
            for index in (0, 2):
                for left, right in zip(bundle["G"].branches[index].parameters(), start["G"].branches[index].parameters()):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
            for key in ("E_control",):
                changed = any(not torch.equal(a, b) for a, b in zip(bundle[key].parameters(), start[key].parameters()))
                self.assertTrue(changed)
            g2_changed = any(not torch.equal(a, b) for a, b in zip(
                bundle["G"].branches[1].parameters(), start["G"].branches[1].parameters()))
            self.assertTrue(g2_changed)
            for left, right in zip(bundle["E_control"].parameters(), imitation["E_control"].parameters()):
                torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-5)
            for left, right in zip(bundle["G"].branches[1].parameters(), imitation["G"].branches[1].parameters()):
                torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-5)
            text = (root / "live.log").read_text()
            self.assertIn("OBJECTIVE standardized_action_mse only", text)
            self.assertIn("adversarial_updates=0", text)
            verify_checkpoint(root / "l2" / "final.pt", "cpu")
            summary_path = root / "l2" / "summary.json"
            tampered = json.loads(summary_path.read_text())
            tampered["adversarial_updates"] = 1
            summary_path.write_text(json.dumps(tampered))
            with self.assertRaises(ValueError):
                verify_checkpoint(root / "l2" / "final.pt", "cpu")
            with self.assertRaises(FileExistsError):
                train(self.cfg(root))


if __name__ == "__main__":
    unittest.main()
