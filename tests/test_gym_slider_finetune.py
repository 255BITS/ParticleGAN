"""Action MSE stays out of the fine-tune graph; only E_control, G2, and R move."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.train_gym_slider_finetune import DEFAULTS, train
from lib.gym_slider_finetune import (action_decoded, build_error_critic, generator_objective,
    initialize_finetune, load_slider_finetune)
from lib.gym_slider_gan import error_loss
from lib.gym_control import build_expert_records, control_action
from particlegan import get_recipe
import tests.test_gym_control_training as control_tests


class SliderFinetuneTests(unittest.TestCase):
    def cfg(self, root, **overrides):
        return {**DEFAULTS, "steps": 4, "batch_size": 4, "checkpoints": [4], "log_interval": 1,
                "device": "cpu", "error_tokens": 2, "error_width": 8, "error_heads": 2,
                "checkpoint": str(root / "initial.pt"), "episodes": str(root / "episodes.json"),
                "out_dir": str(root / "action_error"), "live_log": str(root / "live.log"), **overrides}

    def test_action_mse_is_diagnostic_and_cap_is_lazy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            control_tests.ControlTrainingTests().fixture(root)
            cfg = self.cfg(root)
            bundle = initialize_finetune(cfg, "cpu")
            data = build_expert_records(root / "episodes.json")
            physical = torch.as_tensor(np.concatenate([data[k] for k in ("states", "actions", "next_states")], 1))
            real = bundle["scaler"](physical[:4])
            bundle["R"] = build_error_critic(cfg, bundle["scaler"](physical)[:, 8:10], "cpu")
            bundle["R"].requires_grad_(False)
            decoded = action_decoded(bundle, physical[:4, :8], torch.as_tensor(data["previous_actions"][:4]),
                                     torch.as_tensor(data["terrain"][:4]), real)
            loss, terms = generator_objective(bundle["R"], decoded, real, 1, torch.Generator().manual_seed(9))
            pure, _ = error_loss(bundle["R"], decoded, real, 1, torch.Generator().manual_seed(9))
            torch.testing.assert_close(loss, pure, atol=0, rtol=0)
            self.assertFalse(terms["action_mse"].requires_grad)
            grad = torch.autograd.grad(loss, decoded, retain_graph=True)[0]
            pure_grad = torch.autograd.grad(pure, decoded)[0]
            torch.testing.assert_close(grad, pure_grad, atol=0, rtol=0)
            self.assertEqual(float(grad[:, :8].abs().sum()), 0.)
            self.assertEqual(float(grad[:, 10:].abs().sum()), 0.)
            self.assertGreater(float(grad[:, 8:10].abs().sum()), 0.)
            cap = get_recipe("mog", z_dim=4, num_particles=8, total_steps=4, batch_size=4).make_gradient_penalty(
                arm="b_cap", lazy_k=4, coeff=1., kappa=1.)
            steep = lambda sample: 2 * sample[:, 0]
            noise = torch.randn(4, 2)
            skipped, _ = cap.penalty(steep, noise, noise, step=1)
            applied, _ = cap.penalty(steep, noise, noise, step=4)
            self.assertEqual(float(skipped), 0.)
            self.assertAlmostEqual(float(applied), 4., places=5)

    def test_train_updates_only_control_path_and_replays(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            states, _ = control_tests.ControlTrainingTests().fixture(root)
            cfg = self.cfg(root)
            start = initialize_finetune(cfg, "cpu")
            summary = train(cfg)
            self.assertEqual(summary["simulator_calls"], 0)
            self.assertEqual(summary["discriminator_optimizer_record_draws"], 0)
            self.assertEqual(summary["removed_losses"], ["standardized action MSE"])
            self.assertIn("marginal GAN", summary["absent_losses"])
            self.assertIn("transition-sample gradient penalty", summary["absent_losses"])
            bundle = load_slider_finetune(root / "action_error" / "final.pt")
            self.assertEqual(bundle["config"]["arm"], "slider_finetune")
            for key in ("E", "prior", "D"):
                for left, right in zip(bundle[key].parameters(), start[key].parameters()):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
            for index in (0, 2):
                for left, right in zip(bundle["G"].branches[index].parameters(), start["G"].branches[index].parameters()):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(
                bundle["G"].branches[1].parameters(), start["G"].branches[1].parameters())))
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(
                bundle["E_control"].parameters(), start["E_control"].parameters())))
            rows = [json.loads(line) for line in (root / "action_error" / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["step"] for row in rows], [1, 2, 3, 4])
            for row in rows:
                self.assertAlmostEqual(row["loss"], row["error_g"], places=6)
                self.assertIn("action_mse", row)
            self.assertEqual([row["error_cap"] for row in rows[:3]], [0., 0., 0.])
            self.assertTrue(np.isfinite(rows[3]["error_cap"]))
            text = (root / "live.log").read_text()
            self.assertIn("[action_error] START", text)
            self.assertIn("action_mse=", text)
            action, route = control_action(bundle, states[0], [-1., 0.], [-.5] * 11)
            restored = load_slider_finetune(root / "action_error" / "checkpoint_4.pt")
            replay, replay_route = control_action(restored, states[0], [-1., 0.], [-.5] * 11)
            self.assertTrue(torch.equal(torch.as_tensor(action), torch.as_tensor(replay)))
            self.assertEqual(route, replay_route)
            with self.assertRaises(FileExistsError):
                train(cfg)


if __name__ == "__main__":
    unittest.main()
