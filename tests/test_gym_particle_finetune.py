"""Arm A finetune: RpGAN and sample b_cap replace paired L2, without a slider critic."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.train_gym_transition import DEFAULTS as WORLD_DEFAULTS, build_models, sha256
from experiments.train_gym_particle_finetune import DEFAULTS, train
from lib.gym_control import build_expert_records, control_action
from lib.gym_particle_finetune import (FAKE_PATHS, MODULE_KEYS, control_decode, diagnostic_l2,
    glue_control_loss, load_particle_checkpoint, particle_game, require_classic_particle_gan,
    transition_batch)
from lib.gym_state_control import training_recipe


class ParticleFinetuneTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

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
        ignored = {**episode, "split": "test", "episode_id": 55}
        episodes = root / "episodes.json"
        episodes.write_text(json.dumps([episode, ignored]))
        from lib.gym_transition import GymTransitionScaler
        scaler = GymTransitionScaler.fit(np.concatenate([states, actions, next_states], 1))
        cfg = {**WORLD_DEFAULTS, "width": 8, "encoder_width": 8, "d_width": 8, "marginal_width": 8,
               "z_dim": 4, "num_particles": 8, "device": "cpu"}
        bundle = build_models(cfg, scaler, "cpu")
        checkpoint = dict(config=cfg, scaler=scaler.state_dict(), step=1000, validation={},
            provenance={"dataset": {"episodes.json": sha256(episodes)}})
        checkpoint.update({key: None if bundle[key] is None else bundle[key].state_dict()
                           for key in ("G", "E", "prior", "D", "direct")})
        torch.save(checkpoint, root / "initial.pt")
        return states, actions

    def test_rejects_auxiliary_l2_and_active_adversary(self):
        with self.assertRaises(ValueError):
            train({**DEFAULTS, "imitation_weight": 1.})
        with self.assertRaises(ValueError):
            train({**DEFAULTS, "adv_weight": 1.})

    def test_control_path_ignores_current_action_and_diagnostics_have_no_grad(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            from lib.gym_particle_finetune import initialize_particle_finetune
            bundle = initialize_particle_finetune(root / "initial.pt", "cpu")
            records = build_expert_records(root / "episodes.json")
            batch = {key: torch.from_numpy(records[key][:4]) for key in
                     ("states", "previous_actions", "actions", "next_states", "terrain")}
            previous = batch["previous_actions"].clone().requires_grad_(True)
            decoded, _ = control_decode(bundle, batch["states"], previous, batch["terrain"])
            self.assertGreater(float(torch.autograd.grad(decoded[:, 8:10].sum(), previous)[0].abs().sum()), 0.)
            replay, _ = control_decode(bundle, batch["states"], batch["previous_actions"], batch["terrain"])
            batch["actions"].fill_(999.)
            batch["next_states"].fill_(999.)
            changed, _ = control_decode(bundle, batch["states"], batch["previous_actions"], batch["terrain"])
            torch.testing.assert_close(replay, changed, atol=0, rtol=0)
            real, _, control = transition_batch(
                bundle, batch["states"], batch["previous_actions"], records_actions(records),
                torch.from_numpy(records["next_states"][:4]), batch["terrain"],
                torch.Generator().manual_seed(1), torch.Generator().manual_seed(2), True)
            diag = diagnostic_l2(control, real)
            self.assertFalse(any(value.requires_grad for value in diag.values()))

    def test_glue_loss_trains_only_the_action_head(self):
        def grad_sum(module):
            return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            from lib.gym_particle_finetune import initialize_particle_finetune
            bundle = initialize_particle_finetune(root / "initial.pt", "cpu")
            records = build_expert_records(root / "episodes.json")
            columns = [torch.from_numpy(records[key][:4]) for key in
                       ("states", "previous_actions", "actions", "next_states", "terrain")]
            loss, terms, diag = glue_control_loss(bundle, *columns)
            self.assertGreater(float(loss.detach()), 0.)
            self.assertIn("action_anchor", terms)
            self.assertFalse(any(value.requires_grad for value in diag.values()))
            loss.backward()
            self.assertGreater(grad_sum(bundle["G"].branches[1]), 0.)
            self.assertEqual(grad_sum(bundle["G"].branches[0]), 0.)
            self.assertEqual(grad_sum(bundle["G"].branches[2]), 0.)
            for name in ("E", "prior", "D", "E_control"):
                self.assertEqual(grad_sum(bundle[name]), 0., name)

    def test_rpgan_bcap_reaches_encoders_generators_and_critics(self):
        def has_grad(module):
            return any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in module.parameters())

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            from lib.gym_particle_finetune import initialize_particle_finetune
            bundle = initialize_particle_finetune(root / "initial.pt", "cpu")
            for key in MODULE_KEYS:
                bundle[key].requires_grad_(True)
            records = build_expert_records(root / "episodes.json")
            columns = [torch.from_numpy(records[key][:4]) for key in
                       ("states", "previous_actions", "actions", "next_states", "terrain")]
            recipe = training_recipe({**bundle["world_config"], "steps": 2, "batch_size": 4})
            gan, reg = recipe.make_loss(), recipe.make_gradient_penalty()
            require_classic_particle_gan(gan, reg)
            expectations = dict(control=("E_control",), prior=(), encoded=("E",), composed=("E",))
            seen = set()
            for path, encoders in expectations.items():
                for key in ("G", "E", "prior", "D", "E_control"):
                    bundle[key].zero_grad(set_to_none=True)
                bundle["D"].requires_grad_(False)
                real, fakes, _ = transition_batch(bundle, *columns, torch.Generator().manual_seed(3),
                                                  torch.Generator().manual_seed(4), True)
                seen.update(fakes)
                loss, _ = particle_game(bundle["D"], real, {path: fakes[path]}, columns[-1], gan)
                loss.backward()
                self.assertTrue(all(has_grad(branch) for branch in bundle["G"].branches), path)
                self.assertTrue(has_grad(bundle["prior"]), path)
                self.assertFalse(has_grad(bundle["D"]), path)
                for name in ("E", "E_control"):
                    self.assertEqual(has_grad(bundle[name]), name in encoders, path + name)
            self.assertEqual(seen, set(FAKE_PATHS))
            for key in ("G", "E", "prior", "D", "E_control"):
                bundle[key].zero_grad(set_to_none=True)
            bundle["D"].requires_grad_(True)
            real, fakes, _ = transition_batch(bundle, *columns, torch.Generator().manual_seed(3),
                                              torch.Generator().manual_seed(4), True)
            loss, terms = particle_game(bundle["D"], real, fakes, columns[-1], gan, reg=reg, step=1,
                                        rngs={role: torch.Generator().manual_seed(5) for role in bundle["D"].roles()})
            loss.backward()
            self.assertTrue(all(has_grad(critic) for critic in bundle["D"].critics.values()))
            self.assertTrue(all(name + "_penalty" in terms for name in ("joint", "action", "state", "next_state")))
            self.assertTrue(all(not has_grad(bundle[key]) for key in ("G", "E", "prior", "E_control")))

    def test_cpu_smoke_writes_flushed_logs_and_replays(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            states, _ = self.fixture(root)
            cfg = {**DEFAULTS, "steps": 2, "checkpoints": [1, 2], "batch_size": 4, "log_interval": 1,
                   "device": "cpu", "checkpoint": str(root / "initial.pt"),
                   "episodes": str(root / "episodes.json"), "out_dir": str(root / "particle"),
                   "live_log": str(root / "live.log")}
            summary = train(cfg)
            self.assertEqual(summary["simulator_calls"], 0)
            self.assertEqual(summary["l2_aux_weight"], 0.)
            self.assertEqual(summary["adversarial_updates"], 0)
            self.assertEqual(summary["real_draws"], 8)
            self.assertIn(summary["selected_step"], (1, 2))
            self.assertTrue((root / "particle" / "selected.pt").is_file())
            text = (root / "live.log").read_text()
            self.assertIn("REMOVED L2", text)
            self.assertIn("sample-point b_cap", text)
            self.assertIn("supervised_only=true", text)
            self.assertIn("l2_aux=0", text)
            self.assertTrue((root / "particle" / "live.log").is_symlink())
            self.assertIn("REMOVED L2", (root / "particle" / "log.txt").read_text())
            recipe = json.loads((root / "particle" / "recipe.json").read_text())
            self.assertEqual(recipe["gan_mode"], "rp")
            self.assertEqual(recipe["loss_type"], "logistic")
            self.assertEqual(recipe["reg_arm"], "b_cap")
            self.assertEqual(recipe["reg_method"], "autograd")
            row = json.loads((root / "particle" / "metrics.jsonl").read_text().splitlines()[-1])
            self.assertEqual(row["l2_aux_weight"], 0.)
            self.assertEqual(row["adv_weight"], 0.)
            self.assertIn("action_mse", row)
            self.assertIn("action_anchor", row)
            bundle = load_particle_checkpoint(root / "particle" / "final.pt")
            replay = load_particle_checkpoint(root / "particle" / "checkpoint_2.pt")
            action, route = control_action(bundle, states[0], [-1., 0.], [-.5] * 11)
            action2, route2 = control_action(replay, states[0], [-1., 0.], [-.5] * 11)
            np.testing.assert_array_equal(action, action2)
            self.assertEqual(route, route2)
            self.assertTrue(np.isfinite(action).all() and (np.abs(action) <= 1).all())
            fresh = initialize_again(root)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(
                bundle["E_control"].parameters(), fresh["E_control"].parameters())))
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(
                bundle["G"].branches[1].parameters(), fresh["G"].branches[1].parameters())))
            with self.assertRaises(FileExistsError):
                train(cfg)


def records_actions(records):
    return torch.from_numpy(records["actions"][:4])


def initialize_again(root):
    from lib.gym_particle_finetune import initialize_particle_finetune
    return initialize_particle_finetune(root / "initial.pt", "cpu")


if __name__ == "__main__":
    unittest.main()
