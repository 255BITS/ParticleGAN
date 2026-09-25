"""Arm A finetune: paired-error RpGAN and sample b_cap, with adv_weight locked at 1."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.train_gym_transition import DEFAULTS as WORLD_DEFAULTS, build_models, sha256
from experiments.train_gym_particle_finetune import DEFAULTS, train
from lib.gym_control import build_expert_records, control_action
from lib.gym_particle_finetune import (FAKE_PATHS, control_decode, diagnostic_l2,
    load_particle_checkpoint, particle_game, require_classic_particle_gan, transition_batch)
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

    def test_rejects_auxiliary_l2(self):
        cfg = {**DEFAULTS, "imitation_weight": 1.}
        with self.assertRaises(ValueError):
            train(cfg)

    def test_rejects_inactive_adversary(self):
        inactive = {**DEFAULTS, "adv_weight": 0.}
        with self.assertRaises(ValueError) as caught:
            train(inactive)
        self.assertIn("adv_weight=0 leaves RpGAN and b_cap configured but not applied",
                      str(caught.exception))
        partial = {**DEFAULTS, "adv_weight": 0.1}
        with self.assertRaises(ValueError) as caught:
            train(partial)
        self.assertIn("adv_weight stays 1", str(caught.exception))

    def test_controller_step_is_live_rpgan_and_b_cap_hits_the_edit_critic(self):
        def has_grad(module):
            return any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in module.parameters())

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            from lib.gym_particle_finetune import (build_edit_critic, configure_control_scope,
                controller_objective, discriminator_objective, edit_cap,
                initialize_particle_finetune, normalized_g2_action)
            bundle = configure_control_scope(initialize_particle_finetune(root / "initial.pt", "cpu"))
            records = build_expert_records(root / "episodes.json")
            states = torch.from_numpy(records["states"])
            previous = torch.from_numpy(records["previous_actions"])
            actions = torch.from_numpy(records["actions"])
            terrain = torch.from_numpy(records["terrain"])
            with torch.no_grad():
                neutrals = normalized_g2_action(bundle, states, previous, terrain)
            targets = bundle["scaler"].action(actions)
            critic = build_edit_critic(targets, neutrals,
                                       {**DEFAULTS, "error_tokens": 4, "error_width": 8, "error_heads": 2})
            self.assertEqual(critic.normalization, "paired_edit_per_coordinate_std_median_rms_gain")
            predicted = normalized_g2_action(bundle, states[:4], previous[:4], terrain[:4])
            target = targets[:4]
            reg = edit_cap()
            loss_d, terms = discriminator_objective(
                critic, predicted, target, 4, torch.Generator().manual_seed(7), reg, 8)
            self.assertEqual(terms["b_cap_applied"], 1.)
            loss_d.backward()
            self.assertTrue(has_grad(critic))
            self.assertTrue(all(not has_grad(bundle[key]) for key in ("G", "E", "prior", "D", "E_control")))
            _, skipped = discriminator_objective(
                critic, predicted.detach(), target, 2, torch.Generator().manual_seed(8), reg, 8)
            self.assertEqual(skipped["b_cap_applied"], 0.)
            for module in (bundle["G"], bundle["E"], bundle["prior"], bundle["D"], bundle["E_control"], critic):
                module.zero_grad(set_to_none=True)
            loss_g, g_terms = controller_objective(
                critic, predicted, target, 1, torch.Generator().manual_seed(9), 8, 1.)
            self.assertEqual(g_terms["adv_weight"], 1.)
            loss_g.backward()
            self.assertTrue(has_grad(bundle["E_control"]))
            self.assertTrue(has_grad(bundle["G"].branches[1]))
            self.assertFalse(has_grad(bundle["G"].branches[0]))
            self.assertFalse(has_grad(bundle["G"].branches[2]))
            self.assertFalse(has_grad(bundle["E"]))
            self.assertFalse(has_grad(bundle["prior"]))
            self.assertFalse(has_grad(bundle["D"]))
            with self.assertRaises(ValueError) as caught:
                controller_objective(critic, predicted, target, 1, torch.Generator().manual_seed(9), 8, 0.)
            self.assertIn("adv_weight=0 leaves RpGAN and b_cap configured but not applied",
                          str(caught.exception))

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

    def test_rpgan_bcap_reaches_encoders_generators_and_critics(self):
        def has_grad(module):
            return any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in module.parameters())

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            from lib.gym_particle_finetune import initialize_particle_finetune
            bundle = initialize_particle_finetune(root / "initial.pt", "cpu")
            records = build_expert_records(root / "episodes.json")
            columns = [torch.from_numpy(records[key][:4]) for key in
                       ("states", "previous_actions", "actions", "next_states", "terrain")]
            recipe = training_recipe({**bundle["world_config"], "steps": 2, "batch_size": 4})
            gan, reg = recipe.make_loss(), recipe.make_gradient_penalty(arm="b_cap", coeff=1., kappa=1.)
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
            self.assertEqual(summary["real_draws"], 16)
            text = (root / "live.log").read_text()
            self.assertIn("REMOVED L2", text)
            self.assertIn("sample-point b_cap", text)
            self.assertIn("l2_aux=0", text)
            self.assertIn("adv_weight=1", text)
            self.assertIn("Not supervised_only", text)
            self.assertIn("SAFE-FAST off", text)
            self.assertEqual(summary["safe_fast_weight"], 0.)
            self.assertIn("edit-normalized", text)
            self.assertNotIn("latent-joint", text)
            self.assertTrue((root / "particle" / "live.log").is_symlink())
            self.assertIn("REMOVED L2", (root / "particle" / "log.txt").read_text())
            recipe = json.loads((root / "particle" / "recipe.json").read_text())
            self.assertEqual(recipe["gan_mode"], "rp")
            self.assertEqual(recipe["loss_type"], "logistic")
            self.assertEqual(recipe["reg_arm"], "b_cap")
            self.assertEqual(recipe["reg_method"], "autograd")
            self.assertEqual(recipe["adv_weight"], 1.)
            self.assertEqual(recipe["l2_aux_weight"], 0.)
            self.assertEqual(recipe["critic"], "gmix_t8_w48_l1")
            self.assertEqual(recipe["train_scope"], "control")
            self.assertEqual(recipe["normalization"], "paired_edit_per_coordinate_std_median_rms_gain")
            self.assertEqual(recipe["safe_fast_weight"], 0.)
            self.assertEqual(recipe["safe_fast_horizon"], 0)
            self.assertIn("paired-error", summary["initialization"])
            self.assertEqual(summary["adv_weight"], 1.)
            self.assertEqual(summary["b_cap_applications"], 0)
            row = json.loads((root / "particle" / "metrics.jsonl").read_text().splitlines()[-1])
            self.assertEqual(row["l2_aux_weight"], 0.)
            self.assertEqual(row["safe_fast_weight"], 0.)
            self.assertEqual(row["safe_fast"], 0.)
            self.assertIn("action_mse", row)
            bundle = load_particle_checkpoint(root / "particle" / "final.pt")
            replay = load_particle_checkpoint(root / "particle" / "checkpoint_2.pt")
            action, route = control_action(bundle, states[0], [-1., 0.], [-.5] * 11)
            action2, route2 = control_action(replay, states[0], [-1., 0.], [-.5] * 11)
            np.testing.assert_array_equal(action, action2)
            self.assertEqual(route, route2)
            self.assertTrue(np.isfinite(action).all() and (np.abs(action) <= 1).all())
            fresh = initialize_again(root)
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(
                bundle["E_control"].parameters(), fresh["E_control"].parameters())))
            for trained, started in zip(bundle["G"].branches[0].parameters(), fresh["G"].branches[0].parameters()):
                torch.testing.assert_close(trained, started)
            for trained, started in zip(bundle["G"].branches[2].parameters(), fresh["G"].branches[2].parameters()):
                torch.testing.assert_close(trained, started)
            with self.assertRaises(FileExistsError):
                train(cfg)

    def test_safe_fast_config_is_optional_and_adv_weight_stays_live(self):
        from experiments.config import read_config
        root = Path(__file__).resolve().parents[1]
        proven = {**DEFAULTS, **read_config(root / "configs/gym/lunar_lander_particle_finetune/particle.yaml")}
        from experiments.train_gym_particle_finetune import validate
        validate(proven)
        self.assertEqual(proven["adv_weight"], 1.)
        self.assertEqual(proven["safe_fast_weight"], 0.)
        self.assertEqual(proven["imitation_weight"], 0.)
        armed = {**DEFAULTS, **read_config(
            root / "configs/gym/lunar_lander_particle_finetune/particle_safe_fast.yaml")}
        validate(armed)
        self.assertEqual(armed["adv_weight"], 1.)
        self.assertEqual(armed["safe_fast_weight"], 1.)
        self.assertEqual(armed["safe_fast_horizon"], 48)
        self.assertEqual(armed["safe_fast_time_weight"], 0.15)
        self.assertNotEqual(armed["out_dir"], proven["out_dir"])
        inactive = {**armed, "adv_weight": 0.}
        with self.assertRaises(ValueError) as caught:
            validate(inactive)
        self.assertIn("adv_weight=0 leaves RpGAN and b_cap configured but not applied",
                      str(caught.exception))
        missing_horizon = {**armed, "safe_fast_horizon": 0}
        with self.assertRaises(ValueError) as caught:
            validate(missing_horizon)
        self.assertIn("safe_fast_horizon", str(caught.exception))

    def test_safe_fast_weight_zero_does_not_attach_and_positive_weight_trains(self):
        def has_grad(module):
            return any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in module.parameters())

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            from lib.gym_particle_finetune import (build_edit_critic, configure_control_scope,
                controller_objective, initialize_particle_finetune, normalized_g2_action)
            from lib.safe_fast_landing import gym_shaping_cost
            bundle = configure_control_scope(initialize_particle_finetune(root / "initial.pt", "cpu"))
            records = build_expert_records(root / "episodes.json")
            states = torch.from_numpy(records["states"])
            previous = torch.from_numpy(records["previous_actions"])
            actions = torch.from_numpy(records["actions"])
            terrain = torch.from_numpy(records["terrain"])
            with torch.no_grad():
                neutrals = normalized_g2_action(bundle, states, previous, terrain)
            critic = build_edit_critic(bundle["scaler"].action(actions), neutrals,
                                       {**DEFAULTS, "error_tokens": 4, "error_width": 8, "error_heads": 2})
            predicted = normalized_g2_action(bundle, states[:4], previous[:4], terrain[:4])
            sentinel = torch.zeros((), requires_grad=True)
            loss, terms = controller_objective(
                critic, predicted, bundle["scaler"].action(actions[:4]), 1,
                torch.Generator().manual_seed(9), 8, 1., sentinel, 0.)
            self.assertEqual(terms["safe_fast_weight"], 0.)
            loss.backward()
            self.assertIsNone(sentinel.grad)
            for module in (bundle["E_control"], bundle["G"], critic):
                module.zero_grad(set_to_none=True)

            def predict_physical(full, prev, terr):
                return bundle["scaler"].inverse_action(normalized_g2_action(bundle, full, prev, terr))

            cost = gym_shaping_cost(states[:4], previous[:4], terrain[:4], predict_physical, 2,
                                    0.15, 4., 2., 0.62, 0.35)
            self.assertTrue(cost.requires_grad)
            cost.backward()
            self.assertTrue(has_grad(bundle["E_control"]))
            self.assertTrue(has_grad(bundle["G"].branches[1]))
            self.assertFalse(has_grad(bundle["G"].branches[0]))
            self.assertFalse(has_grad(bundle["E"]))

            cfg = {**DEFAULTS, "steps": 1, "checkpoints": [1], "batch_size": 4, "log_interval": 1,
                   "device": "cpu", "checkpoint": str(root / "initial.pt"),
                   "episodes": str(root / "episodes.json"), "out_dir": str(root / "safe"),
                   "live_log": str(root / "safe_live.log"), "safe_fast_weight": 1.,
                   "safe_fast_time_weight": 0.15, "safe_fast_crash_weight": 4.,
                   "safe_fast_success_bonus": 2., "safe_fast_speed_limit": 0.62,
                   "safe_fast_pad_half": 0.35, "safe_fast_horizon": 2}
            summary = train(cfg)
            text = (root / "safe_live.log").read_text()
            self.assertIn("SAFE-FAST on", text)
            self.assertIn("adv_weight=1", text)
            self.assertIn("Not adv_weight=0", text)
            self.assertEqual(summary["adv_weight"], 1.)
            self.assertEqual(summary["safe_fast_weight"], 1.)
            self.assertEqual(summary["l2_aux_weight"], 0.)
            row = json.loads((root / "safe" / "metrics.jsonl").read_text().splitlines()[-1])
            self.assertEqual(row["adv_weight"], 1.)
            self.assertEqual(row["safe_fast_weight"], 1.)
            self.assertTrue(torch.isfinite(torch.tensor(row["safe_fast"])))

    def test_locked_shared_config_uses_stamp_builders(self):
        from unittest.mock import patch

        from experiments.config import read_config
        from experiments.train_gym_particle_finetune import validate
        from lib.gym_particle_finetune import EDIT_CAP_EVERY, adversarial_game
        from lib.vendor.concept_slider_core.reference import rp_d_loss, rp_g_loss
        from particlegan.locked_shared import LOCKED_SHARED, make_b_cap, make_gan_loss

        root = Path(__file__).resolve().parents[1]
        raw = read_config(root / "configs/gym/lunar_lander_particle_finetune/locked_shared.yaml")
        for pin in ("lazy_k", "loss_type", "gan_mode", "reg_coeff", "reg_kappa", "reg_arm"):
            self.assertNotIn(pin, raw)
        cfg = {**DEFAULTS, **raw}
        validate(cfg)
        self.assertEqual(cfg["adv_posture"], "locked_shared")
        self.assertEqual(cfg["safe_fast_weight"], 0.)
        self.assertEqual(cfg["adv_weight"], 1.)
        self.assertNotEqual(cfg["out_dir"], DEFAULTS["out_dir"])
        self.assertNotEqual(cfg["live_log"], DEFAULTS["live_log"])
        yue = {**DEFAULTS, **read_config(
            root / "configs/gym/lunar_lander_particle_finetune/particle.yaml")}
        validate(yue)
        self.assertEqual(yue["adv_posture"], "yue2")
        self.assertNotIn("adv_posture", read_config(
            root / "configs/gym/lunar_lander_particle_finetune/particle.yaml"))

        with patch("lib.gym_particle_finetune.make_gan_loss", wraps=make_gan_loss) as loss, \
             patch("lib.gym_particle_finetune.make_b_cap", wraps=make_b_cap) as cap:
            gan, reg = adversarial_game("locked_shared")
        self.assertEqual(loss.call_count, 1)
        self.assertEqual(cap.call_count, 1)
        self.assertIs(loss.call_args.args[0], LOCKED_SHARED)
        self.assertIs(cap.call_args.args[0], LOCKED_SHARED)
        self.assertEqual(gan.loss_type, LOCKED_SHARED.loss_type)
        self.assertEqual(gan.mode, LOCKED_SHARED.gan_mode)
        self.assertEqual(reg.lazy_k, LOCKED_SHARED.lazy_k)
        self.assertEqual(reg.lazy_k, 1)
        self.assertNotEqual(reg.lazy_k, EDIT_CAP_EVERY)
        self.assertEqual((reg.arm, reg.coeff, reg.kappa, reg.norm, reg.method, reg.target_anneal),
                         ("b_cap", 1.0, 1.0, "l2", "autograd", "none"))
        real = torch.tensor([0.2, -0.4])
        fake = torch.tensor([0.5, 0.1])
        torch.testing.assert_close(gan.d_loss(real, fake), rp_d_loss(real, fake))
        torch.testing.assert_close(gan.g_loss(fake, real), rp_g_loss(real, fake))

        with patch("lib.gym_particle_finetune.make_gan_loss", wraps=make_gan_loss) as loss, \
             patch("lib.gym_particle_finetune.make_b_cap", wraps=make_b_cap) as cap:
            yue_gan, yue_reg = adversarial_game("yue2")
        self.assertEqual(loss.call_count, 0)
        self.assertEqual(cap.call_count, 0)
        self.assertIsNone(yue_gan)
        self.assertEqual(yue_reg.lazy_k, EDIT_CAP_EVERY)
        with self.assertRaises(ValueError):
            adversarial_game("music")
        with self.assertRaises(ValueError) as caught:
            validate({**cfg, "adv_posture": "music"})
        self.assertIn("adv_posture", str(caught.exception))
        armed = {**cfg, "safe_fast_weight": 1., "safe_fast_time_weight": 0.15,
                 "safe_fast_crash_weight": 4., "safe_fast_success_bonus": 2.,
                 "safe_fast_speed_limit": 0.62, "safe_fast_pad_half": 0.35,
                 "safe_fast_horizon": 48}
        with self.assertRaises(ValueError) as caught:
            validate(armed)
        self.assertIn("locked_shared does not add safe-fast", str(caught.exception))

    def test_locked_shared_objective_calls_the_gan_object(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            from lib.gym_particle_finetune import (adversarial_game, build_edit_critic,
                configure_control_scope, controller_objective, discriminator_objective,
                initialize_particle_finetune, normalized_g2_action)
            bundle = configure_control_scope(initialize_particle_finetune(root / "initial.pt", "cpu"))
            records = build_expert_records(root / "episodes.json")
            states = torch.from_numpy(records["states"])
            previous = torch.from_numpy(records["previous_actions"])
            actions = torch.from_numpy(records["actions"])
            terrain = torch.from_numpy(records["terrain"])
            with torch.no_grad():
                neutrals = normalized_g2_action(bundle, states, previous, terrain)
            critic = build_edit_critic(bundle["scaler"].action(actions), neutrals,
                                       {**DEFAULTS, "error_tokens": 4, "error_width": 8, "error_heads": 2})
            predicted = normalized_g2_action(bundle, states[:4], previous[:4], terrain[:4])
            target = bundle["scaler"].action(actions[:4])
            _, reg = adversarial_game("locked_shared")

            class Mark:
                def __init__(self):
                    self.d = self.g = 0

                def d_loss(self, real, fake):
                    self.d += 1
                    return (real * 0).sum() + real.new_tensor(1.25)

                def g_loss(self, fake, real):
                    self.g += 1
                    return (fake * 0).sum() + fake.new_tensor(0.5)

            mark = Mark()
            _, terms = discriminator_objective(
                critic, predicted, target, 1, torch.Generator().manual_seed(7), reg, 8, mark)
            self.assertEqual(mark.d, 1)
            self.assertEqual(mark.g, 0)
            self.assertAlmostEqual(float(terms["error_d"]), 1.25)
            self.assertEqual(terms["b_cap_applied"], 1.)
            _, g_terms = controller_objective(
                critic, predicted, target, 1, torch.Generator().manual_seed(9), 8, 1., gan=mark)
            self.assertEqual(mark.g, 1)
            self.assertAlmostEqual(float(g_terms["error_g"]), 0.5)
            _, skipped = discriminator_objective(
                critic, predicted.detach(), target, 2, torch.Generator().manual_seed(8), reg, 8, mark)
            self.assertEqual(skipped["b_cap_applied"], 1.)

    def test_locked_shared_cpu_smoke_applies_stamp_every_step(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            from unittest.mock import patch

            import experiments.train_gym_particle_finetune as trainer
            from particlegan.gan_loss import GANLoss
            from particlegan.locked_shared import LOCKED_SHARED
            seen = {}
            real_d, real_g = trainer.discriminator_objective, trainer.controller_objective

            def watch_d(*args, **kwargs):
                seen["d_gan"] = kwargs.get("gan", args[-1] if args else None)
                return real_d(*args, **kwargs)

            def watch_g(*args, **kwargs):
                seen["g_gan"] = kwargs.get("gan", args[-1] if args else None)
                return real_g(*args, **kwargs)

            cfg = {**DEFAULTS, "adv_posture": "locked_shared", "steps": 2, "checkpoints": [1, 2],
                   "batch_size": 4, "log_interval": 1, "device": "cpu",
                   "error_tokens": 4, "error_width": 8, "error_heads": 2,
                   "checkpoint": str(root / "initial.pt"), "episodes": str(root / "episodes.json"),
                   "out_dir": str(root / "locked"), "live_log": str(root / "locked_live.log")}
            with patch.object(trainer, "discriminator_objective", watch_d), \
                 patch.object(trainer, "controller_objective", watch_g):
                summary = train(cfg)
            self.assertIsInstance(seen["d_gan"], GANLoss)
            self.assertIs(seen["d_gan"], seen["g_gan"])
            self.assertEqual(seen["d_gan"].loss_type, LOCKED_SHARED.loss_type)
            self.assertEqual(seen["d_gan"].mode, LOCKED_SHARED.gan_mode)
            self.assertEqual(summary["b_cap_applications"], 2)
            self.assertEqual(summary["l2_aux_weight"], 0.)
            self.assertEqual(summary["adv_weight"], 1.)
            self.assertEqual(summary["simulator_calls"], 0)
            self.assertIn("no landing", summary["selection"])
            text = (root / "locked_live.log").read_text()
            self.assertIn("STAMP particlegan.locked_shared", text)
            self.assertIn("lazy_k=1", text)
            self.assertIn("make_gan_loss+make_b_cap", text)
            self.assertIn(f"Not YuE2 EDIT_CAP_EVERY=4", text)
            self.assertNotIn("POSTURE yue2", text)
            recipe = json.loads((root / "locked" / "recipe.json").read_text())
            self.assertEqual(recipe["adv_posture"], "locked_shared")
            self.assertEqual(recipe["stamp"], "particlegan.locked_shared.LOCKED_SHARED")
            self.assertEqual(recipe["loss_type"], LOCKED_SHARED.loss_type)
            self.assertEqual(recipe["gan_mode"], LOCKED_SHARED.gan_mode)
            self.assertEqual(recipe["reg_every"], LOCKED_SHARED.lazy_k)
            self.assertEqual(recipe["reg_arm"], LOCKED_SHARED.reg_arm)
            self.assertEqual(recipe["reg_coeff"], LOCKED_SHARED.reg_coeff)
            self.assertEqual(recipe["reg_kappa"], LOCKED_SHARED.reg_kappa)
            self.assertEqual(recipe["fm_weight"], 0.)
            self.assertFalse(recipe["cover_applied"])
            self.assertFalse(recipe["particles_applied"])
            self.assertEqual(recipe["critic_pin"], "host")
            self.assertEqual(recipe["pairing"], "live")
            provenance = json.loads((root / "locked" / "provenance.json").read_text())
            self.assertIn("particlegan.locked_shared.make_b_cap", provenance["gradient_penalty"])
            self.assertIn("every 1 steps", provenance["gradient_penalty"])
            rows = [json.loads(line) for line in (root / "locked" / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["b_cap_applied"] for row in rows], [1., 1.])


def records_actions(records):
    return torch.from_numpy(records["actions"][:4])


def initialize_again(root):
    from lib.gym_particle_finetune import initialize_particle_finetune
    return initialize_particle_finetune(root / "initial.pt", "cpu")


if __name__ == "__main__":
    unittest.main()
