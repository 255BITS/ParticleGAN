"""Sampler, frozen circle protocol, and paired-error controller scope."""
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.train_circle_transition import DEFAULTS, train, validate
from lib.circle_transition import (CENTER_BINS, CENTER_LIMIT, RADIUS_BINS, RADIUS_RANGE, SPEED_BINS,
    SPEED_RANGE, CircleEncoder, assert_aligned, evaluate_panel, evaluation_panel, expert_policy,
    expert_transition, in_split, parameter_cell, reversed_policy, rollout, sample_rows, zero_policy)
from lib.gym_particle_finetune import (build_edit_critic, configure_control_scope, controller_objective,
    discriminator_objective, edit_cap, paired_noise, require_live_adversary)
from lib.vendor.concept_slider_core.reference import rp_g_loss
from particlegan import MoGParticlePrior


class SamplerTests(unittest.TestCase):
    def test_analytic_successor_and_alignment(self):
        position = torch.tensor([[1., 0.], [1.2, 0.]])
        center = torch.zeros(2, 2)
        radius = torch.ones(2)
        omega = torch.tensor([torch.pi / 2, 0.])
        action, nxt = expert_transition(position, center, radius, omega)
        torch.testing.assert_close(action[0], torch.tensor([-1., 1.]), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(nxt[0], torch.tensor([0., 1.]), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(action[1], torch.tensor([-0.05, 0.]), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(nxt[1], torch.tensor([1.15, 0.]), atol=1e-5, rtol=1e-5)
        batch = sample_rows(32, torch.Generator().manual_seed(7), "train", "mixed")
        perm = torch.randperm(len(batch))
        shuffled = batch.index(perm)
        assert_aligned(shuffled)
        torch.testing.assert_close(shuffled.position, batch.position[perm])
        torch.testing.assert_close(shuffled.action, batch.action[perm])
        torch.testing.assert_close(shuffled.next_position, batch.next_position[perm])
        gap = (batch.position[1:] - batch.next_position[:-1]).norm(dim=1).median()
        self.assertGreater(float(gap), 0.05)
        self.assertEqual(tuple(batch.context().shape), (32, 4))
        self.assertFalse(hasattr(batch, "phase"))
        self.assertEqual(int((batch.angular_step > 0).sum()), 16)

    def test_parameter_cells_are_a_partition(self):
        def point(index, bins, low, high):
            return low + (index + 0.5) * (high - low) / bins
        counts = {"train": 0, "val": 0, "test": 0}
        for ix in range(CENTER_BINS):
            for iy in range(CENTER_BINS):
                for ir in range(RADIUS_BINS):
                    for isp in range(SPEED_BINS):
                        center = torch.tensor([[point(ix, CENTER_BINS, -CENTER_LIMIT, CENTER_LIMIT),
                                                point(iy, CENTER_BINS, -CENTER_LIMIT, CENTER_LIMIT)]])
                        radius = torch.tensor([point(ir, RADIUS_BINS, *RADIUS_RANGE)])
                        omega = torch.tensor([point(isp, SPEED_BINS, *SPEED_RANGE)])
                        hits = [name for name in counts if bool(in_split(center, radius, omega, name))]
                        self.assertEqual(hits, [hits[0]] if hits else [])
                        self.assertEqual(len(hits), 1)
                        counts[hits[0]] += 1
                        self.assertEqual(int(parameter_cell(center, radius, -omega)),
                                         int(parameter_cell(center, radius, omega)))
        self.assertEqual(sum(counts.values()), CENTER_BINS * CENTER_BINS * RADIUS_BINS * SPEED_BINS)
        self.assertTrue(all(count > 0 for count in counts.values()))
        for split in counts:
            batch = sample_rows(16, torch.Generator().manual_seed(11), split, "on")
            self.assertTrue(bool(in_split(batch.center, batch.radius, batch.angular_step, split).all()))

    def test_rows_are_independent_in_a_rollout(self):
        panel = evaluation_panel("test", "main", 4)
        alone = rollout(expert_policy, panel.index(torch.tensor([0])), 8)
        other = rollout(expert_policy, panel.index(torch.tensor([1])), 8)
        both = rollout(expert_policy, panel.index(torch.tensor([0, 1])), 8)
        torch.testing.assert_close(both[:, 0], alone[:, 0])
        torch.testing.assert_close(both[:, 1], other[:, 0])


class ProtocolTests(unittest.TestCase):
    def test_controls_match_the_circle_protocol(self):
        panel = evaluation_panel("test", "main", 128)
        recovery = evaluation_panel("test", "recovery", 128)
        expert = evaluate_panel(expert_policy, panel)[1024]
        zero = evaluate_panel(zero_policy, panel)[1024]
        reversed_row = evaluate_panel(reversed_policy, panel)[1024]
        expert_recovery = evaluate_panel(expert_policy, recovery, recovery_window=64)[1024]
        self.assertEqual(expert["success"], 1.)
        self.assertEqual(expert["worst_direction_success"], 1.)
        self.assertLess(expert["radial_rmse"], 1e-5)
        self.assertLess(expert["signed_speed_error"], 1e-5)
        self.assertEqual(zero["success"], 0.)
        self.assertLess(zero["completed_turns"], 1e-6)
        self.assertEqual(zero["direction_agreement"], 0.)
        self.assertEqual(reversed_row["success"], 0.)
        self.assertLess(reversed_row["direction_agreement"], 0.05)
        self.assertLess(reversed_row["completed_turns"], 0.)
        self.assertEqual(expert_recovery["success"], 1.)
        self.assertIn("after_recovery_window", expert_recovery)
        self.assertLess(expert_recovery["after_recovery_window"]["radial_rmse"], 0.05)


class EncoderTests(unittest.TestCase):
    def test_encoder_has_no_hidden_state(self):
        prior = MoGParticlePrior(8, 4, generator=torch.Generator().manual_seed(3))
        encoder = CircleEncoder(6, 4, 16)
        features = torch.randn(5, 6)
        first = encoder(features, prior).codes
        _ = encoder(torch.randn(2, 6), prior)
        second = encoder(features, prior).codes
        order = torch.tensor([4, 0, 3, 1, 2])
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(encoder(features[order], prior).codes, first[order])
        with self.assertRaises(ValueError):
            encoder(torch.randn(5, 8), prior)
        self.assertFalse(any(isinstance(module, (torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN, torch.nn.BatchNorm1d))
                             for module in encoder.modules()))


class ControllerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        targets = torch.randn(64, 2)
        self.critic = build_edit_critic(targets, torch.zeros_like(targets),
                                        dict(error_tokens=8, error_width=48, error_heads=4))
        self.reg = edit_cap()
        self.predicted = torch.randn(16, 2, requires_grad=True)
        self.target = torch.randn(16, 2)

    def test_adv_weight_zero_is_rejected(self):
        with self.assertRaises(ValueError):
            require_live_adversary(0)
        cfg = {**DEFAULTS, "adv_weight": 0.}
        with self.assertRaises(ValueError):
            validate(cfg)
        with self.assertRaises(ValueError):
            controller_objective(self.critic, self.predicted, self.target, 1, torch.Generator().manual_seed(1), 8, 0)

    def test_controller_loss_is_paired_error_rpgan(self):
        left, right = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
        loss, terms = controller_objective(self.critic, self.predicted, self.target, 2, left, 20, 1.)
        noise, fake = paired_noise(self.critic, self.predicted, self.target, 2, right, 20)
        with torch.no_grad():
            real_score = self.critic(noise)
        expected = rp_g_loss(real_score, self.critic(fake))
        torch.testing.assert_close(loss, expected)
        self.assertTrue(loss.requires_grad)
        self.assertEqual(float(terms["adv_weight"]), 1.)
        diagnostic = torch.nn.functional.mse_loss(self.predicted.detach(), self.target.detach())
        self.assertIsNone(diagnostic.grad_fn)
        self.assertGreater(abs(float(loss.detach()) - float(diagnostic)), 1e-6)

    def test_b_cap_is_every_fourth_update(self):
        self.assertEqual(self.reg.lazy_k, 4)
        self.assertEqual(self.reg.arm, "b_cap")
        _, skipped = discriminator_objective(self.critic, self.predicted.detach(), self.target, 1,
                                             torch.Generator().manual_seed(1), self.reg, 8)
        _, applied = discriminator_objective(self.critic, self.predicted.detach(), self.target, 4,
                                            torch.Generator().manual_seed(2), self.reg, 8)
        self.assertEqual(skipped["b_cap_applied"], 0.)
        self.assertEqual(applied["b_cap_applied"], 1.)
        self.assertGreaterEqual(float(applied["b_cap"]), 0.)

    def test_finetune_scope_trains_only_control_and_g2(self):
        from experiments.train_circle_transition import _build
        bundle = _build({**DEFAULTS, "width": 16, "encoder_width": 16, "d_width": 16, "z_dim": 4,
                         "num_particles": 8}, torch.device("cpu"))
        bundle["scaler"] = torch.nn.Identity()
        configure_control_scope(bundle)
        self.assertTrue(any(parameter.requires_grad for parameter in bundle["E_control"].parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in bundle["G"].branches[1].parameters()))
        frozen = (bundle["E"], bundle["prior"], bundle["D"], bundle["G"].branches[0], bundle["G"].branches[2])
        self.assertFalse(any(parameter.requires_grad for module in frozen for parameter in module.parameters()))


class SmokeTests(unittest.TestCase):
    def test_short_run_logs_and_keeps_training_off_the_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            cfg = {**DEFAULTS, "width": 16, "encoder_width": 16, "d_width": 16, "z_dim": 4, "num_particles": 8,
                   "pretrain_steps": 1, "finetune_steps": 4, "batch_size": 8, "normalization_samples": 64,
                   "eval_episodes": 4, "log_interval": 1, "out_dir": str(path / "run"),
                   "live_log": str(path / "live.log"), "device": "cpu"}
            summary = train(cfg)
            text = (path / "live.log").read_text()
            self.assertIn("adv_weight=1", text)
            self.assertIn("diag_action_mse", text)
            self.assertIn("l2_aux=0", text)
            self.assertEqual(summary["simulator_calls_during_training"], 0)
            self.assertGreaterEqual(summary["b_cap_applications"], 1)
            self.assertGreater(summary["gan_grad_abs"], 0.)
            self.assertEqual(summary["evaluation"]["controls"]["test"]["expert"]["main"]["1024"]["success"], 1.)
            self.assertEqual(summary["evaluation"]["controls"]["test"]["zero"]["main"]["1024"]["success"], 0.)
            self.assertLess(summary["evaluation"]["controls"]["test"]["reversed"]["main"]["1024"]["direction_agreement"],
                            0.05)
            self.assertTrue((path / "run" / "metrics.jsonl").is_file())
            self.assertTrue((path / "run" / "final.pt").is_file())


if __name__ == "__main__":
    unittest.main()
