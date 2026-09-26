"""Sparse training and evaluation regressions; runs on CPU with unittest."""

import copy
import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import analyze_sparse
from experiments.train_sparse import DEFAULTS, train
from particlegan import get_recipe
from particlegan.ka2 import CriticPenalty
from lib.sparse_metrics import particle_class_purity
from lib.sparse_models import JointCritic, SparseCondGenerator, XOnlyCritic
from lib.sparse_toy import SparseMixedToy


class SparseRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_particle_specialization_uses_associated_ids(self):
        # Every particle serves only its own class; every requested class is
        # tested equally often for every particle.
        particles, classes = 8, 4
        idx = torch.arange(particles).repeat_interleave(classes * 8)
        requested = torch.arange(classes).repeat(particles * 8)
        generated = idx % classes
        self.assertEqual(particle_class_purity(idx, requested, generated, particles, classes), 1.0)
        torch.manual_seed(42)
        unrelated_idx = torch.randint(particles, idx.shape)
        self.assertLess(particle_class_purity(unrelated_idx, requested, generated, particles, classes), 0.75)

    def test_class_conditioned_shared_particles_can_have_low_purity(self):
        idx = torch.arange(8).repeat_interleave(32)
        requested = torch.arange(4).repeat(64)
        self.assertEqual(particle_class_purity(idx, requested, requested, 8, 4), 0.25)
        self.assertTrue(math.isnan(particle_class_purity(idx, requested, (requested + 1) % 4, 8, 4)))
        self.assertTrue(math.isnan(particle_class_purity(idx[:3], requested[:3], requested[:3], 8, 4)))

    def test_eval_hardness_does_not_establish_soft_head_confidence(self):
        generator = SparseCondGenerator(4, 4, 6, 4, 2, hidden=16, cat_mode="soft")
        with torch.no_grad():
            generator.cat.weight.zero_()
            generator.cat.bias.zero_()
            z, c = torch.zeros(8, 4), torch.arange(8) % 4
            soft = generator(z, c)
            generator.eval()
            hard = generator(z, c)
        self.assertEqual(float(soft["y"].max(1).values.mean()), 0.25)
        self.assertEqual(float(hard["y"].max(1).values.mean()), 1.0)
        self.assertEqual(float(hard["logits"].softmax(1).max(1).values.mean()), 0.25)

    def test_analyzer_separates_legacy_gp_and_omits_invalid_purity(self):
        legacy = {"config": {"gp_on_y": False, "arm": "g_interp_cap"},
                  "final": {"modes": 8, "joint_acc": 1.0, "particle_class_purity": 0.91},
                  "bar_held": False, "bar_step": None}
        corrected = copy.deepcopy(legacy)
        corrected["implementation_versions"] = {"x_only_gp": 2, "particle_class_purity": 2}
        corrected["final"]["particle_class_purity"] = 0.25
        with tempfile.TemporaryDirectory() as directory:
            for seed, summary in enumerate((legacy, corrected), 1):
                run = Path(directory) / "ucd" / f"ucd_gpx_s{seed}"
                run.mkdir(parents=True)
                (run / "summary.json").write_text(json.dumps(summary))
            with patch.object(analyze_sparse, "RUNS_ROOT", Path(directory)):
                runs, _, missing = analyze_sparse.load_stage("ucd")
        self.assertEqual(missing, 0)
        self.assertEqual(set(runs), {"ucd_gpx", "ucd_gpx [legacy x-only GP]"})
        rendered = analyze_sparse.table("ucd", runs, missing)
        self.assertNotIn("0.91", rendered)
        self.assertIn("0.25", rendered)
        self.assertIn("Historical ppur values are omitted", rendered)

    def test_cpu_training_retains_final_particle_ids(self):
        cfg = {**DEFAULTS, "total_steps": 6, "eval_interval": 3,
               "batch_size": 32, "eval_n": 128, "final_n": 256,
               "d": 6, "k": 2, "n_modes": 8, "n_classes": 2, "n_symbols": 2,
               "hidden": 16, "n_hidden": 2, "z_dim": 4, "num_particles": 16,
               "emb_dim": 4, "prior_partition": "none", "real_head": "gated",
               "gate_start_frac": 0.4, "fourier": 1, "gp_on_y": False}
        original_penalty = CriticPenalty.__call__
        penalty_calls = []

        def check_penalty(penalty, critic, real, fake):
            self.assertFalse(critic.grad_on_y)
            self.assertEqual(penalty.regularizer.coeff, get_recipe().reg_coeff)
            self.assertEqual(real.shape, (cfg["batch_size"], cfg["d"] + cfg["n_symbols"]))
            self.assertFalse(torch.equal(real, fake))
            penalty_calls.append(penalty.optimizer.record.observed_steps)
            return original_penalty(penalty, critic, real, fake)

        with tempfile.TemporaryDirectory() as directory:
            cfg["out_dir"] = directory
            with patch.object(CriticPenalty, "__call__", check_penalty), redirect_stdout(io.StringIO()):
                summary = train(cfg, torch.device("cpu"))
            self.assertEqual(penalty_calls, list(range(cfg["total_steps"])))
            self.assertEqual(summary["implementation_versions"], {"x_only_gp": 2, "particle_class_purity": 2})
            with np.load(Path(directory) / "final_samples.npz") as saved:
                x = torch.from_numpy(saved["x"])
                c = torch.from_numpy(saved["c"])
                idx = torch.from_numpy(saved["particle_idx"])
            checkpoint = torch.load(Path(directory) / "ckpt.pt", weights_only=True)
            generator = SparseCondGenerator(
                cfg["z_dim"], cfg["n_classes"], cfg["d"], cfg["n_symbols"], cfg["k"],
                hidden=cfg["hidden"], n_hidden=cfg["n_hidden"], emb_dim=cfg["emb_dim"],
                real_head=cfg["real_head"], cat_mode=cfg["cat_mode"])
            generator.load_state_dict(checkpoint["ema_G"])
            generator.eval()
            with torch.no_grad():
                regenerated = generator(checkpoint["ema_prior"]["z"][idx], c)["x"]
            torch.testing.assert_close(regenerated, x, rtol=0, atol=0)
            toy = SparseMixedToy(d=6, k=2, n_modes=8, n_classes=2, n_symbols=2, seed=cfg["data_seed"])
            assigned, _ = toy.assign(x)
            recomputed = particle_class_purity(idx, c, toy.class_of_mode[assigned], cfg["num_particles"], cfg["n_classes"])
            self.assertTrue(math.isfinite(recomputed))
            self.assertAlmostEqual(recomputed, summary["final"]["particle_class_purity"])
            rows = [json.loads(line) for line in (Path(directory) / "metrics.jsonl").read_text().splitlines()]
            self.assertTrue(all(math.isfinite(row["pen"]) for row in rows[1:]))
            for artifact in ("summary.json", "heatmap.png", "confusion.png", "pca.png"):
                self.assertTrue((Path(directory) / artifact).is_file())


if __name__ == "__main__":
    unittest.main()
