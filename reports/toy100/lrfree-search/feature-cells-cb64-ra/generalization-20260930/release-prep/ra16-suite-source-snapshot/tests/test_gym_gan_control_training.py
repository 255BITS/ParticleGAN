"""GAN gradient paths, structural observation masks, and hidden-label nonleakage."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


from experiments.train_gym_gan_control import DEFAULTS, train
from lib.gym_sparse_action import build_sparse_records, fit_sparse_scaler, sparse_task_losses
from lib.gym_state_control import training_recipe, state_control_action
from particlegan.grad_regularizers import GradientPenalty
from lib.gym_gan_control import (build_gan_models, initial_hashes, load_gan_control_checkpoint,
    real_views, fake_views, discriminator_loss, generator_loss)


class GanControlTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = {**DEFAULTS, "z_dim": 4, "num_particles": 8, "width": 8,
                    "encoder_width": 8, "d_width": 8, "marginal_width": 8, "device": "cpu", "batch_size": 4}

    def fixture(self, root):
        rng = np.random.default_rng(19)
        episodes = []
        for eid in range(7):
            n = eid + 4
            states = rng.normal(size=(n, 8)).astype(np.float32)
            states[:, 6:] = rng.integers(0, 2, size=(n, 2))
            actions = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
            successor = states.copy()
            successor[:, :2] += actions * .1
            episodes.append(dict(episode_id=eid, split="train", behavior="heuristic", states=states.tolist(),
                actions=actions.tolist(), next_states=successor.tolist(), terrain=[-.5]*11))
        episodes.append({**episodes[0], "episode_id": 88, "split": "test"})
        path = root / "episodes.json"
        path.write_text(json.dumps(episodes))
        records, selection = build_sparse_records(path)
        hidden = copy.deepcopy(episodes)
        for episode in hidden:
            if episode["episode_id"] not in selection["labeled_episode_ids"]:
                # Stronger than numeric perturbation: hidden commands can be absent.
                episode.pop("actions")
        hidden_path = root / "hidden_changed.json"
        hidden_path.write_text(json.dumps(hidden))
        return path, hidden_path, records, selection

    def batch(self, records):
        labeled = records["labeled_indices"][:4]
        return {k: torch.from_numpy(v) for k,v in dict(
            labeled_states=records["states"][labeled], labeled_actions=records["labeled_actions"][:4],
            labeled_terrain=records["terrain"][labeled], labeled_next_states=records["next_states"][labeled], states=records["states"][-5:],
            next_states=records["next_states"][-5:], terrain=records["terrain"][-5:]).items()}

    def test_mask_structural_inside_critic_and_matched_shared_initialization(self):
        with tempfile.TemporaryDirectory() as td:
            _, _, records, _ = self.fixture(Path(td))
            bundles = [build_gan_models({**self.cfg, "arm": arm}, fit_sparse_scaler(records))
                       for arm in ("joint", "marginals")]
            hashes = [initial_hashes(bundle) for bundle in bundles]
            for key in ("G", "E", "prior", "D.joint"):
                self.assertEqual(hashes[0][key], hashes[1][key])
            fake_results, rng_states = [], []
            for bundle in bundles:
                views = real_views(bundle, self.batch(records))
                self.assertTrue((views["all"]["real"][:, 8:10] == 0).all())
                critic = bundle["D"].critic_for("joint")
                view = views["all"]
                inputs, context = bundle["D"].inputs("joint", view["real"], view["terrain"], view["observed"])
                raw = inputs.clone().requires_grad_(True)
                score = critic(raw, context)[0]
                gradient = torch.autograd.grad(score.sum(), raw)[0]
                self.assertTrue((gradient[:, 8:10] == 0).all())
                changed = raw.detach().clone()
                changed[:, 8:10] = float("nan")
                torch.testing.assert_close(score, critic(changed, context)[0], atol=0, rtol=0)
                penalty = GradientPenalty(kappa=0.)
                original_penalty, _ = penalty.penalty(lambda x: critic(x, context)[0], raw, raw, 1)
                changed_penalty, _ = penalty.penalty(lambda x: critic(x, context)[0], changed, changed, 1)
                self.assertGreater(float(original_penalty.detach()), 0.)
                torch.testing.assert_close(original_penalty, changed_penalty, atol=0, rtol=0)
                latent, contact = [torch.Generator().manual_seed(i) for i in (19, 23)]
                fakes = fake_views(bundle, views, latent, contact, straight_through=True)
                fake_results.append(fakes)
                rng_states.append((latent.get_state(), contact.get_state()))
                for name, view in views.items():
                    for fake in fakes[name].values():
                        xr, cr = bundle["D"].inputs("joint", view["real"], view["terrain"], view["observed"])
                        xf, cf = bundle["D"].inputs("joint", fake, view["terrain"], view["observed"])
                        torch.testing.assert_close(cr, cf, atol=0, rtol=0)
                        if name == "all":
                            self.assertTrue((xr[:, 8:10] == 0).all() and (xf[:, 8:10] == 0).all())
            for name in fake_results[0]:
                for path in fake_results[0][name]:
                    torch.testing.assert_close(fake_results[0][name][path], fake_results[1][name][path], atol=0, rtol=0)
            for a,b in zip(rng_states[0], rng_states[1]):
                torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_adversarial_path_gradients_and_discriminator_isolation(self):
        def has_grad(module):
            return any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters())
        with tempfile.TemporaryDirectory() as td:
            _, _, records, _ = self.fixture(Path(td))
            batch = self.batch(records)
            for arm in ("joint", "marginals"):
                bundle = build_gan_models({**self.cfg, "arm": arm}, fit_sparse_scaler(records))
                recipe = training_recipe(bundle["config"])
                gan = recipe.make_loss()
                opt_d = recipe.make_critic_optimizer(bundle["D"], ema_critic=copy.deepcopy(bundle["D"]))
                views = real_views(bundle, batch)
                for path in ("prior", "encoded"):
                    for key in ("E", "G", "prior", "D"):
                        bundle[key].zero_grad(set_to_none=True)
                    bundle["D"].requires_grad_(False)
                    fakes = fake_views(bundle, views, torch.Generator().manual_seed(5),
                                       torch.Generator().manual_seed(6), straight_through=True)
                    single = {name: {path: value[path]} for name,value in fakes.items()}
                    loss, terms = generator_loss(bundle["D"], views, single, gan)
                    loss.backward()
                    self.assertTrue(all(has_grad(branch) for branch in bundle["G"].branches))
                    self.assertTrue(has_grad(bundle["prior"]))
                    self.assertEqual(has_grad(bundle["E"]), path == "encoded")
                    self.assertFalse(has_grad(bundle["D"]))
                for key in ("E", "G", "prior", "D"):
                    bundle[key].zero_grad(set_to_none=True)
                bundle["D"].requires_grad_(True)
                # Intentionally pass LIVE fake graphs: D helper must detach them itself.
                fakes = fake_views(bundle, views, torch.Generator().manual_seed(7),
                                   torch.Generator().manual_seed(8), straight_through=True)
                rngs = {role: torch.Generator().manual_seed(50+i) for i,role in enumerate(bundle["D"].roles())}
                penalties = {role: recipe.make_critic_penalty(opt_d) for role, rng in rngs.items()}
                loss, terms = discriminator_loss(bundle["D"], views, fakes, gan, penalties)
                loss.backward()
                self.assertTrue(has_grad(bundle["D"]))
                self.assertTrue(all(not has_grad(bundle[key]) for key in ("E", "G", "prior")))
                for role in bundle["D"].roles():
                    self.assertIn(role + "_gan", terms)
                    self.assertIn(role + "_penalty", terms)
                # Both GAN variants keep auxiliary gradients in E/prior.
                for key in ("E", "G", "prior", "D"):
                    bundle[key].zero_grad(set_to_none=True)
                _, rec = sparse_task_losses(bundle, **{k:v for k,v in batch.items() if k != "labeled_next_states"})
                (rec["state_loss"] + rec["next_loss"]).backward()
                self.assertTrue(has_grad(bundle["E"]) and has_grad(bundle["prior"]))
                self.assertFalse(has_grad(bundle["G"].branches[1]))

    def test_hidden_actions_do_not_change_draws_losses_or_model_updates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path, hidden, records, _ = self.fixture(root)
            summaries = []
            for arm in ("joint", "marginals"):
                for variant, source in (("original", path), ("hidden_removed", hidden)):
                    out = root / f"{arm}_{variant}"
                    cfg = {**self.cfg, "arm": arm, "steps": 2, "checkpoints": [1, 2], "log_interval": 1,
                           "episodes": str(source), "out_dir": str(out), "live_log": str(root / "live.log")}
                    summary = train(cfg)
                    summaries.append(summary)
                    self.assertEqual(summary["unique_training_records"], len(records["states"]))
                    self.assertEqual(summary["unique_labeled_records"], len(records["labeled_actions"]))
                    self.assertEqual(summary["labeled_record_draws"], 8)
                    self.assertEqual(summary["auxiliary_record_draws"], 8)
                    self.assertEqual(summary["simulator_calls"], 0)
                    self.assertTrue(summary["gan_training"])
                    self.assertEqual(summary["gan_steps"], 2)
                    self.assertGreater(summary["discriminator_parameters"], 0)
                    bundle = load_gan_control_checkpoint(out / "final.pt")
                    current_hashes = initial_hashes(bundle)
                    for key in summary["provenance"]["initial_parameters"]:
                        self.assertNotEqual(current_hashes[key], summary["provenance"]["initial_parameters"][key], key)
                    action, component = state_control_action(bundle, records["states"][0], records["terrain"][0])
                    self.assertTrue(np.isfinite(action).all() and (np.abs(action) <= 1).all())
                    replay = load_gan_control_checkpoint(out / "checkpoint_2.pt")
                    replay_action, replay_component = state_control_action(replay, records["states"][0], records["terrain"][0])
                    np.testing.assert_array_equal(action, replay_action)
                    self.assertEqual(component, replay_component)
                    if variant == "original":
                        original_hashes = initial_hashes(bundle)
                        original_metrics = [json.loads(row) for row in (out / "metrics.jsonl").read_text().splitlines()]
                    else:
                        self.assertEqual(original_hashes, initial_hashes(bundle))
                        metrics = [json.loads(row) for row in (out / "metrics.jsonl").read_text().splitlines()]
                        for first, second in zip(original_metrics, metrics):
                            first.pop("elapsed_seconds")
                            second.pop("elapsed_seconds")
                            self.assertEqual(first, second)
                    with self.assertRaises(FileExistsError):
                        train(cfg)
            for summary in summaries[1:]:
                for key in ("E", "G", "prior", "D.joint"):
                    self.assertEqual(summaries[0]["provenance"]["initial_parameters"][key], summary["provenance"]["initial_parameters"][key])
                self.assertEqual(summaries[0]["provenance"]["expert_data"]["arrays"], summary["provenance"]["expert_data"]["arrays"])
                self.assertEqual(summaries[0]["data_draw_sha256"], summary["data_draw_sha256"])


if __name__ == "__main__":
    unittest.main()
