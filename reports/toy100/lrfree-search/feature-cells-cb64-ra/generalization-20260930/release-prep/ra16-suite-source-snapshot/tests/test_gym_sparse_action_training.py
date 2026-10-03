"""Sparse labels cannot leak through normalization, batching, losses or updates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.train_gym_sparse_action import DEFAULTS, train
from lib.gym_sparse_action import build_sparse_records, fit_sparse_scaler, sparse_task_losses
from lib.gym_state_control import build_state_models, parameter_hashes, load_state_control_checkpoint, state_control_action


class SparseActionTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = {**DEFAULTS, "z_dim": 4, "num_particles": 8, "width": 8,
                    "encoder_width": 8, "device": "cpu", "batch_size": 4}

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
            labeled_terrain=records["terrain"][labeled], states=records["states"][-5:],
            next_states=records["next_states"][-5:], terrain=records["terrain"][-5:]).items()}

    def test_whole_episode_selection_hidden_action_exclusion_and_scaler(self):
        with tempfile.TemporaryDirectory() as td:
            path, hidden, records, selection = self.fixture(Path(td))
            other, other_selection = build_sparse_records(hidden)
            self.assertEqual(selection, other_selection)
            self.assertEqual(len(selection["labeled_episode_ids"]), 5)
            self.assertNotIn("actions", records)
            self.assertNotIn("previous_actions", records)
            self.assertNotIn(88, records["episode_ids"])
            for key in records:
                np.testing.assert_array_equal(records[key], other[key])
            for eid in np.unique(records["episode_ids"]):
                mask = records["label_mask"][records["episode_ids"] == eid]
                self.assertTrue((mask == (eid in selection["labeled_episode_ids"])).all())
            scaler, other_scaler = fit_sparse_scaler(records), fit_sparse_scaler(other)
            for key in scaler.state_dict():
                torch.testing.assert_close(scaler.state_dict()[key], other_scaler.state_dict()[key], rtol=0, atol=0)
            torch.testing.assert_close(scaler.action_mean, torch.from_numpy(records["labeled_actions"]).mean(0))
            # States from unlabeled episodes still determine the state scaler.
            changed = {**records, "states": records["states"].copy()}
            changed["states"][~records["label_mask"], :6] += 3
            self.assertFalse(torch.equal(scaler.state_mean, fit_sparse_scaler(changed).state_mean))

    def test_gradient_scopes_and_action_reduction_independent_of_unlabeled_count(self):
        with tempfile.TemporaryDirectory() as td:
            _, _, records, _ = self.fixture(Path(td))
            batch = self.batch(records)
            hashes = []
            for arm in ("probes", "auxiliary"):
                bundle = build_state_models({**self.cfg, "arm": arm}, fit_sparse_scaler(records))
                hashes.append(parameter_hashes(bundle))
                modules = dict(E=bundle["E"], prior=bundle["prior"],
                    **{f"G{i+1}": branch for i,branch in enumerate(bundle["G"].branches)})
                def scope():
                    return {key for key,module in modules.items() if any(
                        p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters())}
                _, terms = sparse_task_losses(bundle, **batch)
                terms["action_loss"].backward()
                self.assertEqual(scope(), {"E", "prior", "G2"})
                action_value = terms["action_loss"].detach().clone()
                for module in modules.values():
                    module.zero_grad(set_to_none=True)
                larger = {key: value.repeat(3, 1) if key in ("states", "next_states", "terrain") else value
                          for key,value in batch.items()}
                _, terms = sparse_task_losses(bundle, **larger)
                torch.testing.assert_close(action_value, terms["action_loss"], rtol=0, atol=0)
                (terms["state_loss"] + terms["next_loss"]).backward()
                self.assertEqual(scope(), {"G1", "G3"} | ({"E", "prior"} if arm == "auxiliary" else set()))
                # Explicit action labels and successors never enter E or G2 inputs.
                _, changed = sparse_task_losses(bundle, **{**batch, "labeled_actions": -batch["labeled_actions"]})
                self.assertNotEqual(float(action_value), float(changed["action_loss"].detach()))
            self.assertEqual(hashes[0], hashes[1])

    def test_hidden_actions_do_not_change_draws_losses_or_model_updates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path, hidden, records, _ = self.fixture(root)
            summaries = []
            for arm in ("probes", "auxiliary"):
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
                    bundle = load_state_control_checkpoint(out / "final.pt")
                    action, component = state_control_action(bundle, records["states"][0], records["terrain"][0])
                    self.assertTrue(np.isfinite(action).all() and (np.abs(action) <= 1).all())
                    replay = load_state_control_checkpoint(out / "checkpoint_2.pt")
                    replay_action, replay_component = state_control_action(replay, records["states"][0], records["terrain"][0])
                    np.testing.assert_array_equal(action, replay_action)
                    self.assertEqual(component, replay_component)
                    if variant == "original":
                        original_hashes = parameter_hashes(bundle)
                        original_metrics = [json.loads(row) for row in (out / "metrics.jsonl").read_text().splitlines()]
                    else:
                        self.assertEqual(original_hashes, parameter_hashes(bundle))
                        metrics = [json.loads(row) for row in (out / "metrics.jsonl").read_text().splitlines()]
                        for first, second in zip(original_metrics, metrics):
                            first.pop("elapsed_seconds")
                            second.pop("elapsed_seconds")
                            self.assertEqual(first, second)
                    with self.assertRaises(FileExistsError):
                        train(cfg)
            for summary in summaries[1:]:
                self.assertEqual(summaries[0]["provenance"]["initial_parameters"], summary["provenance"]["initial_parameters"])
                self.assertEqual(summaries[0]["provenance"]["expert_data"]["arrays"], summary["provenance"]["expert_data"]["arrays"])
                self.assertEqual(summaries[0]["data_draw_sha256"], summary["data_draw_sha256"])


if __name__ == "__main__":
    unittest.main()
