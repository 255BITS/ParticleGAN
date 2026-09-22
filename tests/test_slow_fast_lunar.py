"""Slow→fast Lunar pairing, crash refusal, and a CPU trainer smoke. No GPU landings."""
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.collect_slow_fast_lunar import DEFAULTS as COLLECT_DEFAULTS, collect, validate as validate_collect
from experiments.config import read_config
from experiments.evaluate_gym_slow_fast import render_report, select_winner
from experiments.train_gym_fast_teacher import (DEFAULTS as FAST_DEFAULTS, hold_decision,
    train as train_fast, validate as validate_fast)
from experiments.train_gym_slow_fast import DEFAULTS as TRAIN_DEFAULTS, train, validate as validate_train
from experiments.train_gym_transition import DEFAULTS as WORLD_DEFAULTS, build_models, sha256
from lib.gym_control_evaluation import freeze_protocol
from lib.gym_control import control_action_details
from lib.gym_particle_finetune import (MODULE_KEYS, initialize_particle_finetune,
    load_paired_controller, playback_action)
from lib.slow_fast_lunar import (EVAL_SEEDS, TEST_SEEDS, VALIDATION_SEEDS, archive_stranger_pairs,
    assert_collection_seeds, build_progress_pairs, build_slow_fast_pairs, is_success, load_pairs,
    mean_success_steps, progress_index, save_pairs, speed_decision, speed_stats,
    synthetic_teacher_rollouts)


class SlowFastLunarTests(unittest.TestCase):
    def test_shared_seeds_match_the_control_protocol_and_collection_stays_out(self):
        source = inspect.getsource(freeze_protocol)
        self.assertIn("range(391000, 391020)", source)
        self.assertIn("range(491000, 491050)", source)
        self.assertEqual(VALIDATION_SEEDS, tuple(range(391000, 391020)))
        self.assertEqual(TEST_SEEDS, tuple(range(491000, 491050)))
        self.assertTrue(set(VALIDATION_SEEDS).isdisjoint(TEST_SEEDS))
        with self.assertRaises(ValueError):
            assert_collection_seeds([391000, 591000])
        with self.assertRaises(ValueError):
            assert_collection_seeds([491000])
        self.assertEqual(assert_collection_seeds([591000, 591001]), [591000, 591001])
        self.assertTrue(EVAL_SEEDS.isdisjoint([591000]))

    def test_same_seed_both_land_aligns_by_progress_and_drops_crashes(self):
        safe, fast = synthetic_teacher_rollouts()
        self.assertFalse(is_success(fast[3]))
        self.assertTrue(is_success(safe[0]))
        self.assertEqual(progress_index(40, 12, 0), 0)
        self.assertEqual(progress_index(40, 12, 39), 11)
        with self.assertRaises(RuntimeError) as caught:
            build_slow_fast_pairs(safe, fast)
        self.assertIn("stranger nearest", str(caught.exception))
        built = build_progress_pairs(safe, fast)
        manifest = built["manifest"]
        arrays = built["arrays"]
        self.assertEqual(manifest["pairing"], "progress_same_seed")
        self.assertEqual(manifest["alignment"], "t/T")
        self.assertEqual(manifest["crashes_excluded"], 1)
        self.assertEqual(manifest["outcomes_in_pools"], ["successful_landing"])
        self.assertTrue(np.all(arrays["slow_seed"] == arrays["fast_seed"]))
        self.assertTrue(np.all(arrays["fast_steps"] < arrays["slow_steps"]))
        self.assertNotIn(4, set(arrays["slow_seed"].tolist()))
        first = np.where(arrays["slow_index"] == 0)[0]
        self.assertGreater(len(first), 0)
        np.testing.assert_allclose(arrays["previous_actions"][first[0]], [-1., 0.])
        self.assertGreater(manifest["mean_action_edit"], 0.05)
        self.assertGreater(manifest["rows"], 2)
        self.assertEqual(manifest["nearest"]["count"], 0)
        with tempfile.TemporaryDirectory() as td:
            path, _ = save_pairs(Path(td) / "pairs_progress.npz", built)
            loaded, sidecar = load_pairs(path)
            np.testing.assert_array_equal(loaded["target_actions"], arrays["target_actions"])
            self.assertEqual(sidecar["pairing"], "progress_same_seed")
            arrays["fast_seed"] = arrays["slow_seed"] + 1
            np.savez_compressed(Path(td) / "cross.npz", **arrays)
            with self.assertRaises(ValueError) as cross:
                load_pairs(Path(td) / "cross.npz")
            self.assertIn("stranger pairs", str(cross.exception))

    def test_pairs_stay_on_their_own_seed(self):
        def episode(seed, steps, terrain, action):
            states = np.zeros((steps, 8), dtype=np.float32)
            states[:, 1] = np.linspace(1.2, 0.05, steps)
            return dict(seed=seed, steps=steps, outcome="successful_landing", states=states,
                        actions=np.tile(action, (steps, 1)).astype(np.float32),
                        initial_state=states[0].tolist(), terrain=[terrain] * 11,
                        game_over=False, lander_awake=False, terminated=True, truncated=False)

        safe = [episode(1, 20, 0., [-0.2, 0.]), episode(2, 30, 1., [-0.2, 0.])]
        fast = [episode(1, 8, 0., [0.4, 0.]), episode(2, 10, 1., [0.5, 0.]),
                episode(3, 6, 0., [1., 0.])]
        fast[2]["outcome"] = "crash"
        fast[2]["game_over"] = True
        fast[2]["lander_awake"] = True
        built = build_progress_pairs(safe, fast)
        seeds = built["arrays"]["slow_seed"]
        self.assertTrue(np.array_equal(seeds, built["arrays"]["fast_seed"]))
        self.assertTrue(np.all(built["arrays"]["terrain"][seeds == 1] == 0))
        self.assertTrue(np.all(built["arrays"]["terrain"][seeds == 2] == 1))
        self.assertNotIn(3, set(seeds.tolist()))
        self.assertEqual(built["manifest"]["crashes_excluded"], 1)

    def test_archive_renames_legacy_pairs_and_collect_refuses_that_name(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            legacy = directory / "pairs.npz"
            legacy.write_bytes(b"stranger")
            (directory / "pairs.json").write_text("{}\n")
            dest = archive_stranger_pairs(directory)
            self.assertEqual(dest.name, "pairs_stranger_do_not_train.npz")
            self.assertFalse(legacy.exists())
            self.assertTrue((directory / "pairs_stranger_do_not_train.json").is_file())
            self.assertIsNone(archive_stranger_pairs(directory))
        with self.assertRaises(ValueError) as caught:
            validate_collect({**COLLECT_DEFAULTS, "out": "results/pairs.npz"})
        self.assertIn("pairs.npz", str(caught.exception))

    def test_hold_decision_keeps_a_faster_landing_and_drops_the_break(self):
        safe = dict(landing_count=20, crash_count=0, oob_count=0, mean_success_steps=200.)
        self.assertEqual(hold_decision(safe, dict(
            landing_count=20, crash_count=0, oob_count=0, mean_success_steps=150.)), "hold")
        self.assertEqual(hold_decision(safe, dict(
            landing_count=18, crash_count=2, oob_count=0, mean_success_steps=80.)), "break")
        self.assertEqual(hold_decision(safe, dict(
            landing_count=20, crash_count=0, oob_count=1, mean_success_steps=150.)), "break")
        self.assertEqual(hold_decision(safe, dict(
            landing_count=20, crash_count=0, oob_count=0, mean_success_steps=200.)), "continue")

    def test_success_steps_ignore_crashes_and_shortcuts_are_refused(self):
        episodes = [dict(outcome="successful_landing", steps=300),
                    dict(outcome="crash", steps=5),
                    dict(outcome="successful_landing", steps=100)]
        self.assertEqual(mean_success_steps(episodes), 200.)
        stats = speed_stats(episodes)
        self.assertEqual(stats["crash_count"], 1)
        self.assertLess(stats["mean_episode_steps"], stats["mean_success_steps"])
        baseline = dict(landing_count=3, crash_count=0, oob_count=0, timeout_count=0,
                        mean_success_steps=300., episodes=3, mean_episode_steps=300.)
        shortcut = dict(landing_count=1, crash_count=2, oob_count=0, timeout_count=0,
                        mean_success_steps=40., episodes=3, mean_episode_steps=20.)
        refused = speed_decision(shortcut, baseline)
        self.assertFalse(refused["eligible"])
        self.assertTrue(any("landing" in reason or "crash" in reason for reason in refused["reasons"]))
        faster = dict(landing_count=3, crash_count=0, oob_count=0, timeout_count=0,
                      mean_success_steps=220., episodes=3, mean_episode_steps=220.)
        self.assertTrue(speed_decision(faster, baseline)["eligible"])
        rows = [
            dict(role="crashy", step=250, seeds=list(VALIDATION_SEEDS), speed=shortcut),
            dict(role="faster", step=1000, seeds=list(VALIDATION_SEEDS), speed=faster),
        ]
        text = render_report("validation", dict(step=2500, seeds=list(VALIDATION_SEEDS), speed=baseline), rows)
        self.assertIn("No speed win" if not any(row["decision"]["eligible"] for row in rows) else "faster", text)
        self.assertIn("not a ranking key", text)
        self.assertFalse(rows[0]["decision"]["eligible"])
        self.assertTrue(rows[1]["decision"]["eligible"])
        winner = select_winner(rows)
        self.assertEqual(winner["role"], "faster")

    def test_configs_lock_adv_weight_and_reject_the_kinematic_cost(self):
        root = Path(__file__).resolve().parents[1]
        collect_cfg = {**COLLECT_DEFAULTS, **read_config(root / "configs/gym/lunar_lander_slow_fast/collect.yaml")}
        validate_collect(collect_cfg)
        self.assertEqual(collect_cfg["device"], "cuda:1")
        self.assertEqual(collect_cfg["seed_start"], 591000)
        self.assertEqual(Path(collect_cfg["out"]).name, "pairs_progress.npz")
        fast_cfg = {**FAST_DEFAULTS, **read_config(root / "configs/gym/lunar_lander_slow_fast/fast_teacher.yaml")}
        validate_fast(fast_cfg)
        self.assertEqual(fast_cfg["device"], "cuda:1")
        self.assertEqual(fast_cfg["probe_seed_start"], 581000)
        self.assertGreater(fast_cfg["speed_bias"], 0)
        self.assertGreater(fast_cfg["anchor_weight"], 0)
        train_cfg = {**TRAIN_DEFAULTS, **read_config(
            root / "configs/gym/lunar_lander_particle_finetune/particle_slow_fast.yaml")}
        validate_train(train_cfg)
        self.assertEqual(train_cfg["pairs"], "results/gym/lunar_lander_slow_fast/pairs_progress.npz")
        self.assertEqual(train_cfg["out_dir"], "results/gym/lunar_lander_slow_fast/student")
        self.assertEqual(train_cfg["adv_weight"], 1.)
        self.assertEqual(train_cfg["safe_fast_weight"], 0.)
        self.assertEqual(train_cfg["imitation_weight"], 0.)
        self.assertEqual(train_cfg["train_scope"], "residual")
        self.assertEqual(train_cfg["residual_scale"], 0.15)
        self.assertEqual(train_cfg["residual_lr"], 0.01)
        with self.assertRaises(ValueError) as caught:
            validate_train({**train_cfg, "adv_weight": 0.})
        self.assertIn("adv_weight=0", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            validate_train({**train_cfg, "safe_fast_weight": 1.})
        self.assertIn("kinematic", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            validate_train({**train_cfg, "train_scope": "control"})
        self.assertIn("residual", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            validate_train({**train_cfg, "residual_scale": 0.20})
        self.assertIn("0.15", str(caught.exception))
        source = Path(train.__code__.co_filename).read_text()
        self.assertNotIn("gym_shaping_cost", source)
        self.assertNotIn("safe_fast_landing", source)
        self.assertNotIn("nearest matched state", source)
        self.assertIn("progress_same_seed", source)
        fast_source = Path(train_fast.__code__.co_filename).read_text()
        self.assertNotIn("safe_fast_landing", fast_source)
        self.assertIn("main_engine * altitude", fast_source)

    def test_cpu_smoke_trains_paired_error_on_fake_pairs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = _tiny_controller(root)
            safe, fast = synthetic_teacher_rollouts()
            built = build_progress_pairs(safe, fast)
            self.assertTrue(np.all(built["arrays"]["slow_seed"] == built["arrays"]["fast_seed"]))
            pairs, _ = save_pairs(root / "pairs_progress.npz", built)
            legacy = root / "pairs.npz"
            legacy.write_bytes(b"stranger")
            manifest = collect({**COLLECT_DEFAULTS, "out": str(root / "smoke.npz"),
                                "live_log": str(root / "collect.log")}, smoke=True)
            self.assertGreater(manifest["rows"], 2)
            self.assertEqual(manifest["crashes_excluded"], 1)
            self.assertEqual(manifest["pairing"], "progress_same_seed")
            self.assertEqual((root / "pairs_stranger_do_not_train.npz").read_bytes(), b"stranger")
            self.assertFalse(legacy.exists())
            cfg = {**TRAIN_DEFAULTS, "steps": 4, "checkpoints": [4], "batch_size": 4, "log_interval": 1,
                   "device": "cpu", "checkpoint": str(checkpoint), "pairs": str(pairs),
                   "out_dir": str(root / "run"), "live_log": str(root / "live.log"),
                   "error_tokens": 4, "error_width": 8, "error_heads": 2}
            named = root / "pairs.npz"
            save_pairs(named, built)
            with self.assertRaises(ValueError) as caught:
                train({**cfg, "pairs": str(named), "out_dir": str(root / "blocked")})
            self.assertIn("stranger pairs", str(caught.exception))
            cross = {key: value.copy() for key, value in built["arrays"].items()}
            cross["fast_seed"] = cross["slow_seed"] + 1
            np.savez_compressed(root / "cross.npz", **cross)
            (root / "cross.json").write_text(json.dumps(built["manifest"]))
            with self.assertRaises(ValueError) as caught:
                train({**cfg, "pairs": str(root / "cross.npz"), "out_dir": str(root / "blocked")})
            self.assertIn("stranger pairs", str(caught.exception))
            fast_summary = train_fast(
                {**FAST_DEFAULTS, "device": "cpu", "checkpoint": str(checkpoint),
                 "out_dir": str(root / "fast"), "live_log": str(root / "fast.log"),
                 "batch_size": 8},
                smoke=True, smoke_steps=2)
            self.assertTrue(fast_summary["weights_moved"])
            self.assertEqual(fast_summary["format"], "gym_fast_teacher_v1")
            fast_text = (root / "fast.log").read_text()
            self.assertIn("[fast-teacher]", fast_text)
            self.assertIn("not a Lunar landing", fast_text)
            self.assertIn("safe_fast_weight=0", fast_text)
            held = load_paired_controller(fast_summary["held"])
            self.assertIsNone(held["residual"])
            self.assertEqual(float(held["config"]["adv_weight"]), 1.)
            self.assertEqual(float(held["config"]["safe_fast_weight"]), 0.)
            self.assertEqual(held["config"]["speed_term"], "main_engine * altitude")
            with self.assertRaises(ValueError) as caught:
                train({**cfg, "checkpoint": fast_summary["held"], "out_dir": str(root / "from_fast")})
            self.assertIn("not the fast teacher", str(caught.exception))
            summary = train(cfg)
            self.assertEqual(summary["adv_weight"], 1.)
            self.assertEqual(summary["safe_fast_weight"], 0.)
            self.assertEqual(summary["l2_aux_weight"], 0.)
            self.assertEqual(summary["simulator_calls"], 0)
            self.assertEqual(summary["b_cap_applications"], 1)
            self.assertEqual(summary["diagnostic_mse"], "outside the loss")
            text = (root / "live.log").read_text()
            self.assertIn("paired-error RpGAN", text)
            self.assertIn("adv_weight=1", text)
            self.assertIn("safe_fast_weight=0", text)
            self.assertIn("diag_action_mse", text)
            self.assertIn("outside the loss", text)
            self.assertIn("kinematic cost", text)
            self.assertIn("RESIDUAL frozen=#18 scale=0.15", text)
            self.assertIn("E_control and G2 are not updated", text)
            row = json.loads((root / "run" / "metrics.jsonl").read_text().splitlines()[-1])
            self.assertEqual(row["adv_weight"], 1.)
            self.assertEqual(row["safe_fast_weight"], 0.)
            self.assertEqual(row["l2_aux_weight"], 0.)
            self.assertNotAlmostEqual(row["loss"], row["action_mse"], places=4)
            before = load_paired_controller(checkpoint)
            bundle = load_paired_controller(root / "run" / "final.pt")
            self.assertIsNone(before["residual"])
            self.assertEqual(bundle["config"]["arm"], "slow_fast")
            self.assertEqual(float(bundle["config"]["adv_weight"]), 1.)
            self.assertEqual(bundle["config"]["train_scope"], "residual")
            self.assertIsNotNone(bundle["residual"])
            self.assertEqual(bundle["residual"].scale, 0.15)
            for left, right in zip(before["E_control"].parameters(), bundle["E_control"].parameters()):
                self.assertTrue(torch.equal(left, right))
            for left, right in zip(before["G"].branches[1].parameters(),
                                   bundle["G"].branches[1].parameters()):
                self.assertTrue(torch.equal(left, right))
            state = np.zeros(8, dtype=np.float32)
            previous = np.array([-1., 0.], dtype=np.float32)
            terrain = np.zeros(11, dtype=np.float32)
            played, _ = playback_action(bundle, state, previous, terrain)
            frozen, _ = control_action_details(bundle, state, previous, terrain)
            self.assertLessEqual(float(np.max(np.abs(played - frozen))), 0.15 + 1e-6)
            self.assertTrue((root / "run" / "live.log").is_symlink())
            replay = load_paired_controller(root / "run" / "checkpoint_4.pt")
            self.assertEqual(sha256(root / "run" / "final.pt"), sha256(root / "run" / "checkpoint_4.pt"))
            self.assertEqual(int(bundle["step"]), int(replay["step"]))


def _tiny_controller(root):
    rng = np.random.default_rng(81)
    states = rng.normal(size=(8, 8)).astype(np.float32)
    states[:, 6:] = rng.integers(0, 2, size=(8, 2))
    actions = rng.uniform(-1, 1, size=(8, 2)).astype(np.float32)
    next_states = states.copy()
    from lib.gym_transition import GymTransitionScaler
    scaler = GymTransitionScaler.fit(np.concatenate([states, actions, next_states], 1))
    world = {**WORLD_DEFAULTS, "width": 8, "encoder_width": 8, "d_width": 8, "marginal_width": 8,
             "z_dim": 4, "num_particles": 8, "device": "cpu"}
    bundle = build_models(world, scaler, "cpu")
    initial = dict(config=world, scaler=scaler.state_dict(), step=1000, validation={}, provenance={})
    initial.update({key: None if bundle[key] is None else bundle[key].state_dict()
                    for key in ("G", "E", "prior", "D", "direct")})
    torch.save(initial, root / "initial.pt")
    loaded = initialize_particle_finetune(root / "initial.pt", "cpu")
    saved = dict(format="gym_particle_finetune_v1", config=dict(TRAIN_DEFAULTS),
                 world_config=loaded["world_config"], scaler=loaded["scaler"].state_dict(),
                 step=2500, provenance={}, validation={})
    saved.update({key: loaded[key].state_dict() for key in MODULE_KEYS})
    path = root / "controller.pt"
    torch.save(saved, path)
    return path


if __name__ == "__main__":
    unittest.main()
