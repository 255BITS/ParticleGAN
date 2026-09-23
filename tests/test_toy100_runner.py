"""Runner invariants that affect the credibility of checkpoint curves."""

import json

import numpy as np

from benchmarks.toy100.train import evaluation_steps, load_config, resolve_config, train


def test_initial_and_final_samples_do_not_depend_on_observation_schedule(tmp_path):
    base = {
        "problem": "grid100", "steps": 3, "seed": 17, "device": "cpu",
        "num_particles": 128, "batch_size": 16,
        "g_hidden": 16, "d_hidden": 16, "n_hidden": 1,
        "eval_samples": 1024, "snapshot_samples": 64,
        "eval_interval": 3, "snapshot_interval": 3,
        "log_interval": 3, "threads": 1,
    }
    sparse = tmp_path / "sparse"
    dense = tmp_path / "dense"
    sparse_summary = train({**base, "early_eval_steps": [0, 3]}, sparse)
    dense_summary = train({**base, "early_eval_steps": [0, 1, 2, 3]}, dense)
    assert sparse_summary["status"] == dense_summary["status"] == "complete"
    assert sparse_summary["completed_steps"] == dense_summary["completed_steps"] == 3

    for step in (0, 3):
        filename = f"step_{step:06d}.npz"
        with np.load(sparse / "snapshots" / filename) as left, np.load(dense / "snapshots" / filename) as right:
            for key in ("live", "ema", "target"):
                np.testing.assert_array_equal(left[key], right[key])

    events = [json.loads(line) for line in (dense / "events.jsonl").read_text().splitlines()]
    assert [(row["event"], row["step"], row.get("model")) for row in events[:2]] == [
        ("eval", 0, "live"), ("eval", 0, "ema"),
    ]
    assert [row["step"] for row in events if row["event"] == "train"] == [1, 2, 3]


def test_evaluation_schedule_keeps_initial_early_periodic_and_final():
    assert evaluation_steps(620, 250) == [0, 1, 10, 25, 50, 100, 250, 500, 620]


def test_json_and_toml_custom_recipe_labels_resolve(tmp_path):
    json_path = tmp_path / "recipe.json"
    toml_path = tmp_path / "recipe.toml"
    json_path.write_text('{"name":"toy100_test","steps":12,"lr":0.001}')
    toml_path.write_text('name = "toy100_test"\nsteps = 12\nlr = 0.001\n')
    for path in (json_path, toml_path):
        config, recipe = resolve_config(load_config(path))
        assert (config["name"], config["steps"], recipe.total_steps, recipe.lr) == (
            "toy100_test", 12, 12, 0.001,
        )
