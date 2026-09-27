"""Verify hashes and independently recompute retained API-run summaries.

Uses only the Python standard library; never imports training code or unpickles
checkpoints. External checkpoint references are descriptive, not verified here.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import zipfile


POINT_KEYS = ("step", "modes", "n_modes", "hq", "cover", "effective_modes")


def read(path):
    return json.loads(path.read_text())


def lines(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def quality(point):
    return point["modes"] == 8 and 0.90 <= point["hq"] <= 1.0


def point(row):
    return {key: row[key] for key in POINT_KEYS}


def summarize_window(points, start, end, interval=10):
    wanted = list(range(start, end + 1, interval))
    wanted_set = set(wanted)
    selected = [p for p in points if p["step"] in wanted_set]
    failures = [p["step"] for p in selected if not quality(p)]
    return {"checks": len(selected), "expected_checks": len(wanted),
            "complete": [p["step"] for p in selected] == wanted,
            "passing_checks": len(selected) - len(failures), "failing_steps": failures,
            "min_hq": min((p["hq"] for p in selected), default=None),
            "min_modes": min((p["modes"] for p in selected), default=None)}


def summarize_recovery(points, end, shift):
    selected = [p for p in points if shift < p["step"] <= end]
    passing = [i for i, p in enumerate(selected) if quality(p)]
    first = selected[passing[0]]["step"] if passing else None
    after = selected[passing[0]:] if passing else []
    last_failure = max((i for i, p in enumerate(selected) if not quality(p)), default=-1)
    suffix = selected[last_failure + 1:]
    return {"observed_through": end, "first_passing_step": first,
            "updates_to_first_pass": None if first is None else first - shift,
            "checks_from_first_pass": len(after),
            "passing_checks_from_first_pass": sum(quality(p) for p in after),
            "failing_steps_after_first_pass": [p["step"] for p in after if not quality(p)],
            "stable_suffix_start": suffix[0]["step"] if suffix else None,
            "stable_suffix_checks": len(suffix),
            "stable_suffix_span_updates": suffix[-1]["step"] - suffix[0]["step"] if suffix else 0,
            "stable_suffix_min_hq": min((p["hq"] for p in suffix), default=None),
            "final": selected[-1] if selected else None}


def scheduled_scale(completed, horizon, start, floor):
    progress = max(0.0, min(1.0, (completed - horizon * start) / (horizon * (1.0 - start))))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def verify_arm(folder):
    declaration, result = read(folder / "declaration.json"), read(folder / "result.json")
    recipe = declaration["recipe"]
    steps, shift = recipe["total_steps"], declaration["evaluation"]["shift_after_step"]
    schedule = declaration["schedule"]
    assert schedule in ("constant", "decay") and result["schedule"] == schedule
    assert result["status"] == "COMPLETE" and result["completed_steps"] == steps
    assert declaration["training_budget"] == steps and declaration["seed"] == 0
    assert declaration["evaluation"]["every"] == 10 and shift == 2400
    assert recipe["input_noise_anneal_end"] * steps == 360
    assert recipe["output_noise_warmup"] * steps == 720
    with zipfile.ZipFile(folder / "source.zip") as source:
        assert set(source.namelist()) == set(declaration["source_sha256"])
        for name, expected in declaration["source_sha256"].items():
            assert hashlib.sha256(source.read(name)).hexdigest() == expected, name

    observations = lines(folder / "metrics.jsonl.gz")
    assert [row["step"] for row in observations] == list(range(10, steps + 1, 10))
    live = [point(row) for row in observations]
    frozen = [row["frozen"] for row in observations if "frozen" in row]
    assert [row["step"] for row in frozen] == list(range(shift + 10, steps + 1, 10))
    recomputed = {
        "stationary": summarize_window(live, 1000, 1200, 50),
        "prehold": summarize_window(live, 1210, shift),
        "recovery_at_3600": summarize_recovery(live, min(3600, steps), shift),
        "recovery_extended": summarize_recovery(live, steps, shift),
        "frozen_control": summarize_window(frozen, shift + 10, steps),
        "final": live[-1], "final_ema": observations[-1]["ema"],
    }
    for name, expected in recomputed.items():
        assert result[name] == expected, f"{folder.name}: {name} does not reproduce"

    actual_lrs = lines(folder / "learning-rates.jsonl.gz")
    assert [row["step"] for row in actual_lrs] == list(range(1, steps + 1))
    initial_lrs = {"generator_0": recipe["lr"], "prior_1": recipe["lr"] * recipe["prior_lr_mult"],
                   "critic_0": recipe["lr"] * recipe["d_lr_mult"]}
    horizon = min(steps, recipe["network_lr_horizon_cap"] or steps)
    network_floor = recipe["network_lr_floor"]
    network_floor = recipe["lr_floor"] if network_floor is None else network_floor
    lr_ranges = {}
    for name, initial in initial_lrs.items():
        recorded = [row[name] for row in actual_lrs]
        assert all(math.isfinite(v) and v > 0 for v in recorded)
        lr_ranges[name] = {"min": min([initial] + recorded), "max": max([initial] + recorded),
                           "observed_steps": len(recorded)}
        if schedule == "constant":
            assert all(v == initial for v in recorded), f"{name}: LR is not constant"
        else:
            own_horizon, floor = (steps, recipe["lr_floor"]) if name == "prior_1" else (horizon, network_floor)
            for completed, actual in enumerate(recorded):
                expected = initial * scheduled_scale(completed, own_horizon, recipe["lr_anneal_start"], floor)
                assert math.isclose(actual, expected, rel_tol=1e-14), f"{name}: LR at {completed + 1}"
    assert result["lr_ranges"] == lr_ranges
    assert result["constant_lr_verified_every_step"] == (schedule == "constant")
    if schedule == "constant":
        assert recipe["lr_floor"] == network_floor == 1.0
    for row in observations:
        completed = row["step"] - 1
        expected_rates = {key: value for key, value in actual_lrs[completed].items() if key != "step"}
        assert row["learning_rates"] == expected_rates
        assert row["input_noise"] == recipe["input_noise_std"] * max(0.0, 1.0 - completed / 360)
        assert row["output_noise"] == recipe["output_noise_std"] * min(1.0, completed / 720)
        assert all(math.isfinite(v) for v in row["losses"].values())
        assert row["penalty"]["s"] == row["controller"]["blend_weight"] == 0.5
        assert row["penalty"]["phase"] == ("a" if row["step"] < 800 else "blend")
        for measurement in (row, row["ema"], *([row["frozen"]] if "frozen" in row else [])):
            assert measurement["n_modes"] == 8 and 0 <= measurement["modes"] <= 8
            assert 0 <= measurement["hq"] <= 1 and math.isfinite(measurement["effective_modes"])

    receipts = lines(folder / "state-hashes.jsonl.gz")
    expected_receipts = sorted(set(range(100, steps + 1, 100)) | {steps})
    assert [row["step"] for row in receipts] == expected_receipts
    assert receipts[-1] == result["final_state"]
    initial = read(folder / "initial.json")
    assert initial["step"] == 0
    return schedule, declaration, initial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = read(root / "manifest.json")
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        assert root in path.parents, f"artifact outside report: {name}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"hash mismatch: {name}"
    folders = sorted({(root / name).parent for name in manifest["files"] if name.endswith("/declaration.json")})
    assert folders, "no experiment artifacts declared"
    arms = {}
    for folder in folders:
        for name in ("declaration.json", "result.json", "initial.json", "source.zip",
                     "metrics.jsonl.gz", "learning-rates.jsonl.gz", "state-hashes.jsonl.gz"):
            assert str((folder / name).relative_to(root)) in manifest["files"], f"unhashed evidence: {folder / name}"
        schedule, declaration, initial = verify_arm(folder)
        assert schedule not in arms, f"duplicate {schedule} arm"
        arms[schedule] = (declaration, initial)
    assert "constant" in arms
    if "decay" in arms:
        constant, decay = (arms[name][0] for name in ("constant", "decay"))
        ignored = {"lr_floor", "network_lr_floor"}
        assert {k: v for k, v in constant["recipe"].items() if k not in ignored} == {
            k: v for k, v in decay["recipe"].items() if k not in ignored}
        for key in ("model", "optimizer_options", "generator_real", "evaluation", "source_sha256"):
            assert constant[key] == decay[key], f"unmatched declaration: {key}"
        a, b = (arms[name][1] for name in ("constant", "decay"))
        # Full trainer hashes include the deliberately different recipe floors.
        assert {k: v for k, v in a.items() if k != "trainer_sha256"} == {
            k: v for k, v in b.items() if k != "trainer_sha256"}, "unmatched initial models/optimizer/RNG"
    print(f"PASS: {len(manifest['files'])} artifact hashes; {', '.join(sorted(arms))} summaries, "
          "source identities, every-step learning rates, and retained state receipts verified.")


if __name__ == "__main__":
    main()
