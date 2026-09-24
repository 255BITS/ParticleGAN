"""Replay the two cold hosts against the exact PR #84 40ec source bytes."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.toy100.warm_equilibrium_probe import _feed_hash
from benchmarks.transfer_suite.compare_defaults import plan
from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
from benchmarks.transfer_suite.protocol import test_verdict
from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy, declared_recipe
from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate

SOURCE = ROOT / "reports/toy100/continuous-evidence/pr84-smoothed-candidate/original_40ec.py"
SOURCE_SHA = "4bcf47e0b806b935d3a3dd52079476d89ebe72f8f188095206085a807a0ce2dc"


def _original_context(task):
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
        raise RuntimeError("original PR #84 source bytes changed")
    spec = importlib.util.spec_from_file_location("pr84_original_40ec", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.alternating_curvature(task=task, curvature_bound=.25, bound_d=True,
                                        d_curvature_bound=3., smooth_critic=True)


def _state_sha(recorder):
    local = recorder._local
    policy = local["noise_policy"]
    stream = local.get("stream")
    values = dict(
        generator=local["generator"].state_dict(),
        critic=local["critic"].state_dict(),
        prior=local["prior"].state_dict(),
        opt_g=local["opt_g"].state_dict(),
        opt_d=local["opt_d"].state_dict(),
        ema_g=local.get("ema_g"),
        ema_z=local.get("ema_z"),
        data_stream=None if stream is None else stream.get_state(),
        torch_rng=torch.get_rng_state(),
        input_stream=policy.input_stream.get_state(),
        output_stream=None if policy.output_stream is None else policy.output_stream.get_state(),
        input_sigma=policy.input_sigma,
        output_sigma=policy.output_sigma,
    )
    digest = hashlib.sha256()
    _feed_hash(digest, values)
    return digest.hexdigest()


def _without_runtime_timing(value):
    if isinstance(value, dict):
        return {key: _without_runtime_timing(row) for key, row in value.items()
                if "seconds" not in key}
    if isinstance(value, list):
        return [_without_runtime_timing(row) for row in value]
    return value


def _run(task, context, recipe, noise, model_policy):
    spec = next(job["spec"] for job in plan() if job["spec"]["name"] == task)
    with context as (recorder, _):
        result, details = run_legacy(spec, recipe, noise, model_policy=model_policy)
    return dict(result=result, context=details, recorder=recorder,
                verdict=test_verdict(spec, result), state_sha256=_state_sha(recorder))


def _compare_records(a, b):
    if len(a) != len(b):
        raise AssertionError(f"record count differs: {len(a)} vs {len(b)}")
    fields = ("outer_step", "critic_advantage", "critic_sharpness", "critic_width")
    role_fields = ("rho", "factor")
    for step, (left, right) in enumerate(zip(a, b), 1):
        for field in fields:
            if left.get(field) != right.get(field):
                raise AssertionError(f"step {step}: {field} differs")
        for role in ("d", "g"):
            for field in role_fields:
                if left[role][field] != right[role][field]:
                    raise AssertionError(f"step {step}: {role}.{field} differs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    config.update(name="pr84_smoothed_parity", lr_floor=1., lr_anneal_start=0.)
    config.pop("network_lr_horizon_cap")
    config.pop("network_lr_floor")
    recipe, noise, _ = declared_recipe(config)
    model_policy = declared_model_policy(config)
    rows = []
    for task in ("trajectory", "mode_hold"):
        original = _run(task, _original_context(task), recipe, noise, model_policy)
        cleaned = _run(task, pr84_smoothed_candidate(task=task), recipe, noise, model_policy)
        if original["state_sha256"] != cleaned["state_sha256"]:
            raise AssertionError(f"{task}: weights, moments, EMA, or training RNG differ")
        if _without_runtime_timing(original["result"]) != _without_runtime_timing(cleaned["result"]):
            raise AssertionError(f"{task}: host metrics differ")
        if original["context"]["applied"] != cleaned["context"]["applied"]:
            raise AssertionError(f"{task}: optimizer rates differ")
        if original["context"]["noise_receipt"] != cleaned["context"]["noise_receipt"]:
            raise AssertionError(f"{task}: noise receipt differs")
        old, new = original["recorder"], cleaned["recorder"]
        if old.rng_replay_verified != new.rng_replay_verified:
            raise AssertionError(f"{task}: RNG replay count differs")
        if old.host_source != new.host_source:
            raise AssertionError(f"{task}: generated host source differs")
        _compare_records(old.records, new.records)
        row = dict(task=task, steps=old.outer_steps,
                   original_source_sha256=SOURCE_SHA,
                   candidate_source_sha256=hashlib.sha256(
                       (ROOT / "reports/toy100/pr84_smoothed_candidate.py").read_bytes()).hexdigest(),
                   state_sha256=original["state_sha256"],
                   generated_host_source=old.host_source,
                   verdict=original["verdict"], live=original["result"]["live"],
                   terminal_observations=[dict(step=item["step"], modes=item.get("modes"),
                                               hq=item.get("hq"))
                                          for item in original["result"]["observations"]
                                          if item["step"] >= 1000] if task == "mode_hold" else [],
                   applied=original["context"]["applied"],
                   rng_replay_verified=old.rng_replay_verified,
                   exact_state_metrics_noise_and_records=True)
        rows.append(row)
        print(json.dumps(dict(event="HOST_PARITY", **row)), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(status="PASS", rows=rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
