"""Run or independently regrade the 100-mode + 19-toy common-recipe gate.

``run`` creates fresh evidence for all three 100-mode problems and all 19
canonical transfer cases. ``regrade`` checks their saved evidence without
training. A separate installed-wheel public-default replay is an optional
control. A common 22/22 PASS requires one identical global recipe *including*
noise on every host, as well as each host's frozen live gate. Candidate runs
without complete training-noise receipts remain INCOMPLETE for that claim.

python -u -m benchmarks.toy_suite run \
    --config configs/toy100/shared_candidate.json --output /tmp/toy-suite-22
python -m benchmarks.toy_suite regrade --output /tmp/toy-suite-22
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import struct
import sys
import tarfile

from benchmarks.toy100.accuracy_gate import (
    HOLDOUT_SEED_OFFSETS, evaluate_suite as accuracy_suite,
)
from benchmarks.toy100.gate import evaluate_suite as coverage_suite
from benchmarks.toy100.problems import PROBLEM_NAMES
from benchmarks.toy100.models import linear_input_noise
from benchmarks.toy100.train import (
    AFFINE_MODEL_POLICIES, EMPIRICAL_INIT_SEED_OFFSET,
    POLICY_SOURCE_SCOPE_V2, policy_source_scope, resolve_config,
)
from benchmarks.transfer_suite.compare_defaults import plan
from benchmarks.transfer_suite.legacy_noise_adapters import EVAL_SCOPES
from benchmarks.transfer_suite.protocol import test_verdict
from benchmarks.transfer_suite.public_default_verification import (
    GLOBAL_RECIPE_FIELDS, declared_spec, host_recipe, load_declaration,
)
from benchmarks.transfer_suite.toy100_compatibility import (
    VECTOR_NAMES, declared_model_policy, declared_recipe, output_noise_at,
)
from particlegan import Recipe, learning_rate_scale


ROOT = Path(__file__).resolve().parents[1]
NOISE_FIELDS = ("output_noise_std", "input_noise_std",
                "input_noise_anneal_end", "output_noise_warmup")
NATIVE_SOURCE_FILES = (
    "benchmarks/toy100/train.py", "benchmarks/toy100/models.py",
    "benchmarks/toy100/problems.py", "benchmarks/toy100/metrics.py",
    "benchmarks/toy100/accuracy.py", "benchmarks/toy100/accuracy_evidence.py",
    "benchmarks/toy100/accuracy_gate.py", "lib/toy_models.py",
    "particlegan/training.py", "particlegan/recipes.py",
)
OUTPUT_NOISE_SEED_OFFSET = 1901
ISOLATED_TRANSFER_RECEIPT_FIELDS = (
    "output_noise_rng", "output_noise_seed_offset", "output_noise_seed",
    "output_noise_training_stream_isolated",
    "output_noise_train_state_initial_sha256", "output_noise_train_state_final_sha256",
    "output_noise_eval_state_pairs", "output_noise_eval_state_preserved",
)


def _noise_identity(noise: dict) -> dict:
    """Validate shared noise fields while preserving absent historical options."""
    if not isinstance(noise, dict):
        raise ValueError("noise declaration is not an object")
    learned = noise.get("output_noise_learnable", False)
    if type(learned) is not bool:
        raise ValueError("output_noise_learnable must be a boolean")
    if learned and (not _positive_scale(noise.get("output_noise_std"))):
        raise ValueError("learnable output noise requires a positive peak")
    if "output_noise_rng" in noise:
        if noise["output_noise_rng"] != "isolated":
            raise ValueError("output_noise_rng must be 'isolated' when declared")
        if not _positive_scale(noise.get("output_noise_std")):
            raise ValueError("isolated output_noise_rng requires a positive peak")
    return {**noise, "output_noise_warmup": noise.get("output_noise_warmup", 0.0),
            "output_noise_learnable": learned}


def _read(path: Path):
    return json.loads(path.read_text())


def _write(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _status(passed: int, required: int, *, complete: bool) -> str:
    return "INCOMPLETE" if not complete else "PASS" if passed == required else "FAIL"


def _json_value(value):
    """Compare resolved tuples and archived JSON with the same representation."""
    return json.loads(json.dumps(value, allow_nan=False))


def _close(actual, expected):
    return (isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14))


def _scale_close(actual, expected):
    """Allow float32 softplus roundoff while still binding scalar receipts."""
    return (isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-7))


def _positive_scale(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def _sha256_hex(value):
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _check_optimizer_receipts(record: dict, base: Recipe):
    receipts = record.get("applied")
    if not isinstance(receipts, list) or not receipts:
        raise ValueError(f"optimizer receipts are absent: {record['name']}")
    roles = [item.get("role") for item in receipts]
    if len(set(roles)) != len(roles) or "d" not in roles or not {"g", "prior"}.intersection(roles):
        raise ValueError(f"optimizer roles differ: {record['name']}")
    if record["spec"]["runner"] != "legacy" and roles != ["g", "prior", "d"]:
        raise ValueError(f"trainer optimizer roles differ: {record['name']}")
    rates = dict(g=base.lr, prior=base.lr * base.prior_lr_mult,
                 d=base.lr * base.d_lr_mult)
    for item in receipts:
        role = item["role"]
        if role not in rates or not _close(item.get("lr"), rates[role]):
            raise ValueError(f"optimizer rate differs from common recipe: {record['name']}.{role}")
        betas = base.prior_betas or base.betas if role == "prior" else base.betas
        if item.get("betas") != list(betas):
            raise ValueError(f"optimizer betas differ from common recipe: {record['name']}.{role}")
        if type(item.get("parameters")) is not int or item["parameters"] <= 0:
            raise ValueError(f"optimizer parameter receipt is invalid: {record['name']}.{role}")
        if record["spec"]["runner"] != "legacy" and item.get("optimizer") != "Adam":
            raise ValueError(f"trainer optimizer type differs: {record['name']}.{role}")


def _check_actions(record: dict, base: Recipe, noise: dict | None,
                   model_policy: dict | None = None):
    spec, result = record["spec"], record["result"]
    name, steps = spec["name"], spec["steps"]
    actions = result.get("actions", [])
    cap = (model_policy or {}).get("network_lr_horizon_cap")
    network_floor = (model_policy or {}).get("network_lr_floor")
    if spec["runner"] == "legacy":
        expected = [(step, role) for step in (
                    range(steps) if cap is not None else range(0, steps, 20))
                    for role in ("d", "g")]
        if [(item.get("step"), item.get("role")) for item in actions] != expected:
            raise ValueError(f"custom-host optimizer trace is incomplete: {name}")
        for item in actions:
            if cap is None:
                scale = learning_rate_scale(item["step"], steps,
                                            base.lr_anneal_start, base.lr_floor)
                if not _close(item.get("multiplier"), scale):
                    raise ValueError(f"custom-host LR schedule differs from common recipe: {name}")
                continue
            from benchmarks.toy100.schedule import policy_multipliers
            network, prior = policy_multipliers(
                item["step"], steps, base.lr_anneal_start, base.lr_floor, cap,
                network_lr_floor=network_floor,
            )
            if (item.get("network_lr_horizon_cap") != cap
                    or not _close(item.get("multiplier"), network)
                    or not _close(item.get("network_multiplier"), network)
                    or not _close(item.get("prior_multiplier"), prior)):
                raise ValueError(f"custom-host network horizon differs: {name}")
            if (network_floor is not None and not _close(
                    item.get("network_lr_floor"), network_floor)) or (
                    network_floor is None and "network_lr_floor" in item):
                raise ValueError(f"custom-host network floor differs: {name}")
            expected_roles = ({"d"} if item["role"] == "d" else
                              {row["role"] for row in record["applied"] if row["role"] != "d"})
            actual_groups = item.get("group_lrs")
            if (not isinstance(actual_groups, list)
                    or not all(isinstance(row, dict) for row in actual_groups)
                    or len(actual_groups) != len(expected_roles)
                    or {row.get("role") for row in actual_groups} != expected_roles):
                raise ValueError(f"custom-host optimizer group receipt differs: {name}")
            for group in actual_groups:
                role = group["role"]
                rate = base.lr * {"g": 1.0, "d": base.d_lr_mult,
                                  "prior": base.prior_lr_mult}[role]
                scale = prior if role == "prior" else network
                if not _close(group.get("lr"), rate * scale):
                    raise ValueError(f"custom-host optimizer rate differs: {name}.{role}")
        return
    if len(actions) != steps or result.get("update_counts") != {"g": steps, "d": steps}:
        raise ValueError(f"trainer action or update trace is incomplete: {name}")
    for completed, action in enumerate(actions, start=1):
        if cap is None:
            network = prior = learning_rate_scale(
                completed - 1, steps, base.lr_anneal_start, base.lr_floor,
            )
            expected_rates = dict(step=completed, multiplier=network)
        else:
            from benchmarks.toy100.schedule import policy_multipliers
            network, prior = policy_multipliers(
                completed - 1, steps, base.lr_anneal_start, base.lr_floor, cap,
                network_lr_floor=network_floor,
            )
            expected_rates = dict(step=completed, network_multiplier=network,
                                  prior_multiplier=prior,
                                  network_lr_horizon_cap=cap)
            if network_floor is not None:
                expected_rates["network_lr_floor"] = network_floor
        expected_rates.update(lr_g=base.lr * network,
                              lr_prior=base.lr * base.prior_lr_mult * prior,
                              lr_d=base.lr * base.d_lr_mult * network)
        if any(not _close(action.get(key), value) for key, value in expected_rates.items()):
            raise ValueError(f"trainer LR action differs from common recipe: {name}.{completed}")
        if network_floor is None and "network_lr_floor" in action:
            raise ValueError(f"undeclared trainer network floor: {name}.{completed}")
        if noise is not None:
            input_sigma = linear_input_noise(
                noise["input_noise_std"], completed - 1, steps,
                noise["input_noise_anneal_end"],
            )
            output_sigma = output_noise_at(
                noise["output_noise_std"], completed - 1, steps,
                noise.get("output_noise_warmup", 0.0),
            )
            if not _close(action.get("input_sigma"), input_sigma):
                raise ValueError(f"trainer input noise schedule differs: {name}.{completed}")
            if ("output_noise_warmup" in noise
                    and not _close(action.get("output_sigma"), output_sigma)):
                raise ValueError(f"trainer output warmup differs: {name}.{completed}")


def _check_learned_transfer_noise(record: dict, receipt: dict, noise: dict):
    """Bind the learned scalar, its optimizer ownership, and its actual sigma."""
    name = record["name"]
    learned = _noise_identity(noise)["output_noise_learnable"]
    reported = receipt.get("output_noise_learnable", False)
    if type(reported) is not bool or reported != learned:
        raise ValueError(f"learnable-noise flag differs from common recipe: {name}")
    if not learned:
        if receipt.get("output_scale_parameter_count", 0) != 0:
            raise ValueError(f"fixed-noise host has a learned output parameter: {name}")
        return

    peak = noise["output_noise_std"]
    steps = record["spec"]["steps"]
    warmup = noise.get("output_noise_warmup", 0.0)
    if (receipt.get("output_scale_parameter_count") != 1
            or receipt.get("output_scale_optimizer_owned") is not True):
        raise ValueError(f"learned output scale lacks one G-owned parameter: {name}")
    base_count = receipt.get("generator_base_parameters")
    total_count = receipt.get("generator_total_parameters")
    if (type(base_count) is not int or base_count < 1
            or type(total_count) is not int or total_count != base_count + 1):
        raise ValueError(f"learned output wrapper parameter count differs: {name}")
    if record["spec"]["runner"] != "legacy":
        from benchmarks.transfer_suite import image_tasks, vector_tasks
        from lib.toy_models import SimpleMLPGenerator
        import torch

        with torch.random.fork_rng(devices=[]):
            if record["spec"]["runner"] == "vector":
                cfg = vector_tasks.resolve(record["spec"])
                bare = SimpleMLPGenerator(cfg["z_dim"], cfg["hidden"], cfg["layers"], 2)
            else:
                bare = image_tasks.Generator(record["spec"])
        expected_base = sum(parameter.numel() for parameter in bare.parameters())
        g_receipts = [item for item in record["applied"] if item.get("role") == "g"]
        if (len(g_receipts) != 1 or g_receipts[0].get("parameters") != total_count
                or record.get("shapes", {}).get("generator_parameters") != total_count
                or base_count != expected_base):
            raise ValueError(f"native generator optimizer or shape count differs: {name}")

    initial_scale = receipt.get("output_scale_initial")
    final_scale = receipt.get("output_scale_final")
    if (not _scale_close(initial_scale, peak) or not _positive_scale(final_scale)):
        raise ValueError(f"learned output scale is invalid: {name}")
    trace = receipt.get("output_sigma_effective_step_trace")
    if not isinstance(trace, list) or len(trace) != steps:
        raise ValueError(f"learned output sigma trace is incomplete: {name}")
    first_base = output_noise_at(peak, 0, steps, warmup)
    last_base = output_noise_at(peak, steps - 1, steps, warmup)
    final_base = output_noise_at(peak, steps, steps, warmup)
    if (not _close(receipt.get("output_sigma_first"), first_base)
            or not _close(receipt.get("output_sigma_last"), last_base)
            or not _close(receipt.get("output_sigma_final_evaluation"), final_base)
            or receipt.get("output_nonzero_steps") != sum(
                output_noise_at(peak, step, steps, warmup) > 0
                for step in range(steps)
            )):
        raise ValueError(f"learned output base schedule differs: {name}")
    for completed, effective in enumerate(trace):
        base_sigma = output_noise_at(peak, completed, steps, warmup)
        if not (_scale_close(effective, 0.0) if base_sigma == 0
                else _positive_scale(effective)):
            raise ValueError(f"learned output sigma is invalid: {name}.{completed}")
        if record["spec"]["runner"] != "legacy":
            action = record["result"]["actions"][completed]
            if not _scale_close(action.get("output_sigma_effective"), effective):
                raise ValueError(f"learned output sigma action differs: {name}.{completed}")
    if (not _scale_close(trace[0], first_base / peak * initial_scale)
            or not _scale_close(receipt.get("output_sigma_effective_first"), trace[0])
            or not _scale_close(receipt.get("output_sigma_effective_last"), trace[-1])
            or not _scale_close(receipt.get("output_sigma_effective_final_evaluation"),
                                final_base / peak * final_scale)):
        raise ValueError(f"learned output sigma endpoint differs: {name}")
    ema_scale = receipt.get("output_scale_ema_final")
    ema_effective = receipt.get("output_sigma_effective_ema_final_evaluation")
    if record["spec"]["runner"] != "legacy" or ema_scale is not None or ema_effective is not None:
        if (not _positive_scale(ema_scale)
                or not _scale_close(ema_effective, final_base / peak * ema_scale)):
            raise ValueError(f"learned EMA output sigma differs: {name}")
    if record["spec"]["runner"] != "legacy":
        last = record["result"]["observations"][-1]
        if (not _scale_close(last.get("output_sigma_live"),
                             receipt["output_sigma_effective_final_evaluation"])
                or not _scale_close(last.get("output_sigma_ema"), ema_effective)):
            raise ValueError(f"learned output evaluation differs from receipt: {name}")


def _check_isolated_transfer_noise(record: dict, receipt: dict, noise: dict):
    """Bind the private output stream and prove each scoped eval restores it."""
    name = record["name"]
    isolated = noise.get("output_noise_rng") == "isolated"
    if not isolated:
        if any(field in receipt for field in ISOLATED_TRANSFER_RECEIPT_FIELDS):
            raise ValueError(f"undeclared isolated output-noise receipt: {name}")
        return
    if (receipt.get("output_noise_rng") != "isolated"
            or receipt.get("output_noise_seed_offset") != OUTPUT_NOISE_SEED_OFFSET
            or receipt.get("output_noise_seed") != OUTPUT_NOISE_SEED_OFFSET
            or receipt.get("output_noise_training_stream_isolated") is not True):
        raise ValueError(f"isolated output-noise stream differs: {name}")
    initial = receipt.get("output_noise_train_state_initial_sha256")
    final = receipt.get("output_noise_train_state_final_sha256")
    if not (_sha256_hex(initial) and _sha256_hex(final) and initial != final):
        raise ValueError(f"isolated output-noise train state differs: {name}")
    train_calls, train_elements = (receipt.get("output_train_calls"),
                                   receipt.get("output_train_elements"))
    eval_calls, eval_elements = (receipt.get("output_eval_calls"),
                                 receipt.get("output_eval_elements"))
    if (type(train_calls) is not int or train_calls <= 0
            or type(train_elements) is not int or train_elements <= 0
            or type(eval_calls) is not int or eval_calls < 0
            or type(eval_elements) is not int or eval_elements < 0
            or (eval_calls == 0) != (eval_elements == 0)):
        raise ValueError(f"isolated output-noise draw counts differ: {name}")
    pairs = receipt.get("output_noise_eval_state_pairs")
    if (not isinstance(pairs, list)
            or receipt.get("output_noise_eval_state_preserved") is not True
            or (eval_calls > 0 and not pairs)):
        raise ValueError(f"isolated output-noise evaluation receipt differs: {name}")
    legacy = record["spec"]["runner"] == "legacy"
    if legacy and receipt.get("eval_scope") != EVAL_SCOPES[name]:
        raise ValueError(f"isolated output-noise evaluation scope differs: {name}")
    if (not legacy or EVAL_SCOPES[name] in (
            "generated_samples", "generated_and_reconstructed_samples")) and eval_calls == 0:
        raise ValueError(f"isolated output-noise evaluation draws are absent: {name}")
    required = (("before_sha256", "after_sha256") if legacy else
                ("live_before_sha256", "live_after_sha256",
                 "ema_before_sha256", "ema_after_sha256"))
    steps = record["spec"]["steps"]
    pair_steps = []
    for pair in pairs:
        if (not isinstance(pair, dict) or type(pair.get("step")) is not int
                or not 0 <= pair["step"] <= steps
                or any(not _sha256_hex(pair.get(field)) for field in required)):
            raise ValueError(f"isolated output-noise evaluation state is invalid: {name}")
        pair_steps.append(pair["step"])
        if (pair[required[0]] != pair[required[1]]
                or not legacy and pair[required[2]] != pair[required[3]]):
            raise ValueError(f"isolated output-noise evaluation advanced training stream: {name}")
    if pair_steps != sorted(pair_steps):
        raise ValueError(f"isolated output-noise evaluation trace is out of order: {name}")
    expected_steps = sorted({math.ceil(i * steps / 24) for i in range(1, 25)})
    if legacy:
        if (EVAL_SCOPES[name] in ("generated_samples", "generated_and_reconstructed_samples")
                and not set(expected_steps) <= set(pair_steps)):
            raise ValueError(f"isolated output-noise evaluation trace is incomplete: {name}")
    else:
        if pair_steps != expected_steps or eval_calls == 0:
            raise ValueError(f"isolated output-noise evaluation trace is incomplete: {name}")


def _check_toy100_learned_noise(directory: Path, summary: dict, config: dict,
                                *, policy_archive_verified: bool = False):
    """Audit the native scalar and its post-update/evaluation noise evidence.

    Train events record the effective sigma after that update. The update
    itself used the preceding completed-step schedule value.
    """
    if not config.get("output_noise_learnable", False):
        if summary.get("learnable_output_noise") is not None:
            raise ValueError("fixed-noise 100-mode run has a learned-scale receipt")
        return
    name = config["problem"]
    peak, steps = config["output_noise_std"], config["steps"]
    warmup = config.get("output_noise_warmup", 0.0)
    receipt = summary.get("learnable_output_noise")
    if not isinstance(receipt, dict):
        raise ValueError(f"learned output-scale receipt is absent: {name}")
    if (not _close(receipt.get("initial_std"), peak)
            or receipt.get("added_trainable_parameters") != 1
            or receipt.get("parameter") != "G.output_scale.raw_scale"
            or receipt.get("optimizer") != "G"
            or receipt.get("optimizer_group") != 0):
        raise ValueError(f"learned output scale is not bound to G optimizer: {name}")
    initial = output_noise_at(peak, 0, steps, warmup)
    final = output_noise_at(peak, steps, steps, warmup)
    for model in ("live", "ema"):
        if (not _scale_close(receipt.get(f"initial_output_sigma_{model}"), initial)
                or not _positive_scale(receipt.get(f"final_base_std_{model}"))
                or not _scale_close(receipt.get(f"final_output_sigma_{model}"),
                                    final / peak * receipt[f"final_base_std_{model}"])):
            raise ValueError(f"learned 100-mode output sigma endpoint differs: {name}.{model}")
    train_events, eval_events = {}, {}
    for line in (directory / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event.get("event") == "train":
            step = event.get("step")
            if step in train_events:
                raise ValueError(f"duplicate learned training sigma: {name}.{step}")
            train_events[step] = event
        if event.get("event") == "eval" and event.get("model") in ("live", "ema"):
            key = (event.get("step"), event["model"])
            if key in eval_events:
                raise ValueError(f"duplicate learned evaluation sigma: {name}.{key}")
            eval_events[key] = event
    if set(train_events) != set(range(1, steps + 1)):
        raise ValueError(f"learned training sigma trace is incomplete: {name}")
    if set(eval_events) != {(step, model) for step in summary["eval_steps"]
                           for model in ("live", "ema")}:
        raise ValueError(f"learned evaluation sigma trace is incomplete: {name}")
    for step, event in train_events.items():
        base_sigma = output_noise_at(peak, step, steps, warmup)
        for model in ("live", "ema"):
            value = event.get(f"output_sigma_{model}")
            if not (_scale_close(value, 0.0) if base_sigma == 0
                    else _positive_scale(value)):
                raise ValueError(f"invalid learned training sigma: {name}.{step}.{model}")
    for (step, model), event in eval_events.items():
        base_sigma = output_noise_at(peak, step, steps, warmup)
        value = event.get("output_sigma")
        if not (_scale_close(value, 0.0) if base_sigma == 0 else _positive_scale(value)):
            raise ValueError(f"invalid learned evaluation sigma: {name}.{step}.{model}")
        if step > 0 and not _scale_close(
            value, train_events[step][f"output_sigma_{model}"],
        ):
            raise ValueError(f"learned evaluation sigma differs from training: {name}.{step}.{model}")
        if step == steps and not _scale_close(
            value, receipt[f"final_output_sigma_{model}"],
        ):
            raise ValueError(f"learned final evaluation sigma differs: {name}.{model}")
    provenance = _read(directory / "provenance.json")
    if (summary.get("provenance") != provenance
            or not set(NATIVE_SOURCE_FILES) <= set(provenance.get("source_sha256", {}))):
        raise ValueError(f"learned 100-mode source provenance differs: {name}")
    if not policy_archive_verified:
        for source in NATIVE_SOURCE_FILES:
            if provenance["source_sha256"][source] != hashlib.sha256(
                (ROOT / source).read_bytes(),
            ).hexdigest():
                raise ValueError(f"learned 100-mode source hash differs: {name}.{source}")


def _check_toy100_output_rng(directory: Path, summary: dict, config: dict):
    """Audit the private output stream through training, evaluation, and holdout."""
    name = config["problem"]
    isolated = config.get("output_noise_rng") == "isolated"
    if not isolated:
        if ("output_noise_rng" in summary or "output_noise_rng_receipt" in summary):
            raise ValueError(f"undeclared isolated output-noise receipt: {name}")
        return
    receipt = summary.get("output_noise_rng_receipt")
    if (summary.get("output_noise_rng") != "isolated"
            or not isinstance(receipt, dict)
            or receipt.get("mode") != "isolated"
            or receipt.get("namespace_offset") != OUTPUT_NOISE_SEED_OFFSET
            or receipt.get("training_seed") != config["seed"] + OUTPUT_NOISE_SEED_OFFSET
            or receipt.get("generator_wrapper_class") != "IsolatedOutputNoise"
            or receipt.get("discriminator_wrapper_class") != (
                "StatefulInputNoise" if config["input_noise_std"] else None)
            or receipt.get("checkpoint_state_key") != "_extra_state"):
        raise ValueError(f"100-mode isolated output-noise policy differs: {name}")
    for field in ("initial_live_state_sha256", "initial_ema_state_sha256",
                  "final_live_state_sha256", "final_ema_state_sha256"):
        if not _sha256_hex(receipt.get(field)):
            raise ValueError(f"100-mode isolated output-noise state is invalid: {name}.{field}")
    if receipt["initial_live_state_sha256"] != receipt["initial_ema_state_sha256"]:
        raise ValueError(f"100-mode isolated output-noise initial streams differ: {name}")
    if config["input_noise_std"]:
        if not (_sha256_hex(receipt.get("initial_input_state_sha256"))
                and _sha256_hex(receipt.get("final_input_state_sha256"))):
            raise ValueError(f"100-mode isolated input-noise state is invalid: {name}")
    elif (receipt.get("initial_input_state_sha256") is not None
          or receipt.get("final_input_state_sha256") is not None):
        raise ValueError(f"undeclared 100-mode isolated input-noise state: {name}")

    train_events, eval_events = {}, {}
    for line in (directory / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event.get("event") == "train":
            step = event.get("step")
            if step in train_events:
                raise ValueError(f"duplicate isolated output-noise training step: {name}.{step}")
            train_events[step] = event
        elif event.get("event") == "eval" and event.get("model") in ("live", "ema"):
            key = (event.get("step"), event["model"])
            if key in eval_events:
                raise ValueError(f"duplicate isolated output-noise evaluation: {name}.{key}")
            eval_events[key] = event
    steps = config["steps"]
    if set(train_events) != set(range(1, steps + 1)):
        raise ValueError(f"100-mode isolated output-noise training trace is incomplete: {name}")
    expected_evals = {(step, model) for step in summary["eval_steps"]
                      for model in ("live", "ema")}
    if set(eval_events) != expected_evals or not summary["eval_steps"]:
        raise ValueError(f"100-mode isolated output-noise evaluation trace is incomplete: {name}")
    previous_calls = previous_elements = 0
    for step in range(1, steps + 1):
        event = train_events[step]
        calls = event.get("output_noise_rng_draw_calls")
        elements = event.get("output_noise_rng_draw_elements")
        if (not _sha256_hex(event.get("output_noise_rng_state_sha256"))
                or type(calls) is not int or calls < previous_calls
                or type(elements) is not int or elements < previous_elements
                or (calls == 0) != (elements == 0)):
            raise ValueError(f"100-mode isolated output-noise train stream differs: {name}.{step}")
        previous_calls, previous_elements = calls, elements
    last = train_events[steps]
    if (previous_calls <= 0 or previous_elements <= 0
            or receipt.get("final_live_draw_calls") != previous_calls
            or receipt.get("final_live_draw_elements") != previous_elements
            or receipt["final_live_state_sha256"] != last["output_noise_rng_state_sha256"]):
        raise ValueError(f"100-mode isolated output-noise endpoint differs: {name}")

    for (step, model), event in eval_events.items():
        state_before = event.get("output_noise_rng_state_before_sha256")
        state_after = event.get("output_noise_rng_state_after_sha256")
        calls_before = event.get("output_noise_rng_draw_calls_before")
        calls_after = event.get("output_noise_rng_draw_calls_after")
        elements_before = event.get("output_noise_rng_draw_elements_before")
        elements_after = event.get("output_noise_rng_draw_elements_after")
        if (event.get("output_noise_rng_eval_seed") != config["seed"] + 402
                or not _sha256_hex(state_before) or state_before != state_after
                or type(calls_before) is not int or calls_before < 0
                or calls_before != calls_after
                or type(elements_before) is not int or elements_before < 0
                or elements_before != elements_after):
            raise ValueError(f"100-mode isolated output-noise evaluation advanced stream: "
                             f"{name}.{step}.{model}")
        if model == "live":
            expected_state = (receipt["initial_live_state_sha256"] if step == 0 else
                              train_events[step]["output_noise_rng_state_sha256"])
            expected_calls = (0 if step == 0 else
                              train_events[step]["output_noise_rng_draw_calls"])
            expected_elements = (0 if step == 0 else
                                 train_events[step]["output_noise_rng_draw_elements"])
            if ((state_before, calls_before, elements_before) !=
                    (expected_state, expected_calls, expected_elements)):
                raise ValueError(f"100-mode isolated output-noise evaluation state differs: "
                                 f"{name}.{step}.live")
        elif step == 0 and (state_before != receipt["initial_ema_state_sha256"]
                             or calls_before != 0 or elements_before != 0):
            raise ValueError(f"100-mode isolated output-noise initial EMA state differs: {name}")
    final_ema = eval_events.get((steps, "ema"))
    if (final_ema is None or receipt["final_ema_state_sha256"] !=
            final_ema["output_noise_rng_state_before_sha256"]):
        raise ValueError(f"100-mode isolated output-noise final EMA state differs: {name}")

    if receipt.get("holdout_eval_seed") != config["seed"] + HOLDOUT_SEED_OFFSETS["noise"]:
        raise ValueError(f"100-mode isolated output-noise holdout seed differs: {name}")
    for model in ("live", "ema"):
        before = receipt.get(f"holdout_{model}_before_state_sha256")
        after = receipt.get(f"holdout_{model}_after_state_sha256")
        if (not _sha256_hex(before) or before != after
                or before != receipt[f"final_{model}_state_sha256"]):
            raise ValueError(f"100-mode isolated output-noise holdout advanced stream: "
                             f"{name}.{model}")


def _check_toy100_policy(directory: Path, summary: dict, config: dict,
                         policy: dict) -> dict[str, str]:
    """Regrade the optional model and LR policy from portable saved evidence."""
    name, steps = config["problem"], config["steps"]
    provenance = _read(directory / "provenance.json")
    if (any(field in provenance or field in summary
            for field in ("trainer_factory", "model_options"))
            or provenance.get("shared_gate_eligible") is False
            or summary.get("shared_gate_eligible") is False
            or (directory / "model_options.json").exists()):
        raise ValueError(f"scratch trainer override cannot enter common gate: {name}")
    sources = provenance.get("source_sha256")
    required = set(NATIVE_SOURCE_FILES) | {
        "benchmarks/toy100/config.py", "benchmarks/toy100/__main__.py",
        "benchmarks/toy100/schedule.py",
    }
    if (summary.get("provenance") != provenance
            or not isinstance(sources, dict) or not required <= set(sources)):
        raise ValueError(f"100-mode policy source provenance differs: {name}")
    policy_source_scope(provenance)
    archive_file = provenance.get("source_archive_file")
    archive_hash = provenance.get("source_archive_sha256")
    if (archive_file != "source.tar.gz"
            or not isinstance(archive_hash, str)):
        raise ValueError(f"100-mode policy source archive receipt differs: {name}")
    archive_bytes = (directory / archive_file).read_bytes()
    if hashlib.sha256(archive_bytes).hexdigest() != archive_hash:
        raise ValueError(f"100-mode policy source archive hash differs: {name}")
    with tarfile.open(directory / archive_file, "r:gz") as archive:
        entries = archive.getmembers()
        members = {entry.name: entry for entry in entries}
        if len(members) != len(entries) or set(members) != set(sources):
            raise ValueError(f"100-mode policy source archive members differ: {name}")
        for path, digest in sources.items():
            entry = members[path]
            stream = archive.extractfile(entry) if entry.isfile() else None
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != digest:
                raise ValueError(f"100-mode policy source differs: {name}.{path}")
    if summary.get("completed_steps") != steps:
        raise ValueError(f"100-mode policy training budget differs: {name}")
    card = summary.get("model_policy")
    expected_model = policy.get("toy100_model", "mlp_v1")
    if (not isinstance(card, dict)
            or card.get("toy100_model") != expected_model
            or card.get("generator_class") != (
                "Linear" if expected_model in AFFINE_MODEL_POLICIES else "SimpleMLPGenerator")
            or card.get("generator_wrapper_class") != (
                ("IsolatedOutputNoise" if config.get("output_noise_rng") == "isolated"
                 else "OutputNoise") if config["output_noise_std"] else None)
            or card.get("discriminator_class") != "SimpleMLPDiscriminator"
            or card.get("discriminator_wrapper_class") != (
                ("StatefulInputNoise" if config.get("output_noise_rng") == "isolated"
                 else "InputNoise") if config["input_noise_std"] else None)
            or card.get("prior_class") != "ParticlePrior"
            or type(card.get("generator_base_parameters")) is not int
            or card["generator_base_parameters"] <= 0
            or card.get("generator_parameters") != card["generator_base_parameters"]
               + int(config.get("output_noise_learnable", False))
            or type(card.get("discriminator_parameters")) is not int
            or card["discriminator_parameters"] <= 0
            or card.get("prior_parameters") != config["num_particles"] * config["z_dim"]):
        raise ValueError(f"100-mode model policy receipt differs: {name}")
    if expected_model in AFFINE_MODEL_POLICIES:
        expected_weight_hash = hashlib.sha256(
            struct.pack("<4f", 1.0, 0.0, 0.0, 1.0),
        ).hexdigest()
        expected_bias_hash = hashlib.sha256(
            struct.pack("<2f", 0.0, 0.0),
        ).hexdigest()
        identity = expected_model in {
            "affine_square_v1", "affine_normal_v1", "affine_empirical_box_v1",
            "affine_moment_box_v1",
        }
        prior_kind = ("uniform_square" if expected_model in {
            "affine_square_v1", "affine_square_random_v1",
        } else "empirical_box" if expected_model in {
            "affine_empirical_box_v1", "affine_empirical_box_random_v1",
        } else "moment_box" if expected_model == "affine_moment_box_v1"
        else "normal")
        weight = card.get("generator_initial_weight")
        bias = card.get("generator_initial_bias")
        if (not isinstance(weight, list) or len(weight) != 2
                or any(not isinstance(row, list) or len(row) != 2 for row in weight)
                or not isinstance(bias, list) or len(bias) != 2
                or not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                           and math.isfinite(value) for row in weight for value in row)
                or not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                           and math.isfinite(value) for value in bias)):
            raise ValueError(f"100-mode affine initialization values differ: {name}")
        weight_hash = hashlib.sha256(struct.pack("<4f", *(x for row in weight for x in row))).hexdigest()
        bias_hash = hashlib.sha256(struct.pack("<2f", *bias)).hexdigest()
        if (card.get("generator_base_parameters") != 6
                or card.get("generator_parameters") != 6 + int(
                    config.get("output_noise_learnable", False))
                or card.get("prior_initialization") != prior_kind
                or card.get("prior_shape") != [config["num_particles"], 2]
                or not isinstance(card.get("prior_initial_sha256"), str)
                or len(card["prior_initial_sha256"]) != 64
                or not isinstance(card.get("prior_initial_min"), (int, float))
                or not isinstance(card.get("prior_initial_max"), (int, float))
                or not math.isfinite(card["prior_initial_min"])
                or not math.isfinite(card["prior_initial_max"])
                or card["prior_initial_min"] > card["prior_initial_max"]
                or card.get("generator_initial_weight_sha256") != weight_hash
                or card.get("generator_initial_bias_sha256") != bias_hash):
            raise ValueError(f"100-mode affine initialization receipt differs: {name}")
        if identity:
            if (weight != [[1.0, 0.0], [0.0, 1.0]] or bias != [0.0, 0.0]
                    or weight_hash != expected_weight_hash or bias_hash != expected_bias_hash
                    or card.get("generator_initialization") not in (
                        ("identity", None) if expected_model == "affine_square_v1"
                        else ("identity",)
                    )):
                raise ValueError(f"100-mode affine identity receipt differs: {name}")
        elif (card.get("generator_initialization") != "torch_linear_default"
              or any(abs(value) > 2 ** -0.5 + 1e-7 for row in weight for value in row)
              or any(abs(value) > 2 ** -0.5 + 1e-7 for value in bias)):
            raise ValueError(f"100-mode default affine receipt differs: {name}")
        if prior_kind == "uniform_square":
            if (not _close(card.get("prior_scale"), 5.0)
                    or not -5.0 <= card["prior_initial_min"] <= card["prior_initial_max"] <= 5.0):
                raise ValueError(f"100-mode square-prior receipt differs: {name}")
        elif prior_kind == "normal":
            if ("prior_scale" in card or not -10.0 <= card["prior_initial_min"]
                    <= card["prior_initial_max"] <= 10.0):
                raise ValueError(f"100-mode normal-prior receipt differs: {name}")
        else:
            lower, upper = card.get("init_data_lower"), card.get("init_data_upper")
            if (card.get("init_data_samples") != config["batch_size"]
                    or card.get("init_data_seed_offset") != EMPIRICAL_INIT_SEED_OFFSET
                    or not _sha256_hex(card.get("init_data_sha256"))
                    or not isinstance(lower, list) or not isinstance(upper, list)
                    or len(lower) != 2 or len(upper) != 2
                    or not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                               and math.isfinite(value) for value in lower + upper)
                    or not all(low < high for low, high in zip(lower, upper))
                    or not min(lower) <= card["prior_initial_min"]
                    <= card["prior_initial_max"] <= max(upper)):
                raise ValueError(f"100-mode data-box receipt differs: {name}")
            if prior_kind == "moment_box":
                mean, std = card.get("init_data_mean"), card.get("init_data_std")
                if (not isinstance(mean, list) or not isinstance(std, list)
                        or len(mean) != 2 or len(std) != 2
                        or not all(type(value) in (int, float) and math.isfinite(value)
                                   for value in mean + std)
                        or not all(value > 0 for value in std)):
                    raise ValueError(f"100-mode moment-box receipt differs: {name}")
                import numpy as np
                import torch
                if card.get("init_data_file") != "initialization-samples.npy":
                    raise ValueError(f"100-mode moment-box data file differs: {name}")
                raw = np.load(directory / card["init_data_file"], allow_pickle=False)
                if (raw.dtype != np.float32
                        or raw.shape != (config["batch_size"], 2)
                        or not np.isfinite(raw).all()):
                    raise ValueError(f"100-mode moment-box data differs: {name}")
                samples = torch.from_numpy(np.ascontiguousarray(raw))
                expected_mean = samples.mean(dim=0)
                expected_std = samples.std(dim=0, unbiased=False)
                half_width = math.sqrt(3.0) * expected_std
                expected_lower = expected_mean - half_width
                expected_upper = expected_mean + half_width
                if (card["init_data_sha256"] != hashlib.sha256(
                        raw.tobytes(),
                    ).hexdigest()
                        or any(not math.isclose(actual, expected, rel_tol=2e-6,
                                                abs_tol=5e-6)
                               for observed, calculated in (
                                   (mean, expected_mean.tolist()),
                                   (std, expected_std.tolist()),
                                   (lower, expected_lower.tolist()),
                                   (upper, expected_upper.tolist()),
                               ) for actual, expected in zip(observed, calculated))):
                    raise ValueError(f"100-mode moment-box sample/formula differs: {name}")
    cap = policy.get("network_lr_horizon_cap")
    network_floor = policy.get("network_lr_floor")
    if cap is not None:
        if summary.get("network_lr_horizon_cap") != cap:
            raise ValueError(f"100-mode network horizon receipt differs: {name}")
        if (network_floor is not None and not _close(
                summary.get("network_lr_floor"), network_floor)) or (
                network_floor is None and "network_lr_floor" in summary):
            raise ValueError(f"100-mode network floor receipt differs: {name}")
        from benchmarks.toy100.schedule import policy_multipliers
        train_events = {}
        for line in (directory / "events.jsonl").read_text().splitlines():
            event = json.loads(line)
            if event.get("event") == "train":
                step = event.get("step")
                if step in train_events:
                    raise ValueError(f"duplicate 100-mode policy action: {name}.{step}")
                train_events[step] = event
        if set(train_events) != set(range(1, steps + 1)):
            raise ValueError(f"100-mode policy action trace is incomplete: {name}")
        for step, event in train_events.items():
            network, prior = policy_multipliers(
                step - 1, steps, config["lr_anneal_start"], config["lr_floor"], cap,
                network_lr_floor=network_floor,
            )
            expected = dict(network_lr_horizon_cap=cap,
                            network_multiplier=network, prior_multiplier=prior,
                            lr_g=config["lr"] * network,
                            lr_prior=config["lr"] * config["prior_lr_mult"] * prior,
                            lr_d=config["lr"] * config["d_lr_mult"] * network)
            if network_floor is not None:
                expected["network_lr_floor"] = network_floor
            if any(not _close(event.get(key), value) for key, value in expected.items()):
                raise ValueError(f"100-mode policy action differs: {name}.{step}")
            if network_floor is None and "network_lr_floor" in event:
                raise ValueError(f"undeclared 100-mode network floor action: {name}.{step}")
    elif summary.get("network_lr_horizon_cap") is not None:
        raise ValueError(f"undeclared 100-mode network horizon receipt: {name}")
    elif "network_lr_floor" in summary:
        raise ValueError(f"undeclared 100-mode network floor receipt: {name}")
    return sources


def _verify_saved_provenance(directory: Path, protocol: dict, *, candidate: bool):
    """Bind the saved executable source and declared config to their hashes."""
    source_hashes = dict(protocol["source_sha256"])
    noise_name = "benchmarks/toy100/models.py"
    if candidate:
        expected_noise_hash = source_hashes.pop(noise_name)
        if (expected_noise_hash != protocol["noise_source_sha256"]
                or hashlib.sha256((directory / "noise_source.py").read_bytes()).hexdigest()
                != expected_noise_hash):
            raise ValueError("saved noise source differs from protocol hash")
        config_name = Path(protocol["config_file"])
        if config_name.is_absolute() or len(config_name.parts) != 1:
            raise ValueError("candidate config path escapes evidence directory")
        config_bytes = (directory / config_name).read_bytes()
        if hashlib.sha256(config_bytes).hexdigest() != protocol["config_sha256"]:
            raise ValueError("saved candidate config differs from protocol hash")
        saved_config = json.loads(config_bytes)
        base, noise, overrides = declared_recipe(saved_config)
        model_policy = declared_model_policy(saved_config)
        if (_json_value(base.to_dict()) != protocol["global_recipe"]
                or _noise_identity(noise) != _noise_identity(protocol["noise"])
                or overrides != protocol["ignored_toy100_resource_overrides"]
                or model_policy != protocol.get("model_policy", {})):
            raise ValueError("saved candidate config does not resolve to declared recipe")
        if (model_policy or noise.get("output_noise_rng") == "isolated") and not {
            "benchmarks/toy100/schedule.py", "benchmarks/toy100/train.py",
            "benchmarks/toy100/config.py", "benchmarks/toy100/__main__.py",
        } <= set(source_hashes):
            raise ValueError("candidate policy and parser source is absent")
    with tarfile.open(directory / "source.tar.gz", "r:gz") as archive:
        all_members = archive.getmembers()
        members = {member.name: member for member in all_members}
        if len(members) != len(all_members) or set(members) != set(source_hashes):
            raise ValueError("source archive members differ from protocol manifest")
        for name, expected_hash in source_hashes.items():
            member = members[name]
            stream = archive.extractfile(member) if member.isfile() else None
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != expected_hash:
                raise ValueError(f"saved source archive differs: {name}")


def _reject_scratch_optimizer(payload: dict) -> None:
    if not isinstance(payload, dict):
        raise ValueError("optimizer policy evidence must be an object")
    if (payload.get("shared_gate_eligible") is False
            or "scratch_optimizer_policy" in payload
            or "optimizer_policy" in payload):
        raise ValueError("scratch or unsupported optimizer policy cannot enter common gate")


def _episode_rows(directory: Path, expected_names: tuple[str, ...], *, candidate: bool,
                  allow_scratch: bool = False):
    """Recompute every live verdict from the compressed episode, not the stamp."""
    if not (directory / "protocol.json").is_file():
        return dict(status="MISSING", passed=0, required=len(expected_names),
                    cases={}, reason="protocol.json is absent")
    try:
        protocol = _read(directory / "protocol.json")
        if not allow_scratch:
            _reject_scratch_optimizer(protocol)
            summary_path = directory / "summary.json"
            if summary_path.is_file():
                summary = _read(summary_path)
                _reject_scratch_optimizer(summary)
                for case in summary.get("cases", []):
                    _reject_scratch_optimizer(case)
        _verify_saved_provenance(directory, protocol, candidate=candidate)
        index = _read(directory / "index.json")
        if not allow_scratch:
            _reject_scratch_optimizer(index)
        rows = index["records"]
        names = [row["name"] for row in rows]
        if len(names) != len(set(names)) or set(names) - set(expected_names):
            raise ValueError("indexed episode names are duplicated or unknown")
        jobs, profile = load_declaration()
        frozen_jobs = {job["spec"]["name"]: job for job in jobs}
        if protocol["jobs"] != [frozen_jobs[job["spec"]["name"]]
                                for job in protocol["jobs"]]:
            raise ValueError("archived jobs differ from frozen declarations")
        if candidate:
            if protocol["frozen_discriminators"] != profile["discriminators"]:
                raise ValueError("candidate discriminator profile differs from frozen card")
            recipe_fields = protocol["global_recipe"]
        else:
            if protocol["frozen_profile"] != profile:
                raise ValueError("public control profile differs from frozen card")
            recipe_fields = protocol["base_get_recipe"]
        base = Recipe(**recipe_fields)
        if _json_value(base.to_dict()) != recipe_fields:
            raise ValueError("archived recipe does not resolve to declared fields")
        cases = {}
        for row in rows:
            if not allow_scratch:
                _reject_scratch_optimizer(row)
            artifact = (directory / row["artifact"]).resolve()
            if not artifact.is_relative_to(directory.resolve()):
                raise ValueError("episode path escapes evidence directory")
            raw = gzip.decompress(artifact.read_bytes())
            if hashlib.sha256(raw).hexdigest() != row["uncompressed_sha256"]:
                raise ValueError(f"episode hash differs: {row['name']}")
            record = json.loads(raw)
            if not allow_scratch:
                _reject_scratch_optimizer(record)
                _reject_scratch_optimizer(record.get("result", {}))
            name = row["name"]
            if record["name"] != name or record["original_spec"] != frozen_jobs[name]["spec"]:
                raise ValueError(f"frozen task declaration differs: {name}")
            expected_spec, _, variant = declared_spec(frozen_jobs[name], profile, base)
            if (record["spec"] != _json_value(expected_spec)
                    or record.get("discriminator_variant") != _json_value(variant)):
                raise ValueError(f"executed spec, budget, threshold, or discriminator differs: {name}")
            expected_host_recipe = (base if expected_spec["runner"] == "legacy"
                                    else host_recipe(base, expected_spec))
            if record.get("host_recipe") != _json_value(expected_host_recipe.to_dict()):
                raise ValueError(f"host resource recipe differs from declaration: {name}")
            if record["source_sha256"] != protocol["source_sha256"]:
                raise ValueError(f"episode source differs: {name}")
            if candidate:
                if (record["recipe"] != protocol["global_recipe"]
                        or _noise_identity(record["noise"]) != _noise_identity(protocol["noise"])
                        or record.get("model_policy", {}) != protocol.get("model_policy", {})):
                    raise ValueError(f"candidate global fields differ between cases: {name}")
                receipt = record.get("noise_receipt")
                if not isinstance(receipt, dict):
                    raise ValueError(f"candidate noise receipt is absent: {name}")
                output_std = protocol["noise"]["output_noise_std"]
                input_std = protocol["noise"]["input_noise_std"]
                warmup = protocol["noise"].get("output_noise_warmup", 0.0)
                steps = record["spec"]["steps"]
                if receipt.get("step_calls") != steps:
                    raise ValueError(f"candidate noise schedule is incomplete: {name}")
                expected_first = output_noise_at(output_std, 0, steps, warmup)
                expected_last = output_noise_at(
                    output_std, steps - 1, steps, warmup,
                )
                expected_input_nonzero = sum(
                    linear_input_noise(input_std, step, steps,
                                       protocol["noise"]["input_noise_anneal_end"]) > 0
                    for step in range(steps)
                )
                if receipt.get("input_nonzero_steps") != expected_input_nonzero:
                    raise ValueError(f"candidate input noise duration differs: {name}")
                if "output_noise_warmup" in protocol["noise"] and warmup:
                    if (not _close(receipt.get("output_sigma_first"), expected_first)
                            or not _close(receipt.get("output_sigma_last"), expected_last)):
                        raise ValueError(f"candidate output warmup differs: {name}")
                    expected_output_nonzero = sum(
                        output_noise_at(output_std, step, steps, warmup) > 0
                        for step in range(steps)
                    )
                    if receipt.get("output_nonzero_steps") != expected_output_nonzero:
                        raise ValueError(f"candidate output warmup duration differs: {name}")
                if record["spec"]["runner"] == "legacy":
                    if (not _close(receipt.get("output_std"), output_std)
                            or not _close(receipt.get("input_std"), input_std)
                            or not _close(receipt.get("input_anneal_end"),
                                          protocol["noise"]["input_noise_anneal_end"])
                            or not _close(receipt.get("input_sigma_first"),
                                          linear_input_noise(input_std, 0, steps,
                                                             protocol["noise"]["input_noise_anneal_end"]))
                            or not _close(receipt.get("input_sigma_last"),
                                          linear_input_noise(input_std, steps - 1, steps,
                                                             protocol["noise"]["input_noise_anneal_end"]))):
                        raise ValueError(f"custom-host noise receipt differs from common policy: {name}")
                    actual_noise = ((output_std == 0 or receipt.get("train_output_applied"))
                                    and (input_std == 0 or receipt.get("train_input_applied")
                                         and receipt.get("input_nonzero_steps", 0) > 0))
                    if not receipt.get("eval_scope"):
                        raise ValueError(f"custom-host evaluation scope is absent: {name}")
                else:
                    output_module = ("IsolatedOutputNoise" if
                                     protocol["noise"].get("output_noise_rng") == "isolated"
                                     else "OutputNoise")
                    input_module = ("StatefulInputNoise" if
                                    protocol["noise"].get("output_noise_rng") == "isolated"
                                    else "InputNoise")
                    actual_noise = ((output_std == 0 or receipt.get("output_module") == output_module)
                                    and (input_std == 0 or receipt.get("input_module") == input_module
                                         and receipt.get("input_nonzero_steps", 0) > 0))
                _check_learned_transfer_noise(record, receipt, protocol["noise"])
                _check_isolated_transfer_noise(record, receipt, protocol["noise"])
                if bool(actual_noise) != record.get("noise_applied"):
                    raise ValueError(f"candidate noise claim differs from receipt: {name}")
            else:
                if record["recipe"] != protocol["base_get_recipe"]:
                    raise ValueError(f"public default recipe differs between cases: {name}")
            result = record["result"]
            _check_optimizer_receipts(record, expected_host_recipe)
            _check_actions(record, expected_host_recipe,
                           protocol["noise"] if candidate else None,
                           protocol.get("model_policy") if candidate else None)
            verdict = test_verdict(record["spec"], result)
            if (verdict["status"] != record["verdict"]["status"]
                    or verdict["passed"] != record["verdict"]["passed"]
                    or verdict["status"] != row["verdict"]["status"]):
                raise ValueError(f"stored verdict differs from independent regrade: {name}")
            observations = result.get("observations", result.get("curve", []))
            expected_checkpoints = sorted({math.ceil(i * record["spec"]["steps"] / 24)
                                           for i in range(1, 25)})
            if [item.get("step") for item in observations] != expected_checkpoints:
                raise ValueError(f"frozen 24-checkpoint schedule differs: {name}")
            cases[name] = dict(status=verdict["status"], passed=verdict["passed"],
                               observations=len(observations), artifact=str(artifact),
                               noise_applied=record.get("noise_applied", not candidate),
                               eval_scope=(record.get("noise_receipt") or {}).get("eval_scope"),
                               final=result.get("live", {}),
                               passing_suffix=verdict.get("convergence", {}).get("passing_suffix"))
        passed = sum(row["passed"] for row in cases.values())
        return dict(status=_status(passed, len(expected_names),
                                   complete=set(cases) == set(expected_names)),
                    passed=passed, required=len(expected_names), cases=cases,
                    protocol=protocol, reason=None)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError,
            tarfile.TarError,
            gzip.BadGzipFile, EOFError) as error:
        return dict(status="INVALID", passed=0, required=len(expected_names),
                    cases={}, reason=str(error))


def _toy100_rows(directory: Path):
    if not (directory / "run_manifest.json").is_file():
        return dict(status="MISSING", passed=0, required=len(PROBLEM_NAMES),
                    cases={}, reason="run_manifest.json is absent")
    try:
        manifest = _read(directory / "run_manifest.json")
        coverage = coverage_suite(directory, write=False)
        accuracy = accuracy_suite(directory, write=False)
        if (coverage["protocol"] != "toy100-v1"
                or accuracy["protocol"] != "toy100-accuracy-v1"
                or coverage["scope"] != "all declared problems"
                or accuracy["scope"] != "all declared problems"
                or set(coverage["problems"]) != set(PROBLEM_NAMES)
                or set(accuracy["problems"]) != set(PROBLEM_NAMES)):
            raise ValueError("100-mode gates do not cover all three declared problems")
        declared = manifest["declared_manifest"]
        if hashlib.sha256(manifest["config_contents"].encode()).hexdigest() != manifest["config_sha256"]:
            raise ValueError("100-mode manifest config hash differs")
        recipe, noise, _ = declared_recipe(declared)
        model_policy = declared_model_policy(declared)
        isolated_output_rng = noise.get("output_noise_rng") == "isolated"
        requires_archive = bool(model_policy) or isolated_output_rng
        config_fields = {name: getattr(recipe, name) for name in GLOBAL_RECIPE_FIELDS}
        policy_sources = None
        policy_scope = None
        for name in PROBLEM_NAMES:
            executed = _read(directory / name / "config.json")
            expected_config, _ = resolve_config(manifest["resolved_problem_configs"][name])
            if executed != json.loads(json.dumps(expected_config)):
                raise ValueError(f"executed 100-mode configuration differs: {name}")
            for key in GLOBAL_RECIPE_FIELDS:
                actual = executed[key]
                expected = config_fields[key]
                if isinstance(expected, tuple):
                    expected = list(expected)
                if actual != expected:
                    raise ValueError(f"100-mode global field varies by problem: {name}.{key}")
            if any(executed.get(key, 0.0) != noise[key] for key in NOISE_FIELDS):
                raise ValueError(f"100-mode noise varies by problem: {name}")
            if bool(executed.get("output_noise_learnable", False)) != bool(
                noise.get("output_noise_learnable", False),
            ):
                raise ValueError(f"100-mode learned-noise flag varies by problem: {name}")
            if executed.get("output_noise_rng") != noise.get("output_noise_rng"):
                raise ValueError(f"100-mode output-noise RNG varies by problem: {name}")
            if declared_model_policy(executed) != model_policy:
                raise ValueError(f"100-mode model policy varies by problem: {name}")
            if requires_archive:
                archived_scope = policy_source_scope(_read(directory / name / "provenance.json"))
                sources = _check_toy100_policy(
                    directory / name,
                    _read(directory / name / "summary.json"),
                    executed, model_policy,
                )
                if policy_sources is None:
                    policy_sources = sources
                    policy_scope = archived_scope
                elif sources != policy_sources:
                    raise ValueError("100-mode policy source differs between problems")
                elif archived_scope != policy_scope:
                    raise ValueError("100-mode policy source scope differs between problems")
            _check_toy100_learned_noise(
                directory / name, _read(directory / name / "summary.json"), executed,
                policy_archive_verified=requires_archive,
            )
            _check_toy100_output_rng(
                directory / name, _read(directory / name / "summary.json"), executed,
            )
        if requires_archive and manifest.get("policy_source_sha256") != policy_sources:
            raise ValueError("100-mode policy manifest source differs from archived runs")
        if isolated_output_rng and policy_scope != POLICY_SOURCE_SCOPE_V2:
            raise ValueError("isolated output-noise RNG requires full V2 source archive")
        if requires_archive and policy_scope == POLICY_SOURCE_SCOPE_V2:
            if (manifest.get("policy_source_scope") != policy_scope
                    or manifest.get("policy_source_version") != 2):
                raise ValueError("100-mode policy manifest source scope differs")
        elif requires_archive and ("policy_source_scope" in manifest
                               or "policy_source_version" in manifest):
            raise ValueError("historical 100-mode policy manifest has a new source scope")
        cases = {name: dict(
            status="PASS" if coverage["problems"][name]["passed"]
                             and accuracy["problems"][name]["passed"] else "FAIL",
            passed=bool(coverage["problems"][name]["passed"]
                        and accuracy["problems"][name]["passed"]),
            coverage=coverage["problems"][name]["status"],
            accuracy=accuracy["problems"][name]["status"],
            holdout=accuracy["problems"][name].get("holdout_metrics"),
        ) for name in PROBLEM_NAMES}
        passed = sum(row["passed"] for row in cases.values())
        return dict(status=_status(passed, len(PROBLEM_NAMES), complete=True),
                    passed=passed, required=len(PROBLEM_NAMES), cases=cases,
                    recipe=recipe.to_dict(), noise=noise,
                    model_policy=model_policy, policy_source_sha256=policy_sources,
                    policy_source_scope=policy_scope,
                    config_sha256=manifest["config_sha256"],
                    coverage_status=coverage["status"],
                    accuracy_status=accuracy["status"], reason=None)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError,
            tarfile.TarError, EOFError) as error:
        return dict(status="INVALID", passed=0, required=len(PROBLEM_NAMES),
                    cases={}, reason=str(error))


def regrade(output: Path):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    toy = _toy100_rows(output / "toy100")
    expected19 = tuple(job["spec"]["name"] for job in plan())
    candidate = _episode_rows(output / "candidate19", expected19, candidate=True)
    vector = _episode_rows(output / "vector6", VECTOR_NAMES, candidate=True)
    if vector["status"] == "MISSING" and candidate["cases"]:
        subset = {name: candidate["cases"][name] for name in VECTOR_NAMES
                  if name in candidate["cases"]}
        passed = sum(row["passed"] for row in subset.values())
        vector = dict(status=_status(passed, len(VECTOR_NAMES),
                                     complete=len(subset) == len(VECTOR_NAMES)),
                      passed=passed, required=len(VECTOR_NAMES), cases=subset,
                      reason="derived from candidate-19 episodes")
    control = _episode_rows(output / "public19", expected19, candidate=False)
    identity = False
    full_source_coverage = None
    reason = None
    if "recipe" in toy and "protocol" in candidate:
        candidate_recipe = candidate["protocol"]["global_recipe"]
        candidate_noise = candidate["protocol"]["noise"]
        identity = (all(_json_value(toy["recipe"][name]) == candidate_recipe[name]
                        for name in GLOBAL_RECIPE_FIELDS)
                    and _noise_identity(toy["noise"]) == _noise_identity(candidate_noise)
                    and toy.get("model_policy", {}) == candidate["protocol"].get("model_policy", {}))
        if not identity:
            reason = "100-mode and candidate-19 global recipe, noise, or model policy fields differ"
        elif toy.get("model_policy") or toy["noise"].get("output_noise_rng") == "isolated":
            native_sources = toy.get("policy_source_sha256") or {}
            transfer_sources = candidate["protocol"]["source_sha256"]
            full_source_coverage = toy.get("policy_source_scope") == POLICY_SOURCE_SCOPE_V2
            shared = set(native_sources) & set(transfer_sources)
            required_shared = {"particlegan/training.py", "particlegan/recipes.py",
                               "lib/toy_models.py", "benchmarks/toy100/models.py",
                               "benchmarks/toy100/config.py",
                               "benchmarks/toy100/__main__.py",
                               "benchmarks/toy100/train.py",
                               "benchmarks/toy100/schedule.py"}
            if (not required_shared <= shared
                    or any(native_sources[key] != transfer_sources[key] for key in shared)):
                identity = False
                reason = "100-mode and candidate-19 executable policy source differs"
            elif full_source_coverage:
                native_public = {name for name in native_sources
                                 if name.startswith("particlegan/") and name.endswith(".py")}
                transfer_public = {name for name in transfer_sources
                                   if name.startswith("particlegan/") and name.endswith(".py")}
                if native_public != transfer_public:
                    identity = False
                    reason = "100-mode and candidate-19 public package source sets differ"
    else:
        reason = "complete 100-mode and candidate-19 evidence is required"
    noise_covered = (len(candidate["cases"]) == len(expected19)
                     and all(row["noise_applied"] for row in candidate["cases"].values()))
    if identity and not noise_covered:
        reason = "common noise mechanism lacks a complete host receipt"
    complete = toy["status"] in ("PASS", "FAIL") and candidate["status"] in ("PASS", "FAIL")
    if not complete or not identity or not noise_covered:
        status = "INCOMPLETE"
    elif toy["status"] == candidate["status"] == "PASS" and full_source_coverage is False:
        status = "INCOMPLETE"
        reason = "historical 13-file policy archive has limited public-source coverage"
    elif toy["status"] == candidate["status"] == "PASS":
        status = "PASS"
    else:
        status = "FAIL"
    observed_passes = toy["passed"] + candidate["passed"]
    report = dict(protocol="toy-suite-common22-v1", status=status,
                  observed_passes=observed_passes, required=22,
                  global_recipe_identical=identity, noise_applied_on_all_19=noise_covered,
                  policy_source_scope=toy.get("policy_source_scope"),
                  full_public_source_coverage=full_source_coverage,
                  reason=reason, toy100=toy, candidate19=candidate,
                  vector6=vector, public_default19_control=control)
    _write(output / "compatibility.json", report)
    lines = [f"# One-recipe 22-toy gate: {status}", "",
             "A PASS requires the same global optimizer, loss, schedule, and noise settings "
             "on all 22 hosts. The three 100-mode problems must pass both coverage and "
             "accuracy gates; the 19 canonical cases must sustain their frozen live gates. "
             "Architecture and host resource sizes follow the frozen task declarations.", "",
             "| Evidence | Live passes | Status |",
             "| --- | ---: | --- |",
             f"| Three 100-mode problems, coverage + accuracy | {toy['passed']}/3 | {toy['status']} |",
             f"| Same candidate on 19 canonical hosts | {candidate['passed']}/19 | {candidate['status']} |",
             f"| Candidate six-vector full-noise screen | {vector['passed']}/6 | {vector['status']} |",
             f"| Public v3 installed-wheel control | {control['passed']}/19 | {control['status']} |",
             "", f"Global fields identical: **{identity}**. Noise applied on all 19: "
             f"**{noise_covered}**. {reason or ''}", "",
             f"Native policy archive scope: **{toy.get('policy_source_scope') or 'not applicable'}**. "
             f"Full public package source coverage: **{full_source_coverage}**.", "",
             "## Per-case result", "",
             "| Group | Case | Live | Detail |",
             "| --- | --- | --- | --- |"]
    for name in PROBLEM_NAMES:
        row = toy["cases"].get(name, {})
        lines.append(f"| 100-mode | `{name}` | {row.get('status', 'MISSING')} | "
                     f"coverage {row.get('coverage', 'MISSING')}; accuracy {row.get('accuracy', 'MISSING')} |")
    for name in expected19:
        row = candidate["cases"].get(name, {})
        lines.append(f"| canonical | `{name}` | {row.get('status', 'MISSING')} | "
                     f"noise applied {row.get('noise_applied', False)}; "
                     f"eval {row.get('eval_scope') or 'generated samples'}; "
                     f"final suffix {row.get('passing_suffix', '—')} |")
    lines += ["", "The public v3 control uses its own recipe and cannot supply missing "
              "candidate passes. A single-case or six-case screen is incomplete for 22/22. "
              "EMA is recorded separately and never determines the live gate.", ""]
    (output / "compatibility.md").write_text("\n".join(lines))
    return report


def _run_command(command: list[str], *, cwd: Path, log: Path, env: dict[str, str]):
    with log.open("w") as stream:
        process = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, check=False)
    print(json.dumps(dict(event="command_complete", command=command,
                          returncode=process.returncode, log=str(log))), flush=True)
    return process.returncode


def run(config: Path, output: Path, *, with_default_control: bool = False):
    config = config.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError("use a new suite output directory")
    output.mkdir(parents=True)
    python = sys.executable
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=str(ROOT))
    commands = [
        ([python, "-u", "-m", "benchmarks.toy100", "run", "--config", str(config),
          "--output", str(output / "toy100"), "--no-render"], ROOT, output / "toy100.log"),
        ([python, "-u", "-m", "benchmarks.toy100.accuracy_gate", "--output",
          str(output / "toy100")], ROOT, output / "accuracy.log"),
        ([python, "-u", "-m", "benchmarks.transfer_suite.toy100_compatibility",
          "--config", str(config), "--all", "--output", str(output / "candidate19")],
         ROOT, output / "candidate19.log"),
    ]
    returns = {}
    for index, (command, cwd, log) in enumerate(commands):
        print(json.dumps(dict(event="command_start", command=command,
                              log=str(log))), flush=True)
        returns[str(index)] = _run_command(command, cwd=cwd, log=log, env=env)
    if with_default_control:
        wheel_dir, site = output / "wheel", output / "site"
        wheel_dir.mkdir()
        returns["wheel"] = _run_command(
            [python, "-m", "pip", "wheel", "--no-deps", ".", "--wheel-dir",
             str(wheel_dir)], cwd=ROOT, log=output / "wheel.log", env=env)
        wheels = sorted(wheel_dir.glob("particlegan-*.whl"))
        if len(wheels) == 1 and returns["wheel"] == 0:
            returns["install"] = _run_command(
                [python, "-m", "pip", "install", "--no-deps", "--target", str(site),
                 str(wheels[0])], cwd=ROOT, log=output / "install.log", env=env)
            installed_env = env | {"PYTHONPATH": str(site) + os.pathsep + str(ROOT)}
            if returns["install"] == 0:
                returns["public19"] = _run_command(
                    [python, "-u", "-m", "benchmarks.transfer_suite.public_default_verification",
                     "--require-installed-root", str(site), "--output", str(output / "public19")],
                    cwd=Path("/tmp"), log=output / "public19.log", env=installed_env)
    _write(output / "command_returns.json", returns)
    return regrade(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="train and grade fresh evidence")
    run_parser.add_argument("--config", type=Path,
                            default=Path("configs/toy100/shared_candidate.json"),
                            help="candidate recipe (default: the verified shared 22-toy candidate)")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--with-default-control", action="store_true")
    regrade_parser = commands.add_parser("regrade", help="independently grade saved evidence")
    regrade_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run(args.config, args.output,
                     with_default_control=args.with_default_control)
    else:
        result = regrade(args.output)
    print(json.dumps(dict(status=result["status"], observed_passes=result["observed_passes"],
                          required=result["required"], report=str(args.output / "compatibility.md"))),
          flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
