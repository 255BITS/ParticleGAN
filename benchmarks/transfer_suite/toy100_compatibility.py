"""Screen a declared 100-mode recipe on the six frozen 2D transfer toys.

The six hosts retain their original data, architecture cards, resource sizes,
budgets, and live gates (``vector_tasks.VectorTask``). They train on the shared
toy runner: the declared recipe at each task's shape, with the configuration's
noise and network-LR policy as recipe fields, builds the optimizers (which own
the schedule), loss, penalty, noise and EMA. The four image hosts
(``image_tasks.ImageTask``) do the same. This is a transfer screen, not part of the canonical 19-case
public-default verification or a search over host settings.

python -u -m benchmarks.transfer_suite.toy100_compatibility --device auto \
    --config configs/toy100/shared_candidate.json --output /tmp/toy100-vector-screen
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import traceback

import torch

from benchmarks.toy100.device import (
    add_device_argument, apply_device_policy, experiment_generator, host_device,
    rng_fork_devices,
)
from benchmarks.toy100.models import (
    InputNoise, IsolatedOutputNoise, OutputNoise, OUTPUT_NOISE_SEED_OFFSET,
    StatefulInputNoise,
    paired_output_noise, linear_input_noise,
    linear_output_noise as output_noise_at,
)
from benchmarks.toy100.train import AFFINE_MODEL_POLICIES, load_config, resolve_config
from particlegan import GANTrainer
from benchmarks.legacy.recipe import get_recipe

from . import image_tasks, suite, vector_tasks
from . import problem_hosts
from .compare_defaults import ema_verdict
from .legacy_noise_adapters import run_legacy as run_noisy_legacy
from .protocol import test_verdict
from .public_default_verification import (
    GLOBAL_RECIPE_FIELDS, declared_spec, host_recipe, load_declaration,
    optimizer_receipts, public_module_manifest, rate_action,
    shape_receipt, write,
)
from benchmarks.gan_v3 import gan_v3_recipe, legacy_dict


ROOT = Path(__file__).resolve().parents[2]
VECTOR_NAMES = (
    "vector_two_broad", "vector_unequal_mass", "vector_unequal_width",
    "vector_anisotropic", "vector_overlap", "vector_spiral",
)
REMAINING_NAMES = (
    "two_pole", "trajectory", "residual_student", "unipolar", "ae_gan_hold",
    "cover_leftover", "unused_token_hold", "mid_scale_identity", "mode_hold",
    "img_stripes2", "img_bars4", "img_blobs4", "img_intensity2",
)
NOISE_SOURCE = ROOT / "benchmarks/toy100/models.py"
MODEL_POLICY_FIELDS = ("toy100_model", "network_lr_horizon_cap", "network_lr_floor")


def declared_model_policy(config: dict) -> dict:
    """Declare one optional native model card and network schedule for all hosts."""
    overrides = config.get("problem_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("problem_overrides must be an object")
    for values in overrides.values():
        if not isinstance(values, dict):
            raise ValueError("problem overrides must contain field objects")
        if any(key in values for key in MODEL_POLICY_FIELDS):
            raise ValueError("model policy cannot vary by 100-mode problem")
    policy = {}
    if "toy100_model" in config:
        if config["toy100_model"] not in AFFINE_MODEL_POLICIES:
            raise ValueError("unsupported toy100_model")
        policy["toy100_model"] = config["toy100_model"]
    if "network_lr_horizon_cap" in config:
        cap = config["network_lr_horizon_cap"]
        if type(cap) is not int or cap <= 0:
            raise ValueError("network_lr_horizon_cap must be a positive integer")
        policy["network_lr_horizon_cap"] = cap
    if "network_lr_floor" in config:
        floor = config["network_lr_floor"]
        if ("network_lr_horizon_cap" not in policy
                or isinstance(floor, bool) or not isinstance(floor, (int, float))
                or not math.isfinite(floor) or not 0 <= floor <= 1):
            raise ValueError("network_lr_floor requires a cap and a finite fraction in [0, 1]")
        policy["network_lr_floor"] = float(floor)
    return policy


def declared_recipe(config: dict):
    """Resolve once, then discard the 100-mode host's resource dimensions."""
    candidate = deepcopy(config)
    resource_overrides = candidate.pop("problem_overrides", {})
    if not isinstance(resource_overrides, dict):
        raise ValueError("problem_overrides must be an object")
    # The transfer screen can probe this optional common schedule before the
    # 100-mode runner adopts it. It is still part of the declared noise policy.
    output_warmup = candidate.pop("output_noise_warmup", 0.0)
    if (isinstance(output_warmup, bool) or not isinstance(output_warmup, (int, float))
            or not math.isfinite(output_warmup) or not 0 <= output_warmup <= 1):
        raise ValueError("output_noise_warmup must be a finite fraction in [0, 1]")
    # This is a common noise mechanism, not a host resource override. Omit it
    # from historical protocols when undeclared so old records retain their
    # exact recipe/noise identity.
    output_rng_declared = "output_noise_rng" in candidate
    output_rng = candidate.pop("output_noise_rng", None)
    if output_rng_declared and output_rng != "isolated":
        raise ValueError("output_noise_rng must be 'isolated' when declared")
    resolved, _ = resolve_config(candidate)
    globals_only = {name: resolved[name] for name in GLOBAL_RECIPE_FIELDS}
    recipe = gan_v3_recipe(**globals_only).replace(name=str(config.get("name", "toy100_transfer")))
    noise = {name: resolved[name] for name in
             ("output_noise_std", "input_noise_std", "input_noise_anneal_end")}
    noise["output_noise_warmup"] = float(output_warmup)
    if output_rng_declared:
        if noise["output_noise_std"] <= 0:
            raise ValueError("isolated output_noise_rng requires output_noise_std > 0")
        noise["output_noise_rng"] = output_rng
    # Omit the optional false value so archived fixed-noise protocols keep
    # their original canonical noise identity. True is always explicit.
    if resolved.get("output_noise_learnable", False):
        noise["output_noise_learnable"] = True
    return recipe, noise, resource_overrides


def _effective_output_std(model) -> float:
    value = model.effective_std()
    return float(value.detach()) if isinstance(value, torch.Tensor) else float(value)


def _output_stream_sha256(model) -> str:
    return hashlib.sha256(model.noise_stream.get_state().cpu().numpy().tobytes()).hexdigest()


RUNNER_NOISE_FIELDS = ("input_noise_std", "input_noise_anneal_end",
                       "output_noise_std", "output_noise_warmup")


def vector_recipe(spec, base, noise, model_policy=None):
    """The declared candidate at the task's shape, noise and LR policy as recipe fields.

    Isolated or learnable output noise has no recipe field on the shared
    runner, so those declarations are refused rather than approximated.
    """
    unsupported = sorted(set(noise) - set(RUNNER_NOISE_FIELDS))
    if unsupported:
        raise ValueError(f"not expressible on the shared toy runner: {unsupported}")
    policy = {key: (model_policy or {}).get(key) for key in ("network_lr_horizon_cap", "network_lr_floor")}
    return host_recipe(base, spec).replace(**{key: noise[key] for key in RUNNER_NOISE_FIELDS}, **policy)


def run_vector(spec, card, base, noise, *, model_policy=None, log_path=None):
    """One vector host on the shared runner; ``log_path`` gets one JSON line per observation."""
    torch.set_num_threads(1)
    recipe = vector_recipe(spec, base, noise, model_policy)
    result = vector_tasks.train(vector_tasks.VectorTask(spec, card), recipe, log_path=log_path)
    context = dict(applied=result.pop("applied"), shapes=result.pop("shapes"), host_recipe=recipe,
                   runner_recipe=recipe)
    return result, context


def _runner_noise_receipt(context, noise, spec, result):
    """What the shared runner applied, read from its recipe and the per-step receipts."""
    recipe, actions = context["runner_recipe"], result["actions"]
    return dict(
        output_module="toy_runner", input_module="InputNoise",
        recipe_noise={key: getattr(recipe, key) for key in RUNNER_NOISE_FIELDS},
        input_nonzero_steps=sum(action["input_sigma"] > 0 for action in actions),
        output_nonzero_steps=sum(action["output_sigma"] > 0 for action in actions),
        output_sigma_first=actions[0]["output_sigma"], output_sigma_last=actions[-1]["output_sigma"],
        evaluation="clean generator output (the runner samples without training noise)",
        output_noise_learnable=False, output_scale_parameter_count=0,
        step_calls=len(actions),
    )


def image_recipe(spec, base, noise, model_policy=None):
    """The host recipe plus the declared noise and network schedule as recipe fields.

    The shared runner applies both: critic input noise and generator output
    noise on the recipe horizon, and the capped network LR schedule inside
    the recipe-built optimizers. Output noise always comes from the runner's
    own stream and is never drawn while measuring, which is what the
    declared ``output_noise_rng="isolated"`` asks for. A learnable output
    scale is not a recipe field, so it cannot be screened on this route.
    """
    if noise.get("output_noise_learnable", False):
        raise NotImplementedError("learnable output noise is not a recipe field; "
                                  "the shared toy runner cannot screen it on the image hosts")
    policy = model_policy or {}
    return host_recipe(base, spec).replace(
        input_noise_std=noise["input_noise_std"], input_noise_anneal_end=noise["input_noise_anneal_end"],
        output_noise_std=noise["output_noise_std"], output_noise_warmup=noise["output_noise_warmup"],
        network_lr_horizon_cap=policy.get("network_lr_horizon_cap"),
        network_lr_floor=policy.get("network_lr_floor"))


class _ObservedNoise(image_tasks.ImageTask):
    """Screen-only receipt: the image problem, plus read-only observation of the
    training noise the runner actually applied (nothing here changes training).

    Output noise is the residual between the clean batch ``fake()`` returns and
    the noisy batch the runner hands ``views()``. Input noise is the residual
    between each view tensor and what the critic network itself receives (a
    forward pre-hook). Both are compared per update with the recipe schedule, so
    a missing, misplaced or mis-scaled noise draw fails the receipt.
    """

    SIGMAS = 6.  # tolerance in sampling s.d. of a residual std (~1/sqrt(2 * elements))

    def __init__(self, spec):
        super().__init__(spec)
        self._clean, self._views, self._out, self._in = None, [], [], []
        self.steps, self.input_sigmas, self.output_calls, self.input_calls = 0, [], 0, 0
        self.mismatches = []

    def networks(self, recipe, seed):
        nets = super().networks(recipe, seed)

        def critic_input(module, args):
            # The first critic call per view tensor; penalty calls come after.
            if self._views:
                x = args[0].detach()
                residuals = [self._residual(x, view) for view in self._views]
                k = min(range(len(residuals)), key=lambda i: residuals[i][0])
                self._views.pop(k)
                self._in.append(residuals[k])
        nets.critics.register_forward_pre_hook(critic_input)
        return nets

    @staticmethod
    def _residual(noisy, clean):
        return float((noisy - clean).std()), noisy.numel()

    def fake(self, nets, n, stream, real):
        sample = super().fake(nets, n, stream, real)
        if real is not None:  # training draw (metrics enumerate the table instead)
            self._clean = sample.x.detach()
        return sample

    def views(self, nets, real, fake):
        if self._clean is not None:
            self._out.append(self._residual(fake.x.detach(), self._clean))
            self._clean = None
        self._views = [real.x.detach(), fake.x.detach()]
        return super().views(nets, real, fake)

    def witness(self, step, toy):
        from particlegan.training import input_noise_std, output_noise_std
        index, out, into = step - 1, self._out, self._in
        self._out, self._in, self.steps = [], [], self.steps + 1

        def matches(observed, count, expected):  # per update: 2 fake batches, 4 critic inputs
            return len(observed) == count and all(
                r == 0 if expected == 0 else abs(r / expected - 1) <= self.SIGMAS / math.sqrt(2 * n)
                for r, n in observed)
        sigma_in, sigma_out = input_noise_std(toy.recipe, index), output_noise_std(toy.recipe, index)
        self.input_sigmas.append(max((r for r, _ in into), default=0.))
        self.output_calls += sum(r > 0 for r, _ in out)
        self.input_calls += sum(r > 0 for r, _ in into)
        if not (matches(out, 2, sigma_out) and matches(into, 4, sigma_in)):
            self.mismatches.append(dict(step=step, input=[r for r, _ in into], output=[r for r, _ in out],
                                        expected_input=sigma_in, expected_output=sigma_out))


def _image_noise_receipt(problem, noise, result):
    """What the witness observed while the runner trained (not the recipe's claim)."""
    matched = not problem.mismatches
    receipt = dict(source="observed: residuals of the noisy vs clean generator batch and critic input, per update",
                   step_calls=problem.steps,
                   train_input_applied=matched and problem.input_calls > 0,
                   train_output_applied=matched and problem.output_calls > 0,
                   input_nonzero_steps=sum(s > 0 for s in problem.input_sigmas),
                   input_train_calls=problem.input_calls,
                   output_nonzero_steps=problem.output_calls // 2, output_train_calls=problem.output_calls,
                   schedule_mismatches=len(problem.mismatches),
                   first_schedule_mismatches=problem.mismatches[:4],
                   output_noise_learnable=False, output_scale_parameter_count=0,
                   eval_scope="finite particle table without training noise")
    if noise.get("output_noise_rng") == "isolated":
        receipt.update(output_noise_rng="runner stream",
                       output_noise_eval_state_preserved=result.get("eval_streams_preserved") is True)
    return receipt


def run_image(spec, base, noise, *, model_policy=None, log=None):
    started = time.perf_counter()
    torch.set_num_threads(1)
    recipe = image_recipe(spec, base, noise, model_policy)
    applied, shapes = image_tasks.receipts(spec, recipe)
    problem = _ObservedNoise(spec)
    result = image_tasks.train(spec, recipe, problem=problem, log=log, witness=problem.witness)
    result["seconds"] = time.perf_counter() - started
    return result, dict(applied=applied, shapes=shapes, host_recipe=recipe,
                        noise_receipt=_image_noise_receipt(problem, noise, result))


def run(config_path: Path, output: Path, *, tasks=VECTOR_NAMES):
    if output.exists():
        raise FileExistsError("use a new output directory")
    config_path = config_path.resolve()
    config_bytes = config_path.read_bytes()
    config = load_config(config_path)
    base, noise, resource_overrides = declared_recipe(config)
    model_policy = declared_model_policy(config)
    jobs, profile = load_declaration()
    tasks = tuple(tasks)
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("select distinct frozen tasks")
    unknown = set(tasks) - {job["spec"]["name"] for job in jobs}
    if unknown:
        raise ValueError(f"unknown frozen tasks: {sorted(unknown)}")
    selected_jobs = [job for job in jobs if job["spec"]["name"] in tasks]
    package = public_module_manifest()
    output.mkdir(parents=True)
    (output / "episodes").mkdir()
    protocol = suite.snapshot(output)
    noise_hash = hashlib.sha256(NOISE_SOURCE.read_bytes()).hexdigest()
    protocol["source_sha256"][str(NOISE_SOURCE.relative_to(ROOT))] = noise_hash
    shutil.copyfile(NOISE_SOURCE, output / "noise_source.py")
    (output / config_path.name).write_bytes(config_bytes)
    protocol.update(version="toy100-compatibility-screen-v1", seed=0, device=str(host_device()),
                    threads=1, fixed_tasks=list(tasks), jobs=selected_jobs,
                    global_recipe=legacy_dict(base), noise=noise,
                    ignored_toy100_resource_overrides=resource_overrides,
                    frozen_discriminators=profile["discriminators"],
                    config_file=config_path.name,
                    config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                    noise_source_sha256=noise_hash,
                    public_package=package)
    if model_policy:
        protocol["model_policy"] = model_policy
    write(output / "protocol.json", protocol)
    records = []
    for job in selected_jobs:
        suite.verify_source(protocol)
        if hashlib.sha256(config_path.read_bytes()).hexdigest() != protocol["config_sha256"]:
            raise RuntimeError("candidate configuration changed during screening")
        spec, card, variant = declared_spec(job, profile, base)
        route = ("benchmarks.toy_runner + recipe noise/schedule" if spec["runner"] in ("vector", "image")
                 else problem_hosts.ROUTE if problem_hosts.is_migrated(spec["name"])
                 else "host-owned optimizers + shared noise policy")
        print(f'START {spec["name"]} route={route} steps={spec["steps"]}', flush=True)
        started = time.perf_counter()
        try:
            if spec["runner"] == "vector":
                result, context = run_vector(
                    spec, card, base, noise, model_policy=model_policy,
                    log_path=output / "logs" / f'{spec["name"]}.jsonl',
                )
            elif spec["runner"] == "image":
                result, context = run_image(
                    spec, base, noise, model_policy=model_policy,
                    log=lambda row: print(json.dumps(row, default=float), flush=True),
                )
            else:
                result, context = run_noisy_legacy(
                    spec, base, noise, model_policy=model_policy,
                    log_path=output / "logs" / f'{spec["name"]}.log',
                )
            observations = result.get("observations", result.get("curve", []))
            if len(observations) != 24:
                raise RuntimeError("incomplete frozen transfer curve")
            if spec["runner"] == "vector" and len(result["actions"]) != spec["steps"]:
                raise RuntimeError("incomplete transfer action trace")
            json.dumps(result, allow_nan=False)
        except Exception:
            result = dict(error=traceback.format_exc(), seconds=time.perf_counter() - started)
            context = dict(applied=[], shapes={}, host_recipe=(
                base if spec["runner"] == "legacy" else host_recipe(base, spec)))
        receipt = context.get("noise_receipt")
        if spec["runner"] == "vector" and not result.get("error"):
            receipt = _runner_noise_receipt(context, noise, spec, result)
            noise_applied = (
                receipt["step_calls"] == spec["steps"]
                and receipt["recipe_noise"] == {key: noise[key] for key in RUNNER_NOISE_FIELDS}
                and (noise["output_noise_std"] == 0 or receipt["output_nonzero_steps"] > 0)
                and (noise["input_noise_std"] == 0 or receipt["input_nonzero_steps"] > 0)
            )
        elif receipt is not None and not result.get("error"):
            noise_applied = (
                receipt["step_calls"] == spec["steps"]
                and (noise["output_noise_std"] == 0 or
                     receipt["train_output_applied"])
                and (noise["input_noise_std"] == 0 or
                     receipt["train_input_applied"]
                     and receipt["input_nonzero_steps"] > 0)
                and (not noise.get("output_noise_learnable", False) or
                     receipt.get("output_scale_parameter_count") == 1
                     and receipt.get("output_scale_optimizer_owned"))
                and (noise.get("output_noise_rng") != "isolated" or
                     receipt.get("output_noise_eval_state_preserved")
                     and receipt.get("output_train_calls", 0) > 0
                     and (receipt.get("eval_scope") not in (
                         "generated_samples", "generated_and_reconstructed_samples",
                     ) or receipt.get("output_eval_calls", 0) > 0))
            )
        else:
            noise_applied = False
        verdict = test_verdict(spec, result)
        ema = ema_verdict(spec, result)
        record = dict(name=spec["name"], route=route,
                      noise_applied=noise_applied,
                      noise_receipt=receipt,
                      recipe=legacy_dict(base), host_recipe=legacy_dict(context["host_recipe"]),
                      noise=noise, original_spec=deepcopy(job["spec"]), spec=spec,
                      discriminator_variant=variant,
                      architecture=variant["name"] if variant else job["architecture"],
                      reference=job["reference"], reference_sha256=job["reference_sha256"],
                      applied=context["applied"], shapes=context["shapes"],
                      recipe_owned=context.get("recipe_owned", spec["runner"] != "legacy"),
                      verdict=verdict, ema_verdict=ema, result=result,
                      source_sha256=protocol["source_sha256"])
        if "executed_recipe" in context:
            record["executed_recipe"] = context["executed_recipe"]
        if model_policy:
            record["model_policy"] = model_policy
        raw = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()
        artifact = f'episodes/{base.name}__{spec["name"]}.json.gz'
        (output / artifact).write_bytes(gzip.compress(raw, mtime=0))
        records.append({k: v for k, v in record.items() if k not in ("result", "source_sha256")}
                       | dict(artifact=artifact,
                              uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
                              live=result.get("live"), ema=result.get("ema"),
                              observations=len(result.get("observations", result.get("curve", []))),
                              seconds=result["seconds"]))
        write(output / "index.json", dict(records=records))
        print(json.dumps(dict(event="DONE", task=spec["name"],
                              status=verdict["status"],
                              suffix=verdict.get("convergence", {}).get("passing_suffix"),
                              live=result.get("live"), error=result.get("error"))), flush=True)
    suite.verify_source(protocol)
    passed = sum(row["verdict"]["passed"] for row in records)
    complete_vector_screen = set(tasks) == set(VECTOR_NAMES)
    complete_all19 = len(tasks) == len(jobs) and set(tasks) == {
        job["spec"]["name"] for job in jobs}
    # A host that still owns its optimizers cannot carry the common recipe.
    full_mechanism = all(row["noise_applied"] and row["recipe_owned"] for row in records)
    summary = dict(version=protocol["version"],
         attempted=len(records), passed=passed,
         host_owned_optimizers=[row["name"] for row in records if not row["recipe_owned"]],
         overall=("PASS" if passed == len(tasks) else "FAIL")
         if complete_vector_screen or complete_all19 and full_mechanism else "INCOMPLETE",
         subset_passed=passed == len(tasks),
         scope="six-vector full mechanism" if complete_vector_screen
         else "all-19 full mechanism" if complete_all19 and full_mechanism
         else "declared subset; full mechanism requires receipts for every selected host",
         full_mechanism=full_mechanism,
         tasks=list(tasks),
         config_sha256=protocol["config_sha256"],
         global_recipe=legacy_dict(base), noise=noise,
         cases=[dict(name=row["name"], live=row["verdict"]["status"],
                     ema=row["ema_verdict"]["status"],
                     observations=row["observations"], live_final=row["live"],
                     route=row["route"], noise_applied=row["noise_applied"],
                     recipe_owned=row["recipe_owned"], noise_receipt=row["noise_receipt"],
                     artifact=row["artifact"]) for row in records])
    if model_policy:
        summary["model_policy"] = model_policy
    write(output / "summary.json", summary)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remaining", action="store_true",
                        help="screen the four image and nine custom hosts with shared noise")
    parser.add_argument("--all", action="store_true",
                        help="screen all 19 with shared noise on every host")
    parser.add_argument("--tasks", nargs="+", help="bounded named-task screen (always INCOMPLETE)")
    add_device_argument(parser)
    parser.add_argument("--init", default=None,
                        help="deterministic weight and particle init name; omit to keep the PyTorch init")
    args = parser.parse_args()
    torch.set_num_threads(1)  # the screen's declared protocol (threads=1) for every route
    apply_device_policy(args.device, log=True)
    from benchmarks.init_research.init_registry import use_init
    use_init(args.init)
    if sum(bool(x) for x in (args.remaining, args.all, args.tasks)) > 1:
        parser.error("--remaining, --all and --tasks are mutually exclusive")
    all_names = tuple(job["spec"]["name"] for job in load_declaration()[0])
    run(args.config, args.output, tasks=args.tasks or
        (REMAINING_NAMES if args.remaining else all_names if args.all else VECTOR_NAMES))
