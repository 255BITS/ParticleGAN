"""Screen a declared 100-mode recipe on the six frozen 2D transfer toys.

The six hosts retain their original data, initialization order, architecture
cards, resource sizes, budgets, and live gates. Only global optimizer/loss
settings and the two target-agnostic noise wrappers come from the supplied
100-mode configuration. This is a transfer screen, not part of the canonical
19-case public-default verification or a search over host settings.

python -u -m benchmarks.transfer_suite.toy100_compatibility \
    --config configs/toy100/shared_candidate.json --output /tmp/toy100-vector-screen
"""
from __future__ import annotations

import argparse
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

from benchmarks.toy100.models import (
    InputNoise, OutputNoise, linear_input_noise,
    linear_output_noise as output_noise_at,
)
from benchmarks.toy100.train import load_config, resolve_config
from lib.toy_models import SimpleMLPGenerator
from particlegan import GANTrainer, get_recipe

from . import image_tasks, suite, vector_tasks
from .compare_defaults import ema_verdict
from .legacy_noise_adapters import run_legacy as run_noisy_legacy
from .protocol import test_verdict
from .public_default_verification import (
    GLOBAL_RECIPE_FIELDS, declared_spec, host_recipe, load_declaration,
    optimizer_receipts, public_module_manifest, rate_action,
    shape_receipt, vector_discriminator, write,
)


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
        if config["toy100_model"] != "affine_square_v1":
            raise ValueError("unsupported toy100_model")
        policy["toy100_model"] = "affine_square_v1"
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
    resolved, _ = resolve_config(candidate)
    globals_only = {name: resolved[name] for name in GLOBAL_RECIPE_FIELDS}
    recipe = get_recipe(**globals_only).replace(name=str(config.get("name", "toy100_transfer")))
    noise = {name: resolved[name] for name in
             ("output_noise_std", "input_noise_std", "input_noise_anneal_end")}
    noise["output_noise_warmup"] = float(output_warmup)
    # Omit the optional false value so archived fixed-noise protocols keep
    # their original canonical noise identity. True is always explicit.
    if resolved.get("output_noise_learnable", False):
        noise["output_noise_learnable"] = True
    return recipe, noise, resource_overrides


def _effective_output_std(model) -> float:
    value = model.effective_std()
    return float(value.detach()) if isinstance(value, torch.Tensor) else float(value)


def _native_noise_receipt(context, noise, spec, result):
    trainer = context["trainer"]
    learned = noise.get("output_noise_learnable", False)
    scale = trainer.G.output_scale if learned else None
    scalar = scale.raw_scale if scale is not None else None
    ownership = bool(
        scalar is not None
        and sum(p is scalar for group in trainer.opt_g.param_groups for p in group["params"]) == 1
        and all(p is not scalar for group in trainer.opt_d.param_groups for p in group["params"])
        and any(p is scalar for p in trainer.opt_g.param_groups[0]["params"])
        and all(p is not scalar for p in trainer.opt_g.param_groups[1]["params"])
    )
    trace = [action["output_sigma_effective"] for action in result["actions"]]
    receipt = dict(
        output_module=type(trainer.G).__name__,
        input_module=type(trainer.D).__name__,
        input_nonzero_steps=sum(action["input_sigma"] > 0 for action in result["actions"]),
        output_nonzero_steps=sum(action["output_sigma"] > 0 for action in result["actions"]),
        output_sigma_first=result["actions"][0]["output_sigma"],
        output_sigma_last=result["actions"][-1]["output_sigma"],
        output_sigma_final_evaluation=output_noise_at(
            noise["output_noise_std"], spec["steps"], spec["steps"],
            noise.get("output_noise_warmup", 0.0),
        ),
        output_noise_learnable=learned,
        output_scale_parameter_count=scalar.numel() if scalar is not None else 0,
        output_scale_optimizer_owned=ownership,
        generator_base_parameters=context["generator_base_parameters"],
        generator_total_parameters=sum(p.numel() for p in trainer.G.parameters()),
        output_scale_initial=context["output_scale_initial"],
        output_scale_final=float(scale().detach()) if scale is not None else None,
        output_sigma_effective_first=trace[0],
        output_sigma_effective_last=trace[-1],
        output_sigma_effective_final_evaluation=_effective_output_std(trainer.G)
            if noise["output_noise_std"] else 0.0,
        output_sigma_effective_step_trace=trace,
        step_calls=len(result["actions"]),
    )
    if learned:
        receipt["output_scale_ema_final"] = float(trainer.ema_G.output_scale().detach())
        receipt["output_sigma_effective_ema_final_evaluation"] = _effective_output_std(
            trainer.ema_G,
        )
    return receipt


def setup_vector(spec, card, base, noise):
    cfg = vector_tasks.resolve(spec)
    if cfg["d_every"] != 1 or cfg["g_every"] != 1:
        raise ValueError("GANTrainer route requires one G and D update per frozen step")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    data_rng = torch.Generator().manual_seed(0)
    latent_rng = torch.Generator().manual_seed(1)
    penalty_rng = torch.Generator().manual_seed(2)
    recipe = host_recipe(base, spec)
    prior = recipe.make_prior(init_std=.5, generator=torch.Generator().manual_seed(0))
    generator = SimpleMLPGenerator(cfg["z_dim"], cfg["hidden"], cfg["layers"], 2)
    generator_base_parameters = sum(p.numel() for p in generator.parameters())
    discriminator = vector_discriminator(spec, card)
    if noise["output_noise_std"]:
        generator = OutputNoise(
            generator, noise["output_noise_std"],
            learnable=noise.get("output_noise_learnable", False),
        )
    output_scale_initial = (float(generator.output_scale().detach())
                            if noise.get("output_noise_learnable", False) else None)
    if noise["input_noise_std"]:
        discriminator = InputNoise(discriminator, seed=901, device=torch.device("cpu"))
    trainer = GANTrainer(recipe, generator, discriminator, prior=prior, seed=0,
                         latent_generator=latent_rng, penalty_generator=penalty_rng)
    # Shape probing must not consume the generator's training-noise stream.
    with torch.random.fork_rng(devices=[]):
        shapes = shape_receipt(trainer, cfg["batch"], (2,))
    # Match the noise wrapper's declared fixed-seed stream after construction.
    if noise["output_noise_std"]:
        torch.manual_seed(0)
    return dict(trainer=trainer, cfg=cfg, data_rng=data_rng, shapes=shapes,
                applied=optimizer_receipts(trainer), host_recipe=recipe,
                generator_base_parameters=generator_base_parameters,
                output_scale_initial=output_scale_initial)


def run_vector(spec, card, base, noise, *, model_policy=None):
    from benchmarks.locked_shared.observation import sustained

    started = time.perf_counter()
    context = setup_vector(spec, card, base, noise)
    trainer, cfg, data_rng = context["trainer"], context["cfg"], context["data_rng"]
    expected = {math.ceil(i * cfg["steps"] / 24) for i in range(1, 25)}
    observations, actions, losses = [], [], []
    for completed in range(1, cfg["steps"] + 1):
        output_sigma = output_noise_at(
            noise["output_noise_std"], trainer.completed_steps, cfg["steps"],
            noise["output_noise_warmup"],
        )
        if noise["output_noise_std"]:
            trainer.G.std = output_sigma
            trainer.ema_G.std = output_sigma
        output_sigma_effective = (
            _effective_output_std(trainer.G) if noise["output_noise_std"] else 0.0
        )
        sigma = linear_input_noise(
            noise["input_noise_std"], trainer.completed_steps, cfg["steps"],
            noise["input_noise_anneal_end"],
        )
        if noise["input_noise_std"]:
            trainer.D.sigma = sigma
        real = vector_tasks.sample_target(cfg, cfg["batch"], data_rng, completed)
        real_g = lambda: vector_tasks.sample_target(cfg, cfg["batch"], data_rng, completed)
        cap = (model_policy or {}).get("network_lr_horizon_cap")
        network_floor = (model_policy or {}).get("network_lr_floor")
        if cap is None:
            stats = trainer.step(real, generator_real=real_g)
            action = rate_action(trainer, completed)
        else:
            from benchmarks.toy100.schedule import policy_rate_action, step_with_policy
            stats = step_with_policy(
                trainer, real, generator_real=real_g,
                network_lr_horizon_cap=cap,
                network_lr_floor=network_floor,
            )
            action = dict(step=completed) | policy_rate_action(
                trainer, completed, network_lr_horizon_cap=cap,
                network_lr_floor=network_floor,
            )
        if not all(torch.isfinite(value) for key, value in stats.items()
                   if key != "step" and isinstance(value, torch.Tensor)):
            raise FloatingPointError("nonfinite transfer-screen loss")
        actions.append(action |
                       {"input_sigma": sigma, "output_sigma": output_sigma,
                        "output_sigma_effective": output_sigma_effective})
        if completed in expected:
            if noise["output_noise_std"]:
                evaluated_sigma = output_noise_at(
                    noise["output_noise_std"], completed, cfg["steps"],
                    noise["output_noise_warmup"],
                )
                trainer.G.std = evaluated_sigma
                trainer.ema_G.std = evaluated_sigma
            with torch.no_grad(), torch.random.fork_rng(devices=[]):
                def measure(model, prior):
                    torch.manual_seed(402)
                    latent = prior.sample(vector_tasks.EVAL_SAMPLES,
                                          generator=torch.Generator().manual_seed(990))[0]
                    return vector_tasks.score_samples(model(latent), cfg, completed)

                live = measure(trainer.G, trainer.prior)
                ema = measure(trainer.ema_G, trainer.ema_prior)
            observations.append(dict(**live, ema=ema, step=completed,
                                     seconds=time.perf_counter() - started,
                                     **({"output_sigma_live": _effective_output_std(trainer.G),
                                         "output_sigma_ema": _effective_output_std(trainer.ema_G)}
                                        if noise.get("output_noise_learnable", False) else {})))
            losses.append(dict(step=completed, d=float(stats["loss_d"]),
                               g=float(stats["loss_g"]), penalty=float(stats["penalty"]),
                               prior=float(stats["prior_regularization"])))
    result = dict(live={k: v for k, v in observations[-1].items()
                        if k not in ("ema", "step", "seconds")},
                  ema=observations[-1]["ema"], observations=observations,
                  actions=actions, losses=losses,
                  update_counts=dict(g=trainer.completed_steps, d=trainer.completed_steps),
                  seconds=time.perf_counter() - started)
    result["convergence"] = sustained(observations, cfg["thresholds"],
                                       expected_steps=expected)
    return result, context


def setup_image(spec, base, noise):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    centers = image_tasks.templates(spec)
    recipe = host_recipe(base, spec)
    # Preserve the frozen host's G, D, then prior construction order.
    generator, discriminator = image_tasks.Generator(spec), image_tasks.Discriminator(spec)
    generator_base_parameters = sum(p.numel() for p in generator.parameters())
    prior = recipe.make_prior()
    if noise["output_noise_std"]:
        generator = OutputNoise(
            generator, noise["output_noise_std"],
            learnable=noise.get("output_noise_learnable", False),
        )
    output_scale_initial = (float(generator.output_scale().detach())
                            if noise.get("output_noise_learnable", False) else None)
    if noise["input_noise_std"]:
        discriminator = InputNoise(discriminator, seed=901, device=torch.device("cpu"))
    global_stream = torch.default_generator
    trainer = GANTrainer(recipe, generator, discriminator, prior=prior, seed=0,
                         latent_generator=global_stream, penalty_generator=global_stream)
    with torch.random.fork_rng(devices=[]):
        shapes = shape_receipt(trainer, spec["batch_size"], (1, 8, 8))
    if noise["output_noise_std"]:
        torch.manual_seed(0)
    return dict(trainer=trainer, centers=centers, shapes=shapes,
                applied=optimizer_receipts(trainer), host_recipe=recipe,
                generator_base_parameters=generator_base_parameters,
                output_scale_initial=output_scale_initial)


def run_image(spec, base, noise, *, model_policy=None):
    from benchmarks.locked_shared.observation import sustained

    started = time.perf_counter()
    context = setup_image(spec, base, noise)
    trainer, centers = context["trainer"], context["centers"]
    expected = image_tasks.evaluation_steps(spec)
    observations, actions, losses = [], [], []
    for completed in range(1, spec["steps"] + 1):
        output_sigma = output_noise_at(
            noise["output_noise_std"], trainer.completed_steps, spec["steps"],
            noise["output_noise_warmup"],
        )
        if noise["output_noise_std"]:
            trainer.G.std = output_sigma
            trainer.ema_G.std = output_sigma
        output_sigma_effective = (
            _effective_output_std(trainer.G) if noise["output_noise_std"] else 0.0
        )
        sigma = linear_input_noise(
            noise["input_noise_std"], trainer.completed_steps, spec["steps"],
            noise["input_noise_anneal_end"],
        )
        if noise["input_noise_std"]:
            trainer.D.sigma = sigma
        real = centers[torch.randint(len(centers), (spec["batch_size"],))]
        real = (real + spec["noise_std"] * torch.randn_like(real)).clamp(0., 1.)
        cap = (model_policy or {}).get("network_lr_horizon_cap")
        network_floor = (model_policy or {}).get("network_lr_floor")
        if cap is None:
            stats = trainer.step(real, generator_real=real)
            action = rate_action(trainer, completed)
        else:
            from benchmarks.toy100.schedule import policy_rate_action, step_with_policy
            stats = step_with_policy(
                trainer, real, generator_real=real,
                network_lr_horizon_cap=cap,
                network_lr_floor=network_floor,
            )
            action = dict(step=completed) | policy_rate_action(
                trainer, completed, network_lr_horizon_cap=cap,
                network_lr_floor=network_floor,
            )
        if not all(torch.isfinite(value) for key, value in stats.items()
                   if key != "step" and isinstance(value, torch.Tensor)):
            raise FloatingPointError("nonfinite transfer-screen loss")
        actions.append(action |
                       {"input_sigma": sigma, "output_sigma": output_sigma,
                        "output_sigma_effective": output_sigma_effective})
        if completed in expected:
            if noise["output_noise_std"]:
                evaluated_sigma = output_noise_at(
                    noise["output_noise_std"], completed, spec["steps"],
                    noise["output_noise_warmup"],
                )
                trainer.G.std = evaluated_sigma
                trainer.ema_G.std = evaluated_sigma
            live = image_tasks.measure(trainer.G, trainer.prior, centers, spec["thresholds"])
            ema = image_tasks.measure(trainer.ema_G, trainer.ema_prior,
                                      centers, spec["thresholds"])
            observations.append(dict(step=completed,
                                     seconds=time.perf_counter() - started,
                                     **live, ema=ema,
                                     **({"output_sigma_live": _effective_output_std(trainer.G),
                                         "output_sigma_ema": _effective_output_std(trainer.ema_G)}
                                        if noise.get("output_noise_learnable", False) else {})))
            losses.append(dict(step=completed, d=float(stats["loss_d"]),
                               g=float(stats["loss_g"]), penalty=float(stats["penalty"]),
                               prior=float(stats["prior_regularization"])))
    result = dict(live={k: v for k, v in observations[-1].items()
                        if k not in ("ema", "step", "seconds")},
                  ema=observations[-1]["ema"], observations=observations,
                  actions=actions, losses=losses,
                  update_counts=dict(g=trainer.completed_steps, d=trainer.completed_steps),
                  seconds=time.perf_counter() - started)
    result["convergence"] = sustained(
        observations,
        [("modes", ">=", spec["thresholds"]["modes"]),
         ("hq", ">=", spec["thresholds"]["hq_min"])],
        expected_steps=expected,
        minimum=spec["thresholds"]["minimum_stable_checks"],
    )
    return result, context


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
    protocol.update(version="toy100-compatibility-screen-v1", seed=0, device="cpu",
                    threads=1, fixed_tasks=list(tasks), jobs=selected_jobs,
                    global_recipe=base.to_dict(), noise=noise,
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
        route = ("GANTrainer + generic noise wrappers" if spec["runner"] in
                 ("vector", "image") else "public primitives custom host + shared noise policy")
        print(f'START {spec["name"]} route={route} steps={spec["steps"]}', flush=True)
        started = time.perf_counter()
        try:
            if spec["runner"] == "vector":
                result, context = run_vector(
                    spec, card, base, noise, model_policy=model_policy,
                )
            elif spec["runner"] == "image":
                result, context = run_image(
                    spec, base, noise, model_policy=model_policy,
                )
            else:
                result, context = run_noisy_legacy(
                    spec, base, noise, model_policy=model_policy,
                )
            observations = result.get("observations", result.get("curve", []))
            if len(observations) != 24:
                raise RuntimeError("incomplete frozen transfer curve")
            if spec["runner"] != "legacy" and len(result["actions"]) != spec["steps"]:
                raise RuntimeError("incomplete transfer action trace")
            json.dumps(result, allow_nan=False)
        except Exception:
            result = dict(error=traceback.format_exc(), seconds=time.perf_counter() - started)
            context = dict(applied=[], shapes={}, host_recipe=host_recipe(base, spec))
        receipt = context.get("noise_receipt")
        if spec["runner"] in ("vector", "image") and not result.get("error"):
            receipt = _native_noise_receipt(context, noise, spec, result)
            noise_applied = (
                receipt["step_calls"] == spec["steps"]
                and (noise["output_noise_std"] == 0 or
                     receipt["output_module"] == "OutputNoise")
                and (noise["input_noise_std"] == 0 or
                     receipt["input_module"] == "InputNoise"
                     and receipt["input_nonzero_steps"] > 0)
                and (not noise.get("output_noise_learnable", False) or
                     receipt["output_scale_parameter_count"] == 1
                     and receipt["output_scale_optimizer_owned"])
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
            )
        else:
            noise_applied = False
        verdict = test_verdict(spec, result)
        ema = ema_verdict(spec, result)
        record = dict(name=spec["name"], route=route,
                      noise_applied=noise_applied,
                      noise_receipt=receipt,
                      recipe=base.to_dict(), host_recipe=context["host_recipe"].to_dict(),
                      noise=noise, original_spec=deepcopy(job["spec"]), spec=spec,
                      discriminator_variant=variant,
                      architecture=variant["name"] if variant else job["architecture"],
                      reference=job["reference"], reference_sha256=job["reference_sha256"],
                      applied=context["applied"], shapes=context["shapes"],
                      verdict=verdict, ema_verdict=ema, result=result,
                      source_sha256=protocol["source_sha256"])
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
    full_mechanism = all(row["noise_applied"] for row in records)
    summary = dict(version=protocol["version"],
         attempted=len(records), passed=passed,
         overall=("PASS" if passed == len(tasks) else "FAIL")
         if complete_vector_screen or complete_all19 and full_mechanism else "INCOMPLETE",
         subset_passed=passed == len(tasks),
         scope="six-vector full mechanism" if complete_vector_screen
         else "all-19 full mechanism" if complete_all19 and full_mechanism
         else "declared subset; full mechanism requires receipts for every selected host",
         full_mechanism=full_mechanism,
         tasks=list(tasks),
         config_sha256=protocol["config_sha256"],
         global_recipe=base.to_dict(), noise=noise,
         cases=[dict(name=row["name"], live=row["verdict"]["status"],
                     ema=row["ema_verdict"]["status"],
                     observations=row["observations"], live_final=row["live"],
                     route=row["route"], noise_applied=row["noise_applied"],
                     noise_receipt=row["noise_receipt"],
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
    args = parser.parse_args()
    if sum(bool(x) for x in (args.remaining, args.all, args.tasks)) > 1:
        parser.error("--remaining, --all and --tasks are mutually exclusive")
    all_names = tuple(job["spec"]["name"] for job in load_declaration()[0])
    run(args.config, args.output, tasks=args.tasks or
        (REMAINING_NAMES if args.remaining else all_names if args.all else VECTOR_NAMES))
