"""Frozen transfer hosts that are problem-only toys, run on the shared runner.

``HOSTS`` names each frozen host's ``benchmarks.toy_runner.ToyProblem`` class
explicitly as (module, class). A host is *migrated* once that attribute exists
and is a ``ToyProblem``; its no-argument construction is the frozen host arm.
The problem's own ``name`` need not equal the host name (``TwoPole.name`` is
``"locked_two_pole"``; ``Unipolar`` sets it per instance), so it is never used
for lookup. ``benchmarks.locked_shared.mode_hold.ModeHold`` is the reference.

The harness contributes only the recipe under test: its global fields, the
declared noise and the model policy's network horizon, at the problem's own
task shape and prior/encoder structure (``problem.recipe()``'s z_dim /
num_particles / batch_size / total_steps, and prior_kind / sigma_rel /
encoder_mode, which ``AEGanHold``'s ``ae_gan`` preset sets). ``benchmarks.toy_runner.ToyRun`` builds the optimizers (which own
the LR schedule), loss, critic penalty, prior, noise and EMA from it; the
harness only calls ``step()`` and records the frozen 24 observations.

Receipts are read from the public ``ToyRun`` after training: each recipe-built
optimizer's saved schedule state, group roles and base rates. Nothing is
recomputed from LR formulas. Hosts that are not migrated yet still own their
optimizers and are reported as such (``recipe_owned`` is False); see
``legacy_noise_adapters``.
"""
from __future__ import annotations

from importlib import import_module
import json
import math
from pathlib import Path
import time

from benchmarks import toy_runner
from benchmarks.locked_shared.observation import checkpoint, recording, sustained
from benchmarks.toy100.device import host_device
from benchmarks.toy100.schedule import policy_recipe
from particlegan.training import input_noise_std, output_noise_std

from .protocol import requirements

ROUTE = "problem-only toy on benchmarks.toy_runner"
TASK_SHAPE = ("z_dim", "num_particles", "batch_size", "total_steps")
# The prior/encoder structure the problem's networks are built against (a MoG
# prior and AE encoder for ``ae_gan``); never a training setting under test.
STRUCTURE = ("prior_kind", "sigma_rel", "encoder_mode")
NOISE_FIELDS = ("output_noise_std", "input_noise_std", "input_noise_anneal_end", "output_noise_warmup")
# Frozen host name -> (module, ToyProblem class).
HOSTS = {
    "two_pole": ("benchmarks.locked_shared.two_pole", "TwoPole"),
    "trajectory": ("benchmarks.locked_shared.trajectory", "Trajectory"),
    "mode_hold": ("benchmarks.locked_shared.mode_hold", "ModeHold"),
    "residual_student": ("benchmarks.locked_shared.hosts.residual_student", "ResidualStudent"),
    "unipolar": ("benchmarks.locked_shared.hosts.unipolar", "Unipolar"),
    "ae_gan_hold": ("benchmarks.locked_shared.hosts.ae_gan_hold", "AEGanHold"),
    "cover_leftover": ("benchmarks.locked_shared.hosts.cover_leftover", "CoverLeftover"),
    "unused_token_hold": ("benchmarks.locked_shared.hosts.unused_token_hold", "UnusedTokenHold"),
    "mid_scale_identity": ("benchmarks.locked_shared.hosts.mid_scale_identity", "MidScaleIdentity"),
}


def problem_class(name: str):
    """The host's ``ToyProblem`` class, or None while the host owns its training loop."""
    if name not in HOSTS:
        return None
    module, attribute = HOSTS[name]
    value = getattr(import_module(module), attribute, None)
    return value if isinstance(value, type) and issubclass(value, toy_runner.ToyProblem) else None


def is_migrated(name: str) -> bool:
    return problem_class(name) is not None


def problem_recipe(problem, base, noise: dict | None = None, model_policy: dict | None = None):
    """``base`` at the problem's task shape and structure, with the declared noise and horizon.

    ``noise=None`` keeps ``base``'s own noise fields (the public-default control).
    """
    noise = {} if noise is None else dict(noise)
    unsupported = [key for key in ("output_noise_learnable", "output_noise_rng") if noise.get(key)]
    if unsupported:
        raise ValueError(f"the shared runner has no {unsupported} option; problem hosts take "
                         "output noise from the recipe on the runner's own noise stream")
    shape = problem.recipe()
    recipe = base.replace(**{key: getattr(shape, key) for key in TASK_SHAPE + STRUCTURE},
                          **{key: float(noise.get(key, 0.0)) for key in NOISE_FIELDS if noise})
    policy = model_policy or {}
    if policy.get("network_lr_horizon_cap") is not None:
        recipe = policy_recipe(recipe, policy["network_lr_horizon_cap"],
                               network_lr_floor=policy.get("network_lr_floor"))
    return recipe


def optimizer_receipts(toy: toy_runner.ToyRun) -> list[dict]:
    """What each recipe-built optimizer of a ``ToyRun`` holds: schedule state and groups."""
    def receipt(role, optimizer):
        return dict(role=role, optimizer=type(optimizer).__name__,
                    lr_schedule=optimizer.lr_schedule.state_dict(),
                    groups=[dict(role=group["role"], base_lr=group["base_lr"], lr=group["lr"],
                                 betas=list(group["betas"]),
                                 parameters=sum(p.numel() for p in group["params"]))
                            for group in optimizer.param_groups])
    return [receipt("generator", toy.opt_g), *(receipt(name, opt) for name, opt in toy.opt_d.items())]


def noise_receipt(recipe, completed_steps: int, eval_scope: str) -> dict:
    """The runner's noise, from its own recipe schedule functions (no draws are counted)."""
    steps = recipe.total_steps
    inputs = [input_noise_std(recipe, step) for step in range(steps)]
    outputs = [output_noise_std(recipe, step) for step in range(steps)]
    return dict(source="recipe", eval_scope=eval_scope,
                output_std=recipe.output_noise_std, input_std=recipe.input_noise_std,
                input_anneal_end=recipe.input_noise_anneal_end,
                output_noise_warmup=recipe.output_noise_warmup, total_steps=steps,
                step_calls=completed_steps,
                input_sigma_first=inputs[0], input_sigma_last=inputs[-1],
                input_nonzero_steps=sum(value > 0 for value in inputs),
                output_sigma_first=outputs[0], output_sigma_last=outputs[-1],
                output_nonzero_steps=sum(value > 0 for value in outputs),
                train_input_applied=any(value > 0 for value in inputs),
                train_output_applied=any(value > 0 for value in outputs),
                output_noise_learnable=False, output_scale_parameter_count=0)


def _jsonl(path):
    """One JSON line per observation (``tail -f``), or a no-op without a path."""
    if path is None:
        return lambda row: None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")

    def emit(row):
        with path.open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False, default=float) + "\n")
    return emit


def run_problem(spec: dict, base, noise: dict | None = None, *, model_policy: dict | None = None,
                eval_scope: str = "generated_samples", log_path: str | Path | None = None):
    """Train one migrated host on a public ``ToyRun``; ``(result, context)`` in harness form."""
    started = time.perf_counter()
    problem = problem_class(spec["name"])()
    recipe = problem_recipe(problem, base, noise, model_policy)
    steps = spec["steps"]
    if recipe.total_steps != steps:
        raise ValueError(f"{spec['name']}: problem budget {recipe.total_steps} != frozen {steps}")
    toy = toy_runner.ToyRun(problem, recipe=recipe, seed=0, device=host_device())
    emit = _jsonl(log_path)
    with recording(steps) as recorder:
        for _ in range(steps):
            losses = {k: float(v) for k, v in toy.step().items() if k != "step"}
            if not all(math.isfinite(value) for value in losses.values()):
                raise FloatingPointError(f"{spec['name']}: non-finite loss at step {toy.completed_steps}")
            observed = len(recorder.curve)
            checkpoint(toy.completed_steps, lambda: {**toy.measure(), "ema": toy.measure(ema=True)})
            if len(recorder.curve) > observed:
                emit({"toy": spec["name"], **recorder.curve[-1], **losses})
    live, ema = toy.measure(), toy.measure(ema=True)
    emit({"toy": spec["name"], "event": "final", "live": live, "ema": ema})
    rules = requirements(spec)
    keys = [key for key, _, _ in rules]
    result = dict(live={k: live[k] for k in keys if k in live},
                  ema={k: ema[k] for k in keys if k in ema},
                  observations=recorder.curve,
                  convergence=sustained(recorder.curve, rules, expected_steps=recorder.steps),
                  optimizers=optimizer_receipts(toy), seconds=time.perf_counter() - started)
    context = dict(applied=[], shapes={"host": ROUTE, "problem": problem.name},
                   host_recipe=base, executed_recipe=recipe.to_dict(),
                   noise_receipt=noise_receipt(recipe, toy.completed_steps, eval_scope),
                   eval_scope=eval_scope)
    return result, context


def check_receipts(record: dict, base, noise: dict | None, model_policy: dict | None) -> None:
    """Regrade a problem-host episode from its optimizers' saved schedule state."""
    name, steps = record["spec"]["name"], record["spec"]["steps"]
    cls = problem_class(name)
    if cls is None:
        raise ValueError(f"problem-host record for a host that declares no ToyProblem: {name}")
    recipe = problem_recipe(cls(), base, noise, model_policy)
    if record.get("executed_recipe") != json.loads(json.dumps(recipe.to_dict())):
        raise ValueError(f"problem-host recipe differs from the declared recipe at its task shape: {name}")
    receipts = record["result"].get("optimizers")
    if not isinstance(receipts, list) or not receipts or receipts[0].get("role") != "generator":
        raise ValueError(f"problem-host optimizer receipts are absent: {name}")
    rates = {("generator", "network"): recipe.lr, ("generator", "prior"): recipe.lr * recipe.prior_lr_mult,
             ("critic", "network"): recipe.lr * recipe.d_lr_mult}
    prior_betas = list(recipe.prior_betas if recipe.prior_betas is not None else recipe.betas)
    for item in receipts:
        side = "generator" if item["role"] == "generator" else "critic"
        expected_type = "K3PGeneratorAdam" if side == "generator" else "K3PCriticAdam"
        if item.get("optimizer") != expected_type:
            raise ValueError(f"problem-host optimizer is not recipe-built: {name}.{item['role']}")
        if (item.get("lr_schedule") or {}).get("completed_steps") != steps:
            raise ValueError(f"problem-host schedule did not apply every update: {name}.{item['role']}")
        for group in item.get("groups") or [None]:
            if not isinstance(group, dict) or (side, group.get("role")) not in rates:
                raise ValueError(f"problem-host param group is invalid: {name}.{item['role']}")
            betas = prior_betas if group["role"] == "prior" else list(recipe.betas)
            if (group.get("base_lr") != rates[side, group["role"]] or group.get("betas") != betas
                    or type(group.get("parameters")) is not int or group["parameters"] <= 0):
                raise ValueError(f"problem-host base rate differs from recipe: {name}.{item['role']}")
