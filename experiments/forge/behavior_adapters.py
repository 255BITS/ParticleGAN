"""Bind the eight frozen behavioral objectives to public ParticleGAN components.

The existing hosts retain their data, objectives, and measurement loops. Their
explicit ``components`` argument replaces construction/update primitives once;
there is no copied training loop and no global optimizer patch. These small CPU
hosts are separate from the scalar GANTrainer path used by distribution tasks.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

import torch
from torch import nn

from particlegan import init
from particlegan.recipes import learning_rate_scales
from particlegan.recipe_schedules import apply_optimizer_schedule
from particlegan.training import input_noise_std, output_noise_std
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy, _InputAdapter, _OutputAdapter

from .api import CapabilityError, FormulationContext, task_policy_blockers
from .contracts import atomic_json
from .mechanisms import MechanismAudit, mechanism_blockers
from .sampling import BEHAVIOR_POLICIES, POLICIES, executed_receipt


HOSTS = ("two_pole", "trajectory", "residual_student", "unipolar", "ae_gan_hold",
         "cover_leftover", "unused_token_hold", "mid_scale_identity")

# These resources/objectives belong to the frozen host, not the formulation.
# Even a matching explicit value must be removed from the candidate declaration:
# it cannot be a shared variable when another host owns a different value.
FROZEN_HOST_RECIPE_FIELDS = frozenset({
    "total_steps", "batch_size", "num_particles", "z_dim", "encoder_mode",
    "model", "num_classes", "conditioning", "ucd_target", "ucd_weight", "alpha_bar",
    "prior_reg", "reconstruction_weight", "observation_sigma",
})


def behavior_preflight(task: dict, candidate: dict) -> list[str]:
    """Return unsupported explicit overrides before constructing any host state."""
    fields = set(candidate.get("recipe_overrides", {})) & FROZEN_HOST_RECIPE_FIELDS
    if task["execution"].get("host", task["id"]) != "ae_gan_hold":
        fields |= set(candidate.get("recipe_overrides", {})) & {"routing_temperature", "distance_reduction"}
    return (task_policy_blockers(task, candidate)
            + [f"{task['id']}: recipe override {name!r} is owned by the frozen host; revise its task specification"
               for name in sorted(fields)])


def _base(module):
    return module.model if isinstance(module, (_InputAdapter, _OutputAdapter)) else module


def _observe_range(receipt, value):
    """Retain compact actual schedule observations, without a per-update stream."""
    value = float(value)
    if not receipt:
        receipt.update(observations=0, first=value, minimum=value, maximum=value)
    receipt.update(observations=receipt["observations"] + 1, last=value,
                   minimum=min(receipt["minimum"], value), maximum=max(receipt["maximum"], value))


class _OptimizerBundle:
    """One host G step over public optimizers with independently owned priors."""
    def __init__(self, optimizers):
        self.optimizers = optimizers
        self.param_groups = [g for optimizer in optimizers for g in optimizer.param_groups]

    def zero_grad(self, *args, **kwargs):
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        for optimizer in self.optimizers:
            optimizer.step(*args, **kwargs)

    def state_dict(self):
        return [optimizer.state_dict() for optimizer in self.optimizers]


class _ConditionedScore(nn.Module):
    def __init__(self, critic, scale):
        super().__init__()
        self.critic, self.scale = critic, scale

    def forward(self, values):
        return self.critic.score(values, self.scale)


class _PenaltyBinding:
    """Adapt old host call syntax to Recipe.make_critic_penalty's public API."""
    def __init__(self, recipe, optimizer, audit):
        self.bound = recipe.make_critic_penalty(optimizer, collect_stats=True)
        self.arm, self.coeff, self.kappa = recipe.reg_arm or recipe.critic_formulation, recipe.reg_coeff, recipe.reg_kappa
        self.norm, self.lazy_k, self.target_anneal = "rms", recipe.reg_every, "none"
        self.calls = 0
        self.audit = audit
        self.coefficient_observations = {}

    def penalty(self, critic, real, fake, step=None, **kwargs):
        self.calls += 1
        value = self.bound(critic, real, fake)
        _observe_range(self.coefficient_observations, self.bound.regularizer.coeff)
        self.audit.observe_penalty(self.bound.last_stats)
        return value, self.bound.last_stats

    def __call__(self, critic, real, fake, step=None, **kwargs):
        return self.penalty(critic, real, fake, step=step, **kwargs)[0]


class _NamedNoise(NoisePolicy):
    """Reuse the shared host wrappers, with public schedules and Forge streams."""
    def __init__(self, components):
        self.components = components
        recipe = components.recipe
        super().__init__(recipe.output_noise_std, recipe.input_noise_std,
                         recipe.input_noise_anneal_end, recipe.total_steps,
                         seed=components.context.streams.seed,
                         output_noise_warmup=recipe.output_noise_warmup,
                         output_noise_rng="isolated" if recipe.output_noise_std else None)
        streams = components.context.streams
        self.input_stream = streams.generator("noise", component="critic", purpose="input")
        self.output_stream = streams.generator("noise", component="generator", purpose="output")
        self._output_stream_initial_sha256 = hashlib.sha256(self.output_stream.get_state().numpy().tobytes()).hexdigest()
        for component, purpose in (("host", "global"), ("critic", "input"), ("generator", "output")):
            streams.generator("eval", component=component, purpose=purpose)

    def _output_sigma_for(self, completed_steps):
        return output_noise_std(self.components.recipe, completed_steps)

    def set_step(self, completed_steps):
        super().set_step(completed_steps)  # retain the host's mechanism-use counters
        self.input_sigma = input_noise_std(self.components.recipe, completed_steps)
        self.output_sigma = output_noise_std(self.components.recipe, completed_steps)
        return self.input_sigma

    def receipt(self):
        receipt = super().receipt()
        # The inherited policy's fixed seed offsets describe its own generators.
        # This adapter replaces them with named streams; report the generators
        # actually used, without drawing or resetting their current state.
        receipt.update(
            rng_derivation=self.components.context.streams.version,
            d_noise_seed=self.input_stream.initial_seed(),
            output_noise_seed=self.output_stream.initial_seed(),
            output_noise_seed_offset=None,
        )
        return receipt

    @contextmanager
    def evaluation(self, step):
        streams = self.components.context.streams
        before = streams.audit()
        old = self.input_stream, self.output_stream, self.input_sigma, self.output_sigma, self._evaluating
        try:
            with streams.preserve(), streams.fork("eval", component="host", purpose="global"):
                self.input_stream = streams.generator("eval", component="critic", purpose="input")
                self.output_stream = streams.generator("eval", component="generator", purpose="output")
                self.input_sigma = 0.0
                self.output_sigma = self._output_sigma_for(step)
                self._evaluating = True
                yield
        finally:
            self.input_stream, self.output_stream, self.input_sigma, self.output_sigma, self._evaluating = old
            self.components.rng_audits.append(streams.compare(before, streams.audit()))


class BehaviorComponents:
    """Explicit shared construction, optimizer, noise, and observation binder."""
    def __init__(self, request, task):
        self.task = task
        candidate = request.get("candidate", {})
        blockers = behavior_preflight(task, candidate)
        if blockers:
            raise CapabilityError(blockers)
        overrides = dict(candidate.get("recipe_overrides", {}))
        # These are frozen host resources; candidate mechanism settings stay shared.
        overrides["total_steps"] = task["execution"]["steps"]
        self.context = FormulationContext(
            recipe_preset=candidate.get("recipe_preset"), recipe_overrides=overrides,
            prior=task["execution"]["prior"],
            seed=request.get("protocol", {}).get("seed", 0), device="cpu",
            extensions=candidate.get("extensions", {}),
            initializer=candidate.get("initializer", "deterministic_orthogonal"),
            execution_path="public_components")
        extension_blockers = behavior_preflight(task, {"recipe_overrides": self.context.bindings["recipe"]})
        if extension_blockers:
            raise CapabilityError(extension_blockers)
        self.recipe = self.context.recipe
        if candidate.get("claim_contract", {}).get("learning") == "clockfree":
            raise CapabilityError(["these scheduled public components do not implement a clock-free optimizer"])
        self.models, self.optimizers, self.base_rates, self.role_parameters = {}, {}, {}, {}
        self.rng_audits, self.observations = [], []
        self.penalty = None
        self.schedule_observations = {}
        self.noise = _NamedNoise(self)
        self.bound = False

    def encoder_recipe(self, cfg):
        self.recipe = self.recipe.replace(encoder_mode="ae", z_dim=2, num_particles=cfg.n_particles,
                                          batch_size=cfg.batch)
        return self.recipe

    def make_prior(self, recipe):
        prior = recipe.make_prior(sigma=self.context.prior_config["sigma"],
                                  generator=self.context.streams.generator("init", component="prior", purpose="locations"))
        return prior

    @staticmethod
    def critic_copy(critic):
        return deepcopy(_base(critic))

    @staticmethod
    def conditioned_score(critic, scale):
        return _ConditionedScore(critic, scale)

    def _sample_binding(self, prior, component):
        original = prior.sample
        streams = self.context.streams
        streams.generator("eval", component=component, purpose="indices")
        streams.generator("eval", component=component, purpose="width")

        def sample(count, *args, generator=None, **kwargs):
            family = "eval" if self.noise._evaluating else "prior"
            kwargs["generator"] = streams.generator(family, component=component, purpose="indices")
            if self.context.prior_config["kind"] == "mog":
                kwargs["noise_generator"] = streams.generator(family, component=component, purpose="width")
            return original(count, *args, **kwargs)
        prior.sample = sample

    def bind(self, *, generator, critic, opt_g, opt_d, priors=(), encoder=None, direct_particles=()):
        if self.bound:
            raise RuntimeError("one component binder cannot own two host runs")
        self.bound = True
        critic = _base(critic)
        for role, model in (("generator", generator), ("encoder", encoder), ("discriminator", critic)):
            if model is None:
                continue
            model = _base(model)
            self.models[role] = model
            self.role_parameters[role] = list(model.parameters())
            if self.task["id"] != "two_pole":
                self.context.initialize(model, component=role)
        for index, prior in enumerate(priors):
            name = f"prior{index}"
            self.models[name] = prior
            self.context.initialize(prior, component=name)
            self._sample_binding(prior, name)
        prior_parameters = [p for prior in priors for p in prior.parameters()] + list(direct_particles)
        if prior_parameters:
            self.role_parameters["prior"] = prior_parameters
        parts = []
        g_parameters = self.role_parameters.get("generator", []) + self.role_parameters.get("encoder", [])
        if g_parameters:
            parts.append(self.recipe.make_generator_optimizer(g_parameters))
        for prior in priors:
            parts.append(self.recipe.make_generator_optimizer(
                [{"params": list(prior.parameters()), "lr": self.recipe.lr * self.recipe.prior_lr_mult,
                  "betas": self.recipe.prior_betas or self.recipe.betas, "forge_role": "prior"}], latent_table=prior.z))
        if direct_particles:
            parts.append(self.recipe.make_generator_optimizer(
                [{"params": list(direct_particles), "forge_role": "prior"}],
                direct_particles=list(direct_particles)))
        if not parts:
            raise CapabilityError(["host exposes no trainable generator-side component"])
        public_g = _OptimizerBundle(parts)
        public_d = self.recipe.make_critic_optimizer(critic, ema_critic=self.critic_copy(critic))
        self.optimizers = {"generator": public_g, "discriminator": public_d}
        for optimizer in self.optimizers.values():
            self.base_rates[id(optimizer)] = [group["lr"] for group in optimizer.param_groups]
        self.mechanism_audit = MechanismAudit(self.recipe, public_d, parts)
        self.penalty = _PenaltyBinding(self.recipe, public_d, self.mechanism_audit)
        # Host constructors may seed global RNG for fixed fixtures. Training
        # starts on the independent data stream after construction is complete.
        torch.set_rng_state(self.context.streams.generator("data", component="host", purpose="batches").get_state())
        return public_g, public_d, self.recipe.make_loss(), self.penalty

    def schedule_optimizer(self, optimizer, completed_updates):
        apply_optimizer_schedule(completed_updates, self.recipe, optimizer)
        network, prior = learning_rate_scales(completed_updates, self.recipe)
        for group, base in zip(optimizer.param_groups, self.base_rates[id(optimizer)]):
            group["lr"] = base * (prior if group.get("forge_role") == "prior" else network)
            role = next(name for name, owned in self.optimizers.items() if owned is optimizer)
            role = "prior" if group.get("forge_role") == "prior" else role
            observed = self.schedule_observations.setdefault(role, {"lr": {}, "beta2": {}})
            _observe_range(observed["lr"], group["lr"])
            _observe_range(observed["beta2"], group["betas"][1])

    def checkpoint(self, step, measure):
        budget = self.task["execution"]["steps"]
        expected = {math.ceil(i * budget / 24) for i in range(1, 25)}
        if step not in expected:
            return
        with self.noise.evaluation(step):
            values = measure()
        self.observations.append({**values, "step": step})
        print(json.dumps(dict(event="observation", task=self.task["id"], step=step,
                              budget=budget, metrics=values), allow_nan=False), flush=True)

    def guards(self):
        steps = {}
        all_state = {}
        for optimizer in self.optimizers.values():
            owned = optimizer.optimizers if isinstance(optimizer, _OptimizerBundle) else [optimizer]
            for sub in owned:
                all_state.update(sub.state)
        for role, params in self.role_parameters.items():
            observed = [int(all_state.get(p, {}).get("step", 0)) for p in params if p.requires_grad]
            steps[role] = min(observed) if observed else 0
        finite = all(torch.isfinite(p).all().item() and (p.grad is None or torch.isfinite(p.grad).all().item())
                     for params in self.role_parameters.values() for p in params)
        def finite_state(value):
            if isinstance(value, torch.Tensor):
                return bool(torch.isfinite(value).all())
            if isinstance(value, dict):
                return all(finite_state(v) for v in value.values())
            if isinstance(value, (list, tuple)):
                return all(finite_state(v) for v in value)
            return math.isfinite(value) if type(value) in (int, float) else True
        finite = finite and all(finite_state(optimizer.state_dict()) for optimizer in self.optimizers.values())
        mechanisms = self.mechanism_audit.receipt()
        return dict(all_finite=finite, optimizer_updates=steps,
                    hooks_exercised=not mechanism_blockers(mechanisms), mechanism_audit=mechanisms,
                    unintended_rng_deviations=sum(a["unintended_rng_deviations"] for a in self.rng_audits))

    def receipt(self):
        return dict(execution_path="public_components", recipe=asdict(self.recipe),
                    prior=self.context.prior_config, active_roles=sorted(self.role_parameters),
                    public_optimizers=[type(o).__name__ for opt in self.optimizers.values()
                                       for o in (opt.optimizers if isinstance(opt, _OptimizerBundle) else [opt])],
                    noise=self.noise.receipt(), rng=self.context.streams.manifest(), rng_audits=self.rng_audits,
                    training_schedules={"clock": "completed host updates; penalty observes completed critic updates",
                                        "horizon": self.recipe.total_steps,
                                        "optimizer_groups": deepcopy(self.schedule_observations),
                                        "penalty_coefficient": deepcopy(self.penalty.coefficient_observations)
                                        if self.penalty is not None else {}},
                    checkpoint_scope="component states; these hosts do not support continuation",
                    mechanism_applicability={"a2": ("disabled by latent_damping_max_rate=0" if self.recipe.latent_damping_max_rate == 0 else
                                               "installed on prior tables; activation is recorded in mechanism_audit" if any(n.startswith("prior") for n in self.models)
                                               else "not applicable: host has no latent-table optimizer"),
                                             "critic_guard": "disabled" if self.recipe.d_guard_ratio == 0 else
                                             "inactive before its declared minimum steps; see targeted component check" if self.task["execution"]["steps"] <= self.recipe.d_guard_min_steps else "installed; see measured activation counters"})


def run_behavior(request: dict, task: dict, output_dir: Path | str, device="cpu") -> dict:
    """Execute one original behavioral objective through the shared public binder."""
    from benchmarks.locked_shared import two_pole, trajectory
    from benchmarks.locked_shared.hosts import (ae_gan_hold, cover_leftover, mid_scale_identity,
                                               residual_student, unipolar, unused_token_hold)
    modules = {module.__name__.rsplit(".", 1)[-1]: module for module in (
        two_pole, trajectory, ae_gan_hold, cover_leftover, mid_scale_identity,
        residual_student, unipolar, unused_token_hold)}
    # These are deliberate zero-origin controls, not unspecified random layers.
    # Declare their constants through the public initializer registry once.
    for cls in (unipolar.FreeOriginResidual, unused_token_hold.SharedSlotStudent,
                mid_scale_identity.MidScaleResidual, cover_leftover._Residual):
        init.register(cls, lambda model: {name: init.KEEP for name, _ in model.named_parameters(recurse=False)})
    name = task["execution"].get("host", task["id"])
    if name not in modules:
        raise CapabilityError([f"unsupported behavioral host {name}"])
    if task["execution"].get("prior_applicability") == "not_sampled" and task["execution"]["prior"]["kind"] != "particle_cloud":
        raise CapabilityError(["a nonsampled parameter host cannot claim MoG sampling support"])
    module = modules[name]
    components = BehaviorComponents(request, task)
    seed = components.context.streams.seed
    steps = task["execution"]["steps"]
    if type(steps) is not int or steps < 1:
        raise ValueError("behavioral task needs a positive step budget")
    started = time.monotonic()
    previous_threads = torch.get_num_threads()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        torch.set_num_threads(1)
        with torch.device("cpu"), components.context.streams.fork("data", component="host", purpose="batches"), ExitStack() as stack:
            stack.enter_context(patch.object(module, "checkpoint", components.checkpoint))
            stack.enter_context(patch.object(module, "schedule_optimizer", components.schedule_optimizer))
            if hasattr(module, "PROTOCOL"):
                overrides = {k: v for k, v in dict(steps=steps, seed=seed).items() if k in module.PROTOCOL}
                stack.enter_context(patch.dict(module.PROTOCOL, overrides))
            # These existing host scorers can run outside checkpoint callbacks.
            # Isolate every call, including AE's extra progress measurements.
            if name == "ae_gan_hold":
                original = module.evaluate
                def evaluate(*args, **kwargs):
                    if components.noise._evaluating:
                        return original(*args, **kwargs)
                    with components.noise.evaluation(steps):
                        return original(*args, **kwargs)
                stack.enter_context(patch.object(module, "evaluate", evaluate))
            kwargs = dict(noise_policy=components.noise, components=components)
            if name == "two_pole":
                stack.enter_context(patch.object(module, "TOY_STEPS", steps))
                stack.enter_context(patch.object(module, "TOY_SEED", seed))
                raw = module.train(**kwargs)
            elif name == "trajectory":
                raw = module.train(diagnostics=True, **kwargs)
            elif name == "residual_student":
                raw = module.train(**kwargs)
            elif name == "unipolar":
                raw = module.run_arm("locked_rpgan", steps=steps, seed=seed, **kwargs)
            elif name == "unused_token_hold":
                raw = module.train(module.UnusedHoldRecipe(name="forge", steps=steps, seed=seed), **kwargs)
            elif name == "ae_gan_hold":
                raw = module.train(module.HoldConfig(name="forge", steps=steps, seed=seed), **kwargs)
                raw.pop("cfg", None)
            elif name == "cover_leftover":
                raw = module.fit_cover_leftover(module.CoverRecipe(steps=steps, seed=seed), **kwargs)
            else:
                raw = module.run_arm("locked", steps=steps, seed=seed, **kwargs)
        live = raw.get("live", raw)
        metrics = {key: live[key] for key, _, _ in task["evaluation"]["thresholds"] if key in live}
        policy = executed_receipt(BEHAVIOR_POLICIES[name], eval_output_noise=POLICIES[BEHAVIOR_POLICIES[name]])
        result = dict(task_id=task["id"], evidence=dict(observations=components.observations, live=metrics,
                      scoring_weights="live", guards=components.guards(), **policy),
                      execution_path="public_components", device="cpu", applied=components.receipt(),
                      raw=raw, cost=dict(wall_seconds=time.monotonic()-started))
        atomic_json(output_dir / "result.json", result)
        return result
    finally:
        torch.set_num_threads(previous_threads)
