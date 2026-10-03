"""Original paired/basis behavioral functions through public routed policy.

Only the new conditional cohort uses these callers.  The complete original
training grid and every host auxiliary term remain present.  Fit/guard pools
partition known training contexts for structural acceptance; their supplied
guard tensors never enter a loss.  This is not a held-out generalization study.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from particlegan import ParticlePrior, ParticleRegularizer, RoutedBatch, RoutedRows, UpdatePolicy, init
from particlegan.training import InputNoise
from benchmarks.locked_shared import trajectory
from benchmarks.locked_shared.hosts import residual_student, unipolar, mid_scale_identity

from .contracts import atomic_json, file_hash, stable_hash
from .mechanisms import MechanismAudit, mechanism_blockers
from .policy_adapters import (DIGEST_KIND, PolicyLifecycleAudit, controls_receipt,
                             evaluation_state, finite_policy_state, observation_receipt,
                             typed_state_digest)
from .rng import NamedStreams
from . import conditional_policy_contracts as contract


ROOT = Path(__file__).resolve().parents[2]


class _BasisBank:
    """Storage-only reform: one matrix owns the original scalar basis vectors."""
    BASIS_ROLES = ()

    def _pack_basis(self):
        values = torch.stack([getattr(self, name).detach().clone() for name in self.BASIS_ROLES])
        for name in self.BASIS_ROLES:
            delattr(self, name)
        self.bank = nn.Parameter(values)

    def __getattr__(self, name):
        parameters = self.__dict__.get("_parameters", {})
        if name in self.BASIS_ROLES and "bank" in parameters:
            return parameters["bank"][self.BASIS_ROLES.index(name)]
        return super().__getattr__(name)


class RoutedUnipolarResidual(_BasisBank, unipolar.FreeOriginResidual):
    BASIS_ROLES = ("odd", "even", "origin")

    def __init__(self):
        super().__init__(); self._pack_basis()


class RoutedMidScaleResidual(_BasisBank, mid_scale_identity.MidScaleResidual):
    BASIS_ROLES = ("odd", "even", "origin", "mid")

    def __init__(self):
        super().__init__(); self._pack_basis()


# Only these new classes gain a declaration.  Original constructor constants
# and strict initializer rejection are unchanged by importing this module.
init.register(RoutedUnipolarResidual, {"bank": init.KEEP})
init.register(RoutedMidScaleResidual, {"bank": init.KEEP})


class RoleRouter(nn.Module):
    def __init__(self, rows, *, device="cpu", dtype=torch.float32):
        super().__init__()
        self.register_buffer("row_roles", torch.arange(rows, device=device).reshape(rows, 1))
        self.register_buffer("log_mass", torch.zeros(rows, device=device, dtype=dtype))


def arc_contexts(slow):
    return torch.cat((torch.arange(len(slow), device=slow.device, dtype=slow.dtype)[:, None], slow), dim=1)


def _role_logits(candidate, roles):
    return torch.where(roles[:, None] == candidate.row_state["row_roles"].flatten()[None, :],
                       candidate.table.new_zeros(()), candidate.table.new_tensor(contract.UNMATCHED_LOGIT))


def complete_arc_forward(models, context, candidate, routing):
    if (context.ndim != 2 or context.shape[1] != 17 or not torch.isfinite(context).all()
            or not bool((context[:, 0] == context[:, 0].round()).all())
            or not bool(((context[:, 0] >= 0) & (context[:, 0] < 12)).all())):
        raise ValueError("trajectory contexts require original identity IDs and 16 finite slow coordinates")
    codes = routing.mix("identity_latent", _role_logits(candidate, context[:, 0].long()))
    return models["generator"](context[:, 1:], codes)


def complete_basis_forward(models, context, candidate, routing):
    if context.ndim != 2 or context.shape[1] != 1 or not torch.isfinite(context).all():
        raise ValueError("basis contexts require one finite original scale per row")
    student = models["generator"]
    values = []
    for index, name in enumerate(student.BASIS_ROLES):
        roles = torch.full((len(context),), index, device=context.device, dtype=torch.long)
        values.append(routing.mix(name, _role_logits(candidate, roles)))
    scale = context
    result = scale * values[0] + scale.abs() * values[1] + values[2]
    if len(values) == 4:
        bump = torch.where((scale > 0) & (scale < 1), 4 * scale * (1 - scale), torch.zeros_like(scale))
        result = result + bump * values[3]
    return result


def paired_arc_features(models, context, samples, targets):
    critic = models["critic"]
    return critic.net[:-1](torch.cat((context[:, 1:], samples), dim=-1))


def paired_basis_features(models, context, samples, targets):
    critic = models["critic"]
    return critic.net[:-1](torch.cat((samples.float() / critic.input_scale, context), dim=-1))


def routing_spec(host):
    arcs = contract.HOSTS[host]["table_owner"] == "prior.z"
    return RoutedRows(model_forward=complete_arc_forward if arcs else complete_basis_forward,
                      features=paired_arc_features if arcs else paired_basis_features,
                      sites=tuple(contract.HOSTS[host]["sites"]), row_buffers=("row_roles",),
                      output_error_guard=True, max_context_harm=0.,
                      max_output_error_increase=0., max_output_context_harm=0.)


class _ConditionedCritic(nn.Module):
    """Original cap coordinates and labels, with public raw-data input noise."""
    def __init__(self, critic, noise, *, slow=None, scale=None, normalized=False):
        super().__init__()
        self.critic, self.noise = critic, noise
        self.slow, self.scale, self.normalized = slow, scale, normalized

    def forward(self, values):
        if self.slow is not None:
            return self.critic(self.slow, self.noise(values)).unsqueeze(-1)
        if self.normalized:
            values = self.noise(values * self.critic.input_scale) / self.critic.input_scale
        else:
            values = self.noise(values) / self.critic.input_scale
        return self.critic.score(values, self.scale).unsqueeze(-1)


def conditional_controls_receipt(policy, completed_steps):
    if policy.row_policy != "routed_paired" or policy.row_semantics != "conditional":
        raise ValueError("conditional receipts require the actual public routed owner")
    result = controls_receipt(policy, completed_steps)
    result.update(cohort=contract.COHORT, family=contract.FAMILY,
                  row_policy="routed_paired", independent_atlas_qualification=False,
                  table_optimizer_semantics=deepcopy(contract._contract("trajectory", {})["table_optimizer_semantics"]))
    return result


def _global_rng():
    state = np.random.get_state()
    return {"torch": torch.get_rng_state().clone(), "python": repr(random.getstate()),
            "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]]}


class _SelectedBasis:
    def __init__(self, served):
        self.served = served

    def state(self, scale):
        context = self.served.table.new_tensor([[float(scale)]])
        return self.served.routed_forward(context, perturb=False, output_noise=False)[0]

    delta = state


class ConditionalPolicyFixture:
    """Full source-owned objective and complete public policy state for one host."""
    def __init__(self, request, task, *, device="cpu", context=None):
        contract.validate_conditional_task(task, root=ROOT)
        self.host = task["execution"]["host"]
        self.task = deepcopy(task); self.max_steps = task["execution"]["steps"]
        self.recipe = contract.resolved_recipe(request["candidate"], task)
        self.context, self.device = context, torch.device(device)
        seed = request.get("protocol", {}).get("seed", 0)
        if type(seed) is not int or seed != 0:
            raise ValueError("conditional parent fixtures retain their original seed 0")
        if context is not None:
            if (stable_hash(context.recipe.to_dict()) != stable_hash(self.recipe.to_dict())
                    or context.device != self.device or context.streams.seed != seed
                    or stable_hash(context.policy_contract) != stable_hash(task["execution"]["policy_contract"])):
                raise ValueError("conditional context differs from declared Recipe/device/streams/contract")
            self.streams = context.streams
        else:
            self.streams = NamedStreams(seed, device=self.device)
        self.arcs = self.host in ("trajectory", "residual_student")
        self.prior = None
        if self.arcs:
            module = trajectory if self.host == "trajectory" else residual_student
            protocol = module.PROTOCOL
            if (protocol["n_particles"], protocol["z_dim"], protocol["frames"], protocol["critic_hidden"],
                    protocol["cover_weight"], protocol["particle_l2"], protocol["vicreg_weight"]) != (12, 4, 8, 64, 1.5, .02, .05):
                raise ValueError("original trajectory resources/objective changed")
            self.slow, self.real = (value.to(self.device) for value in trajectory.trajectories())
            self.train_context = arc_contexts(self.slow)
            constructor = trajectory._Generator if self.host == "trajectory" else residual_student.ResidualHead
            with self.streams.fork("init", component="generator", purpose="construction", device="cpu"):
                args = (16, 4, 16, 64) if self.host == "trajectory" else (16, 4, 64)
                self.G = constructor(*args).to(self.device)
            with self.streams.fork("init", component="discriminator", purpose="construction", device="cpu"):
                self.D = module._Critic(32, 64).to(self.device)
            self.prior = ParticlePrior(12, 4, init_std=.1, device=self.device,
                generator=self.streams.generator("init", component="prior", purpose="construction"))
            self.table = self.prior.z
            self.mask = residual_student.both_land_mask(self.slow, self.real,
                torch.arange(12, device=self.device)) if self.host == "residual_student" else None
            self.fit_context, self.fit_targets = self.train_context[::2].clone(), self.real[::2].clone()
            self.guard_context, self.guard_targets = self.train_context[1::2].clone(), self.real[1::2].clone()
            self.spread = ParticleRegularizer(weight=.05)
        else:
            module = unipolar if self.host == "unipolar" else mid_scale_identity
            if (module.DIM, module.N_ROWS, module.CRITIC_HIDDEN) != (4, 8, 64):
                raise ValueError("original scale-host dimensions changed")
            self.scales = unipolar.SCALES if self.host == "unipolar" else mid_scale_identity.EVAL_SCALES
            self.teacher = None if self.host == "unipolar" else mid_scale_identity.smile_teacher()
            if self.teacher is not None:
                # Teacher source owns these constants; moving to a device is
                # a prospective execution transform, never a new target.
                from dataclasses import replace
                self.teacher = replace(self.teacher, **{key: getattr(self.teacher, key).to(self.device)
                    for key in ("identity", "concept", "stranger", "retain_unit", "plus", "minus")})
            self.targets = ({0.: torch.zeros(4, device=self.device), 1.: unipolar.PLUS.to(self.device)}
                if self.host == "unipolar" else {scale: self.teacher.train_target("locked", scale) for scale in self.scales})
            constructor = RoutedUnipolarResidual if self.host == "unipolar" else RoutedMidScaleResidual
            with self.streams.fork("init", component="generator", purpose="construction", device="cpu"):
                self.G = constructor().to(self.device)
            cloud = torch.stack([self.targets[scale] for scale in self.scales])
            with self.streams.fork("init", component="discriminator", purpose="construction", device="cpu"):
                self.D = module.ScaleCritic(4, cloud, hidden=64).to(self.device)
            self.table = self.G.bank
            self.train_context = self.table.new_tensor([[scale] for scale in self.scales]).repeat_interleave(8, 0)
            self.real = cloud.repeat_interleave(8, 0)
            fit = torch.tensor([scale == 1. if self.host == "unipolar" else abs(scale) == 1.
                                for scale in self.scales], device=self.device).repeat_interleave(8)
            self.fit_context, self.fit_targets = self.train_context[fit].clone(), self.real[fit].clone()
            self.guard_context, self.guard_targets = self.train_context[~fit].clone(), self.real[~fit].clone()
        self.initialization = {}
        owners = [("generator", self.G), ("discriminator", self.D)] + ([] if self.prior is None else [("prior0", self.prior)])
        for name, model in owners:
            if context is None:
                seeds = {key: self.streams.seed_for("init", component=name, purpose=key)
                         for key, value in model.named_parameters() if value.requires_grad and value.numel()}
                init.deterministic_orthogonal_(model, parameter_seeds=seeds)
                self.initialization[name] = {"initializer": "deterministic_orthogonal_named_parameters_v1",
                    "state_sha256": typed_state_digest(model.state_dict()), "parameter_seeds": seeds}
            else:
                context.initialize(model, component=name)
        if context is not None:
            self.initialization = deepcopy(context.initialization)
        self.router = RoleRouter(len(self.table), device=self.device, dtype=self.table.dtype)
        network = [value for value in self.G.parameters() if value is not self.table]
        groups = ([{"params": network, "lr": self.recipe.lr}] if network else []) + [
            {"params": [self.table], "lr": self.recipe.lr * self.recipe.prior_lr_mult,
             "betas": self.recipe.prior_betas or self.recipe.betas}]
        self.opt_g = self.recipe.make_generator_optimizer(groups, latent_table=self.table)
        self.opt_d = self.recipe.make_critic_optimizer(self.D, ema_critic=deepcopy(self.D))
        self.loss = self.recipe.make_loss(); self.penalty = self.recipe.make_critic_penalty(self.opt_d, collect_stats=True)
        self.policy = UpdatePolicy(self.recipe, self.G, self.D, prior=self.prior, table=self.table,
            router=self.router, generator_optimizer=self.opt_g, critic_optimizer=self.opt_d,
            routed_rows=routing_spec(self.host), row_semantics="conditional", penalty=self.penalty,
            seed=seed, streams={
                "latent_generator": self.streams.generator("prior", component="latent", purpose="indices"),
                "penalty_generator": self.streams.generator("noise", component="penalty", purpose="training"),
                "noise_generator": self.streams.generator("noise", component="generator", purpose="output"),
                "eval_generator": self.streams.generator("eval", component="sampler", purpose="samples")})
        if context is not None:
            context.bind_update_policy(self.policy, external_max_steps=self.max_steps)
        self.audit = PolicyLifecycleAudit(self.policy)
        self.mechanisms = MechanismAudit(self.recipe, self.opt_d, [self.opt_g])
        self.input_noise = InputNoise(nn.Identity(), generator=self.streams.generator("noise", component="critic", purpose="input"))
        self.last_update = {}; self.last_views = None
        self.purity, self.policy_observations, self.rng_audits = [], [], []

    @property
    def completed_steps(self):
        return self.policy.completed_steps

    def clean_original_auxiliary(self, output):
        """Complete original auxiliary terms, including every training row."""
        if self.arcs:
            cover = 1.5 * trajectory._cover(output, self.real)
            l2 = .02 * self.table.square().mean()
            spread = self.spread(self.table)
            residual = (residual_student.RESIDUAL_WEIGHT * (output[self.mask] - self.real[self.mask]).square().mean()
                        if self.mask is not None and bool(self.mask.any()) else output.new_zeros(()))
            return {"cover_loss": cover, "particle_l2_loss": l2, "spread_loss": spread, "residual_loss": residual}
        if self.host == "mid_scale_identity":
            cover = output.new_zeros(())
            for index, scale in enumerate(self.scales):
                cover = cover + F.mse_loss(output[index], self.targets[scale])
            return {"cover_loss": mid_scale_identity.FORMULATION["cover_weight"] * cover / float(len(self.scales))}
        return {}

    def step(self):
        flags = [parameter.requires_grad for parameter in self.D.parameters()]
        try:
            routed = RoutedBatch(self.fit_context, self.fit_targets, self.guard_context, self.guard_targets)
            noise = self.policy.begin_step(self.real, routed=routed, execution_limit=self.max_steps)
            self.input_noise.std = noise.input_sigma
            self.opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake = self.policy.routed_generate(self.train_context, sigma=noise.output_sigma, perturb=True)
            self.policy.observe_critic_pair(self.real, fake)
            adversarial_d = self.table.new_zeros(()); cap = self.table.new_zeros(())
            if self.arcs:
                critic = _ConditionedCritic(self.D, self.input_noise, slow=self.slow)
                adversarial_d = self.loss.d_loss(critic(self.real), critic(fake))
                cap = self.penalty(critic, self.real, fake)
                self.mechanisms.observe_penalty(self.penalty.last_stats)
            else:
                for index, scale in enumerate(self.scales):
                    rows = slice(8 * index, 8 * (index + 1))
                    critic = _ConditionedCritic(self.D, self.input_noise, scale=scale)
                    normalized = _ConditionedCritic(self.D, self.input_noise, scale=scale, normalized=True)
                    adversarial_d = adversarial_d + self.loss.d_loss(critic(self.real[rows]), critic(fake[rows])) / len(self.scales)
                    cap = cap + self.penalty(normalized, self.real[rows] / self.D.input_scale,
                                             fake[rows] / self.D.input_scale) / len(self.scales)
                    self.mechanisms.observe_penalty(self.penalty.last_stats)
            loss_d = adversarial_d + cap
            self.policy.before_critic_backward(); loss_d.backward(); self.policy.after_critic_backward()
            self.opt_d.step(); self.policy.after_critic_step()

            self.D.requires_grad_(False); self.opt_g.zero_grad(set_to_none=True)
            generated = self.policy.routed_generate(self.train_context, sigma=noise.output_sigma, perturb=True)
            if self.arcs:
                critic = _ConditionedCritic(self.D, self.input_noise, slow=self.slow)
                gan = self.loss.g_loss(critic(generated), critic(self.real).detach())
                auxiliary = self.clean_original_auxiliary(generated)
            else:
                gan = self.table.new_zeros(())
                for index, scale in enumerate(self.scales):
                    rows = slice(8 * index, 8 * (index + 1))
                    critic = _ConditionedCritic(self.D, self.input_noise, scale=scale)
                    gan = gan + self.loss.g_loss(critic(generated[rows]), critic(self.real[rows]).detach()) / len(self.scales)
                clean_context = self.table.new_tensor([[scale] for scale in self.scales])
                clean = self.policy.routed_generate(clean_context, sigma=0., perturb=False)
                auxiliary = self.clean_original_auxiliary(clean)
            loss_g = gan + sum(auxiliary.values())
            self.policy.before_generator_backward(); loss_g.backward()
            self.policy.after_generator_backward(loss_gan=gan, loss_critic=adversarial_d)
            self.opt_g.step(); self.policy.after_generator_step()
            for parameter, flag in zip(self.D.parameters(), flags):
                parameter.requires_grad_(flag)
            event = self.policy.finish_step()
            values = {"loss_d": loss_d, "loss_g": loss_g, "loss_gan": gan, "penalty": cap, **auxiliary}
            if not all(bool(torch.isfinite(value)) for value in values.values()):
                raise FloatingPointError("nonfinite conditional public-policy update")
            self.last_update = {key: float(value.detach()) for key, value in values.items()}
            return {"step": self.completed_steps, **self.last_update, "row_event": deepcopy(event)}
        except Exception:
            self.policy.abort_step()
            raise
        finally:
            for parameter, flag in zip(self.D.parameters(), flags):
                parameter.requires_grad_(flag)

    def state_dict(self):
        return {"schema_version": 1, "kind": "forge_conditional_policy_v1", "family": contract.FAMILY,
                "host": self.host, "task_sha256": stable_hash(self.task), "recipe": self.recipe.to_dict(),
                "external_max_steps": self.max_steps, "caller_cursor": self.completed_steps,
                "initialization": deepcopy(self.initialization), "streams": self.streams.state_dict(),
                "policy": self.policy.state_dict(), "lifecycle_audit": self.audit.receipt(self.completed_steps),
                "mechanism_audit_state": deepcopy(self.mechanisms.rows), "last_update": deepcopy(self.last_update)}

    def load_state_dict(self, state):
        expected = self.state_dict()
        if not isinstance(state, dict) or state.keys() != expected.keys():
            raise ValueError("invalid conditional checkpoint schema")
        for key in ("schema_version", "kind", "family", "host", "task_sha256", "recipe", "external_max_steps", "initialization"):
            if typed_state_digest(state[key]) != typed_state_digest(expected[key]):
                raise ValueError(f"conditional checkpoint {key} differs")
        steps, audit = state["caller_cursor"], state["lifecycle_audit"]
        if (type(steps) is not int or not 0 <= steps <= self.max_steps
                or state["policy"].get("completed_steps") != steps or not finite_policy_state(state)
                or audit.get("end_completed_steps") != steps or audit.get("complete") is not True
                or audit.get("start_completed_steps") != 0 or audit.get("calls") != {name: steps for name in self.audit.calls}
                or audit.get("pending") or audit.get("order_errors") != 0
                or audit.get("last_order") != (list(self.audit.calls) if steps else [])):
            raise ValueError("invalid conditional checkpoint health/cursor/lifecycle")
        mechanisms = state["mechanism_audit_state"]
        if not isinstance(mechanisms, dict) or mechanisms.keys() != self.mechanisms.rows.keys():
            raise ValueError("invalid conditional mechanism-audit schema")
        for name, row in mechanisms.items():
            original = self.mechanisms.rows[name]
            cap = steps * (len(self.scales) if not self.arcs and name in ("critic_penalty", "critic_anchor") else 1)
            if (not isinstance(row, dict) or row.keys() != original.keys()
                    or any(type(row[key]) is not bool or row[key] != original[key] for key in ("requested", "enabled"))
                    or any(type(row[key]) is not int or row[key] < 0 for key in ("calls", "eligible", "applied"))
                    or not row["applied"] <= row["eligible"] <= row["calls"] <= cap):
                raise ValueError("conditional mechanism counters disagree with original context count")
        fields = {"loss_d", "loss_g", "loss_gan", "penalty"}
        fields |= ({"cover_loss", "particle_l2_loss", "spread_loss", "residual_loss"} if self.arcs else
                   {"cover_loss"} if self.host == "mid_scale_identity" else set())
        if (not isinstance(state["last_update"], dict) or set(state["last_update"]) != (fields if steps else set())
                or any(type(value) is not float or not math.isfinite(value) for value in state["last_update"].values())):
            raise ValueError("invalid conditional last-update health")
        self.streams.validate_state_dict(state["streams"])
        for name, value in state["policy"]["streams"].items():
            stream = getattr(self.policy, name)
            key = next((key for key, owned in self.streams._streams.items() if stream is owned), None)
            if key is None or not torch.equal(value, state["streams"]["states"].get(key, torch.empty(0))):
                raise ValueError("conditional policy and named streams disagree")
        self.policy.load_state_dict(deepcopy(state["policy"]))
        self.streams.load_state_dict(deepcopy(state["streams"]))
        self.audit.start_steps, self.audit.calls = audit["start_completed_steps"], dict(audit["calls"])
        self.audit.pending, self.audit.last_order = list(audit["pending"]), list(audit["last_order"])
        self.audit.order_errors = audit["order_errors"]
        self.mechanisms.rows, self.last_update = deepcopy(mechanisms), deepcopy(state["last_update"])

    def observe(self):
        before = typed_state_digest(evaluation_state(self.state_dict())); rng_before = self.streams.audit()
        global_before = typed_state_digest(_global_rng())
        served = self.policy.served_model()
        with torch.no_grad(), self.streams.preserve():
            if self.arcs:
                output = served.routed_forward(self.train_context, perturb=False, output_noise=True,
                    generator=self.streams.generator("eval", component="sampler", purpose="samples"))
                metrics = {"identity_mse": trajectory.identity_mse(output, self.real)}
                if self.host == "residual_student":
                    # The original scorer builds CPU row IDs internally.
                    # Copying the observed arrays preserves the exact gate;
                    # no model/sampling or alternative target is introduced.
                    metrics.update(residual_student.landing_stats(output.cpu(), self.real.cpu()))
                target = self.real
                given = self.slow
            else:
                selected = _SelectedBasis(served)
                if self.host == "unipolar":
                    # The original fixed scorer constants are CPU tensors.
                    # Its interface reads actual selected outputs on CPU.
                    class CpuBasis:
                        def delta(_, scale):
                            return selected.delta(scale).cpu()
                    all_metrics = unipolar.score_residual(CpuBasis())
                    metrics = {key: value for key, value in all_metrics.items() if type(value) in (int, float)}
                else:
                    all_metrics = mid_scale_identity.score_hold(selected, teacher=self.teacher)
                    metrics = {key: value for key, value in all_metrics.items() if type(value) in (int, float)}
                given = self.table.new_tensor([[scale] for scale in self.scales])
                output = served.routed_forward(given, perturb=False, output_noise=False)
                target = torch.stack([self.targets[scale] for scale in self.scales])
        after = typed_state_digest(evaluation_state(self.state_dict())); rng_after = self.streams.audit()
        audit = self.streams.compare(rng_before, rng_after); global_after = typed_state_digest(_global_rng())
        pure = before == after and global_before == global_after and audit["unintended_rng_deviations"] == 0
        self.rng_audits.append(audit)
        self.purity.append({"completed_steps": self.completed_steps, "digest_kind": DIGEST_KIND,
            "before_sha256": before, "after_sha256": after, "global_rng_before_sha256": global_before,
            "global_rng_after_sha256": global_after, "pure": pure})
        observed = {**observation_receipt(self.task, self.policy), "family": contract.FAMILY,
                    "row_policy": self.policy.row_policy, "output_sigma_used": served.output_sigma if self.arcs else 0.,
                    "latent_perturbation_applied": False}
        self.policy_observations.append(observed)
        self.last_views = {"given_context": given.detach().cpu().clone(), "target": target.detach().cpu().clone(),
                           "samples": output.detach().cpu().clone(), "selected_table": served.table.detach().cpu().clone(),
                           "selected_row_roles": served.models["router"].row_roles.detach().cpu().clone(),
                           "selected_log_mass": served.models["router"].log_mass.detach().cpu().clone()}
        if not pure:
            raise RuntimeError("conditional selected observation changed training state or global/named RNG")
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("nonfinite original conditional measurement")
        return metrics

    def receipt(self):
        controls = conditional_controls_receipt(self.policy, self.completed_steps)
        return {"execution_path": "public_components", "family": contract.FAMILY, "task_cohort": contract.COHORT,
                "host": self.host, "recipe": self.recipe.to_dict(), "rng": self.streams.manifest(),
                "execution_phase": "gpu_numerical" if self.device.type == "cuda" else "cpu_structural_only",
                "actual_device": str(self.device),
                "initialization": deepcopy(self.initialization),
                "policy_lifecycle": {"owner": "particlegan.UpdatePolicy", "completed_steps": self.completed_steps,
                                     "external_max_steps": self.max_steps, "controls": controls},
                "table_ownership": {"owner": contract.HOSTS[self.host]["table_owner"], "shape": list(self.table.shape),
                                    "roles": list(contract.HOSTS[self.host]["sites"]), "independent_atom_claim": False},
                "original_objective": contract.HOSTS[self.host]["objective"],
                "all_original_training_contexts_preserved": True, "heldout_generalization_claim": False,
                "serving_law": {"selected_source": controls["served_source"],
                                "training_latent_policy": "public_DV12_mixed_codes",
                                "evaluation_latent_perturbation": False,
                                "evaluation_output_noise": "selected_public_policy_sigma" if self.arcs else "not_applied",
                                "original_class_initializer_modified": False},
                "independent_atlas_qualification": False}

    def guards(self):
        def count(optimizer, parameters):
            values = [int(optimizer.state[value]["step"]) for value in parameters
                      if value in optimizer.state and "step" in optimizer.state[value]]
            return min(values) if values else 0
        controls = conditional_controls_receipt(self.policy, self.completed_steps)
        mechanisms = self.mechanisms.receipt()
        direct = mechanisms["mechanisms"]["direct_particle_gain"]
        direct.update(applicable_to_routed_table=False, host_activation_credit=False,
                      owner_semantics="conditional_table_without_independent_direct_response",
                      synthetic_probe_scope="independent_scratch_hook_only_not_routed_transport")
        return {"all_finite": finite_policy_state(self.state_dict()),
                "optimizer_updates": {"generator": count(self.opt_g, self.G.parameters()),
                                      "table": count(self.opt_g, [self.table]),
                                      "discriminator": count(self.opt_d, self.D.parameters())},
                "hooks_exercised": controls["implementation_observed"] and not mechanism_blockers(mechanisms),
                "mechanism_audit": mechanisms,
                "unintended_rng_deviations": sum(value["unintended_rng_deviations"] for value in self.rng_audits)}


def run_behavior(request, task, output_dir, device="cuda", *, context=None):
    """Prospective full-budget entry point; CPU prefixes are fixture controls.

    Root's runner owns admission, supervision and GIF publication.  Each saved
    view contains actual given inputs, desired outputs and selected predictions
    so later media rendering needs no model, new samples or metric recomputation.
    """
    if torch.device(device).type != "cuda":
        raise ValueError("conditional numerical acquisition requires the declared GPU device; CPU fixtures are structural only")
    from .artifacts import manifest_artifacts
    from .sampling import executed_receipt
    started = time.monotonic()
    fixture = ConditionalPolicyFixture(request, task, device=device, context=context)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    artifacts = output / "conditional-policy-behavior"; artifacts.mkdir()
    views = artifacts / "observations"; views.mkdir()
    cadence = {math.ceil(i * fixture.max_steps / 24) for i in range(1, 25)}
    observations = []
    for step in range(1, fixture.max_steps + 1):
        fixture.step()
        if step in cadence:
            metrics = fixture.observe(); observations.append({"step": step, **metrics})
            np.savez_compressed(views / f"step_{step:06d}.npz",
                                **{key: value.numpy() for key, value in fixture.last_views.items()})
            print(json.dumps({"event": "observation", "task": task["id"], "step": step,
                              "budget": fixture.max_steps, "family": contract.FAMILY, "metrics": metrics},
                             allow_nan=False), flush=True)
    state = fixture.state_dict(); state_path = artifacts / "state.pt"; torch.save(state, state_path)
    evidence = {"observations": observations, "live": observations[-1], "scoring_weights": "state_selected",
                "guards": fixture.guards(),
                "policy_controls": conditional_controls_receipt(fixture.policy, fixture.completed_steps),
                "policy_observation": fixture.policy_observations[-1],
                "policy_observations": fixture.policy_observations, "policy_purity": fixture.purity,
                "rng_audits": fixture.rng_audits,
                **executed_receipt(task["evaluation"]["sampling_law"],
                                   eval_output_noise=task["evaluation"]["eval_output_noise"]),
                "checkpoint": {"path": "state.pt", "sha256": file_hash(state_path),
                               "state_sha256": typed_state_digest(state), "digest_kind": DIGEST_KIND},
                "artifact_root": str(artifacts.resolve()), "artifact_manifest": manifest_artifacts(artifacts),
                "host": {"source": contract.HOSTS[fixture.host]["source"],
                         "objective": contract.HOSTS[fixture.host]["objective"],
                         "all_original_training_contexts_preserved": True,
                         "protected_context_scope": "known original training contexts; separate guard tensors used only for structural acceptance",
                         "heldout_generalization_claim": False,
                         "family": contract.FAMILY, "independent_atlas_qualification": False,
                         "media_scope": "actual given context, target and selected output; root exporter supplies goal GIF"}}
    result = {"task_id": task["id"], "evidence": evidence, "execution_path": "public_components",
              "device": str(device), "applied": fixture.receipt(),
              "cost": {"wall_seconds": time.monotonic() - started, "completed_steps": fixture.completed_steps,
                       "optimizer_updates": evidence["guards"]["optimizer_updates"]}}
    atomic_json(output / "result.json", result)
    return result
