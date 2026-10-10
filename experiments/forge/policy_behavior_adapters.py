"""Explicit policy variants of behavioral goals through the public lifecycle.

The direct two-pole host has one independent table and a scalar critic, so
its original objective can bind UpdatePolicy without a second generator.
Conditional, multi-bank and MoG/encoder hosts require distinct routed
contracts; they are never silently certified as independent Atlas rows.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import time

import torch
from torch import nn

from particlegan import UpdatePolicy
from particlegan.training import InputNoise

from .api import CapabilityError, task_formulation_context, task_policy_blockers
from .artifacts import manifest_artifacts
from .contracts import atomic_json, file_hash
from .mechanisms import MechanismAudit, mechanism_blockers
from .policy_adapters import (DIGEST_KIND, PolicyLifecycleAudit, controls_receipt,
                             evaluation_state, finite_policy_state,
                             observation_receipt, typed_state_digest)


def behavior_preflight(task, candidate):
    """Separate actual host ownership boundaries from numerical failures."""
    blockers = task_policy_blockers(task, candidate)
    host = task.get("execution", {}).get("host", task.get("id"))
    boundaries = {
        "unused_token_hold": "shared/slot parameters have no independent sampled row table; a named public RoutedRows binding must preserve protected unused contexts",
        "ae_gan_hold": "the original MoG posterior/encoder law conflicts with independent cloud row controls; a named routed encoder/cloud task must attest its revised conditional law",
        "trajectory": "the original source-conditioned generation requires a named conditional routing and row-evidence/birth-death contract",
        "residual_student": "the original paired residual mapping requires a named conditional routing and protected-context contract",
        "cover_leftover": "the original objective owns two prior banks and conditioned polar contexts; a single independent table is not their joint law",
        "unipolar": "the original residual generator consumes conditioned teacher coordinates; a faithful public callback/routing binding is not yet declared",
        "mid_scale_identity": "the original mid-scale paired mapping consumes context coordinates; a faithful public callback/routing binding is not yet declared",
    }
    if host != "two_pole":
        blockers.append(f"{task.get('id', '<task>')}: " + boundaries.get(host, "no declared policy behavioral host"))
    else:
        from benchmarks.legacy.locked_shared import LOCKED_SHARED
        expected = {"num_particles": LOCKED_SHARED.n_particles, "z_dim": 1,
                    "batch_size": LOCKED_SHARED.n_particles}
        if task["execution"].get("resources") != expected:
            blockers.append(f"{task['id']}: two-pole policy resources must preserve the original {expected}")
        observation = task.get("evaluation", {}).get("policy_observation", {})
        if (observation.get("sampler") != "served_snapshot"
                or observation.get("parameter_measurement") != "selected_table_and_critic_gradient"
                or observation.get("latent_policy") != "not_applied_to_parameter_measurement"):
            blockers.append(f"{task['id']}: two-pole measurement must use selected table/critic without a sampling draw")
    return blockers


class TwoPolePolicyFixture:
    """Original direct-table game, with one complete ordered public owner."""
    def __init__(self, request, task, *, device="cpu"):
        from benchmarks.locked_shared import two_pole
        from benchmarks.legacy.locked_shared import LOCKED_SHARED
        blockers = behavior_preflight(task, request["candidate"])
        if blockers:
            raise CapabilityError(blockers)
        self.task = task
        self.context = task_formulation_context(request["candidate"], task,
            request.get("protocol"), device=device)
        self.recipe = self.context.recipe
        self.max_steps = task["execution"]["steps"]
        if self.recipe.total_steps is not None:
            raise CapabilityError(["policy-selected two-pole preserves schedule-free Recipe.total_steps=None"])
        self.G = nn.Identity().to(device)
        self.D = self.context.construct(two_pole.HostCritic, component="discriminator").to(device)
        self.prior = self.recipe.make_prior(device=self.context.device,
            dtype=next(self.D.parameters()).dtype, learnable=True, init_std=0.,
            generator=self.context.streams.generator("init", component="prior", purpose="locations"))
        # The host starts every coordinate at zero and pins its critic values.
        # These are original fixture constants, not an optimized witness.
        self.context.initialization.update(
            generator={"initializer": "parameterless_identity_for_original_direct_table"},
            discriminator={"initializer": "stored_host_weights", "source": "benchmarks/locked_shared/two_pole.py",
                           "state_sha256": typed_state_digest(self.D.state_dict())},
            prior={"initializer": "zeros", "shape": list(self.prior.z.shape),
                   "state_sha256": typed_state_digest(self.prior.state_dict())})
        self.opt_g = self.recipe.make_generator_optimizer([
            {"params": [self.prior.z], "lr": self.recipe.lr, "betas": self.recipe.betas}],
            latent_table=self.prior.z, direct_particles=[self.prior.z])
        self.opt_d = self.recipe.make_critic_optimizer(self.D, ema_critic=deepcopy(self.D))
        self.loss = self.recipe.make_loss()
        self.penalty = self.recipe.make_critic_penalty(self.opt_d, collect_stats=True)
        self.policy = UpdatePolicy(self.recipe, self.G, self.D, prior=self.prior,
            generator_optimizer=self.opt_g, critic_optimizer=self.opt_d, penalty=self.penalty,
            seed=self.context.streams.seed, streams={
                "latent_generator": self.context.streams.generator("prior", component="latent", purpose="indices"),
                "penalty_generator": self.context.streams.generator("noise", component="penalty", purpose="training"),
                "noise_generator": self.context.streams.generator("noise", component="generator", purpose="output"),
                "eval_generator": self.context.streams.generator("eval", component="sampler", purpose="samples")})
        self.context.bind_update_policy(self.policy, external_max_steps=self.max_steps)
        self.policy._forge_task_cohort = task["task_cohort"]
        self.audit = PolicyLifecycleAudit(self.policy)
        self.mechanisms = MechanismAudit(self.recipe, self.opt_d, [self.opt_g])
        self.critic = InputNoise(self.D, generator=self.context.streams.generator(
            "noise", component="critic", purpose="input"))
        self.real = two_pole.real_batch(LOCKED_SHARED.n_particles).to(device)
        self.rows = torch.arange(LOCKED_SHARED.n_particles, device=device)
        self.particle_l2 = LOCKED_SHARED.particle_l2
        self.last_update = {}
        self.rng_audits = []
        self.purity = []
        self.policy_observations = []
        self.last_views = None

    @property
    def completed_steps(self):
        return self.policy.completed_steps

    def step(self):
        """Keep original enumeration, critic, RpGAN pairing and particle L2."""
        try:
            noise = self.policy.begin_step(self.real, execution_limit=self.max_steps)
            self.critic.std = noise.input_sigma
            self.opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake = self.policy.generate(self.prior.z.detach(), rows=self.rows,
                    sigma=noise.output_sigma, stream=self.policy.noise_generator)
            self.policy.observe_critic_pair(self.real, fake)
            adversarial_d = self.loss.d_loss(self.critic(self.real), self.critic(fake))
            penalty = self.penalty(self.critic, self.real, fake)
            loss_d = adversarial_d + penalty
            self.policy.before_critic_backward()
            loss_d.backward()
            self.policy.after_critic_backward()
            self.opt_d.step()
            self.policy.after_critic_step()

            self.opt_g.zero_grad(set_to_none=True)
            d_real = self.critic(self.real).detach()
            generated = self.policy.generate(self.prior.z, rows=self.rows,
                sigma=noise.output_sigma, stream=self.policy.noise_generator)
            loss_gan = self.loss.g_loss(self.critic(generated), d_real)
            l2 = self.particle_l2 * self.prior.z.square().mean()
            loss_g = loss_gan + l2
            self.policy.before_generator_backward()
            loss_g.backward()
            self.policy.after_generator_backward(loss_gan=loss_gan, loss_critic=adversarial_d)
            self.opt_g.step()
            self.policy.after_generator_step()
            self.policy.finish_step()
            self.mechanisms.observe_penalty(self.penalty.last_stats)
            values = {"loss_d": loss_d, "loss_g": loss_g, "loss_gan": loss_gan,
                      "penalty": penalty, "particle_l2": l2}
            if not all(bool(torch.isfinite(value)) for value in values.values()):
                raise FloatingPointError("nonfinite public two-pole update")
            self.last_update = {key: float(value.detach()) for key, value in values.items()}
            return {**self.last_update, "step": self.completed_steps}
        except Exception:
            self.policy.abort_step()
            raise

    def state_dict(self):
        return {"schema_version": 1, "kind": "forge_policy_two_pole_v1",
                "recipe": self.recipe.to_dict(), "prior": deepcopy(self.context.prior_config),
                "policy_contract": deepcopy(self.task["execution"]["policy_contract"]),
                "external_max_steps": self.max_steps, "caller_cursor": self.completed_steps,
                "initialization": deepcopy(self.context.initialization),
                "streams": self.context.streams.state_dict(), "policy": self.policy.state_dict(),
                "lifecycle_audit": self.audit.receipt(self.completed_steps)}

    def load_state_dict(self, state):
        expected = self.state_dict()
        if not isinstance(state, dict) or state.keys() != expected.keys():
            raise ValueError("invalid direct two-pole checkpoint schema")
        for key in ("schema_version", "kind", "recipe", "prior", "policy_contract",
                    "external_max_steps", "initialization"):
            if state[key] != expected[key]:
                raise ValueError(f"direct two-pole checkpoint {key} differs")
        steps = state["caller_cursor"]
        audit = state["lifecycle_audit"]
        if (type(steps) is not int or not 0 <= steps <= self.max_steps
                or state["policy"].get("completed_steps") != steps or not finite_policy_state(state)
                or audit.get("end_completed_steps") != steps or audit.get("complete") is not True
                or type(audit.get("start_completed_steps")) is not int or audit["start_completed_steps"] != 0
                or audit.get("calls") != {name: steps - audit.get("start_completed_steps", 0)
                                         for name in self.audit.calls}
                or audit.get("pending") or audit.get("order_errors") != 0
                or audit.get("last_order") != (list(self.audit.calls) if steps else [])):
            raise ValueError("invalid direct two-pole checkpoint health/cursor/lifecycle")
        self.context.streams.validate_state_dict(state["streams"])
        for name, value in state["policy"]["streams"].items():
            stream = getattr(self.policy, name)
            key = next((key for key, owned in self.context.streams._streams.items() if stream is owned), None)
            if key is None or not torch.equal(value, state["streams"]["states"].get(key, torch.empty(0))):
                raise ValueError("policy and named checkpoint streams disagree")
        # Public validation occurs before its loader mutates learned state.
        self.policy.load_state_dict(state["policy"])
        self.context.streams.load_state_dict(state["streams"])
        self.audit.start_steps = audit["start_completed_steps"]
        self.audit.calls = dict(audit["calls"])
        self.audit.pending = list(audit["pending"])
        self.audit.order_errors = audit["order_errors"]
        self.audit.last_order = list(audit["last_order"])

    def observe(self):
        """Selected table/critic measurement, without a latent/noise draw."""
        from benchmarks.locked_shared.two_pole import _grad_median
        before = typed_state_digest(evaluation_state(self.state_dict()))
        streams = self.context.streams.audit()
        snapshot = self.policy.served_model()
        captured = {}
        # Read the input gradient already computed by the unchanged host
        # scorer.  The hook returns None and adds no forward/backward/draw.
        def capture_input(module, args):
            values = args[0]
            if values.requires_grad:
                captured["critic_inputs"] = values.detach().clone()
                def capture_gradient(gradient):
                    captured["critic_gradient"] = gradient.detach().clone()
                values.register_hook(capture_gradient)
        handle = snapshot.critic.register_forward_pre_hook(capture_input)
        try:
            metrics = {"mean_abs": float(snapshot.table.abs().mean()),
                       "grad_med": _grad_median(snapshot.critic, self.real, snapshot.table)}
        finally:
            handle.remove()
        if set(captured) != {"critic_inputs", "critic_gradient"}:
            raise RuntimeError("original two-pole scorer did not expose its measured input gradient")
        self.last_views = {"real": self.real.detach().cpu().clone(),
                           "particles": snapshot.table.detach().cpu().clone(),
                           **{name: values.detach().cpu().clone() for name, values in captured.items()}}
        after = typed_state_digest(evaluation_state(self.state_dict()))
        audit = self.context.streams.compare(streams, self.context.streams.audit())
        pure = before == after and audit["unintended_rng_deviations"] == 0
        self.rng_audits.append(audit)
        self.purity.append({"completed_steps": self.completed_steps, "before_sha256": before,
                           "after_sha256": after, "digest_kind": DIGEST_KIND, "pure": pure,
                           "allowed_changes": "none; selected parameter measurement draws no samples"})
        self.policy_observations.append(observation_receipt(self.task, self.policy))
        if not pure:
            raise RuntimeError("selected two-pole measurement changed public training state/RNG")
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("nonfinite selected two-pole measurement")
        return metrics

    def guards(self):
        state = self.state_dict()
        controls = controls_receipt(self.policy, self.completed_steps)
        mechanisms = self.mechanisms.receipt()
        def updates(optimizer, parameters):
            counts = [int(optimizer.state[p]["step"]) for p in parameters
                      if p in optimizer.state and "step" in optimizer.state[p]]
            return min(counts) if counts else 0
        return {"all_finite": finite_policy_state(state),
                "optimizer_updates": {"prior": updates(self.opt_g, [self.prior.z]),
                                      "discriminator": updates(self.opt_d, self.D.parameters())},
                "hooks_exercised": controls["implementation_observed"] and not mechanism_blockers(mechanisms),
                "mechanism_audit": mechanisms,
                "unintended_rng_deviations": sum(row["unintended_rng_deviations"] for row in self.rng_audits)}


def run_behavior(request, task, output_dir, device="cpu"):
    """Run one declared host; its original 24 observations and bounds remain."""
    from .adapters import _checkpoints, _event
    from .sampling import executed_receipt
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    artifacts = output / "policy-behavior"; artifacts.mkdir()
    started = time.monotonic()
    fixture = TwoPolePolicyFixture(request, task, device=device)
    observations = []
    checkpoints = set(_checkpoints(task))
    views = artifacts / "observations"; views.mkdir()
    for step in range(1, task["execution"]["steps"] + 1):
        fixture.step()
        if step in checkpoints:
            metrics = fixture.observe()
            observations.append({"step": step, **metrics})
            import numpy as np
            np.savez_compressed(views / f"step_{step:06d}.npz",
                **{name: value.numpy() for name, value in fixture.last_views.items()})
            _event("observation", task=task["id"], step=step, metrics=metrics)
    state = fixture.state_dict(); state_path = artifacts / "state.pt"
    torch.save(state, state_path)
    controls = controls_receipt(fixture.policy, fixture.completed_steps)
    sampling = executed_receipt(task["evaluation"]["sampling_law"],
        eval_output_noise=task["evaluation"]["eval_output_noise"])
    evidence = {"observations": observations, "live": observations[-1],
                "scoring_weights": "state_selected", "guards": fixture.guards(),
                "policy_controls": controls, "policy_observation": fixture.policy_observations[-1],
                "policy_observations": fixture.policy_observations, "policy_purity": fixture.purity,
                "rng_audits": fixture.rng_audits, **sampling,
                "checkpoint": {"path": "state.pt", "sha256": file_hash(state_path),
                               "state_sha256": typed_state_digest(state), "digest_kind": DIGEST_KIND},
                "artifact_root": str(artifacts.resolve()), "artifact_manifest": manifest_artifacts(artifacts),
                "host": {"source": "benchmarks/locked_shared/two_pole.py", "goal": "travel off origin with bounded critic slope",
                         "objective": "original RpGAN direct-table pairing plus particle_l2 * table.square().mean()",
                         "particle_l2": fixture.particle_l2, "critic_initialization": "stored_host_weights",
                         "particle_initialization": "zeros", "training_row_strategy": "enumerate all original rows",
                         "measurement": "selected table plus current critic; no DV12/noise evaluation draw"}}
    result = {"task_id": task["id"], "evidence": evidence,
              "execution_path": "public_components", "device": str(device),
              "applied": fixture.context.receipt(),
              "cost": {"wall_seconds": time.monotonic() - started,
                       "completed_steps": fixture.completed_steps,
                       "optimizer_updates": evidence["guards"]["optimizer_updates"]}}
    atomic_json(output / "result.json", result)
    return result
