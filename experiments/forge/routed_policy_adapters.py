"""Original shared/slot hold goal through a complete public routed policy.

``atlas_routed`` owns conditional paired evidence and guarded mass transport.
It is not an independent-atom Atlas result. The fixed slot lookup is expressed
as a complete, mass-aware public routing site; fit and guard observations keep
their separate ownership, while the original known-anchor hold loss remains.
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

from particlegan import RoutedBatch, RoutedRows, UpdatePolicy, init
from particlegan.training import InputNoise
from benchmarks.locked_shared.hosts.unused_token_hold import SharedSlotStudent

from .contracts import atomic_json, file_hash, stable_hash
from .mechanisms import MechanismAudit, mechanism_blockers
from .policy_adapters import (DIGEST_KIND, PolicyLifecycleAudit, controls_receipt,
                             evaluation_state, finite_policy_state, observation_receipt,
                             typed_state_digest)
from .rng import NamedStreams
from . import routed_policy_contracts as contract


ROOT = Path(__file__).resolve().parents[2]


class DeclaredSharedSlotStudent(SharedSlotStudent):
    """Original scaffold with additive, class-local public KEEP declarations.

    The original class has intentional zero residuals but no initializer
    declarations. Registering only this new subclass preserves those zeros
    without changing the original class or its strict initializer refusal.
    """


init.register(DeclaredSharedSlotStudent, {"shared": init.KEEP, "slot": init.KEEP})


class SlotRouter(nn.Module):
    """Nontrainable original slot roles; row labels/mass are transported."""
    def __init__(self, *, device="cpu", dtype=torch.float32):
        super().__init__()
        self.register_buffer("log_mass", torch.zeros(2, device=device, dtype=dtype))
        self.register_buffer("slot_ids", torch.arange(2, device=device).reshape(2, 1))


def slot_contexts(slots, *, scale=1., device="cpu", dtype=torch.float32):
    indices = torch.as_tensor(slots, device=device, dtype=dtype).reshape(-1, 1)
    return torch.cat((indices, torch.full_like(indices, float(scale))), dim=1)


def complete_slot_forward(models, context, candidate, routing):
    """Full conditional function; every counterfactual reads its own rows.

    The finite -2**20 logit gives the exact original FP32 slot lookup at
    uniform mass. This is a declared categorical lookup, not a tuned
    temperature, hidden noise floor or independent-particle sampling law.
    Deletion uses actual additive log mass; splits clone their slot-ID buffer.
    """
    if (context.ndim != 2 or context.shape[1] != 2 or not torch.isfinite(context).all()
            or not bool(((context[:, 0] == 0) | (context[:, 0] == 1)).all())):
        raise ValueError("unused-token contexts require original slot IDs and finite scales")
    ids = context[:, 0].long()
    labels = candidate.row_state["slot_ids"].flatten()
    logits = torch.where(ids[:, None] == labels[None, :],
                         candidate.table.new_zeros(()), candidate.table.new_tensor(contract.UNMATCHED_LOGIT))
    codes = routing.mix("slot_lookup", logits)
    student = models["generator"]
    return student.neu[ids] + context[:, 1:2] * (student.shared + codes)


def paired_features(models, context, samples, targets):
    return models["critic"].features(samples)


def routing_spec():
    return RoutedRows(model_forward=complete_slot_forward, features=paired_features,
                      sites=("slot_lookup",), row_buffers=("slot_ids",),
                      output_error_guard=True, max_context_harm=0.,
                      max_output_error_increase=0., max_output_context_harm=0.)


def routed_controls_receipt(policy, completed_steps):
    """Actual public owners under their separate conditional-family identity."""
    if policy.row_policy != "routed_paired" or policy.row_semantics != "conditional":
        raise ValueError("routed receipts require the actual conditional public owner")
    receipt = controls_receipt(policy, completed_steps)
    receipt.update(cohort=contract.COHORT, family=contract.FAMILY,
                   row_policy="routed_paired", independent_atlas_qualification=False,
                   table_optimizer_semantics=deepcopy(contract.TABLE_OPTIMIZER_SEMANTICS))
    return receipt


def _global_rng():
    state = np.random.get_state()
    return {"torch": torch.get_rng_state().clone(), "python": repr(random.getstate()),
            "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]]}


class _SelectedStudent:
    """Read-only original scorer interface over the selected complete function."""
    def __init__(self, served):
        self.served = served
        self.neu = served.models["generator"].neu

    def embeds(self, scale):
        context = slot_contexts([0, 1], scale=scale, device=self.neu.device, dtype=self.neu.dtype)
        return self.served.routed_forward(context, perturb=False, output_noise=False)


class UnusedTokenRoutedFixture:
    """Caller-owned UpdatePolicy with the original two learned slot residuals."""
    def __init__(self, request, task, *, device="cpu", context=None):
        from benchmarks.locked_shared.hosts import unused_token_hold as host
        contract.validate_routed_task(task, root=ROOT)
        self.task = deepcopy(task)
        self.recipe = contract.resolved_recipe(request["candidate"], task)
        self.max_steps = task["execution"]["steps"]
        self.context = context
        self.device = torch.device(device)
        seed = request.get("protocol", {}).get("seed", 0)
        if (host.N_SLOTS, host.DIM, host.N_ROWS, host.STEPS) != (2, 2, 8, 200):
            raise ValueError("original unused-token host resources changed")
        if context is not None:
            if (stable_hash(context.recipe.to_dict()) != stable_hash(self.recipe.to_dict())
                    or context.device != self.device or context.streams.seed != seed
                    or stable_hash(context.policy_contract) != stable_hash(task["execution"]["policy_contract"])):
                raise ValueError("routed context differs from the declared Recipe/device/streams/contract")
            self.streams = context.streams
        else:
            self.streams = NamedStreams(seed, device=self.device)
        # Match the parent task's actual named public orthogonal initializer.
        # Constructor zeros/constants are retained until that declared policy
        # runs; no fitted or historical model/optimizer/EMA state is imported.
        with self.streams.fork("init", component="generator", purpose="construction", device="cpu"):
            self.G = DeclaredSharedSlotStudent().to(self.device)
        with self.streams.fork("init", component="discriminator", purpose="construction", device="cpu"):
            self.D = host.SlotCritic().to(self.device)
        self.initialization = {}
        for name, model in (("generator", self.G), ("discriminator", self.D)):
            if context is None:
                seeds = {key: self.streams.seed_for("init", component=name, purpose=key)
                         for key, value in model.named_parameters() if value.requires_grad and value.numel()}
                init.deterministic_orthogonal_(model, parameter_seeds=seeds)
                self.initialization[name] = {
                    "initializer": "deterministic_orthogonal_named_parameters_v1",
                    "state_sha256": typed_state_digest(model.state_dict()), "parameter_seeds": seeds}
            else:
                context.initialize(model, component=name)
        if context is not None:
            self.initialization = deepcopy(context.initialization)
        self.router = SlotRouter(device=self.device, dtype=self.G.slot.dtype)
        self.opt_g = self.recipe.make_generator_optimizer([
            {"params": [self.G.shared], "lr": self.recipe.lr},
            {"params": [self.G.slot], "lr": self.recipe.lr * self.recipe.prior_lr_mult,
             "betas": self.recipe.prior_betas or self.recipe.betas}],
            latent_table=self.G.slot)
        self.opt_d = self.recipe.make_critic_optimizer(self.D, ema_critic=deepcopy(self.D))
        self.loss = self.recipe.make_loss()
        self.penalty = self.recipe.make_critic_penalty(self.opt_d, collect_stats=True)
        self.policy = UpdatePolicy(self.recipe, self.G, self.D, table=self.G.slot,
            router=self.router, generator_optimizer=self.opt_g, critic_optimizer=self.opt_d,
            routed_rows=routing_spec(), row_semantics="conditional", penalty=self.penalty,
            seed=seed, streams={
                "latent_generator": self.streams.generator("prior", component="latent", purpose="indices"),
                "penalty_generator": self.streams.generator("noise", component="penalty", purpose="training"),
                "noise_generator": self.streams.generator("noise", component="generator", purpose="output"),
                "eval_generator": self.streams.generator("eval", component="sampler", purpose="samples")})
        if context is not None:
            context.bind_update_policy(self.policy, external_max_steps=self.max_steps)
        self.audit = PolicyLifecycleAudit(self.policy)
        self.mechanisms = MechanismAudit(self.recipe, self.opt_d, [self.opt_g])
        self.critic = InputNoise(self.D, generator=self.streams.generator("noise", component="critic", purpose="input"))
        self.real = host._batch(host.CONCEPT_DIR, host.N_ROWS).to(self.device)
        self.fit_context = slot_contexts([host.CONCEPT] * host.N_ROWS, device=self.device)
        self.guard_context = slot_contexts([host.UNUSED] * host.N_ROWS, device=self.device)
        self.guard_targets = host._batch(host.NEU[host.UNUSED], host.N_ROWS).to(self.device).clone()
        self.train_hold_context = slot_contexts([host.UNUSED, host.CONCEPT], device=self.device)
        self.pairs = host.hold_pairs("matched")
        self.hold_weight = host.HOLD_WEIGHT
        self.last_update = {}
        self.last_views = None
        self.purity, self.policy_observations, self.rng_audits = [], [], []

    @property
    def completed_steps(self):
        return self.policy.completed_steps

    def step(self):
        from benchmarks.locked_shared.hosts.unused_token_hold import unused_hold_loss
        flags = [parameter.requires_grad for parameter in self.D.parameters()]
        try:
            routed = RoutedBatch(self.fit_context, self.real, self.guard_context, self.guard_targets)
            noise = self.policy.begin_step(self.real, routed=routed, execution_limit=self.max_steps)
            self.critic.std = noise.input_sigma
            self.opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake = self.policy.routed_generate(self.fit_context, sigma=noise.output_sigma, perturb=True)
            self.policy.observe_critic_pair(self.real, fake)
            adversarial_d = self.loss.d_loss(self.critic(self.real), self.critic(fake))
            penalty = self.penalty(self.critic, self.real, fake)
            loss_d = adversarial_d + penalty
            self.policy.before_critic_backward()
            loss_d.backward()
            self.policy.after_critic_backward()
            self.opt_d.step()
            self.policy.after_critic_step()

            self.D.requires_grad_(False)
            self.opt_g.zero_grad(set_to_none=True)
            generated = self.policy.routed_generate(self.fit_context, sigma=noise.output_sigma, perturb=True)
            loss_gan = self.loss.g_loss(self.critic(generated), self.critic(self.real).detach())
            # Original matched hold MSE uses the public clean complete function,
            # with fresh training constants, never the supplied guard tensors.
            held = self.policy.routed_generate(self.train_hold_context, sigma=0., perturb=False)
            hold = self.hold_weight * unused_hold_loss(held, self.G.neu, self.pairs)
            loss_g = loss_gan + hold
            self.policy.before_generator_backward()
            loss_g.backward()
            self.policy.after_generator_backward(loss_gan=loss_gan, loss_critic=adversarial_d)
            self.opt_g.step()
            self.policy.after_generator_step()
            for parameter, flag in zip(self.D.parameters(), flags):
                parameter.requires_grad_(flag)
            event = self.policy.finish_step()
            self.mechanisms.observe_penalty(self.penalty.last_stats)
            values = {"loss_d": loss_d, "loss_g": loss_g, "loss_gan": loss_gan,
                      "penalty": penalty, "unused_hold_loss": hold}
            if not all(bool(torch.isfinite(value)) for value in values.values()):
                raise FloatingPointError("nonfinite routed unused-token update")
            self.last_update = {key: float(value.detach()) for key, value in values.items()}
            return {"step": self.completed_steps, **self.last_update, "row_event": deepcopy(event)}
        except Exception:
            self.policy.abort_step()
            raise
        finally:
            for parameter, flag in zip(self.D.parameters(), flags):
                parameter.requires_grad_(flag)

    def state_dict(self):
        return {"schema_version": 1, "kind": "forge_routed_unused_token_hold_v1", "family": contract.FAMILY,
                "task_sha256": stable_hash(self.task), "recipe": self.recipe.to_dict(),
                "external_max_steps": self.max_steps, "caller_cursor": self.completed_steps,
                "initialization": deepcopy(self.initialization), "streams": self.streams.state_dict(),
                "policy": self.policy.state_dict(), "lifecycle_audit": self.audit.receipt(self.completed_steps),
                "mechanism_audit_state": deepcopy(self.mechanisms.rows), "last_update": deepcopy(self.last_update)}

    def load_state_dict(self, state):
        expected = self.state_dict()
        if not isinstance(state, dict) or state.keys() != expected.keys():
            raise ValueError("invalid routed shared/slot checkpoint schema")
        for key in ("schema_version", "kind", "family", "task_sha256", "recipe", "external_max_steps", "initialization"):
            if typed_state_digest(state[key]) != typed_state_digest(expected[key]):
                raise ValueError(f"routed checkpoint {key} differs")
        steps, audit = state["caller_cursor"], state["lifecycle_audit"]
        if (type(steps) is not int or not 0 <= steps <= self.max_steps
                or state["policy"].get("completed_steps") != steps or not finite_policy_state(state)
                or audit.get("end_completed_steps") != steps or audit.get("complete") is not True
                or audit.get("start_completed_steps") != 0
                or audit.get("calls") != {name: steps for name in self.audit.calls}
                or audit.get("pending") or audit.get("order_errors") != 0
                or audit.get("last_order") != (list(self.audit.calls) if steps else [])):
            raise ValueError("invalid routed checkpoint health/cursor/lifecycle")
        mechanisms = state["mechanism_audit_state"]
        if not isinstance(mechanisms, dict) or mechanisms.keys() != self.mechanisms.rows.keys():
            raise ValueError("invalid routed mechanism-audit state")
        for name, row in mechanisms.items():
            initial = self.mechanisms.rows[name]
            if (not isinstance(row, dict) or row.keys() != initial.keys()
                    or any(type(row[key]) is not bool or row[key] != initial[key]
                           for key in ("requested", "enabled"))
                    or any(type(row[key]) is not int or row[key] < 0 for key in ("calls", "eligible", "applied"))
                    or not row["applied"] <= row["eligible"] <= row["calls"] <= steps):
                raise ValueError("routed mechanism counters disagree with public ownership/update cursor")
        if (not isinstance(state["last_update"], dict)
                or set(state["last_update"]) != ({"loss_d", "loss_g", "loss_gan", "penalty", "unused_hold_loss"} if steps else set())
                or any(type(value) is not float or not math.isfinite(value) for value in state["last_update"].values())):
            raise ValueError("invalid routed last-update statistics")
        self.streams.validate_state_dict(state["streams"])
        for name, value in state["policy"]["streams"].items():
            stream = getattr(self.policy, name)
            key = next((key for key, owned in self.streams._streams.items() if stream is owned), None)
            if key is None or not torch.equal(value, state["streams"]["states"].get(key, torch.empty(0))):
                raise ValueError("routed policy and named checkpoint streams disagree")
        # Public policy performs full schema validation before writing owners;
        # deepcopy keeps the supplied checkpoint untouched even on success.
        self.policy.load_state_dict(deepcopy(state["policy"]))
        self.streams.load_state_dict(deepcopy(state["streams"]))
        self.audit.start_steps = audit["start_completed_steps"]
        self.audit.calls, self.audit.pending = dict(audit["calls"]), list(audit["pending"])
        self.audit.order_errors, self.audit.last_order = audit["order_errors"], list(audit["last_order"])
        self.mechanisms.rows, self.last_update = deepcopy(mechanisms), deepcopy(state["last_update"])

    def observe(self):
        from benchmarks.locked_shared.hosts.unused_token_hold import score_student, CONCEPT_DIR
        before = typed_state_digest(evaluation_state(self.state_dict()))
        rng_before = self.streams.audit()
        global_before = typed_state_digest(_global_rng())
        served = self.policy.served_model()
        selected = _SelectedStudent(served)
        with torch.no_grad():
            metrics = score_student(selected)
            embeds = selected.embeds(1.)
        after = typed_state_digest(evaluation_state(self.state_dict()))
        audit = self.streams.compare(rng_before, self.streams.audit())
        global_after = typed_state_digest(_global_rng())
        pure = before == after and global_before == global_after and audit["unintended_rng_deviations"] == 0
        self.rng_audits.append(audit)
        self.purity.append({"completed_steps": self.completed_steps, "digest_kind": DIGEST_KIND,
                            "before_sha256": before, "after_sha256": after,
                            "global_rng_before_sha256": global_before,
                            "global_rng_after_sha256": global_after, "pure": pure})
        observed = {**observation_receipt(self.task, self.policy), "family": contract.FAMILY,
                    "backend": self.policy._feature_selection.state["actual_backend"],
                    "row_policy": self.policy.row_policy}
        self.policy_observations.append(observed)
        self.last_views = {"neu": selected.neu.detach().cpu().clone(), "embeds": embeds.detach().cpu().clone(),
                           "concept_target": (selected.neu[1] + CONCEPT_DIR.to(selected.neu)).detach().cpu().clone(),
                           "selected_shared": served.models["generator"].shared.detach().cpu().clone(),
                           "selected_slot_table": served.table.detach().cpu().clone(),
                           "selected_slot_ids": served.models["router"].slot_ids.detach().cpu().clone(),
                           "selected_log_mass": served.models["router"].log_mass.detach().cpu().clone()}
        if not pure:
            raise RuntimeError("selected routed measurement changed training state or global/named RNG")
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("nonfinite selected shared/slot measurement")
        return metrics

    def guards(self):
        def count(optimizer, parameters):
            values = [int(optimizer.state[value]["step"]) for value in parameters
                      if value in optimizer.state and "step" in optimizer.state[value]]
            return min(values) if values else 0
        controls = routed_controls_receipt(self.policy, self.completed_steps)
        mechanisms = self.mechanisms.receipt()
        direct = mechanisms["mechanisms"]["direct_particle_gain"]
        direct.update(applicable_to_routed_table=False, host_activation_credit=False,
                      owner_semantics="conditional_table_with_no_independent_direct_response",
                      synthetic_probe_scope="independent_scratch_hook_only_not_routed_transport")
        return {"all_finite": finite_policy_state(self.state_dict()),
                "optimizer_updates": {"generator": count(self.opt_g, [self.G.shared]),
                                      "table": count(self.opt_g, [self.G.slot]),
                                      "discriminator": count(self.opt_d, self.D.parameters())},
                "hooks_exercised": controls["implementation_observed"] and not mechanism_blockers(mechanisms),
                "mechanism_audit": mechanisms,
                "unintended_rng_deviations": sum(value["unintended_rng_deviations"] for value in self.rng_audits)}

    def receipt(self):
        if self.context is not None:
            applied = self.context.receipt()
        else:
            applied = {"execution_path": "public_components", "recipe": self.recipe.to_dict(),
                       "rng": self.streams.manifest(), "initialization": deepcopy(self.initialization)}
        return {**applied, "family": contract.FAMILY, "task_cohort": contract.COHORT,
                "policy_lifecycle": {"owner": "particlegan.UpdatePolicy", "completed_steps": self.completed_steps,
                                     "external_max_steps": self.max_steps,
                                     "controls": routed_controls_receipt(self.policy, self.completed_steps)},
                "table_ownership": {"model": "SharedSlotStudent", "parameter": "slot", "shape": [2, 2],
                                    "shared_parameter": "shared", "sampled_independent_prior": False,
                                    "latent_damping_owner": True,
                                    "independent_direct_particle_response_owner": False,
                                    "direct_response_ineligibility": "public_routed_transport_rejects_coupled_direct_row_history",
                                    "row_buffers": ["slot_ids"], "log_mass": "router.log_mass"},
                "independent_atlas_qualification": False}


def run_behavior(request, task, output_dir, device="cpu", *, context=None):
    """Actual bounded runner; only root may launch scientific acquisition."""
    from .artifacts import manifest_artifacts
    from .sampling import executed_receipt
    started = time.monotonic()
    fixture = UnusedTokenRoutedFixture(request, task, device=device, context=context)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    artifacts = output / "routed-policy-behavior"; artifacts.mkdir()
    views = artifacts / "observations"; views.mkdir()
    cadence = {math.ceil(i * fixture.max_steps / 24) for i in range(1, 25)}
    observations = []
    for step in range(1, fixture.max_steps + 1):
        fixture.step()
        if step in cadence:
            metrics = fixture.observe()
            observations.append({"step": step, **metrics})
            np.savez_compressed(views / f"step_{step:06d}.npz",
                                **{key: value.numpy() for key, value in fixture.last_views.items()})
            print(json.dumps({"event": "observation", "task": task["id"], "step": step,
                                           "budget": fixture.max_steps, "metrics": metrics}), flush=True)
    state = fixture.state_dict(); state_path = artifacts / "state.pt"; torch.save(state, state_path)
    evidence = {"observations": observations, "live": observations[-1], "scoring_weights": "state_selected",
                "guards": fixture.guards(), "policy_controls": routed_controls_receipt(fixture.policy, fixture.completed_steps),
                "policy_observation": fixture.policy_observations[-1], "policy_observations": fixture.policy_observations,
                "policy_purity": fixture.purity, "rng_audits": fixture.rng_audits,
                **executed_receipt(task["evaluation"]["sampling_law"], eval_output_noise=task["evaluation"]["eval_output_noise"]),
                "checkpoint": {"path": "state.pt", "sha256": file_hash(state_path),
                               "state_sha256": typed_state_digest(state), "digest_kind": DIGEST_KIND},
                "artifact_root": str(artifacts.resolve()), "artifact_manifest": manifest_artifacts(artifacts),
                "host": {"source": contract.HOST_SOURCE, "objective": "original RpGAN concept + matched unused hold MSE",
                         "hold_weight": fixture.hold_weight, "hold_pairs": fixture.pairs,
                         "routing": "complete fixed slot_lookup + additive candidate mass",
                         "protected_context_scope": "known training anchor protected during structural acceptance; not unseen generalization",
                         "measurement": "selected complete shared/slot function; no DV12 or output-noise draw",
                         "family": contract.FAMILY, "independent_atlas_qualification": False}}
    result = {"task_id": task["id"], "evidence": evidence, "execution_path": "public_components",
              "device": str(device), "applied": fixture.receipt(),
              "cost": {"wall_seconds": time.monotonic() - started, "completed_steps": fixture.completed_steps,
                       "optimizer_updates": evidence["guards"]["optimizer_updates"]}}
    atomic_json(output / "result.json", result)
    return result
