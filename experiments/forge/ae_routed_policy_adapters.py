"""Original AE hold objective under an explicitly distinct routed-MoG law.

No cloud substitution, historical state, independent-Atlas credit or optimizer
patch is used. Public UpdatePolicy owns all adaptive state and update hooks.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from particlegan import RoutedRows, RoutedBatch, UpdatePolicy, init
from particlegan.particle_prior import MoGParticlePrior
from particlegan.training import InputNoise
from benchmarks.locked_shared.hosts import ae_gan_hold as host
from . import ae_routed_policy_contracts as contract
from .contracts import atomic_json, file_hash, stable_hash
from .rng import NamedStreams
from .policy_adapters import (DIGEST_KIND, PolicyLifecycleAudit, controls_receipt,
    evaluation_state, finite_policy_state, observation_receipt, typed_state_digest)
from .mechanisms import MechanismAudit, mechanism_blockers

ROOT = Path(__file__).resolve().parents[2]


class AERouter(nn.Module):
    def __init__(self, recipe, *, device="cpu", dtype=torch.float32):
        super().__init__()
        self.recipe = recipe
        self.register_buffer("log_mass", torch.zeros(recipe.num_particles, device=device, dtype=dtype))


class FunctionalMoG(MoGParticlePrior):
    """Non-owning MoG view for public encode over a supplied candidate table.

    This creates no parameters, random draws or captured live-table reference.
    The candidate keeps its autograd graph; only the explicit fixed sigma is a
    buffer. It never becomes an optimizer/policy/checkpoint owner.
    """
    def __init__(self, table, sigma):
        nn.Module.__init__(self)
        object.__setattr__(self, "z", table)
        self.standardize = False
        self.register_buffer("sigma", sigma)


def ae_context(data):
    return torch.cat((data.new_zeros((len(data), 1)), data, data.new_zeros((len(data), 3))), 1)


def mog_context(uniform, normal):
    return torch.cat((normal.new_ones((len(normal), 1)), normal.new_zeros((len(normal), 2)),
                      uniform.reshape(-1, 1), normal), 1)


def complete_ae_forward(models, context, candidate, routing):
    """Hard AE + mass-aware MoG generation, using actual routed DV12 codes."""
    if (context.ndim != 2 or context.shape[1] != 6 or not torch.isfinite(context).all()
            or not bool(((context[:, 0] == 0) | (context[:, 0] == 1)).all())):
        raise ValueError("AE context needs finite branch/data/uniform/normal fields")
    recipe, prior = models["router"].recipe, models["prior"]
    if (not isinstance(prior, MoGParticlePrior) or prior.standardize
            or prior.sigma_rel != 0 or float(prior.sigma) != float(prior.sigma.new_tensor(.025))):
        raise ValueError("AE routed law requires explicit fixed-width nonstandardized MoG")
    table = candidate.table
    query, offset = models["encoder"](context[:, 1:3]).chunk(2, 1)
    distance = query.square().sum(1, keepdim=True) + table.detach().square().sum(1)[None] - 2 * query @ table.detach().T
    if recipe.distance_reduction == "mean":
        distance = distance / table.shape[1]
    ae_logits = -distance / recipe.routing_temperature
    generation = context[:, 0] == 1
    uniform = context[:, 3]
    if bool((generation & ((uniform < 0) | (uniform >= 1))).any()):
        raise ValueError("MoG generation uniforms must lie in [0,1)")
    mass = candidate.log_mass.softmax(0)
    indices = torch.searchsorted(mass.cumsum(0), uniform.contiguous(), right=True).clamp_max(len(table) - 1)
    # Fixed categorical lookup retains the original hard generation component
    # at equal mass. Actual candidate mass chooses the component after moves.
    generated_logits = torch.full_like(ae_logits, -1048576.).scatter(1, indices[:, None], 0.)
    logits = torch.where(generation[:, None], generated_logits, ae_logits)
    mixed = routing.mix("ae_mog_bank", logits)
    weights = (logits + candidate.log_mass).softmax(1)
    # DV12 computes its support clipping and bandwidth increment without an
    # autograd path. Detaching this recovered increment avoids introducing
    # cancelling soft-table gradients into the original hard AE graph.
    actual_displacement = (mixed - weights @ table).detach()

    # Call the actual public hard AE. Its query proxy uses detached means;
    # selected centers retain table gradients. Candidate mass adds an
    # explicit correction only when it changes the original uniform law.
    encoded = recipe.encode(query, FunctionalMoG(table, prior.sigma), offset=offset).codes[:, 0]
    if torch.equal(candidate.log_mass, candidate.log_mass[0].expand_as(candidate.log_mass)):
        reconstruction_codes = encoded
    else:
        new_weights = (ae_logits + candidate.log_mass).softmax(1)
        new_proxy = new_weights @ table.detach()
        new_hard = table[(ae_logits + candidate.log_mass).argmax(1)] + (new_proxy - new_proxy.detach())
        reconstruction_codes = new_hard + prior.sigma * 3. * torch.tanh(offset / 3.)
    generation_codes = table[indices] + prior.sigma * context[:, 4:6]
    codes = torch.where(generation[:, None], generation_codes, reconstruction_codes) + actual_displacement
    return models["generator"](codes)


def paired_features(models, context, samples, targets):
    return models["critic"].features(samples)


def routing_spec():
    return RoutedRows(model_forward=complete_ae_forward, features=paired_features,
        sites=("ae_mog_bank",), output_error_guard=True,
        max_context_harm=0., max_output_error_increase=0., max_output_context_harm=0.)


def ae_controls_receipt(policy, steps):
    if policy.row_policy != "routed_paired" or policy.row_semantics != "conditional":
        raise ValueError("AE controls require the actual conditional routed owner")
    value = controls_receipt(policy, steps)
    value.update(cohort=contract.COHORT, family=contract.FAMILY, row_policy="routed_paired",
                 independent_atlas_qualification=False,
                 table_optimizer_semantics=contract._contract({})["table_optimizer_semantics"])
    return value


def ae_sampling_receipt():
    """Recorded at the actual clean selected hard-AE/MoG observation branch."""
    return {"sampling_contract_version": 1, "sampling_law": contract.SAMPLING_LAW,
            "eval_output_noise": contract.EVAL_OUTPUT_NOISE}


def global_rng():
    numpy = np.random.get_state()
    return {"torch_cpu": torch.get_rng_state().clone(), "python": random.getstate(),
            "numpy": [numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]],
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


class AERoutedFixture:
    def __init__(self, request, task, *, device="cpu", context=None):
        contract.validate_ae_task(task, root=ROOT)
        self.task = deepcopy(task)
        self.recipe = contract.resolved_recipe(request["candidate"], task)
        self.device, self.max_steps = torch.device(device), task["execution"]["steps"]
        self.context = context
        self.streams = context.streams if context is not None else NamedStreams(request.get("protocol", {}).get("seed", 0), device=self.device)
        if context is not None and (stable_hash(context.recipe.to_dict()) != stable_hash(self.recipe.to_dict())
                or context.device != self.device or context.streams.seed != request.get("protocol", {}).get("seed", 0)
                or context.policy_contract != task["execution"]["policy_contract"]):
            raise ValueError("AE caller context differs from declared Recipe/law/device/seed")
        self.initialization = {}
        self.models = {}
        for name, dimensions in (("encoder", (2, 4)), ("generator", (2, 2)), ("discriminator", (2, 1))):
            with self.streams.fork("init", component=name, purpose="construction", device="cpu"):
                model = host.MLP(*dimensions).to(self.device)
            self._initialize(model, name)
            self.models[name] = model
        self.E, self.G, self.D = (self.models[n] for n in ("encoder", "generator", "discriminator"))
        self.prior = self.recipe.make_prior(sigma=.025, learnable=True, generator=self.streams.generator(
            "init", component="prior", purpose="locations"), device=self.device, dtype=next(self.G.parameters()).dtype)
        self._initialize(self.prior, "prior0")
        self.router = AERouter(self.recipe, device=self.device, dtype=self.prior.z.dtype)
        self.opt_g = self.recipe.make_generator_optimizer([
            {"params": list(self.G.parameters()), "lr": self.recipe.lr},
            {"params": list(self.E.parameters()), "lr": self.recipe.lr},
            {"params": [self.prior.z], "lr": self.recipe.lr * self.recipe.prior_lr_mult,
             "betas": self.recipe.prior_betas or self.recipe.betas}], latent_table=self.prior.z)
        self.opt_d = self.recipe.make_critic_optimizer(self.D, ema_critic=deepcopy(self.D))
        self.loss = self.recipe.make_loss()
        self.penalty = self.recipe.make_critic_penalty(self.opt_d, collect_stats=True)
        self.policy = UpdatePolicy(self.recipe, self.G, self.D, encoder=self.E, prior=self.prior, router=self.router,
            row_semantics="conditional", routed_rows=routing_spec(), generator_optimizer=self.opt_g,
            critic_optimizer=self.opt_d, penalty=self.penalty, seed=self.streams.seed, streams={
                "latent_generator": self.streams.generator("prior", component="latent", purpose="indices"),
                "penalty_generator": self.streams.generator("noise", component="penalty", purpose="training"),
                "noise_generator": self.streams.generator("noise", component="generator", purpose="output"),
                "eval_generator": self.streams.generator("eval", component="sampler", purpose="samples")})
        if context is not None:
            context.bind_update_policy(self.policy, external_max_steps=self.max_steps)
        self.audit = PolicyLifecycleAudit(self.policy)
        self.mechanisms = MechanismAudit(self.recipe, self.opt_d, [self.opt_g])
        self.critic = InputNoise(self.D, generator=self.streams.generator("noise", component="critic", purpose="input"))
        self.anchors = torch.tensor(host.ANCHORS, device=self.device, dtype=self.prior.z.dtype)
        generation_guard = mog_context(self.anchors.new_tensor([.25, .75]), torch.zeros_like(self.anchors))
        self.guard_context = torch.cat((ae_context(self.anchors), generation_guard), 0).clone()
        self.guard_targets = self.anchors.repeat(2, 1).clone()
        self.last_update, self.last_views = {}, None
        self.purity, self.policy_observations, self.rng_audits = [], [], []
        # Evaluation bindings exist before purity/checkpoint comparisons.
        for component, purpose in (("target", "indices"), ("target", "width"), ("mog", "indices"), ("mog", "width")):
            self.streams.generator("eval", component=component, purpose=purpose)

    def _initialize(self, model, name):
        if self.context is not None:
            self.context.initialize(model, component=name)
            self.initialization = deepcopy(self.context.initialization)
        else:
            seeds = {n: self.streams.seed_for("init", component=name, purpose=n)
                     for n, p in model.named_parameters() if p.requires_grad and p.numel()}
            init.deterministic_orthogonal_(model, parameter_seeds=seeds)
            self.initialization[name] = {"initializer": "deterministic_orthogonal_named_parameters_v1",
                "state_sha256": typed_state_digest(model.state_dict()), "parameter_seeds": seeds}

    @property
    def completed_steps(self):
        return self.policy.completed_steps

    def data(self, n, family):
        indices = torch.randint(2, (n,), device=self.device, generator=self.streams.generator(family, component="target", purpose="indices"))
        noise = torch.randn((n, 2), device=self.device, dtype=self.prior.z.dtype,
                            generator=self.streams.generator(family, component="target", purpose="width"))
        return self.anchors[indices] + host.DATA_STD * noise

    def generation_context(self, n, family):
        uniform = torch.rand(n, device=self.device, dtype=self.prior.z.dtype,
                             generator=self.streams.generator(family, component="mog", purpose="indices"))
        noise = torch.randn((n, 2), device=self.device, dtype=self.prior.z.dtype,
                            generator=self.streams.generator(family, component="mog", purpose="width"))
        return mog_context(uniform, noise)

    def step(self):
        if self.completed_steps >= self.max_steps:
            raise RuntimeError("AE original 250-update allowance exhausted")
        flags = [p.requires_grad for p in self.D.parameters()]
        try:
            real = self.data(self.recipe.batch_size, "data")
            fitted = ae_context(real)
            noise = self.policy.begin_step(real, execution_limit=self.max_steps,
                routed=RoutedBatch(fitted, real, self.guard_context, self.guard_targets))
            self.critic.std = noise.input_sigma
            self.opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake = self.policy.routed_generate(self.generation_context(self.recipe.batch_size, "prior"),
                                                   sigma=noise.output_sigma, perturb=True)
            self.policy.observe_critic_pair(real, fake)
            adversarial_d = self.loss.d_loss(self.critic(real).squeeze(-1), self.critic(fake).squeeze(-1))
            penalty = self.penalty(self.critic, real, fake)
            self.policy.before_critic_backward()
            (adversarial_d + penalty).backward()
            self.policy.after_critic_backward(); self.opt_d.step(); self.policy.after_critic_step()
            self.D.requires_grad_(False); self.opt_g.zero_grad(set_to_none=True)
            reconstructed = self.policy.routed_generate(fitted, sigma=noise.output_sigma, perturb=True)
            generated = self.policy.routed_generate(self.generation_context(self.recipe.batch_size, "prior"),
                                                    sigma=noise.output_sigma, perturb=True)
            recon = (reconstructed - real).square().mean()
            adversarial_g = self.loss.g_loss(self.critic(generated).squeeze(-1), self.critic(real).squeeze(-1).detach())
            cover = torch.cdist(self.anchors, generated).min(1).values.mean()
            particle_l2 = self.prior.z.square().mean()
            loss_g = recon + adversarial_g + host.DEMO_COVER * cover + host.PARTICLE_L2 * particle_l2
            self.policy.before_generator_backward(); loss_g.backward()
            self.policy.after_generator_backward(loss_gan=adversarial_g, loss_critic=adversarial_d)
            self.opt_g.step(); self.policy.after_generator_step()
            for parameter, flag in zip(self.D.parameters(), flags): parameter.requires_grad_(flag)
            event = self.policy.finish_step()
            self.mechanisms.observe_penalty(self.penalty.last_stats)
            values = {"loss_d": adversarial_d + penalty, "loss_g": loss_g, "loss_gan": adversarial_g,
                      "penalty": penalty, "reconstruction": recon, "cover": cover, "particle_l2": particle_l2}
            if not all(bool(torch.isfinite(v)) for v in values.values()):
                raise FloatingPointError("nonfinite AE routed public update")
            self.last_update = {key: float(value.detach()) for key, value in values.items()}
            return {**self.last_update, "step": self.completed_steps, "policy": event}
        except BaseException:
            self.policy.abort_step()
            for parameter, flag in zip(self.D.parameters(), flags): parameter.requires_grad_(flag)
            raise

    def state_dict(self):
        return {"schema_version": 1, "kind": "forge_ae_routed_policy_v1", "task_id": self.task["id"],
            "recipe": self.recipe.to_dict(), "contract": deepcopy(self.task["execution"]["policy_contract"]),
            "external_max_steps": self.max_steps, "caller_cursor": self.completed_steps,
            "initialization": deepcopy(self.initialization), "policy": self.policy.state_dict(),
            "streams": self.streams.state_dict(), "lifecycle_audit": self.audit.receipt(self.completed_steps),
            "mechanism_audit_state": deepcopy(self.mechanisms.rows), "last_update": deepcopy(self.last_update),
            "module_modes": self.module_modes()}

    def module_modes(self):
        owners = {"encoder": self.E, "generator": self.G, "critic": self.critic, "prior": self.prior,
                  "router": self.router, "averaged_generator": self.policy.ema_G,
                  "averaged_encoder": self.policy.ema_encoder, "averaged_prior": self.policy.ema_prior,
                  "averaged_router": self.policy.ema_router}
        return {name: {path: module.training for path, module in owner.named_modules()}
                for name, owner in owners.items()}

    def load_state_dict(self, state):
        expected = self.state_dict()
        if not isinstance(state, dict) or state.keys() != expected.keys():
            raise ValueError("invalid complete AE routed checkpoint schema")
        for key in ("schema_version", "kind", "task_id", "recipe", "contract", "external_max_steps", "initialization", "module_modes"):
            if typed_state_digest(state[key]) != typed_state_digest(expected[key]):
                raise ValueError("AE checkpoint immutable binding changed: " + key)
        for group in ("models", "averages"):
            prior = state["policy"].get(group, {}).get("prior", {})
            original = expected["policy"][group]["prior"]
            if (not torch.equal(prior.get("sigma", torch.empty(0)), original["sigma"])
                    or not torch.equal(prior.get("d0", torch.empty(0)), original["d0"])
                    or prior.get("_extra_state") != original["_extra_state"]):
                raise ValueError("AE fixed MoG width/standardization metadata changed")
        steps, audit = state["caller_cursor"], state["lifecycle_audit"]
        if (type(steps) is not int or not 0 <= steps <= self.max_steps or not finite_policy_state(state)
                or state["policy"].get("completed_steps") != steps or audit.get("start_completed_steps") != 0
                or audit.get("end_completed_steps") != steps or audit.get("complete") is not True
                or audit.get("pending") or audit.get("order_errors") != 0
                or audit.get("calls") != {name: steps for name in self.audit.calls}
                or audit.get("last_order") != (list(self.audit.calls) if steps else [])):
            raise ValueError("AE checkpoint health/cursor/complete lifecycle differs")
        mechanisms = state["mechanism_audit_state"]
        if not isinstance(mechanisms, dict) or mechanisms.keys() != self.mechanisms.rows.keys():
            raise ValueError("AE mechanism owners differ")
        for name, row in mechanisms.items():
            initial = self.mechanisms.rows[name]
            if (row.keys() != initial.keys() or any(type(row[k]) is not bool or row[k] != initial[k] for k in ("requested", "enabled"))
                    or any(type(row[k]) is not int or row[k] < 0 for k in ("calls", "eligible", "applied"))
                    or not row["applied"] <= row["eligible"] <= row["calls"] <= steps):
                raise ValueError("AE mechanism counters differ from real updates")
        if (set(state["last_update"]) != ({"loss_d", "loss_g", "loss_gan", "penalty", "reconstruction", "cover", "particle_l2"} if steps else set())
                or any(type(v) is not float or not math.isfinite(v) for v in state["last_update"].values())):
            raise ValueError("AE last-update numeric metadata invalid")
        self.streams.validate_state_dict(state["streams"])
        for name, value in state["policy"]["streams"].items():
            owned = getattr(self.policy, name)
            key = next((k for k, stream in self.streams._streams.items() if stream is owned), None)
            if key is None or not torch.equal(value, state["streams"]["states"].get(key, torch.empty(0))):
                raise ValueError("AE policy/named stream checkpoint disagreement")
        self.policy.load_state_dict(deepcopy(state["policy"]))
        self.streams.load_state_dict(deepcopy(state["streams"]))
        self.audit.start_steps, self.audit.calls = 0, dict(audit["calls"])
        self.audit.pending, self.audit.order_errors, self.audit.last_order = list(audit["pending"]), audit["order_errors"], list(audit["last_order"])
        self.mechanisms.rows, self.last_update = deepcopy(mechanisms), deepcopy(state["last_update"])

    def observe(self):
        before, global_before = typed_state_digest(evaluation_state(self.state_dict())), typed_state_digest(global_rng())
        streams_before = self.streams.audit()
        with self.streams.preserve(), torch.no_grad():
            served = self.policy.served_model()
            data = self.data(1024, "eval")
            recon = served.routed_forward(ae_context(data), perturb=False, output_noise=False)
            fake = served.routed_forward(self.generation_context(1024, "eval"), perturb=False, output_noise=False)
            metrics = {"recon_mse": float((recon - data).square().mean()),
                       "hold": float(torch.cdist(self.anchors, fake).min(1).values.mean())}
            self.last_views = {"target": data.cpu().clone(), "reconstruction": recon.cpu().clone(),
                               "samples": fake.cpu().clone(), "anchors": self.anchors.cpu().clone(),
                               "selected_table": served.table.cpu().clone(), "selected_log_mass": served.router.log_mass.cpu().clone()}
        after, global_after = typed_state_digest(evaluation_state(self.state_dict())), typed_state_digest(global_rng())
        audit = self.streams.compare(streams_before, self.streams.audit())
        pure = before == after and global_before == global_after and audit["unintended_rng_deviations"] == 0
        self.rng_audits.append(audit)
        self.purity.append({"completed_steps": self.completed_steps, "before_sha256": before, "after_sha256": after,
                           "global_before_sha256": global_before, "global_after_sha256": global_after,
                           "digest_kind": DIGEST_KIND, "pure": pure})
        self.policy_observations.append({**observation_receipt(self.task, self.policy), "family": contract.FAMILY,
                                        "row_policy": self.policy.row_policy})
        if not pure: raise RuntimeError("AE selected observation changed complete public state/RNG")
        if not all(math.isfinite(v) for v in metrics.values()): raise FloatingPointError("nonfinite AE observation")
        return metrics

    def guards(self):
        def count(opt, parameters):
            values = [int(opt.state[p]["step"]) for p in parameters if p in opt.state and "step" in opt.state[p]]
            return min(values) if values else 0
        mechanisms = self.mechanisms.receipt()
        mechanisms["mechanisms"]["direct_particle_gain"].update(applicable_to_routed_table=False, host_activation_credit=False)
        controls = ae_controls_receipt(self.policy, self.completed_steps)
        return {"all_finite": finite_policy_state(self.state_dict()),
            "optimizer_updates": {"encoder": count(self.opt_g, self.E.parameters()), "generator": count(self.opt_g, self.G.parameters()),
                                  "prior": count(self.opt_g, [self.prior.z]), "discriminator": count(self.opt_d, self.D.parameters())},
            "hooks_exercised": controls["implementation_observed"] and not mechanism_blockers(mechanisms),
            "mechanism_audit": mechanisms, "unintended_rng_deviations": sum(r["unintended_rng_deviations"] for r in self.rng_audits)}

    def receipt(self):
        applied = (self.context.receipt() if self.context is not None else {
            "execution_path": "public_components", "recipe": self.recipe.to_dict(),
            "rng": self.streams.manifest(), "initialization": deepcopy(self.initialization)})
        return {**applied, "family": contract.FAMILY, "task_cohort": contract.COHORT,
            "policy_lifecycle": {"owner": "particlegan.UpdatePolicy", "completed_steps": self.completed_steps,
                "external_max_steps": self.max_steps, "controls": ae_controls_receipt(self.policy, self.completed_steps)},
            "table_ownership": {"parameter": "prior.z", "kind": "mog", "shape": [12, 2],
                "sigma": .025, "sigma_rel": 0., "standardize": False, "encoder": "original_hard_AE_query_offset",
                "log_mass": "router.log_mass", "latent_damping_owner": True,
                "independent_direct_particle_response_owner": False,
                "optimizer_groups": "separate_generator_encoder_prior_and_public_output_noise"},
            "independent_atlas_qualification": False}


def run_behavior(request, task, output_dir, device="cpu", *, context=None):
    """Full declared producer; root alone owns scientific admission/launch."""
    from .artifacts import manifest_artifacts
    started = time.monotonic()
    fixture = AERoutedFixture(request, task, device=device, context=context)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    artifacts = output / "ae-routed-policy"; views = artifacts / "observations"; views.mkdir(parents=True)
    cadence = {math.ceil(i * fixture.max_steps / 24) for i in range(1, 25)}
    observations = []
    for step in range(1, fixture.max_steps + 1):
        fixture.step()
        if step in cadence:
            metrics = fixture.observe(); observations.append({"step": step, **metrics})
            np.savez_compressed(views / f"step_{step:06d}.npz", **{k: v.numpy() for k, v in fixture.last_views.items()})
            print(json.dumps({"event": "observation", "task": task["id"], "step": step,
                              "budget": fixture.max_steps, "metrics": metrics}), flush=True)
    state_path = artifacts / "state.pt"; state = fixture.state_dict(); torch.save(state, state_path)
    evidence = {"observations": observations, "live": observations[-1], "scoring_weights": "state_selected",
        "guards": fixture.guards(), "policy_controls": ae_controls_receipt(fixture.policy, fixture.completed_steps),
        "policy_observation": fixture.policy_observations[-1], "policy_observations": fixture.policy_observations,
        "policy_purity": fixture.purity, "rng_audits": fixture.rng_audits,
        **ae_sampling_receipt(),
        "checkpoint": {"path": "state.pt", "sha256": file_hash(state_path), "state_sha256": typed_state_digest(state), "digest_kind": DIGEST_KIND},
        "artifact_root": str(artifacts.resolve()), "artifact_manifest": manifest_artifacts(artifacts),
        "host": {"source": contract.HOST_SOURCE, "family": contract.FAMILY,
                 "objective": contract._contract({})["objective"], "prior_sigma": .025,
                 "guard_scope": "separate known original anchors, structural acceptance only",
                 "independent_atlas_qualification": False, "evidence_reuse": False}}
    result = {"task_id": task["id"], "evidence": evidence, "execution_path": "public_components", "device": str(device),
        "applied": fixture.receipt(),
        "cost": {"wall_seconds": time.monotonic() - started, "completed_steps": fixture.completed_steps,
                 "optimizer_updates": evidence["guards"]["optimizer_updates"]}}
    atomic_json(output / "result.json", result)
    return result
