"""Original two-pole leftover objective with an explicit public master bank."""
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from particlegan import ParticlePrior, ParticleRegularizer, RoutedBatch, RoutedRows, UpdatePolicy, init
from particlegan.birth_death import ScalarHeadFeatures
from particlegan.training import InputNoise
from benchmarks.locked_shared.hosts import cover_leftover as host

from . import multibank_policy_contracts as declaration
from .contracts import atomic_json, file_hash, stable_hash
from .mechanisms import MechanismAudit, mechanism_blockers
from .policy_adapters import (PolicyLifecycleAudit, controls_receipt, evaluation_state, finite_policy_state,
                             observation_receipt, typed_state_digest, DIGEST_KIND)
from .rng import NamedStreams
from .routed_policy_adapters import _global_rng

ROOT = Path(__file__).resolve().parents[2]


class DeclaredResidual(host._Residual):
    """Original vectors and function, with new-class-only zero-init declarations."""


init.register(DeclaredResidual, {"w_odd": init.KEEP, "w_even": init.KEEP})


class _SelectedResidual:
    """Original CPU scorer interface over independent selected outputs."""
    def __init__(self, residual):
        self.residual = residual

    def delta(self, scale):
        # The source scorer creates CPU basis vectors. Preserve each selected
        # forward on its original device and detach only its scoring values.
        return self.residual.delta(scale).detach().cpu().clone()


class PolarRouter(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.register_buffer("log_mass", torch.zeros(24, device=device))
        self.register_buffer("bank_ids", torch.arange(24, device=device) // 12)
        self.register_buffer("neu", host.LeftoverField().poles()[2].to(device))


def polar_context(bank, local_rows, jitter):
    rows = torch.as_tensor(local_rows, device=jitter.device).reshape(-1)
    labels = torch.full_like(rows, bank)
    return torch.cat((labels[:, None].to(jitter), rows[:, None].to(jitter), jitter), 1)


def complete_polar_forward(models, context, candidate, routing):
    if (context.ndim != 2 or context.shape[1] != 6 or not torch.isfinite(context).all()
            or not bool(((context[:, 0] == 0) | (context[:, 0] == 1)).all())
            or not bool(((context[:, 1] >= 0) & (context[:, 1] < 12) & (context[:, 1] == context[:, 1].long())).all())):
        raise ValueError("original bank/local-row/jitter context required")
    bank, local = context[:, 0].long(), context[:, 1].long()
    router = models["router"]
    if not torch.equal(router.bank_ids, torch.arange(24, device=candidate.table.device) // 12):
        raise ValueError("fixed original bank ownership changed")
    wanted = bank * 12 + local
    row = torch.arange(24, device=candidate.table.device)
    same_bank = bank[:, None] == router.bank_ids[None]
    logits = torch.where(same_bank, candidate.table.new_tensor(-1048576.), candidate.table.new_tensor(-2097152.))
    logits = torch.where(wanted[:, None] == row[None], candidate.table.new_zeros(()), logits)
    codes = routing.mix("polar_bank_lookup", logits)
    # Keep the original two delta branches (and their gradient reductions),
    # then restore input order for reservoirs containing several mixed batches.
    positions = [(bank == value).nonzero().flatten() for value in (0, 1)]
    outputs = [router.neu + models["generator"].delta(scale) + codes[indices] + context[indices, 2:]
               for indices, scale in zip(positions, (1., -1.))]
    order = torch.cat(positions)
    return torch.cat(outputs)[order.argsort()]


def polar_features(models, context, samples, targets):
    return ScalarHeadFeatures(models["critic"])(samples)


def polar_spec():
    return RoutedRows(model_forward=complete_polar_forward, features=polar_features,
                      sites=("polar_bank_lookup",), output_error_guard=True,
                      max_context_harm=0., max_output_error_increase=0., max_output_context_harm=0.)


class CompleteRoutedState:
    """Reusable new-test envelope; only the actual public loader writes owners."""
    @property
    def completed_steps(self):
        return self.policy.completed_steps

    def state_dict(self):
        return dict(schema_version=1, kind=self.KIND, family=self.contract.FAMILY,
            task_sha256=stable_hash(self.task), recipe=self.recipe.to_dict(), external_max_steps=self.max_steps,
            caller_cursor=self.completed_steps, initialization=deepcopy(self.initialization),
            streams=self.streams.state_dict(), policy=self.policy.state_dict(),
            lifecycle_audit=self.audit.receipt(self.completed_steps), mechanism_audit_state=deepcopy(self.mechanisms.rows),
            last_update=deepcopy(self.last_update))

    def load_state_dict(self, state):
        expected = self.state_dict()
        if not isinstance(state, dict) or state.keys() != expected.keys():
            raise ValueError("invalid complete routed checkpoint schema")
        for key in ("schema_version", "kind", "family", "task_sha256", "recipe", "external_max_steps", "initialization"):
            if typed_state_digest(state[key]) != typed_state_digest(expected[key]):
                raise ValueError("routed checkpoint identity differs: " + key)
        steps, audit = state["caller_cursor"], state["lifecycle_audit"]
        if (type(steps) is not int or not 0 <= steps <= self.max_steps or state["policy"].get("completed_steps") != steps
                or not finite_policy_state(state) or audit.get("complete") is not True
                or audit.get("start_completed_steps") != 0 or audit.get("end_completed_steps") != steps
                or audit.get("calls") != {key: steps for key in self.audit.calls}
                or audit.get("last_order") != (list(self.audit.calls) if steps else [])
                or audit.get("pending") != [] or audit.get("order_errors") != 0):
            raise ValueError("invalid routed cursor/lifecycle/health")
        mechanisms = state["mechanism_audit_state"]
        if not isinstance(mechanisms, dict) or mechanisms.keys() != self.mechanisms.rows.keys():
            raise ValueError("invalid routed mechanism state")
        for name, row in mechanisms.items():
            current = self.mechanisms.rows[name]
            if (not isinstance(row, dict) or row.keys() != current.keys()
                    or any(type(row[k]) is not bool or row[k] != current[k] for k in ("requested", "enabled"))
                    or any(type(row[k]) is not int or row[k] < 0 for k in ("calls", "eligible", "applied"))
                    or not row["applied"] <= row["eligible"] <= row["calls"] <= steps):
                raise ValueError("invalid routed mechanism counters")
        if (not isinstance(state["last_update"], dict) or (bool(state["last_update"]) != bool(steps))
                or any(type(v) is not float or not math.isfinite(v) for v in state["last_update"].values())):
            raise ValueError("invalid routed last-update statistics")
        for name, module in self.policy._training_modules().items():
            for key, value in module.named_buffers():
                if name == "router" and key == "log_mass":
                    continue
                if typed_state_digest(state["policy"]["models"][name].get(key)) != typed_state_digest(value):
                    raise ValueError("original fixed routed model buffer changed: " + name + "." + key)
        self.streams.validate_state_dict(state["streams"])
        for name, value in state["policy"]["streams"].items():
            stream = getattr(self.policy, name)
            key = next((k for k, owned in self.streams._streams.items() if owned is stream), None)
            if key is None or not torch.equal(value, state["streams"]["states"].get(key, torch.empty(0))):
                raise ValueError("public/named routed stream mismatch")
        self.policy.load_state_dict(deepcopy(state["policy"]))
        self.streams.load_state_dict(deepcopy(state["streams"]))
        self.audit.start_steps = 0; self.audit.calls = dict(audit["calls"])
        self.audit.pending, self.audit.last_order, self.audit.order_errors = [], list(audit["last_order"]), 0
        self.mechanisms.rows, self.last_update = deepcopy(mechanisms), deepcopy(state["last_update"])

    def controls(self):
        row = controls_receipt(self.policy, self.completed_steps)
        row.update(cohort=self.contract.COHORT, family=self.contract.FAMILY,
                   independent_atlas_qualification=False, row_policy="routed_paired")
        return row

    def pure_observe(self, reader):
        before = typed_state_digest(evaluation_state(self.state_dict())); rng = self.streams.audit()
        global_before = typed_state_digest(_global_rng())
        result = reader(self.policy.served_model())
        after = typed_state_digest(evaluation_state(self.state_dict()))
        allowed = [k for k, v in self.streams.manifest()["bindings"].items() if v["family"] == "eval"]
        audit = self.streams.compare(rng, self.streams.audit(), allowed=allowed)
        global_after = typed_state_digest(_global_rng())
        pure = before == after and global_before == global_after and audit["unintended_rng_deviations"] == 0
        self.rng_audits.append(audit)
        self.purity.append(dict(completed_steps=self.completed_steps, digest_kind=DIGEST_KIND,
            before_sha256=before, after_sha256=after, global_rng_before_sha256=global_before,
            global_rng_after_sha256=global_after, pure=pure))
        self.policy_observations.append({**observation_receipt(self.task, self.policy),
            "family": self.contract.FAMILY, "row_policy": "routed_paired"})
        if not pure:
            raise RuntimeError("routed observation changed training state/global/named RNG")
        return result

    def guards(self):
        def count(optimizer, params):
            values = [int(optimizer.state[p]["step"]) for p in params if "step" in optimizer.state.get(p, {})]
            return min(values) if values else 0
        mechanisms = self.mechanisms.receipt()
        mechanisms["mechanisms"]["direct_particle_gain"].update(
            applicable_to_routed_table=False, host_activation_credit=False,
            synthetic_probe_scope="independent_scratch_hook_not_routed_transport")
        return dict(all_finite=finite_policy_state(self.state_dict()), optimizer_updates={
            role: count(optimizer, params) for role, (optimizer, params) in self.role_parameters.items()},
            hooks_exercised=self.controls()["implementation_observed"] and not mechanism_blockers(mechanisms),
            mechanism_audit=mechanisms, unintended_rng_deviations=sum(a["unintended_rng_deviations"] for a in self.rng_audits))

    def receipt(self):
        base = self.context.receipt() if self.context is not None else dict(execution_path="public_components",
            recipe=self.recipe.to_dict(), rng=self.streams.manifest(), initialization=deepcopy(self.initialization))
        return {**base, "family": self.contract.FAMILY, "task_cohort": self.contract.COHORT,
                "policy_lifecycle": dict(owner="particlegan.UpdatePolicy", completed_steps=self.completed_steps,
                    external_max_steps=self.max_steps, controls=self.controls()), "independent_atlas_qualification": False}


def initialize_named(fixture, model, role):
    seeds = {name: fixture.streams.seed_for("init", component=role, purpose=name)
             for name, p in model.named_parameters() if p.requires_grad and p.numel()}
    init.deterministic_orthogonal_(model, parameter_seeds=seeds)
    fixture.initialization[role] = dict(initializer="deterministic_orthogonal_named_parameters_v1",
        parameter_seeds=seeds, state_sha256=typed_state_digest(model.state_dict()))


def complete_owner(fixture, *, encoder=None):
    fixture.opt_d = fixture.recipe.make_critic_optimizer(fixture.D, ema_critic=deepcopy(fixture.D))
    fixture.loss = fixture.recipe.make_loss()
    fixture.penalty = fixture.recipe.make_critic_penalty(fixture.opt_d, collect_stats=True)
    fixture.policy = UpdatePolicy(fixture.recipe, fixture.G, fixture.D, prior=fixture.prior, encoder=encoder,
        router=fixture.router, generator_optimizer=fixture.opt_g, critic_optimizer=fixture.opt_d,
        routed_rows=fixture.routing_spec(), row_semantics="conditional", penalty=fixture.penalty,
        seed=fixture.streams.seed, streams={
            "latent_generator": fixture.streams.generator("prior", component="latent", purpose="indices"),
            "penalty_generator": fixture.streams.generator("noise", component="penalty", purpose="training"),
            "noise_generator": fixture.streams.generator("noise", component="generator", purpose="output"),
            "eval_generator": fixture.streams.generator("eval", component="sampler", purpose="samples")})
    if fixture.context is not None:
        fixture.context.bind_update_policy(fixture.policy, external_max_steps=fixture.max_steps)
    fixture.audit = PolicyLifecycleAudit(fixture.policy)
    fixture.mechanisms = MechanismAudit(fixture.recipe, fixture.opt_d, [fixture.opt_g])
    fixture.critic = InputNoise(fixture.D, generator=fixture.streams.generator("noise", component="critic", purpose="input"))
    fixture.last_update = {}; fixture.last_views = None
    fixture.purity, fixture.policy_observations, fixture.rng_audits = [], [], []
    fixture.role_parameters = {"generator": (fixture.opt_g, list(fixture.G.parameters())),
        "prior": (fixture.opt_g, list(fixture.prior.parameters())), "discriminator": (fixture.opt_d, list(fixture.D.parameters()))}
    if encoder is not None:
        fixture.role_parameters["encoder"] = (fixture.opt_g, list(encoder.parameters()))


class CoverMultibankFixture(CompleteRoutedState):
    KIND = "forge_cover_multibank_policy_v1"
    routing_spec = staticmethod(polar_spec)
    def __init__(self, request, task, *, device="cpu", context=None):
        self.contract = declaration; declaration.validate_task(task, root=ROOT)
        self.task, self.context, self.device = deepcopy(task), context, torch.device(device)
        self.recipe = declaration.resolved_recipe(request["candidate"], task); self.max_steps = 800
        self.streams = NamedStreams(request.get("protocol", {}).get("seed", 0), device=device) if context is None else context.streams
        if context is not None and (stable_hash(context.recipe.to_dict()) != stable_hash(self.recipe.to_dict())
                or stable_hash(context.policy_contract) != stable_hash(task["execution"]["policy_contract"])):
            raise ValueError("multibank injected context differs")
        self.initialization = {}
        self.G = DeclaredResidual(4).to(device)
        with self.streams.fork("init", component="discriminator", purpose="construction", device="cpu"):
            self.D = host._FourierCritic(4, n_rand=16, hidden=64, seed=self.streams.seed).to(device)
        initialize_named(self, self.G, "generator"); initialize_named(self, self.D, "discriminator")
        originals = []
        for i in range(2):
            prior = ParticlePrior(12, 4, init_std=.05, device=device,
                generator=self.streams.generator("init", component=f"prior{i}", purpose="construction"))
            initialize_named(self, prior, f"prior{i}"); originals.append(prior.z.detach().clone())
        self.prior = ParticlePrior(24, 4, init_std=0., device=device,
            generator=self.streams.generator("init", component="master", purpose="construction"))
        with torch.no_grad(): self.prior.z.copy_(torch.cat(originals))
        self.initialization["master"] = dict(initializer="exact_concat_original_two_initialized_clouds",
            original_shapes=[[12, 4], [12, 4]], state_sha256=typed_state_digest(self.prior.state_dict()))
        self.router = PolarRouter(device)
        self.opt_g = self.recipe.make_generator_optimizer([
            {"params": list(self.G.parameters()), "lr": self.recipe.lr},
            {"params": [self.prior.z], "lr": self.recipe.lr * self.recipe.prior_lr_mult,
             "betas": self.recipe.prior_betas or self.recipe.betas}], latent_table=self.prior.z)
        self.field = host.LeftoverField()
        self.poles_p, self.poles_m, self.neu = [x.to(device) for x in host.teacher_poles(self.field, host.LOCKED_TEACHER)]
        self.spread = ParticleRegularizer(target_std=.05, weight=.05)
        complete_owner(self)
        empty = torch.zeros(12, 4, device=device)
        self.guard_context = torch.cat((polar_context(0, torch.arange(12, device=device), empty),
                                        polar_context(1, torch.arange(12, device=device), empty)))
        self.guard_targets = torch.cat((self.poles_p.expand(12, -1), self.poles_m.expand(12, -1))).clone()

    def sampled_context(self):
        chunks = []
        for bank in range(2):
            rows = torch.randint(12, (16,), device=self.device,
                generator=self.streams.generator("prior", component=f"prior{bank}", purpose="indices"))
            jitter = .01 * torch.randn(16, 4, device=self.device,
                generator=self.streams.generator("noise", component=f"prior{bank}", purpose="original_host_jitter"))
            chunks.append(polar_context(bank, rows, jitter))
        return torch.cat(chunks)

    def auxiliary_loss(self, parts=None, generator=None):
        parts = self.prior.z if parts is None else parts; generator = self.G if generator is None else generator
        cover = (self.neu + generator.delta(1.) - self.poles_p).square().mean()
        cover = cover + (self.neu + generator.delta(-1.) - self.poles_m).square().mean()
        return self.spread(parts) + .02 * parts.square().mean() + 1.5 * cover

    def step(self):
        flags = [p.requires_grad for p in self.D.parameters()]
        try:
            with torch.device(self.device), self.streams.fork("data", component="host", purpose="batches"):
                real = torch.cat([host._sample_real_cloud(pole, self.neu, 16, cloud_std=.03,
                    span_frac=.4, end_margin=.6) for pole in (self.poles_p, self.poles_m)])
            fit = self.sampled_context()
            noise = self.policy.begin_step(real, routed=RoutedBatch(fit, real, self.guard_context, self.guard_targets), execution_limit=self.max_steps)
            self.critic.std = noise.input_sigma; self.opt_d.zero_grad(set_to_none=True)
            with torch.no_grad(): fake = self.policy.routed_generate(fit, sigma=noise.output_sigma)
            self.policy.observe_critic_pair(real, fake)
            adversarial_d = self.loss.d_loss(self.critic(real), self.critic(fake)); penalty = self.penalty(self.critic, real, fake)
            loss_d = adversarial_d + penalty; self.policy.before_critic_backward(); loss_d.backward()
            self.policy.after_critic_backward(); self.opt_d.step(); self.policy.after_critic_step()
            self.D.requires_grad_(False); self.opt_g.zero_grad(set_to_none=True)
            fake = self.policy.routed_generate(self.sampled_context(), sigma=noise.output_sigma)
            gan = self.loss.g_loss(self.critic(fake), self.critic(real).detach()); aux = self.auxiliary_loss(); loss_g = gan + aux
            self.policy.before_generator_backward(); loss_g.backward()
            self.policy.after_generator_backward(loss_gan=gan, loss_critic=adversarial_d)
            self.opt_g.step(); self.policy.after_generator_step()
            for p, flag in zip(self.D.parameters(), flags): p.requires_grad_(flag)
            self.policy.finish_step(); self.mechanisms.observe_penalty(self.penalty.last_stats)
            self.last_update = {k: float(v.detach()) for k, v in dict(loss_d=loss_d, loss_g=loss_g, loss_gan=gan, auxiliary=aux, penalty=penalty).items()}
            if not all(math.isfinite(v) for v in self.last_update.values()): raise FloatingPointError("nonfinite multibank update")
            return dict(step=self.completed_steps, **self.last_update)
        except Exception:
            self.policy.abort_step(); raise
        finally:
            for p, flag in zip(self.D.parameters(), flags): p.requires_grad_(flag)

    def observe(self):
        def read(served):
            selected = _SelectedResidual(served.models["generator"])
            scored = host.score_geometry(selected, self.field,
                self.poles_p.detach().cpu().clone(), self.poles_m.detach().cpu().clone(),
                self.neu.detach().cpu().clone())
            self.last_views = dict(neu=self.neu.cpu().clone(), targets=torch.stack((self.poles_p, self.poles_m)).cpu(),
                residual=torch.stack((selected.delta(1.), selected.delta(-1.))),
                selected_bank=served.table.cpu().clone(), bank_ids=served.models["router"].bank_ids.cpu().clone(),
                log_mass=served.models["router"].log_mass.cpu().clone())
            return {k: v for k, v in scored.items() if type(v) in (int, float)}
        with torch.no_grad(): return self.pure_observe(read)


def run_behavior(request, task, output_dir, device="cpu", *, context=None):
    from .artifacts import manifest_artifacts
    from .sampling import executed_receipt
    started = time.monotonic(); fixture = CoverMultibankFixture(request, task, device=device, context=context)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    artifacts = output / "multibank-policy"; artifacts.mkdir(); views = artifacts / "observations"; views.mkdir()
    observations = []; cadence = {math.ceil(i * 800 / 24) for i in range(1, 25)}
    for step in range(1, 801):
        fixture.step()
        if step in cadence:
            metrics = fixture.observe(); observations.append(dict(step=step, **metrics))
            np.savez_compressed(views / f"step_{step:06d}.npz", **{k: v.detach().numpy() for k, v in fixture.last_views.items()})
            print(json.dumps(dict(event="observation", task=task["id"], step=step, metrics=metrics)), flush=True)
    state = fixture.state_dict(); target = artifacts / "state.pt"; torch.save(state, target)
    evidence = dict(observations=observations, live=observations[-1], scoring_weights="state_selected", guards=fixture.guards(),
        policy_controls=fixture.controls(), policy_observation=fixture.policy_observations[-1],
        policy_observations=fixture.policy_observations, policy_purity=fixture.purity, rng_audits=fixture.rng_audits,
        artifact_root=str(artifacts.resolve()), artifact_manifest=manifest_artifacts(artifacts),
        checkpoint=dict(path="state.pt", sha256=file_hash(target), state_sha256=typed_state_digest(state), digest_kind=DIGEST_KIND),
        **executed_receipt(task["evaluation"]["sampling_law"], eval_output_noise=task["evaluation"]["eval_output_noise"]))
    result = dict(task_id=task["id"], evidence=evidence, applied=fixture.receipt(), execution_path="public_components",
        device=str(device), cost=dict(wall_seconds=time.monotonic()-started, completed_steps=800, optimizer_updates=evidence["guards"]["optimizer_updates"]))
    atomic_json(output / "result.json", result); return result
