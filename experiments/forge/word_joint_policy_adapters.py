"""Original free-encoder BiGAN with a named complete joint-cloud callback.

This caller implements no optimizer or controller.  The public UpdatePolicy
owns independent row evidence, birth/death, DV12, averaging, stationarity,
noise and all original optimizer memories.  A complete generated atom joins
the same effective code to the word output.  Learned noise is explicitly
restricted to word coordinates.  Neither adaptation claims numerical
equivalence to the old split-code DV12 loop or independent-Atlas credit.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from particlegan import UpdatePolicy, init
from particlegan.training import InputNoise
from benchmarks.toy_audit.api_images import (WordGenerator, WordEncoder, WordJointCritic,
    WORD_DIM, WORD_LENGTH, WORD_CHARS, WORDS, word_bank, score_words)

from . import word_joint_policy_contracts as declaration
from .contracts import atomic_json, file_hash, stable_hash
from .mechanisms import MechanismAudit, mechanism_blockers
from .policy_adapters import (DIGEST_KIND, PolicyLifecycleAudit, controls_receipt,
    evaluation_state, finite_policy_state, observation_receipt, typed_state_digest)
from .rng import NamedStreams
from .routed_policy_adapters import _global_rng

ROOT = Path(__file__).resolve().parents[2]


def word_global_rng():
    """Capture already-owned global CUDA streams without creating a context."""
    value = _global_rng()
    value["cuda"] = ([torch.cuda.get_rng_state(i).clone() for i in range(torch.cuda.device_count())]
                     if torch.cuda.is_initialized() else [])
    return value


def join_words(words, code):
    return torch.cat((words.flatten(1), code), 1)


def joint_generation(model, effective_code):
    """One actual atom; callback operates only on the supplied selected G."""
    return join_words(model(effective_code), effective_code)


def words_only_noise(joint, sigma, generator):
    """Original 28x6 output-noise draw; the two joint-code values stay exact."""
    if joint.ndim != 2 or joint.shape[1] != WORD_DIM + 2:
        raise ValueError("complete word/code joint required")
    if sigma == 0:
        return joint
    draw = torch.randn((len(joint), len(WORD_CHARS), WORD_LENGTH), device=joint.device,
                       dtype=joint.dtype, generator=generator).flatten(1)
    return torch.cat((joint[:, :WORD_DIM] + sigma * draw, joint[:, WORD_DIM:]), 1)


def _initialize_named(fixture, model, role):
    if fixture.context is not None:
        fixture.context.initialize(model, component=role)
        fixture.initialization = deepcopy(fixture.context.initialization)
        return
    seeds = {name: fixture.streams.seed_for("init", component=role, purpose=name)
             for name, p in model.named_parameters() if p.requires_grad and p.numel()}
    init.deterministic_orthogonal_(model, parameter_seeds=seeds)
    fixture.initialization[role] = dict(initializer="deterministic_orthogonal_named_parameters_v1",
        parameter_seeds=seeds, state_sha256=typed_state_digest(model.state_dict()))


def _word_declaration(task):
    """Fixed declaration dispatch; no injected resolver or runtime rate patch."""
    cohort = task.get("task_cohort")
    if cohort == declaration.COHORT:
        return declaration
    if cohort == "word_joint_policy_min11_rates_v1":
        from . import word_joint_rate_policy_contracts
        return word_joint_rate_policy_contracts
    raise ValueError("unknown explicit min11 word policy cohort")


class WordJointPolicyFixture:
    KIND = "forge_word_joint_policy_min11_v1"

    def __init__(self, request, task, *, device="cpu", context=None):
        self.declaration = _word_declaration(task)
        self.declaration.validate_task(task, root=ROOT)
        if self.declaration.COHORT == "word_joint_policy_min11_rates_v1":
            from .policy_cohorts import policy_task_declaration
            task = policy_task_declaration(task)
            self.recipe = self.declaration.validate_request(request, task, root=ROOT)
            self.rate_profile = self.declaration.profile_for(request["candidate"])
            self.KIND = self.declaration.KIND
        else:
            self.recipe = declaration.resolved_recipe(request["candidate"], task)
            self.rate_profile = None
        self.task, self.context, self.device = deepcopy(task), context, torch.device(device)
        self.max_steps = task["execution"]["steps"]
        seed = request.get("protocol", {}).get("seed", 0)
        if context is not None and (stable_hash(context.recipe.to_dict()) != stable_hash(self.recipe.to_dict())
                or context.device != self.device or context.streams.seed != seed
                or stable_hash(context.policy_contract) != stable_hash(task["execution"]["policy_contract"])):
            raise ValueError("word joint context differs from declared owners/Recipe/device/streams")
        self.streams = NamedStreams(seed, device=self.device) if context is None else context.streams
        self.initialization = {}
        for factory, role, attribute in ((WordGenerator, "generator", "G"),
                (WordJointCritic, "discriminator", "D"), (WordEncoder, "encoder", "E")):
            with self.streams.fork("init", component=role, purpose="construction", device="cpu"):
                model = factory().to(device)
            _initialize_named(self, model, role); setattr(self, attribute, model)
        if context is None:
            self.prior = self.recipe.make_prior(device=self.device,
                generator=self.streams.generator("init", component="prior", purpose="locations"))
            _initialize_named(self, self.prior, "prior")
        else:
            self.prior = context.build_prior()
            self.initialization = deepcopy(context.initialization)
        if (self.prior.z.shape != (11, 2) or self.recipe.encoder_mode != "none"
                or self.recipe.row_policy != "independent" or self.recipe.conditioning != "scalar"):
            raise ValueError("explicit eleven independent rows with original free encoder required")
        self.opt_g = self.recipe.make_generator_optimizer([
            dict(params=list(self.G.parameters()), lr=self.recipe.lr),
            dict(params=list(self.E.parameters()), lr=self.recipe.lr),
            dict(params=list(self.prior.parameters()), lr=self.recipe.lr * self.recipe.prior_lr_mult,
                 betas=self.recipe.prior_betas or self.recipe.betas)], latent_table=self.prior.z)
        self.opt_d = self.recipe.make_critic_optimizer(self.D, ema_critic=deepcopy(self.D))
        self.loss = self.recipe.make_loss()
        self.spread = self.recipe.make_prior_regularizer(weight=1.)
        self.penalty = self.recipe.make_critic_penalty(self.opt_d, collect_stats=True)
        self.policy = UpdatePolicy(self.recipe, self.G, self.D, prior=self.prior, encoder=self.E,
            generator_optimizer=self.opt_g, critic_optimizer=self.opt_d,
            generation=joint_generation, row_semantics="independent", penalty=self.penalty, seed=seed,
            streams={"latent_generator": self.streams.generator("prior", component="latent", purpose="indices"),
                "penalty_generator": self.streams.generator("noise", component="penalty", purpose="training"),
                "noise_generator": self.streams.generator("noise", component="generator", purpose="output"),
                "eval_generator": self.streams.generator("eval", component="sampler", purpose="samples")})
        if context is not None:
            context.bind_update_policy(self.policy, external_max_steps=self.max_steps)
        self.audit = PolicyLifecycleAudit(self.policy)
        self.mechanisms = MechanismAudit(self.recipe, self.opt_d, [self.opt_g])
        self.critic = InputNoise(self.D, generator=self.streams.generator("noise", component="critic", purpose="input"))
        self.data_stream = self.streams.generator("data", component="target", purpose="training")
        self.words = word_bank(device=device)
        self.last_update = {}; self.last_views = None
        self.purity, self.policy_observations, self.rng_audits = [], [], []
        self.role_parameters = {"generator": (self.opt_g, list(self.G.parameters())),
            "encoder": (self.opt_g, list(self.E.parameters())), "prior": (self.opt_g, list(self.prior.parameters())),
            "discriminator": (self.opt_d, list(self.D.parameters()))}

    @property
    def completed_steps(self):
        return self.policy.completed_steps

    def fake_joint(self, latent, rows, sigma):
        effective = self.policy.generate(latent, sigma=0., rows=rows)
        return words_only_noise(effective, sigma, self.policy.noise_generator)

    def generator_objective(self, real_words, joint_fake):
        joint_real = join_words(real_words, self.E(real_words))
        gan = self.loss.g_loss(self.critic(joint_fake), self.critic(joint_real))
        auxiliary = self.recipe.prior_reg * self.spread(self.prior.z)
        return gan + auxiliary, gan, auxiliary

    def step(self):
        if self.completed_steps >= self.max_steps:
            raise RuntimeError("word joint original execution budget exhausted")
        flags = [p.requires_grad for p in self.D.parameters()]
        try:
            ids = torch.randint(5, (256,), device=self.device, generator=self.data_stream)
            real_words = self.words[ids]
            with torch.no_grad(): joint_real = join_words(real_words, self.E(real_words))
            noise = self.policy.begin_step(joint_real, execution_limit=self.max_steps)
            self.critic.std = noise.input_sigma; self.opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                latent, rows = self.prior.sample(256, generator=self.policy.latent_generator)
                joint_fake = self.fake_joint(latent, rows, noise.output_sigma)
            self.policy.observe_critic_pair(joint_real, joint_fake)
            adversarial_d = self.loss.d_loss(self.critic(joint_real), self.critic(joint_fake))
            penalty = self.penalty(self.critic, joint_real, joint_fake); loss_d = adversarial_d + penalty
            self.policy.before_critic_backward(); loss_d.backward(); self.policy.after_critic_backward()
            self.opt_d.step(); self.policy.after_critic_step()
            self.D.requires_grad_(False); self.opt_g.zero_grad(set_to_none=True)
            latent, rows = self.prior.sample(256, generator=self.policy.latent_generator)
            joint_fake = self.fake_joint(latent, rows, noise.output_sigma)
            loss_g, gan, auxiliary = self.generator_objective(real_words, joint_fake)
            self.policy.before_generator_backward(); loss_g.backward()
            self.policy.after_generator_backward(loss_gan=gan.detach(), loss_critic=adversarial_d.detach())
            self.opt_g.step(); self.policy.after_generator_step()
            for p, flag in zip(self.D.parameters(), flags): p.requires_grad_(flag)
            self.policy.finish_step(); self.mechanisms.observe_penalty(self.penalty.last_stats)
            self.last_update = {k: float(v.detach()) for k, v in dict(loss_d=loss_d, loss_g=loss_g,
                loss_gan=gan, auxiliary=auxiliary, penalty=penalty).items()}
            if not all(math.isfinite(v) for v in self.last_update.values()):
                raise FloatingPointError("nonfinite word joint update")
            return dict(step=self.completed_steps, **self.last_update)
        except Exception:
            self.policy.abort_step(); raise
        finally:
            for p, flag in zip(self.D.parameters(), flags): p.requires_grad_(flag)

    def state_dict(self):
        result = dict(schema_version=1, kind=self.KIND, family=self.declaration.FAMILY,
            task_sha256=stable_hash(self.task), recipe=self.recipe.to_dict(), external_max_steps=self.max_steps,
            caller_cursor=self.completed_steps, initialization=deepcopy(self.initialization),
            module_modes=self.module_modes(),
            canonical_words=self.words.detach().clone(), streams=self.streams.state_dict(), policy=self.policy.state_dict(),
            lifecycle_audit=self.audit.receipt(self.completed_steps), mechanism_audit_state=deepcopy(self.mechanisms.rows),
            last_update=deepcopy(self.last_update))
        if self.rate_profile is not None:
            result["word_rate_binding"] = self.declaration.binding_receipt(self.recipe, self.rate_profile)
        return result

    def modules(self):
        return {role + ("." + name if name else ""): module
            for role, model in (("generator", self.G), ("encoder", self.E),
                                ("critic", self.D), ("prior", self.prior),
                                ("averaged_generator", self.policy.ema_G),
                                ("averaged_encoder", self.policy.ema_encoder),
                                ("averaged_prior", self.policy.ema_prior))
            for name, module in model.named_modules()}

    def module_modes(self):
        return {name: module.training for name, module in self.modules().items()}

    def load_state_dict(self, state):
        expected = self.state_dict()
        if not isinstance(state, dict) or state.keys() != expected.keys():
            raise ValueError("invalid complete word joint checkpoint schema")
        modes = state["module_modes"]
        if (not isinstance(modes, dict) or modes.keys() != self.module_modes().keys()
                or any(type(mode) is not bool for mode in modes.values())):
            raise ValueError("invalid complete word joint model modes")
        for key in ("table", "averaged_table"):
            table = state["policy"].get(key)
            if (not isinstance(table, torch.Tensor) or table.shape != (11, 2)
                    or table.dtype != self.prior.z.dtype):
                raise ValueError("word joint checkpoint requires eleven actual prior rows")
        for key in ("schema_version", "kind", "family", "task_sha256", "recipe", "external_max_steps",
                    "initialization", "canonical_words", *(["word_rate_binding"] if self.rate_profile is not None else [])):
            if typed_state_digest(state[key]) != typed_state_digest(expected[key]):
                raise ValueError("word joint checkpoint identity differs: " + key)
        steps, audit = state["caller_cursor"], state["lifecycle_audit"]
        if (type(steps) is not int or not 0 <= steps <= self.max_steps or state["policy"].get("completed_steps") != steps
                or not finite_policy_state(state) or audit.get("complete") is not True
                or audit.get("start_completed_steps") != 0 or audit.get("end_completed_steps") != steps
                or audit.get("calls") != {k: steps for k in self.audit.calls}
                or audit.get("last_order") != (list(self.audit.calls) if steps else [])
                or audit.get("pending") != [] or audit.get("order_errors") != 0):
            raise ValueError("invalid word joint cursor/lifecycle/health")
        rows = state["mechanism_audit_state"]
        if not isinstance(rows, dict) or rows.keys() != self.mechanisms.rows.keys():
            raise ValueError("invalid word joint mechanism state")
        for name, row in rows.items():
            original = self.mechanisms.rows[name]
            if (not isinstance(row, dict) or row.keys() != original.keys()
                    or any(type(row[k]) is not bool or row[k] != original[k] for k in ("requested", "enabled"))
                    or any(type(row[k]) is not int or row[k] < 0 for k in ("calls", "eligible", "applied"))
                    or not row["applied"] <= row["eligible"] <= row["calls"] <= steps):
                raise ValueError("invalid word joint mechanism counters")
        if (not isinstance(state["last_update"], dict) or bool(state["last_update"]) != bool(steps)
                or any(type(v) is not float or not math.isfinite(v) for v in state["last_update"].values())):
            raise ValueError("invalid word joint last-update statistics")
        for name, module in self.policy._training_modules().items():
            for key, value in module.named_buffers():
                if typed_state_digest(state["policy"]["models"][name].get(key)) != typed_state_digest(value):
                    raise ValueError("fixed word model buffer changed")
        self.streams.validate_state_dict(state["streams"])
        for name, value in state["policy"]["streams"].items():
            stream = getattr(self.policy, name)
            key = next((k for k, owner in self.streams._streams.items() if owner is stream), None)
            if key is None or not torch.equal(value, state["streams"]["states"].get(key, torch.empty(0))):
                raise ValueError("word joint public/named stream mismatch")
        self.policy.load_state_dict(deepcopy(state["policy"]))
        self.streams.load_state_dict(deepcopy(state["streams"]))
        self.audit.start_steps = 0; self.audit.calls = dict(audit["calls"])
        self.audit.pending, self.audit.last_order, self.audit.order_errors = [], list(audit["last_order"]), 0
        self.mechanisms.rows, self.last_update = deepcopy(rows), deepcopy(state["last_update"])
        for name, module in self.modules().items():
            module.training = modes[name]

    def observe(self, n=1024):
        if type(n) is not int or n != self.task["evaluation"]["eval_samples"]:
            raise ValueError("word observer requires the original full1024 population")
        before = typed_state_digest(evaluation_state(self.state_dict())); rng = self.streams.audit()
        global_before = typed_state_digest(word_global_rng())
        with torch.no_grad():
            served = self.policy.served_model()
            generated_stream = self.streams.generator("eval", component="selected", purpose="generated_words")
            reconstructed_stream = self.streams.generator("eval", component="selected", purpose="paired_reconstruction")
            code, rows = served.prior.sample(n, generator=generated_stream)
            generated_joint = served.generate(code, generator=generated_stream, output_noise=False, rows=rows)
            encoded = served.encoder(self.words)
            reconstructed_joint = served.generate(encoded, generator=reconstructed_stream, output_noise=False)
            generated = generated_joint[:, :WORD_DIM].reshape(-1, len(WORD_CHARS), WORD_LENGTH)
            reconstructed = reconstructed_joint[:, :WORD_DIM].reshape(-1, len(WORD_CHARS), WORD_LENGTH)
            scored = score_words(generated.cpu().numpy(), reconstructed.cpu().numpy())
            metrics = deepcopy(scored["metrics"]); metrics["reconstruction_exact"] = int(metrics["reconstruction_exact"])
            self.last_views = {key: value.detach().cpu().clone() for key, value in dict(
                target=self.words, generated=generated, reconstruction=reconstructed,
                prior=served.table, generated_raw_code=code, generated_effective_code=generated_joint[:, WORD_DIM:],
                encoded_code=encoded, reconstruction_effective_code=reconstructed_joint[:, WORD_DIM:]).items()}
        after = typed_state_digest(evaluation_state(self.state_dict()))
        allowed = [k for k, v in self.streams.manifest()["bindings"].items() if v["family"] == "eval"]
        audit = self.streams.compare(rng, self.streams.audit(), allowed=allowed)
        global_after = typed_state_digest(word_global_rng())
        pure = before == after and global_before == global_after and audit["unintended_rng_deviations"] == 0
        self.rng_audits.append(audit)
        self.purity.append(dict(completed_steps=self.completed_steps, digest_kind=DIGEST_KIND,
            before_sha256=before, after_sha256=after, global_rng_before_sha256=global_before,
            global_rng_after_sha256=global_after, pure=pure))
        self.policy_observations.append({**observation_receipt(self.task, self.policy), "family": self.declaration.FAMILY})
        if not pure: raise RuntimeError("word observation changed training state/global/named RNG")
        return metrics

    def controls(self):
        row = controls_receipt(self.policy, self.completed_steps)
        row.update(cohort=self.declaration.COHORT, family=self.declaration.FAMILY, independent_atlas_qualification=False,
                   joint_atom_code="same_effective_code", output_noise_coordinates="words168_only",
                   actual_prior_rows=len(self.prior.z), canonical_target_words=len(self.words),
                   resource_adaptation=deepcopy(self.declaration.RESOURCE_ADAPTATION),
                   actual_birth_death=dict(rows=self.policy.birth_death.N, neighbours=self.policy.birth_death.k,
                       isolation=self.policy.birth_death.isolation, reference_half=(len(self.prior.z)+1)//2))
        if self.rate_profile is not None:
            row["word_rate_binding"] = self.declaration.binding_receipt(self.recipe, self.rate_profile)
        return row

    def guards(self):
        def count(optimizer, params):
            values = [int(optimizer.state[p]["step"]) for p in params if "step" in optimizer.state.get(p, {})]
            return min(values) if values else 0
        mechanisms = self.mechanisms.receipt()
        mechanisms["mechanisms"]["direct_particle_gain"].update(applicable_to_direct_output=False,
            host_activation_credit=False, synthetic_probe_scope="independent_scratch_hook_not_word_output_response")
        return dict(all_finite=finite_policy_state(self.state_dict()), optimizer_updates={
            role: count(optimizer, params) for role, (optimizer, params) in self.role_parameters.items()},
            hooks_exercised=self.controls()["implementation_observed"] and not mechanism_blockers(mechanisms),
            mechanism_audit=mechanisms, unintended_rng_deviations=sum(a["unintended_rng_deviations"] for a in self.rng_audits))

    def receipt(self):
        base = self.context.receipt() if self.context is not None else dict(execution_path="public_components",
            recipe=self.recipe.to_dict(), rng=self.streams.manifest(), initialization=deepcopy(self.initialization))
        return {**base, "family": self.declaration.FAMILY, "task_cohort": self.declaration.COHORT,
            "policy_lifecycle": dict(owner="particlegan.UpdatePolicy", completed_steps=self.completed_steps,
                external_max_steps=self.max_steps, controls=self.controls()),
            "actual_resources": deepcopy(self.declaration.HOST_RESOURCES),
            "resource_adaptation": deepcopy(self.declaration.RESOURCE_ADAPTATION),
            "table_ownership": dict(parameter="prior.z", kind="particle_cloud", shape=[11,2],
                canonical_target_words=5, masses="uniform_actual_eleven_rows", sigma=0., standardize=False,
                auxiliary_encoder="original_free_continuous_WordEncoder", encoder_mode="none",
                optimizer_roles=self.policy.roles, joint_callback="joint_generation", row_semantics="independent"),
            "independent_atlas_qualification": False}


def run_word(request, task, output_dir, device="cpu", *, context=None):
    from .artifacts import manifest_artifacts
    from .sampling import executed_receipt
    started = time.monotonic(); fixture = WordJointPolicyFixture(request, task, device=device, context=context)
    binding = fixture.declaration
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    artifacts = output / "word-joint-policy"; artifacts.mkdir(); views = artifacts / "observations"; views.mkdir()
    observations = []; cadence = {math.ceil(i * 20001 / 24) for i in range(1, 25)}
    for step in range(1, 20002):
        fixture.step()
        if step in cadence:
            metrics = fixture.observe(); observations.append(dict(step=step, **metrics))
            np.savez_compressed(views / f"step_{step:06d}.npz", **{k: v.numpy() for k, v in fixture.last_views.items()})
            print(json.dumps(dict(event="observation", task=task["id"], step=step, metrics=metrics)), flush=True)
    state = fixture.state_dict(); target = artifacts / "state.pt"; torch.save(state, target)
    evidence = dict(observations=observations, live=observations[-1], scoring_weights="state_selected", guards=fixture.guards(),
        policy_controls=fixture.controls(), policy_observation=fixture.policy_observations[-1],
        policy_observations=fixture.policy_observations, policy_purity=fixture.purity, rng_audits=fixture.rng_audits,
        artifact_root=str(artifacts.resolve()), artifact_manifest=manifest_artifacts(artifacts),
        checkpoint=dict(path="state.pt", sha256=file_hash(target), state_sha256=typed_state_digest(state), digest_kind=DIGEST_KIND),
        host=dict(family=binding.FAMILY, task_cohort=binding.COHORT,
            canonical_words=list(WORDS), actual_resources=deepcopy(binding.HOST_RESOURCES),
            resource_adaptation=deepcopy(binding.RESOURCE_ADAPTATION),
            objective=task["execution"]["policy_contract"]["objective"],
            reconstruction_training_loss=False, original_capacity_or_qualification_credit=False),
        **executed_receipt(task["evaluation"]["sampling_law"], eval_output_noise=task["evaluation"]["eval_output_noise"]))
    result = dict(task_id=task["id"], evidence=evidence, applied=fixture.receipt(), execution_path="public_components",
        device=str(device), cost=dict(wall_seconds=time.monotonic()-started, completed_steps=fixture.completed_steps,
                                    optimizer_updates=evidence["guards"]["optimizer_updates"]))
    atomic_json(output / "result.json", result); return result
