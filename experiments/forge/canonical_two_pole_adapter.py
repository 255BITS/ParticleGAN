"""Prospective first canonical two-pole owner; importing this file is inert.

Scientific calls stay in the pinned original host. The derived function adds
only ordered public policy hooks and replaces its construction with the explicit
public owner. Structural tests do not import any scientific package.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace

SCHEMA = "pg_canonical_two_pole_policy_binding_v1"
OWNER_SCHEMA = "pg_canonical_two_pole_initial_owner_v1"
STEPS = 80
CLOCKS = [math.ceil(i * STEPS / 24) for i in range(1, 25)]
TASK_PATH = "configs/forge/tasks/two_pole.json"
CONFIG_PATH = "configs/100gaussians/atlas.json"
HOST_PATH = "benchmarks/locked_shared/two_pole.py"
PROTOCOL_PATH = "configs/forge/protocols/screening.json"
TASK_BINDINGS = dict(num_particles=12, z_dim=1, batch_size=12,
                     sigma_rel=0., standardize=False)
SOURCE_PINS = {'benchmarks/legacy/locked_shared.py': {'bytes': 4756, 'sha256': '9bac7def8e5c10d4d4532723ebbf7d7689d5e85c8823dbf07fb7225f91b455c0'}, 'benchmarks/locked_shared/observation.py': {'bytes': 4351, 'sha256': 'bd6f9845b44f1ec2a58d445727990ba5068c7aca3b6f981cf38d738a37c4513b'}, 'benchmarks/locked_shared/two_pole.py': {'bytes': 6753, 'sha256': 'eee207af6d12475f7dbb106087f9d9e269d284274e69e4d199f6d8a38cb54f5e'}, 'benchmarks/transfer_suite/protocol.py': {'bytes': 9964, 'sha256': '99469b022b790a18a74021a6fe49424d95f535afaa220643688a1ddd7a70ab89'}, 'configs/100gaussians/atlas.json': {'bytes': 1798, 'sha256': 'a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4'}, 'configs/forge/tasks/two_pole.json': {'bytes': 3099, 'sha256': '55ac2d3883ba6c173da304fa7f10648a0b559c202fc35b451b1d0c8870f61cf5'}, 'experiments/forge/mechanisms.py': {'bytes': 11661, 'sha256': '6e170f776c2db1f8986836a1dad2ff18c87311103b3ef771d5b1d340955714c3'}, 'experiments/forge/rng.py': {'bytes': 6893, 'sha256': 'ae7b9a8d61da42136d970188a6f168e03e2d7c5b90cf6fdbd93ad929aba2f293'}, 'experiments/forge/sampling.py': {'bytes': 8632, 'sha256': 'b208ae580347d9f70375035ce26d0f7b628a273aa3443037ee8d5f8ee3855e70'}, 'experiments/forge/views.py': {'bytes': 39045, 'sha256': '50d985f0de1e0e2db7f01df66fdc225b1135ee69f6e297d800d1a51a84602565'}, 'particlegan/birth_death.py': {'bytes': 45396, 'sha256': 'b14c50c611a8cf188347e391739fca50a5171200eb6be90e4e3d1509d64e47ed'}, 'particlegan/continuous.py': {'bytes': 60875, 'sha256': '4ceab49c7d51d1769ae91b7f8bc892eaf380ca6fbd77a948a086ade6627e9a41'}, 'particlegan/feature_policy.py': {'bytes': 14650, 'sha256': 'de1a50f70a37d4c9174c8850ce3eac2a2597ed01f19fc4fe1a2ce0ec3176a8b2'}, 'particlegan/k3p.py': {'bytes': 33012, 'sha256': '200ead2c27ab0b2aa6068f1b1b1b5c826b8125aa8602bfec08b31aaa2894401c'}, 'particlegan/policy.py': {'bytes': 81022, 'sha256': '8370e36b5b93afe95ae24d2b385aac8735a587f86ef2b87ed369ecfdfcb6fec5'}, 'particlegan/recipe_schedules.py': {'bytes': 6044, 'sha256': 'd05e459b3e9e5f938b01a854273b74343f2b6d382bec4ae5151dd1f1cd032500'}, 'particlegan/recipes.py': {'bytes': 53612, 'sha256': '1a7f9df746e242a2774068c819d70d50cd72dc4f7f6b22b0bd13e1345dcf8ab2'}, 'configs/forge/protocols/screening.json': {'sha256': '3fefb4d47fd2cd8aa6ed110c0a9f5bffefaae700431d1f57ca7b162c8efbb803', 'bytes': 972}, 'particlegan/ka2.py': {'sha256': '95fdaa58a2bdc229d5d146ef89548bc3fe07582d06181a149ed1bb4f5d632703', 'bytes': 16551}}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _source(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if path.resolve() != root / relative:
        raise ValueError("foreign source alias")
    raw = path.read_bytes()
    if (hashlib.sha256(raw).hexdigest(), len(raw)) != (
            SOURCE_PINS[relative]["sha256"], SOURCE_PINS[relative]["bytes"]):
        raise ValueError("source drift: " + relative)
    return raw


def _declaration(task):
    task = deepcopy(task)
    if "preflight_blockers" in task:
        if task.pop("preflight_blockers") != []:
            raise ValueError("blocked compiled task")
    if "field_ownership" in task and not isinstance(task.pop("field_ownership"), dict):
        raise ValueError("malformed compiler ownership annotation")
    return task


def _recipe_fields(source):
    """Read dataclass field defaults without importing or executing Recipe."""
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.ClassDef) and n.name == "Recipe")
    return {n.target.id: ast.literal_eval(n.value) for n in node.body
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}


def serialized_recipe(fields, source):
    """Apply only the pinned public metadata serializer to an inert field view."""
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "Recipe")
    method = deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "to_dict"))
    scope = dict(asdict=lambda obj: deepcopy(vars(obj)))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 "<pinned-Recipe-metadata-only-serializer>", "exec"), scope)
    return json.loads(canonical(scope["to_dict"](SimpleNamespace(**fields))))


def resolve_binding(root, candidate, task, protocol):
    """Source-only declaration validation; no constructors, RNG or models."""
    task_source = json.loads(_source(root, TASK_PATH))
    reference = json.loads(_source(root, CONFIG_PATH))
    if _declaration(task) != task_source or task_source["id"] != "two_pole":
        raise ValueError("only the exact canonical two_pole declaration is supported")
    if (candidate.get("recipe_preset") != "atlas"
            or candidate.get("recipe_overrides") != reference
            or candidate.get("extensions", {}) != {}
            or candidate.get("initializer", "deterministic_orthogonal") != "deterministic_orthogonal"):
        raise ValueError("full original Atlas reference declaration required; tuning is unsupported")
    base_protocol = deepcopy(protocol)
    if "scientific_repeat" in base_protocol and not isinstance(base_protocol.pop("scientific_repeat"), dict):
        raise ValueError("malformed recognized repeat intent")
    if base_protocol != json.loads(_source(root, PROTOCOL_PATH)):
        raise ValueError("complete ordinary screening protocol, seed0 and named RNG required")
    for relative in SOURCE_PINS:
        _source(root, relative)
    fields = _recipe_fields(_source(root, "particlegan/recipes.py"))
    fields.update(name="atlas", **reference)
    fields.update(TASK_BINDINGS)
    # Recipe.__post_init__ owns this explicit legacy-arm normalization.
    if fields["reg_arm"] is not None:
        fields["critic_formulation"] = "k3p"
    fields = json.loads(canonical(fields))
    if fields["total_steps"] is not None or fields["continuous_policy"] != "dv12":
        raise ValueError("original schedule-free DV12 Recipe must be preserved")
    contract = dict(schema="pg_canonical_two_pole_public_owner_contract_v1",
        files=deepcopy(SOURCE_PINS), task_id="two_pole", external_max_steps=80,
        resources=dict(device="cpu", cpu_threads=1, gpus=0, **{k: TASK_BINDINGS[k]
                       for k in ("num_particles", "z_dim", "batch_size")}),
        observation=dict(sampling_law="learned_particles_and_critic_gradient",
            eval_output_noise="not_applied_to_measurement", scoring_weights="live",
            observations=CLOCKS, final_five=CLOCKS[-5:], owner="live_table_and_live_critic"),
        initializer=dict(critic="stored_HostCritic_weights", particles="zeros12x1",
                         generator="parameter_free_Identity"),
        objective=dict(gan="Recipe.make_loss", particle_l2=.02, pairing="live_all12_rows"),
        effective_task_bindings=deepcopy(TASK_BINDINGS),
        inapplicable=dict(prior_lr_mult="no separate latent prior; direct coordinates use Recipe.lr",
                          prior_betas="direct_response owns step betas",
                          prior_reg="no latent prior; original direct-coordinate L2 .02 remains",
                          a2="no latent_table; direct_response is the direct-coordinate owner",
                          serving_to_metric="averages remain owned; ordinary parameter metrics inspect live owners"),
        schedule="preserve_original_Recipe.total_steps_None_external_limit80",
        lifecycle="particlegan.UpdatePolicy_ordered_public_update",
        scratch_probe="source_only_AMSGrad_max_exp_avg_sq_fixture_correction",
        scientific_credit=False)
    return dict(schema=SCHEMA, recipe=fields,
                recipe_serialized=serialized_recipe(fields, _source(root, "particlegan/recipes.py")),
                source_contract=contract,
                source_contract_sha256=digest(contract), task_sha256=SOURCE_PINS[TASK_PATH]["sha256"],
                config_sha256=SOURCE_PINS[CONFIG_PATH]["sha256"],
                protocol_sha256=SOURCE_PINS[PROTOCOL_PATH]["sha256"],
                adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def derive_host_function(source):
    """Return an explicit source overlay; every original scientific call remains."""
    tree = ast.parse(source)
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    start = next(i for i, n in enumerate(train.body) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "real")
    original_tail = deepcopy(train.body[start:])
    loop = next(n for n in original_tail if isinstance(n, ast.For))
    body = []
    for node in loop.body:
        text = ast.unparse(node)
        if text.startswith("(d_loss + regularizer("):
            body.extend(ast.parse("owner.policy.before_critic_backward()").body)
        if text == "g_loss.backward()":
            body.extend(ast.parse("owner.policy.before_generator_backward()").body)
        body.append(node)
        if text == "opt_d.step()":
            body.extend(ast.parse("owner.after_critic_step()").body)
        elif text.startswith("g_loss = gan.g_loss("):
            body.extend(ast.parse("loss_gan = g_loss").body)
        elif text == "g_loss.backward()":
            body.extend(ast.parse("owner.after_generator_backward(loss_gan, d_loss)").body)
        elif text == "opt_p.step()":
            body.extend(ast.parse("owner.after_generator_step()\nowner.finish_step()").body)
    loop.body = body
    prefix = ast.parse("""pairing = 'live'
noise_policy = owner.noise
base_critic = owner.critic
critic = owner.critic
particles = owner.table
opt_d = owner.opt_d
opt_p = owner.opt_g
gan = owner.loss
regularizer = owner.penalty
particle_l2 = owner.particle_l2
""").body
    prefix.append(ast.parse("owner.bind_real(real)").body[0])
    # bind_real must follow the source's original real_batch construction.
    head = prefix[:-1] + [original_tail[0], prefix[-1]]
    derived = ast.FunctionDef(name="_owned_train",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="owner")],
            kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=head + original_tail[1:], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[derived], type_ignores=[]))
    result = ast.unparse(module) + "\n"
    required = ("owner.policy.before_critic_backward()", "owner.after_critic_step()",
                "owner.policy.before_generator_backward()", "owner.after_generator_backward(loss_gan, d_loss)",
                "owner.after_generator_step()", "owner.finish_step()")
    if any(result.count(text) != 1 for text in required):
        raise ValueError("source lifecycle insertion boundary changed")
    return result


def corrected_mechanism_source(source):
    marker = '            parameter.grad = torch.full_like(parameter, recipe.d_guard_ratio * 2)'
    replacement = ('            if optimizer.param_groups[0]["amsgrad"]:\n'
                   '                optimizer.state[parameter]["max_exp_avg_sq"] = '\
                   'optimizer.state[parameter]["exp_avg_sq"].clone()\n' + marker)
    if source.count(marker) != 1:
        raise ValueError("AMSGrad scratch-state source boundary changed")
    return source.replace(marker, replacement)


def typed_digest(value, tensor_bytes):
    """Exact typed state hash; raw diagnostic sentinels remain in the digest."""
    h = hashlib.sha256()
    def add(tag, raw):
        h.update(tag + len(raw).to_bytes(8, "big") + raw)
    def visit(v):
        tensor = tensor_bytes(v)
        if tensor is not None:
            description, raw = tensor
            add(b"tensor", canonical(description).encode()); add(b"bytes", raw)
        elif type(v) in (dict, OrderedDict):
            add(b"ordered" if type(v) is OrderedDict else b"mapping", str(len(v)).encode())
            for k, item in v.items(): visit(k); visit(item)
            if type(v) is OrderedDict: visit(getattr(v, "_metadata", None))
        elif type(v) in (list, tuple):
            add(b"list" if type(v) is list else b"tuple", str(len(v)).encode())
            for item in v: visit(item)
        elif v is None: add(b"none", b"")
        elif type(v) is bool: add(b"bool", b"1" if v else b"0")
        elif type(v) is int: add(b"int", str(v).encode())
        elif type(v) is float: add(b"float", struct.pack("!d", v))
        elif type(v) is str: add(b"str", v.encode())
        else: raise TypeError("unknown policy state leaf: " + type(v).__name__)
    visit(value)
    return h.hexdigest()


class TwoPoleOwner:
    """Actual public owner. Construction occurs only in construct_owner."""
    def bind_real(self, real): self.real = real
    def after_critic_step(self):
        self.policy.after_critic_step(); self.calls["after_critic_step"] += 1
    def after_generator_backward(self, loss_gan, loss_critic):
        self.policy.after_generator_backward(loss_gan=loss_gan, loss_critic=loss_critic)
        self.calls["after_generator_backward"] += 1
    def after_generator_step(self):
        self.policy.after_generator_step(); self.calls["after_generator_step"] += 1
    def finish_step(self):
        self.policy.finish_step(); self.calls["finish_step"] += 1
    def schedule_optimizer(self, optimizer, step):
        if optimizer not in (self.opt_g, self.opt_d) or self.policy.completed_steps != step:
            raise ValueError("unowned optimizer or wrong policy clock")
        # begin_step already applies public schedules/stationarity exactly once.
    def checkpoint(self, step, measure):
        if step not in CLOCKS: return
        if self.policy.completed_steps != step or self.policy._phase != "ready":
            raise ValueError("observation outside completed public update")
        before = self.state_sha256()
        named = self.streams.audit()
        global_before = self.global_sha256()
        self._reading_step = step
        try: values = measure()
        finally: self._reading_step = None
        after = self.state_sha256()
        global_after = self.global_sha256()
        rng = self.streams.compare(named, self.streams.audit())
        pure = before == after and global_before == global_after and not rng["unintended_rng_deviations"]
        if not pure: raise ValueError("ordinary live measurement changed owned state or RNG")
        self.observations.append({**values, "step": step})
        self.purity.append(dict(step=step, before_sha256=before, after_sha256=after,
            global_before_sha256=global_before, global_after_sha256=global_after,
            named_rng=rng, pure=True, owner="live_table_and_live_critic", eval_draws=0))
    def state_sha256(self):
        return typed_digest(self.complete_state(), self.tensor_bytes)
    def complete_state(self):
        return dict(policy=self.policy.state_dict(), named_streams=self.streams.state_dict(),
            gradients={"table": self.table.grad, "noise": self.policy.log_output_sigma.grad,
                       "critic": [p.grad for p in self.critic.parameters()]},
            modes={"generator": self.generator.training, "critic": self.critic.training,
                   "ema_generator": self.policy.ema_G.training},
            completed_steps=self.policy.completed_steps, observations=deepcopy(self.observations),
            calls=deepcopy(self.calls), binding=deepcopy(self.binding))
    def capture_gradient(self, gradient):
        if self._reading_step is not None:
            self.retained.append(dict(step=self._reading_step,
                particles=self.table.detach().cpu().clone(), gradient=gradient.detach().cpu().clone()))


class _PolicyNoise:
    def __init__(self, owner): self.owner = owner
    def set_step(self, step):
        owner = self.owner
        if step != owner.policy.completed_steps: raise ValueError("wrong external update clock")
        owner.policy.begin_step(owner.real, execution_limit=80)
        owner.calls["begin_step"] += 1
    def output(self, coordinates, *, generator_step):
        owner = self.owner
        owner.policy.observe_support(coordinates)
        if generator_step:
            return owner.policy.generate(coordinates, rows=owner.row_indices)
        with owner.torch.no_grad():
            fake = owner.policy.generate(coordinates, rows=owner.row_indices)
        owner.policy.observe_critic_pair(owner.real, fake)
        return fake


def _loaded_source(root, module, relative):
    if Path(inspect.getfile(module)).resolve() != (Path(root) / relative).resolve():
        raise ValueError("foreign imported scientific source: " + relative)
    _source(root, relative)


def construct_owner(root, binding, *, source_guard):
    """ROOT-only late constructor; import/source guards precede any model."""
    source_guard()
    import torch
    import numpy as np
    import random
    from torch import nn
    from particlegan import Recipe
    from particlegan.policy import UpdatePolicy
    from benchmarks.locked_shared import two_pole
    from benchmarks.legacy.locked_shared import LOCKED_SHARED
    from experiments.forge.rng import NamedStreams
    source_guard()
    for module, relative in ((two_pole, HOST_PATH), (UpdatePolicy, "particlegan/policy.py"),
                             (Recipe, "particlegan/recipes.py"), (NamedStreams, "experiments/forge/rng.py")):
        _loaded_source(root, module, relative)
    if binding.get("schema") != SCHEMA or digest(binding["source_contract"]) != binding["source_contract_sha256"]:
        raise ValueError("malformed source-bound owner contract")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != binding["adapter_sha256"]:
        raise ValueError("foreign adapter source")
    if torch.get_default_dtype() != torch.float32 or torch.get_num_threads() != 1:
        raise ValueError("canonical host requires original float32 tensors and CPU1")
    reference = json.loads(_source(root, CONFIG_PATH))
    expected = resolve_binding(root, dict(recipe_preset="atlas", recipe_overrides=reference),
        json.loads(_source(root, TASK_PATH)), json.loads(_source(root, PROTOCOL_PATH)))
    if binding != expected:
        raise ValueError("forged effective Recipe or owner contract before model construction")
    recipe = Recipe(**binding["recipe"])
    if (canonical(asdict(recipe)) != canonical(binding["recipe"])
            or canonical(recipe.to_dict()) != canonical(binding["recipe_serialized"])):
        raise ValueError("actual complete public Recipe differs before model construction")
    owner = TwoPoleOwner()
    owner.torch, owner.recipe, owner.binding = torch, recipe, deepcopy(binding)
    owner.streams = NamedStreams(0, device="cpu")
    owner.restored, owner.construction_id = False, object()
    with owner.streams.fork("init", component="discriminator", purpose="construction", device="cpu"):
        owner.critic = two_pole.HostCritic()
    owner.table = nn.Parameter(torch.zeros(12, 1))
    owner.generator = nn.Identity()
    owner.opt_g = recipe.make_generator_optimizer(
        [{"params": [owner.table], "forge_role": "prior"}], direct_particles=[owner.table])
    owner.opt_d = recipe.make_critic_optimizer(owner.critic, ema_critic=deepcopy(owner.critic))
    penalty = recipe.make_critic_penalty(owner.opt_d, collect_stats=True)
    streams = {"latent_generator": owner.streams.generator("prior", component="latent", purpose="indices"),
        "penalty_generator": owner.streams.generator("noise", component="penalty", purpose="training"),
        "noise_generator": owner.streams.generator("noise", component="generator", purpose="output"),
        "eval_generator": owner.streams.generator("eval", component="sampler", purpose="samples")}
    owner.policy = UpdatePolicy(recipe, owner.generator, owner.critic, table=owner.table,
        generator_optimizer=owner.opt_g, critic_optimizer=owner.opt_d, table_optimizer=owner.opt_g,
        roles=[["table"], ["critic"]], streams=streams, seed=0, penalty=penalty)
    owner.loss, owner.particle_l2 = recipe.make_loss(), LOCKED_SHARED.particle_l2
    owner.row_indices = torch.arange(12, dtype=torch.long)
    owner.noise = _PolicyNoise(owner)
    owner.calls = {key: 0 for key in ("begin_step", "after_critic_step", "after_generator_backward",
                                    "after_generator_step", "finish_step")}
    owner.observations, owner.purity, owner.retained, owner._reading_step = [], [], [], None
    def tensor_bytes(value):
        if not isinstance(value, torch.Tensor): return None
        array = value.detach().cpu().contiguous()
        return dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
                    requires_grad=value.requires_grad), array.numpy().tobytes()
    owner.tensor_bytes = tensor_bytes
    owner.global_sha256 = lambda: typed_digest(
        (torch.get_rng_state(), random.getstate(), np.random.get_state()),
        lambda v: (dict(shape=list(v.shape), dtype=str(v.dtype)), v.tobytes())
        if isinstance(v, np.ndarray) else tensor_bytes(v))
    # Only the known delayed-hook scratch fixture changes; public training state does not.
    namespace = dict(__name__="_canonical_two_pole_scratch_probe")
    scratch_source = corrected_mechanism_source(_source(root, "experiments/forge/mechanisms.py").decode())
    exec(compile(scratch_source, "<pinned-AMSGrad-scratch-probe>", "exec"), namespace)
    owner.audit = namespace["MechanismAudit"](recipe, owner.opt_d, [owner.opt_g])
    owner.penalty = lambda critic, real, fake, step: _penalty(owner, penalty, critic, real, fake)
    source_guard()
    return owner


def _penalty(owner, penalty, critic, real, fake):
    value = penalty(critic, real, fake)
    owner.audit.observe_penalty(penalty.last_stats)
    return value


def owner_initial_receipt(owner):
    """Physical witness; manual receipt objects and restored owners are refused."""
    import torch
    from torch import nn
    from particlegan.policy import UpdatePolicy
    from particlegan.k3p import K3PGeneratorAdam
    from particlegan.ka2 import KA2CriticAdam
    from benchmarks.locked_shared import two_pole
    if (type(owner) is not TwoPoleOwner or type(owner.policy) is not UpdatePolicy
            or type(owner.critic) is not two_pole.HostCritic or type(owner.generator) is not nn.Identity
            or type(owner.opt_g) is not K3PGeneratorAdam or type(owner.opt_d) is not KA2CriticAdam):
        raise ValueError("actual declared public owner types required")
    policy = owner.policy
    params = [p for opt in (owner.opt_g, owner.opt_d) for group in opt.param_groups for p in group["params"]]
    stored = ((owner.critic.fc1.weight, two_pole._HOST_W1), (owner.critic.fc1.bias, two_pole._HOST_B1),
              (owner.critic.fc2.weight, two_pole._HOST_W2), (owner.critic.fc2.bias, two_pole._HOST_B2))
    receipt = dict(schema=OWNER_SCHEMA, completed_steps=policy.completed_steps, phase=policy._phase,
        restored=owner.restored, seed=owner.streams.seed, rng_version=owner.streams.version,
        table=dict(shape=list(owner.table.shape), all_zero=bool((owner.table == 0).all()),
                   requires_grad=owner.table.requires_grad),
        critic=dict(type="benchmarks.locked_shared.two_pole.HostCritic",
                    stored_weights_exact=all(torch.equal(p, p.new_tensor(values).reshape_as(p)) for p, values in stored),
                    state_sha256=typed_digest(owner.critic.state_dict(), owner.tensor_bytes)),
        generator=dict(type="torch.nn.Identity", trainable_parameters=sum(p.numel() for p in owner.generator.parameters())),
        optimizer_updates=dict(prior=0, discriminator=0, noise=0),
        optimizer_state_entries=dict(prior=len(owner.opt_g.state), discriminator=len(owner.opt_d.state)),
        object_bindings=dict(policy_table=policy.table is owner.table, policy_critic=policy.D is owner.critic,
            policy_generator=policy.G is owner.generator, table_optimizer=policy.table_optimizer is owner.opt_g,
            unique_parameter_ownership=len(params) == len({id(p) for p in params})),
        resolved_recipe_sha256=digest(asdict(policy.recipe)),
        source_contract_sha256=owner.binding["source_contract_sha256"])
    if (receipt["completed_steps"] != 0 or receipt["phase"] != "ready" or receipt["restored"]
            or not receipt["table"]["all_zero"] or receipt["table"]["shape"] != [12, 1]
            or not receipt["critic"]["stored_weights_exact"] or any(receipt["optimizer_state_entries"].values())
            or not all(receipt["object_bindings"].values()) or any(owner.calls.values())
            or owner.opt_d.record.observed_steps != 0 or owner.policy.controller.updates != 0):
        raise ValueError("fresh zero-update original initialization required")
    return receipt


def run_first_case(root, request, task=None, *, source_guard, fresh_repeat_guard):
    """ROOT-only child: exact original loop, ordinary measurement, no serving draws."""
    task = request["task"] if task is None else task
    binding = resolve_binding(root, request["candidate"], task, request["protocol"])
    if binding != request["binding"]:
        raise ValueError("request effective Recipe/source binding drift")
    owner = fresh_repeat_guard.construct(lambda: construct_owner(root, binding, source_guard=source_guard), owner_initial_receipt)
    import torch
    from benchmarks.locked_shared import two_pole
    source_guard()
    namespace = dict(vars(two_pole))
    namespace.update(checkpoint=owner.checkpoint, schedule_optimizer=owner.schedule_optimizer)
    # Capture the already-computed gradient. This adds no critic/gradient call.
    grad_source = inspect.getsource(two_pole._grad_median)
    marker = '    return float(grad.flatten().abs().median())'
    if grad_source.count(marker) != 1: raise ValueError("original metric source changed")
    grad_source = grad_source.replace(marker, '    _capture_gradient(grad)\n' + marker)
    namespace["_capture_gradient"] = owner.capture_gradient
    exec(compile(grad_source, "<pinned-two-pole-gradient-capture>", "exec"), namespace)
    overlay = derive_host_function(_source(root, HOST_PATH).decode())
    exec(compile(overlay, "<pinned-two-pole-public-policy-overlay>", "exec"), namespace)
    raw = namespace["_owned_train"](owner)
    if owner.policy.completed_steps != 80 or [p["step"] for p in owner.observations] != CLOCKS:
        raise ValueError("incomplete original external horizon or ordinary cadence")
    mechanism_audit = owner.audit.receipt()
    def updates(parameter, optimizer):
        return int(optimizer.state.get(parameter, {}).get("step", 0))
    d_counts = [updates(p, owner.opt_d) for p in owner.critic.parameters()]
    counts = dict(prior=updates(owner.table, owner.opt_g), discriminator=min(d_counts),
                  noise=updates(owner.policy.log_output_sigma, owner.opt_g))
    if any(n != 80 for n in counts.values()) or max(d_counts) != 80 or any(n != 80 for n in owner.calls.values()):
        raise ValueError("actual optimizer/lifecycle clocks do not complete the original80")
    actual_owners = {"dv12": owner.policy.controller, "stationarity": owner.policy.lr_settle,
        "row_evidence": owner.policy.row_evidence, "birth_death": owner.policy.birth_death,
        "averaging": owner.policy.averaged_table, "learned_output_noise": owner.policy.log_output_sigma,
        "automatic_features": owner.policy._feature_selection, "settled_reopen": owner.policy.reopen_guard}
    if any(value is None for value in actual_owners.values()):
        raise ValueError("a requested original Atlas policy owner is missing")
    def finite(v):
        if isinstance(v, torch.Tensor): return bool(torch.isfinite(v).all())
        if isinstance(v, dict): return all(finite(x) for x in v.values())
        if isinstance(v, (list, tuple)): return all(finite(x) for x in v)
        return math.isfinite(v) if type(v) in (int, float) else True
    all_finite = all(finite(p) and finite(p.grad) for p in (owner.table, owner.policy.log_output_sigma, *owner.critic.parameters()))
    all_finite = all_finite and finite([owner.opt_g.state_dict(), owner.opt_d.state_dict()])
    from experiments.forge.mechanisms import mechanism_blockers
    source_guard()
    result = dict(task_id="two_pole", execution_path="public_components", device="cpu", raw=raw,
        evidence=dict(observations=owner.observations, live={k: raw[k] for k in ("mean_abs", "grad_med")},
            scoring_weights="live", sampling_contract_version=1,
            sampling_law="learned_particles_and_critic_gradient", eval_output_noise="not_applied_to_measurement",
            guards=dict(all_finite=all_finite, optimizer_updates=counts,
                hooks_exercised=not mechanism_blockers(mechanism_audit), mechanism_audit=mechanism_audit,
                unintended_rng_deviations=0), measurement_purity=owner.purity),
        applied=dict(recipe=asdict(owner.recipe), recipe_serialized=owner.recipe.to_dict(),
            source_contract=deepcopy(binding["source_contract"]),
            source_contract_sha256=binding["source_contract_sha256"], policy_owner="particlegan.UpdatePolicy",
            actual_policy_owners={key: type(value).__module__ + "." + type(value).__qualname__
                                  for key, value in actual_owners.items()},
            lifecycle_calls=owner.calls, optimizer_updates=counts, roles=owner.policy.roles,
            ordinary_observation_owner="live_table_and_live_critic", evaluation_sampler_calls=0,
            direct_row_base_lr=owner.policy.initial_lrs[0][0],
            a2_inapplicable=True, prior_lr_mult_consumed=False, prior_betas_consumed=False,
            streams=owner.streams.manifest()), retained_goal_states=owner.retained,
        complete_state=owner.complete_state())
    source_guard()
    return result
