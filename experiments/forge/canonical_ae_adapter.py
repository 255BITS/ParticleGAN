"""Prospective caller-owned AE host; module import performs no science.

The full Atlas Recipe controls the independently sampled decoder/prior game.
The original free encoder and deterministic particle_ae reconstruction loss
remain a separately owned caller objective. Ordinary metrics read live owners
with the original scheduled output noise, rather than a serving-law sample.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace

SCHEMA = "pg_canonical_ae_policy_binding_v1"
OWNER_SCHEMA = "pg_canonical_ae_initial_owner_v1"
TASK_ID = "ae_gan_hold"
TASK_PATH = "configs/forge/tasks/ae_gan_hold.json"
CONFIG_PATH = "configs/100gaussians/atlas.json"
PROTOCOL_PATH = "configs/forge/protocols/screening.json"
HOST_PATH = "benchmarks/locked_shared/hosts/ae_gan_hold.py"
STEPS = 250
CLOCKS = [math.ceil(i * STEPS / 24) for i in range(1, 25)]
TASK_BINDINGS = dict(num_particles=12, z_dim=2, batch_size=64)
ACTUAL_PRIOR = dict(kind="mog", sigma=.025, standardize=False, learnable=True)
PRIOR_FACTORY_OVERRIDES = dict(prior_kind="mog", sigma=.025, standardize=False)
# Filled from the authorized copied SOURCE tree; no generated evidence input.
SOURCE_PINS = {'benchmarks/__init__.py': {'sha256': '53ba1483505036be295d73f5f3308bffb951e9414bf2b1139d8756e260e3b830', 'bytes': 71}, 'benchmarks/gan_v3.py': {'sha256': '00ca856749c82ce3d4e554ced924dda4399311d2e31ca8dbea8b8d5829141538', 'bytes': 2573}, 'benchmarks/legacy/__init__.py': {'sha256': '83de2dd27bb1fec5ba74b066b49274ba161e4e480037129e3c99e13b04767f04', 'bytes': 393}, 'benchmarks/legacy/grad_regularizers.py': {'sha256': 'cfe252f52a6d11274e4be78bd13934fc3d919abe7d0ba349e5bd0199a45dcb5d', 'bytes': 27997}, 'benchmarks/legacy/recipe.py': {'sha256': 'b31cdb23913b3c71432ecbdee40a0c0293c067bbcd1f2e5cdfb192426a43c543', 'bytes': 9954}, 'benchmarks/locked_shared/__init__.py': {'sha256': 'a33bb985c9801d5fe3ad19666a43f8448a7aaebe2ff86555bd05d7a20e76fc2f', 'bytes': 79}, 'benchmarks/locked_shared/baseline.py': {'sha256': '8f1f58fc9cb90159c555788e593769d50070deaa078d0637bf881cf074f6680b', 'bytes': 30125}, 'benchmarks/locked_shared/hosts/__init__.py': {'sha256': '3f0d91417b7c3b167e7b9a13b2053eca53394ca454ea62d27f1c26c723a096fb', 'bytes': 78}, 'benchmarks/locked_shared/hosts/ae_gan_hold.py': {'sha256': 'c25ea8f8b998a137719df7fc18e5b047b456cdb5a65969430a652db0524509a9', 'bytes': 8676}, 'benchmarks/locked_shared/observation.py': {'sha256': 'bd6f9845b44f1ec2a58d445727990ba5068c7aca3b6f981cf38d738a37c4513b', 'bytes': 4351}, 'benchmarks/toy100/device.py': {'sha256': '7fa0f53db44e8824d3497b471e9fe8db56cbf8b4e02ddf997f65beed63f81911', 'bytes': 10288}, 'benchmarks/transfer_suite/protocol.py': {'sha256': '99469b022b790a18a74021a6fe49424d95f535afaa220643688a1ddd7a70ab89', 'bytes': 9964}, 'configs/100gaussians/atlas.json': {'sha256': 'a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4', 'bytes': 1798}, 'configs/forge/protocols/screening.json': {'sha256': '3fefb4d47fd2cd8aa6ed110c0a9f5bffefaae700431d1f57ca7b162c8efbb803', 'bytes': 972}, 'configs/forge/tasks/ae_gan_hold.json': {'sha256': '53a400c3f2b27ef347076f3cc603345e1442d2d8f97f8052f0b9496ba35bae79', 'bytes': 2897}, 'experiments/forge/mechanisms.py': {'sha256': '9e4e1fe1f8cb9a24148cab88b7bf2f980e805a2e488c16a982768519bbe1b23f', 'bytes': 11826}, 'experiments/forge/policy_adapters.py': {'sha256': '714046ea1655f1946e09dc2aeed27c43cf93f2841237c2be47f0f0049a5df9fa', 'bytes': 17992}, 'experiments/forge/rng.py': {'sha256': 'ae7b9a8d61da42136d970188a6f168e03e2d7c5b90cf6fdbd93ad929aba2f293', 'bytes': 6893}, 'particlegan/__init__.py': {'sha256': 'b56efa562acc551d639548c299002e2cd76338b3fd7062699780add51f4f5e8c', 'bytes': 1811}, 'particlegan/_qr.py': {'sha256': 'b59cd62cc27ed557b41b2d17ed86ed578c762a3a8a392d63731a5a3b2b018916', 'bytes': 1951}, 'particlegan/anchor_birth.py': {'sha256': 'aa50983d8b61304acf4e1138f16fd4784798983df91e931863bcacd3b511f76f', 'bytes': 6401}, 'particlegan/autoencoder.py': {'sha256': 'f3fbb9184e9f44112e15eccf0c0246dfba5ed490d3c7b81b0f837a5335916402', 'bytes': 6194}, 'particlegan/birth_death.py': {'sha256': 'b14c50c611a8cf188347e391739fca50a5171200eb6be90e4e3d1509d64e47ed', 'bytes': 45396}, 'particlegan/birth_phase.py': {'sha256': '5ff929e5e9858aa24c4cc2f8999b5d0603e25e42ca3844d1da4cc28b1981d3d7', 'bytes': 21144}, 'particlegan/capabilities.py': {'sha256': '39df9c67dbedf612740057e5fa743bc724678a1bc323d1f7cf550b37c0c01177', 'bytes': 2285}, 'particlegan/conditioning.py': {'sha256': '8734fe338868ace4b5851a8fdb1f071e46d47396f66a3765e6b1ef37f0481486', 'bytes': 4878}, 'particlegan/continuous.py': {'sha256': '4ceab49c7d51d1769ae91b7f8bc892eaf380ca6fbd77a948a086ade6627e9a41', 'bytes': 60875}, 'particlegan/diffusion.py': {'sha256': '19c3aa8772703891551b9122fde8e74c07996f779892b7cb6be09553691efa61', 'bytes': 5100}, 'particlegan/discriminators.py': {'sha256': '0e2efb125ffb314577612ab7a2eba66b0a1a2d28ad25f403f42c18b6f6ee333f', 'bytes': 5833}, 'particlegan/feature_cells.py': {'sha256': 'e0ea05f61abd7845c7437eed9a79c3a2511d5551ff04f4e619b6b9c8992e221a', 'bytes': 125325}, 'particlegan/feature_policy.py': {'sha256': 'de1a50f70a37d4c9174c8850ce3eac2a2597ed01f19fc4fe1a2ce0ec3176a8b2', 'bytes': 14650}, 'particlegan/feature_reference.py': {'sha256': '58e076e9e04600be25e28afafc537a340de727371f4acbbfb2b7dd8bebac18b1', 'bytes': 32368}, 'particlegan/gan_loss.py': {'sha256': '1c1019dfe71c583e32a05df0d2f794f9fff6d9ee1ea0332f57f6379ae70cf6b7', 'bytes': 1016}, 'particlegan/grad_regularizers.py': {'sha256': 'a540d05a4a992540b6a5f3b5ff7ccc11ebaf18592135b87c95638adbe68aa747', 'bytes': 14853}, 'particlegan/init.py': {'sha256': 'e15d4de01eb49f7097bc249abacab907ff067385412346f54ab0d510cb8649d0', 'bytes': 24559}, 'particlegan/k3p.py': {'sha256': '200ead2c27ab0b2aa6068f1b1b1b5c826b8125aa8602bfec08b31aaa2894401c', 'bytes': 33012}, 'particlegan/ka2.py': {'sha256': '95fdaa58a2bdc229d5d146ef89548bc3fe07582d06181a149ed1bb4f5d632703', 'bytes': 16551}, 'particlegan/mean_transport.py': {'sha256': '57a68e0a65c5424a4460f65002f3862a69e90af330a4d47c700dc773683a2c06', 'bytes': 39633}, 'particlegan/output_moments.py': {'sha256': 'dcf3e27228d2c3c9738e473c5b22ee9e6d5047ddcd60c538b66526adb53ef188', 'bytes': 8387}, 'particlegan/particle_prior.py': {'sha256': '0220878bebea227da63abbf9f5b1ebdabdc6aea54fd2332fd2cab02ed734238f', 'bytes': 17899}, 'particlegan/policy.py': {'sha256': '8370e36b5b93afe95ae24d2b385aac8735a587f86ef2b87ed369ecfdfcb6fec5', 'bytes': 81022}, 'particlegan/population_continuity.py': {'sha256': '378cd75cca45bc8da4da33e686a12cf99df853042687978314e37c532bc17fe6', 'bytes': 20329}, 'particlegan/recipe_schedules.py': {'sha256': 'd05e459b3e9e5f938b01a854273b74343f2b6d382bec4ae5151dd1f1cd032500', 'bytes': 6044}, 'particlegan/recipes.py': {'sha256': '1a7f9df746e242a2774068c819d70d50cd72dc4f7f6b22b0bd13e1345dcf8ab2', 'bytes': 53612}, 'particlegan/routing.py': {'sha256': '63249d808fccb3d4e113eb1d638a245ee665c3cade20a1542bb706f316b4abcc', 'bytes': 64904}, 'particlegan/row_evidence.py': {'sha256': '78be40141ef3946860458e9842159be9321f05c217a4800dd4134113b6eba93e', 'bytes': 9397}, 'particlegan/training.py': {'sha256': '7dadc219135bc3e6bf657bfe9bc3fa81dd14d4b2a31c70d5ff61e2941db35c94', 'bytes': 39880}, 'particlegan/vicreg_loss.py': {'sha256': 'ab1c4dc266dec2c35337f449917240eb38afede7590d2154b63f6f471dc45a36', 'bytes': 2297}}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _source(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if path.resolve() != path or relative not in SOURCE_PINS:
        raise ValueError("foreign or undeclared source alias")
    raw = path.read_bytes()
    if (hashlib.sha256(raw).hexdigest(), len(raw)) != (
            SOURCE_PINS[relative]["sha256"], SOURCE_PINS[relative]["bytes"]):
        raise ValueError("source drift: " + relative)
    return raw


def _loaded_source(root, value, relative):
    path = Path(inspect.getfile(value)).resolve()
    if path != Path(root).resolve() / relative or path.read_bytes() != _source(root, relative):
        raise ValueError("foreign imported source: " + relative)


def _declaration(task):
    task = deepcopy(task)
    if "preflight_blockers" in task and task.pop("preflight_blockers") != []:
        raise ValueError("compiled task still has preflight blockers")
    if "field_ownership" in task and not isinstance(task.pop("field_ownership"), dict):
        raise ValueError("malformed compiler ownership annotation")
    return task


def _recipe_fields(source):
    cls = next(n for n in ast.parse(source).body
               if isinstance(n, ast.ClassDef) and n.name == "Recipe")
    return {n.target.id: ast.literal_eval(n.value) for n in cls.body
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}


def serialized_recipe(fields, source):
    """Execute only the pinned metadata serializer on an inert field view."""
    cls = next(n for n in ast.parse(source).body
               if isinstance(n, ast.ClassDef) and n.name == "Recipe")
    method = deepcopy(next(n for n in cls.body
                           if isinstance(n, ast.FunctionDef) and n.name == "to_dict"))
    scope = dict(asdict=lambda value: deepcopy(vars(value)))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 "<pinned-AE-Recipe-metadata-serializer>", "exec"), scope)
    return json.loads(canonical(scope["to_dict"](SimpleNamespace(**fields))))


def resolve_binding(root, candidate, task, protocol):
    """Pure declaration checks. Construction belongs to an admitted child."""
    reference = json.loads(_source(root, CONFIG_PATH))
    declared_task = json.loads(_source(root, TASK_PATH))
    if canonical(_declaration(task)) != canonical(declared_task) or declared_task["id"] != TASK_ID:
        raise ValueError("only the complete canonical ae_gan_hold task is supported")
    if (candidate.get("recipe_preset") != "atlas"
            or canonical(candidate.get("recipe_overrides")) != canonical(reference)
            or candidate.get("extensions", {}) != {}
            or candidate.get("initializer", "deterministic_orthogonal") != "deterministic_orthogonal"):
        raise ValueError("exact full original Atlas declaration required")
    base_protocol = deepcopy(protocol)
    if "scientific_repeat" in base_protocol and not isinstance(base_protocol.pop("scientific_repeat"), dict):
        raise ValueError("malformed repeat declaration")
    if canonical(base_protocol) != canonical(json.loads(_source(root, PROTOCOL_PATH))):
        raise ValueError("complete ordinary seed0/named-stream screening protocol required")
    for relative in SOURCE_PINS:
        _source(root, relative)
    fields = _recipe_fields(_source(root, "particlegan/recipes.py"))
    fields.update(name="atlas", **reference)
    fields.update(TASK_BINDINGS)
    fields = json.loads(canonical(fields))
    if (fields["total_steps"] is not None or fields["encoder_mode"] != "none"
            or fields["row_policy"] != "independent" or fields["continuous_policy"] != "dv12"
            or not fields["particle_birth_death"] or not fields["row_evidence_gate"]):
        raise ValueError("full original independent policy must remain unchanged")
    contract = dict(schema="pg_canonical_ae_public_owner_contract_v1", files=deepcopy(SOURCE_PINS),
        task_id=TASK_ID, external_max_steps=STEPS,
        resources=dict(device="cpu", gpus=0, gpu_memory_mb=0, cpu_threads=1, timeout_seconds=300,
                       **TASK_BINDINGS),
        actual_prior=deepcopy(ACTUAL_PRIOR),
        prior_factory_overrides=deepcopy(PRIOR_FACTORY_OVERRIDES),
        factory_defaults_are_not_actual_prior=True,
        ownership=dict(generator="host.MLP(2,2,32)", encoder="host.MLP(2,4,32)",
                       discriminator="host.MLP(2,1,32)", prior="MoGParticlePrior.z12x2",
                       homogeneous_roles=[["generator", "encoder", "table", "noise"], ["critic"]]),
        initializer="deterministic_orthogonal_whole_owned_modules_and_R2_prior_named_init_streams",
        objective=dict(encoder="free_query_and_offset", encoding="public particle_ae",
                       reconstruction_weight=1., adversarial_weight=1., cover_weight=1.5,
                       particle_l2=.02, feature_matching_weight=0.,
                       gan="Recipe.make_loss", penalty="Recipe.make_critic_penalty",
                       policy_generation="UpdatePolicy.generate_DV12_and_learned_output_kernel",
                       joint_inverse_or_routed_loss_added=False),
        observation=dict(sampling_law=declared_task["evaluation"]["sampling_law"],
                         eval_output_noise="public_recipe_schedule", scoring_weights="live",
                         sample_count=1024, observations=CLOCKS, final_five=CLOCKS[-5:],
                         original_evaluate_source=HOST_PATH + ":evaluate",
                         selected_or_averaged_sampling=False, evaluation_DV12=False,
                         all_named_and_global_rng_preserved=True),
        applicability=dict(independent_controls="uniform sampled unconditional decoder/prior branch",
                           auxiliary_encoder="caller-owned reconstruction objective; policy encoder_mode remains none",
                           feature_backend="actual encoder callback selects reference knn; feature-cell controls inactive",
                           a2="current source nonstandardized MoG row-local capability, not historical PR223 factory equality",
                           direct_particle_response="no direct-coordinate parameter group"),
        lifecycle="one ordered public UpdatePolicy update per original host step",
        schedule="Recipe.total_steps_None_external_limit250",
        source_overlay="original host loss and data formulas plus explicit policy hooks and named RNG",
        excluded_console_measurements="unscored log-only extra measures removed; terminal reuses recorded step250",
        original_scores_transferred=False, qualification=False, default_adoption=False, speed_ranking=False)
    return dict(schema=SCHEMA, recipe=fields,
        recipe_serialized=serialized_recipe(fields, _source(root, "particlegan/recipes.py")),
        actual_prior=deepcopy(ACTUAL_PRIOR), prior_factory_overrides=deepcopy(PRIOR_FACTORY_OVERRIDES),
        source_contract=contract, source_contract_sha256=digest(contract),
        task_sha256=SOURCE_PINS[TASK_PATH]["sha256"], config_sha256=SOURCE_PINS[CONFIG_PATH]["sha256"],
        protocol_sha256=SOURCE_PINS[PROTOCOL_PATH]["sha256"],
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def derive_host_train(source):
    """Keep the pinned host objective AST; add explicit owned lifecycle calls."""
    original = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == "train")
    start = next(i for i, n in enumerate(original.body)
                 if isinstance(n, ast.FunctionDef) and n.name == "measure")
    tail = deepcopy(original.body[start:])
    loop = next(n for n in tail if isinstance(n, ast.For))
    # Host data/prior formula calls use named owners; encoded prior is still
    # the actual MoG, so public particle_ae validates its actual type.
    class NamedDraws(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "sample_data":
                node.func = ast.Attribute(value=ast.Name(id="owner", ctx=ast.Load()),
                                          attr="sample_data", ctx=ast.Load())
            elif (isinstance(node.func, ast.Attribute) and node.func.attr == "sample"
                  and isinstance(node.func.value, ast.Name) and node.func.value.id == "prior"):
                node.func = ast.Attribute(value=ast.Name(id="owner", ctx=ast.Load()),
                                          attr="sample_prior", ctx=ast.Load())
            return node
    loop = NamedDraws().visit(loop)
    def add(text):
        return ast.parse(text).body
    body = []
    for node in loop.body:
        text = ast.unparse(node)
        if text.startswith("if step == 1 or "):
            continue  # console-only extra observations, never a task gate
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple)
                and [getattr(value, "id", None) for value in node.targets[0].elts] == ["query", "offset"]):
            body.extend(add("owner.generator_phase()"))
        if text == "loss.backward()":
            body.extend(add("owner.before_generator_backward()"))
        if isinstance(node, ast.If) and ast.unparse(node.test) == "cfg.adversarial_weight > 0":
            inner = []
            for part in node.body:
                statement = ast.unparse(part)
                if statement.startswith("d_loss = gan.d_loss"):
                    inner.extend(add("owner.observe_pair(data, fake)"))
                if statement == "(d_loss + penalty).backward()":
                    inner.extend(add("owner.before_critic_backward()"))
                inner.append(part)
                if statement == "opt_d.step()":
                    inner.extend(add("owner.after_critic_step()"))
            node.body = inner
        body.append(node)
        if text == "data = owner.sample_data(cfg.batch)":
            body.extend(add("owner.begin_step(data, step - 1)"))
        elif text == "loss.backward()":
            body.extend(add("owner.after_generator_backward(adv, d_loss)"))
        elif text == "opt_g.step()":
            body.extend(add("owner.after_generator_step()\nowner.finish_step(loss, d_loss, adv, recon)"))
    loop.body = body
    cleaned = []
    for node in tail:
        text = ast.unparse(node)
        if text.startswith("_log("):
            continue
        if text == "final = measure(cfg.steps)":
            node = add("final = owner.final_metrics()")[0]
        cleaned.append(node)
    header = add("""cfg = owner.cfg
noise_policy = owner.noise
recipe = owner.encoding
prior = owner.prior
encoder = owner.encoder
decoder = owner.decode
critic = owner.critic
opt_g, opt_d = owner.opt_g, owner.opt_d
gan, regularizer = owner.loss, owner.regularizer
""")
    derived = ast.FunctionDef(name="_owned_train",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="owner")], kwonlyargs=[],
                           kw_defaults=[], defaults=[]),
        body=header + cleaned, decorator_list=[])
    result = ast.unparse(ast.fix_missing_locations(ast.Module(body=[derived], type_ignores=[]))) + "\n"
    required = ("owner.begin_step(data, step - 1)", "owner.observe_pair(data, fake)",
                "owner.before_critic_backward()", "owner.after_critic_step()", "owner.generator_phase()",
                "owner.before_generator_backward()", "owner.after_generator_backward(adv, d_loss)",
                "owner.after_generator_step()", "owner.finish_step(loss, d_loss, adv, recon)")
    if any(result.count(call) != 1 for call in required):
        raise ValueError("pinned host lifecycle insertion boundary changed")
    return result


def typed_digest(value, tensor_bytes):
    """Typed state hash preserves float diagnostic sentinels without grading."""
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
            for key, item in v.items():
                visit(key); visit(item)
            if type(v) is OrderedDict:
                visit(getattr(v, "_metadata", None))
        elif type(v) in (list, tuple):
            add(b"list" if type(v) is list else b"tuple", str(len(v)).encode())
            for item in v:
                visit(item)
        elif v is None: add(b"none", b"")
        elif type(v) is bool: add(b"bool", b"1" if v else b"0")
        elif type(v) is int: add(b"int", str(v).encode())
        elif type(v) is float: add(b"float", struct.pack("!d", v))
        elif type(v) is str: add(b"str", v.encode())
        else: raise TypeError("unknown state leaf: " + type(v).__name__)
    visit(value)
    return h.hexdigest()


class _EncodingObjective:
    def __init__(self, owner, particle_ae):
        self.owner, self.particle_ae = owner, particle_ae
    def encode(self, query, prior, *, offset):
        if prior is not self.owner.prior:
            raise ValueError("auxiliary encoding must use the same actual owned MoG")
        recipe = self.owner.recipe
        return self.particle_ae(query, offset, prior, temperature=recipe.routing_temperature,
                                distance_reduction=recipe.distance_reduction)


class _PenaltyView:
    def __init__(self, owner, bound):
        self.owner, self.bound = owner, bound
    def penalty(self, critic, real, fake, step=None):
        if critic is not self.owner.critic or step != self.owner.policy.completed_steps + 1:
            raise ValueError("foreign penalty owner or clock")
        value = self.bound(critic, real, fake)
        self.owner.audit.observe_penalty(self.bound.last_stats)
        return value, self.bound.last_stats


class _PolicyNoise:
    def __init__(self, owner):
        self.owner = owner
    def set_step(self, step):
        if self.owner.policy._phase != "ready" or self.owner.policy.completed_steps != step:
            raise ValueError("foreign external policy clock")
    @contextmanager
    def discriminator(self):
        owner = self.owner
        modes = [(module, module.training) for module in owner.generator.modules()]
        try:
            owner.generator.eval()
            owner.critic.train()
            with owner.torch.no_grad():
                yield
        finally:
            for module, mode in modes:
                module.training = mode
    def evaluation(self, step):
        return self.owner.evaluation(step)


class AEOwner:
    """Caller owns the actual modules and public policy; no scheduler here."""
    def _hook(self, name, *args, **kwargs):
        result = getattr(self.policy, name)(*args, **kwargs)
        self.calls[name] += 1
        return result
    def begin_step(self, real, step):
        self.require_fresh()
        if self.policy.completed_steps != step:
            raise ValueError("wrong host update clock")
        self._hook("begin_step", real, execution_limit=STEPS)
    def observe_pair(self, real, fake):
        self.policy.observe_critic_pair(real, fake)
    def before_critic_backward(self): self._hook("before_critic_backward")
    def after_critic_step(self): self._hook("after_critic_step")
    def before_generator_backward(self): self._hook("before_generator_backward")
    def after_generator_backward(self, adv, d_loss):
        self._hook("after_generator_backward", loss_gan=adv, loss_critic=d_loss)
    def after_generator_step(self): self._hook("after_generator_step")
    def finish_step(self, loss, d_loss, adv, recon):
        self._hook("finish_step")
        values = dict(step=self.policy.completed_steps, loss=float(loss.detach()),
                      critic_loss=float(d_loss.detach()), adversarial_loss=float(adv.detach()),
                      reconstruction_loss=float(recon.detach()),
                      training_output_sigma=self.policy.output_sigma())
        if any(not math.isfinite(v) for key, v in values.items() if key != "step"):
            raise ValueError("nonfinite original objective or physical output noise")
        self.losses.append(values)
    def generator_phase(self):
        self.generator.train(); self.encoder.train(); self.critic.eval()
    def sample_data(self, n):
        with self.streams.fork("data", component="host", purpose="global"):
            return self.host.sample_data(n)
    def sample_prior(self, n):
        return self.prior.sample(n, generator=self.policy.latent_generator,
            noise_generator=self.streams.generator("prior", component="latent", purpose="gaussian"))
    def decode(self, latent):
        self.require_fresh()
        if self._reading_step is not None:
            generated = self.generator(latent)
            sigma = self.output_noise_std(self.recipe, self._reading_step)
            if sigma:
                generated = generated + sigma * self.torch.randn(generated.shape,
                    generator=self.eval_output, device=generated.device, dtype=generated.dtype)
            self._capture_decoded.append(generated.detach().cpu().clone())
            return generated
        return self.policy.generate(latent)
    def schedule_optimizer(self, optimizer, step):
        if optimizer not in (self.opt_g, self.opt_d) or self.policy.completed_steps != step:
            raise ValueError("foreign optimizer or schedule clock")
        # begin_step owns original public schedules/stationarity once; the
        # legacy fixed finite-horizon cosine must not replace total_steps=None.
    def checkpoint(self, step, measure):
        if step not in CLOCKS:
            return
        if step in [point["step"] for point in self.observations]:
            raise ValueError("duplicate ordinary observation")
        values = measure()
        self.observations.append(dict(step=step, **values))
        self.retained.append(dict(step=step, target=self._capture_target,
            reconstructed=self._capture_decoded[0], generated=self._capture_decoded[1],
            prior=self.prior.z.detach().cpu().clone(),
            anchors=self.host._anchors().detach().cpu().clone(), metrics=deepcopy(values)))
    def final_metrics(self):
        if not self.observations or self.observations[-1]["step"] != STEPS:
            raise ValueError("terminal observation is absent")
        return {key: self.observations[-1][key] for key in ("recon_mse", "hold")}
    def evaluate(self, encoder, decoder, prior, recipe):
        if (encoder is not self.encoder or prior is not self.prior or recipe is not self.encoding
                or decoder != self.decode or self._reading_step is None):
            raise ValueError("foreign ordinary observation owners")
        def capture_encoder(data):
            self._capture_target = data.detach().cpu().clone()
            return self.encoder(data)
        return self.host.evaluate(capture_encoder, decoder, prior, recipe)
    @contextmanager
    def evaluation(self, step):
        self.require_fresh()
        if (type(step) is not int or step not in [0, *CLOCKS]
                or self.policy._phase != "ready" or self.policy.completed_steps != step):
            raise ValueError("observation outside original completed clocks")
        self.source_guard()
        before = self.state_sha256()
        global_before = self.global_sha256()
        named_before = self.streams.audit()
        modes = [(module, module.training) for model in self.policy._training_modules().values()
                 for module in model.modules()]
        self._reading_step, self._capture_decoded, self._capture_target = step, [], None
        try:
            for model in self.policy._training_modules().values():
                model.eval()
            with self.streams.preserve(), self.streams.fork("eval", component="host", purpose="global"):
                with self.torch.no_grad():
                    yield
        finally:
            self._reading_step = None
            for module, mode in modes:
                module.training = mode
        after = self.state_sha256()
        global_after = self.global_sha256()
        named_after = self.streams.audit()
        comparison = self.streams.compare(named_before, named_after)
        pure = (before == after and global_before == global_after
                and comparison["unintended_rng_deviations"] == 0)
        if not pure or len(self._capture_decoded) != 2 or self._capture_target is None:
            raise ValueError("ordinary evaluator changed state/RNG or lost a source output")
        self.purity.append(dict(step=step, before_sha256=before, after_sha256=after,
            global_rng_before_sha256=global_before, global_rng_after_sha256=global_after,
            named_rng_before=named_before, named_rng_after=named_after, pure=True,
            unintended_rng_deviations=0, sample_count=1024, forward_calls=dict(encoder=1, decoder=2),
            scoring_weights="live", evaluation_DV12=False,
            scheduled_output_sigma=self.output_noise_std(self.recipe, step)))
        self.source_guard()
    def _modes(self):
        return {name: {key: module.training for key, module in model.named_modules()}
                for name, model in self.policy._training_modules().items()}
    def _gradients(self):
        return {name: {key: None if p.grad is None else p.grad.detach().clone()
                       for key, p in model.named_parameters()}
                for name, model in self.policy._training_modules().items()}
    def state_sha256(self):
        return typed_digest(dict(policy=self.policy.state_dict(), named_rng=self.streams.state_dict(),
                                 gradients=self._gradients(), modes=self._modes(), calls=self.calls,
                                 mechanism_counters=self.audit.rows), self.tensor_bytes)
    def complete_state(self):
        return dict(schema="pg_canonical_ae_complete_owner_state_v1", binding=deepcopy(self.binding),
            completed_steps=self.policy.completed_steps, policy=self.policy.state_dict(),
            named_rng=self.streams.state_dict(), global_rng=self.global_state(),
            gradients=self._gradients(), modes=self._modes(), lifecycle_calls=deepcopy(self.calls),
            initial_owner=deepcopy(self.initial_receipt),
            observation_cursor=[p["step"] for p in self.observations],
            measurement_purity=deepcopy(self.purity), original_cfg=asdict(self.cfg),
            mechanism_counters=deepcopy(self.audit.rows))


def construct_owner(root, binding, *, device, source_guard):
    """Actual CPU owner factory, called once by a fresh guard after admission."""
    if str(device) != "cpu" or not callable(source_guard):
        raise ValueError("canonical AE requires CPU ownership and a real source guard")
    source_guard()
    reference = json.loads(_source(root, CONFIG_PATH))
    expected = resolve_binding(root, dict(recipe_preset="atlas", recipe_overrides=reference),
                               json.loads(_source(root, TASK_PATH)), json.loads(_source(root, PROTOCOL_PATH)))
    if canonical(binding) != canonical(expected):
        raise ValueError("forged full AE binding before model construction")
    import torch
    import numpy as np
    import random
    from particlegan import Recipe
    from particlegan.policy import UpdatePolicy, output_noise_std
    from particlegan.autoencoder import particle_ae
    from particlegan.init import deterministic_orthogonal_
    from particlegan.capabilities import prior_mechanisms
    from benchmarks.locked_shared.hosts import ae_gan_hold
    from experiments.forge.rng import NamedStreams
    from experiments.forge.mechanisms import MechanismAudit
    from experiments.forge.policy_adapters import finite_policy_state
    source_guard()
    for value, relative in ((Recipe, "particlegan/recipes.py"), (UpdatePolicy, "particlegan/policy.py"),
                           (particle_ae, "particlegan/autoencoder.py"),
                           (deterministic_orthogonal_, "particlegan/init.py"),
                           (ae_gan_hold, HOST_PATH), (NamedStreams, "experiments/forge/rng.py"),
                           (MechanismAudit, "experiments/forge/mechanisms.py"),
                           (finite_policy_state, "experiments/forge/policy_adapters.py")):
        _loaded_source(root, value, relative)
    if torch.get_default_dtype() != torch.float32 or torch.get_num_threads() != 1:
        raise ValueError("original float32/CPU1 execution required")
    recipe = Recipe(**binding["recipe"])
    if (canonical(asdict(recipe)) != canonical(binding["recipe"])
            or canonical(recipe.to_dict()) != canonical(binding["recipe_serialized"])):
        raise ValueError("actual complete Recipe differs before construction")
    owner = AEOwner()
    owner.torch, owner.recipe, owner.binding, owner.host = torch, recipe, deepcopy(binding), ae_gan_hold
    owner.source_guard, owner.streams = source_guard, NamedStreams(0, device="cpu")
    owner.restored, owner.construction_id = False, object()
    owner.init = dict(kind="deterministic_orthogonal", networks={}, prior="R2_explicit_width_preserved")
    for role, shape in (("generator", (2, 2)), ("encoder", (2, 4)), ("critic", (2, 1))):
        component = "discriminator" if role == "critic" else role
        with owner.streams.fork("init", component=component, purpose="construction"):
            model = ae_gan_hold.MLP(*shape).to(device="cpu")
        seed = owner.streams.seed_for("init", component=component, purpose="weights")
        deterministic_orthogonal_(model, seed=seed)
        setattr(owner, role, model)
        owner.init["networks"][role] = dict(seed=seed, constructor="host.MLP", dimensions=list(shape), hidden=32)
    with owner.streams.fork("init", component="prior", purpose="construction"):
        owner.prior = recipe.make_prior(**PRIOR_FACTORY_OVERRIDES, device="cpu")
    deterministic_orthogonal_(owner.prior,
        seed=owner.streams.seed_for("init", component="prior", purpose="weights"))
    mechanisms = prior_mechanisms(owner.prior, latent_damping_max_rate=recipe.latent_damping_max_rate,
                                  prior_beta1=(recipe.prior_betas or recipe.betas)[0])
    if not mechanisms["prior"]["a2_eligible"] or not mechanisms["a2"]["enabled"]:
        raise ValueError("actual nonstandardized MoG must support the requested current A2")
    owner.opt_g = recipe.make_generator_optimizer([
        dict(params=list(owner.generator.parameters()), lr=recipe.lr, forge_role="generator"),
        dict(params=list(owner.encoder.parameters()), lr=recipe.lr, forge_role="encoder"),
        dict(params=[owner.prior.z], lr=recipe.lr * recipe.prior_lr_mult,
             betas=recipe.prior_betas or recipe.betas, forge_role="prior",
             **({"eps": recipe.prior_eps} if recipe.prior_eps is not None else {}))], latent_table=owner.prior.z)
    owner.opt_g.prior_mechanisms = deepcopy(mechanisms)
    owner.opt_d = recipe.make_critic_optimizer(owner.critic, ema_critic=deepcopy(owner.critic))
    penalty = recipe.make_critic_penalty(owner.opt_d, collect_stats=True)
    streams = dict(latent_generator=owner.streams.generator("prior", component="latent", purpose="indices"),
                   penalty_generator=owner.streams.generator("noise", component="penalty", purpose="training"),
                   noise_generator=owner.streams.generator("noise", component="generator", purpose="output"),
                   eval_generator=owner.streams.generator("eval", component="sampler", purpose="samples"))
    owner.policy = UpdatePolicy(recipe, owner.generator, owner.critic, prior=owner.prior, encoder=owner.encoder,
        generator_optimizer=owner.opt_g, critic_optimizer=owner.opt_d, table_optimizer=owner.opt_g,
        roles=[["generator", "encoder", "table"], ["critic"]],
        row_semantics="independent", streams=streams, seed=0, penalty=penalty)
    owner.prior_mechanisms = deepcopy(mechanisms)
    owner.loss, owner.encoding, owner.output_noise_std = recipe.make_loss(), _EncodingObjective(owner, particle_ae), output_noise_std
    owner.cfg = ae_gan_hold.HoldConfig(name="atlas", lr=recipe.lr, reconstruction_weight=recipe.reconstruction_weight)
    owner.noise = _PolicyNoise(owner)
    owner.calls = {name: 0 for name in ("begin_step", "before_critic_backward", "after_critic_step",
        "before_generator_backward", "after_generator_backward", "after_generator_step", "finish_step")}
    owner.observations, owner.purity, owner.retained, owner.losses = [], [], [], []
    owner._reading_step, owner._capture_decoded, owner._capture_target = None, [], None
    owner.eval_output = owner.streams.generator("eval", component="generator", purpose="output")
    owner.streams.generator("eval", component="critic", purpose="input")
    owner.streams.generator("eval", component="host", purpose="global")
    owner.streams.generator("data", component="host", purpose="global")
    owner.streams.generator("prior", component="latent", purpose="gaussian")
    def tensor_bytes(value):
        if not isinstance(value, torch.Tensor):
            return None
        data = value.detach().cpu().contiguous()
        return dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
                    requires_grad=value.requires_grad), data.numpy().tobytes()
    owner.tensor_bytes = tensor_bytes
    owner.global_state = lambda: dict(torch_cpu=torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state())
    def global_bytes(value):
        if isinstance(value, np.ndarray):
            return dict(shape=list(value.shape), dtype=str(value.dtype)), value.tobytes()
        return tensor_bytes(value)
    owner.global_sha256 = lambda: typed_digest(owner.global_state(), global_bytes)
    owner.audit = MechanismAudit(recipe, owner.opt_d, [owner.opt_g])
    owner.finite_policy_state = finite_policy_state
    owner.regularizer = _PenaltyView(owner, penalty)
    owner.initial_state_sha256 = owner.state_sha256()
    # A fresh guard must replace this before the first initial observation.
    def unverified():
        raise ValueError("actual owner lacks admitted fresh-repeat attestation")
    owner.require_fresh = unverified
    source_guard()
    return owner


def owner_initial_receipt(owner):
    """Inspect the actual newly constructed owner before every forward."""
    import torch
    from particlegan.policy import UpdatePolicy
    from particlegan.particle_prior import MoGParticlePrior
    from particlegan.k3p import K3PGeneratorAdam
    from particlegan.ka2 import KA2CriticAdam
    from benchmarks.locked_shared.hosts import ae_gan_hold
    if (type(owner) is not AEOwner or type(owner.policy) is not UpdatePolicy
            or any(type(model) is not ae_gan_hold.MLP for model in (owner.generator, owner.encoder, owner.critic))
            or type(owner.prior) is not MoGParticlePrior or type(owner.opt_g) is not K3PGeneratorAdam
            or type(owner.opt_d) is not KA2CriticAdam):
        raise ValueError("actual declared AE public owner types required")
    policy = owner.policy
    parameters = [p for opt in (owner.opt_g, owner.opt_d) for group in opt.param_groups for p in group["params"]]
    bindings = dict(policy_generator=policy.G is owner.generator, policy_encoder=policy.encoder is owner.encoder,
        policy_critic=policy.D is owner.critic, policy_prior=policy.prior is owner.prior,
        policy_table=policy.table is owner.prior.z, table_optimizer=policy.table_optimizer is owner.opt_g,
        encoder_average=policy.ema_encoder is not None, live_and_average_owners_distinct=all(
            a is not b for name, live in policy._training_modules().items() if name != "critic"
            for a, b in zip(live.parameters(), policy._average_modules()[name].parameters())),
        unique_parameter_ownership=len(parameters) == len({id(p) for p in parameters}),
        current_a2_enabled=owner.opt_g.latent_damping is not None,
        all_original_policy_owners=all(value is not None for value in (
            policy.controller, policy.lr_settle, policy.birth_death, policy.row_evidence,
            policy._feature_selection, policy.reopen_guard, policy.log_output_sigma)))
    finite = all(bool(torch.isfinite(p).all()) for p in parameters)
    zero = (policy.completed_steps == 0 and policy._phase == "ready" and not owner.restored
            and not owner.opt_g.state and not owner.opt_d.state and not any(owner.calls.values())
            and owner.opt_d.record.observed_steps == 0 and policy.controller.updates == 0
            and owner.opt_g.latent_damping.total == 0)
    if (not zero or not finite or not all(bindings.values()) or owner.prior.standardize is not False
            or owner.prior.z.shape != (12, 2) or not owner.prior.z.requires_grad
            or not torch.equal(owner.prior.sigma, owner.prior.sigma.new_tensor(.025))
            or policy.roles != [["generator", "encoder", "table", "noise"], ["critic"]]
            or owner.state_sha256() != owner.initial_state_sha256):
        raise ValueError("new finite original initialization with empty optimizers and zero clocks required")
    return dict(schema=OWNER_SCHEMA, task_id=TASK_ID, completed_steps=0, phase="ready", restored=False,
        seed=owner.streams.seed, rng_version=owner.streams.version, device="cpu", all_finite=True,
        optimizer_updates=dict(generator=0, encoder=0, prior=0, discriminator=0, noise=0),
        optimizer_state_entries=dict(generator=len(owner.opt_g.state), discriminator=len(owner.opt_d.state)),
        table=dict(shape=[12, 2], requires_grad=True, initialization="R2", state_sha256=typed_digest(owner.prior.state_dict(), owner.tensor_bytes)),
        actual_prior=deepcopy(ACTUAL_PRIOR), prior_factory_overrides=deepcopy(PRIOR_FACTORY_OVERRIDES),
        object_bindings=bindings, roles=deepcopy(policy.roles), initializer=deepcopy(owner.init),
        initial_owner_state_sha256=owner.initial_state_sha256,
        initial_global_rng_sha256=owner.global_sha256(), initial_named_rng=owner.streams.audit(),
        resolved_recipe_sha256=digest(asdict(owner.recipe)), full_resolved_recipe=asdict(owner.recipe),
        source_contract_sha256=owner.binding["source_contract_sha256"])


def run_case(root, request, *, output_dir, device, source_guard, fresh_repeat_guard):
    """ROOT-only admitted child; this owner never grades or writes files."""
    binding = resolve_binding(root, request["candidate"], request["task"], request["protocol"])
    if canonical(binding) != canonical(request["binding"]) or str(device) != "cpu":
        raise ValueError("request binding/device drift")
    owner = fresh_repeat_guard.construct(lambda: construct_owner(root, binding, device=device,
                                        source_guard=source_guard), owner_initial_receipt)
    owner.require_fresh = lambda: fresh_repeat_guard.require_owned(owner)
    owner.require_fresh()
    owner.initial_receipt = owner_initial_receipt(owner)
    namespace = dict(vars(owner.host))
    namespace.update(evaluate=owner.evaluate, checkpoint=owner.checkpoint,
                     schedule_optimizer=owner.schedule_optimizer)
    overlay = derive_host_train(_source(root, HOST_PATH).decode())
    exec(compile(overlay, "<pinned-AE-public-policy-owner-overlay>", "exec"), namespace)
    source_guard()
    raw = namespace["_owned_train"](owner)
    if owner.policy.completed_steps != STEPS or [row["step"] for row in owner.observations] != CLOCKS:
        raise ValueError("original complete250/24 horizon is absent")
    counts = {}
    for role, parameters, optimizer in (("generator", owner.generator.parameters(), owner.opt_g),
            ("encoder", owner.encoder.parameters(), owner.opt_g), ("prior", [owner.prior.z], owner.opt_g),
            ("discriminator", owner.critic.parameters(), owner.opt_d),
            ("noise", [owner.policy.log_output_sigma], owner.opt_g)):
        clocks = [int(optimizer.state.get(p, {}).get("step", 0)) for p in parameters]
        if not clocks or min(clocks) != STEPS or max(clocks) != STEPS:
            raise ValueError("actual " + role + " optimizer clock is incomplete")
        counts[role] = STEPS
    if any(count != STEPS for count in owner.calls.values()) or len(owner.losses) != STEPS:
        raise ValueError("ordered public lifecycle did not complete250")
    def finite(value):
        if isinstance(value, owner.torch.Tensor): return bool(owner.torch.isfinite(value).all())
        if isinstance(value, dict): return all(finite(v) for v in value.values())
        if isinstance(value, (list, tuple)): return all(finite(v) for v in value)
        return math.isfinite(value) if type(value) in (float, int) else True
    all_finite = all(finite(p) and finite(p.grad) for opt in (owner.opt_g, owner.opt_d)
                     for group in opt.param_groups for p in group["params"])
    # Learned/Adam tensors must be finite; source-defined diagnostic sentinels
    # remain typed state, rather than a new blanket public-state health law.
    all_finite = all_finite and all(finite(state) for opt in (owner.opt_g, owner.opt_d) for state in opt.state.values())
    policy_health = owner.finite_policy_state(owner.policy.state_dict())
    all_finite = all_finite and policy_health
    audit = owner.audit.receipt()
    from experiments.forge.mechanisms import mechanism_blockers
    source_guard()
    backend = deepcopy(owner.policy._feature_selection.state_dict())
    raw["cfg"] = asdict(owner.cfg)
    raw.update(spec=dict(task_id=TASK_ID, steps=STEPS, thresholds=deepcopy(request["task"]["evaluation"]["thresholds"])),
               observations=deepcopy(owner.observations), live=owner.final_metrics(), losses=deepcopy(owner.losses))
    result = dict(task_id=TASK_ID, execution_path="public_components", device="cpu", raw=raw,
        evidence=dict(observations=deepcopy(owner.observations), live=owner.final_metrics(),
            scoring_weights="live", sampling_contract_version=1,
            sampling_law="generated_and_reconstructed_prior_with_scheduled_output_noise",
            eval_output_noise="public_recipe_schedule", measurement_purity=deepcopy(owner.purity),
            guards=dict(all_finite=all_finite, public_policy_state_finite=policy_health, optimizer_updates=counts,
                        hooks_exercised=not mechanism_blockers(audit), mechanism_audit=audit,
                        unintended_rng_deviations=0)),
        applied=dict(recipe=asdict(owner.recipe), recipe_serialized=owner.recipe.to_dict(),
            source_contract=deepcopy(binding["source_contract"]), source_contract_sha256=binding["source_contract_sha256"],
            actual_prior=deepcopy(ACTUAL_PRIOR), prior_factory_overrides=deepcopy(PRIOR_FACTORY_OVERRIDES),
            prior_mechanisms=deepcopy(owner.prior_mechanisms), auxiliary_encoder="free_host_MLP_query_offset",
            policy_owner="particlegan.UpdatePolicy", row_semantics="independent", backend_selection=backend,
            initial_owner=deepcopy(owner.initial_receipt), roles=deepcopy(owner.policy.roles),
            initial_lrs=deepcopy(owner.policy.initial_lrs), lifecycle_calls=deepcopy(owner.calls), optimizer_updates=counts,
            streams=owner.streams.manifest(), observation_owner="live_G_E_and_fixed_width_actual_MoG",
            evaluation_DV12=False, selected_or_averaged_metric=False,
            external_horizon=STEPS, intrinsic_horizon=owner.recipe.total_steps,
            qualification=False, default_adoption=False, speed_ranking=False),
        retained_goal_states=owner.retained, complete_state=owner.complete_state())
    owner.require_fresh(); source_guard()
    return result
