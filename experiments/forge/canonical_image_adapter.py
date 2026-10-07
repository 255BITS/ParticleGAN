"""Prospective four-image full-original-Atlas owner; import is metadata-only.

The public GANTrainer owns every update. A pinned source overlay omits only its
final compatibility serving swap, so the original finite-center measurement
reads the live training iterate. ROOT supplies the admission and source guards.
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
from types import MethodType, SimpleNamespace

SCHEMA = "pg_canonical_image_policy_binding_v1"
OWNER_SCHEMA = "pg_canonical_image_initial_owner_v1"
TASK_IDS = ("img_stripes2", "img_bars4", "img_blobs4", "img_intensity2")
CONFIG_PATH = "configs/100gaussians/atlas.json"
HOST_PATH = "benchmarks/transfer_suite/image_tasks.py"
PROTOCOL_PATH = "configs/forge/protocols/screening.json"
TRAINER_PATH = "particlegan/training.py"
STEPS = 600
CLOCKS = list(range(25, 601, 25))
MEDIA_INDICES = (0, 3, 6, 9, 12, 14, 17, 20, 23)
TASK_BINDINGS = dict(num_particles=32, z_dim=8, batch_size=32,
                     prior_kind="particles", sigma_rel=0., standardize=False)
RECIPE_SHA256 = "c64791736011b9eda199a66e89bf26e454184720e8df23393cc1d3dc39806eff"
SOURCE_PINS = {'benchmarks/locked_shared/observation.py': {'bytes': 4351,
                                             'sha256': 'bd6f9845b44f1ec2a58d445727990ba5068c7aca3b6f981cf38d738a37c4513b'},
 'benchmarks/toy100/device.py': {'bytes': 10288,
                                 'sha256': '7fa0f53db44e8824d3497b471e9fe8db56cbf8b4e02ddf997f65beed63f81911'},
 'benchmarks/transfer_suite/image_tasks.py': {'bytes': 15788,
                                              'sha256': '4f070d0879cbaaaa076f82b4683cfe74ea9ed8d85480d1922e249444a752ce58'},
 'benchmarks/transfer_suite/protocol.py': {'bytes': 9964,
                                           'sha256': '99469b022b790a18a74021a6fe49424d95f535afaa220643688a1ddd7a70ab89'},
 'configs/100gaussians/atlas.json': {'bytes': 1798,
                                     'sha256': 'a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4'},
 'configs/forge/protocols/screening.json': {'bytes': 972,
                                            'sha256': '3fefb4d47fd2cd8aa6ed110c0a9f5bffefaae700431d1f57ca7b162c8efbb803'},
 'configs/forge/tasks/img_bars4.json': {'bytes': 3108,
                                        'sha256': '562c05c07de33e5a5e088cff097af9cf9f6f8b37bb436aed37bf502fb7aff86f'},
 'configs/forge/tasks/img_blobs4.json': {'bytes': 3111,
                                         'sha256': '15c7ec3cfe80539b14d24824afc1a6fffae99e27b5a38e75c991823b5aceb336'},
 'configs/forge/tasks/img_intensity2.json': {'bytes': 3123,
                                             'sha256': '6a69162bee7dd1c50c9f596f29ed339a5e475c296156c4324e5092099cddf90d'},
 'configs/forge/tasks/img_stripes2.json': {'bytes': 3116,
                                           'sha256': '010a96d3829b639d13f1150f30a5ff44119bcc10a23822f4191773bab177eb70'},
 'configs/forge/view-history/discriminator_stability-v3.json': {'bytes': 4104,
                                                                'sha256': '007660b1a48ed597ffe4a9b3bdfc33ed50fe3aa8086502908a2a4ca63f24d48b'},
 'experiments/forge/api.py': {'bytes': 36703,
                              'sha256': 'e745bfc9f7246055e4d3cea0b3943493ccdf3769527a54684182514a6dc767cd'},
 'experiments/forge/boundaries.py': {'bytes': 17716,
                                     'sha256': '17ce3e0da65d62158955ed03f5a7ba6f4a3526c74730019a5dc51f2d189b7218'},
 'experiments/forge/mechanisms.py': {'bytes': 11826,
                                     'sha256': '9e4e1fe1f8cb9a24148cab88b7bf2f980e805a2e488c16a982768519bbe1b23f'},
 'experiments/forge/policy_adapters.py': {'bytes': 17992,
                                          'sha256': '714046ea1655f1946e09dc2aeed27c43cf93f2841237c2be47f0f0049a5df9fa'},
 'experiments/forge/rng.py': {'bytes': 6893,
                              'sha256': 'ae7b9a8d61da42136d970188a6f168e03e2d7c5b90cf6fdbd93ad929aba2f293'},
 'particlegan/__init__.py': {'bytes': 1811,
                             'sha256': 'b56efa562acc551d639548c299002e2cd76338b3fd7062699780add51f4f5e8c'},
 'particlegan/_qr.py': {'bytes': 1951,
                        'sha256': 'b59cd62cc27ed557b41b2d17ed86ed578c762a3a8a392d63731a5a3b2b018916'},
 'particlegan/anchor_birth.py': {'bytes': 6401,
                                 'sha256': 'aa50983d8b61304acf4e1138f16fd4784798983df91e931863bcacd3b511f76f'},
 'particlegan/autoencoder.py': {'bytes': 6194,
                                'sha256': 'f3fbb9184e9f44112e15eccf0c0246dfba5ed490d3c7b81b0f837a5335916402'},
 'particlegan/birth_death.py': {'bytes': 45396,
                                'sha256': 'b14c50c611a8cf188347e391739fca50a5171200eb6be90e4e3d1509d64e47ed'},
 'particlegan/birth_phase.py': {'bytes': 21144,
                                'sha256': '5ff929e5e9858aa24c4cc2f8999b5d0603e25e42ca3844d1da4cc28b1981d3d7'},
 'particlegan/capabilities.py': {'bytes': 2285,
                                 'sha256': '39df9c67dbedf612740057e5fa743bc724678a1bc323d1f7cf550b37c0c01177'},
 'particlegan/conditioning.py': {'bytes': 4878,
                                 'sha256': '8734fe338868ace4b5851a8fdb1f071e46d47396f66a3765e6b1ef37f0481486'},
 'particlegan/continuous.py': {'bytes': 60875,
                               'sha256': '4ceab49c7d51d1769ae91b7f8bc892eaf380ca6fbd77a948a086ade6627e9a41'},
 'particlegan/diffusion.py': {'bytes': 5100,
                              'sha256': '19c3aa8772703891551b9122fde8e74c07996f779892b7cb6be09553691efa61'},
 'particlegan/discriminators.py': {'bytes': 5833,
                                   'sha256': '0e2efb125ffb314577612ab7a2eba66b0a1a2d28ad25f403f42c18b6f6ee333f'},
 'particlegan/feature_cells.py': {'bytes': 125325,
                                  'sha256': 'e0ea05f61abd7845c7437eed9a79c3a2511d5551ff04f4e619b6b9c8992e221a'},
 'particlegan/feature_policy.py': {'bytes': 14650,
                                   'sha256': 'de1a50f70a37d4c9174c8850ce3eac2a2597ed01f19fc4fe1a2ce0ec3176a8b2'},
 'particlegan/feature_reference.py': {'bytes': 32368,
                                      'sha256': '58e076e9e04600be25e28afafc537a340de727371f4acbbfb2b7dd8bebac18b1'},
 'particlegan/gan_loss.py': {'bytes': 1016,
                             'sha256': '1c1019dfe71c583e32a05df0d2f794f9fff6d9ee1ea0332f57f6379ae70cf6b7'},
 'particlegan/grad_regularizers.py': {'bytes': 14853,
                                      'sha256': 'a540d05a4a992540b6a5f3b5ff7ccc11ebaf18592135b87c95638adbe68aa747'},
 'particlegan/init.py': {'bytes': 24559,
                         'sha256': 'e15d4de01eb49f7097bc249abacab907ff067385412346f54ab0d510cb8649d0'},
 'particlegan/k3p.py': {'bytes': 33012,
                        'sha256': '200ead2c27ab0b2aa6068f1b1b1b5c826b8125aa8602bfec08b31aaa2894401c'},
 'particlegan/ka2.py': {'bytes': 16551,
                        'sha256': '95fdaa58a2bdc229d5d146ef89548bc3fe07582d06181a149ed1bb4f5d632703'},
 'particlegan/mean_transport.py': {'bytes': 39633,
                                   'sha256': '57a68e0a65c5424a4460f65002f3862a69e90af330a4d47c700dc773683a2c06'},
 'particlegan/output_moments.py': {'bytes': 8387,
                                   'sha256': 'dcf3e27228d2c3c9738e473c5b22ee9e6d5047ddcd60c538b66526adb53ef188'},
 'particlegan/particle_prior.py': {'bytes': 17899,
                                   'sha256': '0220878bebea227da63abbf9f5b1ebdabdc6aea54fd2332fd2cab02ed734238f'},
 'particlegan/policy.py': {'bytes': 81022,
                           'sha256': '8370e36b5b93afe95ae24d2b385aac8735a587f86ef2b87ed369ecfdfcb6fec5'},
 'particlegan/population_continuity.py': {'bytes': 20329,
                                          'sha256': '378cd75cca45bc8da4da33e686a12cf99df853042687978314e37c532bc17fe6'},
 'particlegan/recipe_schedules.py': {'bytes': 6044,
                                     'sha256': 'd05e459b3e9e5f938b01a854273b74343f2b6d382bec4ae5151dd1f1cd032500'},
 'particlegan/recipes.py': {'bytes': 53612,
                            'sha256': '1a7f9df746e242a2774068c819d70d50cd72dc4f7f6b22b0bd13e1345dcf8ab2'},
 'particlegan/routing.py': {'bytes': 64904,
                            'sha256': '63249d808fccb3d4e113eb1d638a245ee665c3cade20a1542bb706f316b4abcc'},
 'particlegan/row_evidence.py': {'bytes': 9397,
                                 'sha256': '78be40141ef3946860458e9842159be9321f05c217a4800dd4134113b6eba93e'},
 'particlegan/training.py': {'bytes': 39880,
                             'sha256': '7dadc219135bc3e6bf657bfe9bc3fa81dd14d4b2a31c70d5ff61e2941db35c94'},
 'particlegan/vicreg_loss.py': {'bytes': 2297,
                                'sha256': 'ab1c4dc266dec2c35337f449917240eb38afede7590d2154b63f6f471dc45a36'}}


def _source(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if path.resolve() != root / relative:
        raise ValueError("foreign source alias")
    raw = path.read_bytes()
    pin = SOURCE_PINS[relative]
    if (hashlib.sha256(raw).hexdigest(), len(raw)) != (pin["sha256"], pin["bytes"]):
        raise ValueError("source drift: " + relative)
    return raw


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


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


def _loaded_source(root, module, relative):
    if Path(inspect.getfile(module)).resolve() != (Path(root) / relative).resolve():
        raise ValueError("foreign imported scientific source: " + relative)
    _source(root, relative)


def resolve_binding(root, candidate, task, protocol):
    """Validate exact source declarations before a constructor or RNG call."""
    task_id = task.get("id")
    if task_id not in TASK_IDS:
        raise ValueError("only the four original canonical image declarations are supported")
    task_path = "configs/forge/tasks/" + task_id + ".json"
    declaration = json.loads(_source(root, task_path))
    reference = json.loads(_source(root, CONFIG_PATH))
    if _declaration(task) != declaration:
        raise ValueError("canonical image declaration changed")
    if (candidate.get("recipe_preset") != "atlas"
            or candidate.get("recipe_overrides") != reference
            or candidate.get("extensions", {}) != {}
            or candidate.get("initializer", "deterministic_orthogonal") != "deterministic_orthogonal"
            or candidate.get("seed", 0) != 0):
        raise ValueError("full frozen original Atlas reference required; tuning is unsupported")
    base_protocol = deepcopy(protocol)
    if "scientific_repeat" in base_protocol and not isinstance(base_protocol.pop("scientific_repeat"), dict):
        raise ValueError("malformed recognized repeat intent")
    if base_protocol != json.loads(_source(root, PROTOCOL_PATH)):
        raise ValueError("complete ordinary seed0/named-stream screening protocol required")
    for relative in SOURCE_PINS:
        _source(root, relative)
    fields = _recipe_fields(_source(root, "particlegan/recipes.py"))
    fields.update(name="atlas", **reference)
    fields.update(TASK_BINDINGS)
    fields = json.loads(canonical(fields))
    if digest(fields) != RECIPE_SHA256 or fields["total_steps"] is not None:
        raise ValueError("complete original Recipe or task-owned image binding changed")
    spec = deepcopy(declaration["execution"]["host_definition"])
    spec["thresholds"] = deepcopy(declaration["evaluation"]["measurement"])
    if (spec["architecture"] != "transpose" or spec["width"] != 12
            or spec["steps"] != 600 or spec["particles"] != 32
            or spec["z_dim"] != 8 or spec["batch_size"] != 32):
        raise ValueError("unsupported original image host capacity")
    contract = dict(schema="pg_canonical_image_public_owner_contract_v1", task_id=task_id,
        files=deepcopy(SOURCE_PINS), external_max_steps=STEPS,
        resources=dict(gpus=1, physical_gpu=1, cpu_threads=1, timeout_seconds=1800,
                       num_particles=32, z_dim=8, batch_size=32),
        host=dict(source=HOST_PATH, generator="benchmarks.transfer_suite.image_tasks.Generator",
                  critic="benchmarks.transfer_suite.image_tasks.Discriminator", specification=spec),
        observation=dict(sampling_law="enumerated_prior_without_output_noise", eval_output_noise="clean",
            scoring_weights="live", owner="live_generator_and_raw_prior_centers", observations=CLOCKS,
            final_five=CLOCKS[-5:], retained_states=24,
            goal_steps=[CLOCKS[i] for i in MEDIA_INDICES], evaluation_draws=0,
            function="benchmarks.transfer_suite.image_tasks.measure", capture="already_computed_images"),
        effective_task_bindings=deepcopy(TASK_BINDINGS),
        initializer="deterministic_orthogonal_named_parameters_v1",
        objective=dict(gan="Recipe.make_loss", penalty="Recipe.make_critic_penalty",
                       prior="Recipe.prior_reg * Recipe.make_prior_regularizer(weight=1)",
                       real_data="uniform_templates_plus_0.01_gaussian_noise_clipped_0_1"),
        training=dict(owner="particlegan.GANTrainer", policy_owner="particlegan.UpdatePolicy",
            update_overlay="omit_only_final_post_finish_compatibility_serve_apply",
            serial_backward=False,
            intrinsic_total_steps=None, external_steps=600, updates_per_outer_step=dict(generator=1, discriminator=1),
            output_kernel="original_learned_training_kernel", row_policy="independent"),
        inapplicable=dict(standardize="raw ParticlePrior has no standardization transform",
            direct_particle_gain="latent prior uses A2; direct-response applies only to direct sample coordinates",
            serving_to_metric="averaging remains active; ordinary image metric reads live owners",
            host_rate_fields="legacy host lr/betas/penalty/prior_weight/EMA fields are inactive provenance"),
        scientific_credit=False)
    return dict(schema=SCHEMA, task_id=task_id, recipe=fields,
        recipe_serialized=serialized_recipe(fields, _source(root, "particlegan/recipes.py")),
        source_contract=contract, source_contract_sha256=digest(contract),
        task_sha256=SOURCE_PINS[task_path]["sha256"], config_sha256=SOURCE_PINS[CONFIG_PATH]["sha256"],
        protocol_sha256=SOURCE_PINS[PROTOCOL_PATH]["sha256"],
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def derive_live_step(source):
    """Remove one serving storage swap; all public scientific AST stays exact."""
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "GANTrainer")
    step = deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_step"))
    matches = [i for i, n in enumerate(step.body)
               if isinstance(n, ast.Expr) and ast.unparse(n) == "self._serve_apply()"]
    if len(matches) != 1 or ast.unparse(step.body[matches[0] - 1]) != "self.policy.finish_step()":
        raise ValueError("public final serving storage boundary changed")
    del step.body[matches[0]]
    return ast.unparse(ast.fix_missing_locations(ast.Module(body=[step], type_ignores=[]))) + "\n"


def derive_live_measure(source):
    """Tap the original single forward/metric call; never create another read."""
    measure = deepcopy(next(n for n in ast.parse(source).body
                            if isinstance(n, ast.FunctionDef) and n.name == "measure"))
    block = measure.body[0]
    if (not isinstance(block, ast.With) or len(block.body) != 1
            or ast.unparse(block.body[0]) != "return image_metrics(generator(prior.z), centers, thresholds)"):
        raise ValueError("original live-center measurement source boundary changed")
    block.body = ast.parse("images = generator(prior.z)\n"
                           "metrics = image_metrics(images, centers, thresholds)\n"
                           "_capture_images(images)\nreturn metrics").body
    return ast.unparse(ast.fix_missing_locations(ast.Module(body=[measure], type_ignores=[]))) + "\n"


def compile_source_preflight(root):
    """Model-free source compilation only; ROOT still owns copied import proof."""
    for relative in SOURCE_PINS:
        raw = _source(root, relative)
        if relative.endswith(".py"):
            ast.parse(raw, filename=relative)
    step = derive_live_step(_source(root, TRAINER_PATH))
    measure = derive_live_measure(_source(root, HOST_PATH))
    compile(step, "<pinned-live-storage-public-step>", "exec")
    compile(measure, "<pinned-original-image-measure-capture>", "exec")
    return dict(schema="pg_canonical_image_source_preflight_v1", status="PASS_SOURCE_ONLY",
        step_overlay_sha256=hashlib.sha256(step.encode()).hexdigest(),
        measure_overlay_sha256=hashlib.sha256(measure.encode()).hexdigest(),
        source_files=deepcopy(SOURCE_PINS), actual_models=0, actual_forwards=0, actual_scorers=0)


def verify_purity(before, after, global_before, global_after, named_rng):
    if (before != after or global_before != global_after
            or named_rng.get("unintended_rng_deviations") != 0):
        raise ValueError("ordinary live image measurement changed complete owner state or RNG")


def optimizer_clock(optimizer, parameters):
    counts = []
    for parameter in parameters:
        state = optimizer.state.get(parameter, {})
        if "step" not in state:
            raise ValueError("an intended optimizer parameter has no update clock")
        raw = state["step"]
        value = float(raw.item()) if hasattr(raw, "item") else float(raw)
        if not math.isfinite(value) or value < 0 or value != int(value):
            raise ValueError("invalid actual optimizer clock")
        counts.append(int(value))
    if not counts or min(counts) != max(counts):
        raise ValueError("missing or incomplete actual optimizer role")
    return counts[0]


class ImageOwner:
    """Physical public owners; instantiated only by the admitted factory."""
    def complete_state(self):
        if self.policy._fast is not None:
            raise ValueError("live owner storage was replaced by serving parameters")
        modules = dict(generator=self.generator, critic=self.critic, prior=self.prior,
                       **{"average_" + k: v for k, v in self.policy._average_modules().items()},
                       ema_critic=self.opt_d.ema_critic)
        return dict(schema="pg_canonical_image_complete_state_v1",
            trainer=self.trainer.state_dict(), policy=self.policy.state_dict(),
            recipe=asdict(self.recipe), named_streams=self.streams.state_dict(),
            gradients={k: {n: p.grad for n, p in module.named_parameters()} for k, module in modules.items()},
            noise_gradient=self.policy.log_output_sigma.grad,
            module_modes={k: {n: m.training for n, m in module.named_modules()} for k, module in modules.items()},
            calls=deepcopy(self.calls), lifecycle_audit=self.lifecycle_audit.receipt(self.trainer.completed_steps),
            data_cursor=self.trainer.completed_steps,
            observations=deepcopy(self.observations), binding=deepcopy(self.binding),
            initialization=deepcopy(self.initialization))

    def checkpoint(self, step):
        if step not in CLOCKS or self.trainer.completed_steps != step or self.policy._phase != "ready":
            raise ValueError("image observation outside the original completed clock")
        before = typed_digest(self.complete_state(), self.tensor_bytes)
        global_before = self.global_sha256()
        named = self.streams.audit()
        captured = []
        def capture(images):
            captured.append(dict(step=step, generated=images.detach().cpu().clone(),
                target=self.centers.detach().cpu().clone(), prior=self.prior.z.detach().cpu().clone()))
        self.measure_namespace["_capture_images"] = capture
        values = self.measure_namespace["measure"](
            self.generator, self.prior, self.centers, self.spec["thresholds"])
        after = typed_digest(self.complete_state(), self.tensor_bytes)
        global_after = self.global_sha256()
        rng = self.streams.compare(named, self.streams.audit())
        verify_purity(before, after, global_before, global_after, rng)
        if len(captured) != 1:
            raise ValueError("original image observer must capture exactly one existing forward")
        canonical(values)  # Existing numeric metadata must be finite.
        self.observations.append(dict(step=step, **values))
        self.retained.extend(captured)
        self.purity.append(dict(step=step, before_sha256=before, after_sha256=after,
            global_before_sha256=global_before, global_after_sha256=global_after,
            named_rng=rng, pure=True, owner="live_generator_and_raw_prior_centers", eval_draws=0,
            forward_calls=1, original_metric_calls=1, served_parameter_swap=False))


def construct_owner(root, binding, *, source_guard, device="cuda:1"):
    """ROOT-only late factory; no model exists at metadata preflight."""
    source_guard()
    import random
    import torch
    import numpy as np
    from particlegan import Recipe, GANTrainer, init
    from experiments.forge.api import require_legacy_autograd_source
    require_legacy_autograd_source(GANTrainer, SOURCE_PINS[TRAINER_PATH]["sha256"],
                                 owner="Original common26 image owner")
    from particlegan.policy import UpdatePolicy
    from experiments.forge.rng import NamedStreams
    from experiments.forge.mechanisms import MechanismAudit
    from experiments.forge.policy_adapters import PolicyLifecycleAudit
    from benchmarks.transfer_suite import image_tasks
    source_guard()
    for module, relative in ((Recipe, "particlegan/recipes.py"), (GANTrainer, TRAINER_PATH),
            (UpdatePolicy, "particlegan/policy.py"), (init, "particlegan/init.py"),
            (NamedStreams, "experiments/forge/rng.py"), (MechanismAudit, "experiments/forge/mechanisms.py"),
            (PolicyLifecycleAudit, "experiments/forge/policy_adapters.py"), (image_tasks, HOST_PATH)):
        _loaded_source(root, module, relative)
    task_id = binding.get("task_id")
    if task_id not in TASK_IDS:
        raise ValueError("unknown canonical image owner")
    expected = resolve_binding(root, dict(recipe_preset="atlas", recipe_overrides=json.loads(_source(root, CONFIG_PATH))),
        json.loads(_source(root, "configs/forge/tasks/" + task_id + ".json")),
        json.loads(_source(root, PROTOCOL_PATH)))
    if binding != expected:
        raise ValueError("forged source/Recipe/owner binding before construction")
    device = torch.device(device)
    if device.type != "cuda" or device.index is None or torch.get_num_threads() != 1 or torch.get_default_dtype() != torch.float32:
        raise ValueError("declared image owner requires one explicit CUDA device, CPU1 and FP32")
    recipe = Recipe(**binding["recipe"])
    if canonical(asdict(recipe)) != canonical(binding["recipe"]) or canonical(recipe.to_dict()) != canonical(binding["recipe_serialized"]):
        raise ValueError("actual complete public Recipe differs")
    owner = ImageOwner()
    owner.torch, owner.recipe, owner.binding = torch, recipe, deepcopy(binding)
    owner.spec = deepcopy(binding["source_contract"]["host"]["specification"])
    owner.streams = NamedStreams(0, device=device)
    owner.restored, owner.construction_id = False, object()
    def tensor_bytes(value):
        if not isinstance(value, torch.Tensor): return None
        array = value.detach().cpu().contiguous()
        return dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
                    requires_grad=value.requires_grad), array.numpy().tobytes()
    owner.tensor_bytes = tensor_bytes
    def global_sha256():
        return typed_digest((torch.get_rng_state(), torch.cuda.get_rng_state(device),
                             random.getstate(), np.random.get_state()),
            lambda v: (dict(shape=list(v.shape), dtype=str(v.dtype)), v.tobytes())
            if isinstance(v, np.ndarray) else tensor_bytes(v))
    owner.global_sha256 = global_sha256
    global_before = global_sha256()
    owner.initialization = {}
    for component, factory in (("generator", image_tasks.Generator), ("discriminator", image_tasks.Discriminator)):
        with owner.streams.fork("init", component=component, purpose="construction", device="cpu"), torch.device("cpu"):
            module = factory(owner.spec)
        seeds = {name: owner.streams.seed_for("init", component=component, purpose=name)
                 for name, p in module.named_parameters() if p.requires_grad and p.numel()}
        init.deterministic_orthogonal_(module, parameter_seeds=seeds)
        module.to(device=device, dtype=torch.float32)
        owner.initialization[component] = dict(initializer="deterministic_orthogonal_named_parameters_v1",
            parameter_seeds=seeds, state_sha256=typed_digest(module.state_dict(), tensor_bytes))
        setattr(owner, "generator" if component == "generator" else "critic", module)
    prior_stream = owner.streams.generator("init", component="prior", purpose="locations")
    owner.prior = recipe.make_prior(device=device, dtype=torch.float32, learnable=True, generator=prior_stream, init_std=1.)
    seeds = {name: owner.streams.seed_for("init", component="prior", purpose=name)
             for name, p in owner.prior.named_parameters() if p.requires_grad and p.numel()}
    init.deterministic_orthogonal_(owner.prior, parameter_seeds=seeds)
    owner.initialization["prior"] = dict(initializer="deterministic_orthogonal_named_parameters_v1",
        parameter_seeds=seeds, state_sha256=typed_digest(owner.prior.state_dict(), tensor_bytes),
        constructor_locations="public factory temporary locations replaced once by declared initializer")
    stream_bindings = {"latent_generator": ("prior", "latent", "indices"),
        "penalty_generator": ("noise", "penalty", "training"),
        "noise_generator": ("noise", "generator", "output"),
        "input_noise_generator": ("noise", "critic", "input"),
        "eval_generator": ("eval", "sampler", "samples"),
        "model_generator": ("noise", "models", "stochastic_layers")}
    options = {name: owner.streams.generator(family, component=component, purpose=purpose)
               for name, (family, component, purpose) in stream_bindings.items()}
    owner.data = owner.streams.generator("data", component="target", purpose="training")
    owner.trainer = GANTrainer(recipe, owner.generator, owner.critic, prior=owner.prior, seed=0,
        max_steps=600, require_latent_damping=True, **options)
    owner.policy = owner.trainer.policy
    owner.opt_g, owner.opt_d = owner.trainer.opt_g, owner.trainer.opt_d
    namespace = dict(vars(inspect.getmodule(GANTrainer)))
    exec(compile(derive_live_step(_source(root, TRAINER_PATH)), "<pinned-live-storage-public-step>", "exec"), namespace)
    owner.trainer._step = MethodType(namespace["_step"], owner.trainer)
    owner.lifecycle_audit = PolicyLifecycleAudit(owner.policy)
    owner.calls = owner.lifecycle_audit.calls
    owner.centers = image_tasks.templates(owner.spec).to(device=device, dtype=torch.float32)
    owner.observations, owner.purity, owner.retained = [], [], []
    owner.measure_namespace = dict(vars(image_tasks))
    # Existing host helper returns this device list; no process device policy changes.
    owner.measure_namespace["rng_fork_devices"] = lambda: [device.index]
    exec(compile(derive_live_measure(_source(root, HOST_PATH)), "<pinned-original-image-measure-capture>", "exec"), owner.measure_namespace)
    owner.audit = MechanismAudit(recipe, owner.opt_d, [owner.opt_g])
    owner.initial_named_rng = owner.streams.audit()
    owner.initial_global_rng = dict(before_sha256=global_before, after_sha256=global_sha256())
    if owner.initial_global_rng["before_sha256"] != owner.initial_global_rng["after_sha256"]:
        raise ValueError("construction consumed unowned global RNG")
    source_guard()
    return owner


def owner_initial_receipt(owner):
    """Actual admitted factory witness, never a premodel freshness checkbox."""
    from particlegan import GANTrainer, Recipe, ParticlePrior
    from particlegan.policy import UpdatePolicy
    from particlegan.k3p import K3PGeneratorAdam
    from particlegan.ka2 import KA2CriticAdam
    from experiments.forge.rng import NamedStreams
    from benchmarks.transfer_suite.image_tasks import Generator, Discriminator
    types = ((owner, ImageOwner), (owner.trainer, GANTrainer), (owner.policy, UpdatePolicy),
        (owner.recipe, Recipe), (owner.generator, Generator), (owner.critic, Discriminator),
        (owner.prior, ParticlePrior), (owner.opt_g, K3PGeneratorAdam), (owner.opt_d, KA2CriticAdam),
        (owner.streams, NamedStreams))
    if any(type(value) is not required for value, required in types):
        raise ValueError("actual declared public image owner types required")
    params = [p for opt in (owner.opt_g, owner.opt_d) for group in opt.param_groups for p in group["params"]]
    identities = dict(trainer_generator=owner.trainer.G is owner.generator,
        trainer_critic=owner.trainer.D is owner.critic, trainer_prior=owner.trainer.prior is owner.prior,
        trainer_policy=owner.trainer.policy is owner.policy,
        policy_generator=owner.policy.G is owner.generator, policy_critic=owner.policy.D is owner.critic,
        policy_prior=owner.policy.prior is owner.prior, policy_table=owner.policy.table is owner.prior.z,
        generator_optimizer=owner.policy.opt_g is owner.opt_g, critic_optimizer=owner.policy.opt_d is owner.opt_d,
        unique_parameter_ownership=len(params) == len({id(p) for p in params}),
            live_storage=owner.policy._fast is None)
    model_hashes = {key: typed_digest(module.state_dict(), owner.tensor_bytes)
                    for key, module in (("generator", owner.generator), ("discriminator", owner.critic), ("prior", owner.prior))}
    receipt = dict(schema=OWNER_SCHEMA, task_id=owner.binding["task_id"], completed_steps=owner.trainer.completed_steps,
        phase=owner.policy._phase, restored=owner.restored, seed=owner.streams.seed, rng_version=owner.streams.version,
        table=dict(shape=list(owner.prior.z.shape), requires_grad=owner.prior.z.requires_grad, sigma=0., standardize=False),
        optimizer_state_entries=dict(generator=len(owner.opt_g.state), discriminator=len(owner.opt_d.state)),
        optimizer_updates=dict(generator=0, prior=0, discriminator=0, noise=0),
        object_bindings=identities, initialization=deepcopy(owner.initialization), model_sha256=model_hashes,
        initial_named_rng=deepcopy(owner.initial_named_rng), global_rng=deepcopy(owner.initial_global_rng),
        resolved_recipe_sha256=digest(asdict(owner.recipe)), source_contract_sha256=owner.binding["source_contract_sha256"],
        initializer="deterministic_orthogonal_named_parameters_v1", device=str(owner.trainer.device),
        base_learning_rates=deepcopy(owner.policy.initial_lrs),
        prior_mechanisms=deepcopy(owner.trainer.prior_mechanisms))
    if (receipt["completed_steps"] != 0 or receipt["phase"] != "ready" or owner.restored
            or owner.trainer.max_steps != 600 or owner.recipe.total_steps is not None
            or owner.trainer.serial_backward
            or not all(identities.values()) or receipt["table"]["shape"] != [32, 8] or not receipt["table"]["requires_grad"]
            or any(receipt["optimizer_state_entries"].values()) or any(owner.calls.values())
            or owner.policy.controller.updates != 0 or owner.opt_d.record.observed_steps != 0
            or any(model_hashes[k] != owner.initialization[k]["state_sha256"] for k in model_hashes)
            or owner.streams.audit() != owner.initial_named_rng or receipt["resolved_recipe_sha256"] != RECIPE_SHA256
            or owner.observations or owner.retained or owner.purity):
        raise ValueError("fresh zero-update initialized image owners required")
    return receipt


def run_case(root, request, *, source_guard, fresh_repeat_guard):
    """ROOT admitted child only; public600 updates and24 existing live reads."""
    binding = resolve_binding(root, request["candidate"], request["task"], request["protocol"])
    if binding != request["binding"]:
        raise ValueError("request source/effective Recipe binding drift")
    device = request.get("runtime", {}).get("device", "cuda:1")
    owner = fresh_repeat_guard.construct(
        lambda: construct_owner(root, binding, source_guard=source_guard, device=device), owner_initial_receipt)
    losses = []
    for step in range(1, 601):
        torch = owner.torch
        indices = torch.randint(len(owner.centers), (32,), device=owner.trainer.device, generator=owner.data)
        real = (owner.centers[indices] + owner.spec["noise_std"] * torch.randn(
            (32, 1, 8, 8), device=owner.trainer.device, generator=owner.data)).clamp(0., 1.)
        last = owner.trainer.step(real, collect_stats=True)
        owner.audit.observe_penalty(last["penalty_stats"])
        if owner.trainer.completed_steps != step or owner.policy._fast is not None:
            raise ValueError("public update clock or live storage mismatch")
        if not all(bool(torch.isfinite(v).all()) for v in last.values() if isinstance(v, torch.Tensor)):
            raise FloatingPointError("nonfinite public training loss")
        if step in CLOCKS:
            source_guard()
            owner.checkpoint(step)
            losses.append(dict(step=step, **{k: float(v) for k, v in last.items() if isinstance(v, torch.Tensor)}))
    counts = dict(generator=optimizer_clock(owner.opt_g, owner.generator.parameters()),
        prior=optimizer_clock(owner.opt_g, owner.prior.parameters()),
        discriminator=optimizer_clock(owner.opt_d, owner.critic.parameters()),
        noise=optimizer_clock(owner.opt_g, [owner.policy.log_output_sigma]))
    lifecycle = owner.lifecycle_audit.receipt(owner.trainer.completed_steps)
    if (lifecycle.get("complete") is not True or lifecycle.get("observed_updates") != 600
            or any(v != 600 for v in counts.values()) or any(v != 600 for v in owner.calls.values())
            or [p["step"] for p in owner.observations] != CLOCKS or [p["step"] for p in owner.retained] != CLOCKS):
        raise ValueError("incomplete full600/24 image owner clocks")
    actual_owners = dict(dv12=owner.policy.controller, stationarity=owner.policy.lr_settle,
        row_evidence=owner.policy.row_evidence, birth_death=owner.policy.birth_death,
        averaging=owner.policy.averaged_table, learned_output_noise=owner.policy.log_output_sigma,
        automatic_features=owner.policy._feature_selection, settled_reopen=owner.policy.reopen_guard)
    if any(value is None for value in actual_owners.values()):
        raise ValueError("a requested original Atlas owner is missing")
    audit = owner.audit.receipt()
    from experiments.forge.mechanisms import mechanism_blockers
    from experiments.forge.policy_adapters import finite_policy_state
    _loaded_source(root, finite_policy_state, "experiments/forge/policy_adapters.py")
    def finite(v):
        if isinstance(v, owner.torch.Tensor): return bool(owner.torch.isfinite(v).all())
        if isinstance(v, (dict, OrderedDict)): return all(finite(x) for x in v.values())
        if isinstance(v, (list, tuple)): return all(finite(x) for x in v)
        return math.isfinite(v) if type(v) in (float, int) else True
    learned = [*owner.generator.parameters(), *owner.critic.parameters(), owner.prior.z, owner.policy.log_output_sigma]
    policy_health = finite_policy_state(owner.policy.state_dict())
    health = policy_health and finite(learned) and finite([p.grad for p in learned]) and finite([owner.opt_g.state_dict(), owner.opt_d.state_dict()])
    source_guard()
    live = deepcopy(owner.observations[-1])
    result = dict(task_id=binding["task_id"], execution_path="public_trainer", device=str(owner.trainer.device),
        raw=dict(spec=deepcopy(owner.spec), observations=deepcopy(owner.observations), live=live, losses=losses),
        evidence=dict(observations=deepcopy(owner.observations), live=live,
            scoring_weights="live", sampling_contract_version=1,
            sampling_law="enumerated_prior_without_output_noise", eval_output_noise="clean",
            measurement_purity=deepcopy(owner.purity), guards=dict(all_finite=health,
                finite_scope="learned_parameters_gradients_Adam_state_losses_observed_metrics_and_public_policy_controls",
                public_policy_all_finite=policy_health,
                policy_health_source="experiments.forge.policy_adapters.finite_policy_state",
                optimizer_updates=counts, hooks_exercised=not mechanism_blockers(audit),
                mechanism_audit=audit, unintended_rng_deviations=0)),
        applied=dict(recipe=asdict(owner.recipe), recipe_serialized=owner.recipe.to_dict(),
            source_contract=deepcopy(binding["source_contract"]), source_contract_sha256=binding["source_contract_sha256"],
            policy_owner="particlegan.UpdatePolicy", trainer_owner="particlegan.GANTrainer",
            actual_policy_owners={k: type(v).__module__ + "." + type(v).__qualname__ for k, v in actual_owners.items()},
            lifecycle_calls=deepcopy(owner.calls), lifecycle_audit=deepcopy(lifecycle),
            optimizer_updates=counts, roles=deepcopy(owner.policy.roles),
            backend_selection=owner.policy._feature_selection.state_dict(),
            prior_mechanisms=deepcopy(owner.trainer.prior_mechanisms),
            initial_learning_rates=deepcopy(owner.policy.initial_lrs),
            ordinary_observation_owner="live_generator_and_raw_prior_centers", evaluation_sampler_calls=0,
            evaluation_generator_calls=24, output_noise_applied_to_metric=False, dv12_applied_to_metric=False,
            serving_averaging_owned=True, live_storage_overlay="omit_only_final_compatibility_serve_apply",
            serial_backward=owner.trainer.serial_backward,
            initialization=deepcopy(owner.initialization), streams=owner.streams.manifest()),
        retained_goal_states=owner.retained, complete_state=owner.complete_state())
    source_guard()
    return result
