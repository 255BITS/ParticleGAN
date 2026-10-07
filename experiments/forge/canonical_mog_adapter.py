"""Original common-task MoG hosts through the documented caller-owned API.

Metadata functions import no scientific packages. Construction and the existing
Forge producer execute only in the caller's admitted, source-guarded child.
The Recipe's prior kind is its factory default; the supplied prior and its
fixed kernel are separately explicit. No Recipe control is disabled here.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace, FunctionType

SCHEMA = "pg_canonical_mog_policy_binding_v1"
OWNER_SCHEMA = "pg_canonical_mog_initial_owner_v1"
CONFIG_PATH = "configs/100gaussians/atlas.json"
PROTOCOL_PATH = "configs/forge/protocols/screening.json"
TASK_IDS = ("ring16_acquisition", "mode_hold", "vector_two_broad",
    "vector_unequal_mass", "vector_unequal_width", "vector_anisotropic",
    "vector_overlap", "vector_spiral", "grid100", "rotated100", "staggered100",
    "ring_hold", "ring_extension")
ACTUAL_PRIOR = dict(kind="mog", sigma=.025, standardize=False, learnable=True)
FACTORY_OVERRIDES = dict(prior_kind="mog", sigma=.025, standardize=False)
# Filled from the authorized copied SOURCE, never from trial evidence.
SOURCE_PINS = {'benchmarks/legacy/gan_loss.py': 'ee26f6d8ef56dde123740f67dd82f48d01f57d4acfa0f3efd299097a1e60600f', 'benchmarks/legacy/grad_regularizers.py': 'cfe252f52a6d11274e4be78bd13934fc3d919abe7d0ba349e5bd0199a45dcb5d', 'benchmarks/legacy/locked_shared.py': '9bac7def8e5c10d4d4532723ebbf7d7689d5e85c8823dbf07fb7225f91b455c0', 'benchmarks/locked_shared/baseline.py': '8f1f58fc9cb90159c555788e593769d50070deaa078d0637bf881cf074f6680b', 'benchmarks/locked_shared/mode_hold.py': '38d4b4030ddaf354d41593395f3d6a26b4acf3668e672ebe56d126b7ee1c7232', 'benchmarks/locked_shared/observation.py': 'bd6f9845b44f1ec2a58d445727990ba5068c7aca3b6f981cf38d738a37c4513b', 'benchmarks/smart_descent/controller.py': 'b2c00ade7ec77d1acba7917d8689bd29db4406fde5cf84cd67976dea274a42cf', 'benchmarks/smart_descent/evaluate.py': '7d053ee8397f13e0579baf671867a16a1207c06e2763b1c0a3e8551e9fd38407', 'benchmarks/toy100/accuracy.py': 'ee5c62049f8cdceb3d93b6d792cbaa96c815e12b3ce79ffaed244fb15a44fed2', 'benchmarks/toy100/accuracy_evidence.py': '654adde5ee19915bc9eb630d754f2a4c0282d694252df7918bae10f542904d39', 'benchmarks/toy100/accuracy_gate.py': 'e66ede95d4b80eedd832a102841c2b0a5d6d710f646bfa34dc00647dc0ab3abe', 'benchmarks/toy100/device.py': '7fa0f53db44e8824d3497b471e9fe8db56cbf8b4e02ddf997f65beed63f81911', 'benchmarks/toy100/gate.py': '2ddadf56a4ccd7645c8b4e391373818b1113fb6113a9c691310fb162684058e6', 'benchmarks/toy100/metrics.py': 'a7fbef19082f8470eb392948da52a336a90a3e99c6b3b893343a9e75bcda9c7b', 'benchmarks/toy100/models.py': '75f33ad0e100930631da2a86a9c0312f6ae02afe6c851d940c7cf44e41dede6a', 'benchmarks/toy100/problems.py': '2581374d451ed9147c031dfcdfbe9e2eea83e41ba49c2b8945b6e729ede26c9d', 'benchmarks/toy100/train.py': '64ad54128085ab684a77ecb155ae449d95ab326775e13b925cfb783e53c3f610', 'benchmarks/toy_audit/gaussian1d_quality.py': '7ec72e07c1aea87e77c85401b7e822f23d5a248ba45d718180045a1ad64ccd8b', 'benchmarks/toy_audit/ring16_quality.py': 'c666909ee8ed22ecb7a72f4185556f82b0b6d4ebd8aac871d58ed75b7ebf9db6', 'benchmarks/transfer_suite/protocol.py': '99469b022b790a18a74021a6fe49424d95f535afaa220643688a1ddd7a70ab89', 'benchmarks/transfer_suite/public_default_verification.py': '65fcb3d70cc026d36c56f7448722de46b8c01235bb0281b16b575019f1a96ace', 'benchmarks/transfer_suite/vector_tasks.py': '3ee4eb27759f61a430029c80b7772cac3dca2fb2ac84919ace0db94efca1e0b0', 'configs/100gaussians/atlas.json': 'a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4', 'configs/forge/protocols/screening.json': '3fefb4d47fd2cd8aa6ed110c0a9f5bffefaae700431d1f57ca7b162c8efbb803', 'configs/forge/tasks/grid100.json': '64709b1544155fd3e19ac1aef96453e2183145cc2be029e7da5df3bfb4b291b5', 'configs/forge/tasks/mode_hold.json': '79e11760a8851a65f3739775b5e4041bacb0a79050e8e21f52ad1eb9500435f5', 'configs/forge/tasks/ring16_acquisition.json': 'e6b53ba29fbe9ead47e842cfa01e40ba57821bd1b4e6aa5b297631fa0f6525c1', 'configs/forge/tasks/ring_extension.json': '19361e1a0817f90a7fe13c1c4b35fff644745258063e87371099936df7461f50', 'configs/forge/tasks/ring_hold.json': '7d3e33f19db943f6f55016b9cbf2fe93221cc3e202c4c86190d5ff981b1fdfc9', 'configs/forge/tasks/rotated100.json': 'f8691aa4152885a34dc4e32eda6bbbc5d7df2699cc7b582f9bc25c091ea9c35d', 'configs/forge/tasks/staggered100.json': '5c39cf3faf007d146f861b9962965b92ec17bcf7d9bd906aa9f1f0d21c82d4a1', 'configs/forge/tasks/vector_anisotropic.json': 'e98512a93ba2d3a10a5fe6640cf74c5d22f88f917e9a5cf313302cc4aed1a24b', 'configs/forge/tasks/vector_overlap.json': '72bde8ab5f9a5222b095ee17849c27823156bd28675f443138d56542c9e69c35', 'configs/forge/tasks/vector_spiral.json': 'd47cf689d77375873c25b5d6aa3610e9d6250ee5a9d2f2592538d49da943ff9f', 'configs/forge/tasks/vector_two_broad.json': '7239390e9da217364f4142d0a90afd60e75317cd5fe362efb9b0e68d0e1bccab', 'configs/forge/tasks/vector_unequal_mass.json': 'ea1b35df2bd0ab55b50d47ade00a758775d29130641fc1e76db28c03009164ff', 'configs/forge/tasks/vector_unequal_width.json': 'e3f18fe65e198563908aa1b2b68a76bf46398427037cddffef4b42e0cd3d0c8f', 'experiments/forge/adapters.py': 'e2902ef9f39f39067a8d275616f78bb07e672af69c562833bc1e7ffa681d08c7', 'experiments/forge/api.py': 'e745bfc9f7246055e4d3cea0b3943493ccdf3769527a54684182514a6dc767cd', 'experiments/forge/artifacts.py': '78da4ed4b4038b8a93918aa4b5de7fa6f7e4d36731b69467e1cd8b7edcd7ee1e', 'experiments/forge/contracts.py': '0b197620cfe44144d1b9a4a7eaf0320f31c207488fd5e4ba0002727e26af3dfc', 'experiments/forge/initialization.py': 'c2674617c306e24337f1dbbf910d2110b7c852557b9c3e43b47f7fe1dce26634', 'experiments/forge/mechanisms.py': '9e4e1fe1f8cb9a24148cab88b7bf2f980e805a2e488c16a982768519bbe1b23f', 'experiments/forge/nativeprofiles.py': '18947c346bda0b14e9cddb25dbb0c1b5ba0556e0edd77df24c195988db9a1025', 'experiments/forge/paired_sampling.py': '4dd4e7acd422388be738689155c2d6f6d39a68a9cfd6a6bc03b37f08b9ac6064', 'experiments/forge/planning.py': 'f4df749ddc012ccc1bd2bdf35cacfc19e4e420293b83b4f093899831bb6ebb9c', 'experiments/forge/priors.py': '874466861600410f361fce6887546e6624d6d88b9884a821cb3bf88482106c2f', 'experiments/forge/rng.py': 'ae7b9a8d61da42136d970188a6f168e03e2d7c5b90cf6fdbd93ad929aba2f293', 'experiments/forge/sampling.py': '10e7c17dd4ca30c1f2f3797aa323690db7f4bfdbfe295f23770bb6293ba3aab1', 'experiments/forge/state.py': '2eb9613f8e85c63240499307a81c5af429b23cfa1f0ed08c126630a76ed887b5', 'experiments/forge/taskrecipes.py': 'c11739bd935256e7083875f4e67d6c29fad1b9ea5915976dd32ed1eb7531d285', 'experiments/forge/telemetry.py': '65f444bb1f643fec121144cbf1dce8574531d99c85b80cbd4089ba631ff4831a', 'experiments/forge/vectorprofiles.py': 'dcbbf669a3457126fce82d073f8d8e4e5254702dca5201f88ea5be7877ee14da', 'experiments/forge/views.py': '673331e6fd93e56d69db83f034fa2e431ee5f2015f368c7de8029009da739e77', 'lib/toy_metrics.py': '12e8975fe31ed4d9973d53649ce940171da78739a3d7b80544ba5f4d2313367c', 'lib/toy_models.py': '70f32797882de0dc88085afcf5e05821b65eb5f248c9690818a2494225d782a3', 'particlegan/__init__.py': 'b56efa562acc551d639548c299002e2cd76338b3fd7062699780add51f4f5e8c', 'particlegan/_qr.py': 'b59cd62cc27ed557b41b2d17ed86ed578c762a3a8a392d63731a5a3b2b018916', 'particlegan/anchor_birth.py': 'aa50983d8b61304acf4e1138f16fd4784798983df91e931863bcacd3b511f76f', 'particlegan/autoencoder.py': 'f3fbb9184e9f44112e15eccf0c0246dfba5ed490d3c7b81b0f837a5335916402', 'particlegan/birth_death.py': 'b14c50c611a8cf188347e391739fca50a5171200eb6be90e4e3d1509d64e47ed', 'particlegan/birth_phase.py': '5ff929e5e9858aa24c4cc2f8999b5d0603e25e42ca3844d1da4cc28b1981d3d7', 'particlegan/capabilities.py': '39df9c67dbedf612740057e5fa743bc724678a1bc323d1f7cf550b37c0c01177', 'particlegan/conditioning.py': '8734fe338868ace4b5851a8fdb1f071e46d47396f66a3765e6b1ef37f0481486', 'particlegan/continuous.py': '4ceab49c7d51d1769ae91b7f8bc892eaf380ca6fbd77a948a086ade6627e9a41', 'particlegan/diffusion.py': '19c3aa8772703891551b9122fde8e74c07996f779892b7cb6be09553691efa61', 'particlegan/discriminators.py': '0e2efb125ffb314577612ab7a2eba66b0a1a2d28ad25f403f42c18b6f6ee333f', 'particlegan/feature_cells.py': 'e0ea05f61abd7845c7437eed9a79c3a2511d5551ff04f4e619b6b9c8992e221a', 'particlegan/feature_policy.py': 'de1a50f70a37d4c9174c8850ce3eac2a2597ed01f19fc4fe1a2ce0ec3176a8b2', 'particlegan/feature_reference.py': '58e076e9e04600be25e28afafc537a340de727371f4acbbfb2b7dd8bebac18b1', 'particlegan/gan_loss.py': '1c1019dfe71c583e32a05df0d2f794f9fff6d9ee1ea0332f57f6379ae70cf6b7', 'particlegan/grad_regularizers.py': 'a540d05a4a992540b6a5f3b5ff7ccc11ebaf18592135b87c95638adbe68aa747', 'particlegan/init.py': 'e15d4de01eb49f7097bc249abacab907ff067385412346f54ab0d510cb8649d0', 'particlegan/k3p.py': '200ead2c27ab0b2aa6068f1b1b1b5c826b8125aa8602bfec08b31aaa2894401c', 'particlegan/ka2.py': '95fdaa58a2bdc229d5d146ef89548bc3fe07582d06181a149ed1bb4f5d632703', 'particlegan/mean_transport.py': '57a68e0a65c5424a4460f65002f3862a69e90af330a4d47c700dc773683a2c06', 'particlegan/output_moments.py': 'dcf3e27228d2c3c9738e473c5b22ee9e6d5047ddcd60c538b66526adb53ef188', 'particlegan/particle_prior.py': '0220878bebea227da63abbf9f5b1ebdabdc6aea54fd2332fd2cab02ed734238f', 'particlegan/policy.py': '8370e36b5b93afe95ae24d2b385aac8735a587f86ef2b87ed369ecfdfcb6fec5', 'particlegan/population_continuity.py': '378cd75cca45bc8da4da33e686a12cf99df853042687978314e37c532bc17fe6', 'particlegan/recipe_schedules.py': 'd05e459b3e9e5f938b01a854273b74343f2b6d382bec4ae5151dd1f1cd032500', 'particlegan/recipes.py': '1a7f9df746e242a2774068c819d70d50cd72dc4f7f6b22b0bd13e1345dcf8ab2', 'particlegan/routing.py': '63249d808fccb3d4e113eb1d638a245ee665c3cade20a1542bb706f316b4abcc', 'particlegan/row_evidence.py': '78be40141ef3946860458e9842159be9321f05c217a4800dd4134113b6eba93e', 'particlegan/training.py': '7dadc219135bc3e6bf657bfe9bc3fa81dd14d4b2a31c70d5ff61e2941db35c94', 'particlegan/vicreg_loss.py': 'ab1c4dc266dec2c35337f449917240eb38afede7590d2154b63f6f471dc45a36', 'reports/toy100/gap-fill-20260925/sources/k3p/convergence_gate.py': 'a29c8c21083195c2944a2b2509ec7d4b96e6fc1eb7c3fa6df70d51dea6fe1ce0', 'experiments/forge/policy_adapters.py': '714046ea1655f1946e09dc2aeed27c43cf93f2841237c2be47f0f0049a5df9fa'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def typed_digest(value, tensor_bytes):
    """Hash complete typed state, preserving opaque policy sentinel bytes."""
    h = hashlib.sha256(b"canonical-MoG-owned-state-v1\0")
    def add(tag, raw):
        h.update(tag + len(raw).to_bytes(8, "big") + raw)
    def visit(item):
        tensor = tensor_bytes(item)
        if tensor is not None:
            description, raw = tensor
            add(b"tensor", canonical(description).encode()); add(b"bytes", raw)
        elif type(item) in (dict, OrderedDict):
            add(b"ordered" if type(item) is OrderedDict else b"mapping", str(len(item)).encode())
            for key, child in item.items(): visit(key); visit(child)
            if type(item) is OrderedDict: visit(getattr(item, "_metadata", None))
        elif type(item) in (list, tuple):
            add(b"list" if type(item) is list else b"tuple", str(len(item)).encode())
            for child in item: visit(child)
        elif item is None: add(b"none", b"")
        elif type(item) is bool: add(b"bool", b"1" if item else b"0")
        elif type(item) is int: add(b"int", str(item).encode())
        elif type(item) is float: add(b"float", struct.pack("!d", item))
        elif type(item) is str: add(b"str", item.encode())
        else: raise TypeError("unknown policy state leaf: " + type(item).__name__)
    visit(value)
    return h.hexdigest()


def compile_receipt(raw):
    """Reuse current strict policy health in the original-law receipt branch."""
    cls = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == "_Run")
    node = deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "receipt"))
    marker = ast.parse("self.finite = self.finite and _finite_tree(state)").body[0]
    class HealthBoundary(ast.NodeTransformer):
        matches = 0
        def visit_Assign(self, current):
            if ast.dump(current) == ast.dump(marker):
                self.matches += 1
                return ast.parse("self.finite = self.finite and _finite_learned_state(trainer)").body[0]
            return self.generic_visit(current)
    boundary = HealthBoundary()
    node = boundary.visit(node)
    if boundary.matches != 1:
        raise ValueError("original state-health boundary changed")
    return ast.fix_missing_locations(node)


def _source(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if path.resolve() != path:
        raise ValueError("foreign source alias: " + relative)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_PINS[relative]:
        raise ValueError("source drift: " + relative)
    return raw


def _declaration(task):
    value = deepcopy(task)
    if "preflight_blockers" in value and value.pop("preflight_blockers") != []:
        raise ValueError("blocked or malformed compiled task")
    if "field_ownership" in value:
        ownership = value.pop("field_ownership")
        required = {"schema_version", "version", "task_id", "recipe_fields", "task_contract",
                    "delegated_reference_values", "inactive_legacy_host_fields", "reference_declarations"}
        if (not isinstance(ownership, dict) or not required <= ownership.keys()
                or ownership.keys() - required - {"protocol"}
                or type(ownership["schema_version"]) is not int or ownership["schema_version"] != 1
                or ownership["version"] != "forge-field-boundaries-v2"
                or ownership["task_id"] != task.get("id")
                or any(not isinstance(ownership[key], dict) for key in required - {
                    "schema_version", "version", "task_id"})
                or ("protocol" in ownership and not isinstance(ownership["protocol"], dict))):
            raise ValueError("malformed task-specific typed ownership annotation")
        canonical(ownership)
    return value


def _recipe_fields(raw):
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == "Recipe")
    return {n.target.id: ast.literal_eval(n.value) for n in node.body
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}


def serialized_recipe(fields, raw):
    """Execute only the pinned metadata serializer on an inert field view."""
    cls = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == "Recipe")
    method = deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "to_dict"))
    scope = dict(asdict=lambda obj: deepcopy(vars(obj)))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 "<pinned-Recipe-metadata-serializer>", "exec"), scope)
    return json.loads(canonical(scope["to_dict"](SimpleNamespace(**fields))))


def _resources(root, task):
    adapter, execution = task["adapter"], task["execution"]
    if adapter == "transfer_vector":
        spec = execution["host_definition"]
        return dict(num_particles=spec["particles"], z_dim=spec["z_dim"], batch_size=spec["batch"])
    if adapter == "native100":
        return deepcopy(execution["resources"])
    cls = next(n for n in ast.parse(_source(root, "benchmarks/legacy/locked_shared.py")).body
               if isinstance(n, ast.ClassDef) and n.name == "LockedShared")
    count = next(ast.literal_eval(n.value) for n in cls.body
                 if isinstance(n, ast.AnnAssign) and n.target.id == "n_particles")
    return dict(num_particles=count, z_dim=4, batch_size=128)


def observation_contract(task):
    """Declare original score clocks and a bounded subset of existing draws."""
    horizon = task["execution"]["steps"]
    if task["adapter"] == "ring_endurance":
        start = task["evaluation"]["start_step"] + 1
        goals = sorted({start + math.ceil(i * (horizon - start) / 7) for i in range(8)})
        return dict(kind="dense_first_convergence_hold_extension", first_step=start,
            last_step="actual stopped public clock, at most7500", observation_steps=None,
            consecutive=True, goal_steps=goals, retain_actual_final_observation=True,
            max_retained_goal_states=9, final_checks="original hold and immediate300 extension",
            all_original_numeric_observations_retained=True)
    if task["adapter"] == "native100":
        evaluation = task["evaluation"]
        clocks = sorted(set(evaluation["early_eval_steps"]) |
                        set(range(evaluation["eval_interval"], horizon + 1, evaluation["eval_interval"])))
        kind, holdout = "native_coverage_and_accuracy", evaluation["holdout_samples"]
    else:
        clocks = sorted({math.ceil(i * horizon / 24) for i in range(1, 25)})
        kind, holdout = "ordinary_scheduled_live", None
    goals = [clocks[math.floor(i * (len(clocks) - 1) / 8)] for i in range(9)]
    return dict(kind=kind, observation_steps=clocks, observation_count=len(clocks),
        final_checks=5, holdout_samples=holdout, goal_steps=goals,
        retain_actual_final_observation=False, max_retained_goal_states=9,
        all_original_numeric_observations_retained=True)


def execution_group_contract(root, task):
    if task["adapter"] != "ring_endurance":
        return None
    members = {name: json.loads(_source(root, f"configs/forge/tasks/{name}.json"))
               for name in ("ring_hold", "ring_extension")}
    return dict(id="ring_endurance", producer_task_id="ring_hold",
        task_ids=list(members), task_definitions=members,
        task_sha256={name: digest(value) for name, value in members.items()},
        physical_attempts=1, factory_calls=1,
        allowance_seconds=max(row["resources"]["timeout_seconds"] for row in members.values()),
        allowance_rule="maintained planning max member timeout; never sum member timeouts",
        prerequisite=dict(task="mode_hold", status="source-bound certified own current PASS required"),
        independent_extension_replay=False, checkpoint_restore=False,
        grade_projection="both original tasks from the same dense evidence and uninterrupted run_id; dependency reducer unchanged")


def resolve_binding(root, candidate, task, protocol):
    """Resolve an exact original declaration with zero construction or RNG."""
    tid = task.get("id")
    if tid not in TASK_IDS:
        raise ValueError("only the original thirteen MoG hosts are supported")
    source_task = json.loads(_source(root, f"configs/forge/tasks/{tid}.json"))
    reference = json.loads(_source(root, CONFIG_PATH))
    if canonical(_declaration(task)) != canonical(source_task):
        raise ValueError("canonical task, gates, resources or dependencies changed")
    if (candidate.get("recipe_preset") != "atlas"
            or canonical(candidate.get("recipe_overrides")) != canonical(reference)
            or candidate.get("extensions", {}) != {}
            or candidate.get("initializer", "deterministic_orthogonal") != "deterministic_orthogonal"
            or type(candidate.get("seed", 0)) is not int or candidate.get("seed", 0) != 0):
        raise ValueError("full original Atlas declaration required; no tuning")
    base_protocol = deepcopy(protocol)
    if "scientific_repeat" in base_protocol:
        repeat = base_protocol.pop("scientific_repeat")
        if not isinstance(repeat, dict):
            raise ValueError("malformed recognized repeat intent")
    if canonical(base_protocol) != canonical(json.loads(_source(root, PROTOCOL_PATH))):
        raise ValueError("exact ordinary screening protocol, seed0 and named RNG required")
    if source_task["execution"]["prior"] != ACTUAL_PRIOR:
        raise ValueError("the actual task prior must remain nonstandardized fixed-width MoG")
    if source_task["evaluation"]["scoring_weights"] != "live" or any(
            source_task["evaluation"].get(k) != v for k, v in dict(
                sampling_contract_version=1, sampling_law="public_prior_without_output_noise",
                eval_output_noise="clean").items()):
        raise ValueError("ordinary public clean/live observation law required")
    source_files = {relative: dict(sha256=expected, bytes=len(_source(root, relative)))
                    for relative, expected in SOURCE_PINS.items()}
    recipe_raw = _source(root, "particlegan/recipes.py")
    fields = _recipe_fields(recipe_raw)
    fields.update(name="atlas", **reference)
    fields.update(**_resources(root, source_task), sigma_rel=0., standardize=False)
    if fields["reg_arm"] is not None:
        fields["critic_formulation"] = "k3p"
    fields = json.loads(canonical(fields))
    if (fields["total_steps"] is not None or fields["continuous_policy"] != "dv12"
            or fields["prior_kind"] != "particles" or not fields["particle_birth_death"]):
        raise ValueError("requested original continuous Atlas controls must remain intact")
    producer = "_vector" if source_task["adapter"] == "transfer_vector" else (
        "_native" if source_task["adapter"] == "native100" else "_ring")
    contract = dict(schema="pg_canonical_mog_source_contract_v1", task_id=tid,
        task=deepcopy(source_task), full_recipe=deepcopy(fields), source_pins=deepcopy(SOURCE_PINS),
        files=source_files,
        public_prior_factory_default=fields["prior_kind"], supplied_prior=deepcopy(ACTUAL_PRIOR),
        local_prior_factory_overrides=deepcopy(FACTORY_OVERRIDES),
        recipe_mutation_for_prior_kind=False, producer=producer,
        update_body="particlegan.training.GANTrainer._step",
        storage="caller_owned_fast; public finish_step still updates serving averages",
        sampling=dict(method="particlegan.training.GANTrainer.sample",
            weights="live", output_noise=False, latent_mog_kernel_sigma=.025,
            latent_policy_perturbation="actual public DV12; not removed by output_noise=False",
            covariance_is_not_fixed_to_prior_kernel=True),
        intrinsic_horizon=None, external_horizon=source_task["execution"]["steps"],
        dependencies=deepcopy(source_task["dependencies"]),
        execution_group=source_task["execution"].get("execution_group"),
        endurance_state_law="original uninterrupted ring_hold plus extension; never a fresh extension replay"
            if source_task["adapter"] == "ring_endurance" else None,
        observation=observation_contract(source_task),
        grouped_execution=execution_group_contract(root, source_task),
        completed_steps_path="complete_state.trainer.completed_steps",
        harness_overlays=dict(health_scope="current maintained finite_policy_state for complete public policy plus actual learned parameters/buffers/gradients/Adam; unchanged loss and metric checks",
            purity_hash="typed full checkpoint including policy sentinel bytes; no sentinel removal",
            scratch_probe="maintained MechanismAudit; upstream AMSGrad scratch repair used unchanged",
            lifecycle_observer="maintained PolicyLifecycleAudit; original public hooks and their successful order",
            selected_cloud_cohort=False),
        historical_applicability="PR223 caller-owned prior override is documented; current public A2 capability is separately source-bound",
        historical_results_are_credit=False)
    return dict(schema=SCHEMA, task_id=tid, recipe=fields,
        recipe_serialized=serialized_recipe(fields, recipe_raw),
        actual_prior=deepcopy(ACTUAL_PRIOR), prior_factory_overrides=deepcopy(FACTORY_OVERRIDES),
        task_sha256=digest(source_task), config_sha256=SOURCE_PINS[CONFIG_PATH],
        protocol_sha256=digest(base_protocol),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_contract=contract, source_contract_sha256=digest(contract))


def _check_binding(root, binding):
    tid = binding.get("task_id")
    if tid not in TASK_IDS:
        raise ValueError("unknown owner task")
    expected = resolve_binding(root, dict(recipe_preset="atlas",
        recipe_overrides=json.loads(_source(root, CONFIG_PATH))),
        json.loads(_source(root, f"configs/forge/tasks/{tid}.json")),
        json.loads(_source(root, PROTOCOL_PATH)))
    if canonical(binding) != canonical(expected):
        raise ValueError("forged or stale full owner binding")
    return expected


class MogOwner:
    """Actual supplied components; a receipt alone cannot construct this owner."""


def _trainer_class():
    """Reuse public updates/checkpoints/sampling in a caller-owned container.

    Its explicit supplied-prior constructor replaces no existing GANTrainer
    guard. The inherited scientific methods are unchanged. Only the legacy
    compatibility storage swap is inactive, as documented for external loops.
    """
    import torch
    from particlegan.training import GANTrainer, InputNoise
    from .api import require_legacy_autograd_source
    require_legacy_autograd_source(GANTrainer, SOURCE_PINS["particlegan/training.py"],
                                 owner="Original common26 supplied-MoG trainer")
    from particlegan import UpdatePolicy

    class SuppliedMogTrainer(GANTrainer):
        def __init__(self, context, generator, critic, prior):
            self.recipe, self.G, self.D, self.prior = context.recipe, generator, critic, prior
            if (self.recipe.row_policy != "independent" or self.recipe.model != "gan"
                    or self.recipe.conditioning != "scalar" or self.recipe.encoder_mode != "none"
                    or not any(p.requires_grad for p in generator.parameters())
                    or not any(p.requires_grad for p in critic.parameters())):
                raise ValueError("the supplied owner requires the original unconditional trainable host")
            self.device, self.dtype = context.device, next(generator.parameters()).dtype
            self.max_steps, self.serial_backward = context.external_horizon, False
            self.optimizer_options, self.penalty_options = {}, {}
            seen = set()
            for model in (generator, critic, prior):
                for tensor in (*model.parameters(), *model.buffers()):
                    if tensor.device != self.device or (tensor.requires_grad and tensor.is_floating_point()
                            and tensor.dtype != self.dtype):
                        raise ValueError("caller-owned components must share device/trainable dtype")
                for parameter in model.parameters():
                    if id(parameter) in seen:
                        raise ValueError("caller-owned parameter ownership overlaps")
                    seen.add(id(parameter))
            self.opt_g, self.opt_d = self.recipe.make_optimizers(generator, critic, prior,
                ema_critic=deepcopy(critic), require_latent_damping=True)
            self.prior_mechanisms = self.opt_g.prior_mechanisms
            self.loss = self.recipe.make_loss()
            self.prior_regularizer = self.recipe.make_prior_regularizer(weight=1.)
            self.penalty = self.recipe.make_critic_penalty(self.opt_d)
            from experiments.forge.api import TRAINER_STREAM_BINDINGS
            streams = {name: context.streams.generator(family, component=component, purpose=purpose)
                for name, (family, component, purpose) in TRAINER_STREAM_BINDINGS.items()}
            if len({id(v) for v in streams.values()}) != len(streams):
                raise ValueError("distinct ordinary named streams required")
            self.policy = UpdatePolicy(self.recipe, generator, critic, prior=prior,
                generator_optimizer=self.opt_g, critic_optimizer=self.opt_d,
                seed=context.streams.seed, penalty=self.penalty,
                streams={name: streams[name] for name in UpdatePolicy._STREAMS})
            self._STREAMS = GANTrainer._STREAMS + ("input_noise_generator", "prior_noise_generator", "model_generator")
            self.input_noise_generator = self._stream(streams["input_noise_generator"], 0)
            self.prior_noise_generator = self._stream(streams["prior_noise_generator"], 0)
            self.model_generator = self._stream(streams["model_generator"], 0)
            global_stream = (torch.default_generator if self.device.type == "cpu"
                             else torch.cuda.default_generators[self.device.index])
            if self.model_generator is global_stream or any(value is global_stream for value in streams.values()):
                raise ValueError("named training/evaluation streams cannot alias global draws")
            self._noisy_D = InputNoise(critic, 0., self.input_noise_generator)

        def _serve_apply(self):
            # Caller-owned loops retain fast storage; policy serving still exists.
            return None

    return SuppliedMogTrainer


class MogContext:
    """Ordinary named construction/initialization around a supplied MoG owner."""
    def __init__(self, recipe, binding, device, streams):
        self.recipe, self.binding, self.device, self.streams = recipe, binding, device, streams
        self.recipe_preset, self.execution_path = "atlas", "public_components"
        # The current _Run has a separate selected-cloud branch. This owner
        # executes the original canonical task and live public sampling law.
        self.policy_task = None
        self.prior_config, self.initializer = deepcopy(ACTUAL_PRIOR), "deterministic_orthogonal"
        self.initialization, self._trainer = {}, None
        self.external_horizon = binding["source_contract"]["external_horizon"]

    def construct(self, factory, *, component):
        with self.streams.fork("init", component=component, purpose="construction", device="cpu"):
            return factory()

    def initialize(self, model, *, component):
        from particlegan import init
        from experiments.forge.state import state_digest
        seeds = {name: self.streams.seed_for("init", component=component, purpose=name)
                 for name, p in model.named_parameters() if p.requires_grad and p.numel()}
        init.deterministic_orthogonal_(model, parameter_seeds=seeds)
        self.initialization[component] = dict(initializer="deterministic_orthogonal_named_parameters_v1",
            parameter_seeds=seeds, state_sha256=state_digest(model.state_dict()))
        return model

    def receipt(self):
        from particlegan import prior_capabilities
        trainer = self._trainer
        return dict(api_version="forge-api-v1", execution_path=self.execution_path,
            recipe_preset="atlas", recipe=self.recipe.to_dict(), prior=deepcopy(ACTUAL_PRIOR),
            resolved_recipe=deepcopy(self.binding["recipe"]),
            capabilities=dict(checkpoint=True, named_rng=True, live_sampling=True, learned_locations=True,
                mog_prior=True, uniform_masses=True, fixed_prior_width=True, a2=True,
                policy_controls=True, policy_serving=True, public_components=True),
            requires_capabilities=deepcopy(self.binding["source_contract"]["task"]["requires_capabilities"]),
            extensions={}, api_changes=[], initializer=self.initializer,
            initialization=deepcopy(self.initialization), rng=self.streams.manifest(),
            prior_mechanisms=deepcopy(trainer.prior_mechanisms),
            actual_prior_capabilities=deepcopy(prior_capabilities(trainer.prior)),
            policy_lifecycle=dict(owner="particlegan.UpdatePolicy", completed_steps=trainer.completed_steps,
                external_max_steps=trainer.max_steps, continuous_policy=self.recipe.continuous_policy,
                lr_control=self.recipe.lr_control, row_policy=self.recipe.row_policy,
                requested_serving=self.recipe.serve_average, observation_weights="fast_live",
                backend_selection=None if trainer.policy._feature_selection is None else
                    deepcopy(trainer.policy._feature_selection.state_dict()), quality_qualification=False),
            original_mog_binding=deepcopy(self.binding))

    def state_dict(self):
        return dict(schema=1, api_version="forge-api-v1", recipe=self.recipe.to_dict(),
            prior=deepcopy(ACTUAL_PRIOR), extensions={}, initializer=self.initializer,
            initialization=deepcopy(self.initialization), streams=self.streams.state_dict(),
            trainer=self._trainer.state_dict())


def construct_owner(root, binding, *, device, source_guard):
    """Only the admitted child may invoke this actual public component factory."""
    source_guard()
    from particlegan.training import GANTrainer
    from .api import require_legacy_autograd_source
    require_legacy_autograd_source(GANTrainer, SOURCE_PINS["particlegan/training.py"],
                                 owner="Original common26 MoG owner")
    _check_binding(root, binding)
    import torch
    from particlegan import Recipe, MoGParticlePrior, prior_capabilities
    from particlegan.k3p import K3PGeneratorAdam
    from particlegan.ka2 import KA2CriticAdam
    from experiments.forge.rng import NamedStreams
    from experiments.forge.policy_adapters import PolicyLifecycleAudit
    from experiments.forge import adapters
    from experiments.forge.vectorprofiles import build_vector_models, resolve_vector_spec
    from experiments.forge.state import state_digest
    source_guard()
    if torch.get_num_threads() != 1 or torch.get_default_dtype() != torch.float32:
        raise ValueError("original float32 CPU1 runtime required")
    rng_before = state_digest(dict(cpu=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None))
    recipe = Recipe(**binding["recipe"])
    if canonical(asdict(recipe)) != canonical(binding["recipe"]) or (
            canonical(recipe.to_dict()) != canonical(binding["recipe_serialized"])):
        raise ValueError("actual complete public Recipe differs before model construction")
    device = torch.device(device)
    context = MogContext(recipe, deepcopy(binding), device, NamedStreams(0, device=device))
    # Match the ordinary registry's fixed names, including unused streams.
    for family, component, purpose in (("init", "prior", "locations"), ("data", "target", "training"),
        ("prior", "latent", "indices"), ("noise", "penalty", "training"),
        ("noise", "generator", "output"), ("noise", "critic", "input"),
        ("noise", "models", "stochastic_layers"), ("noise", "prior", "gaussian"),
        ("eval", "sampler", "samples"), ("eval", "target", "reference")):
        context.streams.generator(family, component=component, purpose=purpose)
    for component in ("generator", "discriminator"):
        context.streams.generator("init", component=component, purpose="construction", device="cpu")
    task = binding["source_contract"]["task"]
    if task["adapter"] == "transfer_vector":
        generator, critic = build_vector_models(context, resolve_vector_spec(task, root=root))
    else:
        spec = task["execution"].get("model", dict(hidden=96, layers=3, fourier=3))
        generator, critic = adapters._models(context, spec)
    context.initialize(generator, component="generator")
    context.initialize(critic, component="discriminator")
    prior = recipe.make_prior(**binding["prior_factory_overrides"], learnable=True,
        generator=context.streams.generator("init", component="prior", purpose="locations"),
        device=device, dtype=next(generator.parameters()).dtype, init_std=1.)
    context.initialize(prior, component="prior")
    if (type(prior) is not MoGParticlePrior or prior.standardize or not prior.z.requires_grad
            or list(prior.z.shape) != [recipe.num_particles, recipe.z_dim]
            or float(prior.sigma.item()) != float(torch.tensor(.025, dtype=prior.sigma.dtype).item())
            or not prior_capabilities(prior)["a2_eligible"]):
        raise ValueError("actual supplied prior width, shape, kind or capability drift")
    owner = MogOwner()
    owner.context, owner.recipe, owner.binding = context, recipe, deepcopy(binding)
    owner.generator, owner.critic, owner.prior, owner.table = generator, critic, prior, prior.z
    owner.streams, owner.restored, owner.construction_id = context.streams, False, object()
    owner.trainer_class = _trainer_class()
    owner.trainer = owner.trainer_class(context, generator, critic, prior)
    context._trainer = owner.trainer
    owner.policy, owner.opt_g, owner.opt_d = owner.trainer.policy, owner.trainer.opt_g, owner.trainer.opt_d
    if type(owner.opt_g) is not K3PGeneratorAdam or type(owner.opt_d) is not KA2CriticAdam:
        raise ValueError("actual public original optimizer types required")
    owner.initialization_digest = state_digest(dict(generator=generator.state_dict(),
        critic=critic.state_dict(), prior=prior.state_dict()))
    owner.lifecycle_audit = PolicyLifecycleAudit(owner.policy)
    owner.calls = owner.lifecycle_audit.calls
    owner.observation_purity = []
    owner.retained_goal_states = []
    def tensor_bytes(value):
        if not isinstance(value, torch.Tensor):
            return None
        data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
                    requires_grad=value.requires_grad), data
    owner.tensor_bytes = tensor_bytes
    owner.initial_named_rng = owner.streams.audit()
    owner.initial_global_rng = dict(before_sha256=rng_before,
        after_sha256=state_digest(dict(cpu=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)))
    if owner.initial_global_rng["before_sha256"] != owner.initial_global_rng["after_sha256"]:
        raise ValueError("construction consumed unowned Torch global RNG")
    source_guard()
    owner.initial_receipt = owner_initial_receipt(owner)
    return owner


def owner_initial_receipt(owner):
    """Measure the actual source-bound public owner before its first forward."""
    import torch
    from particlegan import Recipe, UpdatePolicy, MoGParticlePrior, prior_capabilities
    from particlegan.k3p import K3PGeneratorAdam
    from particlegan.ka2 import KA2CriticAdam
    from experiments.forge.state import state_digest
    from experiments.forge.rng import NamedStreams
    from lib.toy_models import SimpleMLPGenerator, SimpleMLPDiscriminator
    if (type(owner) is not MogOwner or type(owner.trainer) is not owner.trainer_class
            or type(owner.recipe) is not Recipe or type(owner.policy) is not UpdatePolicy
            or type(owner.prior) is not MoGParticlePrior
            or type(owner.generator) is not SimpleMLPGenerator
            or type(owner.critic) is not SimpleMLPDiscriminator
            or type(owner.streams) is not NamedStreams
            or type(owner.opt_g) is not K3PGeneratorAdam or type(owner.opt_d) is not KA2CriticAdam):
        raise TypeError("receipt requires the actual new public component owner")
    parameters = [p for optimizer in (owner.opt_g, owner.opt_d)
                  for group in optimizer.param_groups for p in group["params"]]
    module_parameters = {id(p) for module in (owner.generator, owner.critic, owner.prior)
                         for p in module.parameters() if p.requires_grad}
    if owner.policy.log_output_sigma is not None:
        module_parameters.add(id(owner.policy.log_output_sigma))
    bindings = dict(trainer_generator=owner.trainer.G is owner.generator,
        trainer_critic=owner.trainer.D is owner.critic, trainer_prior=owner.trainer.prior is owner.prior,
        trainer_policy=owner.trainer.policy is owner.policy,
        policy_generator=owner.policy.G is owner.generator,
        policy_critic=owner.policy.D is owner.critic, policy_prior=owner.policy.prior is owner.prior,
        policy_table=owner.policy.table is owner.table,
        table_optimizer=owner.policy.table_optimizer is owner.opt_g,
        unique_parameter_ownership=len({id(p) for p in parameters}) == len(parameters)
            and {id(p) for p in parameters} == module_parameters,
        live_storage=owner.policy._fast is None,
        separate_averages=owner.policy.ema_G is not owner.generator and owner.policy.ema_prior is not owner.prior)
    current = state_digest(dict(generator=owner.generator.state_dict(), critic=owner.critic.state_dict(),
                               prior=owner.prior.state_dict()))
    models = {name: state_digest(module.state_dict()) for name, module in
              (("generator", owner.generator), ("discriminator", owner.critic), ("prior", owner.prior))}
    counts = dict(policy=owner.policy.completed_steps, controller=owner.policy.controller.updates,
                  critic_record=owner.opt_d.record.observed_steps,
                  lifecycle_order_errors=owner.lifecycle_audit.order_errors, **owner.calls)
    if (owner.restored is not False or any(counts.values()) or owner.policy._phase != "ready"
            or owner.opt_g.state or owner.opt_d.state or owner.lifecycle_audit.pending
            or owner.observation_purity or owner.retained_goal_states
            or not all(bindings.values()) or current != owner.initialization_digest):
        raise ValueError("actual fresh initialization, empty optimizers and zero lifecycle clocks required")
    if (owner.streams.audit() != owner.initial_named_rng
            or any(models[name] != owner.context.initialization[name]["state_sha256"] for name in models)):
        raise ValueError("actual initialized weights or named initial RNG changed before first forward")
    return dict(schema=OWNER_SCHEMA, task_id=owner.binding["task_id"], completed_steps=0, phase="ready",
        restored=False, seed=owner.streams.seed, rng_version=owner.streams.version,
        table=dict(shape=list(owner.table.shape), requires_grad=owner.table.requires_grad,
            sigma=float(owner.prior.sigma.item()), standardize=owner.prior.standardize),
        actual_prior=deepcopy(ACTUAL_PRIOR), actual_prior_capabilities=prior_capabilities(owner.prior),
        optimizer_state_entries=dict(generator=len(owner.opt_g.state), discriminator=len(owner.opt_d.state)),
        optimizer_updates=dict(generator=0, prior=0, discriminator=0, noise=0),
        zero_clocks=counts, object_bindings=bindings, initialization_sha256=current,
        initialization=deepcopy(owner.context.initialization), named_rng=owner.streams.manifest(),
        model_sha256=models, initial_named_rng=deepcopy(owner.initial_named_rng),
        global_rng=deepcopy(owner.initial_global_rng), global_rng_scope="Torch CPU plus the declared CUDA device",
        full_resolved_recipe=asdict(owner.recipe),
        resolved_recipe_sha256=digest(asdict(owner.recipe)),
        source_contract_sha256=owner.binding["source_contract_sha256"],
        device=str(owner.context.device), factory_default_prior_kind=owner.recipe.prior_kind)


def compile_producer(raw, name):
    """Replace only original producer construction with the admitted factory.

    All scientific statements after construction remain AST-exact, including
    target draws, updates, observation clocks, scorers and saved artifacts.
    This hook is also usable by model-free private source controls.
    """
    if name not in {"_vector", "_ring", "_native"}:
        raise ValueError("unknown original producer")
    node = deepcopy(next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == name))
    start = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "context" for t in n.targets))
    stop = next(i for i in range(start, len(node.body)) if isinstance(node.body[i], ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "trainer" for t in node.body[i].targets)) + 1
    setup = node.body[start:stop]
    expected = 2 if name == "_ring" else 3
    if len(setup) != expected or any(not isinstance(n, ast.Assign) for n in setup):
        raise ValueError("original construction boundary changed")
    replacement = ast.parse("owner = _owned_factory()\ncontext = owner.context\ntrainer = owner.trainer\ng, d = owner.generator, owner.critic\n").body
    node.body[start:stop] = replacement
    taps = []
    class CaptureExisting(ast.NodeTransformer):
        def visit_Assign(self, item):
            self.generic_visit(item)
            if name == "_vector" and any(isinstance(t, ast.Name) and t.id == "samples" for t in item.targets):
                taps.append("vector scored samples")
                return [item, ast.parse("_capture_existing(step, samples, None)").body[0]]
            if name == "_native" and any(isinstance(t, ast.Name) and t.id == "draw" for t in item.targets):
                taps.append("native live scored samples and existing reference")
                return [item, ast.parse("if model == 'live':\n    _capture_existing(step, draw[:config['snapshot_samples']], reference[:config['snapshot_samples']])").body[0]]
            if name == "_native" and any(isinstance(t, ast.Name) and t.id == "event" for t in item.targets):
                taps.append("native existing live metric event")
                return [item, ast.parse("if model == 'live':\n    _record_existing_native(event)").body[0]]
            return item
        def visit_Call(self, item):
            self.generic_visit(item)
            if name == "_ring" and isinstance(item.func, ast.Name) and item.func.id == "diversity":
                if len(item.args) != 2:
                    raise ValueError("ring observation boundary changed")
                taps.append("ring existing single sample draw and declared means")
                item.args[0] = ast.Call(func=ast.Name(id="_capture_identity", ctx=ast.Load()),
                    args=[ast.Name(id="step", ctx=ast.Load()), item.args[0], deepcopy(item.args[1])], keywords=[])
            return item
    node = CaptureExisting().visit(node)
    if name == "_vector":
        # Clone only the original scorer's globals to observe its existing
        # calibration target draw. Its function bytecode remains exact.
        position = start + len(replacement)
        node.body.insert(position, ast.parse("score_samples = _observe_vector_score(score_samples)").body[0])
    return ast.fix_missing_locations(node), dict(original_setup=ast.dump(ast.Module(body=setup, type_ignores=[]),
        include_attributes=False), replacement_statements=len(replacement), insertion_start=start,
        capture_sites=taps, scientific_body_unchanged_except_detached_output_taps=True)


def observe_existing_target(original, target_callback, no_grad):
    """Keep scorer bytecode and no-grad, tapping its one existing target draw.

    The original scorer has a public no-grad wrapper; cloning only that
    wrapper's globals would miss its closed-over scientific function. Clone
    the source-bound underlying function and restore the same no-grad context.
    No original module globals or function are modified.
    """
    def clone(function, namespace):
        result = FunctionType(function.__code__, namespace, function.__name__,
                              function.__defaults__, function.__closure__)
        result.__kwdefaults__ = function.__kwdefaults__
        return result
    core = getattr(original, "__wrapped__", None)
    if core is None or hasattr(core, "__wrapped__"):
        raise ValueError("the frozen scorer requires one public no-grad wrapper")
    outer_namespace = dict(core.__globals__)
    base = outer_namespace.get("vector_score_samples", original)
    base_core = getattr(base, "__wrapped__", None)
    if base_core is None or hasattr(base_core, "__wrapped__"):
        raise ValueError("the frozen base scorer requires one public no-grad wrapper")
    base_namespace = dict(base_core.__globals__)
    original_target = base_namespace.get("sample_target")
    if original_target is None:
        raise ValueError("original vector score target tap is unavailable")
    def measured_target(*args, **kwargs):
        value = original_target(*args, **kwargs)
        target_callback(value)
        return value
    base_namespace["sample_target"] = measured_target
    captured = no_grad(clone(base_core, base_namespace))
    if base is original:
        return captured
    outer_namespace["vector_score_samples"] = captured
    return no_grad(clone(core, outer_namespace))


def compile_source_preflight(root):
    """Validate source closure and compile hooks without importing ML/scorers."""
    for relative in SOURCE_PINS:
        _source(root, relative)
    raw = _source(root, "experiments/forge/adapters.py")
    overlays = {}
    for name in ("_vector", "_ring", "_native"):
        node, proof = compile_producer(raw, name)
        compile(ast.Module(body=[node], type_ignores=[]), "<metadata-only-producer>", "exec")
        overlays[name] = proof
    compile(ast.Module(body=[compile_receipt(raw)], type_ignores=[]), "<metadata-only-health>", "exec")
    compile(_source(root, "experiments/forge/mechanisms.py").decode(),
            "<metadata-only-scratch-probe>", "exec")
    # The inherited methods remain actual public functions at runtime. No
    # copied scientific training loop or scorer is executed by preflight.
    return dict(schema="pg_canonical_mog_compile_preflight_v1", source_files=len(SOURCE_PINS),
        task_ids=list(TASK_IDS), overlays=overlays, model_constructors=0, forwards=0,
        sampler_calls=0, scorer_calls=0, optimizer_updates=0, queue_calls=0)


def run_case(root, request, *, output_dir, device, source_guard, fresh_repeat_guard):
    """Run only the unchanged original producer inside the admitted child."""
    source_guard()
    expected = resolve_binding(root, request["candidate"], request["task"], request["protocol"])
    if canonical(request["binding"]) != canonical(expected):
        raise ValueError("request owner binding drift")
    task = _declaration(request["task"])
    if task["id"] == "ring_extension":
        raise ValueError("ring extension belongs to the original uninterrupted ring_hold group")
    prerequisite_digest = None
    if task["adapter"] == "ring_endurance":
        prerequisite_digest = request["protocol"].get("scientific_repeat", {}).get("prerequisite_proof_sha256")
        if (not isinstance(prerequisite_digest, str) or len(prerequisite_digest) != 64
                or any(c not in "0123456789abcdef" for c in prerequisite_digest)):
            raise ValueError("ring group repeat must bind the controller-validated current prerequisite proof")
        group = expected["source_contract"]["grouped_execution"]
        proof = request.get("prerequisites", {}).get("mode_hold")
        if (proof is None or digest(proof) != prerequisite_digest
                or canonical(request.get("grouped_tasks")) != canonical(group["task_definitions"])
                or request["protocol"]["scientific_repeat"].get("grouped_tasks_sha256")
                    != digest(group["task_definitions"])):
            raise ValueError("group declarations or current prerequisite proof differ from the repeated intent")
        # The callback is the controller's artifact/terminal/source validator,
        # not a caller PASS flag. It checks the proof before any construction.
        source_guard()
    import torch
    from experiments.forge import adapters
    from experiments.forge.policy_adapters import finite_policy_state
    source_guard()
    held = []
    observation = expected["source_contract"]["observation"]
    goal_steps = set(observation["goal_steps"])
    last_observed = None
    native_observations = []
    def owned_factory():
        if held:
            raise ValueError("a scientific attempt constructs exactly one fresh owner")
        owner = fresh_repeat_guard.construct(lambda: construct_owner(root, expected, device=device,
            source_guard=source_guard), owner_initial_receipt)
        held.append(owner)
        return owner
    def capture_existing(step, generated, target):
        # Copies only arrays already used by the original score. No forward,
        # sample, target RNG draw or metric recomputation occurs here.
        nonlocal last_observed
        # One current draw reference is overwritten each check. In endurance
        # this prevents thousands of cloud copies while preserving the final
        # scored draw. The producer creates new outputs; no live parameter is
        # referenced by these detached generated/target values.
        last_observed = dict(step=step, generated=generated.detach(),
                             target=None if target is None else target.detach())
        if step in goal_steps and target is not None:
            retain_existing(last_observed)
    def retain_existing(state):
        held[0].retained_goal_states.append(dict(step=state["step"],
            generated=state["generated"].clone().cpu(), target=state["target"].clone().cpu()))
    def capture_identity(step, generated, target):
        capture_existing(step, generated, target)
        return generated
    def observe_vector_score(original):
        def target_callback(value):
            state = last_observed
            if state is None:
                raise ValueError("an original target draw preceded its scored samples")
            if state["target"] is not None:
                raise ValueError("the original vector score unexpectedly drew multiple targets")
            state["target"] = value.detach()
            if state["step"] in goal_steps:
                retain_existing(state)
        return observe_existing_target(original, target_callback, torch.no_grad())
    def record_existing_native(event):
        # Copy the original recorded metrics; never recompute a metric here.
        native_observations.append(dict(step=event["step"],
            **deepcopy(event["metrics"]), accuracy=deepcopy(event["accuracy"])))
    def training_state(trainer):
        state = trainer.state_dict()
        state["streams"].pop("eval_generator")
        state.pop("cpu_rng"); state.pop("cuda_rng")
        state["observed_lifecycle"] = held[0].lifecycle_audit.receipt(trainer.completed_steps)
        return typed_digest(state, held[0].tensor_bytes)
    def complete_digest(state):
        return typed_digest(state, held[0].tensor_bytes)
    def finite_learned_state(trainer):
        models = [trainer.G, trainer.D, trainer.prior, trainer.ema_G, trainer.ema_prior, trainer.ema_D]
        values = [tensor for model in models for tensor in (*model.parameters(), *model.buffers())]
        values.extend(p.grad for model in models for p in model.parameters())
        values.extend([trainer.log_output_sigma, trainer.opt_g.state_dict(), trainer.opt_d.state_dict()])
        return adapters._finite_tree(values) and finite_policy_state(trainer.policy.state_dict())
    run_namespace = dict(adapters.__dict__, _finite_learned_state=finite_learned_state)
    exec(compile(ast.Module(body=[compile_receipt(_source(root, "experiments/forge/adapters.py"))],
                           type_ignores=[]), "<source-bound-original-run-health>", "exec"), run_namespace)
    class PureObservationRun(adapters._Run):
        receipt = run_namespace["receipt"]
        def evaluate(self, function):
            before = training_state(self.trainer)
            result = super().evaluate(function)
            if before != training_state(self.trainer):
                raise ValueError("observation changed model, optimizer, policy or training RNG")
            held[0].observation_purity.append(dict(status="PURE", before_sha256=before,
                training_draws_added=0, optimizer_updates_added=0))
            return result
    name = expected["source_contract"]["producer"]
    node, overlay = compile_producer(_source(root, "experiments/forge/adapters.py"), name)
    namespace = dict(adapters.__dict__, _owned_factory=owned_factory, _Run=PureObservationRun,
        _capture_existing=capture_existing, _capture_identity=capture_identity,
        _observe_vector_score=observe_vector_score, _record_existing_native=record_existing_native,
        state_digest=complete_digest)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<source-bound-original-MoG-producer>", "exec"), namespace)
    wire = deepcopy(request)
    wire["tasks"] = {task["id"]: task}
    options = dict(endurance=True) if task["adapter"] == "ring_endurance" else {}
    result = namespace[name](wire, task, Path(output_dir), device, **options)
    if len(held) != 1:
        raise ValueError("original producer did not construct one fresh owner")
    owner = held[0]
    lifecycle = owner.lifecycle_audit.receipt(owner.policy.completed_steps)
    if not lifecycle["complete"]:
        raise ValueError("each actual public lifecycle hook must execute once per update")
    if observation["retain_actual_final_observation"] and last_observed is not None and (
            not owner.retained_goal_states or owner.retained_goal_states[-1]["step"] != last_observed["step"]):
        retain_existing(last_observed)
    if len(owner.retained_goal_states) > observation["max_retained_goal_states"]:
        raise ValueError("retained goal clouds exceeded the frozen memory bound")
    if native_observations:
        result["evidence"]["observations"] = native_observations
    result["evidence"]["original_mog_owner"] = dict(binding=deepcopy(expected),
        initialization=deepcopy(owner.initial_receipt),
        lifecycle_counts=deepcopy(owner.calls), lifecycle=deepcopy(lifecycle),
        observations=deepcopy(owner.observation_purity),
        actual_prior_capabilities=deepcopy(owner.context.receipt()["actual_prior_capabilities"]),
        sampling=deepcopy(expected["source_contract"]["sampling"]))
    result["evidence"]["guards"]["finite_scope"] = expected["source_contract"]["harness_overlays"]["health_scope"]
    if any(s["target"] is None for s in owner.retained_goal_states):
        raise ValueError("an original scored target was not captured")
    result["retained_goal_states"] = owner.retained_goal_states
    result["complete_state"] = owner.context.state_dict()
    if task["adapter"] == "ring_endurance":
        result["execution_group"] = dict(
            **deepcopy(expected["source_contract"]["grouped_execution"]),
            prerequisite_proof_sha256=prerequisite_digest,
            grouped_tasks_sha256=digest(group["task_definitions"]),
            grouped_task_ids=["ring_hold", "ring_extension"],
            continuity=deepcopy(result["evidence"]["continuity"]),
            completed_steps=result["complete_state"]["trainer"]["completed_steps"],
            shared_raw_evidence=True)
    source_guard()
    return result
