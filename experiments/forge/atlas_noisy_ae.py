"""Prior-type-only Atlas AE owner for the isolated686 branch.

Original free encoder/reconstruction and sampled unconditional decoder policy,
250 updates,24 live scheduled-noise reads and original gates are retained.
Import defines Source/metadata functions only; no models or tensors are made.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
from importlib.machinery import NamespaceLoader
import inspect
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
from types import SimpleNamespace

from .canonical_ae_adapter import (AEOwner, _EncodingObjective, _PenaltyView,
    _PolicyNoise, derive_host_train, typed_digest)

SCHEMA = "forge_atlas_noisy_ae_binding_v1"
OWNER_SCHEMA = "forge_atlas_noisy_ae_initial_owner_v1"
TASK_ID = "ae_gan_hold_noisy_prior686_v1"
TASK_PATH = "configs/forge/tasks/ae_gan_hold.json"
VARIANT_PATH = "configs/forge/task-variants/noisy-prior686/ae_gan_hold_noisy_prior686_v1.json"
PROTOCOL_PATH = "configs/forge/protocols/screening.json"
HOST_PATH = "benchmarks/locked_shared/hosts/ae_gan_hold.py"
STEPS = 250
CLOCKS = [math.ceil(i * STEPS / 24) for i in range(1, 25)]
TASK_BINDINGS = dict(num_particles=12, z_dim=2, batch_size=64, standardize=False,
                     sigma_rel=0., prior_kind="noisy_particles")
ACTUAL_PRIOR = dict(kind="noisy_particle_cloud", sigma=.025, standardize=False, learnable=True)
PRIOR_FACTORY_OVERRIDES = dict(prior_kind="noisy_particles", sigma=.025, standardize=False)
RESOLVED_RECIPE = {'name': 'atlas', 'critic_formulation': 'ka2', 'model': 'gan', 'z_dim': 2, 'num_particles': 12, 'prior_kind': 'noisy_particles', 'sigma_rel': 0.0, 'standardize': False, 'num_classes': None, 'conditioning': 'scalar', 'ucd_target': 'class', 'ucd_weight': 0.02, 'alpha_bar': [1.0, 0.9, 0.5, 0.05, 0.0001], 'batch_size': 64, 'total_steps': None, 'continuous_policy': 'dv12', 'lr': 0.00425, 'd_lr_mult': 1.0, 'prior_lr_mult': 2.0, 'betas': [0.0, 0.999], 'prior_betas': None, 'reg_arm': None, 'reg_coeff': 3.0, 'reg_kappa': 1.0, 'reg_every': 1, 'prior_reg': 0.0, 'ema_decay': 0.995, 'lr_anneal_start': 0.6, 'lr_floor': 0.05, 'network_lr_floor': 0.01, 'network_lr_horizon_cap': 1600, 'reg_anchor_min_decay': 0.9, 'reg_anchor_weight': 1.0, 'direct_particle_gain': True, 'd_guard_ratio': 5.0, 'd_guard_min_steps': 200, 'latent_damping_max_rate': 0.5, 'direct_particle_betas': [0.0, 0.9], 'input_noise_std': 0.0, 'input_noise_anneal_end': 0.1, 'output_noise_std': 0.029, 'output_noise_warmup': 0.0, 'encoder_mode': 'none', 'routing_temperature': 0.25, 'distance_reduction': 'sum', 'observation_sigma': 0.03, 'reconstruction_weight': 1.0, 'amsgrad': True, 'critic_r1_real': True, 'critic_payoff_damping': True, 'output_noise_mode': 'learnable', 'lr_control': 'stationarity', 'particle_birth_death': True, 'row_evidence_gate': True, 'table_release_rule': 'anchor', 'row_evidence_hot': True, 'row_evidence_exclude': True, 'row_evidence_hold': True, 'birth_death_space': 'critic', 'serve_average': 4.0, 'reopen_signal': 'optimizer', 'reopen_anchor': 'release', 'reopen_guard': 'settled', 'row_evidence_null': 'scaled', 'birth_death_isolation': True, 'birth_death_feature_scale': 'std', 'birth_death_backend': 'auto', 'birth_death_cells': 128, 'birth_death_metric_rank': 8, 'birth_death_chunk': 256, 'birth_death_parent_policy': 'real_anchor', 'row_policy': 'independent', 'optimizer_family': 'formulation', 'eps': 1e-08, 'beta2_end': None, 'beta2_anneal_end': 0.2, 'reg_coeff_end': None, 'reg_coeff_anneal_end': 0.2, 'loss': 'relativistic'}
SOURCE_PINS = {'benchmarks/__init__.py': {'bytes': 71, 'sha256': '53ba1483505036be295d73f5f3308bffb951e9414bf2b1139d8756e260e3b830'}, 'benchmarks/gan_v3.py': {'bytes': 2573, 'sha256': '00ca856749c82ce3d4e554ced924dda4399311d2e31ca8dbea8b8d5829141538'}, 'benchmarks/legacy/__init__.py': {'bytes': 393, 'sha256': '83de2dd27bb1fec5ba74b066b49274ba161e4e480037129e3c99e13b04767f04'}, 'benchmarks/legacy/grad_regularizers.py': {'bytes': 27997, 'sha256': 'cfe252f52a6d11274e4be78bd13934fc3d919abe7d0ba349e5bd0199a45dcb5d'}, 'benchmarks/legacy/recipe.py': {'bytes': 9954, 'sha256': 'b31cdb23913b3c71432ecbdee40a0c0293c067bbcd1f2e5cdfb192426a43c543'}, 'benchmarks/locked_shared/__init__.py': {'bytes': 79, 'sha256': 'a33bb985c9801d5fe3ad19666a43f8448a7aaebe2ff86555bd05d7a20e76fc2f'}, 'benchmarks/locked_shared/baseline.py': {'bytes': 30125, 'sha256': '8f1f58fc9cb90159c555788e593769d50070deaa078d0637bf881cf074f6680b'}, 'benchmarks/locked_shared/hosts/__init__.py': {'bytes': 78, 'sha256': '3f0d91417b7c3b167e7b9a13b2053eca53394ca454ea62d27f1c26c723a096fb'}, 'benchmarks/locked_shared/hosts/ae_gan_hold.py': {'bytes': 8676, 'sha256': 'c25ea8f8b998a137719df7fc18e5b047b456cdb5a65969430a652db0524509a9'}, 'benchmarks/locked_shared/observation.py': {'bytes': 4351, 'sha256': 'bd6f9845b44f1ec2a58d445727990ba5068c7aca3b6f981cf38d738a37c4513b'}, 'benchmarks/toy100/device.py': {'bytes': 10288, 'sha256': '7fa0f53db44e8824d3497b471e9fe8db56cbf8b4e02ddf997f65beed63f81911'}, 'benchmarks/transfer_suite/protocol.py': {'bytes': 9964, 'sha256': '99469b022b790a18a74021a6fe49424d95f535afaa220643688a1ddd7a70ab89'}, 'configs/forge/protocols/screening.json': {'bytes': 972, 'sha256': '3fefb4d47fd2cd8aa6ed110c0a9f5bffefaae700431d1f57ca7b162c8efbb803'}, 'configs/forge/task-variants/noisy-prior686/ae_gan_hold_noisy_prior686_v1.json': {'bytes': 3284, 'sha256': '0942d40f40ededeaa27be959e1568d6e52afaa151a9b53fecd7710cf2d9d8ca9'}, 'configs/forge/tasks/ae_gan_hold.json': {'bytes': 2897, 'sha256': '53a400c3f2b27ef347076f3cc603345e1442d2d8f97f8052f0b9496ba35bae79'}, 'experiments/forge/mechanisms.py': {'bytes': 11826, 'sha256': '9e4e1fe1f8cb9a24148cab88b7bf2f980e805a2e488c16a982768519bbe1b23f'}, 'experiments/forge/noisy_prior_tier1.py': {'bytes': 8177, 'sha256': 'd8eacc15850c803cb907a39c8b09dbe2777c35bb946e7d8170cff224582c3f4a'}, 'experiments/forge/policy_adapters.py': {'bytes': 17992, 'sha256': '714046ea1655f1946e09dc2aeed27c43cf93f2841237c2be47f0f0049a5df9fa'}, 'experiments/forge/rng.py': {'bytes': 6893, 'sha256': 'ae7b9a8d61da42136d970188a6f168e03e2d7c5b90cf6fdbd93ad929aba2f293'}, 'particlegan/__init__.py': {'bytes': 1886, 'sha256': '18506efd720327de46f4bd5f4f66dab6e1b0574c5388e5496795400a45aa4070'}, 'particlegan/_qr.py': {'bytes': 1951, 'sha256': 'b59cd62cc27ed557b41b2d17ed86ed578c762a3a8a392d63731a5a3b2b018916'}, 'particlegan/anchor_birth.py': {'bytes': 6401, 'sha256': 'aa50983d8b61304acf4e1138f16fd4784798983df91e931863bcacd3b511f76f'}, 'particlegan/autoencoder.py': {'bytes': 6194, 'sha256': 'f3fbb9184e9f44112e15eccf0c0246dfba5ed490d3c7b81b0f837a5335916402'}, 'particlegan/birth_death.py': {'bytes': 45396, 'sha256': 'b14c50c611a8cf188347e391739fca50a5171200eb6be90e4e3d1509d64e47ed'}, 'particlegan/birth_phase.py': {'bytes': 21144, 'sha256': '5ff929e5e9858aa24c4cc2f8999b5d0603e25e42ca3844d1da4cc28b1981d3d7'}, 'particlegan/capabilities.py': {'bytes': 2548, 'sha256': '74f57e8485bd72193e535c4f78978b5f908a6c965bb417c2fd57b2cf446720a0'}, 'particlegan/conditioning.py': {'bytes': 4878, 'sha256': '8734fe338868ace4b5851a8fdb1f071e46d47396f66a3765e6b1ef37f0481486'}, 'particlegan/continuous.py': {'bytes': 60875, 'sha256': '4ceab49c7d51d1769ae91b7f8bc892eaf380ca6fbd77a948a086ade6627e9a41'}, 'particlegan/diffusion.py': {'bytes': 5100, 'sha256': '19c3aa8772703891551b9122fde8e74c07996f779892b7cb6be09553691efa61'}, 'particlegan/discriminators.py': {'bytes': 5833, 'sha256': '0e2efb125ffb314577612ab7a2eba66b0a1a2d28ad25f403f42c18b6f6ee333f'}, 'particlegan/feature_cells.py': {'bytes': 125325, 'sha256': 'e0ea05f61abd7845c7437eed9a79c3a2511d5551ff04f4e619b6b9c8992e221a'}, 'particlegan/feature_policy.py': {'bytes': 14650, 'sha256': 'de1a50f70a37d4c9174c8850ce3eac2a2597ed01f19fc4fe1a2ce0ec3176a8b2'}, 'particlegan/feature_reference.py': {'bytes': 32368, 'sha256': '58e076e9e04600be25e28afafc537a340de727371f4acbbfb2b7dd8bebac18b1'}, 'particlegan/gan_loss.py': {'bytes': 3203, 'sha256': '2ba031aa3c162bc6c4666c7dd43a2aac295ed2a841f7855e6b0a28b29eca6b2f'}, 'particlegan/grad_regularizers.py': {'bytes': 14853, 'sha256': 'a540d05a4a992540b6a5f3b5ff7ccc11ebaf18592135b87c95638adbe68aa747'}, 'particlegan/init.py': {'bytes': 24559, 'sha256': 'e15d4de01eb49f7097bc249abacab907ff067385412346f54ab0d510cb8649d0'}, 'particlegan/k3p.py': {'bytes': 33012, 'sha256': '200ead2c27ab0b2aa6068f1b1b1b5c826b8125aa8602bfec08b31aaa2894401c'}, 'particlegan/ka2.py': {'bytes': 16551, 'sha256': '95fdaa58a2bdc229d5d146ef89548bc3fe07582d06181a149ed1bb4f5d632703'}, 'particlegan/mean_transport.py': {'bytes': 39633, 'sha256': '57a68e0a65c5424a4460f65002f3862a69e90af330a4d47c700dc773683a2c06'}, 'particlegan/noisy_particle_prior.py': {'bytes': 3328, 'sha256': 'a59eb7340cae67b0f8f40729bd89cd2071bba63c7ae66d516246554b6c621316'}, 'particlegan/output_moments.py': {'bytes': 8387, 'sha256': 'dcf3e27228d2c3c9738e473c5b22ee9e6d5047ddcd60c538b66526adb53ef188'}, 'particlegan/particle_prior.py': {'bytes': 17899, 'sha256': '0220878bebea227da63abbf9f5b1ebdabdc6aea54fd2332fd2cab02ed734238f'}, 'particlegan/policy.py': {'bytes': 81022, 'sha256': '8370e36b5b93afe95ae24d2b385aac8735a587f86ef2b87ed369ecfdfcb6fec5'}, 'particlegan/population_continuity.py': {'bytes': 20329, 'sha256': '378cd75cca45bc8da4da33e686a12cf99df853042687978314e37c532bc17fe6'}, 'particlegan/recipe_schedules.py': {'bytes': 6044, 'sha256': 'd05e459b3e9e5f938b01a854273b74343f2b6d382bec4ae5151dd1f1cd032500'}, 'particlegan/recipes.py': {'bytes': 55930, 'sha256': '55601319bae017171c63219c058399e889466f574bdef9626ca8425e72bfab62'}, 'particlegan/routing.py': {'bytes': 64904, 'sha256': '63249d808fccb3d4e113eb1d638a245ee665c3cade20a1542bb706f316b4abcc'}, 'particlegan/row_evidence.py': {'bytes': 9397, 'sha256': '78be40141ef3946860458e9842159be9321f05c217a4800dd4134113b6eba93e'}, 'particlegan/training.py': {'bytes': 40093, 'sha256': '740c01b14ce4542f5fcef43c59f9eed09b122f349a2308fb17bcc87e97786eb2'}, 'particlegan/vicreg_loss.py': {'bytes': 2297, 'sha256': 'ab1c4dc266dec2c35337f449917240eb38afede7590d2154b63f6f471dc45a36'}, 'experiments/forge/canonical_ae_adapter.py': {'sha256': 'e09306a6ab520939d38651d8ea707bb900b237381f2144d15498d9b5ad6364e5', 'bytes': 47595}}

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


def _loaded_source(root, value, relative, *, no_grad=None):
    inspected = value
    if relative == "particlegan/init.py" and inspect.isfunction(value):
        # This one pinned function has one source-declared @torch.no_grad().
        # Check the public export AND the actual wrapper; __wrapped__ alone is
        # not permission to replace a foreign decorator with an owned target.
        module = sys.modules.get("particlegan.init")
        expected_path = Path(root).resolve() / relative
        module_file = getattr(module, "__file__", None)
        if (module is None or not inspect.ismodule(module)
                or type(module_file) is not str or not module_file
                or value.__module__ != "particlegan.init"
                or value.__name__ != "deterministic_orthogonal_"
                or value.__qualname__ != "deterministic_orthogonal_"
                or getattr(module, "deterministic_orthogonal_", None) is not value
                or Path(module_file).resolve() != expected_path):
            raise ValueError("foreign initializer export: " + relative)
        inspected = inspect.unwrap(value)
        if (inspected is value or not inspect.isfunction(inspected)
                or getattr(value, "__wrapped__", None) is not inspected
                or inspected.__globals__ is not module.__dict__
                or inspected.__module__ != value.__module__
                or inspected.__name__ != value.__name__
                or inspected.__qualname__ != value.__qualname__
                or not callable(no_grad)):
            raise ValueError("unexpected initializer decorator: " + relative)
        # Creating this context wrapper does not enter it or call the function.
        # Match code/globals and both closure owners against the imported
        # no_grad factory, including the unchanged fresh context attributes.
        probe = no_grad()(inspected)
        if (not inspect.isfunction(probe) or value.__code__ is not probe.__code__
                or value.__globals__ is not probe.__globals__
                or value.__defaults__ != probe.__defaults__
                or value.__kwdefaults__ != probe.__kwdefaults__):
            raise ValueError("foreign initializer wrapper: " + relative)
        actual = tuple(value.__closure__ or ())
        expected = tuple(probe.__closure__ or ())
        if len(actual) != len(expected) or not expected:
            raise ValueError("foreign initializer closure: " + relative)
        target_cells = context_cells = 0
        for actual_cell, expected_cell in zip(actual, expected):
            actual_owner, expected_owner = actual_cell.cell_contents, expected_cell.cell_contents
            if expected_owner is inspected:
                target_cells += 1
                same = actual_owner is inspected
            elif inspect.ismethod(expected_owner):
                context_cells += 1
                same = (inspect.ismethod(actual_owner)
                        and actual_owner.__func__ is expected_owner.__func__
                        and type(actual_owner.__self__) is type(expected_owner.__self__)
                        and vars(actual_owner.__self__) == vars(expected_owner.__self__))
            else:
                same = actual_owner is expected_owner
            if not same:
                raise ValueError("foreign initializer closure owner: " + relative)
        if target_cells != 1 or context_cells != 1:
            raise ValueError("unexpected initializer closure layout: " + relative)
    path = Path(inspect.getfile(inspected)).resolve()
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


def _supports_candidate(candidate):
    return (isinstance(candidate, dict) and candidate.get("recipe_preset") == "atlas"
            and candidate.get("recipe_overrides", {}) == {}
            and candidate.get("extensions", {}) == {} and candidate.get("host_adaptation") is None
            and not candidate.get("implementation")
            and candidate.get("initializer", "deterministic_orthogonal") == "deterministic_orthogonal")


def _metadata_recipe(source):
    """Resolve only the pinned public preset metadata, never a real constructor."""
    tree = ast.parse(source)
    method = deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_recipe"))
    fields = _recipe_fields(source)
    scope = {"Recipe": lambda **overrides: {**deepcopy(fields), **overrides}}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 "<pinned-Noisy-AE-Atlas-preset-metadata>", "exec", dont_inherit=True), scope)
    result = scope["get_recipe"]("atlas")
    result.update(TASK_BINDINGS)
    if result["reg_arm"] is not None:
        result["critic_formulation"] = "k3p"
    return json.loads(canonical(result))


def blockers(task, candidate, root=None):
    try:
        from .noisy_prior_tier1 import validate
        metadata = validate(task, root=root)
        if metadata["parent_task_id"] != "ae_gan_hold" or not _supports_candidate(candidate):
            raise ValueError("only the fixed Noisy AE task and unchanged current Atlas preset are supported")
        if root is not None:
            resolve_binding(root, candidate, task, json.loads(_source(root, PROTOCOL_PATH)))
        return []
    except (KeyError, TypeError, ValueError, OSError) as error:
        return [str(error)]


def resolve_binding(root, candidate, task, protocol):
    """Current79 source-bound metadata; no model, draw or owner is constructed."""
    from .noisy_prior_tier1 import validate
    metadata = validate(task, root=root)
    if metadata["parent_task_id"] != "ae_gan_hold" or not _supports_candidate(candidate):
        raise ValueError("fixed AE prior-only task and unchanged Atlas preset required")
    if canonical(_declaration(task)) != canonical(json.loads(_source(root, VARIANT_PATH))):
        raise ValueError("complete Noisy AE declaration differs")
    if canonical(protocol) != canonical(json.loads(_source(root, PROTOCOL_PATH))):
        raise ValueError("original screening seed0/streams protocol required; no repeat/restore flags")
    for relative in SOURCE_PINS:
        _source(root, relative)
    own_file = Path(__file__).resolve()
    if own_file != Path(root).resolve() / "experiments/forge/atlas_noisy_ae.py":
        raise ValueError("Noisy AE metadata must use its frozen owned module")
    fields = _metadata_recipe(_source(root, "particlegan/recipes.py").decode())
    if (len(fields) != 79 or fields["total_steps"] is not None or fields["encoder_mode"] != "none"
            or fields["row_policy"] != "independent" or fields["continuous_policy"] != "dv12"
            or not fields["particle_birth_death"] or not fields["row_evidence_gate"]
            or fields["prior_kind"] != "noisy_particles" or fields["standardize"] is not False):
        raise ValueError("one complete current Atlas79/independent/schedule-free tuple required")
    files = deepcopy(SOURCE_PINS)
    files["experiments/forge/atlas_noisy_ae.py"] = dict(sha256=hashlib.sha256(own_file.read_bytes()).hexdigest(),
                                                      bytes=own_file.stat().st_size)
    contract = dict(schema="forge_atlas_noisy_ae_public_owner_contract_v1", files=files,
        task_id=TASK_ID, parent_task_id="ae_gan_hold", external_max_steps=STEPS,
        actual_prior=deepcopy(ACTUAL_PRIOR), prior_factory_overrides=deepcopy(PRIOR_FACTORY_OVERRIDES),
        ownership=dict(generator="host.MLP(2,2,32)", encoder="host.MLP(2,4,32)",
                       discriminator="host.MLP(2,1,32)", prior="NoisyParticlePrior.z12x2",
                       homogeneous_roles=[["generator", "encoder", "table", "noise"], ["critic"]]),
        objective=dict(encoder="free_query_and_offset", encoding="unchanged_public_particle_ae",
                       original_host=HOST_PATH, original_derived_train="canonical_ae_adapter.derive_host_train",
                       reconstruction_weight=1., adversarial_weight=1., cover_weight=1.5,
                       particle_l2=.02, feature_matching_weight=0., gan="Recipe.make_loss",
                       penalty="Recipe.make_critic_penalty", policy_generation="UpdatePolicy.generate",
                       joint_inverse_or_routed_loss_added=False),
        observation=dict(sampling_law=task["evaluation"]["sampling_law"], eval_output_noise="public_recipe_schedule",
                         scoring_weights="live", sample_count=1024, observations=CLOCKS,
                         final_five=CLOCKS[-5:], selected_or_averaged_sampling=False,
                         evaluation_DV12=False, all_named_and_global_rng_preserved=True),
        resources=deepcopy(task["resources"]), initializer="original_named_MLP_and_R2_prior_initializers",
        schedule="current_global_Atlas_total_steps_None_external250",
        substitution="only_actual_prior_type; parent_absolute_sigma.025_stdFalse_retained",
        original_scores_transferred=False, ordinary_parent_credit=False, qualification=False,
        default_adoption=False, speed_ranking=False)
    return dict(schema=SCHEMA, candidate=deepcopy(candidate), task=deepcopy(_declaration(task)),
        protocol=deepcopy(protocol), recipe=fields,
        recipe_serialized=serialized_recipe(fields, _source(root, "particlegan/recipes.py")),
        actual_prior=deepcopy(ACTUAL_PRIOR), prior_factory_overrides=deepcopy(PRIOR_FACTORY_OVERRIDES),
        source_contract=contract, source_contract_sha256=digest(contract),
        task_sha256=SOURCE_PINS[VARIANT_PATH]["sha256"], parent_task_sha256=SOURCE_PINS[TASK_PATH]["sha256"],
        protocol_sha256=SOURCE_PINS[PROTOCOL_PATH]["sha256"], adapter_sha256=files["experiments/forge/atlas_noisy_ae.py"]["sha256"])


def construct_owner(root, binding, *, device, source_guard):
    """Actual CPU owner factory, called once by a fresh guard after admission."""
    if str(device) != "cpu" or not callable(source_guard):
        raise ValueError("canonical AE requires CPU ownership and a real source guard")
    source_guard()
    expected = resolve_binding(root, binding["candidate"], binding["task"], binding["protocol"])
    if canonical(binding) != canonical(expected):
        raise ValueError("forged complete Noisy AE binding before model construction")
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
                           (finite_policy_state, "experiments/forge/policy_adapters.py"),
                           (AEOwner, "experiments/forge/canonical_ae_adapter.py"),
                           (derive_host_train, "experiments/forge/canonical_ae_adapter.py")):
        _loaded_source(root, value, relative, no_grad=torch.no_grad)
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
        raise ValueError("actual nonstandardized Noisy prior must support the requested current A2")
    owner.opt_g = recipe.make_generator_optimizer([
        dict(params=list(owner.generator.parameters()), lr=recipe.lr, forge_role="generator"),
        dict(params=list(owner.encoder.parameters()), lr=recipe.lr, forge_role="encoder"),
        dict(params=[owner.prior.z], lr=recipe.lr * recipe.prior_lr_mult,
             betas=recipe.prior_betas or recipe.betas, forge_role="prior")], latent_table=owner.prior.z)
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
    from particlegan.noisy_particle_prior import NoisyParticlePrior
    from particlegan.k3p import K3PGeneratorAdam
    from particlegan.ka2 import KA2CriticAdam
    from benchmarks.locked_shared.hosts import ae_gan_hold
    if (type(owner) is not AEOwner or type(owner.policy) is not UpdatePolicy
            or any(type(model) is not ae_gan_hold.MLP for model in (owner.generator, owner.encoder, owner.critic))
            or type(owner.prior) is not NoisyParticlePrior or type(owner.opt_g) is not K3PGeneratorAdam
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
            policy.controller, policy.lr_settle, policy.birth_death, policy.row_evidence, policy.surprise,
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
        public_prior_type="particlegan.noisy_particle_prior.NoisyParticlePrior",
        prior_kernel=owner.prior.kernel_contract(),
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
            streams=owner.streams.manifest(), observation_owner="live_G_E_and_fixed_width_actual_Noisy",
            evaluation_DV12=False, selected_or_averaged_metric=False,
            external_horizon=STEPS, intrinsic_horizon=owner.recipe.total_steps,
            qualification=False, default_adoption=False, speed_ranking=False),
        retained_goal_states=owner.retained, complete_state=owner.complete_state())
    owner.require_fresh(); source_guard()
    return result


def source_guard(request, root):
    """Same ordinary snapshot/namespace boundary; no source alias or bypass."""
    from .sources import verify_snapshot
    root = Path(root).resolve()
    source = request["source"]
    if (root != Path(source["snapshot_path"]).resolve()
            or Path(__file__).resolve() != root / "experiments/forge/atlas_noisy_ae.py"):
        raise ValueError("Noisy AE owner must execute from the actual frozen snapshot")
    verify_snapshot(root, source)
    for name, module in tuple(sys.modules.items()):
        if name.split(".", 1)[0] not in {"particlegan", "benchmarks", "experiments", "lib"} or module is None:
            continue
        loaded = getattr(module, "__file__", None)
        namespaces = tuple(getattr(module, "__path__", ()))
        if loaded is None and (not namespaces or not isinstance(
                getattr(getattr(module, "__spec__", None), "loader", None), NamespaceLoader)):
            raise ValueError("missing-file scientific module is not an owned namespace: " + name)
        if loaded is not None:
            path = Path(loaded).resolve()
            if not path.is_relative_to(root) or path.relative_to(root).as_posix() not in source["files"]:
                raise ValueError("foreign already-imported scientific module: " + name)
        for namespace in namespaces:
            if not Path(namespace).resolve().is_relative_to(root):
                raise ValueError("foreign scientific package namespace: " + name)


def _admitted_ordinary_attempt(request, task, output):
    """Ordinary immutable CPU1/full300 Queue request and inherited lease."""
    from .contracts import read_json, stable_hash
    resolved = read_json(output / "request.json")
    if resolved.get("schema_version") != 1 or canonical(resolved.get("request")) != canonical(request):
        raise ValueError("ordinary immutable admitted request differs")
    worker, job = resolved["worker"], resolved["job"]
    if (job["task_id"] != TASK_ID or job.get("task_ids", [job["task_id"]]) != [TASK_ID]
            or job["budget_seconds"] != 300 or job["resources"]["gpus"] != 0
            or job["resources"]["cpu_threads"] != 1 or worker["device"] != "cpu"
            or Path(worker["directory"]).resolve() != output
            or not isinstance(worker.get("token"), str) or not worker["token"]
            or not isinstance(worker.get("attempt"), str) or not worker["attempt"]
            or job["compatibility_key"] != stable_hash(job["science"])
            or job["science"]["candidate_revision"] != request["candidate_revision"]
            or canonical(job["science"]["protocol"]) != canonical(request["protocol"])
            or job["science"]["compute"]["backend"] != "cpu"):
        raise ValueError("ordinary CPU1 full300 AE job/worker admission differs")
    lease_fd = int(os.environ["FORGE_LEASE_FD"])
    actual, declared = os.fstat(lease_fd), (output / "execution.lock").stat()
    if (actual.st_dev, actual.st_ino) != (declared.st_dev, declared.st_ino):
        raise ValueError("ordinary inherited lease does not own this AE attempt")
    if task["id"] not in request["tasks"] or canonical(task) != canonical(request["tasks"][task["id"]]):
        raise ValueError("ordinary request task differs")
    return dict(schema="forge_atlas_noisy_ae_admitted_owner_v1", attempt=worker["attempt"],
        compatibility_key=job["compatibility_key"], source_digest=request["source"]["digest"],
        candidate_revision=request["candidate_revision"], budget_seconds=300, device="cpu", cpu_threads=1,
        inherited_lease_checked=True)


class _OrdinaryFreshOwner:
    """One actual admitted factory; zero update evidence precedes every forward."""
    def __init__(self, guard, admission, output):
        self.guard, self.admission, self.output = guard, admission, output
        self.owner = None
        self.initialization = None

    def construct(self, factory, reader):
        if self.owner is not None or self.initialization is not None:
            raise ValueError("one fresh AE owner per ordinary attempt; no reuse")
        self.guard()
        owner = factory()
        if type(owner) is not AEOwner or reader is not owner_initial_receipt:
            raise ValueError("actual source-owned original AE class and Noisy reader required")
        receipt = reader(owner)
        self.guard()
        from .contracts import atomic_json
        self.owner, self.initialization = owner, receipt
        atomic_json(self.output / "INITIALIZATION.json", dict(
            schema="forge_atlas_noisy_ae_initialization_v1", admission=self.admission,
            actual_owner=receipt, pid=os.getpid()))
        start = dict(schema="forge_atlas_noisy_ae_model_started_v1", admission=self.admission,
                     completed_steps=0, pid=os.getpid(), initialization_sha256=digest(receipt))
        atomic_json(self.output / "MODEL_STARTED.json", start)
        print(canonical(dict(event="actual_model_started", **start)), flush=True)
        return owner

    def require_owned(self, owner):
        if owner is not self.owner or self.initialization is None or owner.restored:
            raise ValueError("unowned or restored AE owner")
        self.guard()
        policy = owner.policy
        if (policy.G is not owner.generator or policy.D is not owner.critic
                or policy.encoder is not owner.encoder or policy.prior is not owner.prior
                or policy.table is not owner.prior.z or policy.table_optimizer is not owner.opt_g
                or policy.roles != [["generator", "encoder", "table", "noise"], ["critic"]]
                or canonical(owner.prior.kernel_contract()) != canonical(self.initialization["prior_kernel"])):
            raise ValueError("actual AE prior/kernel/roles/optimizer ownership drift")


def validate_evidence(task, evidence):
    """Truthful owner/control joins only; the original numerical grader follows."""
    try:
        from .noisy_prior_tier1 import validate
        if validate(task)["parent_task_id"] != "ae_gan_hold":
            raise ValueError("Noisy AE evidence must name its exact fixed variant")
        if not isinstance(evidence, dict):
            raise ValueError("Noisy AE evidence must be an object")
        receipt = evidence["noisy_prior686"]
        if (receipt["schema"] != "forge_atlas_noisy_ae_evidence_v1" or receipt["task_id"] != TASK_ID
                or canonical(receipt["actual_prior"]) != canonical(ACTUAL_PRIOR)
                or receipt["public_prior_type"] != "particlegan.noisy_particle_prior.NoisyParticlePrior"
                or receipt["table_shape"] != [12, 2]
                or receipt["same_location_parameter"] is not True
                or receipt["table_optimizer_alias"] is not True
                or type(receipt["initial_completed_steps"]) is not int or receipt["initial_completed_steps"] != 0
                or canonical(receipt["initial_optimizer_state_entries"]) != canonical({"generator": 0, "discriminator": 0})
                or type(receipt["completed_steps"]) is not int or receipt["completed_steps"] != STEPS
                or receipt["source_contract_sha256"] != digest(receipt["source_contract"])
                or receipt["full_recipe_sha256"] != digest(receipt["full_recipe"])
                or canonical(receipt["full_recipe"]) != canonical(RESOLVED_RECIPE)
                or receipt["source_contract"]["task_id"] != TASK_ID):
            raise ValueError("actual Noisy AE type/kernel/table/recipe/source/clock receipt differs")
        files = receipt["source_contract"]["files"]
        declared_files = deepcopy(SOURCE_PINS)
        own_file = Path(__file__).resolve()
        declared_files["experiments/forge/atlas_noisy_ae.py"] = dict(
            sha256=hashlib.sha256(own_file.read_bytes()).hexdigest(), bytes=own_file.stat().st_size)
        if (canonical(files) != canonical(declared_files)
                or receipt["source_contract"]["parent_task_id"] != "ae_gan_hold"
                or receipt["source_contract"]["external_max_steps"] != STEPS
                or canonical(receipt["source_contract"]["actual_prior"]) != canonical(ACTUAL_PRIOR)
                or canonical(receipt["source_contract"]["observation"]["observations"]) != canonical(CLOCKS)
                or receipt["source_contract"]["observation"]["selected_or_averaged_sampling"] is not False
                or receipt["source_contract"]["observation"]["evaluation_DV12"] is not False):
            raise ValueError("complete actual source/parent/observer contract differs")
        kernel = receipt["prior_kernel"]
        if (kernel["kind"] != "noisy_particle_cloud"
                or not math.isclose(kernel["sigma"], .025, rel_tol=1e-7, abs_tol=0.)
                or kernel["standardize"] is not False or kernel["learned_width"] is not False
                or kernel["row_weights"] != "uniform"
                or kernel["code_path"] != "particlegan.noisy_particle_prior.NoisyParticlePrior"):
            raise ValueError("fixed original .025 raw uniform prior kernel differs")
        if (evidence["scoring_weights"] != "live"
                or evidence["sampling_law"] != task["evaluation"]["sampling_law"]
                or evidence["eval_output_noise"] != "public_recipe_schedule"
                or canonical([x["step"] for x in evidence["observations"]]) != canonical(CLOCKS)
                or canonical([x["step"] for x in evidence["measurement_purity"]]) != canonical([0, *CLOCKS])
                or any(x["pure"] is not True or x["evaluation_DV12"] is not False
                       or x["unintended_rng_deviations"] != 0 for x in evidence["measurement_purity"])):
            raise ValueError("original250/24/live scheduled observation or purity differs")
        guards = evidence["guards"]
        expected_counts = {role: STEPS for role in ("generator", "encoder", "prior", "discriminator", "noise")}
        expected_hooks = {name: STEPS for name in ("begin_step", "before_critic_backward", "after_critic_step",
            "before_generator_backward", "after_generator_backward", "after_generator_step", "finish_step")}
        if (guards["all_finite"] is not True or guards["public_policy_state_finite"] is not True
                or guards["hooks_exercised"] is not True or guards["unintended_rng_deviations"] != 0
                or canonical(guards["optimizer_updates"]) != canonical(expected_counts)
                or canonical(receipt["ordered_lifecycle_calls"]) != canonical(expected_hooks)):
            raise ValueError("full original AE role/optimizer/ordered lifecycle health is absent")
        return None
    except (KeyError, TypeError, ValueError, OSError) as error:
        return {"status": "INVALID", "reason": str(error)}


def run_behavior(request, task, output_dir, device="cpu"):
    """Ordinary Queue bridge; same actual AE250/24 owner, only prior type changed."""
    from .api import CapabilityError
    from .contracts import atomic_json
    if str(device) != "cpu":
        raise CapabilityError(["fixed Noisy AE task requires original CPU1/noGPU"])
    root = Path(request["source"]["snapshot_path"]).resolve()
    output = Path(output_dir).resolve()
    errors = blockers(task, request["candidate"], root=root)
    if errors:
        raise CapabilityError(errors)
    admission = _admitted_ordinary_attempt(request, task, output)
    guard = lambda: source_guard(request, root)
    guard()
    binding = resolve_binding(root, request["candidate"], task, request["protocol"])
    fresh = _OrdinaryFreshOwner(guard, admission, output)
    owner_request = dict(candidate=request["candidate"], task=task, protocol=request["protocol"], binding=binding)
    started = time.monotonic()
    result = run_case(root, owner_request, output_dir=output, device="cpu",
                      source_guard=guard, fresh_repeat_guard=fresh)
    retained = result.pop("retained_goal_states")
    result.pop("complete_state")  # Tensors remain outside the ordinary JSON receipt.
    initial = fresh.initialization
    applied = result["applied"]
    result["evidence"]["noisy_prior686"] = dict(schema="forge_atlas_noisy_ae_evidence_v1",
        task_id=TASK_ID, parent_task_id="ae_gan_hold", actual_prior=deepcopy(ACTUAL_PRIOR),
        public_prior_type=initial["public_prior_type"], prior_kernel=initial["prior_kernel"],
        table_shape=initial["table"]["shape"], same_location_parameter=initial["object_bindings"]["policy_table"],
        table_optimizer_alias=initial["object_bindings"]["table_optimizer"], initial_completed_steps=initial["completed_steps"],
        initial_optimizer_state_entries=initial["optimizer_state_entries"], completed_steps=fresh.owner.policy.completed_steps,
        full_recipe=applied["recipe"], full_recipe_sha256=digest(applied["recipe"]),
        source_contract=applied["source_contract"], source_contract_sha256=applied["source_contract_sha256"],
        ordered_lifecycle_calls=applied["lifecycle_calls"])
    if [row["step"] for row in retained] != CLOCKS:
        raise ValueError("retained original AE observation clocks differ")
    # Existing captured scored tensors only: no evaluation or media redraw here.
    result["evidence"]["saved_ae_observations"] = [dict(step=row["step"],
        generated=row["generated"].tolist(), reconstructed=row["reconstructed"].tolist(),
        target=row["target"].tolist(), prior=row["prior"].tolist(), anchors=row["anchors"].tolist(),
        metrics=deepcopy(row["metrics"])) for row in retained]
    result["evidence"]["ae_goal"] = dict(original_thresholds=deepcopy(task["evaluation"]["thresholds"]),
        measurement_owner="original_live_G_E_with_scheduled_output_noise", extra_evaluation_draws=0)
    applied.update(initialization=initial, admission=admission, retained_state_exported=False)
    invalid = validate_evidence(task, result["evidence"])
    if invalid is not None:
        raise ValueError(invalid["reason"])
    result["cost"] = dict(wall_seconds=time.monotonic() - started)
    guard()
    atomic_json(output / "result.json", result)
    return result
