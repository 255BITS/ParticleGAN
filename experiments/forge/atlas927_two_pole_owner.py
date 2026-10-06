"""927 table-only AMSGrad ablation; importing this module is inert.

The base79 Recipe stays pinned; the direct table group alone disables AMSGrad.
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
import os
import sys
import time
from importlib.machinery import NamespaceLoader
from pathlib import Path
import struct
from types import SimpleNamespace

SCHEMA = "forge_atlas927_two_pole_optimizer_binding_v1"
OWNER_SCHEMA = "forge_atlas927_two_pole_initial_owner_v1"
STEPS = 80
CLOCKS = [math.ceil(i * STEPS / 24) for i in range(1, 25)]
TASK_PATH = "configs/forge/tasks/two_pole.json"
CONFIG_PATH = "configs/100gaussians/atlas.json"
HOST_PATH = "benchmarks/locked_shared/two_pole.py"
PROTOCOL_PATH = "configs/forge/protocols/screening.json"
TASK_BINDINGS = dict(num_particles=12, z_dim=1, batch_size=12,
                     sigma_rel=0., standardize=False)
SOURCE_PINS = {'benchmarks/legacy/locked_shared.py': {'sha256': '9bac7def8e5c10d4d4532723ebbf7d7689d5e85c8823dbf07fb7225f91b455c0', 'bytes': 4756}, 'benchmarks/locked_shared/baseline.py': {'sha256': '8f1f58fc9cb90159c555788e593769d50070deaa078d0637bf881cf074f6680b', 'bytes': 30125}, 'benchmarks/locked_shared/observation.py': {'sha256': 'bd6f9845b44f1ec2a58d445727990ba5068c7aca3b6f981cf38d738a37c4513b', 'bytes': 4351}, 'benchmarks/locked_shared/two_pole.py': {'sha256': 'eee207af6d12475f7dbb106087f9d9e269d284274e69e4d199f6d8a38cb54f5e', 'bytes': 6753}, 'benchmarks/transfer_suite/protocol.py': {'sha256': '99469b022b790a18a74021a6fe49424d95f535afaa220643688a1ddd7a70ab89', 'bytes': 9964}, 'configs/100gaussians/atlas.json': {'sha256': 'a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4', 'bytes': 1798}, 'configs/forge/ideas/atlas-existing-mog-nearest-positive791-v1.json': {'sha256': 'f0b58b467852f76dd40ab0cf2d4543b62df577995c7faa72bf32e036251e23f2', 'bytes': 1767}, 'configs/forge/ideas/atlas-original-two-pole-passive889-v1.json': {'sha256': '253800cb61c8cb89da2331654b641191931cc02b15ad96f9d4e9da2299dd96a9', 'bytes': 1550}, 'configs/forge/ideas/atlas-two-pole-particle-amsgrad-off927-v1.json': {'sha256': 'a64c72cec3c12f127c15e07fab155564a37b62e1f2432215553ad88618557517', 'bytes': 2284}, 'configs/forge/protocols/screening.json': {'sha256': '3fefb4d47fd2cd8aa6ed110c0a9f5bffefaae700431d1f57ca7b162c8efbb803', 'bytes': 972}, 'configs/forge/task-variants/noisy-prior686/two_pole_noisy_prior686_v1.json': {'sha256': '8ff9d2e25fc2a1c5ebb432e9aaeebcdac8483b29ba5ec777342901ede7a7ed2f', 'bytes': 3472}, 'configs/forge/tasks/two_pole.json': {'sha256': '55ac2d3883ba6c173da304fa7f10648a0b559c202fc35b451b1d0c8870f61cf5', 'bytes': 3099}, 'configs/forge/views/atlas_original_two_pole_passive889_v1.json': {'sha256': '0e581c815faa6a91d372a94a30456e94bb0f51182f13d475ec7aa259e3f57894', 'bytes': 838}, 'configs/forge/views/atlas_two_pole_particle_amsgrad_off927_v1.json': {'sha256': '54d0afcf2443ed06851230de15986d3b59def59c301425d204910d56253fad8a', 'bytes': 1627}, 'experiments/forge/adapters.py': {'sha256': '7757e74b16f916adf42b1b2b3bcc595085e5d7ea7d2fa78c5fb87db6ff5c94f0', 'bytes': 40019}, 'experiments/forge/api.py': {'sha256': '671e12770c9a6fed73bcbf58de0354f9003e21c380826e90580ccd70bd727e18', 'bytes': 52201}, 'experiments/forge/atlas889_contract.py': {'sha256': 'e4850919918d913600dce641f1c1e421c4e1fe483a4dedac74a520c03458043d', 'bytes': 5520}, 'experiments/forge/atlas889_two_pole_observer.py': {'sha256': '0c2070cbaa5da9f056ded548b629165ace50746472aa45e0dc8abd20a4e2a646', 'bytes': 23311}, 'experiments/forge/atlas927_contract.py': {'sha256': '7a2be1b94d723776a4a938868cefc8cc6962fd37946d5b41e1fe0c49ea15e01d', 'bytes': 10860}, 'experiments/forge/atlas927_reference.py': {'sha256': 'f582b24ef950b065e48618300176161488a0174955af337a1c8495a2a663421e', 'bytes': 8247}, 'experiments/forge/atlas_existing_mog.py': {'sha256': 'bc6ef1f03b623560762c0f1555eebdf08818c9796f0e7b3002e5ffe64105a387', 'bytes': 49246}, 'experiments/forge/atlas_existing_mog_ae.py': {'bytes': 54426, 'sha256': 'c2a2e2b5928859036a4ce11d19ae91a3d4199ed162008440bd891815ef04af07'}, 'experiments/forge/atlas_noisy_ae.py': {'sha256': '391a8255b26ed44195d3c148ea0983578acfdfa1885ab37372848d82e85528d4', 'bytes': 53563}, 'experiments/forge/atlas_two_pole.py': {'sha256': '4bc5b7c7acb7f3adc4af6dcbe1d7e1388120815522f3071452d787c5c351ab04', 'bytes': 48584}, 'experiments/forge/boundaries.py': {'sha256': '9440e0a27b997f7b808a411b1965206ea22e71bc072c656d2bc8f91c17821f85', 'bytes': 18676}, 'experiments/forge/evaluate.py': {'sha256': '458aa3babbac76ca8bed8c53e21bbe97544d5d71e71c2a6269ef767089ed1d36', 'bytes': 2177}, 'experiments/forge/initialization.py': {'sha256': 'c2674617c306e24337f1dbbf910d2110b7c852557b9c3e43b47f7fe1dce26634', 'bytes': 1489}, 'experiments/forge/mechanisms.py': {'sha256': '9e4e1fe1f8cb9a24148cab88b7bf2f980e805a2e488c16a982768519bbe1b23f', 'bytes': 11826}, 'experiments/forge/noisy_prior_adapters.py': {'sha256': '157c15b13445c92da10f4a5634743216364ab43c29320f3c52e404638ba851b4', 'bytes': 12074}, 'experiments/forge/noisy_prior_tier1.py': {'sha256': 'd8eacc15850c803cb907a39c8b09dbe2777c35bb946e7d8170cff224582c3f4a', 'bytes': 8177}, 'experiments/forge/planning.py': {'sha256': 'a17cb2d13cfbd1f66cff0658effa1124f37d26ad5afaa3ab362ab6a17ef6b81f', 'bytes': 20821}, 'experiments/forge/policy_adapters.py': {'sha256': '714046ea1655f1946e09dc2aeed27c43cf93f2841237c2be47f0f0049a5df9fa', 'bytes': 17992}, 'experiments/forge/priors.py': {'sha256': '4267847c5e116ae9eeacad3a66980bde5a4eb220c89e7182e56b108f9478bda4', 'bytes': 2964}, 'experiments/forge/rng.py': {'sha256': 'ae7b9a8d61da42136d970188a6f168e03e2d7c5b90cf6fdbd93ad929aba2f293', 'bytes': 6893}, 'experiments/forge/sampling.py': {'sha256': '10e7c17dd4ca30c1f2f3797aa323690db7f4bfdbfe295f23770bb6293ba3aab1', 'bytes': 9148}, 'experiments/forge/studies.py': {'sha256': '23180555910bee7ac3bf0f8465dc5e0bb4d222a3ad6fc763dcfbefaf2c2c49a0', 'bytes': 18047}, 'experiments/forge/taskrecipes.py': {'sha256': 'c11739bd935256e7083875f4e67d6c29fad1b9ea5915976dd32ed1eb7531d285', 'bytes': 3525}, 'experiments/forge/views.py': {'sha256': '62ade2d5e9e7c2019bfc0437effd0f8e0343a87f08eb151f51c4912428827136', 'bytes': 41404}, 'particlegan/__init__.py': {'bytes': 1886, 'sha256': '18506efd720327de46f4bd5f4f66dab6e1b0574c5388e5496795400a45aa4070'}, 'particlegan/_qr.py': {'bytes': 1951, 'sha256': 'b59cd62cc27ed557b41b2d17ed86ed578c762a3a8a392d63731a5a3b2b018916'}, 'particlegan/anchor_birth.py': {'bytes': 6401, 'sha256': 'aa50983d8b61304acf4e1138f16fd4784798983df91e931863bcacd3b511f76f'}, 'particlegan/autoencoder.py': {'bytes': 6194, 'sha256': 'f3fbb9184e9f44112e15eccf0c0246dfba5ed490d3c7b81b0f837a5335916402'}, 'particlegan/birth_death.py': {'sha256': '7482b29f0065be446a6b5e087a4b55d0961c8a6f2602cf9919ca502edca78c91', 'bytes': 48741}, 'particlegan/birth_phase.py': {'bytes': 21144, 'sha256': '5ff929e5e9858aa24c4cc2f8999b5d0603e25e42ca3844d1da4cc28b1981d3d7'}, 'particlegan/capabilities.py': {'bytes': 2548, 'sha256': '74f57e8485bd72193e535c4f78978b5f908a6c965bb417c2fd57b2cf446720a0'}, 'particlegan/conditioning.py': {'bytes': 4878, 'sha256': '8734fe338868ace4b5851a8fdb1f071e46d47396f66a3765e6b1ef37f0481486'}, 'particlegan/continuous.py': {'bytes': 60875, 'sha256': '4ceab49c7d51d1769ae91b7f8bc892eaf380ca6fbd77a948a086ade6627e9a41'}, 'particlegan/diffusion.py': {'bytes': 5100, 'sha256': '19c3aa8772703891551b9122fde8e74c07996f779892b7cb6be09553691efa61'}, 'particlegan/discriminators.py': {'bytes': 5833, 'sha256': '0e2efb125ffb314577612ab7a2eba66b0a1a2d28ad25f403f42c18b6f6ee333f'}, 'particlegan/feature_cells.py': {'bytes': 125325, 'sha256': 'e0ea05f61abd7845c7437eed9a79c3a2511d5551ff04f4e619b6b9c8992e221a'}, 'particlegan/feature_policy.py': {'bytes': 14650, 'sha256': 'de1a50f70a37d4c9174c8850ce3eac2a2597ed01f19fc4fe1a2ce0ec3176a8b2'}, 'particlegan/feature_reference.py': {'bytes': 32368, 'sha256': '58e076e9e04600be25e28afafc537a340de727371f4acbbfb2b7dd8bebac18b1'}, 'particlegan/gan_loss.py': {'bytes': 3203, 'sha256': '2ba031aa3c162bc6c4666c7dd43a2aac295ed2a841f7855e6b0a28b29eca6b2f'}, 'particlegan/grad_regularizers.py': {'bytes': 14853, 'sha256': 'a540d05a4a992540b6a5f3b5ff7ccc11ebaf18592135b87c95638adbe68aa747'}, 'particlegan/init.py': {'bytes': 24559, 'sha256': 'e15d4de01eb49f7097bc249abacab907ff067385412346f54ab0d510cb8649d0'}, 'particlegan/k3p.py': {'bytes': 33012, 'sha256': '200ead2c27ab0b2aa6068f1b1b1b5c826b8125aa8602bfec08b31aaa2894401c'}, 'particlegan/ka2.py': {'bytes': 16551, 'sha256': '95fdaa58a2bdc229d5d146ef89548bc3fe07582d06181a149ed1bb4f5d632703'}, 'particlegan/mean_transport.py': {'bytes': 39633, 'sha256': '57a68e0a65c5424a4460f65002f3862a69e90af330a4d47c700dc773683a2c06'}, 'particlegan/noisy_particle_prior.py': {'bytes': 3328, 'sha256': 'a59eb7340cae67b0f8f40729bd89cd2071bba63c7ae66d516246554b6c621316'}, 'particlegan/output_moments.py': {'bytes': 8387, 'sha256': 'dcf3e27228d2c3c9738e473c5b22ee9e6d5047ddcd60c538b66526adb53ef188'}, 'particlegan/particle_prior.py': {'bytes': 17899, 'sha256': '0220878bebea227da63abbf9f5b1ebdabdc6aea54fd2332fd2cab02ed734238f'}, 'particlegan/policy.py': {'sha256': '7a6ac1410260d720f58b6903310443f3f8dc3a283c1f3b5abd921965835a4967', 'bytes': 81173}, 'particlegan/population_continuity.py': {'bytes': 20329, 'sha256': '378cd75cca45bc8da4da33e686a12cf99df853042687978314e37c532bc17fe6'}, 'particlegan/recipe_schedules.py': {'bytes': 6044, 'sha256': 'd05e459b3e9e5f938b01a854273b74343f2b6d382bec4ae5151dd1f1cd032500'}, 'particlegan/recipes.py': {'sha256': '79905bb948cb967f80a087063960ea72e72c04d3b572b4b90ff4fa1dda5f299f', 'bytes': 56022}, 'particlegan/routing.py': {'bytes': 64904, 'sha256': '63249d808fccb3d4e113eb1d638a245ee665c3cade20a1542bb706f316b4abcc'}, 'particlegan/row_evidence.py': {'bytes': 9397, 'sha256': '78be40141ef3946860458e9842159be9321f05c217a4800dd4134113b6eba93e'}, 'particlegan/training.py': {'bytes': 40093, 'sha256': '740c01b14ce4542f5fcef43c59f9eed09b122f349a2308fb17bcc87e97786eb2'}, 'particlegan/vicreg_loss.py': {'bytes': 2297, 'sha256': 'ab1c4dc266dec2c35337f449917240eb38afede7590d2154b63f6f471dc45a36'}, 'tests/test_atlas889_two_pole_observer.py': {'sha256': '52f82b6b0bd5bc313ee341466463cee415d61e9804658dfc554280931d144145', 'bytes': 29610}, 'tests/test_atlas921_completed_lr_clock.py': {'sha256': '3da79006816bcb88488333c200f55077af5d6e3c97f2dafbbedd863fc7279a40', 'bytes': 13934}}


TASK_DECLARATION = {'schema_version': 1, 'id': 'two_pole', 'adapter': 'transfer_behavior', 'execution': {'initializer': 'deterministic_orthogonal', 'host': 'two_pole', 'steps': 80, 'prior': {'kind': 'particle_cloud', 'sigma': 0.0, 'standardize': False, 'learnable': True, 'exception_reason': 'This host optimizes explicit sample-particle coordinates directly, without a separate generator or latent sampling kernel.'}, 'protocol': 'screening', 'produces_state': False, 'host_source': 'benchmarks/locked_shared/baseline.py', 'host_definition': {'family': 'existing_behavior', 'runner': 'legacy', 'split': 'development', 'phase': 'fit', 'steps': 80}, 'execution_path': 'public_components', 'device': 'cpu', 'schedule_policy': 'public_recipe_with_frozen_host_horizon', 'fixed_initialization': {'critic': 'stored_host_weights', 'particles': 'zeros'}}, 'evaluation': {'sampling_contract_version': 1, 'sampling_law': 'learned_particles_and_critic_gradient', 'eval_output_noise': 'not_applied_to_measurement', 'kind': 'transfer_sustained', 'evaluator': 'benchmarks.transfer_suite.protocol:test_verdict', 'thresholds': [['mean_abs', '>=', 0.3], ['grad_med', '<=', 1.0]], 'observations': 24, 'minimum_stable_checks': 5, 'scoring_weights': 'live', 'sources': {'benchmarks/transfer_suite/protocol.py': '99469b022b790a18a74021a6fe49424d95f535afaa220643688a1ddd7a70ab89', 'benchmarks/locked_shared/observation.py': 'bd6f9845b44f1ec2a58d445727990ba5068c7aca3b6f981cf38d738a37c4513b', 'benchmarks/locked_shared/baseline.py': '8f1f58fc9cb90159c555788e593769d50070deaa078d0637bf881cf074f6680b'}, 'guards': {'finite_state': True, 'optimizer_roles': ['prior', 'discriminator'], 'mechanism_exercised': True, 'rng_isolation': True}, 'evaluator_revision': {'id': 'explicit-equality-and-word-prior-betas-v3', 'previous_source_sha256': {'benchmarks/transfer_suite/protocol.py': '7626dd8236c0765645c304891d8fa79be0c05fd10876a0694d58db831ff2dc97', 'benchmarks/locked_shared/observation.py': '04067872ce228e4702e48c440222cd2f553ffb30b10ce113a4d52101e9abd9d8', 'benchmarks/locked_shared/baseline.py': '72ebe4ef6d3cfe28f1ab6073fc781a72b45173f3ce2a0b63040e84b27bed03cc'}, 'previous_revision': None, 'change': 'Correct == grading and reject unsupported operators; >= and <= numerical gates unchanged. Word fixture now honors declared prior_betas. Bind current sources for future execution only; original receipts and verdicts remain unchanged.'}}, 'resources': {'gpus': 0, 'gpu_memory_mb': 0, 'cpu_threads': 1, 'timeout_seconds': 300, 'allow_cpu': True}, 'requires_capabilities': ['named_rng', 'live_sampling', 'learned_locations', 'particle_cloud'], 'dependencies': []}
BASE_RECIPE_SHA256 = "d5f20a8c4a9a7a3e2f0ac6d4562ae0a364677ffe73611ff9f9b7432434024b95"
OPTIMIZER_VARIANT = {'schema': 'forge_atlas927_two_pole_particle_amsgrad_variant_v1', 'base_recipe_sha256': 'd5f20a8c4a9a7a3e2f0ac6d4562ae0a364677ffe73611ff9f9b7432434024b95', 'groups': [{'optimizer': 'generator', 'group_index': 0, 'role': 'table', 'amsgrad': False}, {'optimizer': 'generator', 'group_index': 1, 'role': 'noise', 'amsgrad': True}, {'optimizer': 'critic', 'group_index': 0, 'role': 'critic', 'amsgrad': True}], 'table_lr_clock': 'completed_direct_adam_lr/base_lr'}
NOISY_EFFECTIVE_RECIPE_SHA256 = '6b542fecd875f9a51b67f75a0b7e8975969f2d52ea25eb9d2427a8454900f91b'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()




def _source(root, relative):
    root = (Path(__file__).resolve().parents[2] if root is None else Path(root)).resolve()
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


def is_task(task):
    if not isinstance(task, dict):
        return False
    scientific = {k: v for k, v in task.items() if k not in ("preflight_blockers", "field_ownership")}
    return canonical(scientific) == canonical(TASK_DECLARATION)


def is_noisy_task(task):
    from .noisy_prior_tier1 import is_noisy_task as recognized, validate
    if not recognized(task):
        return False
    try:
        return validate(task)["parent_task_id"] == "two_pole"
    except (ValueError, KeyError, TypeError):
        return False


def supports(task, candidate):
    return is_task(task) and supports_candidate(candidate)


def validate_effective_recipe(recipe, *, noisy=False):
    expected = NOISY_EFFECTIVE_RECIPE_SHA256 if noisy else BASE_RECIPE_SHA256
    if not isinstance(recipe, dict) or len(recipe) != 79 or digest(recipe) != expected:
        raise ValueError("ordinary two-pole requires the complete unchanged base79 Atlas recipe and fixed host binding")


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
    noisy = is_noisy_task(task)
    if noisy:
        from .noisy_prior_adapters import _parent
        if _parent(task, root=root) != task_source:
            raise ValueError("Noisy two-pole differs from the complete original parent")
    if (not noisy and _declaration(task) != task_source) or task_source["id"] != "two_pole":
        raise ValueError("only the exact canonical two_pole declaration is supported")
    if not supports(task, candidate):
        raise ValueError("the unchanged current Atlas preset is required; tuning and other tasks are unsupported")
    protocol = json.loads(_source(root, PROTOCOL_PATH)) if protocol is None else protocol
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
    if noisy:
        fields["prior_kind"] = "noisy_particles"
    # Recipe.__post_init__ owns this explicit legacy-arm normalization.
    if fields["reg_arm"] is not None:
        fields["critic_formulation"] = "k3p"
    fields = json.loads(canonical(fields))
    validate_effective_recipe(fields, noisy=noisy)
    if fields["total_steps"] is not None or fields["continuous_policy"] != "dv12":
        raise ValueError("original schedule-free DV12 Recipe must be preserved")
    contract = dict(schema="forge_atlas927_two_pole_public_owner_contract_v1",
        files=deepcopy(SOURCE_PINS), task_id=task["id"], external_max_steps=80,
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
        learning_rate_owner="public UpdatePolicy StationarityLR; direct bank uses Recipe.lr without latent prior multipliers",
        lifecycle="particlegan.UpdatePolicy_ordered_public_update",
        mechanism_audit="maintained_public_MechanismAudit_base_Recipe_synthetic_probes_do_not_validate_effective_optimizer",
        base_recipe_sha256=BASE_RECIPE_SHA256,
        effective_optimizer_variant=deepcopy(OPTIMIZER_VARIANT),
        effective_optimizer_variant_sha256=digest(OPTIMIZER_VARIANT),
        scientific_credit=False)
    if noisy:
        contract["prior_type_substitution"] = dict(cohort=task["task_cohort"],
            parent=deepcopy(task["prior_substitution_parent"]), actual_type="NoisyParticlePrior",
            sigma=0., same_table_alias=True, sampler_calls=0, constructor_rng_draws=0,
            birth_death="same direct bank remains owned", scientific_credit=False)
    return dict(schema=SCHEMA, recipe=fields,
                base_recipe_sha256=BASE_RECIPE_SHA256,
                effective_optimizer_variant=deepcopy(OPTIMIZER_VARIANT),
                effective_optimizer_variant_sha256=digest(OPTIMIZER_VARIANT),
                **({"noisy_task": _declaration(task)} if noisy else {}),
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
        if self.prior is not None:
            from .noisy_prior_adapters import observation_receipt
            self.noisy_observations.append(observation_receipt(self.binding["noisy_task"], self.policy))
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
                particles=self.table.detach().cpu().clone(), gradient=gradient.detach().cpu().clone(),
                target=self.real.detach().cpu().clone()))


def install_completed_table_lr_clock(owner):
    """Use the completed direct Adam LR for this owner's table settling clock.

    Adam and restored groups are unchanged. The capture is transient, consumed
    once, cleared on failure/restore, and absent from every checkpoint schema.
    No callback is added to the vars-serialized table tester.
    """
    policy, optimizer, table = owner.policy, owner.opt_g, owner.table
    if (owner.prior is not None or tuple(table.shape) != (12, 1)
            or str(table.device) != "cpu" or policy.table is not table
            or policy.table_optimizer is not optimizer or policy.optimizers[0] is not optimizer):
        raise ValueError("LR clock correction requires the original direct CPU12x1 owner")
    matches = [(j, group) for j, group in enumerate(optimizer.param_groups)
               if len(group["params"]) == 1 and group["params"][0] is table]
    if len(matches) != 1 or optimizer.direct_response is None:
        raise ValueError("LR clock correction requires one owned direct table group")
    j, group = matches[0]
    response = optimizer.direct_response
    if len(response.params) != 1 or response.params[0] is not table or policy.lr_settle.testers[0][j] is None:
        raise ValueError("LR clock correction requires the original table response/tester aliases")
    original_end, original_step = response.end, optimizer.step
    original_settle, original_load = policy._settle_observe, policy.load_state_dict
    capture = dict(active=None, ended=0, lr=None, token_valid=False, completed=None)

    def clear():
        capture.update(active=None, ended=0, lr=None, token_valid=False, completed=None)

    def end(token):
        active = capture["active"] is not None
        valid = token is None or (isinstance(token, tuple) and len(token) == 3 and token[0] is group)
        # K3P calls end in finally. This is attempted-step evidence until the
        # delegated optimizer step returns successfully, including restoration.
        used_lr = float(group["lr"]) if active and valid and token is not None else None
        value = original_end(token)
        if active:
            capture["ended"] += 1
            capture["lr"], capture["token_valid"] = used_lr, valid
        return value

    def step(*args, **kwargs):
        if capture["completed"] is not None or capture["active"] is not None:
            raise RuntimeError("previous direct Adam LR was not consumed")
        clear()
        clock = policy.completed_steps
        capture["active"] = clock
        try:
            value = original_step(*args, **kwargs)
            if capture["ended"] != 1 or not capture["token_valid"]:
                raise RuntimeError("completed direct Adam step has no exact restoration witness")
            lr = capture["lr"]
            if lr is not None and (not math.isfinite(lr) or lr < 0.):
                raise RuntimeError("completed direct Adam LR is invalid")
            capture["completed"] = (clock, lr)
            capture["active"] = None
            return value
        except BaseException:
            clear()
            raise

    def settle(index):
        if index != 0:
            return original_settle(index)
        completed = capture["completed"]
        if completed is None or completed[0] != policy.completed_steps:
            clear()
            raise RuntimeError("table settling requires this update's completed direct Adam LR")
        used_lr = completed[1]
        clear()  # one-shot, even if the original consumer raises
        if used_lr is None:
            # No direct token means no table gradient/Adam update. Preserve the
            # original no-gradient consumer path, without reusing an old LR.
            return original_settle(index)
        base = float(policy.initial_lrs[0][j])
        if not math.isfinite(base) or base <= 0.:
            raise RuntimeError("table settling base LR is invalid")
        # Exact original loop/order: only the owned table ratio differs.
        # In particular surprise.observe still reads the restored group.
        for k, (current, rate, tester) in enumerate(zip(optimizer.param_groups,
                policy.initial_lrs[0], policy.lr_settle.testers[0])):
            if tester is not None:
                ratio = used_lr / base if current is group else current["lr"] / rate
                tester.observe(current["params"], ratio, step=policy.completed_steps + 1)
                if policy.surprise is not None:
                    policy.surprise.observe(f"0.{k}", optimizer, current)

    def load(*args, **kwargs):
        nonlocal j, group
        clear()
        value = original_load(*args, **kwargs)
        # Adam restores fresh group dictionaries. Rejoin the live table alias
        # after successful original restore; no parameter/state is replaced here.
        matches = [(k, current) for k, current in enumerate(optimizer.param_groups)
                   if len(current["params"]) == 1 and current["params"][0] is table]
        if len(matches) != 1 or policy.table_optimizer is not optimizer:
            raise RuntimeError("restored direct table group alias is invalid")
        j, group = matches[0]
        if policy.lr_settle.testers[0][j] is None:
            raise RuntimeError("restored direct table tester is absent")
        return value

    response.end, optimizer.step = end, step
    policy._settle_observe, policy.load_state_dict = settle, load


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


def make_particle_optimizer(recipe, table):
    """Only the table group overrides the unchanged base Recipe/defaults."""
    validate_effective_recipe(asdict(recipe))
    return recipe.make_generator_optimizer(
        [{"params": [table], "forge_role": "prior", "amsgrad": False}],
        direct_particles=[table])


def validate_optimizer_variant_receipt(receipt):
    if (not isinstance(receipt, dict) or receipt.get("base_recipe_sha256") != BASE_RECIPE_SHA256
            or canonical(receipt.get("effective_optimizer_variant")) != canonical(OPTIMIZER_VARIANT)
            or receipt.get("effective_optimizer_variant_sha256") != digest(OPTIMIZER_VARIANT)
            or canonical(receipt.get("optimizer_defaults")) != canonical({"generator_amsgrad": True, "critic_amsgrad": True})
            or canonical(receipt.get("object_bindings")) != canonical({"table": True, "noise": True, "critic": True})):
        raise ValueError("actual table-only AMSGrad-off927 optimizer law is absent")
    return receipt


def optimizer_variant_receipt(owner):
    """Read the actual constructed group flags/roles; never alter an optimizer."""
    validate_effective_recipe(asdict(owner.recipe))
    g, d = owner.opt_g.param_groups, owner.opt_d.param_groups
    if len(g) != 2 or len(d) != 1:
        raise ValueError("927 requires the original table/noise/critic group roster")
    bindings = dict(table=len(g[0]["params"]) == 1 and g[0]["params"][0] is owner.table
        and owner.policy.table_optimizer is owner.opt_g,
        noise=len(g[1]["params"]) == 1 and g[1]["params"][0] is owner.policy.log_output_sigma,
        critic=[id(p) for p in d[0]["params"]] == [id(p) for p in owner.critic.parameters()])
    flags = [g[0].get("amsgrad"), g[1].get("amsgrad"), d[0].get("amsgrad")]
    defaults = [owner.opt_g.defaults.get("amsgrad"), owner.opt_d.defaults.get("amsgrad")]
    if any(type(v) is not bool for v in flags + defaults):
        raise ValueError("927 AMSGrad group/default flags must be actual booleans")
    variant = deepcopy(OPTIMIZER_VARIANT)
    for declaration, flag in zip(variant["groups"], flags):
        declaration["amsgrad"] = flag
    receipt = dict(base_recipe_sha256=digest(asdict(owner.recipe)),
        effective_optimizer_variant=variant, effective_optimizer_variant_sha256=digest(variant),
        optimizer_defaults=dict(generator_amsgrad=defaults[0], critic_amsgrad=defaults[1]),
        object_bindings=bindings)
    return validate_optimizer_variant_receipt(receipt)


def construct_owner(root, binding, *, source_guard, context=None):
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
    expected = resolve_binding(root, json.loads(_source(root, "configs/forge/ideas/" + CANDIDATE_ID + ".json")),
        binding.get("noisy_task", json.loads(_source(root, TASK_PATH))), json.loads(_source(root, PROTOCOL_PATH)))
    if binding != expected:
        raise ValueError("forged effective Recipe or owner contract before model construction")
    recipe = Recipe(**binding["recipe"])
    if (canonical(asdict(recipe)) != canonical(binding["recipe"])
            or canonical(recipe.to_dict()) != canonical(binding["recipe_serialized"])):
        raise ValueError("actual complete public Recipe differs before model construction")
    owner = TwoPoleOwner()
    owner.torch, owner.recipe, owner.binding = torch, recipe, deepcopy(binding)
    owner.streams = NamedStreams(0, device="cpu") if context is None else context.streams
    owner.restored, owner.construction_id = False, object()
    with owner.streams.fork("init", component="discriminator", purpose="construction", device="cpu"):
        owner.critic = two_pole.HostCritic()
    owner.table = nn.Parameter(torch.zeros(12, 1))
    owner.prior = None
    if "noisy_task" in binding:
        from particlegan.noisy_particle_prior import NoisyParticlePrior
        _loaded_source(root, NoisyParticlePrior, "particlegan/noisy_particle_prior.py")
        owner.prior = NoisyParticlePrior.from_table(owner.table, sigma=0.)
    owner.generator = nn.Identity()
    owner.opt_g = make_particle_optimizer(recipe, owner.table)
    owner.opt_d = recipe.make_critic_optimizer(owner.critic, ema_critic=deepcopy(owner.critic))
    penalty = recipe.make_critic_penalty(owner.opt_d, collect_stats=True)
    streams = {"latent_generator": owner.streams.generator("prior", component="latent", purpose="indices"),
        "penalty_generator": owner.streams.generator("noise", component="penalty", purpose="training"),
        "noise_generator": owner.streams.generator("noise", component="generator", purpose="output"),
        "eval_generator": owner.streams.generator("eval", component="sampler", purpose="samples")}
    owner.policy = UpdatePolicy(recipe, owner.generator, owner.critic, prior=owner.prior, table=owner.table,
        generator_optimizer=owner.opt_g, critic_optimizer=owner.opt_d, table_optimizer=owner.opt_g,
        roles=[["table"], ["critic"]], streams=streams, seed=0, penalty=penalty)
    install_completed_table_lr_clock(owner)
    optimizer_variant_receipt(owner)
    owner.loss, owner.particle_l2 = recipe.make_loss(), LOCKED_SHARED.particle_l2
    owner.row_indices = torch.arange(12, dtype=torch.long)
    owner.noise = _PolicyNoise(owner)
    owner.calls = {key: 0 for key in ("begin_step", "after_critic_step", "after_generator_backward",
                                    "after_generator_step", "finish_step")}
    owner.observations, owner.purity, owner.retained, owner._reading_step = [], [], [], None
    owner.noisy_observations = []
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
    from experiments.forge.mechanisms import MechanismAudit
    from experiments.forge.policy_adapters import PolicyLifecycleAudit
    _loaded_source(root, MechanismAudit, "experiments/forge/mechanisms.py")
    _loaded_source(root, PolicyLifecycleAudit, "experiments/forge/policy_adapters.py")
    owner.audit = MechanismAudit(recipe, owner.opt_d, [owner.opt_g])
    owner.lifecycle = PolicyLifecycleAudit(owner.policy)
    if context is not None:
        context.bind_update_policy(owner.policy, external_max_steps=80)
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
    if "noisy_task" in owner.binding:
        from particlegan.noisy_particle_prior import NoisyParticlePrior
        if (type(owner.prior) is not NoisyParticlePrior or policy.prior is not owner.prior
                or owner.prior.z is not owner.table or float(owner.prior.sigma) != 0.):
            raise ValueError("actual unsampled zero-width Noisy prior must wrap the original table")
    elif owner.prior is not None or policy.prior is not None:
        raise ValueError("ordinary direct table must not gain a prior")
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
        base_recipe_sha256=digest(asdict(policy.recipe)),
        optimizer_variant=optimizer_variant_receipt(owner),
        source_contract_sha256=owner.binding["source_contract_sha256"])
    if (receipt["completed_steps"] != 0 or receipt["phase"] != "ready" or receipt["restored"]
            or not receipt["table"]["all_zero"] or receipt["table"]["shape"] != [12, 1]
            or not receipt["critic"]["stored_weights_exact"] or any(receipt["optimizer_state_entries"].values())
            or not all(receipt["object_bindings"].values()) or any(owner.calls.values())
            or owner.opt_d.record.observed_steps != 0 or owner.policy.controller.updates != 0
            or not owner.lifecycle.receipt(0)["complete"]):
        raise ValueError("fresh zero-update original initialization required")
    if owner.prior is not None:
        receipt["prior"] = owner.prior.kernel_contract()
        receipt["prior_table_alias"] = owner.prior.z is owner.table
        receipt["prior_sampler_calls"] = 0
    return receipt


def run_first_case(root, request, task=None, *, source_guard, fresh_repeat_guard, context=None):
    """ROOT-only child: exact original loop, ordinary measurement, no serving draws."""
    task = request["task"] if task is None else task
    binding = resolve_binding(root, request["candidate"], task, request["protocol"])
    if binding != request["binding"]:
        raise ValueError("request effective Recipe/source binding drift")
    owner = fresh_repeat_guard.construct(
        lambda: construct_owner(root, binding, source_guard=source_guard, context=context), owner_initial_receipt)
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
    from .atlas889_two_pole_observer import PassiveTwoPoleRecorder
    observer_install_error = None
    try:
        recorder = PassiveTwoPoleRecorder(owner).install()
    except Exception as error:
        recorder = None
        observer_install_error = type(error).__name__
    try:
        raw = namespace["_owned_train"](owner)
    finally:
        if recorder is not None:
            recorder.uninstall()
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
    from experiments.forge.policy_adapters import finite_policy_state
    all_finite = all_finite and finite_policy_state(owner.policy.state_dict())
    lifecycle = owner.lifecycle.receipt(80)
    if not lifecycle["complete"]:
        raise ValueError("incomplete actual ordered public lifecycle")
    from experiments.forge.mechanisms import mechanism_blockers
    source_guard()
    result = dict(task_id=task["id"], execution_path="public_components", device="cpu", raw=raw,
        evidence=dict(observations=owner.observations, live={k: raw[k] for k in ("mean_abs", "grad_med")},
            scoring_weights="live", sampling_contract_version=1,
            sampling_law="learned_particles_and_critic_gradient", eval_output_noise="not_applied_to_measurement",
            guards=dict(all_finite=all_finite, optimizer_updates=counts,
                hooks_exercised=not mechanism_blockers(mechanism_audit), mechanism_audit=mechanism_audit,
                unintended_rng_deviations=0), measurement_purity=owner.purity),
        applied=dict(recipe=asdict(owner.recipe), recipe_serialized=owner.recipe.to_dict(),
            base_recipe_sha256=digest(asdict(owner.recipe)),
            optimizer_variant=optimizer_variant_receipt(owner),
            source_contract=deepcopy(binding["source_contract"]),
            source_contract_sha256=binding["source_contract_sha256"], policy_owner="particlegan.UpdatePolicy",
            actual_policy_owners={key: type(value).__module__ + "." + type(value).__qualname__
                                  for key, value in actual_owners.items()},
            lifecycle_calls=owner.calls, lifecycle=lifecycle, optimizer_updates=counts, roles=owner.policy.roles,
            ordinary_observation_owner="live_table_and_live_critic", evaluation_sampler_calls=0,
            direct_row_base_lr=owner.policy.initial_lrs[0][0],
            a2_inapplicable=True, prior_lr_mult_consumed=False, prior_betas_consumed=False,
            streams=owner.streams.manifest()), retained_goal_states=owner.retained,
        complete_state=owner.complete_state())
    if owner.prior is not None:
        from .policy_adapters import controls_receipt
        from .noisy_prior_adapters import observation_receipt
        controls = controls_receipt(owner.policy, 80)
        controls["cohort"] = task["task_cohort"]
        result["evidence"].update(policy_controls=controls,
            policy_observations=deepcopy(owner.noisy_observations),
            policy_purity=[dict(row, completed_steps=row["step"]) for row in owner.purity])
    try:
        if recorder is None:
            raise ValueError('observer install failed: ' + str(observer_install_error))
        trace = recorder.receipt(completed_steps=owner.policy.completed_steps,
            scored_clocks=[row['step'] for row in owner.observations])
    except Exception as error:
        trace = dict(schema='forge_atlas889_two_pole_passive_trace_v1', status='UNKNOWN',
            complete=False, unknown=['observer receipt failed: ' + type(error).__name__])
    result['evidence']['two_pole_passive889'] = trace
    source_guard()
    return result


def source_guard(request, root):
    """Use the ordinary snapshot boundary, including loaded module origins."""
    from .sources import verify_snapshot
    root = Path(root).resolve()
    source = request["source"]
    if (root != Path(source["snapshot_path"]).resolve()
            or Path(__file__).resolve() != root / "experiments/forge/atlas927_two_pole_owner.py"):
        raise ValueError("ordinary owner must execute from the actual frozen snapshot")
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
    """Require the ordinary Queue request and inherited execution lease."""
    from .contracts import read_json, stable_hash
    resolved = read_json(output / "request.json")
    if resolved.get("schema_version") != 1 or canonical(resolved.get("request")) != canonical(request):
        raise ValueError("ordinary immutable admitted request differs")
    worker, job = resolved["worker"], resolved["job"]
    if (job["task_id"] != task["id"] or job.get("task_ids", [job["task_id"]]) != [task["id"]]
            or job["budget_seconds"] != 300 or job["resources"]["gpus"] != 0
            or job["resources"]["cpu_threads"] != 1 or worker["device"] != "cpu"
            or Path(worker["directory"]).resolve() != output
            or not isinstance(worker.get("token"), str) or not worker["token"]
            or not isinstance(worker.get("attempt"), str) or not worker["attempt"]
            or job["compatibility_key"] != stable_hash(job["science"])
            or job["science"]["candidate_revision"] != request["candidate_revision"]
            or canonical(job["science"]["protocol"]) != canonical(request["protocol"])
            or job["science"]["compute"]["backend"] != "cpu"):
        raise ValueError("ordinary CPU1 full300 job/worker admission differs")
    lease_fd = int(os.environ["FORGE_LEASE_FD"])
    actual, declared = os.fstat(lease_fd), (output / "execution.lock").stat()
    if (actual.st_dev, actual.st_ino) != (declared.st_dev, declared.st_ino):
        raise ValueError("ordinary inherited lease does not own this attempt")
    if task["id"] not in request["tasks"] or canonical(task) != canonical(request["tasks"][task["id"]]):
        raise ValueError("ordinary request task differs")
    return dict(schema="forge_atlas_two_pole_admitted_owner_v1",
                attempt=worker["attempt"], compatibility_key=job["compatibility_key"],
                source_digest=request["source"]["digest"], candidate_revision=request["candidate_revision"],
                budget_seconds=300, device="cpu", cpu_threads=1, inherited_lease_checked=True)


class _OrdinaryFreshOwner:
    """One admitted factory and its actual zero-update public-owner witness."""
    def __init__(self, guard, admission, output):
        self.guard, self.admission, self.output = guard, admission, output
        self.owner = None
        self.initialization = None

    def construct(self, factory, reader):
        if self.owner is not None or self.initialization is not None:
            raise ValueError("one fresh owner per ordinary attempt; no reuse")
        self.guard()
        owner = factory()
        if type(owner) is not TwoPoleOwner or reader is not owner_initial_receipt:
            raise ValueError("actual source-owned two-pole constructor and reader required")
        receipt = reader(owner)
        self.guard()
        from .contracts import atomic_json
        self.owner, self.initialization = owner, receipt
        atomic_json(self.output / "INITIALIZATION.json", dict(
            schema="forge_atlas_two_pole_initialization_v1", admission=self.admission,
            actual_owner=receipt, pid=os.getpid()))
        start = dict(schema="forge_atlas_two_pole_model_started_v1", admission=self.admission,
                     completed_steps=0, pid=os.getpid(), initialization_sha256=digest(receipt))
        atomic_json(self.output / "MODEL_STARTED.json", start)
        print(canonical(dict(event="actual_model_started", **start)), flush=True)
        return owner


def run_behavior(request, task, output_dir, device="cpu"):
    """Ordinary worker bridge; the existing owner performs all scientific calls."""
    from .api import CapabilityError, task_formulation_context
    from .contracts import atomic_json
    if not supports_candidate(request["candidate"]) or not is_task(task) or str(device) != "cpu":
        raise CapabilityError(["only the unchanged ordinary Atlas two_pole CPU task is supported"])
    root = Path(request["source"]["snapshot_path"]).resolve()
    output = Path(output_dir).resolve()
    admission = _admitted_ordinary_attempt(request, task, output)
    guard = lambda: source_guard(request, root)
    guard()
    context = task_formulation_context(request["candidate"], task, request["protocol"], root=root)
    binding = context.noisy_two_pole_binding if is_noisy_task(task) else context.ordinary_two_pole_binding
    fresh = _OrdinaryFreshOwner(guard, admission, output)
    owner_request = dict(candidate=request["candidate"], task=task,
                         protocol=request["protocol"], binding=binding)
    started = time.monotonic()
    result = run_first_case(root, owner_request, source_guard=guard,
                            fresh_repeat_guard=fresh, context=context)
    retained = result.pop("retained_goal_states")
    result.pop("complete_state")  # Tensor checkpoint data cannot enter ordinary atomic_json.
    observations = {row["step"]: row for row in result["evidence"]["observations"]}
    purity = {row["step"]: row for row in result["evidence"]["measurement_purity"]}
    if [row["step"] for row in retained] != CLOCKS:
        raise ValueError("retained original observation clocks differ")
    # These are the already captured live banks/gradients, not another evaluator draw.
    result["evidence"]["saved_particle_observations"] = [dict(
        step=row["step"], particles=row["particles"].tolist(), critic_gradient=row["gradient"].tolist(),
        target=row["target"].tolist(),
        metrics=deepcopy(observations[row["step"]]), purity=deepcopy(purity[row["step"]]))
        for row in retained]
    result["evidence"]["particle_goal"] = dict(
        poles=[-1., 1.], original_thresholds=deepcopy(task["evaluation"]["thresholds"]),
        measurement_owner="live_table_and_live_critic", extra_evaluation_draws=0)
    result["applied"].update(api=context.receipt(), initialization=fresh.initialization,
                             admission=admission, retained_state_exported=False)
    result['evidence']['existing_mog717'] = direct_control_receipt(task, result)
    result['evidence']['two_pole_passive889'].update(candidate_id=CANDIDATE_ID, task_id='two_pole',
        source_digest=request['source']['digest'], candidate_revision=request['candidate_revision'],
        request_id=request['request_id'])
    result["cost"] = dict(wall_seconds=time.monotonic() - started)
    guard()
    atomic_json(output / "result.json", result)
    return result


# Strict additive method/provenance identity. The original task law is unchanged.
CANDIDATE_ID = 'atlas-two-pole-particle-amsgrad-off927-v1'
VIEW_ID = 'atlas_two_pole_particle_amsgrad_off927_v1'
STUDY_ID = 'atlas-two-pole-particle-amsgrad-off927-study-v1'
TRACK_ID = 'atlas927_two_pole_particle_amsgrad_off'


def is_candidate(candidate):
    return isinstance(candidate, dict) and candidate.get('id') == CANDIDATE_ID


def supports_candidate(candidate):
    return (is_candidate(candidate) and candidate.get('claim_contract', {}).get('experimental_track') == TRACK_ID
        and canonical(candidate.get('claim_contract', {}).get('optimizer_variant')) == canonical(OPTIMIZER_VARIANT)
        and candidate.get('recipe_preset') == 'atlas' and candidate.get('recipe_overrides', {}) == {}
        and candidate.get('extensions', {}) == {} and candidate.get('host_adaptation') is None
        and not candidate.get('implementation')
        and candidate.get('initializer', 'deterministic_orthogonal') == 'deterministic_orthogonal')


def blockers(task, candidate, root=None):
    try:
        if not supports_candidate(candidate) or not is_task(task):
            raise ValueError('927 requires its exact optimizer variant candidate and unchanged original two_pole task')
        resolve_binding(root, candidate, task, None)
        return []
    except (ValueError, KeyError, TypeError, OSError) as error:
        return ['two_pole particle AMSGrad-off927: ' + str(error)]


def supporting_source_paths(task, candidate=None, root=None):
    if not is_candidate(candidate) or task.get('id') != 'two_pole':
        return ()
    if not is_task(task):
        raise ValueError('927 support paths require the unchanged original task')
    catalog = ()
    if root is not None:
        base = Path(root).resolve()
        catalog = tuple(sorted({p.relative_to(base).as_posix()
            for directory in ('configs/forge/tasks', 'configs/forge/task-variants')
            for p in (base / directory).rglob('*.json') if p.is_file()}))
    return (*catalog, *SOURCE_PINS, 'experiments/forge/atlas927_two_pole_owner.py',
        'configs/forge/defaults.json', 'configs/forge/ideas/atlas.json',
        'configs/forge/legacy-ideas-v1.json', 'configs/forge/ideas/ka2.json',
        'configs/forge/views/' + VIEW_ID + '.json', 'configs/forge/ideas/' + CANDIDATE_ID + '.json',
        'configs/forge/studies/' + STUDY_ID + '.json', 'tests/test_atlas889_two_pole_observer.py',
        'tests/test_atlas927_particle_amsgrad.py', 'tests/test_atlas921_completed_lr_clock.py',
        'experiments/forge/atlas927_contract.py')


def direct_control_receipt(task, result):
    if not is_task(task):
        raise ValueError('927 retains only original unsampled direct two_pole')
    applied = result['applied']
    return dict(schema='forge_atlas927_direct_optimizer_control_v1', candidate_id=CANDIDATE_ID,
        cohort=VIEW_ID, task_id='two_pole', initial_owner=deepcopy(applied['initialization']),
        recipe=deepcopy(applied['recipe']), base_recipe_sha256=digest(applied['recipe']),
        optimizer_variant=deepcopy(applied['optimizer_variant']),
        source_contract_sha256=applied['source_contract_sha256'],
        lifecycle=deepcopy(applied['lifecycle']), lifecycle_calls=deepcopy(applied['lifecycle_calls']),
        optimizer_updates=deepcopy(applied['optimizer_updates']), actual_sampled_prior=None,
        evaluation_sampler_calls=applied['evaluation_sampler_calls'],
        a2_inapplicable=applied['a2_inapplicable'], prior_lr_mult_consumed=applied['prior_lr_mult_consumed'])


def validate_evidence(task, evidence):
    try:
        if not is_task(task):
            raise ValueError('AMSGrad-off927 numerical evidence has a different original task')
        receipt = evidence['existing_mog717']
        if (receipt['schema'] != 'forge_atlas927_direct_optimizer_control_v1'
                or receipt['candidate_id'] != CANDIDATE_ID or receipt['cohort'] != VIEW_ID
                or receipt['task_id'] != 'two_pole' or receipt['actual_sampled_prior'] is not None
                or receipt['evaluation_sampler_calls'] != 0 or receipt['a2_inapplicable'] is not True
                or receipt['prior_lr_mult_consumed'] is not False
                or receipt['base_recipe_sha256'] != digest(receipt['recipe'])
                or receipt['lifecycle']['complete'] is not True
                or any(count != 80 for count in receipt['lifecycle_calls'].values())
                or [row['step'] for row in evidence['observations']] != CLOCKS
                or [row['step'] for row in evidence['measurement_purity']] != CLOCKS
                or any(row['pure'] is not True for row in evidence['measurement_purity'])):
            raise ValueError('original direct owner/lifecycle/purity differs')
        validate_effective_recipe(receipt['recipe'])
        validate_optimizer_variant_receipt(receipt['optimizer_variant'])
        initial = receipt['initial_owner']
        if (initial['completed_steps'] != 0 or initial['restored'] is not False
                or initial['table']['all_zero'] is not True or initial['table']['shape'] != [12, 1]
                or initial['base_recipe_sha256'] != BASE_RECIPE_SHA256
                or initial['optimizer_variant'] != receipt['optimizer_variant']
                or initial['critic']['stored_weights_exact'] is not True
                or initial['optimizer_state_entries'] != dict(prior=0, discriminator=0)):
            raise ValueError('original zero12x1/stored critic initialization is absent')
        trace = evidence['two_pole_passive889']
        if (trace['schema'] != 'forge_atlas889_two_pole_passive_trace_v1'
                or trace.get('candidate_id') != CANDIDATE_ID or trace.get('task_id') != 'two_pole'
                or trace.get('status') not in {'COMPLETE_DECLARED_SCOPE', 'UNKNOWN'}):
            raise ValueError('passive trace has missing or mismatched diagnostic identity')
        # Diagnostic incompleteness is explicit; the original numerical grader
        # consumes its original observations/gates without substituting a trace.
        return None
    except (KeyError, TypeError, ValueError, IndexError) as error:
        return dict(status='INVALID', reason='malformed AMSGrad-off927 owner evidence: ' + str(error))
