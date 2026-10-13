"""Independently audit the completed energy-distance publication from saved evidence.

Run only after collection finishes. --write writes a compact receipt; no scorer,
model, optimizer, training, sampling or RNG draw is permitted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from dataclasses import MISSING, fields
import importlib.util
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.planning import candidate_revision_for
from experiments.forge.state import state_digest
from particlegan.recipes import Recipe

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ARCHIVE = Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-energy-distance-repair-20261012')
ORIGINAL = Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-tier1-search-20261012')
CAMPAIGN = 'r1r2-energy-distance-repair-v1'
PYTHON = '/home/martyn/dev/ParticleGAN/.venv/bin/python'
PREVIOUS = ROOT / 'reports/forge/r1r2-weak-penalty-repair'
HELPER_PATH = ROOT / 'reports/forge/r1r2-objective-memory-repair/audit_collection.py'
HELPER_SHA = '596ba6cbefbfcea8dc059450ba0c157c7cc72980e9a8f5e384324d619f3bf384'
ROLES = {'non_saturating', 'energy_u'}
CLOCK = 'clockfree_audit_measurement_v1'
TASKS = {'gaussian1d_smoke', 'two_pole', 'unused_token_hold', 'ae_gan_hold',
         'ring16_acquisition', 'five_word_joint_smoke', CLOCK}
BUDGETS = dict(gaussian1d_smoke=120, two_pole=300, unused_token_hold=300,
               ae_gan_hold=300, ring16_acquisition=300, five_word_joint_smoke=900,
               clockfree_audit_measurement_v1=300)
PRIOR_ARCHIVES = (ORIGINAL, *(ARCHIVE.parent / f'{name}-20261012' for name in (
    'r1r2-five-track-repair', 'r1r2-role-repair', 'r1r2-direction-floor-repair',
    'r1r2-spectral-repair', 'r1r2-objective-memory-repair', 'r1r2-score-law-repair',
    'r1r2-weak-penalty-repair')))


def require(condition, message):
    if not condition:
        raise ValueError(message)


@contextmanager
def forbid_live_execution(helper):
    """Also forbid recipe/optimizer constructors and Python/NumPy draws."""
    def forbidden(*_args, **_kwargs):
        raise RuntimeError('Saved-evidence audit forbids constructors and random draws')
    methods = [(Recipe, '__init__'), (torch.optim.Optimizer, '__init__')]
    methods += [(random, name) for name in ('random', 'randint', 'randrange', 'choice', 'choices', 'sample', 'shuffle', 'uniform', 'gauss')]
    methods += [(np.random, name) for name in ('random', 'rand', 'randn', 'randint', 'choice', 'shuffle', 'permutation', 'normal', 'uniform', 'default_rng')]
    with helper.forbid_live_execution(), ExitStack() as stack:
        for owner, name in methods:
            stack.enter_context(patch.object(owner, name, forbidden))
        yield


def load_helper():
    require(file_hash(HELPER_PATH) == HELPER_SHA, 'Frozen collection helper changed')
    spec = importlib.util.spec_from_file_location('frozen_objective_collection', HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def actual_recipe(saved, task, helper):
    return complete_recipe(helper.role_groups(saved, task)[1])


def complete_recipe(compact):
    """Read declared dataclass defaults without invoking Recipe validators.

    Recipe.__post_init__ constructs auxiliary modules even when inactive.
    Saved provenance and public preflight own validation; this audit only
    restores the complete documented field table for exact comparisons.
    """
    declared = fields(Recipe)
    require(set(compact) <= {field.name for field in declared}, 'Unknown saved recipe fields')
    require(all(field.default is not MISSING and field.default_factory is MISSING for field in declared),
            'Recipe acquired a required/dynamic default needing separate forensic handling')
    return {field.name: deepcopy(compact.get(field.name, field.default)) for field in declared}


def validate_energy_state(value, *, weight=1., steps=None):
    """Validate signed aggregate accounting without inferring per-step forces."""
    expected = {'schema_version', 'consumer', 'weight', 'calls', 'active_calls',
                'loss_sum', 'minimum_loss', 'maximum_loss'}
    require(isinstance(value, dict) and set(value) == expected, 'Energy state field contract changed')
    require(type(value['schema_version']) is int and value['schema_version'] == 1 and
            value['consumer'] == 'output_energy_u_v1' and
            type(value['weight']) in (int, float) and value['weight'] == weight,
            'Energy consumer/version/weight changed')
    count = value['calls']
    require(type(count) is int and type(value['active_calls']) is int and count >= 0 and
            value['active_calls'] == count and (steps is None or count == steps),
            'Energy successful G-call coverage changed')
    numbers = [value[key] for key in ('loss_sum', 'minimum_loss', 'maximum_loss')]
    require(all(type(number) in (int, float) and math.isfinite(number) for number in numbers),
            'Energy aggregate contains nonfinite/invalid signed values')
    total, lower, upper = numbers
    if count == 0:
        require(total == lower == upper == 0, 'Uncalled energy state has nonzero aggregates')
    else:
        tolerance = 1e-12 * max(1., abs(total), count * abs(lower), count * abs(upper))
        require(lower <= upper and count * lower - tolerance <= total <= count * upper + tolerance,
                'Signed energy extrema/sum are inconsistent')
    return dict(value)


def energy_binding_receipt(value, *, word=False):
    expected = dict(version=1, consumer='output_energy_u_v1',
        native_loop='benchmarks.toy_audit.api_images.WordFixture.step' if word else
                    'experiments.forge.behavior_adapters.run_behavior',
        source='experiments/forge/energy_adapters.py',
        source_sha256=file_hash(ROOT / 'experiments/forge/energy_adapters.py'),
        objective_source='particlegan/energy_distance.py',
        objective_source_sha256=file_hash(ROOT / 'particlegan/energy_distance.py'))
    require(value == expected, 'Actual external energy/native-loop source binding changed')
    return dict(value)


def fresh_control_identity(root, registration, old_registration):
    old = old_registration['candidates']['non_saturating']
    current = registration['candidates']['non_saturating']
    card = f"configs/forge/ideas/{old['candidate']}.json"
    require(current['candidate'] == old['candidate'] and
            file_hash(root / card) == old_registration['declarations'][card] and
            current['candidate_revision'] != old['candidate_revision'],
            'Fresh control must retain the original NS card bytes under its new source-bound revision')


def clock_receipt(proof):
    """Own warmed starts, cross-role immutable constructor/RNG identities only."""
    initial = proof['initial']
    branches = {'reference', 'step_label', 'horizon', 'evaluation_cadence', 'restart'}
    require(set(proof['trajectories']) == set(proof['branch_initial']) == branches and
            all(len(frames) == 4 for frames in proof['trajectories'].values()),
            'Original five own-clock branches/four updates required')
    require(initial['initializer'] == 'deterministic_orthogonal' and
            set(initial['initialization']) == {'generator', 'discriminator', 'prior'} and
            all(isinstance(value.get('initial_state_sha256'), str) and
                len(value['initial_state_sha256']) == 64 for value in initial['initialization'].values()),
            'Clock original constructor initialization receipts missing')
    for branch in proof['branch_initial'].values():
        for field in ('models', 'optimizers', 'streams'):
            require(state_digest(branch['trainer'][field]) == state_digest(initial['trainer'][field]),
                    'Each clock branch must start from its own warmed tensors, optimizer and streams')
    expected = {('data', 'clock_probe', 'training'), ('data', 'target', 'training'),
        ('eval', 'clock_probe', 'samples'), ('eval', 'sampler', 'samples'), ('eval', 'target', 'reference'),
        ('init', 'discriminator', 'construction'), ('init', 'generator', 'construction'),
        ('init', 'prior', 'locations'), ('noise', 'critic', 'input'), ('noise', 'generator', 'output'),
        ('noise', 'models', 'stochastic_layers'), ('noise', 'penalty', 'training'),
        ('noise', 'prior', 'gaussian'), ('prior', 'latent', 'indices')}
    trainer_keys = {'latent_generator', 'penalty_generator', 'eval_generator', 'noise_generator',
                    'input_noise_generator', 'prior_noise_generator', 'model_generator'}
    keys = set(initial['streams']['states'])
    require(len(keys) == 14 and {tuple(json.loads(key)[:3]) for key in keys} == expected and
            set(initial['streams']['manifest']['bindings']) == keys and
            initial['streams']['manifest']['seed'] == 0,
            'Original fourteen seed-zero clock stream bindings required')
    snapshots = [initial, *proof['branch_initial'].values(),
                 *(saved for frames in proof['trajectories'].values() for saved in frames)]
    for saved in snapshots:
        require(set(saved['streams']['states']) == keys and set(saved['trainer']['streams']) == trainer_keys and
                all(isinstance(value, torch.Tensor) and value.dtype == torch.uint8
                    for value in saved['streams']['states'].values()) and
                all(value is None or isinstance(value, torch.Tensor) and value.dtype == torch.uint8
                    for value in saved['trainer']['streams'].values()),
                'All consumed own-clock streams must be checkpointed in every saved state')
    return dict(constructor_initialization_sha256=state_digest(initial['initialization']),
        own_warmed_tensors_sha256=state_digest(initial['trainer']['models']),
        pre_warmup_tensor_copy_available=False,
        initial_streams_sha256=state_digest({json.dumps(json.loads(key)[:3]):
            dict(state=value, seed=initial['streams']['manifest']['bindings'][key]['seed'])
            for key, value in initial['streams']['states'].items()}),
        initial_trainer_streams_sha256=state_digest(initial['trainer']['streams']),
        saved_states_checked=len(snapshots), all_named_streams_checkpointed=True)


def matched_clock_receipts(receipts):
    require(set(receipts) == ROLES and all(len({receipt[field] for receipt in receipts.values()}) == 1
            for field in ('constructor_initialization_sha256', 'initial_streams_sha256', 'initial_trainer_streams_sha256')),
            'Fresh clock constructor receipts or semantic consumed-stream states differ')
    return receipts


def initial_metadata_signature(value):
    """Preserve shapes/dtypes/backend; worker CUDA ordinals are caller labels."""
    if isinstance(value, dict):
        return {key: ('cuda' if key == 'device' and isinstance(child, str) and
                       (child == 'cuda' or child.startswith('cuda:') and child[5:].isdigit())
                       else initial_metadata_signature(child)) for key, child in value.items()}
    if isinstance(value, list):
        return [initial_metadata_signature(child) for child in value]
    return value


def matched_ordinary_comparisons(parent, actual):
    require(all(stable_hash(parent[field]) == stable_hash(actual[field])
                for field in ('initial_tensors_sha256', 'actual_target_batches')) and
            stable_hash(initial_metadata_signature(parent['initial_tensor_metadata'])) ==
            stable_hash(initial_metadata_signature(actual['initial_tensor_metadata'])),
            'Actual initialization, shape/dtype/backend or ordered target batches differ from fresh control')


def energy_receipt(saved, row, task, role, helper, *, steps=None):
    recipe = actual_recipe(saved, task, helper)
    weight = 1. if role == 'energy_u' else 0.
    require(recipe['energy_distance_weight'] == weight, 'Actual raw energy recipe weight changed')
    if 'trainer' in saved:
        require('component_energy' not in saved and 'component_energy' not in saved['trainer'],
                'Stateless public energy acquired a fabricated component counter')
        if steps is not None:
            require(saved['trainer']['completed_steps'] == steps, 'Native full G update allowance changed')
        return dict(kind='stateless_public_G_phase_same_batch_source_recipe_and_checkpoint',
                    weight=weight, completed_steps=saved['trainer']['completed_steps'],
                    scalar_counter_available=False, signed_U_unclamped=True, extra_draws=0)
    owner = saved['fixture'] if 'fixture' in saved else saved
    evidence, applied = row['evidence'], saved.get('applied', {})
    if weight == 0:
        require(all('component_energy' not in part and 'energy_binding' not in part
                    for part in (owner, applied, evidence)) and
                all(key not in evidence.get('guards', {}) for key in
                    ('component_energy_requested', 'component_energy_active_calls')),
                'Weight-zero original component path acquired energy state/binding/guards')
        return dict(kind='original_component_path_energy_omitted', weight=0.)
    require('component_energy' in owner, 'Active component energy checkpoint missing')
    value = validate_energy_state(owner['component_energy'], steps=steps)
    if 'fixture' in saved:
        require(value == evidence['component_energy'] and
                saved['fixture']['api_state']['completed_steps'] == value['active_calls'],
                'Word signed energy evidence/checkpoint/G counts differ')
        binding = energy_binding_receipt(evidence.get('energy_binding'), word=True)
    else:
        require(value == applied['component_energy'] == evidence['component_energy'],
                'Behavior signed energy consumer/evidence/G counts differ')
        binding = energy_binding_receipt(applied.get('energy_binding'))
    require(evidence['guards']['component_energy_requested'] is True and
            evidence['guards']['component_energy_active_calls'] == value['active_calls'] and
            evidence['guards']['hooks_exercised'] is True,
            'Actual energy hook coverage changed')
    return dict(kind='actual_separate_output_energy_u_consumer', **value,
                energy_binding=binding, signed_values_preserved=True, targets_detached=True, extra_draws=0)


def storage_receipt(archive):
    locations = (archive, *PRIOR_ARCHIVES)
    require(len(set(locations)) == 9 and all(path.is_dir() for path in locations),
            'Nine historical/current archive identities required')
    sizes = {str(path): int(subprocess.check_output(['du', '-sb', str(path)], text=True).split()[0])
             for path in locations}
    free = shutil.disk_usage(archive).free
    require(sum(sizes.values()) < 8 * 1024**3 and free >= 5 * 1024**3,
            'Combined archive cap or filesystem headroom exhausted')
    return dict(archives_bytes=sizes, combined_bytes=sum(sizes.values()), free_bytes=free,
                combined_cap_bytes=8 * 1024**3, minimum_free_bytes=5 * 1024**3)


def committed_sources(root, execution, files):
    """Current SHA256 plus Git blob equality binds every source to execution."""
    tree = subprocess.check_output(['git', 'ls-tree', '-r', '-z', execution], cwd=root)
    blobs = {}
    for item in tree.split(b'\0'):
        if item:
            metadata, path = item.split(b'\t', 1)
            kind, oid = metadata.split()[1:]
            if kind == b'blob':
                blobs[path.decode()] = oid.decode()
    paths = sorted(files)
    require(all('\n' not in path for path in paths), 'Unsafe scientific source path')
    actual = subprocess.check_output(['git', 'hash-object', '--stdin-paths'], cwd=root,
        input=''.join(json.dumps(path) + '\n' for path in paths).encode()).decode().splitlines()
    require(len(actual) == len(paths), 'Incomplete committed-source blob check')
    for path, blob in zip(paths, actual, strict=True):
        require(file_hash(root / path) == files[path] and blobs.get(path) == blob,
                f'Executed source file is absent/changed under supplied commit: {path}')
    return len(paths)


def optimizer_roles(saved, task, role, helper, *, clock=False):
    """Native saved histories; inspect the native positive gamma .1 histories."""
    require(role in ROLES, 'Only the two fresh native AMS recipes are admitted')
    groups, compact = helper.role_groups(saved, task)
    recipe = complete_recipe(compact)
    expected = {"D": .5, "G": .5, "E": .5, "P": 0.} if role != "baseline" else dict.fromkeys(("D", "G", "E", "P"), 0.)
    amsgrad = role != "baseline"
    require(recipe.get("optimizer_family", "adam") == "adam" and recipe.get("d_eps") is None and
            recipe.get("amsgrad", False) == amsgrad, "Saved native Adam family/epsilon/AMSGrad changed")
    require(recipe.get("kinetic_transport_weight", 0) == 0 and
            recipe.get("kinetic_transport_local_weight", 0) == 0 and recipe.get("kinetic_transport_projections", 32) == 32,
            "Disabled global/local transport or fixed projection count changed")
    observed = {}
    for player, optimizer, group in groups:
        require("dualnorm" not in optimizer and "algorithm" not in group, "Native Adam was replaced by normalized updates")
        require(group["eps"] == 1e-8 and group["betas"][0] == expected[player] and group["amsgrad"] == amsgrad,
                f"{role}/{task}/{player}: actual epsilon, beta1 or AMSGrad changed")
        require(group["betas"][1] == .99 if role != "baseline" or not clock else .9 <= group["betas"][1] <= .99,
                f"{role}/{task}/{player}: actual beta2 changed")
        if role != "baseline":
            require(group["lr"] == .0085, f"{role}/{task}/{player}: actual global step size changed")
        history_fields = {"step", "exp_avg", "exp_avg_sq"} | ({"max_exp_avg_sq"} if amsgrad else set())
        require(set(optimizer["state"]) <= {parameter for g in optimizer["param_groups"] for parameter in g["params"]},
                "Optimizer history contains an unknown parameter")
        for parameter in group["params"]:
            history = optimizer["state"].get(parameter)
            require(history and set(history) == history_fields, "Actual Adam/AMSGrad update histories changed")
            step = history["step"]
            step = float(step.item()) if isinstance(step, torch.Tensor) else step
            require(step > 0 and int(step) == step, "Actual optimizer updates missing")
            for field in history_fields - {"step"}:
                value = history[field]
                require(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()) and
                        value.shape == history["exp_avg_sq"].shape, "Invalid saved optimizer history")
                if field in {"exp_avg_sq", "max_exp_avg_sq"}:
                    require(bool((value >= 0).all()), "Invalid negative second moments")
            if amsgrad:
                require(bool((history["max_exp_avg_sq"] >= history["exp_avg_sq"]).all()), "AMSGrad maximum history is below current second moment")
        observed[player] = dict(family="adam", algorithm="adam", betas=list(group["betas"]),
                               amsgrad=amsgrad, eps=group["eps"], lr=group["lr"], history_fields=sorted(history_fields))
    if task == "ae_gan_hold":
        parameters = saved["role_parameters"]
        require(parameters["encoder"] and len(groups[0][2]["params"]) ==
                len(parameters["generator"]) + len(parameters["encoder"]), "Actual AE G/E joint group membership changed")
        observed["E"] = dict(observed["G"])
    if role != "baseline":
        require(recipe.get("beta2_end") is None and recipe["lr_schedule"] == "constant" and
                recipe["lr"] == .0085 and recipe["reg_coeff"] == .1 and
                recipe.get("prior_update", "learned") == "learned" and recipe["prior_reg"] == 0,
                "Actual role-specific gamma or constant learned-prior law changed")
        for player, field in (("G", "betas"), ("D", "d_betas"), ("P", "prior_betas")):
            require(list(recipe.get(field) or recipe["betas"]) == [expected[player], .99], "Global role moments changed")
    return observed, len(groups)


def score_and_penalty_binding(saved, task, role, helper, *, ordinary_steps=None):
    """Actual recipe/history plus immutable public loss/penalty ownership."""
    groups, compact = helper.role_groups(saved, task)
    recipe = complete_recipe(compact)
    require(role in ROLES and recipe['energy_distance_weight'] == (1. if role == 'energy_u' else 0.),
            'Unknown recipe role or raw energy coefficient')
    require(recipe['loss'] == 'non_saturating',
            f'{role}/{task}: actual score law changed')
    require(tuple(recipe['loss_labels']) == (0., 1., 1.) and recipe['reg_arm'] == 'a_r1r2',
            'Actual labels or paired R1/R2 penalty changed')
    if role != 'baseline':
        require(recipe['reg_coeff'] == .1 and recipe['reg_every'] == 1 and
                recipe['reg_coeff_end'] is None and recipe['amsgrad'] and
                recipe['kinetic_transport_weight'] == recipe['kinetic_transport_local_weight'] == 0,
                'Constant positive paired penalties or exact AMS parent controls changed')
        critic = next(optimizer for player, optimizer, _ in groups if player == 'D')
        record = critic['regularizer']['record']
        require(record['calls'] == record['observed_steps'] and not record['anchor_started'] and
                critic['regularizer']['ema'] is None and critic['regularizer']['guard'] is None,
                'Every-update raw-score R1/R2 history or native state changed')
        if ordinary_steps is not None:
            require(record['observed_steps'] == ordinary_steps, 'Full original critic allowance changed')
        for _, optimizer, group in groups:
            require(all(int(optimizer['state'][key]['step']) == record['observed_steps']
                        for key in group['params']), 'Actual player update counters differ from critic history')
    return dict(loss=recipe['loss'], loss_labels=list(recipe['loss_labels']), penalty_arm='a_r1r2',
        gamma=recipe['reg_coeff'], gamma_end=recipe['reg_coeff_end'], every=recipe['reg_every'],
        constant_gamma=recipe['reg_coeff_end'] is None or recipe['reg_coeff_end'] == recipe['reg_coeff'],
        joint_encoder_reversed_real_stream=task == 'five_word_joint_smoke',
        binding='immutable public factory/source, actual saved recipe and completed optimizer/penalty histories',
        proof_limit='No stored per-update scores, real/fake pairings or paired-force decomposition is asserted.')


def audit(root, archive, execution, helper):
    require(len(execution) == 40 and all(c in '0123456789abcdef' for c in execution), 'Expected execution SHA required')
    require(sys.executable == PYTHON and torch.get_num_threads() == 1, 'Use declared interpreter and one CPU thread')
    root, archive = root.resolve(), archive.resolve()
    out, previous = root / OUT.relative_to(ROOT), root / PREVIOUS.relative_to(ROOT)
    paths = dict(publication=out / 'publication/results.json', registration=out / 'registration.json',
                 progress=archive / f'{CAMPAIGN}-progress.json', queue=archive / 'queue/queue/state.json',
                 previous_publication=previous / 'publication/results.json', previous_audit=previous / 'collection-audit.json',
                 manifest=out / 'manifest.json', previous_registration=previous / 'registration.json')
    hashes = {name: file_hash(path) for name, path in paths.items()}
    publication, registration, progress, queue, old, old_audit, manifest, old_registration = [read_json(paths[name]) for name in paths]
    require(publication['complete'] and progress['phase'] == 'complete' and
            publication['source_commit'] == progress['source_commit'] == execution, 'Campaign is not completely collected')
    require(old['complete'] and old_audit['status'] == 'PASS' and
            old_audit['inputs_sha256']['publication'] == hashes['previous_publication'] and
            old_audit['inputs_sha256']['registration'] == hashes['previous_registration'],
            'Closed predecessor evidence changed')
    require(paths['queue'].resolve() == (ORIGINAL / 'queue/queue/state.json').resolve(), 'Shared queue identity changed')
    require(publication['registration_sha256'] == progress['registration_sha256'] == hashes['registration'], 'Registration changed')
    require(registration['manifest'] == manifest and registration['manifest_sha256'] == hashes['manifest'] and
            manifest['source_digest'] == registration['source_digest'] and
            manifest['source_digest'] != old['source_digest'], 'Explicit fresh-source manifest binding changed')
    require(set(progress['requests']) == set(registration['candidates']) == ROLES and
            publication['round_id'] == CAMPAIGN and
            publication['full_reservation_seconds'] == registration['reservation_seconds'] == 5040 and
            publication['paid_cap_seconds'] == registration['paid_cap_seconds'] == 8000, 'Finite scope/budgets changed')
    require(registration['archive'] == str(archive) and archive == ARCHIVE and
            (archive / 'queue').is_symlink() and (archive / 'queue').resolve() == ORIGINAL / 'queue',
            'Declared ninth archive/shared queue identity changed')
    declarations = registration['declarations']
    require({str((out / name).relative_to(root)) for name in ('audit_collection.py', 'density_audit.py')} <= set(declarations)
            and all(file_hash(root / path) == digest for path, digest in declarations.items()),
            'Frozen supporting declarations/auditors changed')
    old_control = old_registration['candidates']['non_saturating']
    for field in ('source_digest', 'runtime_sha256', 'protocol_sha256', 'view_sha256', 'task_contracts'):
        require(len({stable_hash(item[field]) for item in registration['candidates'].values()}) == 1,
                f'Fixed {field} differs across roles')
    for plan in registration['candidates'].values():
        require(all(plan[field] == old_control[field] for field in
                    ('runtime_sha256', 'protocol_sha256', 'view_sha256', 'task_contracts', 'execution_policy', 'full_allowance_seconds')),
                'Original task, gate, protocol, runtime or full allowance changed')
    fresh_control_identity(root, registration, old_registration)
    require(len(publication['certified_results']) == len(publication['task_cells']) ==
            len(publication['media']) == len(publication['costs']) == 14, 'Exactly 14 fresh cells/media/costs required')
    require(publication['optimizer_updates_added_by_collection'] == publication['sampling_draws_added_by_collection'] == 0 and
            not publication['selection_or_defaults_changed'], 'Collection side effects declared')
    charges = {item['attempt_id']: item for item in queue['charges']}
    require(len(charges) == len(queue['charges']) == 231, 'Aggregate charges duplicated/missing')
    old_ids = set(old['aggregate_charge_attempt_ids'])
    old_charges = {aid: charges[aid] for aid in old_ids}
    require(len(old_ids) == old['aggregate_paid_attempts'] == 217 and
            stable_hash(old_charges) == old_audit['inputs_sha256']['aggregate217_charge_index'], 'Historical217 charge identities/owners/values changed')
    prior_total = sum(item['seconds'] for item in old_charges.values())
    require(math.isclose(prior_total, old['aggregate_total_paid_seconds'], abs_tol=1e-7, rel_tol=0) and
            math.isclose(prior_total, 6586.162437112915, abs_tol=1e-7, rel_tol=0), 'Historical217 paid seconds changed')
    old_jobs = {}
    for job in queue['jobs'].values():
        result = job.get('result')
        if result and result['attempt_id'] in old_ids:
            aid = result['attempt_id']
            require(aid not in old_jobs and len(job['attempts']) == 1 and
                    job['attempts'][0]['attempt_id'] == aid and
                    job['cost_owner'] == result['cost_owner'] == old_charges[aid]['owner'] and
                    math.isclose(math.fsum(row['cost']['wall_seconds'] for row in result['task_results']),
                                 old_charges[aid]['seconds'], abs_tol=1e-7, rel_tol=0),
                    'Historical result/retry/original charge-owner contract changed')
            old_jobs[aid] = job
    require(set(old_jobs) == old_ids, 'Historical paid result disappeared')
    current_requests = set(progress['requests'].values())
    new_charges = {aid: item for aid, item in charges.items() if item['owner']['request'] in current_requests}
    require(set(new_charges) == set(charges) - old_ids and len(new_charges) == 14, 'New charge ownership/count changed')
    new_total, total = sum(item['seconds'] for item in new_charges.values()), sum(item['seconds'] for item in charges.values())
    require(set(publication['aggregate_charge_attempt_ids']) == set(charges) and publication['aggregate_paid_attempts'] == 231 and
            math.isclose(publication['total_paid_seconds'], new_total, abs_tol=1e-7, rel_tol=0) and
            math.isclose(publication['aggregate_total_paid_seconds'], total, abs_tol=1e-7, rel_tol=0) and total <= 8000,
            'Aggregate costs or paid ceiling changed')
    cells = {(row['role'], row['task_id']): row for row in publication['task_cells']}
    media = {(row['role'], row['task_id']): row for row in publication['media']}
    require(len(cells) == len(media) == 14 and set(cells) == set(media) == {(r, t) for r in ROLES for t in TASKS}, 'Duplicate/missing task cells')
    require(len({(row['role'], row['task_id']) for row in publication['certified_results']}) == 14, 'Duplicate certified rows')
    for row in cells.values():
        require(row['required'] == (row['task_id'] != CLOCK), 'Ordinary/clock denominator changed')
    standings = {row['role']: row for row in publication['recipes']}
    require(set(standings) == ROLES, 'Recipe summary coverage changed')
    for role, plan in registration['candidates'].items():
        own = [cells[(role, task)] for task in TASKS]
        require(standings[role] == dict(role=role, candidate_id=plan['candidate'],
            required_tier1=dict(Counter(row['status'] for row in own if row['required'])),
            clock_status=cells[(role, CLOCK)]['status'], all_seven_pass=all(row['status'] == 'PASS' for row in own)),
            'Standings differ from original stored gates')
    counts, source_files, moments, score_laws, recipes, energies, fresh_comparisons = Counter(), {}, {}, {}, {}, {}, {}
    clocks, clock_inputs = {}, {}
    for role in ('non_saturating', 'energy_u'):
        rid = progress['requests'][role]
        request, plan = queue['submissions'][rid]['request'], registration['candidates'][role]
        require(helper.projection(request) == {k: v for k, v in plan.items() if k != 'card'}, 'Admitted projection changed')
        require(request['candidate_revision'] == candidate_revision_for(request['source']['digest'], request['candidate']),
                'Actual formulation/source-bound candidate revision changed')
        jobs = [job for job in request['jobs'] if job['task_id'] in TASKS]
        require(len(jobs) == 7 and {job['task_id'] for job in jobs} == TASKS and
                all(job.get('task_ids', [job['task_id']]) == [job['task_id']] and
                    job['budget_seconds'] == BUDGETS[job['task_id']] for job in jobs),
                'Seven original full task allowances changed')
        require(plan['source_digest'] == publication['source_digest'] == registration['source_digest'] != old['source_digest'] and
                plan['runtime_sha256'] == old_audit['runtime_sha256'], 'Frozen scientific source/runtime changed')
        for path, digest in request['source']['files'].items():
            require(path not in source_files or source_files[path] == digest, 'Conflicting source file identities')
            source_files[path] = digest
        require(request['source']['origin_commit'] == execution, 'Fresh control/treatment does not bind supplied execution SHA')
        require(request['rng']['seed'] == request['protocol']['seed'] == 0 and
                all(request['source']['files'][path] == digest == file_hash(root / path)
                    for task in TASKS for path, digest in request['tasks'][task]['evaluation']['sources'].items()),
                'Public initialization seed or byte-pinned original evaluator sources changed')
        for task, key in plan['task_keys'].items():
            job = queue['jobs'][key]
            require(job['status'] == 'terminal' and len(job['attempts']) == 1 and job['result'], 'Missing result or scientific retry')
            aid = job['result']['attempt_id']
            saved_request, result, local, compact = helper.read_attempt(root, aid)
            require(result == job['result'] and helper.projection(saved_request) == helper.projection(request), 'Certified/admitted identity differs')
            raw_row = next(row for row in result['task_results'] if row['task_id'] == task)
            row = next(row for row in publication['certified_results'] if row['role'] == role and row['task_id'] == task)
            projected = next(row for row in compact['task_results'] if row['task_id'] == task)
            require(row['candidate_id'] == plan['candidate'] and row['candidate_revision'] == plan['candidate_revision'] and
                    row['source_digest'] == plan['source_digest'] and row['attempt_id'] == aid and not row['qualification_input'], 'Published attribution changed')
            require(all(row.get(k) == value for k, value in projected.items()) and
                    row['independent_grading'] == helper.stored_grade(saved_request, result), 'Original independent grades/metrics changed')
            require(raw_row['raw_status'] == 'completed' and row['gate_status'] == cells[(role, task)]['status'], 'Completion/gate cell changed')
            require(set(publication['exact_certificate_hashes'][aid]) == {'request', 'result', 'evidence'}, 'Original certificate hash missing')
            for name, digest in publication['exact_certificate_hashes'][aid].items():
                require(file_hash(root / f'reports/forge/attempts/{aid}/{name}.json') == digest, 'Original certificate bytes changed')
            charge, cost = charges[aid], publication['costs'][aid]
            require(charge['owner'] == result['cost_owner'] and math.isclose(charge['seconds'],
                    sum(row['cost']['wall_seconds'] for row in result['task_results']), abs_tol=1e-7, rel_tol=0), 'Actual original charge changed')
            reused = False
            require(cost['reused'] == reused and cost['original_charge_owner'] == charge['owner'] and
                    cost['original_paid_seconds'] == charge['seconds'] and cost['seconds'] == (0 if reused else charge['seconds']) and
                    cost['full_allowance_seconds'] == job['definition']['budget_seconds'], 'Exact reuse/full allowance accounting changed')
            require(aid not in old_ids and charge['owner']['request'] == rid, 'Fresh cell borrowed an old-source grade/charge')
            proof = None
            if task == CLOCK:
                evidence = raw_row['evidence']
                directory = Path(evidence['artifact_root'])
                helper.verify_artifacts(directory, evidence['artifact_manifest'])
                proof = torch.load(directory / 'comparisons.pt', map_location='cpu', weights_only=True)
                clocks[role] = clock_receipt(proof)
                clock_inputs[str(directory / 'comparisons.pt')] = file_hash(directory / 'comparisons.pt')
                require(proof['execution'] == request['tasks'][task]['execution'], 'Clock execution law changed')
                branches = {'reference', 'step_label', 'horizon', 'evaluation_cadence', 'restart'}
                require(set(proof['trajectories']) == set(proof['branch_initial']) == branches and
                        all(len(states) == proof['execution']['probe_steps'] for states in proof['trajectories'].values()), 'Clock saved branch coverage changed')
                for initial in proof['branch_initial'].values():
                    for field in ('models', 'optimizers', 'streams'):
                        require(state_digest(initial['trainer'][field]) == state_digest(proof['initial']['trainer'][field]), 'Clock own trained branch initial state changed')
                snapshots = [proof['initial'], *proof['branch_initial'].values(),
                             *(state for trajectory in proof['trajectories'].values() for state in trajectory)]
                for snapshot in snapshots:
                    counts['clock_stream_states'] += helper.named_streams(snapshot)
                    counts['clock_saved_snapshots'] += 1
                    optimizer_roles(snapshot, task, role, helper, clock=True)
                    score_and_penalty_binding(snapshot, task, role, helper)
                    energy_receipt(snapshot, raw_row, task, role, helper)
                saved = proof['initial']
                moments.setdefault(role, {})[task] = optimizer_roles(saved, task, role, helper, clock=True)[0]
                counts['clock_proofs'] += 1
            else:
                saved, checkpoint = helper.checkpoint(raw_row)
                require(checkpoint == row['checkpoint'], 'Published checkpoint identity changed')
                comparison = raw_row['evidence']['comparison_protocol']
                require(row['comparison_protocol'] == comparison, 'Published comparison receipt changed')
                if role == 'non_saturating':
                    fresh_comparisons[task] = comparison
                else:
                    parent = fresh_comparisons[task]
                    matched_ordinary_comparisons(parent, comparison)
                counts['ordinary_stream_states'] += helper.named_streams(saved, comparison)
                moments.setdefault(role, {})[task], n = optimizer_roles(saved, task, role, helper)
                counts['optimizer_groups'] += n
                counts['checkpoints'] += 1
            recipes[(role, task)] = actual_recipe(saved, task, helper)
            score_laws.setdefault(role, {})[task] = score_and_penalty_binding(saved, task, role, helper,
                ordinary_steps=None if task == CLOCK else request['tasks'][task]['execution']['steps'])
            energies.setdefault(role, {})[task] = energy_receipt(saved, raw_row, task, role, helper,
                steps=None if task == CLOCK else request['tasks'][task]['execution']['steps'])
            helper.saved_media(root, media[(role, task)], raw_row, request['tasks'][task], local, proof)
            counts['gifs'] += 1
            counts['grades'] += 1
    deltas = {}
    for task in TASKS:
        parent, actual = recipes[('non_saturating', task)], recipes[('energy_u', task)]
        changed = [name for name in parent if stable_hash(parent[name]) != stable_hash(actual[name])]
        require(changed == ['energy_distance_weight'], f'{task}: complete task-bound recipe difference is {changed}')
        deltas[task] = dict(changed_fields=changed, before=parent['energy_distance_weight'], after=actual['energy_distance_weight'])
    require(set(fresh_comparisons) == TASKS - {CLOCK}, 'All six fresh ordinary comparison receipts required')
    matched_clock_receipts(clocks)
    comparisons = publication['comparison_audit']
    require(len(comparisons) == 7 and {row['task_id'] for row in comparisons} == TASKS and
            all(row['status'] == 'PASS' and set(row['roles']) == ROLES for row in comparisons),
            'All seven fresh paired protocol comparison receipts required')
    clock_comparison = next(row for row in comparisons if row['task_id'] == CLOCK)
    require(clock_comparison == dict(task_id=CLOCK, status='PASS', roles=sorted(ROLES),
            saved_inputs=clock_inputs, own_clock_saved_proofs=clocks, optimizer_updates_added=0, sampling_draws_added=0),
            'Published own-clock initializer/stream proof differs from saved evidence')
    require({'particlegan/energy_distance.py', 'experiments/forge/energy_adapters.py'} <= set(source_files),
            'New raw primitive or external native-loop binding omitted from source ownership')
    committed_sources(root, execution, source_files)
    committed_sources(root, execution, declarations)
    # Actual immutable source paths, not a fabricated historical force trace.
    loss_source = (root / 'particlegan/gan_loss.py').read_text()
    require('def joint_g_loss' in loss_source and
            'self.g_loss(fake_logits) + F.softplus(real_logits).mean()' in loss_source and
            'F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()' in loss_source,
            'Native NS/joint reversed-label public score law changed')
    require('self.loss.joint_g_loss(' in (root / 'benchmarks/toy_audit/api_images.py').read_text(), 'Actual joint host removed encoder stream')
    trainer_source = (root / 'particlegan/training.py').read_text()
    require('recipe.energy_distance_loss(fake_g, real_g)' in trainer_source and
            'loss_g = loss_g + energy' in trainer_source,
            'Stateless native energy must use the existing actual G-phase real/fake batch and objective')
    storage = storage_receipt(archive)
    require(all(file_hash(path) == hashes[name] for name, path in paths.items()), 'Inputs changed during audit')
    return dict(schema_version=1, status='PASS', qualification_input=False,
        scope='independent_saved_fresh_source_energy_u_collection_audit', source_commit=execution,
        source_digest=publication['source_digest'], runtime_sha256=old_audit['runtime_sha256'],
        inputs_sha256={**hashes, 'historical217_charge_index': stable_hash(old_charges),
                      'aggregate231_charge_index': stable_hash(charges), 'audit_source': file_hash(Path(__file__))},
        imported_collection_helper_sha256=HELPER_SHA,
        coverage=dict(roles=2, cases=14, exact_cached_control_cases=0, new_paid_attempts=14,
                      aggregate_unique_paid_attempts=231, source_files=len(source_files), **dict(counts)),
        costs=dict(prior_paid_seconds=prior_total, new_paid_seconds=new_total, aggregate_paid_seconds=total,
                   cached_new_paid_seconds=0, shared_paid_ceiling_seconds=8000),
        storage=storage,
        checks={name: 'PASS' for name in ('original_certificates_grades_metrics_and_training_GIFs',
            'zero_cache_fourteen_fresh_source_controls_and_treatments', 'complete_task_bound_recipe_delta_only_raw_energy_weight_zero_to_one_on_all_seven_hosts',
            'native_NS_and_joint_encoder_reversed_stream_immutable_public_source',
            'both_fresh_recipes_positive_R1_R2_point1_every_update_actual_penalty_histories',
            'actual_AMSGrad_role_betas_maxima_and_full_optimizer_histories',
            'actual_initial_tensor_and_ordered_target_batch_receipts_match_parent',
            'all_named_stream_receipts_match_checkpoints', 'own_clock_branches',
            'separate_finite_signed_energy_state_and_weight_zero_omission',
            'all_fresh_source_files_committed_under_execution_SHA',
            'original_fixed_task_evaluator_pins_and_frozen_supporting_inputs',
            'nine_archives_original_combined_cap_and_free_space_floor',
            'all217_historical_charges_retained_fourteen_new231_aggregate', 'no_scientific_retries')},
        actual_optimizer_groups=moments, actual_score_law_bindings=score_laws, actual_energy_bindings=energies,
        own_clock_saved_proofs=clocks, task_bound_recipe_deltas=deltas,
        optimizer_updates_added=0, sampling_draws_added=0, live_evaluators_or_rescoring=False,
        model_construction_training_sampling_and_random_draws_forbidden=True,
        limits='Fourteen fresh stored independent grades only; old-source grades supply no fresh cell. Loss/penalty binding uses actual complete saved recipes, '
            'optimizer/penalty counters and immutable public source, not saved per-update paired scores or forces. '
            'Initial tensors/data are verified six-ordinary-host receipt identities, not reconstructed; original device labels are retained and CUDA worker ordinals normalized for metadata comparisons. Clock keeps its own warmed cohort. '
            'Native scalar/clock energy is stateless and bound by source, recipe, actual batch receipts and update counts; no missing counter is invented. '
            'EMA first moments do not recover instantaneous G/D gradients or historical sampled-row ownership.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-commit', required=True)
    parser.add_argument('--repository', type=Path, default=ROOT)
    parser.add_argument('--artifacts', type=Path, default=ARCHIVE)
    parser.add_argument('--output', type=Path, default=OUT / 'collection-audit.json')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    helper = load_helper()
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    with forbid_live_execution(helper), torch.no_grad():
        receipt = audit(args.repository, args.artifacts, args.execution_commit, helper)
    after = random.getstate(), np.random.get_state(), torch.get_rng_state()
    require(before[0] == after[0] and before[1][0] == after[1][0] and np.array_equal(before[1][1], after[1][1]) and
            before[1][2:] == after[1][2:] and torch.equal(before[2], after[2]), 'Global RNG changed during audit')
    receipt['checks']['python_numpy_torch_cpu_rng_unchanged'] = 'PASS'
    if args.write:
        atomic_json(args.output.resolve(), receipt)
        print(json.dumps(dict(status=receipt['status'], coverage=receipt['coverage'], costs=receipt['costs']), sort_keys=True))
    else:
        print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
