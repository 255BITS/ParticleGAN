"""Independently audit the completed score-law publication from saved evidence.

Run only after collection finishes. --write writes a compact receipt; no scorer,
model, optimizer, training, sampling or RNG draw is permitted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import MISSING, fields
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.state import state_digest
from particlegan.recipes import Recipe

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ARCHIVE = Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-score-law-repair-20261012')
ORIGINAL = Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-tier1-search-20261012')
CAMPAIGN = 'r1r2-score-law-repair-v1'
PYTHON = '/home/martyn/dev/ParticleGAN/.venv/bin/python'
PREVIOUS = ROOT / 'reports/forge/r1r2-objective-memory-repair'
HELPER_PATH = PREVIOUS / 'audit_collection.py'
HELPER_SHA = '596ba6cbefbfcea8dc059450ba0c157c7cc72980e9a8f5e384324d619f3bf384'
ROLES = {'baseline', 'ams_m110', 'non_saturating'}
CONTROLS = {'baseline', 'ams_m110'}
CLOCK = 'clockfree_audit_measurement_v1'
TASKS = {'gaussian1d_smoke', 'two_pole', 'unused_token_hold', 'ae_gan_hold',
         'ring16_acquisition', 'five_word_joint_smoke', CLOCK}


def require(condition, message):
    if not condition:
        raise ValueError(message)


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


def score_and_penalty_binding(saved, task, role, helper, *, ordinary_steps=None):
    """Actual recipe/history plus immutable public loss/penalty ownership."""
    groups, compact = helper.role_groups(saved, task)
    recipe = complete_recipe(compact)
    require(recipe['loss'] == ('non_saturating' if role == 'non_saturating' else 'relativistic'),
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
                 previous_publication=previous / 'publication/results.json', previous_audit=previous / 'collection-audit.json')
    hashes = {name: file_hash(path) for name, path in paths.items()}
    publication, registration, progress, queue, old, old_audit = [read_json(paths[name]) for name in paths]
    require(publication['complete'] and progress['phase'] == 'complete' and
            publication['source_commit'] == progress['source_commit'] == execution, 'Campaign is not completely collected')
    require(old['complete'] and old_audit['status'] == 'PASS' and
            old_audit['inputs_sha256']['publication'] == hashes['previous_publication'], 'Closed predecessor evidence changed')
    require(paths['queue'].resolve() == (ORIGINAL / 'queue/queue/state.json').resolve(), 'Shared queue identity changed')
    require(publication['registration_sha256'] == progress['registration_sha256'] == hashes['registration'], 'Registration changed')
    require(set(progress['requests']) == set(registration['candidates']) == ROLES and
            publication['round_id'] == CAMPAIGN and
            publication['full_reservation_seconds'] == registration['reservation_seconds'] == 7560 and
            publication['paid_cap_seconds'] == registration['paid_cap_seconds'] == 8000, 'Finite scope/budgets changed')
    for field in ('source_digest', 'runtime_sha256', 'protocol_sha256', 'view_sha256', 'task_contracts'):
        require(len({stable_hash(item[field]) for item in registration['candidates'].values()}) == 1,
                f'Fixed {field} differs across roles')
    require(len(publication['certified_results']) == len(publication['task_cells']) ==
            len(publication['media']) == len(publication['costs']) == 21, 'Exactly 21 cells/media/costs required')
    require(publication['optimizer_updates_added_by_collection'] == publication['sampling_draws_added_by_collection'] == 0 and
            not publication['selection_or_defaults_changed'], 'Collection side effects declared')
    original_controls = {role: {row['task_id']: row for row in old['certified_results'] if row['role'] == role}
                         for role in CONTROLS}
    require(all(set(rows) == TASKS for rows in original_controls.values()), 'Exact cached control coverage changed')
    charges = {item['attempt_id']: item for item in queue['charges']}
    require(len(charges) == len(queue['charges']) == 210, 'Aggregate charges duplicated/missing')
    old_ids = set(old['aggregate_charge_attempt_ids'])
    old_charges = {aid: charges[aid] for aid in old_ids}
    require(len(old_ids) == old['aggregate_paid_attempts'] == 203 and
            stable_hash(old_charges) == old_audit['inputs_sha256']['aggregate203_charge_index'], 'Historical203 charge identities/owners/values changed')
    prior_total = sum(item['seconds'] for item in old_charges.values())
    require(math.isclose(prior_total, old['aggregate_total_paid_seconds'], abs_tol=1e-7) and
            math.isclose(prior_total, 6222.199867211952, abs_tol=1e-7), 'Historical203 paid seconds changed')
    current_requests = set(progress['requests'].values())
    new_charges = {aid: item for aid, item in charges.items() if item['owner']['request'] in current_requests}
    require(set(new_charges) == set(charges) - old_ids and len(new_charges) == 7, 'New charge ownership/count changed')
    new_total, total = sum(item['seconds'] for item in new_charges.values()), sum(item['seconds'] for item in charges.values())
    require(set(publication['aggregate_charge_attempt_ids']) == set(charges) and publication['aggregate_paid_attempts'] == 210 and
            math.isclose(publication['total_paid_seconds'], new_total, abs_tol=1e-7) and
            math.isclose(publication['aggregate_total_paid_seconds'], total, abs_tol=1e-7) and total <= 8000,
            'Aggregate costs or paid ceiling changed')
    cells = {(row['role'], row['task_id']): row for row in publication['task_cells']}
    media = {(row['role'], row['task_id']): row for row in publication['media']}
    require(len(cells) == len(media) == 21 and set(cells) == set(media) == {(r, t) for r in ROLES for t in TASKS}, 'Duplicate/missing task cells')
    require(len({(row['role'], row['task_id']) for row in publication['certified_results']}) == 21, 'Duplicate certified rows')
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
    counts, source_files, moments, score_laws, recipes = Counter(), {}, {}, {}, {}
    for role, rid in progress['requests'].items():
        request, plan = queue['submissions'][rid]['request'], registration['candidates'][role]
        require(helper.projection(request) == {k: v for k, v in plan.items() if k != 'card'}, 'Admitted projection changed')
        require(plan['source_digest'] == publication['source_digest'] == registration['source_digest'] == old['source_digest'] and
                plan['runtime_sha256'] == old_audit['runtime_sha256'], 'Frozen scientific source/runtime changed')
        for path, digest in request['source']['files'].items():
            require(path not in source_files or source_files[path] == digest, 'Conflicting source file identities')
            source_files[path] = digest
        if role == 'non_saturating':
            require(request['source']['origin_commit'] == execution, 'New run does not bind supplied execution SHA')
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
                    sum(row['cost']['wall_seconds'] for row in result['task_results']), abs_tol=1e-7), 'Actual original charge changed')
            reused = role in CONTROLS
            require(cost['reused'] == reused and cost['original_charge_owner'] == charge['owner'] and
                    cost['original_paid_seconds'] == charge['seconds'] and cost['seconds'] == (0 if reused else charge['seconds']) and
                    cost['full_allowance_seconds'] == job['definition']['budget_seconds'], 'Exact reuse/full allowance accounting changed')
            if reused:
                old_row = original_controls[role][task]
                require({k: value for k, value in row.items() if k != 'role'} ==
                        {k: value for k, value in old_row.items() if k != 'role'}, 'Exact cached control changed')
            proof = None
            semantic_role = 'baseline' if role == 'baseline' else 'ams_m110'
            if task == CLOCK:
                evidence = raw_row['evidence']
                directory = Path(evidence['artifact_root'])
                helper.verify_artifacts(directory, evidence['artifact_manifest'])
                proof = torch.load(directory / 'comparisons.pt', map_location='cpu', weights_only=True)
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
                    helper.optimizer_roles(snapshot, task, semantic_role, clock=True)
                    score_and_penalty_binding(snapshot, task, role, helper)
                saved = proof['initial']
                moments.setdefault(role, {})[task] = helper.optimizer_roles(saved, task, semantic_role, clock=True)[0]
                counts['clock_proofs'] += 1
            else:
                saved, checkpoint = helper.checkpoint(raw_row)
                require(checkpoint == row['checkpoint'], 'Published checkpoint identity changed')
                comparison = raw_row['evidence']['comparison_protocol']
                require(row['comparison_protocol'] == comparison, 'Published comparison receipt changed')
                parent = original_controls['ams_m110'][task]['comparison_protocol']
                for field in ('initial_tensors_sha256', 'initial_tensor_metadata', 'actual_target_batches'):
                    require(stable_hash(comparison[field]) == stable_hash(parent[field]), 'Actual initial tensors/ordered target batches differ from parent')
                counts['ordinary_stream_states'] += helper.named_streams(saved, comparison)
                moments.setdefault(role, {})[task], n = helper.optimizer_roles(saved, task, semantic_role)
                counts['optimizer_groups'] += n
                counts['checkpoints'] += 1
            recipes[(role, task)] = actual_recipe(saved, task, helper)
            score_laws.setdefault(role, {})[task] = score_and_penalty_binding(saved, task, role, helper,
                ordinary_steps=None if task == CLOCK else request['tasks'][task]['execution']['steps'])
            helper.saved_media(root, media[(role, task)], raw_row, request['tasks'][task], local, proof)
            counts['gifs'] += 1
            counts['grades'] += 1
    deltas = {}
    for task in TASKS:
        parent, actual = recipes[('ams_m110', task)], recipes[('non_saturating', task)]
        changed = [name for name in parent if stable_hash(parent[name]) != stable_hash(actual[name])]
        require(changed == ['loss'], f'{task}: complete task-bound recipe difference is {changed}')
        deltas[task] = dict(changed_fields=changed, before=parent['loss'], after=actual['loss'])
    for path, digest in source_files.items():
        require(file_hash(root / path) == digest, 'Executed scientific source changed')
    # Actual immutable source paths, not a fabricated historical force trace.
    loss_source = (root / 'particlegan/gan_loss.py').read_text()
    require('def joint_g_loss' in loss_source and
            'self.g_loss(fake_logits) + F.softplus(real_logits).mean()' in loss_source and
            'F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()' in loss_source,
            'Native NS/joint reversed-label public score law changed')
    require('self.loss.joint_g_loss(' in (root / 'benchmarks/toy_audit/api_images.py').read_text(), 'Actual joint host removed encoder stream')
    require(all(file_hash(path) == hashes[name] for name, path in paths.items()), 'Inputs changed during audit')
    return dict(schema_version=1, status='PASS', qualification_input=False,
        scope='independent_saved_score_law_collection_audit', source_commit=execution,
        source_digest=publication['source_digest'], runtime_sha256=old_audit['runtime_sha256'],
        inputs_sha256={**hashes, 'historical203_charge_index': stable_hash(old_charges),
                      'aggregate210_charge_index': stable_hash(charges), 'audit_source': file_hash(Path(__file__))},
        imported_collection_helper_sha256=HELPER_SHA,
        coverage=dict(roles=3, cases=21, exact_cached_control_cases=14, new_paid_attempts=7,
                      aggregate_unique_paid_attempts=210, source_files=len(source_files), **dict(counts)),
        costs=dict(prior_paid_seconds=prior_total, new_paid_seconds=new_total, aggregate_paid_seconds=total,
                   cached_new_paid_seconds=0, shared_paid_ceiling_seconds=8000),
        checks={name: 'PASS' for name in ('original_certificates_grades_metrics_and_training_GIFs',
            'both_exact_cached_controls', 'complete_task_bound_recipe_delta_only_loss_on_all_seven_hosts',
            'native_NS_and_joint_encoder_reversed_stream_immutable_public_source',
            'new_and_AMS_parent_both_R1_R2_positive_constant_every_update_actual_penalty_histories',
            'actual_AMSGrad_role_betas_maxima_and_full_optimizer_histories',
            'actual_initial_tensor_and_ordered_target_batch_receipts_match_parent',
            'all_named_stream_receipts_match_checkpoints', 'own_clock_branches',
            'all203_historical_charges_retained_seven_new210_aggregate', 'no_scientific_retries')},
        actual_optimizer_groups=moments, actual_score_law_bindings=score_laws, task_bound_recipe_deltas=deltas,
        optimizer_updates_added=0, sampling_draws_added=0, live_evaluators_or_rescoring=False,
        model_construction_training_sampling_and_random_draws_forbidden=True,
        limits='Original stored independent grades only. Loss/penalty binding uses actual complete saved recipes, '
            'optimizer/penalty counters and immutable public source, not saved per-update paired scores or forces. '
            'Initial tensors/data are verified receipt identities, not reconstructed. Clock keeps its own cohort. '
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
    with helper.forbid_live_execution(), torch.no_grad():
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
