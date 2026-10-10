"""Read-only audit of all frozen round-five arms and durable attempt certificates."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import subprocess

import torch
from experiments.forge.contracts import stable_hash

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repository', type=Path, default=Path.cwd())
parser.add_argument('--brief', type=Path, default=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/handoff'))
options = parser.parse_args()
REPO = options.repository.resolve()
BRIEF = options.brief.resolve()
GENERATED = {'field_ownership', 'preflight_blockers'}
EXPECTED_ARMS = {'conditional_integration': 4, 'gaussian_regression': 3,
                 'mass_allocation': 2, 'component_tails': 2, 'force_distortion': 2}
ANCILLARY_ALLOWANCES = {'conditional_integration': 1200, 'gaussian_regression': 420,
                       'mass_allocation': 360, 'component_tails': 720, 'force_distortion': 1920}


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def task_conditions(request):
    return {name: {key: value for key, value in task.items() if key not in GENERATED}
            for name, task in request['tasks'].items()}


def original_conditions(slug, request, original):
    proofs = {}
    for tid, card in task_conditions(request).items():
        if tid in original:
            assert card == original[tid], (slug, tid, 'original task changed')
            proofs[tid] = dict(original_id=tid, unchanged_original=True, numerical_conditions_preserved=True)
            continue
        assert slug == 'conditional_integration' and tid.endswith('_transport_round5_v1'), (slug, tid)
        oid = tid.removesuffix('_transport_round5_v1')
        assert oid in {'trajectory', 'residual_student', 'mid_scale_identity'}
        restored = copy.deepcopy(card)
        assert restored.pop('retained_question_ids') == [oid]
        assert restored.pop('task_cohort') == 'conditional_transport_round5_v1'
        description = restored.pop('description')
        assert 'original objectives, conditions and full gates retained' in description
        restored['id'] = oid
        assert restored['execution'].pop('transport_consumer') == 'output_marginal_v1'
        contract = restored['execution'].pop('transport_contract')
        assert contract == dict(conditioning='four_original_training_scales' if oid == 'mid_scale_identity' else 'slow_arc',
                                conditioning_in_distance=False,
                                gradients='generated_outputs_to_original_network_and_learned_prior_locations',
                                new_paired_supervision=False, paired_correctness='original_objectives_and_original_full_identity_gates',
                                sample_space='output_panel', target_law='original_training_target_panel')
        sources = restored['evaluation'].pop('sources')
        old_sources = original[oid]['evaluation'].get('sources', {})
        assert all(sources[path] == expected for path, expected in old_sources.items())
        host_source = 'benchmarks/locked_shared/trajectory.py' if oid == 'trajectory' else f'benchmarks/locked_shared/hosts/{oid}.py'
        assert set(sources) - set(old_sources) == {host_source, 'experiments/forge/behavior_adapters.py',
                                                 'particlegan/conditional_transport.py', 'particlegan/kinetic_transport.py'}
        if 'sources' in original[oid]['evaluation']:
            restored['evaluation']['sources'] = old_sources
        for path, expected in sources.items():
            assert request['source']['files'][path] == expected, (slug, tid, path)
        assert restored == original[oid], (slug, tid, 'variant changed other conditions')
        proofs[tid] = dict(original_id=oid, unchanged_original=False, explicit_variant=True,
                          numerical_conditions_preserved=True, declared_consumer=contract,
                          variant_source_pins=sources, original_task_hash=stable_hash(original[oid]))
    return proofs


def role_for(slug, candidate):
    if slug == 'component_tails':
        assert candidate in {'kinetic_transport_local_v2', 'component_prior_transport_r5'}
        return 'control' if candidate == 'kinetic_transport_local_v2' else 'candidate'
    role = candidate.split('-round5-')[-1].removesuffix('-v1')
    assert role in {'winner', 'direction', 'transport', 'both', 'local', 'finite', 'control', 'candidate'}, (slug, candidate)
    return role


original = read_json(BRIEF / 'original-task-conditions.json')
tracks, final_attempts, historical_attempts = [], [], []
for track in read_json(BRIEF / 'manifest.json'):
    slug = track['slug']
    done = read_json(BRIEF / slug / 'done.json')
    worktree = Path(track['worktree'])
    state = read_json(Path(track['queue_root']) / 'queue/state.json')
    submissions = list(state['submissions'].values())
    assert len(submissions) == EXPECTED_ARMS[slug], (slug, len(submissions))
    assert all(entry['status'] == 'concluded' for entry in submissions), slug
    request_roles = {entry['request']['request_id']: role_for(slug, entry['request']['candidate']['id']) for entry in submissions}
    if done.get('requests'):
        assert request_roles == {request_id: role for role, request_id in done['requests'].items()}
    requests = [entry['request'] for entry in submissions]
    assert set(request_roles) == {request['request_id'] for request in requests}
    reference = requests[0]
    original_parity = original_conditions(slug, reference, original)
    bindings = {key: all(request[key] == reference[key] for request in requests)
                for key in ['protocol', 'rng', 'runtime', 'compute_profiles']}
    bindings.update(task_scientific_declarations=all(task_conditions(request) == task_conditions(reference) for request in requests),
                    executed_source_files=all(request['source']['files'] == reference['source']['files'] for request in requests),
                    executed_source_digest=all(request['source']['digest'] == reference['source']['digest'] for request in requests))
    assert all(bindings.values()), (slug, bindings)
    snapshot = Path(reference['source']['snapshot_path'])
    for relative, expected in reference['source']['files'].items():
        assert sha(snapshot / relative) == expected, (slug, 'snapshot changed', relative)
        assert sha(worktree / relative) == expected, (slug, 'measured publication changed', relative)

    arms, checkpoint_proofs = [], defaultdict(list)
    track_histories, track_paid = [], 0.0
    for entry in submissions:
        request = entry['request']
        role = request_roles[request['request_id']]
        workers, grades = [], {}
        for job in state['jobs'].values():
            if (job.get('cost_owner') or {}).get('request') != request['request_id']:
                continue
            assert job['status'] not in {'running', 'queued'}, (slug, 'live work')
            if job['status'] != 'terminal':
                assert len(job['attempts']) == 0 and job['charged_seconds'] == 0 and job['reserved_seconds'] == 0
                continue
            paid_attempts = 0.0
            for attempt in job['attempts']:
                directory = Path(attempt['path'])
                result = read_json(directory / 'result.json')
                certificate = worktree / 'reports/forge/attempts' / result['attempt_id']
                evidence = read_json(certificate / 'evidence.json')
                assert stable_hash(result) == evidence['result_hash']
                assert result == read_json(certificate / 'result.json')
                certified_request = read_json(certificate / 'request.json')['request']
                assert certified_request['source']['digest'] == reference['source']['digest']
                assert certified_request['request_id'] == request['request_id']
                assert certified_request['candidate_revision'] == request['candidate_revision']
                assert len(result['task_results']) == 1
                row = result['task_results'][0]
                paid = row['cost']['wall_seconds']
                paid_attempts += paid
                receipt = dict(attempt_id=result['attempt_id'], task_id=row['task_id'],
                               gate_status=row['gate_status'], paid_worker_seconds=paid,
                               result_hash=stable_hash(result),
                               certificate_sha256={filename: sha(certificate / filename) for filename in ['request.json', 'result.json', 'evidence.json']})
                if result['attempt_id'] != job['result']['attempt_id']:
                    assert slug in {'gaussian_regression', 'mass_allocation', 'force_distortion'}
                    assert row['gate_status'] == 'INCOMPLETE' and result['raw']['attempt_status'] == 'timeout'
                    assert row['task_id'] == 'gaussian1d_smoke'
                    assert job['retry_of']['attempt_id'] in {earlier['attempt_id'] for earlier in job['attempts'][:-1]}
                    receipt.update(role=role, artifact_root=str(directory), retained_status='INCOMPLETE',
                                   superseded_for_execution_only_by=job['result']['attempt_id'],
                                   reason=result['raw']['reason'], full_allowance_seconds=job['definition']['budget_seconds'])
                    track_histories.append(receipt)
                    historical_attempts.append(receipt)
                    continue
                assert result == job['result'] and result['raw']['attempt_status'] == 'completed'
                assert row['gate_status'] in {'PASS', 'FAIL'}
                assert row['task_id'] not in grades
                grades[row['task_id']] = row['gate_status']
                receipt['final_metrics'] = row.get('metrics', {})
                evaluator = row.get('evaluator_result', {})
                receipt['sustained_summary'] = {key: evaluator[key] for key in ['convergence', 'coverage_status'] if key in evaluator}
                if 'terminal_checks' in evaluator:
                    receipt['sustained_summary']['terminal_checks'] = [
                        {'step': check['step'], 'passed': check['passed']}
                        for check in evaluator['terminal_checks']]
                if 'gaussian_grade' in row:
                    receipt['gaussian_retention_summary'] = row['gaussian_grade']
                workers.append(receipt)
                final_attempts.append(receipt)
                assert all(audit.get('unintended_rng_deviations', 0) == 0 for audit in row.get('evidence', {}).get('rng_audits', []))
                path = directory / 'provenance/provenance-state.pt'
                certified_checkpoint = row['evidence']['provenance_checkpoint']
                assert (Path(certified_checkpoint['artifact_root']) / certified_checkpoint['path']).resolve() == path.resolve()
                assert sha(path) == certified_checkpoint['sha256'], (slug, row['task_id'], 'checkpoint certificate changed')
                saved = torch.load(path, map_location='cpu', weights_only=False)
                metadata = saved.get('applied', saved)
                assert metadata.get('initializer') is not None and metadata.get('initialization') is not None
                assert metadata.get('prior') is not None
                assert all(audit.get('unintended_rng_deviations', 0) == 0 for audit in metadata.get('rng_audits', []))
                checkpoint_proofs[row['task_id']].append(dict(role=role, initialization=metadata['initialization'],
                                                            initializer=metadata['initializer'], prior=metadata['prior'],
                                                            streams=saved['streams'], checkpoint_sha256=sha(path)))
            assert math.isclose(paid_attempts, job['charged_seconds'], abs_tol=1e-7), (slug, paid_attempts, job['charged_seconds'])
            track_paid += job['charged_seconds']
        for tid in request['tasks']:
            if tid not in grades:
                grades[tid] = 'BLOCKED'
        record = read_json(worktree / 'reports/forge/records' / (entry['readout_record_id'] + '.json'))
        certified_ids = {worker['attempt_id'] for worker in workers}
        assert certified_ids <= set(record['attempt_ids'])
        arms.append(dict(role=role, candidate_id=request['candidate']['id'], candidate_revision=request['candidate_revision'],
                         request_id=request['request_id'], study_id=request['study']['id'],
                         declared_primary_control=request['study']['control']['candidate_id'],
                         source_origin_commit=request['source']['origin_commit'], readout_record_id=entry['readout_record_id'],
                         task_grades=grades, outcomes=dict(Counter(grades.values())),
                         workers=sorted(workers, key=lambda worker: worker['task_id'])))

    proof_parity = {}
    for tid, proofs in checkpoint_proofs.items():
        first = proofs[0]
        if len(proofs) < 2:
            continue
        checks = {key: all(equal(first[key], proof[key]) for proof in proofs[1:]) for key in ['initialization', 'initializer', 'prior']}
        checks['named_stream_bindings'] = all(equal(first['streams']['manifest'], proof['streams']['manifest']) for proof in proofs[1:])
        stream_states = first['streams']['states']
        non_eval = [key for key in stream_states if json.loads(key)[0] != 'eval']
        checks['consumed_training_stream_states'] = all(equal(stream_states[key], proof['streams']['states'][key]) for key in non_eval for proof in proofs[1:])
        assert all(checks.values()), (slug, tid, checks)
        differing_evaluation_streams = sorted({key for proof in proofs[1:] for key in stream_states if json.loads(key)[0] == 'eval' and not equal(stream_states[key], proof['streams']['states'][key])})
        assert all(json.loads(key)[2] == 'smoke_confirmation' for key in differing_evaluation_streams), (slug, tid, differing_evaluation_streams)
        proof_parity[tid] = dict(arms=len(proofs), checks=checks,
                                adaptive_independent_smoke_confirmation_streams=differing_evaluation_streams)

    publication = json.loads(subprocess.check_output(['gh', 'pr', 'view', done.get('pr_url', done.get('PR')), '--json', 'headRefOid,isDraft,state,url,baseRefName']))
    head = subprocess.check_output(['git', '-C', str(worktree), 'rev-parse', 'HEAD'], text=True).strip()
    assert head == done['publication_commit'] == publication['headRefOid'] and publication['state'] == 'OPEN' and not publication['isDraft']
    assert publication['baseRefName'] == track['pr_base']
    report_rel = track['report_relative_path']
    report_bytes = subprocess.check_output(['git', '-C', str(worktree), 'show', f'{head}:{report_rel}'])
    gifs = [name for name in subprocess.check_output(['git', '-C', str(worktree), 'ls-tree', '-r', '--name-only', head, str(Path(report_rel).parent)], text=True).splitlines() if name.endswith('.gif')]
    assert len(gifs) == sum(len(arm['workers']) for arm in arms), (slug, len(gifs))
    paid = sum(campaign['spent_seconds'] for campaign in state['campaigns'].values())
    reserved = sum(campaign['reserved_seconds'] for campaign in state['campaigns'].values())
    ceiling = sum(campaign['definition']['budget_seconds'] for campaign in state['campaigns'].values())
    planned = sum(task['resources']['timeout_seconds'] for request in requests for task in request['tasks'].values())
    executed = sum(job['definition']['budget_seconds'] * len(job['attempts']) for job in state['jobs'].values())
    assert reserved == 0 and ceiling <= track['budget_seconds'] and paid <= ceiling
    assert math.isclose(paid, track_paid, abs_tol=1e-7)
    assert planned + sum(item['full_allowance_seconds'] for item in track_histories) + ANCILLARY_ALLOWANCES[slug] <= track['budget_seconds']
    tracks.append(dict(track=slug, pr_url=publication['url'], pr_base=publication['baseRefName'], publication_commit=head,
                       report_url=f'https://github.com/255BITS/ParticleGAN/blob/{head}/{report_rel}',
                       report_sha256=hashlib.sha256(report_bytes).hexdigest(), published_actual_training_gifs=len(gifs),
                       original_task_conditions_preserved=original_parity,
                       frozen_task_condition_hashes={tid: stable_hash(card) for tid, card in task_conditions(reference).items()},
                       actual_matched_execution_proofs=proof_parity, executed_source_digest=reference['source']['digest'],
                       trained_source_origin_commit=reference['source']['origin_commit'], matched_frozen_bindings=bindings,
                       task_parity_excludes_generated_fields=sorted(GENERATED), arms=sorted(arms, key=lambda arm: arm['role']),
                       paid_worker_seconds=paid, campaign_ceiling_seconds=ceiling, track_ceiling_seconds=track['budget_seconds'],
                       remaining_reserved_seconds=reserved, queue_root=track['queue_root'],
                       declared_full_allowances_seconds=planned, executed_full_allowances_including_retry_seconds=executed,
                       retries=sum(max(0, len(job['attempts']) - 1) for job in state['jobs'].values()), retained_attempt_history=track_histories,
                       scientific_snapshot_files_verified=len(reference['source']['files']),
                       ancillary_full_allowance_seconds=ANCILLARY_ALLOWANCES[slug],
                       published_scientific_files_matching_measured=len(reference['source']['files']),
                       unmeasured_publication_changes=[], recommendation=done['recommendation']))
    print(f"{slug}: audited {len(arms)} arms, {sum(len(arm['workers']) for arm in arms)} final workers, {len(track_histories)} retained earlier attempts", flush=True)

assert len(final_attempts) == len({attempt['attempt_id'] for attempt in final_attempts})
preserved = read_json(BRIEF / 'qualification-before.json')
for relative, expected in preserved.items():
    assert sha(REPO / relative) == expected, (relative, 'archived qualification/telemetry changed')
totals = dict(completed_runnable_scientific_jobs=len(final_attempts),
              measured_task_grades=dict(Counter(attempt['gate_status'] for attempt in final_attempts)),
              declared_task_cells=sum(len(arm['task_grades']) for track in tracks for arm in track['arms']),
              unexecuted_BLOCKED_cells=sum(arm['outcomes'].get('BLOCKED', 0) for track in tracks for arm in track['arms']),
              paid_attempts=len(final_attempts) + len(historical_attempts), retained_earlier_INCOMPLETE_attempts=len(historical_attempts),
              paid_worker_seconds=sum(track['paid_worker_seconds'] for track in tracks),
              campaign_ceiling_seconds=sum(track['campaign_ceiling_seconds'] for track in tracks),
              track_ceiling_seconds=sum(track['track_ceiling_seconds'] for track in tracks),
              ancillary_full_allowance_seconds=sum(ANCILLARY_ALLOWANCES.values()),
              planned_full_allowances_seconds=sum(track['declared_full_allowances_seconds'] for track in tracks),
              executed_full_allowances_including_retry_seconds=sum(track['executed_full_allowances_including_retry_seconds'] for track in tracks),
              execution_retries=sum(track['retries'] for track in tracks),
              published_actual_training_gifs=sum(track['published_actual_training_gifs'] for track in tracks),
              verified_scientific_snapshot_files=sum(track['scientific_snapshot_files_verified'] for track in tracks),
              original_root_qualification_telemetry_snapshots_preserved=len(preserved), remaining_reserved_seconds=0)
assert totals['track_ceiling_seconds'] == 129600
result = dict(schema_version=1, scope='source_bound_mechanism_diagnostic_comparison', round=5,
              qualification_input=False, protocol_seed=0, tracks=tracks, totals=totals,
              interpretation=dict(full_ordinary_suite_qualification=False, default_adoption=False,
                                  ranking='Scoped arm comparisons; differing controls and applicability prohibit pooled optimizer ranking',
                                  source_overlap='Deterministic repeated controls and shared-task arms are not statistical replications',
                                  cost_scope='Main scientific worker wall time includes four retained timeouts and four execution-only retries. Disposable saved probes and synthetic software checks have separate declared allowances/runtimes in track reports.',
                                  evaluation_streams='Adaptive smoke confirmation is separately named; fixed scoring/data/noise/prior streams retain matched consumption.',
                                  source_changes='Every frozen scientific file matches each ready publication; later helpers report existing evidence.'))
output = REPO / 'reports/forge/bcap-physics/round5/comparison.json'
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(totals, indent=2))
