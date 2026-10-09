"""Project certified integration results and saved-training GIFs; never execute science.

Ordinary and optional diagnostic scopes stay separate. Unreached requirements
remain visible beside all 27 required questions. Raw curves, events, tensors and
checkpoints remain in their original artifact archives.
"""
from collections import Counter
from pathlib import Path
import argparse
import importlib.util
import json
import math
import operator
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ARCHIVE = Path('/mnt/ml7tb/ParticleGAN-forge/bcap-develop-integration-next')
sys.path.insert(0, str(ROOT))
from experiments.forge.artifacts import verify_artifacts
from experiments.forge.contracts import atomic_json, atomic_text, file_hash, read_json, stable_hash
from experiments.forge.state import state_digest
from experiments.forge.tier1_media import _indices, _scored_outputs, render
from reports.forge.regenerate_technique_inventory import project_receipt

ROLES = ('winner', 'combined')
ACTIVE = {'queued', 'running', 'paused'}


def arguments(description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--repository', type=Path, default=ROOT)
    parser.add_argument('--queue', type=Path, default=ARCHIVE / 'queue')
    parser.add_argument('--progress', type=Path, default=ARCHIVE / 'progress.json')
    parser.add_argument('--diagnostic-queue', type=Path)
    parser.add_argument('--diagnostic-progress', type=Path)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--allow-partial', action='store_true', help='Label a running snapshot as partial; adds no training.')
    return parser


def require(condition, message):
    if not condition:
        raise ValueError(message)


def required_questions(root):
    view = read_json(root / 'configs/forge/views/discriminator_stability.json')
    rows = sorted((a for a in view['assignments'] if a.get('importance') == 'required'
                   and a['qualification_tier'] <= 2), key=lambda a: (a['qualification_tier'], a['order']))
    require(view['revision'] == 8 and Counter(a['qualification_tier'] for a in rows) == {1: 6, 2: 21},
            'Expected the frozen revision-8 six/21 required task contract')
    return rows


def scopes(options):
    require(bool(options.diagnostic_queue) == bool(options.diagnostic_progress),
            'Supply both --diagnostic-queue and --diagnostic-progress')
    definitions = [('ordinary', options.queue, options.progress)]
    if options.diagnostic_queue:
        definitions.append(('research_diagnostic', options.diagnostic_queue, options.diagnostic_progress))
    loaded = []
    for scope, queue, progress_path in definitions:
        state = read_json(queue / 'queue/state.json')
        progress = read_json(progress_path)
        request_ids = progress['requests']
        require(set(request_ids) == set(ROLES) if scope == 'ordinary' else
                bool(request_ids) and set(request_ids) <= set(ROLES), f'{scope}: invalid winner/combined request roles')
        submissions = {role: state['submissions'][rid] for role, rid in request_ids.items()}
        jobs = [j for j in state['jobs'].values() if set(j.get('subscribers', [])) & set(request_ids.values())]
        active = [role for role, sub in submissions.items() if sub['status'] in ACTIVE]
        active += [j['definition']['task_id'] for j in jobs if j['status'] in ACTIVE]
        require(not active or options.allow_partial, f'{scope}: work remains active: {active}')
        for role, sub in submissions.items():
            request = sub['request']
            actual_scope = request['view'].get('evidence_scope', 'ordinary')
            require((scope == 'ordinary' and actual_scope not in {'research_diagnostic', 'calibration_diagnostic'})
                    or (scope == 'research_diagnostic' and actual_scope == scope),
                    f'{scope}/{role}: request scope differs: {actual_scope}')
        blockers = {}
        admission = options.repository / 'reports/forge/bcap-develop-integration/deeper-preregistration.json'
        if scope == 'research_diagnostic' and admission.is_file():
            declared = read_json(admission)
            combined = submissions.get('combined')
            require(combined is not None and declared['source_digest'] == combined['request']['source']['digest']
                    and declared['candidate_revision'] == combined['request']['candidate_revision'],
                    'Diagnostic blocker declaration does not bind this candidate/source')
            blockers = {row['task_id']: row for row in declared.get('blocked_holds', [])}
        loaded.append(dict(scope=scope, queue=queue, state=state, requests=request_ids,
                           submissions=submissions, jobs=jobs, active=active, diagnostic_blockers=blockers))
    return loaded


def certified_attempt(root, attempt_id):
    compact = project_receipt(root, attempt_id)
    directory = root / 'reports/forge/attempts' / attempt_id
    envelope, result, certificate = [read_json(directory / f'{name}.json')
                                     for name in ('request', 'result', 'evidence')]
    request = envelope.get('request', envelope)
    local = Path(certificate['local_artifact_root'])
    require(read_json(local / 'result.json') == result, f'{attempt_id}: local result differs from certificate')
    require(read_json(local / 'request.json') == envelope, f'{attempt_id}: local request differs from certificate')
    if result.get('retry_of'):
        require(result['retry_of'] == envelope.get('retry_of'), f'{attempt_id}: retry chain differs')
    return dict(compact=compact, request=request, result=result, certificate=certificate, local=local)


def checkpoint(row):
    descriptor = row.get('evidence', {}).get('provenance_checkpoint')
    if not descriptor:
        return None, None
    root = Path(descriptor['artifact_root']).resolve()
    verify_artifacts(root, descriptor['artifact_manifest'])
    path = (root / descriptor['path']).resolve()
    require(path.is_relative_to(root) and file_hash(path) == descriptor['sha256'],
            f'{row["task_id"]}: provenance checkpoint bytes differ')
    require(path.stat().st_size == descriptor['bytes'], f'{row["task_id"]}: checkpoint size differs')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    require(state_digest(saved) == descriptor['state_sha256'], f'{row["task_id"]}: saved state digest differs')
    require(saved['streams']['states'].keys() == descriptor['named_stream_state_sha256'].keys(),
            f'{row["task_id"]}: named checkpoint stream keys differ')
    for key, value in saved['streams']['states'].items():
        require(state_digest(value) == descriptor['named_stream_state_sha256'][key],
                f'{row["task_id"]}: named checkpoint stream digest differs: {key}')
    compact = {key: descriptor[key] for key in ('path', 'sha256', 'bytes', 'state_sha256', 'completed_steps')}
    compact.update(artifact_root=str(root), artifact_manifest_sha256=descriptor['artifact_manifest']['sha256'],
                   purpose=descriptor.get('purpose'), prerequisite_eligible=descriptor.get('prerequisite_eligible'))
    return saved, compact


def mechanism_stats(saved):
    found = {}
    def visit(value, path='state'):
        if isinstance(value, dict):
            for name in ('constraint_geometry', 'direction_blend', 'strict_progress'):
                if isinstance(value.get(name), dict) and 'stats' in value[name]:
                    found[f'{path}.{name}'] = value[name]['stats']
            transport = value.get('component_transport')
            if isinstance(transport, dict):
                found[f'{path}.component_transport'] = transport
            for key, child in value.items():
                if key not in {'models', 'role_parameters', 'streams', 'initialization'}:
                    visit(child, f'{path}.{key}')
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f'{path}[{index}]')
    visit(saved)
    return found


def collect(options):
    root = options.repository.resolve()
    requirements = required_questions(root)
    loaded = scopes(options)
    attempts, final, cells, accounting = {}, [], [], []
    for context in loaded:
        state, scope = context['state'], context['scope']
        used_ids = set()
        for job in context['jobs']:
            for attempt in job.get('attempts', []):
                aid = attempt['attempt_id']
                if not (root / 'reports/forge/attempts' / aid / 'evidence.json').is_file() and job['status'] in ACTIVE:
                    require(options.allow_partial, f'{aid}: active attempt has no completed certificate')
                    continue
                if aid not in attempts:
                    attempts[aid] = certified_attempt(root, aid)
                used_ids.add(aid)
            if job.get('result'):
                aid = job['result']['attempt_id']
                if aid not in attempts:
                    attempts[aid] = certified_attempt(root, aid)
                require(job['result'] == attempts[aid]['result'], f'{aid}: queue result differs from certificate')
        charges = [c for c in state['charges'] if c['attempt_id'] in used_ids]
        require(len(charges) == len({c['attempt_id'] for c in charges}), f'{scope}: duplicate paid charges')
        charged = {c['attempt_id']: c['seconds'] for c in charges}
        for aid in used_ids:
            paid = sum(r.get('cost', {}).get('wall_seconds', 0) for r in attempts[aid]['result']['task_results'])
            require(aid in charged and math.isclose(paid, charged[aid], abs_tol=1e-7), f'{aid}: paid cost differs')
        for job in context['jobs']:
            actual = sum(charged.get(a['attempt_id'], 0) for a in job.get('attempts', []))
            require(math.isclose(actual, job.get('charged_seconds', 0), abs_tol=1e-7),
                    f'{scope}/{job["definition"]["task_id"]}: job charge total differs')
        accounting.append(dict(scope=scope, queue_root=str(context['queue']),
            campaign_accounting=state['campaigns'], selected_paid_seconds=sum(charged.values()),
            paid_attempts=len(charged), execution_retries=sum(max(0, len(j['attempts']) - 1) for j in context['jobs']),
            executed_full_allowances_seconds=sum(j['definition']['budget_seconds'] * len(j['attempts']) for j in context['jobs']),
            pending_work=context['active']))
        for role, sub in context['submissions'].items():
            request = sub['request']
            rid = context['requests'][role]
            available = {}
            for job in context['jobs']:
                if rid not in job.get('subscribers', []) or not job.get('result'):
                    continue
                aid = job['result']['attempt_id']
                attempt = attempts[aid]
                require(attempt['request']['source'] == request['source']
                        and attempt['request']['runtime'] == request['runtime']
                        and attempt['result']['candidate_revision'] == request['candidate_revision'],
                        f'{scope}/{role}/{aid}: certified source/runtime/candidate differs')
                for row in attempt['result']['task_results']:
                    tid = row['task_id']
                    require(tid not in available, f'{scope}/{role}/{tid}: multiple selected final results')
                    task = request['tasks'][tid]
                    declared = next(j['compatibility_key'] for j in request['jobs']
                                    if tid in j.get('task_ids', [j['task_id']]))
                    require(row['compatibility_key'] == declared, f'{scope}/{role}/{tid}: task binding differs')
                    compact = next(r for r in attempt['compact']['task_results'] if r['task_id'] == tid)
                    saved, proof = checkpoint(row)
                    item = dict(scope=scope, role=role, request_id=rid, candidate_id=request['candidate']['id'],
                        candidate_revision=request['candidate_revision'], task_id=tid, attempt_id=aid,
                        study_id=request.get('study', {}).get('id'), source_commit=request['source'].get('origin_commit'),
                        source_digest=request['source']['digest'], task_contract_sha256=stable_hash(task),
                        provenance_checkpoint=proof,
                        mechanism_stats=mechanism_stats(saved) if saved is not None else {})
                    item.update(compact)
                    assignment = next(a for a in request['view']['assignments'] if a['task'] == tid)
                    item.update(importance=assignment.get('importance', 'required'), tier=assignment['qualification_tier'])
                    available[tid] = item
                    final.append(dict(item=item, row=row, task=task, request=request,
                                      attempt=attempt, saved=saved, context=context))
            for requirement in requirements:
                tid = requirement['task']
                if tid in available:
                    item = available[tid]
                    cells.append(dict(scope=scope, role=role, task_id=tid, tier=requirement['qualification_tier'],
                                      gate_status=item['gate_status'], attempt_id=item['attempt_id'], reason=item.get('reason', '')))
                    continue
                jobs = [j for j in context['jobs'] if rid in j.get('subscribers', [])
                        and tid in j['definition'].get('task_ids', [j['definition']['task_id']])]
                task = request['tasks'].get(tid)
                blockers = (task or {}).get('preflight_blockers', [])
                if role == 'combined' and tid in context['diagnostic_blockers']:
                    blockers = [context['diagnostic_blockers'][tid]]
                submission_blocked = bool(jobs) and sub.get('status') == 'blocked'
                status = 'BLOCKED' if blockers or submission_blocked or any(j['status'] == 'blocked' for j in jobs) else 'UNMEASURED'
                reason = blockers or [j.get('reason') for j in jobs if j.get('reason')]
                if submission_blocked and sub.get('reason') is not None:
                    reason = [*reason, sub['reason']]
                if not reason:
                    reason = ['Not declared in this diagnostic scope' if scope != 'ordinary' and not jobs
                              else 'No certified final result; ordinary eligibility or own checkpoint remains unmet']
                cells.append(dict(scope=scope, role=role, task_id=tid, tier=requirement['qualification_tier'],
                                  gate_status=status, reason=reason, paid_attempts=sum(len(j['attempts']) for j in jobs)))
        for role in set(ROLES) - set(context['submissions']):
            cells.extend(dict(scope=scope, role=role, task_id=a['task'], tier=a['qualification_tier'],
                gate_status='UNMEASURED', reason=['No request for this role in the declared diagnostic scope'],
                paid_attempts=0) for a in requirements)
    return dict(requirements=requirements, scopes=loaded, attempts=attempts, final=final,
                cells=cells, accounting=accounting)


def _saved_renderer(root):
    path = root / 'reports/forge/gaussian-smoke-inventory/export_media.py'
    spec = importlib.util.spec_from_file_location('integration_saved_media', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def saved_outputs(task, evidence, local):
    """Select an existing scored cadence from retained component-view supersets."""
    try:
        samples, inputs = _scored_outputs(task, evidence, local)
        return samples, inputs, None
    except ValueError as error:
        require(task['adapter'] == 'transfer_behavior'
                and str(error) == 'scored output and numerical observation schedules differ', str(error))
    descriptor = evidence['saved_observer_outputs']
    root = Path(local).resolve()
    path = (root / descriptor['path']).resolve()
    require(path.is_relative_to(root) and file_hash(path) == descriptor['sha256']
            and path.stat().st_size == descriptor['bytes'], 'Retained component views differ from certified bytes')
    retained = torch.load(path, map_location='cpu', weights_only=True)
    require(isinstance(retained, list) and all(isinstance(point, dict) for point in retained),
            'Retained component views must be observation objects')
    selected, indices = [], []
    for observed in evidence['observations']:
        metrics = {key: value for key, value in observed.items() if key != 'step'}
        matches = [index for index, point in enumerate(retained)
                   if point.get('step') == observed['step'] and point.get('metrics') == metrics]
        require(matches, f'{task["id"]}: no saved view with the certified step and exact numerical metrics')
        # Duplicate final views are actual saved records from the same scoring
        # point. Select the first matching record and retain its source index.
        indices.append(matches[0])
        selected.append(retained[matches[0]])
    return selected, {str(path): file_hash(path)}, dict(kind='exact_scored_subset_of_retained_views',
        retained_records=len(retained), source_record_indices=indices, added_observations=0)


def render_saved(entry, output, renderer):
    task, row, local = entry['task'], entry['row'], entry['attempt']['local']
    output.parent.mkdir(parents=True, exist_ok=True)
    require(entry['request']['source']['files']['experiments/forge/tier1_media.py'] == file_hash(Path(sys.modules[render.__module__].__file__)),
            f'{task["id"]}: saved-observation renderer differs from executed source')
    if task['adapter'] == 'native100':
        return renderer.render_native(task, row, output)
    samples, inputs, selection = saved_outputs(task, row['evidence'], local)
    require(renderer.finite_tree(samples) and renderer.finite_tree(row['evidence'].get('observations', []))
            and row['evidence'].get('guards', {}).get('all_finite') is not False,
            f'{task["id"]}: nonfinite saved training evidence')
    from benchmarks.toy_audit import api_run
    from experiments.forge import tier1_media
    require(entry['request']['source']['files']['benchmarks/toy_audit/api_run.py'] == file_hash(Path(api_run.__file__)),
            f'{task["id"]}: API visualization source differs from executed source')
    original = api_run.render_gif
    def scalar_display(case, records, *args, **kwargs):
        records = [{**r, 'metrics': {k: v for k, v in r['metrics'].items() if type(v) in (int, float)}}
                   for r in records]
        return original(case, records, *args, **kwargs)
    with patch.object(api_run, 'render_gif', scalar_display), patch.object(tier1_media, '_scored_outputs',
            lambda *_args: (samples, inputs)):
        if samples and task['adapter'] == 'transfer_image':
            observations = row['evidence']['observations']
            indices = _indices(len(samples))
            ops = {'<=': operator.le, '>=': operator.ge, '==': operator.eq, '<': operator.lt, '>': operator.gt}
            def failures(point):
                return [f'{name} {op} {bound}' for name, op, bound in task['evaluation']['thresholds']
                        if not ops[op](point[name], bound)]
            records = [dict(step=samples[i]['step'], metrics=observations[i],
                passed=not failures(observations[i]), failed_bounds=failures(observations[i]),
                views=[dict(kind='image', title='Original target templates and saved generated outputs',
                    target=samples[i]['targets'], samples=samples[i]['samples'],
                    caption='Actual scored training outputs; display preserves the original clean enumeration.')]) for i in indices]
            case = dict(id=task['id'], goal=task.get('description', task['id']),
                        default_steps=task['execution']['steps'], sampling=row['evidence']['sampling_law'])
            api_run.render_gif(case, records, output, full_budget=True, requested_steps=task['execution']['steps'],
                               final_verdict=row['gate_status'])
            receipt = dict(schema_version=1, task_id=task['id'], recorded_grade=row['gate_status'],
                kind='actual_training_saved_images_gif', source_inputs=inputs, observation_count=len(observations),
                selected_observation_indices=indices, observations_sha256=stable_hash(observations),
                gif_sha256=file_hash(output), optimizer_updates_added=0, sampling_draws_added=0,
                qualification_input=False, renderer_sha256=file_hash(Path(__file__)))
        else:
            receipt = render(task, row, local, output)
    receipt['display_metrics'] = 'Saved scalar metrics; original structured metadata remains in certified evidence.'
    if selection is not None:
        receipt['saved_record_selection'] = selection
    atomic_json(output.with_suffix('.json'), receipt)
    return receipt


def task_table(data):
    lookup = {(c['scope'], c['role'], c['task_id']): c for c in data['cells']}
    columns = [(s['scope'], role) for s in data['scopes'] for role in ROLES]
    lines = ['# Required-task comparison', '',
             'Display projection only. Ordinary qualification and research diagnostics retain separate columns.', '',
             '| Tier | Original question | ' + ' | '.join(f'{scope}: {role}' for scope, role in columns) + ' |',
             '| --- | --- | ' + ' | '.join('---' for _ in columns) + ' |']
    for requirement in data['requirements']:
        tid = requirement['task']
        values = []
        for scope, role in columns:
            cell = lookup[(scope, role, tid)]
            status = cell['gate_status']
            if cell.get('attempt_id'):
                status = f'[{status}](results.json)'
            values.append(status)
        lines.append(f'| {requirement["qualification_tier"]} | {tid} | ' + ' | '.join(values) + ' |')
    lines += ['', 'Final metrics, grading reasons, blockers, paid retries and checkpoint/source identities are in [results.json](results.json).', '']
    return '\n'.join(lines)


def main():
    parser = arguments()
    parser.add_argument('--skip-media', action='store_true', help='Write an explicitly media-incomplete projection.')
    options = parser.parse_args()
    torch.set_num_threads(1)
    data = collect(options)
    out = options.output.resolve()
    media = []
    renderer = _saved_renderer(options.repository) if not options.skip_media else None
    for entry in data['final']:
        item = entry['item']
        if item['gate_status'] not in {'PASS', 'FAIL'}:
            item['media_status'] = 'Unavailable: no completed scientific gate'
            continue
        if options.skip_media:
            item['media_status'] = 'Not exported: --skip-media'
            continue
        gif = out / 'media' / f'{item["scope"]}-{item["role"]}-{item["task_id"]}.gif'
        from PIL import Image
        try:
            with renderer.forbid_live_execution():
                receipt = render_saved(entry, gif, renderer)
            with Image.open(gif) as saved_gif:
                frames = saved_gif.n_frames
            require(frames >= 2, f'{item["task_id"]}: actual-training GIF needs at least two saved states')
            media.append(dict(scope=item['scope'], role=item['role'], attempt_id=item['attempt_id'],
                gif=str(gif.relative_to(out)), frames=frames, **receipt))
            item['media_status'] = 'Exported from certified saved training observations'
        except Exception as error:
            for path in (gif, gif.with_suffix('.json')):
                if path.exists():
                    path.unlink()
            item.update(media_status='BLOCKED: saved-training export could not be verified',
                        media_blocker=dict(error_type=type(error).__name__, reason=str(error)))
            media.append(dict(scope=item['scope'], role=item['role'], task_id=item['task_id'],
                attempt_id=item['attempt_id'], gif=None, recorded_grade_unchanged=True, **item['media_blocker']))
        print(json.dumps(dict(event='media_projection', scope=item['scope'], role=item['role'],
            task_id=item['task_id'], status=item['media_status'], blocker=item.get('media_blocker'))), flush=True)
    selected_ids = {e['item']['attempt_id'] for e in data['final']}
    histories = [dict(a['compact'], final_selected=aid in selected_ids)
                 for aid, a in sorted(data['attempts'].items())]
    result = dict(schema_version=1, qualification_input=False,
        scope='matched_develop_integration_with_separate_ordinary_and_diagnostic_results',
        required_counts=dict(tier1=6, tier2=21), protocol_seed=0,
        partial=any(s['active'] for s in data['scopes']),
        required_task_cells=data['cells'], task_results=[e['item'] for e in data['final']],
        outcomes={s['scope']: {role: dict(Counter(c['gate_status'] for c in data['cells']
                  if c['scope'] == s['scope'] and c['role'] == role)) for role in ROLES} for s in data['scopes']},
        accounting=data['accounting'], paid_attempt_history=histories,
        media_export_complete=not options.skip_media and all(m.get('gif') for m in media),
        actual_training_gifs=sum(bool(m.get('gif')) for m in media),
        optimizer_updates_added=0, sampling_draws_added=0,
        interpretation='Passing diagnostic subsets cannot fill ordinary qualification cells or change public defaults.')
    atomic_json(out / 'results.json', result)
    atomic_json(out / 'media/index.json', dict(schema_version=1, qualification_input=False, media=media,
        optimizer_updates_added=0, sampling_draws_added=0))
    atomic_text(out / 'task-table.md', task_table(data))
    print(json.dumps(dict(event='publication_complete', outcomes=result['outcomes'], media=result['actual_training_gifs'],
                         media_blockers=sum(not m.get('gif') for m in media),
                         paid_attempts=len(histories), output=str(out))), flush=True)


if __name__ == '__main__':
    main()
