"""Saved-only publication: compact results and the hydraulic goal's single leaderboard.

Reads the archived winner control (bcap-default-baseline results.json) and the
completed queue state of this study. Adds no training, sampling or updates.

    python reports/forge/bcap-physics/hydraulic/winner/publish.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
sys.path.insert(0, str(OUT))
from workflow import ARCHIVE, ARMS, CONTROL, NOT_CONSUMED, TASKS  # noqa: E402

CONTROL_RESULTS = ROOT / 'reports/forge/bcap-default-baseline/results.json'
LABELS = {'winner': 'A0 winner', 'travel': 'A1 travel (v1)', 'gap': 'A2 gap-adaptive'}


def rows():
    control = json.loads(CONTROL_RESULTS.read_text())
    result = {'winner': {r['task_id']: r for r in control['task_results'] if r['candidate_id'] == CONTROL}}
    state = json.loads((ARCHIVE / 'queue/queue/state.json').read_text())
    ids = {cid: arm for arm, cid in ARMS.items()}
    for arm in ARMS:
        result[arm] = {}
    for job in state['jobs'].values():
        if not job.get('result'):
            continue
        record = ROOT / 'reports/forge/attempts' / job['result']['attempt_id']
        envelope = json.loads((record / 'request.json').read_text())
        arm = ids.get(envelope.get('request', envelope)['candidate']['id'])
        certified = json.loads((record / 'result.json').read_text())
        assert certified == job['result'], 'queue result differs from certified attempt'
        for row in certified['task_results']:
            if arm and row['task_id'] in TASKS:
                result[arm][row['task_id']] = dict(row, attempt_id=certified['attempt_id'],
                    source_digest=envelope.get('request', envelope)['source']['digest'])
    return result, control


def summary(row):
    return row.get('evaluator_summary') or row.get('evaluator_result') or {}


def conv(row):
    return summary(row).get('convergence') or {}


def key_metrics(task, row):
    if row is None:
        return {}
    m, s = row.get('metrics') or {}, summary(row)
    if task == 'gaussian1d_smoke':
        return dict(first_confirmed=s.get('first_confirmed_step'), ks=m.get('cdf_ks'))
    if task == 'gaussian1d_stability':
        return dict(stationary=f"{s.get('stationary_passes')}/{s.get('stationary_checks')}",
                    shift=f"{s.get('shift_hold_passes')}/{s.get('shift_hold_checks')}",
                    reacquisition=s.get('reacquisition_status'), ks=m.get('cdf_ks'), std_ratio=m.get('std_ratio'),
                    mean_error_sigma=m.get('mean_error_sigma'))
    if task in ('grid100', 'rotated100', 'staggered100'):
        return {k: m.get(k) for k in ('precision', 'center_rms_sigma', 'cov_trace_bias', 'radial_ks', 'mass_tv')}
    if task.startswith('img_'):
        return dict(modes=m.get('modes'), hq=m.get('hq'), tv=m.get('distribution_tv'), suffix=conv(row).get('passing_suffix'))
    if task == 'mode_hold':
        return dict(modes=m.get('modes'), hq=m.get('hq'), suffix=conv(row).get('passing_suffix'))
    keys = ('component_covariance_error', 'min_mass_ratio', 'mass_tv', 'sw1_normalized', 'component_min_eigen_ratio')
    return {**{k: m.get(k) for k in keys}, 'suffix': conv(row).get('passing_suffix'),
            'confirmed': conv(row).get('confirmed_step')}


def hydraulic(row):
    h = ((row or {}).get('evidence') or {}).get('hydraulic') or (row or {}).get('hydraulic')
    if not h:
        return None
    s, n = h['summary'], max(1, h['summary']['updates'])
    return dict(updates=s['updates'], limited=s['limited'] / n, rejected=s['rejected'],
                mean_scale=s['scale_sum'] / n, mean_radius=s['radius_sum'] / n,
                mean_proposed_rms=s['proposed_rms_sum'] / n, mean_accepted_rms=s['accepted_rms_sum'] / n,
                gap_wider=s['gap_wider'] / n, shared_fraction=s['shared_fraction_sum'] / n)


def fmt(value):
    if isinstance(value, float):
        return f'{value:.4g}'
    return '-' if value is None else str(value)


def main():
    result, control = rows()
    missing = {arm: [t for t in TASKS if t not in result[arm]] for arm in ARMS}
    cells = {}
    for task in TASKS:
        cells[task] = {arm: dict(gate=(result[arm].get(task) or {}).get('gate_status', 'NOT_RUN'),
                                 metrics=key_metrics(task, result[arm].get(task)),
                                 hydraulic=hydraulic(result[arm].get(task)),
                                 wall_seconds=((result[arm].get(task) or {}).get('cost') or {}).get('wall_seconds'),
                                 attempt_id=(result[arm].get(task) or {}).get('attempt_id'),
                                 source_digest=(result[arm].get(task) or {}).get('source_digest'))
                       for arm in ('winner', *ARMS)}
    passes = {arm: sum(cells[t][arm]['gate'] == 'PASS' for t in TASKS) for arm in ('winner', *ARMS)}
    winner_passes = [t for t in TASKS if cells[t]['winner']['gate'] == 'PASS']
    lost = {arm: [t for t in winner_passes if cells[t][arm]['gate'] != 'PASS'] for arm in ARMS}
    gained = {arm: [t for t in TASKS if cells[t][arm]['gate'] == 'PASS' and cells[t]['winner']['gate'] != 'PASS'] for arm in ARMS}
    out = dict(schema_version=1, scope='research_diagnostic', qualification_claim=False,
        control=dict(candidate_id=CONTROL, source_commit=control['source_commit'], source_digest=control['source_digest'],
                     reused='archived ordinary results; not rerun'),
        arms={arm: cid for arm, cid in ARMS.items()}, protocol_seed=0, measured_tasks=list(TASKS),
        not_consumed_caller_owned=list(NOT_CONSUMED), missing=missing, passes=passes, lost_winner_passes=lost,
        gained_passes=gained, cells=cells,
        paid_seconds={arm: sum((r.get('cost') or {}).get('wall_seconds') or 0 for r in result[arm].values()) for arm in ARMS},
        archive=str(ARCHIVE))
    (OUT / 'results.json').write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    lines = ['# Hydraulic leaderboard (current)', '',
             'The one current leaderboard for the hydraulic goal. The winner is the archived ordinary direction-blend arm (source d378734f). '
             'Arms ran on the 17 public-GANTrainer cells of revision 8, at seed 0, on CUDA A6000. Research diagnostic; no qualification.', '',
             f'| Task | {" | ".join(LABELS[a] for a in ("winner", *ARMS))} |', '| --- | --- | --- | --- |']
    for task in TASKS:
        row = [task]
        for arm in ('winner', *ARMS):
            c = cells[task][arm]
            row.append(f"**{c['gate']}** " + ', '.join(f'{k} {fmt(v)}' for k, v in c['metrics'].items()))
        lines.append('| ' + ' | '.join(row) + ' |')
    lines.append(f"| **Passes (17)** | {' | '.join(str(passes[a]) for a in ('winner', *ARMS))} |")
    lines += ['', 'Caller-owned cells (not consumed by the bound; winner results unchanged): ' + ', '.join(NOT_CONSUMED) + '.']
    (OUT / 'LEADERBOARD.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(passes=passes, lost=lost, gained=gained, missing=missing)))


def media():
    """Actual-training GIFs from saved observations only (shared saved renderer)."""
    import importlib.util
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location('hydraulic_saved_evidence', ROOT / 'reports/forge/bcap-develop-integration/publish.py')
    saved = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(saved)
    saved.ROLES = tuple(ARMS)
    saved.required_questions = lambda _root: [dict(task=t, qualification_tier=1) for t in TASKS]
    def scopes(_options):
        state = json.loads((ARCHIVE / 'queue/queue/state.json').read_text())
        requests = json.loads((ARCHIVE / 'progress.json').read_text())['requests']
        jobs = [j for j in state['jobs'].values() if set(j.get('subscribers', [])) & set(requests.values())]
        return [dict(scope='research_diagnostic', queue=ARCHIVE / 'queue', state=state, requests=requests,
                     submissions={r: state['submissions'][i] for r, i in requests.items()}, jobs=jobs,
                     active=[j['definition']['task_id'] for j in jobs if j['status'] in saved.ACTIVE],
                     diagnostic_blockers={})]
    saved.scopes = scopes
    collection = saved.collect(SimpleNamespace(repository=ROOT, allow_partial=False))
    renderer = saved._saved_renderer(ROOT)
    index = []
    for entry in collection['final']:
        item = entry['item']
        if item['gate_status'] not in {'PASS', 'FAIL'}:
            continue
        gif = OUT / 'media' / f"{item['role']}-{item['task_id']}.gif"
        with renderer.forbid_live_execution():
            receipt = saved.render_saved(entry, gif, renderer)
        index.append(dict(receipt, role=item['role'], task_id=item['task_id'], gif=str(gif.relative_to(OUT))))
        print(json.dumps(dict(event='saved_gif', role=item['role'], task=item['task_id'])), flush=True)
    (OUT / 'media/index.json').write_text(json.dumps(dict(schema_version=1, media=index, optimizer_updates_added=0,
                                                          sampling_draws_added=0), indent=1, default=str) + '\n')


if __name__ == '__main__':
    main()
    if '--media' in sys.argv:
        media()
