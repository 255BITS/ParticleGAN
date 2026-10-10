"""Hydraulic v1 on the selected direction-blend BCAP winner: prepare and run.

Two candidate arms (exact v1 travel bound; gap-adaptive radius) on every task
of the ordinary revision-8 suite that executes through the public GANTrainer.
The winner control is the archived ordinary direction-blend arm
(bcap-default-baseline, source d378734f); it is not rerun. Caller-owned hosts
do not consume the bound and are declared unsupported before reservation.

    python reports/forge/bcap-physics/hydraulic/winner/workflow.py prepare
    python reports/forge/bcap-physics/hydraulic/winner/workflow.py run --gpus 0,1
    tail -f /mnt/ml7tb/ParticleGAN-forge/bcap-hydraulic-winner-20261010/logs/driver.log
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
ARCHIVE = Path('/mnt/ml7tb/ParticleGAN-forge/bcap-hydraulic-winner-20261010')
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import atomic_json, read_json, stable_hash, utc_now  # noqa: E402
from experiments.forge.planning import resolve_idea  # noqa: E402
from experiments.forge.studies import validate_study  # noqa: E402

VIEW_ID = 'bcap-hydraulic-winner-diagnostic-v1'
CAMPAIGN = 'bcap-hydraulic-winner-v1'
CONTROL = 'bcap-default-baseline-direction-v1'
ARMS = {'travel': 'bcap-hydraulic-winner-travel-v1', 'gap': 'bcap-hydraulic-winner-gap-v1'}
TASKS = ('gaussian1d_smoke', 'ring16_acquisition', 'gaussian1d_stability', 'mode_hold',
         'vector_two_broad', 'vector_unequal_mass', 'vector_unequal_width', 'vector_anisotropic',
         'vector_overlap', 'vector_spiral', 'img_stripes2', 'img_bars4', 'img_blobs4',
         'img_intensity2', 'grid100', 'rotated100', 'staggered100')
NOT_CONSUMED = ('two_pole', 'unused_token_hold', 'ae_gan_hold', 'five_word_joint_smoke',
                'five_word_joint_hold', 'trajectory', 'residual_student', 'unipolar',
                'cover_leftover', 'mid_scale_identity')
PER_ARM = 31620
BUDGET = 64000
OUTCOMES = dict(falsified='stop_revision', incomplete='request_missing_evidence',
                inconclusive='stop_and_readout', prediction_observed='review_saved_diagnostics')
PREDICTIONS = {
    'travel': dict(prediction=dict(task_id='grid100', metric='precision', op='>=', threshold=.48, phase='final'),
                   falsifier=dict(task_id='grid100', metric='precision', op='<', threshold=.48, phase='final'),
                   hypothesis='The exact PR360 v1 travel bound composes with direction blend (inactive on GANTrainer hosts) and reproduces its archived retention/precision signal on the selected winner: Gaussian stationary >=70/72, shifted hold >=22/24, grid100 precision >=.48.'),
    'gap': dict(prediction=dict(task_id='gaussian1d_stability', metric='cdf_ks', op='<=', threshold=.05, phase='final'),
                falsifier=dict(task_id='gaussian1d_stability', metric='cdf_ks', op='>', threshold=.05, phase='final'),
                hypothesis='A gap-adaptive radius keeps the v1 stationary bound but removes its travel-limited acquisition cost: Gaussian smoke confirms by update 500, stationary/shift retention stays >=70/72 and >=22/24, and grid100 precision stays >=.40.'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def prepare():
    view = deepcopy(read_json(ROOT / 'configs/forge/views/discriminator_stability.json'))
    require(view['revision'] == 8, 'Bind the ordinary revision-8 task contracts.')
    ordinary = {a['task'] for a in view['assignments']}
    require(set(TASKS) <= ordinary and set(NOT_CONSUMED) <= ordinary, 'Tasks must come from revision 8.')
    view.update(id=VIEW_ID, revision=1, evidence_scope='research_diagnostic',
        policy_change_reason='Hydraulic v1 on the selected direction-blend winner: every revision-8 task executed by the public GANTrainer, original gates and own checkpoint dependencies, no tier veto and no qualification credit. Caller-owned hosts do not consume the bound.',
        assignments=[dict(task=t, qualification_tier=1, importance='diagnostic', order=i) for i, t in enumerate(TASKS)])
    atomic_json(ROOT / f'configs/forge/views/{VIEW_ID}.json', view)
    for arm, cid in ARMS.items():
        study = dict(schema_version=1, id=f'{cid}-study', status='ready', candidate=cid,
            control=dict(candidate_id=CONTROL, task_map={}), max_rounds=1,
            hypothesis=PREDICTIONS[arm]['hypothesis'],
            competing_explanation='A batch-spacing speed limit slows every transport, so retention may improve only by slowing the dynamics, while acquisition deadlines, density shape and native covariance regress; the remaining Gaussian KS misses may be shape/mean drift that no scalar travel bound removes.',
            scope=dict(view=VIEW_ID, through_tier=1, execution_backend='cuda', cuda_model='NVIDIA RTX A6000'),
            campaign=dict(id=CAMPAIGN, budget_seconds=BUDGET, candidate_budget_seconds=PER_ARM),
            prior_evidence=[dict(path='reports/forge/bcap-default-baseline/results.json', selector=[],
                identity=dict(source_commit='d378734f40b09ce223a389e8f54a9783ec6a0c75'),
                use='motivation_only')],
            prediction=PREDICTIONS[arm]['prediction'], falsifier=PREDICTIONS[arm]['falsifier'],
            terminal_rules=OUTCOMES)
        validate_study(study)
        atomic_json(ROOT / f'configs/forge/studies/{study["id"]}.json', study)
    plans = {}
    for arm, cid in ARMS.items():
        request = resolve_idea(ROOT, cid, study=f'{cid}-study', queue_root=ARCHIVE / 'queue')
        require(request['study_review']['status'] == 'READY', request['study_review'])
        require(not request['preflight_blockers'], request['preflight_blockers'])
        reservation = sum(j['budget_seconds'] for j in request['jobs'])
        require(reservation == PER_ARM, f'reservation {reservation} != {PER_ARM}')
        plans[arm] = dict(candidate_id=cid, candidate_revision=request['candidate_revision'],
                          source_digest=request['source']['digest'], reservation_seconds=reservation,
                          protocol_sha256=stable_hash(request['protocol']), runtime_sha256=stable_hash(request['runtime']),
                          task_keys={m: j['compatibility_key'] for j in request['jobs'] for m in j.get('task_ids', [j['task_id']])})
    require(len({p['source_digest'] for p in plans.values()}) == 1, 'Arms need identical scientific source.')
    require(len({p['protocol_sha256'] for p in plans.values()}) == 1, 'Arms need identical protocol.')
    print(json.dumps(dict(event='prepared', plans={k: {x: v[x] for x in v if x != 'task_keys'} for k, v in plans.items()})), flush=True)
    return plans


def run(gpus, workers):
    from experiments.forge.queue import Queue, drain
    (ARCHIVE / 'logs').mkdir(parents=True, exist_ok=True)
    queue = Queue(ARCHIVE / 'queue', report_root=ROOT / 'reports/forge', on_completion=None)
    path = ARCHIVE / 'progress.json'
    if not path.exists():
        progress = dict(phase='submitting', requests={}, source_commit=head(), started_at=utc_now())
        for arm, cid in ARMS.items():
            request = resolve_idea(ROOT, cid, study=f'{cid}-study', queue_root=ARCHIVE / 'queue', freeze_source=True)
            require(not request['preflight_blockers'], request['preflight_blockers'])
            receipt = queue.submit(request, request['study']['campaign'])
            progress['requests'][arm] = receipt['request']['request_id']
            atomic_json(path, progress)
            print(json.dumps(dict(event='submitted', arm=arm, request=progress['requests'][arm], time=utc_now())), flush=True)
    progress = read_json(path)
    progress.update(phase='running', updated_at=utc_now())
    atomic_json(path, progress)
    drain(queue, gpus.split(','), workers_per_gpu=workers, allow_sharing=workers > 1, watch=False, campaign=CAMPAIGN)
    progress.update(phase='complete', updated_at=utc_now())
    atomic_json(path, progress)
    print(json.dumps(dict(event='complete', time=utc_now(), requests=progress['requests'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    options = parser.parse_args()
    prepare() if options.action == 'prepare' else run(options.gpus, options.workers_per_gpu)
