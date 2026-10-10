"""Declare, freeze, execute, and publish bounded original two-pole diagnostics."""
from copy import deepcopy
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

import torch

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash, utc_now
from experiments.forge.planning import resolve_idea
from experiments.forge.queue import Queue, drain

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ARCHIVE = Path('/mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next/stability')
PREFIX = 'bcap-tier1-stability'
ARMS = {
    'incumbent': ('none', 0., 0.),
    'projection': ('direction_blend', 0., 0.),
    'transport': ('none', 1., 1.),
    'projection-global': ('direction_blend', 1., 0.),
    'projection-local': ('direction_blend', 0., 1.),
    'combined': ('direction_blend', 1., 1.),
}


def declare():
    view = deepcopy(read_json(ROOT/'configs/forge/views/discriminator_stability.json'))
    view.update(id=PREFIX+'-diagnostic-v1', revision=1, evidence_scope='research_diagnostic',
        policy_change_reason='Six fixed trainer ablations on the original two-pole question; no ordinary qualification.',
        assignments=[dict(task='two_pole', qualification_tier=1, importance='diagnostic', order=0)])
    atomic_json(ROOT/f'configs/forge/views/{view["id"]}.json', view)
    base = read_json(ROOT/'configs/forge/ideas/bcap-develop-integration-winner-v1.json')
    study_base = read_json(ROOT/'configs/forge/studies/bcap-develop-integration-winner-study-v1.json')
    for role, (projection, global_weight, local_weight) in ARMS.items():
        card = deepcopy(base)
        card.update(id=f'{PREFIX}-{role}-v1', guide='reports/forge/bcap-tier1-stability/README.md',
            mechanism_class='structural', changed_factors=[f'constraint_geometry_mode={projection}; kinetic_transport_weight={global_weight}; kinetic_transport_local_weight={local_weight}'],
            mechanism_rationale='Matched component ablation of the incumbent and measured combined Tier1 regression; no task-specific settings.')
        card['recipe_overrides'].update(constraint_geometry_mode=projection,
            kinetic_transport_weight=global_weight, kinetic_transport_local_weight=local_weight,
            kinetic_transport_projections=32)
        atomic_json(ROOT/f'configs/forge/ideas/{card["id"]}.json', card)
        study = deepcopy(study_base)
        study.update(id=f'{PREFIX}-{role}-study-v1', candidate=card['id'], status='ready',
            control=dict(candidate_id=f'{PREFIX}-{"combined" if role == "incumbent" else "incumbent"}-v1', task_map={}),
            hypothesis='Isolate projection, global quantile matching, and local kernel matching as causes of the observed critic-cap terminal-suffix regression, preserving movement and the original full sustained gate.',
            competing_explanation='Projection could affect a proposed displacement without a recorded conflict, or local transport rather than global transport could drive critic excursions; the combined comparison alone cannot distinguish these.',
            prior_evidence=[dict(path='reports/forge/bcap-tier1-stability/existing-evidence.json', selector=[],
                identity=dict(scientific_source_digest='c68be4ae40db26959c1b2876ed33ec044aba9cf08cab84be71ed8a0264a2eb17',
                    incumbent_attempt='f80ae62046f14be8b0ad459cfbd1c9ee', combined_attempt='aa76d8f268014f84a7a553e0b47482b0'), use='motivation_only')],
            prediction=dict(task_id='two_pole', metric='mean_abs', op='>=', threshold=.3, phase='final'),
            falsifier=dict(task_id='two_pole', metric='mean_abs', op='<', threshold=.3, phase='final'))
        study['scope'].update(view=view['id'], through_tier=1)
        study['campaign'].update(id=PREFIX+'-ablations-v1', budget_seconds=1800, candidate_budget_seconds=300)
        atomic_json(ROOT/f'configs/forge/studies/{study["id"]}.json', study)
    plans = {}
    for role in ARMS:
        request = resolve_idea(ROOT, f'{PREFIX}-{role}-v1', study=f'{PREFIX}-{role}-study-v1',
            queue_root=ARCHIVE/'queue', freeze_source=False)
        assert request['study_review']['status'] == 'READY', request['study_review']
        assert not request['preflight_blockers'], request['preflight_blockers']
        assert len(request['jobs']) == 1 and request['jobs'][0]['budget_seconds'] == 300
        plans[role] = dict(admission='READY', source_digest=request['source']['digest'],
            candidate_revision=request['candidate_revision'], task_hash=stable_hash(request['tasks']['two_pole']))
    assert len({p['source_digest'] for p in plans.values()}) == 1
    atomic_json(OUT/'preregistration.json', dict(schema_version=1, qualification_input=False,
        source_base='79f7ddb512f0bc1e1457257e85c97adbe6d11bd1', evidence_scope='research_diagnostic',
        arms=ARMS, plans=plans, seed=0, full_reservation_seconds=1800,
        repair_screens=dict(maximum_arms=2, total_reservation_seconds=600, status='not_admitted'),
        exact_task='two_pole', steps=80, observations=24, minimum_stable_checks=5,
        gates=[['mean_abs','>=',.3],['grad_med','<=',1.]], timeout_seconds=300,
        matched_conditions=['stored critic and zero-particle fixture', 'architecture', 'actual fixed target panel',
            'prior', 'sampling', 'constructor/data/noise/evaluation named streams', 'schedule horizon', 'updates', 'cadence', 'all original gates'],
        causal_predictions=['Projection-only equals incumbent actual models/particles and all metrics when conflict count is zero.',
            'Transport-only equals combined actual models/particles and all metrics when conflict count is zero.',
            'Global-only versus local-only isolates which finite signal creates persistent critic-cap excursions.'],
        causal_falsifiers=['Any paired state/curve discrepancy despite zero conflicts falsifies projection inactivity.',
            'Neither separated transport signal reproducing the combined excursion supports interaction rather than a single-factor explanation.'],
        stopping_rule='Complete six original-gate arms once. Analyze before declaring at most two global repairs; distinguish structural from constant changes. No seeds, full suite, Tier2, default adoption, or merge.'))
    print(json.dumps(dict(event='declared', plans=plans), sort_keys=True), flush=True)


def run():
    assert not subprocess.check_output(['git','status','--porcelain'], cwd=ROOT).strip(), 'Freeze committed clean source'
    queue = Queue(ARCHIVE/'queue', report_root=ROOT/'reports/forge', on_completion=None)
    progress = dict(phase='submitting', requests={}, source_digest=None, source_commit=None)
    for role in ARMS:
        request = resolve_idea(ROOT, f'{PREFIX}-{role}-v1', study=f'{PREFIX}-{role}-study-v1',
            queue_root=ARCHIVE/'queue', freeze_source=True)
        assert request['study_review']['status']=='READY' and not request['preflight_blockers']
        assert progress['source_digest'] in (None,request['source']['digest'])
        progress.update(source_digest=request['source']['digest'], source_commit=request['source']['origin_commit'])
        receipt = queue.submit(request, request['study']['campaign'])
        progress['requests'][role] = receipt['request']['request_id']
        atomic_json(ARCHIVE/'progress.json',progress)
        print(json.dumps(dict(event='submitted', role=role, request=progress['requests'][role],
            source_commit=progress['source_commit'], source_digest=progress['source_digest'], time=utc_now())),flush=True)
    progress['phase']='running'
    atomic_json(ARCHIVE/'progress.json',progress)
    drain(queue,['0','1'],workers_per_gpu=1,allow_sharing=True,watch=False,campaign=PREFIX+'-ablations-v1')
    progress.update(phase='complete', time=utc_now())
    atomic_json(ARCHIVE/'progress.json',progress)
    print(json.dumps(dict(event='complete',time=utc_now())),flush=True)


def publish():
    spec=importlib.util.spec_from_file_location('integration_publication',ROOT/'reports/forge/bcap-develop-integration/publish.py')
    publication=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publication)
    state,progress=read_json(ARCHIVE/'queue/queue/state.json'),read_json(ARCHIVE/'progress.json')
    assert progress['phase']=='complete'
    roles={rid:role for role,rid in progress['requests'].items()}
    rows=[]; initial=[]
    for job in state['jobs'].values():
        owner=[roles[rid] for rid in job['subscribers'] if rid in roles]
        if not owner: continue
        assert len(owner)==1 and job['status']=='terminal' and job['result']
        aid=job['result']['attempt_id']; attempt=publication.certified_attempt(ROOT,aid)
        row=attempt['result']['task_results'][0]; assert row['task_id']=='two_pole'
        saved,proof=publication.checkpoint(row)
        stats=publication.mechanism_stats(saved)
        curve=row['evidence']['observations']; flags=[p['mean_abs']>=.3 and p['grad_med']<=1. for p in curve]
        suffix=0
        for flag in reversed(flags):
            if not flag: break
            suffix+=1
        target=OUT/'media'/f'{owner[0]}.gif'
        entry=dict(task=attempt['request']['tasks']['two_pole'],row=row,attempt=attempt,
            request=attempt['request'],saved=saved)
        # The unchanged existing renderer consumes certified retained outputs only.
        media=publication.render_saved(entry,target,publication._saved_renderer(ROOT))
        item=dict(role=owner[0], gate_status=row['gate_status'], metrics=row['metrics'],
            passing_checks=sum(flags), terminal_suffix=suffix, observations=len(curve),
            worst_grad_med=max(p['grad_med'] for p in curve), grad_cap_violations=sum(p['grad_med']>1 for p in curve),
            final_five=curve[-5:], mechanism_stats=stats, guards=row['evidence']['guards'],
            attempt_id=aid, result_hash=attempt['certificate']['result_hash'], checkpoint=proof,
            source=attempt['request']['source'], cost=row['cost'], media=str(target.relative_to(ROOT)),
            media_sha256=file_hash(target))
        rows.append(item)
        initial.append(saved)
    assert len(rows)==6
    rows.sort(key=lambda r:list(ARMS).index(r['role']))
    # Keep full stream/tensor state outside Git; compact provenance stays here.
    for item in rows:
        item['source']={key:item['source'][key] for key in ('origin_commit','digest')}
    paid=sum(c['seconds'] for c in state['charges'])
    atomic_json(OUT/'results.json',dict(schema_version=1,qualification_input=False,
        evidence_scope='research_diagnostic',rows=rows,paid_seconds=paid,paid_attempts=len(state['charges']),
        execution_retries=sum(max(0,len(j['attempts'])-1) for j in state['jobs'].values()),
        full_reservation_seconds=1800,archive=str(ARCHIVE),requests=progress['requests']))
    print(json.dumps(dict(event='published',paid_seconds=paid,
        outcomes=[{k:r[k] for k in ('role','gate_status','metrics','passing_checks','terminal_suffix','worst_grad_med')} for r in rows])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('declare','run','publish'))
    globals()[parser.parse_args().action]()
