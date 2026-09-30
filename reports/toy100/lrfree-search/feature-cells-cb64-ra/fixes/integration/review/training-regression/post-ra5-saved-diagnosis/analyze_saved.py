"""CPU-only read-only RA5 checkpoint diagnosis; no proposals or training."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PREV=Path('/ml2/hypergan/gan-attempts/scaling-portability-20260929/validation')
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--package-root',type=Path,required=True)
parser.add_argument('--source-ready',type=Path,required=True)
args=parser.parse_args()
PACKAGE=args.package_root.resolve()
sys.path.insert(0,str(PACKAGE));sys.path.insert(0,str(PREV))
from particlegan import feature_cells as module
from models_metrics import oracle_centres
sys.path.insert(0,str(HERE.parent/'post-ra4-quality'))
from measure_saved_utils import tensor_state_hash
if (args.output/'receipt.json').exists():raise SystemExit('existing receipt; choose a new output')
paths=sorted(args.run_dir.glob('checkpoint-*.pt'))
if not any(p.name!='checkpoint-0000.pt' for p in paths):raise SystemExit('no sealed nonzero checkpoint available')
args.output.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
centres=oracle_centres();NMODES=len(centres)


def forward(points,weights,head=False):
    chunks=[]
    for x in points.split(256):
        for index in (0,2):x=F.leaky_relu(F.linear(x,weights[f'{index}.weight'],weights[f'{index}.bias']),.2)
        chunks.append(x if head else F.linear(x,weights['4.weight'],weights['4.bias']))
    return torch.cat(chunks)


def oracle(points):
    distance,mode=torch.cdist(points,centres).min(1)
    return distance,mode,distance<=.09


def score(points):
    distance,mode,accepted=oracle(points)
    counts=torch.bincount(mode[accepted],minlength=NMODES)
    mass=counts.double()/len(points);precision=float(accepted.float().mean());covered=mass>=.01
    return dict(precision=precision,coverage=int(covered.sum()),
        mass_tv=float((mass-.04).abs().sum()/2+(1-precision)/2),counts=counts.tolist(),
        covered_modes=covered.nonzero().flatten().tolist(),missing_modes=(~covered).nonzero().flatten().tolist(),
        unsupported_rows=int((~accepted).sum()),nearest_mode_counts=torch.bincount(mode,minlength=NMODES).tolist(),
        mean_nearest_centre_distance=float(distance.mean()))


def metric_prefix(run,label):
    path=run/'metrics.jsonl'
    if not path.exists():return {},None
    raw=path.read_bytes();complete=raw[:raw.rfind(b'\n')+1]
    records=[json.loads(line) for line in complete.splitlines() if line]
    capture=args.output/(label+'-metrics-captured.jsonl');capture.write_bytes(complete)
    return {r['step']:r for r in records},dict(original_path=str(path),captured_path=str(capture),
        captured_sha256=sha(capture),captured_bytes=len(complete),captured_complete_rows=len(records),
        source_live_file_may_append=True)


metrics,captured=metric_prefix(args.run_dir,'candidate')
reference_runs=dict(RA4=ROOT/'validation-ra4/learned/training/toy/CB64-RA4',
    E22=PREV/'runs/toy/E22')
reference_metrics={};prefixes=[captured] if captured else []
for label,path in reference_runs.items():
    reference_metrics[label],prefix=metric_prefix(path,label.lower())
    if prefix:prefixes.append(prefix)
sources=[*sorted(PACKAGE.rglob('*.py')),Path(__file__),HERE/'PROTOCOL.md',PREV/'models_metrics.py',
    HERE.parent/'post-ra4-quality/measure_saved_utils.py',args.source_ready]
source_before={str(p):sha(p) for p in sources}
ready=json.loads(args.source_ready.read_text())
assert {str(p.relative_to(PACKAGE/'particlegan')):sha(p) for p in PACKAGE.rglob('*.py')}==ready['package_source_sha256']
checkpoint_before={str(p):sha(p) for p in paths}
reference_paths=[];reference_input_sha={};rng=torch.get_rng_state().clone();records=[]
last_cohort_records=[]
for path in paths:
    state=torch.load(path,map_location='cpu',weights_only=False)['trainer'];state_hash=tensor_state_hash(state)
    step=state['completed_steps'];weights=state['models'];bd=state['birth_death']
    raw=forward(weights['prior']['z'],weights['G']);average=forward(weights['ema_prior']['z'],weights['ema_G'])
    fast=score(raw);ema=score(average)
    tester=state['lr_settle'][0][1];serve='EMA' if tester['last_decisive']==-1 else 'FAST'
    population=dict(last_decisive=tester['last_decisive'],s=tester['s'],b=tester['b'],
        population_active=tester.get('population_active'),stationary_participating_rows=None,
        required_rows=len(raw)-int(.05*len(raw)),stationary_undo_s=tester.get('stationary_undo_s'),
        population_expiries=tester['counts'].get('population_expiries',0),
        population_coverage_rejections=tester['counts'].get('population_coverage_rejections',0),
        last_population=tester.get('last_population'))
    if tester.get('stationary_rows') is not None:
        population['stationary_participating_rows']=int(tester['stationary_rows'].sum())
    latest=bd['last'];born=latest.get('novel_birth_children',torch.empty(0,dtype=torch.long))
    born=torch.as_tensor(born,dtype=torch.long)
    counters=bd['counters']
    event=dict(reaction_step=latest.get('step'),snapshot=latest.get('snapshot'),
        age_updates=None if latest.get('step') is None else step-latest['step'],
        latest_births=latest.get('ordinary_novel_birth_moves',0),latest_attempts=latest.get('novel_birth_attempts',0),
        cumulative_births=counters.get('novel_birth_moves',0),cumulative_attempts=counters.get('novel_birth_attempts',0),
        cumulative_reactions=counters.get('evals',0),ordinary_moves=latest.get('ordinary_moves',0),
        children=born.tolist(),source_seed_rows=latest.get('novel_birth_seed_rows',[]),
        historical_gpu_target_cells=latest.get('novel_birth_target_cells',[]),
        historical_target_cells_not_reused_in_cpu_refit=True)
    diagnostic=None;birth_rows=[]
    if bd['reservoir'] is not None and bd['fill']>=len(raw):
        real_raw=bd['reservoir'];r=forward(real_raw,weights['D'],head=True).double()
        q=forward(raw,weights['D'],head=True).double();eq=forward(average,weights['D'],head=True).double()
        snap=module.FeatureCellSnapshot.fit(r,generator=torch.Generator().set_state(state['cpu_rng']),
            cells=bd['settings']['cells'],rank=bd['settings']['rank'],chunk=bd['settings']['chunk'])
        flags,pvalues,_=snap.support(q);ef,ep,_=snap.support(eq)
        ids,_=snap.assign(q);eid,_=snap.assign(eq);rid,_=snap.assign(r)
        categories=snap.count_categories(q);ec=snap.count_categories(eq)
        target=snap._mass_targets(len(q));groups=snap._mass_topology()
        _,rm,ra=oracle(real_raw);_,qm,qa=oracle(raw);_,em,ea=oracle(average)
        contingency=torch.bincount(rid*NMODES+rm,minlength=snap.cells*NMODES).reshape(snap.cells,NMODES)
        cell_mode=contingency.argmax(1)
        purity=contingency.max(1).values.double()/contingency.sum(1).clamp_min(1)
        clean=torch.bincount(ids[~flags],minlength=snap.cells)
        eligible=(~flags)&(pvalues>.05);inside=categories.remainder(2)==0
        pool=torch.bincount(ids[eligible&inside],minlength=snap.cells)
        vacancy=(target-clean).clamp_min(0)
        group_vacancy=(snap._group_counts(target)-snap._group_counts(clean)).clamp_min(0)
        cell_capacity=torch.minimum(vacancy,pool)
        per_mode=[]
        for mode in range(NMODES):
            cells=cell_mode==mode;raw_rows=qm==mode
            annotated_capacity=torch.minimum(snap._group_counts(cell_capacity*cells),group_vacancy).sum()
            per_mode.append(dict(mode=mode,fast_raw_supported=fast['counts'][mode],ema_raw_supported=ema['counts'][mode],
                reference_raw_rows=int((rm==mode).sum()),reference_supported_rows=int(((rm==mode)&ra).sum()),
                nearest_fast_rows=int(raw_rows.sum()),raw_supported_pQ_rejections=int((raw_rows&qa&(pvalues<=.05)).sum()),
                raw_supported_inside_rejections=int((raw_rows&qa&eligible&~inside).sum()),
                raw_supported_eligible_inside=int((raw_rows&qa&eligible&inside).sum()),
                eligible_inside_by_reference_cell_mode=int(pool[cells].sum()),
                real_target_by_reference_cell_mode=int(target[cells].sum()),
                supported_by_reference_cell_mode=int(clean[cells].sum()),
                cell_vacancy_by_reference_cell_mode=int(vacancy[cells].sum()),
                physical_unique_parent_capacity_upper_bound=int(annotated_capacity),
                empty_inside_parent_cells=int((cells&(pool==0)&(vacancy>0)&(group_vacancy[groups]>0)).sum()),
                cell_annotation_scope='majority raw-reference mode; labels are annotation only'))
        for row in born.tolist():
            birth_rows.append(dict(row=row,reaction_step=event['reaction_step'],age_updates=event['age_updates'],
                fast_nearest_mode=int(qm[row]),fast_raw_supported=bool(qa[row]),fast_pvalue=float(pvalues[row]),
                fast_cpu_inside=bool(inside[row]),fast_cpu_cell=int(ids[row]),
                ema_nearest_mode=int(em[row]),ema_raw_supported=bool(ea[row]),ema_pvalue=float(ep[row]),
                ema_cpu_inside=bool(ec[row]%2==0),ema_cpu_cell=int(eid[row]),
                birth_time_support_not_observed=True))
        diagnostic=dict(scope='saved current FIFO/head CPU refit; not historical GPU geometry or reserved postplanning ledger',
            flags=int(flags.sum()),eligible_pQ=int(eligible.sum()),eligible_inside=int((eligible&inside).sum()),
            ema_eligible_pQ=int((ep>.05).sum()),ema_eligible_inside=int(((ep>.05)&(ec.remainder(2)==0)).sum()),
            groups=snap.mass_groups,cell_reference_purity=purity.tolist(),
            cell_vacancies=int(vacancy.sum()),group_vacancies=int(group_vacancy.sum()),
            total_unique_copy_capacity_upper_bound=int(torch.minimum(snap._group_counts(cell_capacity),group_vacancy).sum()),
            modes=per_mode)
    saved_metric=metrics.get(step,{}).get('metrics')
    served=ema if serve=='EMA' else fast
    counts_match=(None if saved_metric is None else served['counts']==[round(x*len(raw)) for x in saved_metric['clean_particle_centres']['supported_mass']])
    references={}
    for label,run in reference_runs.items():
        rp=run/path.name
        if not rp.exists():continue
        reference_paths.append(rp)
        reference_input_sha[str(rp)]=sha(rp)
        rs=torch.load(rp,map_location='cpu',weights_only=False)['trainer'];wm=rs['models']
        references[label]=dict(fast=score(forward(wm['prior']['z'],wm['G'])),
            ema=score(forward(wm['ema_prior']['z'],wm['ema_G'])),
            saved_metrics=reference_metrics[label].get(step,{}).get('metrics'),checkpoint_sha256=sha(rp))
    record=dict(step=step,checkpoint_sha256=checkpoint_before[str(path)],fast=fast,ema=ema,reported_serving_table=serve,
        saved_metrics=saved_metric,served_clean_counts_match_saved=counts_match,population=population,
        birth_event=event,current_birth_rows=birth_rows,parent_supply=diagnostic,references=references,
        unchanged_checkpoint_tensors=tensor_state_hash(state)==state_hash)
    if records:
        previous=records[-1];dn=event['cumulative_reactions']-previous['birth_event']['cumulative_reactions']
        db=event['cumulative_births']-previous['birth_event']['cumulative_births']
        record['since_previous']=dict(updates=step-previous['step'],reactions=dn,births=db,
            births_per_reaction=None if dn==0 else db/dn,
            fast_lost_modes=sorted(set(previous['fast']['covered_modes'])-set(fast['covered_modes'])),
            fast_restored_modes=sorted(set(fast['covered_modes'])-set(previous['fast']['covered_modes'])),
            ema_lost_modes=sorted(set(previous['ema']['covered_modes'])-set(ema['covered_modes'])),
            ema_restored_modes=sorted(set(ema['covered_modes'])-set(previous['ema']['covered_modes'])),
            population_expiries=population['population_expiries']-previous['population']['population_expiries'])
        for cohort in previous['current_birth_rows']:
            row=cohort['row'];d,m,a=oracle(raw[row:row+1]);ed,em,ea=oracle(average[row:row+1])
            last_cohort_records.append(dict(first_observed_step=previous['step'],observed_step=step,row=row,
                first_fast_mode=cohort['fast_nearest_mode'],current_fast_mode=int(m[0]),current_fast_supported=bool(a[0]),
                first_ema_mode=cohort['ema_nearest_mode'],current_ema_mode=int(em[0]),current_ema_supported=bool(ea[0]),
                intervening_reactions=dn,incarnation_continuity_unverified=dn>0,
                scope='row persistence only; unlogged copy/birth overwrites prevent exact birth-survival attribution'))
    records.append(record)
    print(json.dumps(dict(event='saved_ra5_checkpoint',step=step,fast_P=fast['precision'],fast_modes=fast['coverage'],
        ema_P=ema['precision'],ema_modes=ema['coverage'],serving=serve,counts_match=counts_match,
        births=event['cumulative_births'],latest_births=event['latest_births'],population_expiries=population['population_expiries'])),flush=True)
assert reference_input_sha=={str(p):sha(p) for p in reference_paths}
assert all(r['unchanged_checkpoint_tensors'] for r in records)
assert source_before=={str(p):sha(p) for p in sources}
assert checkpoint_before=={str(p):sha(p) for p in paths} and torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
def plain(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value
receipt=dict(status='COMPLETE_READ_ONLY_SAVED_DIAGNOSIS',records=records,row_persistence=last_cohort_records,
    source_sha256=source_before,checkpoint_sha256=checkpoint_before,reference_checkpoint_sha256=reference_input_sha,
    captured_metric_prefixes=prefixes,all_sources_and_checkpoints_unchanged=True,global_rng_unchanged=True,
    cuda_initialized=False,cpu_only=True,new_seeds=0,new_training_steps=0,new_emissions=0,new_proposals=0,
    quality_verdict=None,scope='saved clean/EMA/support/population/birth diagnosis; saved metrics are quality authority',
    limits=['CPU geometry refit is not historical GPU geometry','no complete per-reaction events',
        'row persistence is not same-incarnation birth survival','per-mode parent capacity is annotated initial-state upper bound'])
(args.output/'receipt.json').write_text(json.dumps(plain(receipt),indent=2)+'\n')
lines=['# Saved RA5 checkpoint diagnosis','',
    'Read-only CPU analysis; no proposals, emissions, updates, seeds or quality verdict. Saved emitted metrics remain authoritative.',
    '', '| Update | Fast P | Fast modes | EMA P | EMA modes | Saved emitted P | Saved emitted modes | Births cumulative | Population expiries |',
    '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for r in records:
    metric=r['saved_metrics'] or {}
    lines.append(f"| {r['step']} | {r['fast']['precision']:.6f} | {r['fast']['coverage']} | {r['ema']['precision']:.6f} | {r['ema']['coverage']} | {metric.get('precision','pending')} | {metric.get('coverage','pending')} | {r['birth_event']['cumulative_births']} | {r['population']['population_expiries']} |")
lines.extend(['','Latest observed birth rows, initial per-mode parent availability/targets and matched RA4/E22 clean/saved metrics are in the receipt.',
    '', 'CPU snapshots describe current saved heads/FIFO and use recorded CPU RNG. They cannot identify historical GPU target-cell IDs. Interval birth averages and subsequent row persistence are reported; unlogged intermediate row overwrites prevent exact birth-expiry attribution.'])
(args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(event='saved_ra5_diagnosis_complete',status=receipt['status'],checkpoints=len(records),output=str(args.output))),flush=True)
