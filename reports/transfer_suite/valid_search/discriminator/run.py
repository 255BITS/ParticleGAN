"""Bounded D-architecture comparison; all formulation and non-D knobs frozen."""
import datetime,gzip,hashlib,json,shutil,sys,time
from copy import deepcopy
from pathlib import Path
sys.path.insert(0,'/ml2/hypergan/ParticleGAN-pr36-valid-d')
import torch
from benchmarks.transfer_suite import suite,vector_tasks
from benchmarks.transfer_suite.protocol import test_verdict
root=Path('/ml2/hypergan/ParticleGAN-pr36-valid-d')
out=Path('/tmp/pr36-valid-data-d-architecture')
HARD=['vector_unequal_mass','vector_unequal_width','vector_overlap']
ALL=[s['name'] for s in vector_tasks.TASKS if s['tier']=='ranking']
CARDS=[{'name':f'd{w}_l{d}_f{f}','overrides':{'d_hidden':w,'d_layers':d,'fourier':f}} for w,d,f in (
    (128,2,2),(256,2,2),(64,3,2),(128,3,2),
    (64,2,3),(128,2,3),(128,3,3),(256,3,3),
    (64,2,5),(128,2,4),(128,3,4),(64,4,3))]
manifest={s['name']:s for s in suite.manifest()['tasks']}
assert len(ALL)==6 and len(CARDS)==12
for name in ALL:
    spec=vector_tasks.resolve(manifest[name])
    assert spec['reg_arm']=='b_cap' and spec['reg_coeff']==3 and spec['reg_kappa']==1.25
    assert spec.get('loss_type','logistic')=='logistic' and spec.get('gan_mode','rp')=='rp'
    assert spec['prior_reg']==.05 and spec['lr']==.001 and spec['betas']==[0.,.99]
    assert spec['prior_lr_mult']==10 and spec['hidden']==64 and spec['layers']==2
    assert spec['particles']==256 and spec['batch']==128 and spec['d_lr_mult']==1.5
    assert spec['steps']==(1600 if name=='vector_spiral' else 1200)
    assert spec['tier']=='ranking' and spec['runner']=='vector'
policy=vector_tasks.fixed_policy('cosine')
torch.set_num_threads(1)
protocol=suite.snapshot(out)
protocol['driver_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
protocol['head']='a11c5304cde01c7fdc96e8a49a5a576b8cb8ebff'
plan={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cards':CARDS,'hard_tasks':HARD,'full_tasks':ALL,'original_specs':[manifest[n] for n in ALL],'policy':policy,'selection':'Screen all12 cards on all3 hard data toys; select top3 by sustained pass count, then lower mean final normalized shortfall, then lower mean confirmation fraction, then card order. Run their other3 data tasks. No diagnostic/stress cases or new seeds participate. Complete six-task results include reused screen episodes.','frozen':'Only D width/depth/Fourier feature count change. G, data, thresholds, original per-task budget, logistic RP, cap3/kappa1.25, prior regularization.05/noL2, LR/betas/priorLR, particles and batch all unchanged.','previous_exact_architecture':'D64x2/Fourier4 already evaluated on all3 hard tasks and failed sustained; retained as existing reference without rerun.','no_training_seed_search':True}
def write(path,value): path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
write(out/'plan.json',plan);write(out/'protocol.json',protocol)
(out/'episodes').mkdir(exist_ok=False);(out/'references').mkdir(exist_ok=False)
records=[];references=[]
# Explicitly retain previous exact baseline and F4 failures without re-training.
for name in ALL:
    path=root/f'reports/transfer_suite/study/episodes/cosine__{name}.json.gz'
    dest=out/'references'/path.name;shutil.copy2(path,dest)
    payload=json.loads(gzip.decompress(path.read_bytes()))
    references.append({'candidate':'original_d64_l2_f2','task':name,'verdict':test_verdict(manifest[name],payload['result']),'live':payload['result']['live'],'artifact':str(dest.relative_to(out)),'artifact_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'reused':True})
for name in HARD:
    path=root/f'reports/transfer_suite/solvability/vectors/screen/episodes/fourier4__{name}.json.gz'
    dest=out/'references'/path.name;shutil.copy2(path,dest)
    payload=json.loads(gzip.decompress(path.read_bytes()))
    references.append({'candidate':'existing_d64_l2_f4','task':name,'verdict':payload['verdict'],'live':payload['result']['live'],'artifact':str(dest.relative_to(out)),'artifact_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'reused':True})
for source,name in ((root/'reports/transfer_suite/study/source.tar.gz','original_cosine_source.tar.gz'),(root/'reports/transfer_suite/solvability/vectors/screen/source.tar.gz','existing_fourier4_source.tar.gz')):
    if source.exists(): shutil.copy2(source,out/'references'/name)
write(out/'references.json',references)

def render():
    lines=['# Fixed-formulation data toys: discriminator architecture search','','Only discriminator width/depth/Fourier representation changes. Seed0, CPU thread1, original data/G/settings/budgets/gates. Six valid data toys; intentionally weak architecture/data diagnostics and forced optimizer stresses are excluded. EMA never qualifies a live failure. Missing cases remain unmeasured, not passes.','','| Discriminator | Unequal mass | Unequal width | Overlap | Broad | Anisotropic | Spiral | Sustained / measured |','|---|---|---|---|---|---|---|---:|']
    for label,group in [('original_d64_l2_f2',[r for r in references if r['candidate']=='original_d64_l2_f2']),('existing_d64_l2_f4',[r for r in references if r['candidate']=='existing_d64_l2_f4'])]+[(c['name'],[r for r in records if r['candidate']['name']==c['name']]) for c in CARDS]:
        by={r['task']:r for r in group}
        def cell(task):
            if task not in by:return 'unmeasured'
            v=by[task]['verdict'];return f'{v["status"]} ({v.get("convergence",{}).get("passing_suffix",0)}/24)'
        passed=sum(r['verdict']['passed'] for r in group)
        lines.append('| '+label+' | '+' | '.join(cell(t) for t in HARD+['vector_two_broad','vector_anisotropic','vector_spiral'])+f' | {passed}/{len(group)} |')
    lines += ['','Cells show sustained verdict and final passing suffix. A complete24-point curve and at least5 final passing observations are required; final bounds alone do not qualify. Baseline/Fourier4 rows are explicitly reused historical controls. The spiral retains its original1600-step budget; other valid vector cases use1200. No training extensions.','','[Frozen plan](plan.json), [runtime/source hashes](protocol.json), [all results](index.json), [exact driver](run.py), [tailable log](progress.log).']
    (out/'README.md').write_text('\n'.join(lines)+'\n')

def episode(card,name,phase):
    spec=deepcopy(manifest[name]);spec.update(card['overrides'])
    assert {k for k in spec if spec[k]!=manifest[name].get(k)}<=set(card['overrides'])
    print(f'START {phase} {card["name"]} {name}',flush=True)
    suite.verify_source(protocol)
    result=suite.run_episode(spec,policy,fixed=True)
    verdict=test_verdict(spec,result)
    payload={'candidate':card,'task':name,'phase':phase,'original_spec':manifest[name],'spec':spec,'policy':policy,'source_sha256':protocol['source_sha256'],'result':result,'verdict':verdict}
    raw=(json.dumps(payload,sort_keys=True,allow_nan=False)+'\n').encode()
    artifact=f'episodes/{card["name"]}__{name}.json.gz'
    (out/artifact).write_bytes(gzip.compress(raw,mtime=0))
    row={'candidate':card,'task':name,'phase':phase,'spec':spec,'policy':policy,'verdict':verdict,'live':result.get('live'),'ema':result.get('ema'),'seconds':result['seconds'],'artifact':artifact,'uncompressed_sha256':hashlib.sha256(raw).hexdigest()}
    records.append(row);write(out/'index.json',{'records':records});render()
    failures=[f'{m["metric"]}={m.get("value")}' for m in verdict.get('metrics',[]) if m['status']!='PASS']
    print(f'DONE {card["name"]} {name}: {verdict["status"]}, tail={verdict.get("convergence",{}).get("passing_suffix",0)}/24, seconds={result["seconds"]:.2f}, failures={"; ".join(failures) or "none"}',flush=True)
    if result.get('error'):print(result['error'],flush=True)
render()
for card in CARDS:
    for name in HARD:episode(card,name,'screen')
def rank(card):
    rows=[r for r in records if r['candidate']['name']==card['name']]
    return (-sum(r['verdict']['passed'] for r in rows),sum(r['verdict']['shortfall'] for r in rows)/3,sum(r['verdict']['confirmation_fraction'] for r in rows)/3,CARDS.index(card))
selected=sorted(CARDS,key=rank)[:3]
write(out/'selected.json',{'selection':plan['selection'],'candidates':selected,'screen_key':{c['name']:rank(c) for c in CARDS}})
print('VALIDATION_SELECTED '+','.join(c['name'] for c in selected),flush=True)
for card in selected:
    for name in ALL:
        if name not in HARD:episode(card,name,'all_six_validation')
suite.verify_source(protocol)
write(out/'completed.json',{'episodes':len(records),'new_cards':len(CARDS),'validated_candidates':[c['name'] for c in selected],'errors':sum(r['verdict']['status']=='ERROR' for r in records),'all_complete':all(r['verdict'].get('convergence',{}).get('complete',False) for r in records)})
print('COMPLETE',flush=True)
