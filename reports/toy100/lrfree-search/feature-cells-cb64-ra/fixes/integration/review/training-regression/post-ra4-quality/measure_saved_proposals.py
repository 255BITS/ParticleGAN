"""CPU-only saved-state diagnosis and the one predeclared anchor-birth policy."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
from pathlib import Path
from types import ModuleType
import hashlib,importlib,json,time
import torch
import torch.nn.functional as F
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
from anchor_birth import propose_inaccessible_anchors
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PREV=Path('/ml2/hypergan/gan-attempts/scaling-portability-20260929/validation')
PACKAGE=ROOT/'pkg-CB64-RA4/particlegan'
holder=ModuleType('anchor_measure_frozen');holder.__path__=[str(PACKAGE)];sys.modules[holder.__name__]=holder
feature=importlib.import_module(holder.__name__+'.feature_cells')
sys.path.insert(0,str(PREV));from models_metrics import oracle_centres
centres=oracle_centres()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=sorted(PACKAGE.glob('*.py'))+[Path(__file__),HERE/'anchor_birth.py',HERE/'PROTOCOL.md',PREV/'models_metrics.py']
checkpoints=[ROOT/f'validation-ra4/learned/training/toy/CB64-RA4/checkpoint-{step:04d}.pt' for step in (1250,2000)]
paths+=checkpoints
before={str(p):sha(p) for p in paths}
global_rng=torch.get_rng_state().clone()

def tensor_state_hash(obj):
    h=hashlib.sha256()
    def walk(v):
        if isinstance(v,torch.Tensor):h.update(str((v.dtype,v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());walk(v[k])
        elif isinstance(v,(list,tuple)):
            for x in v:walk(x)
        else:h.update(repr(v).encode())
    walk(obj);return h.hexdigest()

def forward(points,w,head=False):
    out=[]
    for x in points.split(256):
        for index in (0,2):x=F.leaky_relu(F.linear(x,w[f'{index}.weight'],w[f'{index}.bias']),.2)
        out.append(x if head else F.linear(x,w['4.weight'],w['4.bias']))
    return torch.cat(out)

def plain(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu().tolist()
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    return v

def oracle(points):
    d,m=torch.cdist(points,centres).min(1);return d,m,d<=.09

records=[]
for checkpoint in checkpoints:
    s=torch.load(checkpoint,map_location='cpu',weights_only=False)['trainer'];step=s['completed_steps']
    original=tensor_state_hash(s);models=s['models'];z=models['prior']['z'];ez=models['ema_prior']['z']
    callback=lambda points:forward(forward(points,models['G']),models['D'],head=True)
    average=lambda points:forward(forward(points,models['ema_G']),models['D'],head=True)
    with torch.no_grad():
        raw=forward(z,models['G']);served=forward(ez,models['ema_G'])
        q=callback(z).double();rraw=s['birth_death']['reservoir'];r=forward(rraw,models['D'],head=True).double()
        snap=feature.FeatureCellSnapshot.fit(r,generator=torch.Generator().set_state(s['cpu_rng']),cells=64,rank=8,chunk=256)
        flags,pvalues,scores=snap.support(q);qid,_=snap.assign(q);rid,_=snap.assign(r);categories=snap.count_categories(q)
        rd,rm,ra=oracle(rraw);d,m,a=oracle(raw);ed,em,ea=oracle(served)
        contingency=torch.bincount(rid*25+rm,minlength=64*25).reshape(64,25);cell_mode=contingency.argmax(1)
        mode_rows=[]
        for mode in sorted(set(m[a].unique().tolist())|set(range(25))):
            accepted=(m==mode)&a
            if int(accepted.sum())>=11:continue
            nearest=(m==mode);cells=cell_mode==mode
            eligible=(~flags)&(pvalues>.05);inside=categories.remainder(2)==0
            def minimum(values,mask):return float(values[mask].min()) if bool(mask.any()) else None
            mode_rows.append(dict(mode=mode,training_accepted=int(accepted.sum()),training_nearest=int(nearest.sum()),
                served_accepted=int(((em==mode)&ea).sum()),
                accepted_rejected_pQ=int((accepted&(pvalues<=.05)).sum()),
                accepted_pQ_pass_inside_fail=int((accepted&eligible&~inside).sum()),
                accepted_pQ_inside_pass=int((accepted&eligible&inside).sum()),
                minimum_nearest_raw_distance=minimum(d,nearest),minimum_nearest_score=minimum(scores,nearest),
                maximum_nearest_pvalue=float(pvalues[nearest].max()) if bool(nearest.any()) else None,
                positive_real_target=int(snap._mass_targets(len(z))[cells].sum()),
                real_reference_rows=int((rm==mode).sum())))
    start=time.perf_counter()
    attempts=propose_inaccessible_anchors(snap,q,flags,pvalues,z,callback,ema_latents=ez,
        ema_feature_of_latent=average,candidate_limit=feature.REAL_ANCHORS,passes=feature.POWER_PASSES)
    proposal_seconds=time.perf_counter()-start
    annotated=[]
    for proposal in attempts:
        cell=proposal['cell'];seed=proposal['seed_row']
        for key,weights in (('current',models['G']),('average',models['ema_G'])):
            value=proposal[key]
            with torch.no_grad():
                generated=forward(value['latent'][None],weights)
                distance,mode,accepted=oracle(generated)
            value['oracle_annotation_only']=dict(raw_output=generated,nearest_mode=int(mode[0]),
                centre_distance=float(distance[0]),oracle_accepted=bool(accepted[0]),target_reference_mode=int(cell_mode[cell]))
        proposal['seed_oracle_annotation_only']=dict(training_raw=raw[seed],training_mode=int(m[seed]),training_distance=float(d[seed]),
            training_pvalue=float(pvalues[seed]),training_score=float(scores[seed]),served_raw=served[seed],served_distance=float(ed[seed]))
        annotated.append(plain(proposal))
    assert tensor_state_hash(s)==original,'G/D/prior/optimizer/stream tensors changed'
    record=dict(step=step,current_support_flags=int(flags.sum()),current_eligible=int((pvalues>.05).sum()),
        mode_diagnosis=mode_rows,attempts=annotated,accepted=sum(x['accepted'] for x in attempts),
        attempted=len(attempts),proposal_seconds=proposal_seconds,training_state_sha256_before=original,
        training_state_sha256_after=tensor_state_hash(s),model_prior_optimizer_stream_unchanged=True,
        fit_scope='saved FIFO/current critic, recorded CPU RNG; not exact last GPU geometry')
    records.append(record)
    print(json.dumps(dict(event='proposal_saved_state',step=step,attempted=record['attempted'],accepted=record['accepted'],
        proposal_seconds=proposal_seconds,summary=[dict(cell=x['cell'],seed_copy_eligible=x['seed_copy_eligible'],
            current_p=x['current']['support_pvalue'],average_p=x['average']['support_pvalue'],
            current_oracle=x['current']['oracle_annotation_only'],average_oracle=x['average']['oracle_annotation_only'],
            accepted=x['accepted']) for x in annotated])),flush=True)
assert before=={p:sha(Path(p)) for p in before} and torch.equal(global_rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
receipt=dict(status='COMPLETE_FIXED_POLICY_CPU_RESPONSE',records=records,sources_sha256=before,sources_unchanged=True,
    global_rng_unchanged=True,cuda_initialized=False,new_seeds=0,optimizer_updates=0,generated_emissions=0,
    committed_births=0,quality_verdict=None,
    protocol='fixed four-anchor/four-linearization paired real-anchor proposal; no gate/seed/cutoff sweep')
(HERE/'saved-proposal-response.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(event='proposal_response_complete',status=receipt['status'],cuda_initialized=False)),flush=True)
