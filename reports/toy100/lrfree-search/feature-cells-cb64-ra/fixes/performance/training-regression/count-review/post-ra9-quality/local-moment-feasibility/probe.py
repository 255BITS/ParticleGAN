"""One fixed saved-only local-moment feasibility probe; no corrections or scoring."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
    MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import ast
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
RUN=ROOT/'validation-cb64-ra9/screens/runs/grid100'
PACKAGE=ROOT/'pkg-CB64-RA9'
HEAD=ROOT/'performance/training-regression/count-review/post-ra8-quality/grid-covariance/diagnose.py'
HASH=ROOT/'integration/review/training-regression/post-ra4-quality/measure_saved_utils.py'
GEOMETRY=Path('/ml2/hypergan/lrfree-20260926/harness/hosts/native100/problems.py')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def verify():
    frozen=read(HERE/'PREPARATION-FROZEN.json')
    for path,digest in frozen['source_and_input_sha256'].items():assert sha(path)==digest,path
    return frozen
def extract(path,names):
    nodes=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    ns=dict(torch=torch,F=F,hashlib=hashlib,math=math,
            PROBLEM_NAMES=('grid100','rotated100','staggered100'),DATA_STD=.03)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return [ns[name] for name in names]
def plain(x):
    if isinstance(x,torch.Tensor):return plain(x.detach().cpu().numpy())
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,(np.integer,np.floating)):return plain(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x
def moments(points,ids,k):
    x=np.asarray(points,dtype=np.float64);ids=np.asarray(ids,dtype=np.int64)
    assert x.ndim==2 and x.shape[1]==2 and len(x)==len(ids) and np.isfinite(x).all()
    assert (ids>=0).all() and (ids<k).all()
    n=np.bincount(ids,minlength=k);den=np.maximum(n,1)
    mean=np.stack([np.bincount(ids,weights=x[:,j],minlength=k)/den for j in range(2)],1)
    centered=x-mean[ids]
    cov=np.stack([np.bincount(ids,weights=centered[:,j]*centered[:,l],minlength=k)/den
                  for j in range(2) for l in range(2)],1).reshape(k,2,2)
    return dict(rows=n,mean=mean,covariance=cov,defined=n>0,
                covariance_eigenvalues=np.linalg.eigvalsh(cov))
def weighted_rms(delta,weights,mask):
    return float(np.sqrt(np.sum(weights[mask]*np.sum(delta[mask]**2,1))))
def comparison(a,b,weights):
    mask=(a['rows']>0)&(b['rows']>0)
    delta=a['mean']-b['mean']
    return dict(defined_groups=mask,used_reference_mass=float(weights[mask].sum()),
        omitted_reference_mass=float(weights[~mask].sum()),mean_difference=delta,
        weighted_mean_difference_rms=weighted_rms(delta,weights,mask),
        unweighted_mean_difference_rms=float(np.sqrt(np.mean(np.sum(delta[mask]**2,1)))) if mask.any() else None)
def reference_agreement(anchor,even,odd,weights):
    valid=(anchor['rows']>0)&(even['rows']>0)&(odd['rows']>0)
    de=anchor['mean']-even['mean'];do=anchor['mean']-odd['mean'];split=even['mean']-odd['mean']
    dots=np.sum(de*do,1);norms=np.linalg.norm(de,axis=1)*np.linalg.norm(do,axis=1)
    cosine=np.full(len(weights),np.nan);np.divide(dots,norms,out=cosine,where=norms>0)
    v=np.trace(even['covariance'],axis1=1,axis2=2)/2
    positive=valid&(v>0);scaled=np.zeros(len(v));scaled[positive]=dots[positive]/v[positive]
    eligible_se=valid&(even['rows']>1)&(odd['rows']>1)
    se2=np.zeros(len(v))
    se2[eligible_se]=(np.trace(even['covariance'][eligible_se],axis1=1,axis2=2)/(even['rows'][eligible_se]-1)
        +np.trace(odd['covariance'][eligible_se],axis1=1,axis2=2)/(odd['rows'][eligible_se]-1))
    return dict(defined_groups=valid,used_reference_mass=float(weights[valid].sum()),
        omitted_reference_mass=float(weights[~valid].sum()),d_even=de,d_odd=do,
        residual_dot_product=dots,residual_cosine=cosine,
        positive_dot_groups=int(((dots>0)&valid).sum()),nonpositive_dot_groups=int(((dots<=0)&valid).sum()),
        positive_dot_reference_mass=float(weights[(dots>0)&valid].sum()),
        A_raw_output_units=float(np.sum(weights[valid]*dots[valid])),
        A_even_within_variance_standardized=float(np.sum(weights[positive]*scaled[positive])),
        standardized_used_reference_mass=float(weights[positive].sum()),
        offset_even_weighted_rms=weighted_rms(de,weights,valid),
        offset_odd_weighted_rms=weighted_rms(do,weights,valid),
        real_split_weighted_rms=weighted_rms(split,weights,valid),
        optimistic_iid_split_mean_noise_energy=float(np.sum(weights[eligible_se]*se2[eligible_se])),
        optimistic_iid_split_mean_noise_defined_mass=float(weights[eligible_se].sum()),
        interpretation='Descriptive split reproducibility; current D/table/EMA share FIFO training, no calibrated null or local action certificate.')
def pair_identity(clean,noisy,ids,k,weights):
    clean=np.asarray(clean,dtype=np.float64);noisy=np.asarray(noisy,dtype=np.float64)
    noise=noisy-clean
    c,e,n=(moments(x,ids,k) for x in (clean,noise,noisy))
    cc=clean-c['mean'][ids];ee=noise-e['mean'][ids]
    den=np.maximum(c['rows'],1)
    cross=np.stack([np.bincount(ids,weights=cc[:,j]*ee[:,l],minlength=k)/den
                    for j in range(2) for l in range(2)],1).reshape(k,2,2)
    cross+=cross.transpose(0,2,1).copy()
    err=float(np.max(np.abs(c['covariance']+e['covariance']+cross-n['covariance'])))
    mean_err=float(np.max(np.abs(c['mean']+e['mean']-n['mean'])))
    assert err<1e-10 and mean_err<1e-10
    mask=c['rows']>0
    trace=lambda x:float(np.sum(weights[mask]*np.trace(x[mask],axis1=1,axis2=2)/2))
    return dict(rows=c['rows'],clean=c,noise=e,noisy=n,symmetric_cross_covariance=cross,
        covariance_identity_max_abs_error=err,mean_identity_max_abs_error=mean_err,
        weighted_conditional_noise_mean_rms=weighted_rms(e['mean'],weights,mask),
        weighted_mean_covariance_per_coordinate=dict(clean=trace(c['covariance']),noise=trace(e['covariance']),
            cross=trace(cross),noisy=trace(n['covariance'])),
        used_reference_mass=float(weights[mask].sum()),
        global_noise_mean=noise.mean(0),global_noise_covariance=np.cov(noise,rowvar=False,bias=True))
def alias(ids,labels,k,modes):
    table=np.bincount(np.asarray(ids)*modes+labels,minlength=k*modes).reshape(k,modes)
    counts=table.sum(1);represented=(table>0).sum(1)
    return dict(contingency=table,group_rows=counts,modes_per_group=represented,
        nonempty_groups=int((counts>0).sum()),multimode_groups=int((represented>1).sum()),
        weighted_majority_purity=float(table.max(1).sum()/len(labels)),
        dominant_mode_per_group=table.argmax(1))
def main():
    assert not (HERE/'result.json').exists()
    frozen=verify();started=time.perf_counter()
    sys.path.insert(0,str(PACKAGE))
    from particlegan.feature_cells import FeatureCellSnapshot
    head,=extract(HEAD,['head_features'])
    state_hash,=extract(HASH,['tensor_state_hash'])
    global_rng=torch.get_rng_state().clone();numpy_rng=np.random.get_state()
    assert not torch.cuda.is_initialized()
    saved=torch.load(RUN/'final-state.pt',map_location='cpu',weights_only=False)
    state=saved['trainer'];state_before=state_hash(state)
    assert state['completed_steps']==7000 and state['recipe']['num_particles']==20000
    bd=state['birth_death'];models=state['models']
    assert state['schema']==5 and bd['backend_schema']==8
    assert bd['settings']['cells']==128 and bd['settings']['rank']==8 and bd['settings']['chunk']==256
    assert bd['fill']==20000
    assert set(models['G'])==set(models['ema_G'])=={'weight','bias'}
    dtype=models['D']['net.0.weight'].dtype
    forward_rows=0
    def features(x):
        nonlocal forward_rows
        t=x if isinstance(x,torch.Tensor) else torch.as_tensor(x)
        assert t.ndim==2 and t.shape[1]==2 and torch.isfinite(t).all()
        forward_rows+=len(t)
        return head(t.to(dtype=dtype),models['D']).double()
    with torch.no_grad():
        raw_real=bd['reservoir'].detach().cpu().numpy()
        real_features=features(bd['reservoir'])
        private=torch.Generator().set_state(state['cpu_rng'])
        private_start=state_hash(private.get_state())
        snapshot=FeatureCellSnapshot.fit(real_features,generator=private,cells=128,rank=8,chunk=256)
        private_end=state_hash(private.get_state())
        assert snapshot.rank==8 and snapshot.cells==128
        group_map=snapshot._mass_topology();k=snapshot.mass_groups
        def assign_features(f):return group_map[snapshot.assign(f)[0]].cpu().numpy()
        def assign_points(x):return assign_features(features(x))
        real_ids=assign_features(real_features)
        even=moments(raw_real[0::2],real_ids[0::2],k)
        odd=moments(raw_real[1::2],real_ids[1::2],k)
        pooled=moments(raw_real,real_ids,k)
        weights=even['rows']/len(raw_real[0::2])
        anchor={};anchor_points={};anchor_ids={}
        for name,role,prior in (('FAST','G','prior'),('EMA','ema_G','ema_prior')):
            points=F.linear(models[prior]['z'],models[role]['weight'],models[role]['bias'])
            anchor_points[name]=points.detach().cpu().numpy();anchor_ids[name]=assign_points(points)
            anchor[name]=moments(anchor_points[name],anchor_ids[name],k)
        agreement={name:reference_agreement(v,even,odd,weights) for name,v in anchor.items()}
        comparisons={name:{target:comparison(v,r,weights) for target,r in (('even',even),('odd',odd),('pooled',pooled))}
                     for name,v in anchor.items()}
        cloud={};cloud_points_for_annotation={}
        for split,filename,rows in (('terminal20k','final_samples.npz',20000),('holdout100k','holdout_samples.npz',100000)):
            with np.load(RUN/'native-clean'/filename,allow_pickle=False) as clean_file,\
                 np.load(RUN/'native-noisy'/filename,allow_pickle=False) as noisy_file:
                assert set(clean_file.files)==set(noisy_file.files)=={'live','ema','target'}
                assert np.array_equal(clean_file['target'],noisy_file['target'])
                target=clean_file['target'].copy()
                assert target.shape==(rows,2)
                target_ids=assign_points(target);tm=moments(target,target_ids,k)
                record=dict(target_real=tm,target_comparisons={name:comparison(tm,r,weights)
                    for name,r in (('even',even),('odd',odd),('pooled',pooled),('FAST_anchor',anchor['FAST']),('EMA_anchor',anchor['EMA']))},views={})
                first_pair=None
                for view in ('live','ema'):
                    c,n=clean_file[view].copy(),noisy_file[view].copy()
                    assert c.shape==n.shape==(rows,2) and np.isfinite(c).all() and np.isfinite(n).all()
                    if first_pair is not None and np.array_equal(c,first_pair[0]) and np.array_equal(n,first_pair[1]):
                        record['views'][view]=dict(numerically_identical_to='live',independent_model_control=False)
                        continue
                    cid,nid=assign_points(c),assign_points(n)
                    cm,nm=moments(c,cid,k),moments(n,nid,k)
                    transitions=np.bincount(cid*k+nid,minlength=k*k).reshape(k,k)
                    record['views'][view]=dict(clean_own_group=cm,noisy_own_group=nm,
                        group_transitions=transitions,group_switch_rows=int((cid!=nid).sum()),
                        group_switch_fraction=float((cid!=nid).mean()),
                        conditional_by_clean_group=pair_identity(c,n,cid,k,weights),
                        conditional_by_noisy_group=pair_identity(c,n,nid,k,weights),
                        clean_comparisons={name:comparison(cm,r,weights) for name,r in
                            (('even',even),('odd',odd),('pooled',pooled),('saved_target_real',tm),('FAST_anchor',anchor['FAST']),('EMA_anchor',anchor['EMA']))},
                        noisy_comparisons={name:comparison(nm,r,weights) for name,r in
                            (('even',even),('odd',odd),('pooled',pooled),('saved_target_real',tm),('FAST_anchor',anchor['FAST']),('EMA_anchor',anchor['EMA']))})
                    cloud_points_for_annotation[(split,view)]=(c,n,cid,nid)
                    if view=='live':first_pair=(c,n)
                cloud[split]=record
        result=read(RUN/'result.json');execution=read(RUN/'execution-receipt.json')
        assert result['completed_steps']==7000 and result['status']=='FAIL'
        assert sha(RUN/'result.json')==execution['result_sha256']
        assert execution['source_integrity_before']==execution['source_integrity_after']
        assert execution['source_integrity_after']['status']=='VALID'
        metrics={r['step']:r for r in (json.loads(line) for line in (RUN/'metrics.jsonl').read_text().splitlines())}
        sigma=float(metrics[7000]['output_sigma'])
        assert math.isfinite(sigma) and sigma>=0
        budget={}
        for name,r in (('even',even),('odd',odd),('pooled',pooled)):
            proxy=r['covariance']-sigma*sigma*np.eye(2)[None]
            eigen=np.linalg.eigvalsh(proxy);valid=r['rows']>0
            budget[name]=dict(covariance_minus_applied_noise_proxy=proxy,eigenvalues=eigen,
                defined_groups=valid,groups_with_negative_proxy_eigenvalue=int(((eigen[:,0]<0)&valid).sum()),
                weighted_proxy_variance_per_coordinate=float(np.sum(weights[valid]*np.trace(proxy[valid],axis1=1,axis2=2)/2)),
                interpretation='Optimistic clean covariance budget only; conditional group/noise selection and latent jitter are not removed or certified.')
        # Oracle geometry appears only after every real-only chart/moment calculation.
        _,geometry=extract(GEOMETRY,['_centers','evaluation_geometry'])
        centers,target_sigma=geometry('grid100',dtype=torch.float64)
        centers=centers.numpy();target_sigma=float(target_sigma)
        def annotate(points):
            x=np.asarray(points,dtype=np.float64)
            return np.concatenate([((b[:,None]-centers[None])**2).sum(2).argmin(1) for b in np.array_split(x,math.ceil(len(x)/256))])
        annotation=dict(scope='Nearest original oracle mode downstream only; no production/chart/statistic input',
            real_even=alias(real_ids[0::2],annotate(raw_real[0::2]),k,len(centers)),
            real_odd=alias(real_ids[1::2],annotate(raw_real[1::2]),k,len(centers)),
            oracle_sigma=target_sigma,
            weighted_offset_even_sigma={name:v['offset_even_weighted_rms']/target_sigma for name,v in agreement.items()},
            weighted_offset_odd_sigma={name:v['offset_odd_weighted_rms']/target_sigma for name,v in agreement.items()},
            reference_split_rms_sigma=agreement['EMA']['real_split_weighted_rms']/target_sigma,
            cloud_oracle_mode_switch_rows={})
        for key,(c,n,cid,nid) in cloud_points_for_annotation.items():
            annotation['cloud_oracle_mode_switch_rows']['/'.join(key)]=int((annotate(c)!=annotate(n)).sum())
    assert state_hash(state)==state_before,'Loaded trainer state mutated'
    assert torch.equal(torch.get_rng_state(),global_rng),'Global Torch RNG advanced'
    after_numpy=np.random.get_state()
    assert all(np.array_equal(a,b) if isinstance(a,np.ndarray) else a==b for a,b in zip(numpy_rng,after_numpy))
    assert not torch.cuda.is_initialized()
    verify()
    verdict=read(RUN/'native-noisy/verdict.json')
    record=dict(status='VALID_FIXED_SAVED_DIAGNOSTIC',source_freeze_sha256=sha(HERE/'PREPARATION-FROZEN.json'),
        saved_step=7000,backend_schema=8,trainer_schema=5,
        chart=dict(requested_cells=128,actual_cells=snapshot.cells,rank=snapshot.rank,width=snapshot.width,
            fitted_rows=len(raw_real[0::2]),calibration_rows=len(raw_real[1::2]),groups=k,
            topology=snapshot.mass_topology,geometry_sha256=state_hash([snapshot.mean,snapshot.scale,snapshot.basis,snapshot.centers,group_map]),
            private_fit_rng_start_sha256=private_start,private_fit_rng_end_sha256=private_end,
            work=dict(snapshot.work),historical_GPU_chart_reconstructed=False),
        actual_saved_paired_average=bd['paired_average'],references=dict(even=even,odd=odd,pooled=pooled),
        raw_anchors=anchor,anchor_reference_agreement=agreement,anchor_reference_comparisons=comparisons,
        clouds=cloud,applied_output_sigma=sigma,optimistic_covariance_budget=budget,annotation_only=annotation,
        original_result=dict(status=result['status'],final=result['final'],thresholds=result['thresholds'],
            original_terminal_checks=verdict['accuracy']['terminal_checks']),
        state_before_sha256=state_before,state_after_sha256=state_hash(state),
        numerical_state_and_input_bytes_unchanged=True,global_Torch_NumPy_trainer_rng_unchanged=True,
        charts=1,private_rng_draws='existing one chart projection draw from cloned savedCPU RNG only',
        model_constructions=0,functional_D_head_rows=forward_rows,training_updates=0,optimizer_steps=0,
        new_emitted_samples=0,new_seeds=0,actions=0,counterfactual_quality_scores=0,cuda_initialized=False,
        statistical_certificate=None,quality_verdict=None,
        limitation='Descriptive split-reference moments with shared trained-D/FIFO, not independence/equivalence/stationarity/action evidence.',
        cpu_seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat())
    with (HERE/'result.json').open('x') as f:f.write(json.dumps(plain(record),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=record['status'],groups=k,
        A_EMA=agreement['EMA']['A_even_within_variance_standardized'],
        positive_direction_groups=agreement['EMA']['positive_dot_groups'],
        positive_direction_mass=agreement['EMA']['positive_dot_reference_mass'],
        EMA_real_even_offset_sigma=annotation['weighted_offset_even_sigma']['EMA'],
        real_split_rms_sigma=annotation['reference_split_rms_sigma'],
        functional_head_rows=forward_rows,cpu_seconds=record['cpu_seconds'],result_sha256=sha(HERE/'result.json'))),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise
