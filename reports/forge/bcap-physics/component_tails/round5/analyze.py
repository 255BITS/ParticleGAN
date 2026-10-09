"""Saved endpoint center/kernel decomposition and uncensored native tail audit."""
from pathlib import Path
import importlib.util
import json
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from experiments.forge.contracts import atomic_json,file_hash,read_json
from benchmarks.toy100.problems import evaluation_geometry
from experiments.forge import tier1_media
OUT=Path(__file__).resolve().parent
ARCHIVE=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/component_tails')


def radial_shape(receipt):
    directory=Path(receipt['artifact_root'])
    request=read_json(directory/'request.json')['request']
    result=read_json(directory/'result.json')
    task=request['tasks'][receipt['task_id']]
    row=next(r for r in result['task_results'] if r['task_id']==receipt['task_id'])
    records,inputs=tier1_media._scored_outputs(task,row['evidence'],directory)
    points=records[-1]['samples'].double()
    definition=task['execution']['host_definition']
    means=points.new_tensor(definition['means']);cov=points.new_tensor(definition['covariances'])
    assigned=torch.cdist(points,means).argmin(1)
    delta=points-means[assigned]
    squared=torch.einsum('ni,nij,nj->n',delta,torch.linalg.inv(cov)[assigned],delta)
    def ks(values):
        if not len(values):return None
        transformed=(1-torch.exp(-.5*values)).sort().values
        indices=torch.arange(1,len(values)+1,dtype=torch.float64)
        return float(torch.maximum(indices/len(values)-transformed,
                                   transformed-(indices-1)/len(values)).max())
    return {'arm':receipt['arm'],'task_id':receipt['task_id'],
        'scope':'uncensored_saved_final_clean_live_radial_diagnostic_not_new_gate',
        'optimizer_updates_added':0,'sampling_draws_added':0,
        'global_whitened_radial_ks':ks(squared),
        'component_whitened_radial_ks':[ks(squared[assigned==k]) for k in range(len(means))],
        'component_counts':torch.bincount(assigned,minlength=len(means)).tolist(),
        'reference':'chi-square2 CDF; nearest-cell assignment can alter the reference when target components overlap',
        'inputs':inputs,'original_full_covariance_metric':row['evidence']['live']['component_covariance_error']}


def native_tail(receipt):
    directory=Path(receipt['artifact_root'])
    request=read_json(directory/'request.json')['request']
    path=directory/'native100/grid100/final_samples.npz'
    with np.load(path,allow_pickle=False) as saved:
        points=torch.from_numpy(saved['live'].copy()).double()
    assert torch.isfinite(points).all()
    means,sigma=evaluation_geometry('grid100',dtype=torch.float64)
    distance,assigned=torch.cdist(points,means).min(1)
    counts=torch.bincount(assigned,minlength=100)
    errors=[];spills=[];ratios=[];missing=[]
    for k in range(100):
        selected=points[assigned==k]
        if len(selected)<10:
            missing.append(k);errors.append(1.);spills.append(1.);ratios.append(0.)
            continue
        delta=selected-selected.mean(0)
        covariance=delta.T@delta/len(selected)/sigma**2
        errors.append(float((covariance-torch.eye(2,dtype=torch.float64)).norm()/2**.5))
        ratios.append(float(torch.linalg.eigvalsh(covariance).min()))
        spills.append(float((distance[assigned==k]>3*sigma).double().mean()))
    transformed=(1-torch.exp(-.5*(distance/sigma).square())).sort().values
    indices=torch.arange(1,len(points)+1,dtype=torch.float64)
    ks=float(torch.maximum(indices/len(points)-transformed,
                            transformed-(indices-1)/len(points)).max())
    return {'arm':receipt['arm'],'task_id':'grid100','scope':'uncensored_final_20k_clean_live_nearest_cells',
        'sampling_draws_added':0,'optimizer_updates_added':0,'samples':len(points),
        'input_path':str(path),'input_sha256':file_hash(path),
        'source_digest':request['source']['digest'],
        'component_counts':counts.tolist(),'missing_or_under_10_sample_components':missing,
        'component_covariance_error':sum(errors)/100,'component_covariance_errors':errors,
        'component_min_eigen_ratio':min(ratios),'component_spill':spills,
        'max_component_spill':max(spills),'global_spill':float((distance>3*sigma).double().mean()),
        'uncensored_radial_ks':ks,'mass_tv':float((counts/len(points)-.01).abs().sum()/2),
        'covariance_definition':'population covariance about each assigned cell own mean, normalized by target sigma squared',
        'missing_component_policy':'count<10 receives error1/spill1/eigen0, remains explicitly missing',
        'qualification_input':False,'authoritative_gates':'original native sustained coverage/accuracy plus independent holdout'}


def main():
    start=time.monotonic();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rng=torch.get_rng_state().clone()
    spec=importlib.util.spec_from_file_location('component_tail_census',OUT/'diagnose.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    provenance=read_json(OUT/'provenance.json')
    rows=[];native=[];radial=[]
    for receipt in provenance['attempts']:
        if not (receipt['task_id'].startswith('vector_') or receipt['task_id']=='grid100'):
            continue
        row=module.diagnose('round5-'+receipt['arm'],receipt)
        row['arm']=receipt['arm'];rows.append(row)
        if receipt['task_id']=='grid100':native.append(native_tail(receipt))
        else:radial.append(radial_shape(receipt))
    full=ARCHIVE/'current-component-full.json'
    atomic_json(full,{'diagnostics':rows,'native_uncensored':native,'vector_radial':radial})
    for row in rows:
        if row['task_id']!='grid100':continue
        components=row.pop('component_summary')
        keys=[k for k,v in next(v for v in components if not v.get('missing_shape')).items() if isinstance(v,float)]
        row['component_summary_aggregate']={k:{'mean':float(np.mean([v[k] for v in components if k in v])),
            'median':float(np.median([v[k] for v in components if k in v])),
            'min':float(np.min([v[k] for v in components if k in v])),
            'max':float(np.max([v[k] for v in components if k in v]))} for k in keys}
        row['missing_or_single_center_components']=[v['component'] for v in components if v.get('missing_shape')]
    assert torch.equal(rng,torch.get_rng_state())
    atomic_json(OUT/'current-component-diagnostics.json',{'schema_version':1,'qualification_input':False,
        'optimizer_updates_added':0,'sampling_draws_added':0,'analysis_seconds':time.monotonic()-start,
        'global_rng_unchanged':True,'scope':'exhaustive_center_outputs; approximate nonlinear Gaussian moments under fixed center assignments',
        'full_artifact':{'path':str(full),'sha256':file_hash(full)},'diagnostics':rows,'native_uncensored':native,
        'vector_radial':radial})
    print({'event':'current_component_analysis_complete','seconds':time.monotonic()-start,
           'rows':len(rows)},flush=True)


if __name__=='__main__':main()
