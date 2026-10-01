"""Fixed saved-row geometry probe; no draws, constructors of networks or updates."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import weakref
import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
FIRST=HERE.parent/'grid-covariance'
RUN=ROOT/'validation-cb64-ra8/screens/runs/grid100'
FC=ROOT/'pkg-CB64-RA8/particlegan/feature_cells.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
SIGMA=.03
grid=np.arange(10,dtype=np.float64)-4.5
CENTERS=np.stack(np.meshgrid(grid,grid,indexing='ij'),axis=-1).reshape(-1,2)
torch.set_num_threads(1)

def assign(x):
    d=np.asarray(x,dtype=np.float64)[:,None,:]-CENTERS[None,:,:]
    return np.argmin(np.einsum('nkd,nkd->nk',d,d),axis=1)

def summarize(x):
    a=np.asarray(x,dtype=np.float64)
    return dict(min=float(a.min()),median=float(np.median(a)),mean=float(a.mean()),max=float(a.max()),p95=float(np.quantile(a,.95)))

def select(points,modes):
    ids=assign(points.numpy());rows=[]
    for mode in modes:
        members=np.flatnonzero(ids==mode)
        radii=np.linalg.norm(points.numpy()[members]-CENTERS[mode],axis=1)
        extremes=members[np.argsort(radii,kind='stable')[::-1][:8]]
        spaced=members[np.linspace(0,len(members)-1,8,dtype=np.int64)]
        rows.extend(np.unique(np.concatenate((extremes,spaced))).tolist())
    return torch.tensor(rows,dtype=torch.long),ids

def reconstruct(geometry,points,queries,rows):
    distances=[];candidate_ids=[]
    for index,(axis,values,axis_rows) in enumerate(geometry._orders(points)):
        count=geometry.neighbors//len(geometry._orders(points))+(index<geometry.neighbors%len(geometry._orders(points)))
        offsets=torch.arange(-(count//2),count-count//2)
        positions=torch.searchsorted(values,queries[:,axis].contiguous())[:,None]+offsets
        valid=(positions>=0)&(positions<len(values))
        candidates=axis_rows[positions.clamp(0,len(values)-1)]
        d=(queries[:,None]-points[candidates]).square().sum(-1)
        d.masked_fill_(~valid|(d==0),float('inf'))
        distances.append(d);candidate_ids.append(candidates)
    candidates=geometry.lineage.neighbors[rows]
    d=(queries[:,None]-points[candidates.clamp_min(0)]).square().sum(-1)
    d.masked_fill_((candidates<0)|(d==0),float('inf'))
    distances.append(d);candidate_ids.append(candidates.clamp_min(0))
    distances=torch.cat(distances,1);candidate_ids=torch.cat(candidate_ids,1)
    nearest=distances.min(1).values.sqrt()
    radius=torch.where(torch.isfinite(nearest),.5*nearest,torch.zeros_like(nearest))
    selected=distances.argsort(dim=1,stable=True)[:,:min(geometry.rank,distances.shape[1])]
    valid=torch.isfinite(distances.gather(1,selected))
    neighbors=points[candidate_ids.gather(1,selected)]
    width=((queries[:,None]-neighbors).square()*valid[:,:,None]).sum(1)/(2*valid.sum(1).clamp_min(1)[:,None])
    return distances,candidate_ids,radius,width.sqrt(),selected,valid

def bounds(weight,radius,width):
    a=weight.double();w=width.double();r=radius.double()
    trace=(w.square()*a.square().sum(0)[None]).sum(1)
    spectral=float(torch.linalg.svdvals(a).max())**2*r.square()
    return torch.minimum(trace,spectral)/SIGMA**2

def main():
    seal=read(HERE/'PREPARATION-FROZEN.json')
    for p,d in seal['source_and_input_sha256'].items():assert sha(p)==d,p
    assert not torch.cuda.is_initialized();rng=torch.get_rng_state().clone()
    first=read(FIRST/'result.json')
    ranks=sorted(first['paired_covariance']['holdout100k/live']['same_noisy_HQ']['per_mode'],key=lambda x:(-x['max_eig_sigma2'],x['mode']))
    modes=[v['mode'] for v in ranks[:3]]+[ranks[50]['mode']]
    state=torch.load(RUN/'final-state.pt',map_location='cpu',weights_only=False)['trainer']
    assert state['completed_steps']==7000
    source=ast.parse(FC.read_text())
    classes=[v for v in source.body if isinstance(v,ast.ClassDef) and v.name in ('LatentLineage','BoundedLatentGeometry')]
    assert len(classes)==2
    namespace=dict(torch=torch,weakref=weakref,PARENT_RESERVOIR=64)
    exec(compile(ast.Module(body=classes,type_ignores=[]),str(FC),'exec'),namespace)
    lineage=namespace['LatentLineage'](20000,8,'cpu')
    lineage.validate(state['birth_death']['lineage_neighbors'])
    lineage.neighbors=state['birth_death']['lineage_neighbors'].clone()
    bandwidth=torch.as_tensor(state['controller']['latent_bandwidth'],dtype=torch.float32)
    records={};total_queries=0;all_pairs=0
    for name,model,prior_name in (('FAST','G','prior'),('EMA','ema_G','ema_prior')):
        z=state['models'][prior_name]['z'].clone()
        weight=state['models'][model]['weight'];bias=state['models'][model]['bias']
        anchors=F.linear(z,weight,bias)
        rows,oracle=select(anchors,modes)
        queries=z[rows];total_queries+=len(rows)
        geometry=namespace['BoundedLatentGeometry'](rank=8,neighbors=64,chunk=256,lineage=lineage)
        radius,width=geometry._local_geometry(queries,SimpleNamespace(z=z),rows=rows)
        d,ci,rr,ww,selected,valid=reconstruct(geometry,z,queries,rows)
        assert torch.equal(radius,rr) and torch.equal(width,ww)
        exact=[]
        for block in queries.split(32):
            exact.append((block[:,None]-z[None]).square().sum(-1))
        full=torch.cat(exact);all_pairs+=full.numel()
        zero=(full==0);duplicate_counts=zero.sum(1)-1
        other=full.clone();other[torch.arange(len(rows)),rows]=float('inf')
        exact_other=other.min(1).values.sqrt()
        full[zero]=float('inf')
        minpositive=full.min(1).values;exact_radius=.5*minpositive.sqrt()
        ei=full.argsort(dim=1,stable=True)[:,:8]
        evalid=torch.isfinite(full.gather(1,ei))
        exact_width=(((queries[:,None]-z[ei]).square()*evalid[:,:,None]).sum(1)/(2*evalid.sum(1).clamp_min(1)[:,None])).sqrt()
        applied=torch.minimum(bandwidth,width);eapplied=torch.minimum(bandwidth,exact_width)
        upper=bounds(weight,radius,applied);eupper=bounds(weight,exact_radius,eapplied)
        row_results=[]
        for j,row in enumerate(rows.tolist()):
            candidates=ci[j][torch.isfinite(d[j])]
            best=full[j].min()
            captures=bool((full[j,candidates]==best).any())
            chosen=ci[j,selected[j]][valid[j]]
            mode=int(oracle[row]);parts=[]
            for axis,values,axis_rows in geometry._orders(z):
                parts.append(dict(axis=axis,tied_coordinate_rows=int((z[:,axis]==queries[j,axis]).sum())))
            ratio=float(radius[j]/exact_radius[j]) if float(exact_radius[j])>0 else None
            row_results.append(dict(row=row,mode=mode,anchor=anchors[row].tolist(),
                 anchor_radius_sigma=float(np.linalg.norm(anchors[row].numpy()-CENTERS[mode])/SIGMA),
                 positive_radius=float(radius[j]),exact_positive_radius=float(exact_radius[j]),radius_inflation=ratio,
                 exact_nearest_other_distance=float(exact_other[j]),duplicate_other_rows=int(duplicate_counts[j]),
                 exact_nearest_positive_candidate_captured=captures,candidate_slots=int(len(candidates)),
                 unique_finite_candidates=int(len(torch.unique(candidates))),
                 local_width=width[j].tolist(),exact_nearest8_width=exact_width[j].tolist(),
                 actual_applied_width=applied[j].tolist(),exact_applied_width=eapplied[j].tolist(),
                 selected_nearest8_rows=chosen.tolist(),selected_nearest8_unique_rows=int(len(torch.unique(chosen))),
                 finite_candidate_other_oracle_mode_fraction=float(np.mean(oracle[candidates.numpy()]!=mode)),
                 selected_nearest8_other_oracle_mode_fraction=float(np.mean(oracle[chosen.numpy()]!=mode)),
                 actual_output_displacement_RMS_upper_sigma2=float(upper[j]),
                 exact_geometry_output_displacement_RMS_upper_sigma2=float(eupper[j]),
                 coordinate_ties=parts))
        per_mode={}
        for mode in modes:
            group=[r for r in row_results if r['mode']==mode]
            per_mode[str(mode)]=dict(queries=len(group),captured_exact_nearest_fraction=float(np.mean([r['exact_nearest_positive_candidate_captured'] for r in group])),
                radius_inflation=summarize([r['radius_inflation'] for r in group]),
                actual_output_displacement_RMS_upper_sigma2=summarize([r['actual_output_displacement_RMS_upper_sigma2'] for r in group]),
                nearest8_cross_oracle_fraction=summarize([r['selected_nearest8_other_oracle_mode_fraction'] for r in group]))
        records[name]=dict(queries=len(rows),rows=row_results,per_mode=per_mode,geometry_work=geometry.work,
             actual_output_displacement_RMS_upper_sigma2=summarize(upper.numpy()),
             radius_inflation=summarize([r['radius_inflation'] for r in row_results]),
             sampled_query_ids_were_not_saved=True)
    assert total_queries<=128 and all_pairs<=128*20000
    assert torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
    for p,d in seal['source_and_input_sha256'].items():assert sha(p)==d,p
    out=dict(status='VALID',scope='Fixed saved-row subset only; no new draws/quality/candidate change',selected_modes=modes,
         selection='holdout_covariance_top3_plus_median;8_radial_extremes_plus8_even_row_positions_permode',
         queried_rows=total_queries,query_by_table_distance_pairs=all_pairs,no_all_table_pair_graph=True,
         global_bandwidth=bandwidth.tolist(),results=records,cpu_only=True,cuda_initialized=False,
         global_CPU_RNG_unchanged=True,source_and_input_sha256=seal['source_and_input_sha256'])
    (HERE/'result.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='VALID',modes=modes,rows=total_queries,pairs=all_pairs,
        FAST_inflation=records['FAST']['radius_inflation'],EMA_inflation=records['EMA']['radius_inflation'],
        result_sha256=sha(HERE/'result.json'))))

if __name__=='__main__':main()
