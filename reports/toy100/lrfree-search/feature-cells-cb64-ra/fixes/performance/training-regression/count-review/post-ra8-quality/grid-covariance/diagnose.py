"""Fixed saved-only CPU covariance identities; no model constructors or draws."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
RUN = ROOT/'validation-cb64-ra8/screens/runs/grid100'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
K, SIGMA = 100, .03
grid = np.arange(10, dtype=np.float64)-4.5
CENTERS = np.stack(np.meshgrid(grid,grid,indexing='ij'),axis=-1).reshape(-1,2)
BOUNDARY = (np.abs(CENTERS)==4.5).any(1)
CORNER = (np.abs(CENTERS)==4.5).all(1)
torch.set_num_threads(1)

def stats(x):
    a=np.asarray(x,dtype=np.float64)
    return dict(mean=float(a.mean()),median=float(np.median(a)),min=float(a.min()),max=float(a.max()),
                p95=float(np.quantile(a,.95)))

def assign(x):
    x=np.asarray(x,dtype=np.float64)
    assert x.ndim==2 and x.shape[1]==2 and np.isfinite(x).all()
    ids=np.empty(len(x),dtype=np.int64)
    for start in range(0,len(x),4096):
        d=x[start:start+4096,None,:]-CENTERS[None,:,:]
        ids[start:start+4096]=np.argmin(np.einsum('nkd,nkd->nk',d,d),axis=1)
    return ids,(x-CENTERS[ids])/SIGMA

def moments(x,ids):
    n=np.bincount(ids,minlength=K)
    den=np.maximum(n,1)
    m=np.stack([np.bincount(ids,weights=x[:,j],minlength=K)/den for j in range(x.shape[1])],axis=1)
    if x.shape[1]!=2:
        return n,m
    second=np.empty((K,2,2),dtype=np.float64)
    for j in range(2):
        for k in range(2):
            second[:,j,k]=np.bincount(ids,weights=x[:,j]*x[:,k],minlength=K)/den
    return n,m,second-m[:,:,None]*m[:,None,:]

def cross_cov(a,b,ids,am,bm,n):
    ac=a-am[ids]; bc=b-bm[ids]
    cross=np.empty((K,2,2))
    for j in range(2):
        for k in range(2):
            cross[:,j,k]=np.bincount(ids,weights=ac[:,j]*bc[:,k],minlength=K)/np.maximum(n,1)
    return cross+cross.transpose(0,2,1)

def spatial(rows):
    out={}
    for name,mask in (('corners',CORNER),('other_boundary',BOUNDARY&~CORNER),('interior',~BOUNDARY)):
        selected=[rows[i] for i in np.flatnonzero(mask)]
        out[name]=dict(modes=len(selected), max_eig=stats([v['max_eig_sigma2'] for v in selected]),
                       tail_fraction=stats([v['tail_fraction'] for v in selected]))
    return out

def paired(clean,noisy):
    clean=np.asarray(clean,dtype=np.float64); noisy=np.asarray(noisy,dtype=np.float64)
    ids, nr=assign(noisy); cid,cr_own=assign(clean)
    c=(clean-CENTERS[ids])/SIGMA; e=(noisy-clean)/SIGMA
    radius=np.linalg.norm(nr,axis=1); hq=radius<=3
    result={}
    per_mode=None
    for name,mask in (('all',np.ones(len(noisy),dtype=bool)),('same_noisy_HQ',hq)):
        ii=ids[mask]; cc=c[mask]; ee=e[mask]; rr=nr[mask]
        n,cm,cv=moments(cc,ii); _,em,ev=moments(ee,ii); _,nm,nv=moments(rr,ii)
        cross=cross_cov(cc,ee,ii,cm,em,n)
        err=float(np.max(np.abs(cv+ev+cross-nv)))
        assert err<1e-10 and np.max(np.abs(cm+em-nm))<1e-10
        vals,vecs=np.linalg.eigh(nv)
        axes=vecs[:,:,1]
        project=lambda cov: np.einsum('ki,kij,kj->k',axes,cov,axes)
        cp,ep,xp=project(cv),project(ev),project(cross)
        rows=[]
        for i in range(K):
            local=ids==i
            row=dict(mode=i,center=CENTERS[i].tolist(), n=int(n[i]),
                     max_eig_sigma2=float(vals[i,1]),min_eig_sigma2=float(vals[i,0]),
                     max_eig_axis=axes[i].tolist(), angle_degrees=float(math.degrees(math.atan2(axes[i,1],axes[i,0]))%180),
                     max_axis_clean=float(cp[i]),max_axis_output_noise=float(ep[i]),max_axis_cross=float(xp[i]),
                     covariance_noisy=nv[i].tolist(),covariance_clean=cv[i].tolist(),
                     covariance_output_noise=ev[i].tolist(),covariance_cross=cross[i].tolist(),
                     tail_fraction=float(np.mean(~hq[local])),noisy_mean_sigma=nm[i].tolist(),
                     clean_mean_sigma=cm[i].tolist(),output_noise_mean_sigma=em[i].tolist())
            rows.append(row)
        result[name]=dict(identity_max_abs_error=err,per_mode=rows,
             top10_max_covariance=[rows[i] for i in np.argsort(vals[:,1],kind='stable')[::-1][:10]],
             modes_over_original1_7=[int(i) for i in np.flatnonzero(vals[:,1]>1.7)],spatial=spatial(rows),
             mean_covariance_trace_per_axis=dict(noisy=float(np.trace(nv,axis1=1,axis2=2).mean()/2),
                 clean=float(np.trace(cv,axis1=1,axis2=2).mean()/2),noise=float(np.trace(ev,axis1=1,axis2=2).mean()/2),
                 cross=float(np.trace(cross,axis1=1,axis2=2).mean()/2)))
        if name=='same_noisy_HQ': per_mode=rows
    energy={}
    for name,mask in (('all',np.ones(len(noisy),dtype=bool)),('HQ',hq),('tail_over3',~hq)):
        cc,ee,rr=c[mask],e[mask],nr[mask]
        a=float(np.mean(np.sum(cc*cc,1))); b=float(np.mean(np.sum(ee*ee,1)))
        cross=float(2*np.mean(np.sum(cc*ee,1))); total=float(np.mean(np.sum(rr*rr,1)))
        assert abs(a+b+cross-total)<1e-10
        energy[name]=dict(n=int(mask.sum()),clean_radius_energy=a,noise_radius_energy=b,cross=cross,noisy_radius_energy=total,
             clean_component_already_over3_fraction=float(np.mean(np.linalg.norm(cc,axis=1)>3)),
             noise_component_already_over3_fraction=float(np.mean(np.linalg.norm(ee,axis=1)>3)))
    _,_,clean_cov=moments(cr_own[cid>=0],cid[cid>=0])
    result.update(n=len(noisy),precision=float(hq.mean()), mode_switches=int(np.sum(ids!=cid)),
        saved_output_noise_global_mean_sigma=e.mean(0).tolist(),
        saved_output_noise_global_covariance_sigma2=np.cov(e,rowvar=False,bias=True).tolist(),
        saved_output_noise_global_variance_per_axis_sigma2=float(np.var(e,axis=0).mean()),
        clean_unconditional_max_eigenvalues=np.linalg.eigvalsh(clean_cov)[:,1].tolist(),radial_energy=energy)
    return result

def anchor_geometry(points,graph):
    ids,r=assign(points); hq=np.linalg.norm(r,axis=1)<=3
    n,m,c=moments(r,ids); hn,hm,hc=moments(r[hq],ids[hq]); vals=np.linalg.eigvalsh(c)
    degree=(graph>=0).sum(1)
    parent=np.arange(len(points),dtype=np.int64)
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=int(parent[i])
        return i
    edges=[]
    for i,neighbors in enumerate(graph):
        for j in neighbors:
            j=int(j)
            if j>i:
                a,b=find(i),find(j)
                if a!=b:parent[b]=a
                edges.append((i,j))
    component=np.array([find(i) for i in range(len(points))])
    rows=[]
    for i in range(K):
        mask=ids==i; _,sizes=np.unique(component[mask],return_counts=True)
        rows.append(dict(mode=i,rows=int(n[i]),anchor_tail_fraction=float(np.mean(~hq[mask])),
                         max_eig_sigma2=float(vals[i,1]),min_eig_sigma2=float(vals[i,0]),
                         covariance_sigma2=c[i].tolist(),conditional_covariance_sigma2=hc[i].tolist(),
                         mean_radius_sigma=float(np.linalg.norm(m[i])),linked_fraction=float(np.mean(degree[mask]>0)),
                         mean_degree=float(degree[mask].mean()),components_in_mode=len(sizes),
                         largest_known_component_fraction=float(sizes.max()/n[i]),
                         component_HHI=float(np.sum((sizes/n[i])**2))))
    e=np.asarray(edges,dtype=np.int64).reshape(-1,2)
    _,sizes=np.unique(component,return_counts=True)
    return dict(rows=len(points),precision_descriptive=float(hq.mean()),per_mode=rows,
                known_copy_graph=dict(undirected_edges=len(e),components=len(sizes),largest_component=int(sizes.max()),
                     degree=stats(degree),cross_oracle_mode_edges=int(np.sum(ids[e[:,0]]!=ids[e[:,1]])),
                     cross_oracle_mode_fraction=float(np.mean(ids[e[:,0]]!=ids[e[:,1]])) if len(e) else None,
                     complete_genealogy=False))

@torch.no_grad()
def head_features(x,weights):
    # Frozen native discriminator: Fourier append and three LeakyReLU(.2) blocks.
    chunks=[]
    for block in x.split(256):
        xf=block.unsqueeze(-1)*weights['freqs']
        h=torch.cat((block,torch.sin(xf).flatten(1),torch.cos(xf).flatten(1)),dim=1)
        for layer in (0,2,4):
            h=F.leaky_relu(F.linear(h,weights[f'net.{layer}.weight'],weights[f'net.{layer}.bias']),.2)
        chunks.append(h)
    return torch.cat(chunks)

def feature_aliasing(state,anchors):
    bd=state['birth_death']; fifo=bd['reservoir']; weights=state['models']['D']
    real=head_features(fifo,weights).double(); ref=real[0::2]
    mean=ref.mean(0); spread=ref.std(0,unbiased=True)
    varying=spread>spread.max()*1e-8
    scale=torch.where(varying,spread,torch.full_like(spread,float('inf')))
    xr=((ref-mean)/scale).numpy(); ids,_=assign(fifo[0::2].numpy())
    n,centroids=moments(xr,ids)
    assert (n>0).all()
    radius=np.linalg.norm(xr-centroids[ids],axis=1)
    _,within=moments((radius**2)[:,None],ids)
    d=centroids[:,None,:]-centroids[None,:,:]
    d2=np.einsum('ijd,ijd->ij',d,d); np.fill_diagonal(d2,np.inf)
    nearest=np.argmin(d2,axis=1)
    separation=np.sqrt(d2[np.arange(K),nearest])/np.sqrt(within[:,0])
    pairs=[]
    for i in np.argsort(separation,kind='stable')[:12]:
        j=int(nearest[i]); pairs.append(dict(mode=int(i),other_mode=j,centers=[CENTERS[i].tolist(),CENTERS[j].tolist()],
            centroid_separation_over_own_rms=float(separation[i]),
            oracle_centers_distance=float(np.linalg.norm(CENTERS[i]-CENTERS[j]))))
    out=dict(metric='even_FIFO_standardized_current_D_head_input_before_projection',
             historical_snapshot_reconstructed=False,stored_mass_topology=bd['last']['mass_topology'],
             stored_paired_average=bd['paired_average'], fitted_reference_mode_counts=n.tolist(),
             nearest_other_centroid_separation_over_own_rms=stats(separation),most_overlapping12=pairs,
             historical_chart_missing=['basis','centers','cell_scale','cell_group_mapping'])
    for model,points in anchors.items():
        f=head_features(points,state['models']['D']).double()
        q=((f-mean)/scale).numpy(); own,_=assign(points.numpy())
        chosen=[]
        for block in np.array_split(q,math.ceil(len(q)/256)):
            dist=np.maximum(np.sum(block*block,1)[:,None]+np.sum(centroids*centroids,1)[None,:]-2*block@centroids.T,0)
            chosen.append(np.argmin(dist,axis=1))
        chosen=np.concatenate(chosen); wrong=chosen!=own
        confusion=np.bincount(own[wrong]*K+chosen[wrong],minlength=K*K)
        pairs2=[dict(oracle_mode=int(j//K),nearest_real_feature_centroid_mode=int(j%K),rows=int(confusion[j]))
                for j in np.argsort(confusion,kind='stable')[::-1][:10] if confusion[j]>0]
        out[model]=dict(rows=len(points),nearest_reference_centroid_matches_oracle_mode_fraction=float(np.mean(~wrong)),
                       mismatched_rows=int(wrong.sum()),top10_confusion=pairs2)
    return out

def main():
    prep=read(HERE/'PREPARATION-FROZEN.json')
    before=prep['source_and_input_sha256']
    for p,digest in before.items():assert sha(p)==digest,p
    rng=torch.get_rng_state().clone();assert not torch.cuda.is_initialized()
    result=read(RUN/'result.json'); execution=read(RUN/'execution-receipt.json')
    assert result['status']=='FAIL' and result['completed_steps']==7000
    assert sha(RUN/'result.json')==execution['result_sha256']
    assert execution['source_integrity_before']==execution['source_integrity_after']
    assert execution['source_integrity_after']['status']=='VALID'
    metrics={v['step']:v for v in (json.loads(line) for line in (RUN/'metrics.jsonl').read_text().splitlines())}
    verdict=read(RUN/'native-noisy/verdict.json')
    failed=lambda value: [dict(metric=k,value=value[k],operator=op,bound=bound) for k,op,bound in result['thresholds']
         if k in value and not(value[k]>=bound if op=='>=' else value[k]<=bound)]
    terminal=[dict(step=v['step'],original_passed=v['passed'],failed_thresholds=failed(metrics[v['step']]))
              for v in verdict['accuracy']['terminal_checks']]
    assert len(terminal)==5 and not any(v['original_passed'] for v in terminal)
    state=torch.load(RUN/'final-state.pt',map_location='cpu',weights_only=False)['trainer']
    assert state['completed_steps']==7000 and state['recipe']['num_particles']==20000
    assert state['birth_death']['backend_schema']==7
    models=state['models'];anchors={}
    for name,model,prior in (('live','G','prior'),('ema','ema_G','ema_prior')):
        anchors[name]=F.linear(models[prior]['z'],models[model]['weight'],models[model]['bias'])
    graph=state['birth_death']['lineage_neighbors'].numpy()
    anchor_stats={name:anchor_geometry(x.numpy(),graph) for name,x in anchors.items()}
    pair_results={}
    for split,fname,count in (('terminal20k','final_samples.npz',20000),('holdout100k','holdout_samples.npz',100000)):
        with np.load(RUN/'native-clean'/fname,allow_pickle=False) as clean,np.load(RUN/'native-noisy'/fname,allow_pickle=False) as noisy:
            assert set(clean.files)==set(noisy.files)=={'live','ema','target'}
            assert np.array_equal(clean['target'],noisy['target'])
            for name in ('live','ema'):
                c,n=clean[name],noisy[name]
                assert c.shape==n.shape==(count,2)
                pair_results[f'{split}/{name}']=paired(c,n)
                # Exact pair component distributions, no new perturbation draws.
    final_top=pair_results['terminal20k/live']['same_noisy_HQ']['top10_max_covariance'][0]
    official_max_delta=float(final_top['max_eig_sigma2']-result['final']['max_cov_eig_ratio'])
    # Cross links and rows are diagnostic associations only; full genealogy was not saved.
    mode_comparisons=[]
    for i in range(K):
        row=dict(mode=i,center=CENTERS[i].tolist())
        for name in ('live','ema'):
            a=anchor_stats[name]['per_mode'][i]
            p=pair_results[f'holdout100k/{name}']['same_noisy_HQ']['per_mode'][i]
            clean=pair_results[f'holdout100k/{name}']['clean_unconditional_max_eigenvalues'][i]
            row[name]=dict(anchor_max_eig_sigma2=a['max_eig_sigma2'],clean_cloud_max_eig_sigma2=clean,
                          noisy_HQ_max_eig_sigma2=p['max_eig_sigma2'],anchor_tail_fraction=a['anchor_tail_fraction'],
                          noisy_tail_fraction=p['tail_fraction'],largest_known_component_fraction=a['largest_known_component_fraction'],
                          mean_known_copy_degree=a['mean_degree'],component_HHI=a['component_HHI'])
        mode_comparisons.append(row)
    fa=feature_aliasing(state,anchors)
    live_ids,_=assign(anchors['live'].numpy());ema_ids,_=assign(anchors['ema'].numpy())
    for p,digest in before.items():assert sha(p)==digest,p
    assert torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
    out=dict(status='VALID',scope='Fixed saved-only diagnostic; no candidate or gate changes',
        quality=dict(original_status='FAIL',evidence_source_integrity='VALID',terminal=terminal,
                     final=result['final'],holdout=result['native']['holdout'],
                     all5terminal_failed=True,accuracy_holdout_PASS_but_frozen_holdout_FAIL=True),
        paired_covariance=pair_results,anchors=anchor_stats,mode_covariance_lineage_comparisons=mode_comparisons,
        diagnostic_float64_minus_authoritative_GPU_float32_max_cov=official_max_delta,
        feature_aliasing=fa,FAST_EMA_raw_anchor_oracle_mode_agreement_fraction=float(np.mean(live_ids==ema_ids)),
        final_reaction=dict(last=state['birth_death']['last'],counters=state['birth_death']['counters']),
        cpu_only=True,cuda_initialized=False,global_CPU_RNG_unchanged=True,
        no_draws=True,no_training=True,no_new_quality_emission=True,
        source_and_input_sha256=before)
    (HERE/'result.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='VALID',top_mode=final_top['mode'],max_cov=final_top['max_eig_sigma2'],
        clean=final_top['max_axis_clean'],output_noise=final_top['max_axis_output_noise'],cross=final_top['max_axis_cross'],
        source_and_input_files=len(before),result_sha256=sha(HERE/'result.json'))))

if __name__=='__main__':main()
