"""Deterministic novel-latent birth candidates at observed real feature anchors.

Copy-parent eligibility remains separate. A seed is a starting coordinate,
not a supported-copy certificate. All acceptance uses the frozen head law.
"""
import math
import time
import torch


def latent_to_anchor(snapshot, feature_of_latent, seed_latent, anchor_cell, prior_latents, *, passes=4):
    """At most four low-rank Newton linearizations; no model/stream update.

    Each step is the numerically truncated least-norm Jacobian solution. The
    trust radius is one current prior RMS coordinate spread, derived without
    labels or a step-size fit. Acceptance retains the existing cell, even-fit
    inside boundary and support p>Q checks. This proposes a new point.
    """
    if (type(passes) is not int or passes<=0 or passes>4 or seed_latent.ndim!=1
            or prior_latents.ndim!=2 or prior_latents.shape[1]!=len(seed_latent)
            or not snapshot.valid_metric):
        raise ValueError('invalid real-anchor latent birth inputs')
    if not 0<=anchor_cell<snapshot.cells:
        raise ValueError('invalid anchor cell')
    start=time.perf_counter()
    seed=seed_latent.detach().clone()
    latent=seed.clone()
    target=snapshot.real_representatives[anchor_cell].detach()
    trust=prior_latents.detach().std(0,unbiased=False).double().square().mean().sqrt()
    progress=[];accepted=False
    for iteration in range(passes+1):
        with torch.no_grad():
            features=feature_of_latent(latent[None]).detach().double()
            projected=snapshot.transform(features)[0]
            flags,pvalues,scores=snapshot.support(features)
            categories=snapshot.count_categories(features)
            residual=(target-projected).norm()
            accepted=bool(torch.isfinite(latent).all() and not flags[0] and pvalues[0]>.05
                and categories[0]==2*anchor_cell)
            progress.append(dict(iteration=iteration,residual=float(residual),score=float(scores[0]),
                support_pvalue=float(pvalues[0]),category=int(categories[0]),accepted=accepted))
        if accepted or iteration==passes or not math.isfinite(float(trust)) or not trust>0:break
        with torch.enable_grad():
            def projected_features(z):
                f=feature_of_latent(z[None]).double()[0]
                return ((f-snapshot.mean)/snapshot.scale)@snapshot.basis
            jacobian=torch.autograd.functional.jacobian(projected_features,latent.detach().requires_grad_(True),
                create_graph=False,vectorize=True).detach().double()
        if not bool(torch.isfinite(jacobian).all()):break
        u,s,vh=torch.linalg.svd(jacobian,full_matrices=False)
        # The callback differentiates the model's original precision. A
        # double SVD cannot recover rank below that source arithmetic noise.
        numerical=torch.finfo(seed.dtype).eps*max(jacobian.shape)*s.max()
        active=s>numerical
        if not bool(active.any()):break
        delta=(vh[active].T@((u[:,active].T@(target-projected))/s[active]))
        delta*=torch.minimum(delta.new_tensor(1.),trust/delta.norm().clamp_min(torch.finfo(delta.dtype).tiny))
        latent=(latent.double()+delta).to(seed.dtype).detach()
    return dict(kind='new_latent_birth',seed_latent=seed,latent=latent,features=features,
        cell=anchor_cell,category=int(categories[0]),support_pvalue=float(pvalues[0]),accepted=accepted,
        trust_radius=float(trust),total_latent_displacement=float((latent-seed).double().norm()),
        linearizations=len(progress)-1,progress=progress,seconds=time.perf_counter()-start)


def propose_inaccessible_anchors(snapshot, query_features, flags, pvalues, latents, feature_of_latent, *,
        ema_latents=None, ema_feature_of_latent=None, reserved_rows=None, supported_counts=None,
        candidate_limit=4, passes=4):
    """Bounded representative-cell proposals; oracle labels are never inputs."""
    if candidate_limit<0 or candidate_limit>4:
        raise ValueError('candidate limit exceeds the fixed existing four-anchor bound')
    n=len(query_features)
    if len(latents)!=n or flags.shape!=(n,) or pvalues.shape!=(n,):raise ValueError('invalid table')
    if (ema_latents is None)!=(ema_feature_of_latent is None):raise ValueError('paired EMA arguments required')
    ids,_=snapshot.assign(query_features);categories=snapshot.count_categories(query_features)
    reserved=torch.zeros(n,dtype=torch.bool,device=ids.device)
    if reserved_rows is not None:reserved[reserved_rows]=True
    eligible=(~flags)&(pvalues>.05)&(categories.remainder(2)==0)&~reserved
    pool_counts=torch.bincount(ids[eligible],minlength=snapshot.cells)
    supported=torch.bincount(ids[~flags],minlength=snapshot.cells) if supported_counts is None else supported_counts.clone()
    target=snapshot._mass_targets(n);vacancies=(target-supported).clamp_min(0)
    groups=snapshot._mass_topology();group_vacancies=(snapshot._group_counts(target)-snapshot._group_counts(supported)).clamp_min(0)
    inaccessible=(pool_counts==0)&(vacancies>0)&(group_vacancies[groups]>0)&(snapshot.reference_counts>0)
    order=vacancies.masked_fill(~inaccessible,-1).argsort(descending=True,stable=True)
    order=order[inaccessible[order]][:candidate_limit]
    projected=snapshot.transform(query_features)
    attempts=[]
    for cell_tensor in order:
        cell=int(cell_tensor)
        distance=(projected-snapshot.real_representatives[cell]).square().sum(1).masked_fill(reserved,float('inf'))
        source=int(distance.argmin())
        if not math.isfinite(float(distance[source])):break
        reserved[source]=True
        current=latent_to_anchor(snapshot,feature_of_latent,latents[source],cell,latents,passes=passes)
        average=None
        if ema_latents is not None:
            average=latent_to_anchor(snapshot,ema_feature_of_latent,ema_latents[source],cell,ema_latents,passes=passes)
        attempts.append(dict(kind='new_latent_birth',cell=cell,seed_row=source,
            seed_copy_eligible=bool((~flags[source])&(pvalues[source]>.05)),
            initial_cell=int(ids[source]),initial_category=int(categories[source]),
            vacancy=int(vacancies[cell]),group_vacancy=int(group_vacancies[groups[cell]]),
            current=current,average=average,accepted=current['accepted'] and (average is None or average['accepted'])))
    return attempts
