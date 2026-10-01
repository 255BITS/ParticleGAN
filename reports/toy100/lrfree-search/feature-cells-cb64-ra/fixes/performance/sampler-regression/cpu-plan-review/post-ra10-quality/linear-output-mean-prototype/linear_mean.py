"""Reserved scratch linear-output mean phase; never a backend9 checkpoint law.

All support predicates/cache entries remain critic features. Only bounded mean
geometry consumes even-real fitted raw coordinate projection. No new RNG draw
fits the frame; old paired noise/packet/row-state equations stay unchanged.
"""
from dataclasses import dataclass,replace
import math
import torch
from mean_owner import mean_transport as base

Q=.05
POLICY='even_linear_output_group_clip_unit_residual_EB_3K_plus_3_scratch_v1'
PROJECTION_POLICY='even_streamed_varying_axes_if_d_le8_else_top8_marginal_variance_stable_v1'
FIT_CHUNK=256
MAX_RANK=8
POOL=base.POOL
group_means=base.group_means
neutral_observation=base.neutral_observation
CandidatePairs=base.CandidatePairs
epoch=base.epoch
tensor_digest=base.tensor_digest

@dataclass(frozen=True)
class OutputProjection:
    output_dim:int
    axes:torch.Tensor
    fitted_rows:int
    even_mean:torch.Tensor
    even_variance:torch.Tensor
    policy:str=PROJECTION_POLICY
    @property
    def rank(self):return len(self.axes)
    def transform(self,raw):
        if raw.ndim<2 or math.prod(raw.shape[1:])!=self.output_dim:
            raise ValueError('raw output width differs from fixed even projection')
        return raw.flatten(1)[:,self.axes.to(raw.device)].double().cpu()

@torch.no_grad()
def fit_projection(even_raw):
    if even_raw.ndim<2 or len(even_raw)<2:
        return None,'insufficient_even_output_rows'
    width=math.prod(even_raw.shape[1:])
    if width<1:return None,'empty_output_width'
    # Two stable streamed passes, no d*d covariance or fit RNG. O(M*d).
    mean=torch.zeros(width,dtype=torch.float64)
    count=0
    for block in even_raw.split(FIT_CHUNK):
        flat=block.flatten(1).double().cpu()
        if not bool(torch.isfinite(flat).all()):return None,'nonfinite_even_outputs'
        next_count=count+len(flat)
        mean+=(flat.mean(0)-mean)*(len(flat)/next_count)
        count=next_count
    ss=torch.zeros_like(mean)
    for block in even_raw.split(FIT_CHUNK):
        centered=block.flatten(1).double().cpu()-mean
        ss+=centered.square().sum(0)
    variance=ss/count
    if not bool(torch.isfinite(variance).all()):return None,'nonfinite_even_output_variance'
    varying=(variance>0).nonzero().flatten()
    if not len(varying):return None,'zero_output_moment_rank'
    if width<=MAX_RANK:axes=varying  # Original raw coordinate order, no RNG.
    else:axes=varying[variance[varying].argsort(descending=True,stable=True)[:MAX_RANK]]
    return OutputProjection(width,axes,count,mean,variance),None

@dataclass(frozen=True)
class FixedOutputMoment:
    projection:OutputProjection
    centers:torch.Tensor
    scales:torch.Tensor
    even_means:torch.Tensor
    ema_means:torch.Tensor
    ema_counts:torch.Tensor
    weights:torch.Tensor
    directions:torch.Tensor
    radius:float
    moment_rank:int
    chart_rank:int
    cells:int
    def psi(self,projected_outputs,groups):
        if projected_outputs.ndim!=2 or projected_outputs.shape[1]!=self.moment_rank:
            raise ValueError('psi requires output metric, never critic metric')
        z=(projected_outputs.double()-self.centers[groups])/self.scales[groups,None]
        fraction=(self.radius/z.norm(dim=1).clamp_min(1e-30)).clamp_max(1.)
        return z*fraction[:,None]
    def energy(self,means=None):
        means=self.ema_means if means is None else means
        return (self.weights*(self.even_means-means).square().sum(1)).sum()

@dataclass(frozen=True)
class OutputObservation:
    features:torch.Tensor
    projected_outputs:torch.Tensor

@dataclass(frozen=True)
class MomentView:
    features:object
    metric:torch.Tensor  # Learned support metric, never raw output phi.
    cells:torch.Tensor
    categories:torch.Tensor
    groups:torch.Tensor
    eligible:torch.Tensor
    pvalues:torch.Tensor
    moment_metric:object

@torch.no_grad()
def observe_moment_view(snapshot,features,projected_outputs,*,coordinates,cached_fast=False):
    chart=base.observe_view(snapshot,features,coordinates=coordinates,cached_fast=cached_fast)
    return MomentView(**vars(chart),moment_metric=projected_outputs)

@torch.no_grad()
def capture_outputs_features(backend,trainer,model,latents,projection):
    """One G pass per chunk; same raw output yields critic and output metrics."""
    roots=(trainer.G,trainer.ema_G,trainer.D)
    streams=(backend.stream,*[getattr(trainer,n) for n in trainer._STREAMS])
    devices=[trainer.device.index] if trainer.device.type=='cuda' else []
    features,projected,finite=[],[],[]
    with neutral_observation(roots,streams,devices):
        for block in latents.detach().split(backend.settings['chunk']):
            raw=model(block)
            if tuple(raw.shape[1:])!=backend.sample_shape:raise ValueError('raw output sample shape mismatch')
            head=backend._features(trainer,raw,chunk=backend.settings['chunk'])
            finite.extend((torch.isfinite(raw).flatten(1).all(1),torch.isfinite(head).all(1)))
            features.append(head);projected.append(projection.transform(raw))
    if not bool(torch.cat(finite).all()):return None
    return OutputObservation(torch.cat(features),torch.cat(projected))

@torch.no_grad()
def freeze_moment(snapshot,even_features,even_raw,ema_observation,projection):
    """No odd data or count-driven actions enter the fixed output geometry."""
    if not snapshot.valid_metric or snapshot.rank<=0 or snapshot.duplicate_fraction>Q:
        return None,'invalid_or_duplicate_chart'
    if projection is None or not 0<projection.rank<=MAX_RANK:return None,'invalid_output_projection'
    topology=snapshot._mass_topology()
    chart=snapshot.transform(even_features)
    even_groups=topology[snapshot._assign_metric(chart)[0]].cpu()
    metric=projection.transform(even_raw)
    centers,counts=group_means(metric,even_groups,snapshot.mass_groups)
    if bool((counts<2).any()):return None,'missing_or_insufficient_even_group'
    squared=(metric-centers[even_groups]).square().sum(1)
    ss=torch.zeros(snapshot.mass_groups,dtype=torch.float64)
    ss.index_add_(0,even_groups,squared)
    scales=(ss/counts/projection.rank).sqrt()
    if not bool(torch.isfinite(scales).all()&(scales>0).all()):return None,'nonfinite_or_zero_even_scale'
    radius=math.sqrt(projection.rank/Q)
    def psi(values,groups):
        z=(values-centers[groups])/scales[groups,None]
        return z*(radius/z.norm(dim=1).clamp_min(1e-30)).clamp_max(1.)[:,None]
    even_means,_=group_means(psi(metric,even_groups),even_groups,snapshot.mass_groups)
    ema_chart=snapshot.transform(ema_observation.features)
    ema_groups=topology[snapshot._assign_metric(ema_chart)[0]].cpu()
    ema_means,ema_counts=group_means(psi(ema_observation.projected_outputs,ema_groups),ema_groups,snapshot.mass_groups)
    if bool((ema_counts==0).any()):return None,'missing_EMA_group'
    residual=even_means-ema_means;lengths=residual.norm(dim=1)
    directions=torch.where(lengths[:,None]>0,residual/lengths.clamp_min(1e-30)[:,None],torch.zeros_like(residual))
    if not bool(torch.isfinite(directions).all()&torch.isfinite(ema_means).all()):return None,'nonfinite_frozen_direction'
    return FixedOutputMoment(projection,centers,scales,even_means,ema_means,ema_counts,
        counts.double()/int(counts.sum()),directions,radius,projection.rank,snapshot.rank,snapshot.cells),None

@torch.no_grad()
def odd_witness(snapshot,fixed,odd_features,odd_raw):
    alpha=Q/(3*snapshot.cells+3)
    result=dict(policy=POLICY,alpha=alpha,multiplicity=3*snapshot.cells+3,observations=len(odd_raw),
        authoritative=False,limit='conditional iid bounded-score algebra; trained shared chart/FIFO empirical evidence only')
    if fixed is None:return dict(result,valid=False,fires=False,reason='no_valid_frozen_moment')
    metric=fixed.projection.transform(odd_raw)
    if len(metric)<=1 or not bool(torch.isfinite(metric).all()):
        return dict(result,valid=False,fires=False,reason='insufficient_or_nonfinite_odd_outputs')
    chart=snapshot.transform(odd_features)
    if not bool(torch.isfinite(chart).all()):return dict(result,valid=False,fires=False,reason='nonfinite_odd_features')
    groups=snapshot._mass_topology()[snapshot._assign_metric(chart)[0]].cpu()
    psi=fixed.psi(metric,groups)
    values=(fixed.directions[groups]*(psi-fixed.ema_means[groups])).sum(1)
    if not bool(torch.isfinite(values).all()):return dict(result,valid=False,fires=False,reason='nonfinite_odd_scalar')
    known_range=4*fixed.radius;variance=values.var(unbiased=True);mean=values.mean();t=math.log(2/alpha)
    variance_penalty=(2*variance*t/len(values)).sqrt()
    range_penalty=(7/3)*known_range*t/(len(values)-1)
    lcb=mean-variance_penalty-range_penalty
    assert bool((values.abs()<=2*fixed.radius+1e-10).all())
    return dict(result,valid=True,fires=bool(lcb>0),reason=None,mean=float(mean),variance_ddof1=float(variance),
        radius=fixed.radius,known_range=known_range,variance_penalty=float(variance_penalty),range_penalty=float(range_penalty),
        lower_bound=float(lcb),zero_direction_observations=int((fixed.directions[groups].norm(dim=1)==0).sum()),
        scalar_min=float(values.min()),scalar_max=float(values.max()),even_mass_objective=float(fixed.energy()))

@dataclass(frozen=True)
class OutputCopyPacket(base.CopyPacket):
    fast_output_metric:torch.Tensor
    ema_output_metric:torch.Tensor
    def content_digest(self):
        return tensor_digest(self.children,self.parents,self.fast_coordinates,self.ema_coordinates,self.parent_row_state,
            self.parent_history,self.fast_features,self.ema_features,self.fast_output_metric,self.ema_output_metric)

@torch.no_grad()
def commit_packet(packet,state,snapshot):
    if (not isinstance(packet,OutputCopyPacket) or packet.fast_output_metric.ndim!=2
            or packet.ema_output_metric.shape!=packet.fast_output_metric.shape
            or packet.fast_output_metric.shape[0]!=len(packet.children)
            or not 0<packet.fast_output_metric.shape[1]<=MAX_RANK
            or packet.fast_output_metric.dtype!=torch.float64 or packet.ema_output_metric.dtype!=torch.float64
            or packet.fast_output_metric.device.type!='cpu' or packet.ema_output_metric.device.type!='cpu'
            or not bool(torch.isfinite(packet.fast_output_metric).all()&torch.isfinite(packet.ema_output_metric).all())):
        raise ValueError('prepared output-metric packet layout mismatch')
    return base.commit_packet(packet,state,snapshot)

@torch.no_grad()
def propose_pairs(snapshot, fixed, fast, ema, *, earlier_ordinary, reserved_rows):
    """Stable bounded pools, no pair matrix, unique disjoint source incarnations."""
    n = len(fast.metric)
    if type(earlier_ordinary) is not int or not 0 <= earlier_ordinary <= math.floor(Q * n):
        raise ValueError("invalid shared ordinary prefix")
    if (reserved_rows.ndim != 1 or reserved_rows.dtype != torch.long
            or bool(((reserved_rows < 0) | (reserved_rows >= n)).any())):
        raise ValueError("invalid complete reaction reservation union")
    budget = math.floor(Q * n) - earlier_ordinary
    empty = torch.empty(0, dtype=torch.long, device=fast.metric.device)
    if fixed is None:
        return CandidatePairs(empty, empty, torch.empty(0, dtype=torch.float64), 0, 0, empty,
                              budget, earlier_ordinary)
    reserved = torch.zeros(n, dtype=torch.bool, device=fast.metric.device)
    reserved[reserved_rows] = True
    legal = fast.eligible & ema.eligible & (fast.groups == ema.groups)
    protected = torch.zeros_like(legal)
    # Preserve an actual jointly eligible survivor in each occupied cell/view.
    for view in (fast, ema):
        for cell in range(snapshot.cells):
            rows = (legal & (view.cells == cell)).nonzero().flatten()
            if len(rows):
                protected[rows[0]] = True
    legal &= ~reserved
    psi = fixed.psi(ema.moment_metric, ema.groups)
    projection = (fixed.directions[ema.groups] * psi).sum(1)
    signature = fast.categories * (2 * snapshot.cells) + ema.categories
    rows = legal.nonzero().flatten()
    order = rows[signature[rows].argsort(stable=True)]
    _, counts = torch.unique_consecutive(signature[order], return_counts=True)
    all_children, all_parents, all_gain = [], [], []
    start = 0
    for size in counts.cpu().tolist():
        bucket = order[start:start + size]
        start += size
        ranked = bucket[projection[bucket].argsort(descending=True, stable=True)]
        parent_count = min(POOL, size // 2)
        parents = ranked[:parent_count]
        candidates = ranked[parent_count:]
        candidates = candidates[~protected[candidates]]
        children = candidates[projection[candidates].argsort(stable=True)[:POOL]]
        count = min(len(children), len(parents))
        if not count:
            continue
        children, parents = children[:count], parents[:count]
        groups = ema.groups[children]
        delta = (psi[parents] - psi[children]) / fixed.ema_counts[groups, None]
        residual = fixed.even_means[groups] - fixed.ema_means[groups]
        gain = fixed.weights[groups] * (2 * (residual * delta).sum(1) - delta.square().sum(1))
        keep = torch.isfinite(gain) & (gain > 0)
        all_children.append(children[keep]); all_parents.append(parents[keep]); all_gain.append(gain[keep])
    children = torch.cat(all_children) if all_children else empty
    parents = torch.cat(all_parents) if all_parents else empty
    gains = torch.cat(all_gain) if all_gain else torch.empty(0, dtype=torch.float64, device=empty.device)
    # Stable ties are parent then child IDs; last stable gain sort is primary.
    ordered = parents.argsort(stable=True)
    ordered = ordered[children[ordered].argsort(stable=True)]
    ordered = ordered[gains[ordered].argsort(descending=True, stable=True)]
    total = len(ordered)
    ordered = ordered[:budget]
    children, parents, gains = children[ordered], parents[ordered], gains[ordered]
    assert len(torch.unique(children)) == len(children)
    assert len(torch.unique(parents)) == len(parents)
    assert not bool(torch.isin(children, parents).any())
    assert not bool(reserved[children].any() | reserved[parents].any())
    return CandidatePairs(children, parents, gains, int(legal.sum()), total,
                          protected.nonzero().flatten(), budget, earlier_ordinary)


@torch.no_grad()
def preview_pairs(snapshot, fixed, fast, ema, pairs, state, *, stream, measure_fast,
                  measure_ema, geometry, roots=(), owned_streams=(), devices=()):
    """Exactly one draw; pure virtual objective; accepted coordinates never redrawn."""
    before_draw = epoch(state, snapshot, include_stream=False)
    if stream is not state.stream:
        raise ValueError("packet stream must be the declared owned reaction stream")
    child_host, parent_host = pairs.children, pairs.parents
    row_pairs = torch.stack((child_host,parent_host),1).to(state.prior.z.device)
    child, parent = row_pairs[:,0],row_pairs[:,1]
    if not len(child):
        return None, dict(attempts=0, accepted=0, reason="no_residual_candidate_slots")
    source_fast, source_ema = state.prior.z[parent], state.ema_prior.z[parent]
    noise = torch.randn(source_fast.shape, dtype=source_fast.dtype, device=source_fast.device, generator=stream)
    proposed_fast = source_fast + geometry.displacement(source_fast, state.prior,
                    state.bandwidth, noise, rows=parent)
    proposed_ema = source_ema + geometry.displacement(source_ema, state.ema_prior,
                    state.bandwidth, noise, rows=parent)
    before = epoch(state, snapshot)
    if epoch(state, snapshot, include_stream=False) != before_draw:
        raise RuntimeError("paired jitter changed semantic pre-action state")
    if not bool(torch.isfinite(proposed_fast).all() & torch.isfinite(proposed_ema).all()):
        return None, dict(attempts=len(child), accepted=0, reason="nonfinite_preview_latents")
    with neutral_observation(roots, (*owned_streams, stream), devices):
        fast_observation = measure_fast(proposed_fast)
        ema_observation = measure_ema(proposed_ema)
    if fast_observation is None or ema_observation is None:
        return None, dict(attempts=len(child),accepted=0,reason='nonfinite_preview_observation')
    new_fast_features,new_ema_features = fast_observation.features,ema_observation.features
    if epoch(state, snapshot) != before:
        raise RuntimeError("observation changed the pre-action packet epoch")
    if (not isinstance(new_fast_features,torch.Tensor) or not isinstance(new_ema_features,torch.Tensor)
            or not bool(torch.isfinite(new_fast_features).all() & torch.isfinite(new_ema_features).all())):
        return None, dict(attempts=len(child), accepted=0, reason="nonfinite_preview_features")
    vf = observe_moment_view(snapshot,new_fast_features,fast_observation.projected_outputs,coordinates=proposed_fast)
    ve = observe_moment_view(snapshot,new_ema_features,ema_observation.projected_outputs,coordinates=proposed_ema)
    retained = (vf.eligible & ve.eligible
        & (vf.cells == fast.cells[child_host]) & (ve.cells == ema.cells[child_host])
        & (vf.categories == fast.categories[child_host]) & (ve.categories == ema.categories[child_host])
        & (vf.groups == fast.groups[child_host]) & (ve.groups == ema.groups[child_host])
        & (vf.groups == ve.groups))
    working = fixed.ema_means.clone()
    old_psi = fixed.psi(ema.moment_metric[child_host], ema.groups[child_host])
    new_psi = fixed.psi(ve.moment_metric, ve.groups)
    # Every scalar loop below is CPU float64; no per-pair device reads.
    energy = fixed.energy(working)
    initial = float(energy)
    accepted, gains = [], []
    for index in range(len(child)):
        if not bool(retained[index]):
            continue
        group = int(ema.groups[child_host[index]])
        proposed_mean = working[group] + (new_psi[index] - old_psi[index]) / fixed.ema_counts[group]
        old_group = fixed.weights[group] * (fixed.even_means[group] - working[group]).square().sum()
        new_group = fixed.weights[group] * (fixed.even_means[group] - proposed_mean).square().sum()
        next_energy = energy - old_group + new_group
        if bool(torch.isfinite(next_energy) & (next_energy < energy)):
            gains.append(float(energy - next_energy)); accepted.append(index)
            working[group] = proposed_mean
            energy = next_energy
    detail = dict(attempts=len(child), category_retained=int(retained.sum()), accepted=len(accepted),
        objective_before=initial, objective_after_virtual=float(energy), actual_gains=gains,
        paired_noise_draws=1, noise_rows=len(child), no_redraw=True,
        preview_work=dict(geometry.work))
    if not accepted:
        return None, dict(detail, reason="no_actual_retained_progress")
    indices = torch.tensor(accepted, dtype=torch.long).to(child.device)
    accepted_child, accepted_parent = child[indices].clone(), parent[indices].clone()
    row_state = {key: value[accepted_parent].clone() for key, value in state.row_state.items()
                 if isinstance(value, torch.Tensor) and value.shape == state.prior.z.shape}
    history = None if state.history is None else state.history[accepted_parent].clone()
    values = (accepted_child, accepted_parent, proposed_fast[indices].clone(), proposed_ema[indices].clone(),
              row_state, history, new_fast_features[indices].clone(), new_ema_features[indices].clone())
    output_values=(vf.moment_metric[indices.cpu()].clone(),ve.moment_metric[indices.cpu()].clone())
    packet = OutputCopyPacket(values[0],values[1],values[2],values[3],values[4],values[5],before,
        tensor_digest(*values,*output_values),values[6],values[7],*output_values)
    return packet, dict(detail, reason=None, children=packet.children.tolist(), parents=packet.parents.tolist())



@torch.no_grad()
def prepare_mean_witness(backend,trainer,snapshot,real_features):
    backend.counters['mean_evals']+=1
    fixed=None;reason=None;projection=None
    if not snapshot.valid_metric or snapshot.duplicate_fraction>Q:reason='invalid_or_duplicate_chart'
    elif not bool(torch.isfinite(trainer.ema_prior.z).all()):reason='nonfinite_pre_mean_EMA_coordinates'
    else:
        projection,reason=fit_projection(backend.reservoir[0::2].reshape(-1,*backend.sample_shape))
        if projection is not None:
            observed=capture_outputs_features(backend,trainer,trainer.ema_G,trainer.ema_prior.z,projection)
            backend.counters['mean_forward_rows']+=backend.N
            if observed is None:reason='nonfinite_pre_mean_EMA_observation'
            else:fixed,reason=freeze_moment(snapshot,real_features[0::2],backend.reservoir[0::2],observed,projection)
    witness=(dict(valid=False,fires=False,reason=reason or 'no_valid_frozen_moment') if fixed is None
        else odd_witness(snapshot,fixed,real_features[1::2],backend.reservoir[1::2]))
    valid=witness['valid'];rank=projection.rank if projection is not None else 0
    stamp=dict(schema='scratch_linear_output_v1',policy=POLICY,status='firing' if witness['fires'] else ('veto' if valid else 'invalid'),
        reason=None if witness['fires'] else ('nonpositive_lower_bound' if valid else witness['reason']),
        step=trainer.completed_steps+1,snapshot=backend.snapshot_serial,cells=snapshot.cells,chart_rank=snapshot.rank,
        moment_rank=rank,output_dim=projection.output_dim if projection else None,projection_policy=PROJECTION_POLICY,
        selected_axes=projection.axes.tolist() if projection else [],fitted_rows=projection.fitted_rows if projection else 0,
        observations=snapshot.calibration_rows,alpha=Q/(3*snapshot.cells+3),radius=math.sqrt(rank/Q) if valid else None,
        known_range=4*math.sqrt(rank/Q) if valid else None,
        **{k:witness.get(k) if valid else None for k in ('mean','variance_ddof1','variance_penalty','range_penalty','lower_bound')},
        attempts=0,moves=0,objective_before_mean=None,objective_after_mean=None,
        purpose='scratch-only output objective, incompatible with backend9 checkpoint metadata')
    backend.counters['mean_witness_fires']+=int(stamp['status']=='firing')
    return fixed,stamp

@torch.no_grad()
def run_mean_phase(backend,trainer,snapshot,fixed,stamp,*,earlier_ordinary,reserved_rows):
    empty=torch.empty(0,dtype=torch.long,device=trainer.prior.z.device)
    result=dict(children=empty,parents=empty,invalidated=None,affected=None,diagnostics=dict(stamp),packet=None)
    diag=result['diagnostics'];budget=math.floor(Q*backend.N)-earlier_ordinary
    if budget<0:raise ValueError('earlier ordinary actions exceed shared budget')
    if diag['status']!='firing':return result
    if backend.dry_run or budget==0:
        diag['reason']='dry_run' if backend.dry_run else 'no_remaining_budget';return result
    observed=capture_outputs_features(backend,trainer,trainer.ema_G,trainer.ema_prior.z,fixed.projection)
    backend.counters['mean_forward_rows']+=backend.N
    if observed is None:diag['reason']='nonfinite_current_EMA_observation';return result
    fast=observe_moment_view(snapshot,None,None,coordinates=trainer.prior.z,cached_fast=True)
    ema=observe_moment_view(snapshot,observed.features,observed.projected_outputs,coordinates=trainer.ema_prior.z)
    means,counts=group_means(fixed.psi(ema.moment_metric,ema.groups),ema.groups,snapshot.mass_groups)
    if bool((counts==0).any()):diag['reason']='missing_current_EMA_group';return result
    action=replace(fixed,ema_means=means,ema_counts=counts)
    baseline=float(action.energy());diag.update(objective_before_mean=baseline,objective_after_mean=baseline)
    pairs=propose_pairs(snapshot,action,fast,ema,earlier_ordinary=earlier_ordinary,reserved_rows=torch.unique(reserved_rows).cpu())
    if not len(pairs.children):diag['reason']='no_legal_positive_pairs';return result
    state=base.copy_state(backend,trainer)
    def fast_observation(points):return capture_outputs_features(backend,trainer,trainer.G,points,fixed.projection)
    def ema_observation(points):return capture_outputs_features(backend,trainer,trainer.ema_G,points,fixed.projection)
    packet,detail=preview_pairs(snapshot,action,fast,ema,pairs,state,stream=backend.stream,
        measure_fast=fast_observation,measure_ema=ema_observation,geometry=backend.latent_geometry)
    diag['attempts']=detail['attempts'];backend.counters['mean_preview_rows']+=2*detail['attempts']
    backend.counters['mean_forward_rows']+=2*detail['attempts']
    if packet is None:diag['reason']=detail['reason'];return result
    psi=action.psi(ema.moment_metric,ema.groups)
    child=packet.children.cpu();groups=ema.groups[child]
    psi[child]=action.psi(packet.ema_output_metric,groups)
    actual_means,actual_counts=group_means(psi,ema.groups,snapshot.mass_groups)
    actual_energy=float(action.energy(actual_means))
    if (not torch.equal(actual_counts,counts) or not actual_energy<baseline
            or not math.isclose(actual_energy,detail['objective_after_virtual'],rel_tol=1e-10,abs_tol=1e-12)):
        raise RuntimeError('prepared output objective differs from exact virtual ledger')
    applied=commit_packet(packet,state,snapshot)
    diag.update(moves=len(child),reason=None,objective_after_mean=actual_energy)
    result.update(children=packet.children,parents=packet.parents,packet=packet,invalidated=applied['invalidated'],affected=applied['affected'])
    return result
