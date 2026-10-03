"""Even-real deterministic raw-output frame for bounded conditional moments.

The critic chart defines support/groups; this separate coordinate projection
only defines the mean witness/objective. All frames are derived and ephemeral.
No fit RNG, d*d covariance, Jacobian or retained generated N*d array is used.
"""
from dataclasses import dataclass
import math
import torch
Q=.05
POLICY='even_linear_output_group_radial_clip_unit_residual_EB_3K_plus_3_v1'
PROJECTION_POLICY='even_streamed_varying_axes_if_d_le8_else_top8_marginal_variance_stable_v1'
FIT_CHUNK=256
MAX_RANK=8

def group_means(values, groups, count):
    counts = torch.bincount(groups, minlength=count)
    sums = torch.zeros(count, values.shape[1], dtype=torch.float64, device=values.device)
    sums.index_add_(0, groups, values.double())
    return sums / counts.clamp_min(1)[:, None], counts


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


