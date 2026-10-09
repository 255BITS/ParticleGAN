"""Training-neighborhood finite shape control with mean-preserving corrections."""
from copy import deepcopy
import math
import torch
from .hydraulic import HydraulicTravel, HydraulicDeformationTravel


class HydraulicShapeTravel(HydraulicTravel):
    """Bound sampled travel and local finite antithetic excess, not all density.

    Constants define one structural rule; no target labels, evaluation thresholds
    or fresh sampling. Corrections act on the actual G/prior proposal after the
    public optimizer advances once. The mean constraint is first order only.
    """
    SETTINGS = dict(schema=1, link_multiplier=4., corrections_per_ray=3,
                    ray_trials=7, correction_over_relaxation=2.,
                    mean_constraint='antithetic_midpoint_mean_v1',
                    zero_spacing_policy='nearest_distinct_or_zero_ray_v1')

    def __init__(self, fraction):
        super().__init__(fraction)
        self.summary.update(zero_spacing=0, shape_active=0, shape_corrections=0,
                            graph_components_sum=0., local_capacity_sum=0.,
                            old_excess_sum=0., accepted_excess_sum=0.,
                            old_odd_energy_sum=0., accepted_odd_energy_sum=0.,
                            max_shape_bound_violation=0., max_mean_linear_residual=0.)

    def radius(self, real):
        return HydraulicDeformationTravel.radius(self, real)

    @torch.no_grad()
    def prepare(self, real):
        points=real.detach().flatten(1)
        if points.shape[1] not in (1,2):
            raise ValueError('hydraulic finite shape supports one or two output coordinates')
        radius=self.radius(real)
        if radius==0: return (radius,None,None)
        distances=torch.cdist(points,points,compute_mode='donot_use_mm_for_euclid_dist')
        # CPU union-find has a deterministic fixed edge order; no SciPy dependency.
        edges=torch.triu(distances <= self.SETTINGS['link_multiplier']*radius/self.fraction,1).cpu().nonzero().tolist()
        parents=list(range(len(points)))
        def find(i):
            while parents[i]!=i:
                parents[i]=parents[parents[i]];i=parents[i]
            return i
        for i,j in edges:
            a,b=find(i),find(j)
            if a!=b: parents[max(a,b)]=min(a,b)
        roots=[find(i) for i in range(len(points))]
        numbers={root:i for i,root in enumerate(sorted(set(roots)))}
        labels=torch.tensor([numbers[r] for r in roots],device=points.device,dtype=torch.long)
        n=len(numbers);counts=torch.bincount(labels,minlength=n)
        sums=points.new_zeros(n,points.shape[1]).index_add_(0,labels,points)
        residual=(points-sums[labels]/counts[labels,None]).square().sum(1)
        variance=points.new_zeros(n).index_add_(0,labels,residual)/(counts-1).clamp_min(1)
        # Singleton or identical neighborhoods have no covariance information.
        # Their positive-spacing fallback is declared; constant batches use zero.
        capacity=torch.where(variance>0,variance,variance.new_tensor((radius/self.fraction)**2))
        return radius,labels,capacity

    @staticmethod
    def score(plus, minus, assignments, capacity):
        odd=((plus-minus)*.5).flatten(1).square().sum(1)
        counts=torch.bincount(assignments,minlength=len(capacity))
        sums=odd.new_zeros(len(capacity)).index_add_(0,assignments,odd)
        means=sums/counts.clamp_min(1)
        excess=((means-capacity).relu()*counts).sum()/len(odd)
        return excess,odd.mean()

    @staticmethod
    def _flat_grad(value, parameters, *, retain_graph):
        grads=torch.autograd.grad(value,parameters,retain_graph=retain_graph,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1).double()
                          for p,g in zip(parameters,grads)])

    @torch.no_grad()
    def step(self, optimizer, real, probe, *, shape_probe, prepared):
        radius,labels,capacity=prepared
        if radius==0:
            self.summary['zero_spacing']+=1
            return super().step(optimizer,real,probe,radius=0.)
        parameters=[p for group in optimizer.param_groups for p in group['params']]
        trainable=[p for p in parameters if p.requires_grad]
        before=[p.detach().clone() for p in parameters]
        _,old=probe();plus,minus=shape_probe()
        midpoint=((plus+minus)*.5).flatten(1)
        closest=torch.cdist(midpoint,real.detach().flatten(1)).argmin(1)
        assignments=labels[closest]
        old_excess,old_energy=self.score(plus,minus,assignments,capacity)
        old_excess=float(old_excess);old_energy=float(old_energy)
        mean_capacity=float(capacity[assignments].mean())
        tolerance=64*torch.finfo(old.dtype).eps*max(mean_capacity,1e-20)
        optimizer.step()
        delta=[p.detach()-v for p,v in zip(parameters,before)]
        network,joint=probe()
        proposed=float(self.rms(joint-old));network_rms=float(self.rms(network-old))
        prior_rms=float(self.rms(joint-network))
        if not all(math.isfinite(value) for value in (proposed,network_rms,prior_rms)):
            for p,v in zip(parameters,before): p.copy_(v)
            raise FloatingPointError('hydraulic finite shape received a nonfinite proposal')
        travel=(joint-old).flatten(1)
        shared=float(travel.mean(0).square().sum()/travel.square().sum(1).mean().clamp_min(1e-30))
        scale=min(1.,radius/proposed) if proposed>0 else 1.
        calls=2;corrections=0;active=False;accepted=0.;accepted_excess=old_excess;accepted_energy=old_energy
        target=old_excess
        for trial in range(self.SETTINGS['ray_trials']):
            if scale!=1. or trial:
                for p,v,d in zip(parameters,before,delta): p.copy_(v+scale*d)
            target=(1-scale)*old_excess
            for _ in range(self.SETTINGS['corrections_per_ray']):
                plus,minus=shape_probe();calls+=1
                excess,_=self.score(plus,minus,assignments,capacity)
                if math.isfinite(float(excess)) and float(excess)<=target+tolerance: break
                active=True
                with torch.enable_grad():
                    plus,minus=shape_probe()
                    excess,_=self.score(plus,minus,assignments,capacity)
                    mean=((plus+minus)*.5).flatten(1).mean(0)
                    gradient=self._flat_grad(excess,trainable,retain_graph=True)
                    mean_rows=torch.stack([self._flat_grad(v,trainable,retain_graph=True) for v in mean])
                gram=mean_rows@mean_rows.T
                null_gradient=gradient-mean_rows.T@(torch.linalg.pinv(gram,hermitian=True,rtol=1e-10)@(mean_rows@gradient))
                norm=float(null_gradient.square().sum())
                if not math.isfinite(norm) or norm<=1e-30: break
                coefficient=self.SETTINGS['correction_over_relaxation']*max(0.,float(excess)-target)/norm
                movement=-coefficient*null_gradient
                offset=0;actual=[]
                for p in trainable:
                    start=p.detach().clone();length=p.numel()
                    p.add_(movement[offset:offset+length].reshape_as(p).to(p.dtype));offset+=length
                    actual.append((p-start).reshape(-1).double())
                residual=float((mean_rows@torch.cat(actual)).abs().max())
                self.summary['max_mean_linear_residual']=max(self.summary['max_mean_linear_residual'],residual)
                corrections+=1
            plus,minus=shape_probe();calls+=1
            excess,odd_energy=self.score(plus,minus,assignments,capacity)
            _,joint=probe();calls+=1
            movement_rms=float(self.rms(joint-old))
            if math.isfinite(movement_rms) and movement_rms<=radius and math.isfinite(float(excess)) and float(excess)<=target+tolerance:
                accepted=movement_rms;accepted_excess=float(excess);accepted_energy=float(odd_energy)
                break
            scale*=.5
        else:
            scale=0.
            for p,v in zip(parameters,before): p.copy_(v)
            target=old_excess
        work=-sum(float((p.grad*(p-v)).sum()) for p,v in zip(parameters,before) if p.grad is not None)
        row=self.summary;row['updates']+=1;row['limited']+=int(scale<1);row['rejected']+=int(scale==0)
        row['shape_active']+=int(active);row['shape_corrections']+=corrections;row['probe_calls']+=calls
        for key,value in [('radius',radius),('proposed_rms',proposed),('accepted_rms',accepted),('network_rms',network_rms),
                          ('prior_rms',prior_rms),('shared_fraction',shared),('scale',scale),('work',work),
                          ('graph_components',len(capacity)),('local_capacity',mean_capacity),('old_excess',old_excess),
                          ('accepted_excess',accepted_excess),('old_odd_energy',old_energy),('accepted_odd_energy',accepted_energy)]:
            row[key+'_sum']+=value
        row['max_accepted_radius_ratio']=max(row['max_accepted_radius_ratio'],accepted/radius)
        row['max_shape_bound_violation']=max(row['max_shape_bound_violation'],accepted_excess-target-tolerance)

    def state_dict(self):
        return {**super().state_dict(),'shape_settings':deepcopy(self.SETTINGS)}

    def validate_state_dict(self,state,steps):
        if not isinstance(state,dict) or state.get('shape_settings')!=self.SETTINGS:
            raise ValueError('hydraulic finite shape checkpoint setting differs')
        super().validate_state_dict({k:v for k,v in state.items() if k!='shape_settings'},steps)
        row=state['summary']
        if not (0<=row['zero_spacing']<=steps and 0<=row['shape_active']<=steps
                and 0<=row['shape_corrections']<=21*steps and row['max_shape_bound_violation']==0):
            raise ValueError('invalid hydraulic finite shape counters')
