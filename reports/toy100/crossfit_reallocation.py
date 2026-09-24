"""Nonlocal data allocation with an independent within-batch improvement check.

The first half of the existing D real minibatch constructs the output target;
the second half only validates native and fitted proposals against pre-G state.
Both halves must strictly improve their C+Q objective for any G/prior movement.
This remains a new sampled-data objective and controller, not a GAN theorem.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from unittest.mock import patch
import torch

from benchmarks.locked_shared import mode_hold
from reports.toy100 import pr84_smoothed_candidate as frozen
from reports.toy100.reallocation_smoothed_candidate import ReallocationRecorder
from reports.toy100.chamfer_discrete_reallocation import greedy_real_reallocate, _cost
from reports.toy100.chamfer_pullback import chamfer_targets
from reports.toy100.joint_output_pullback import fit_output_targets
from reports.toy100.pr84_critic_refinement_capture import _sha, _rng

METHOD='pr84_crossfit_data_reallocation_joint_output_fit'


class CrossfitRecorder(ReallocationRecorder):
    @torch.no_grad()
    def correct(self,optimizer):
        local=self._local
        clean=getattr(local['generator'],'model',local['generator'])
        prior=local['prior']
        parameters=self._params(optimizer)
        native=[p.detach().clone() for p in parameters]
        rng=_sha(_rng(local))
        owner=_sha(dict(d=local['critic'].state_dict(),od=local['opt_d'].state_dict(),og=optimizer.state_dict()))
        if len(self.real)%2:
            raise ValueError('cross-fit needs two equal existing real-batch halves')
        training,validation=self.real.chunk(2)
        costs=lambda points: [_cost(training,points),_cost(validation,points)]
        before=costs(self.pre_points)
        native_cost=costs(clean(prior.z).detach())
        allocated,allocation=greedy_real_reallocate(training,self.pre_points)
        target,counts,_,_=chamfer_targets(training.double(),allocated.double())
        target_cost=costs(target)
        if target_cost[0] > before[0]+1e-10*max(1.,before[0]):
            raise RuntimeError('training output-target construction increased C+Q')
        fit=fit_output_targets(clean,prior.z,target)
        fitted=costs(clean(prior.z).detach())
        thresholds=[64*torch.finfo(torch.float64).eps*max(1.,value) for value in before]

        def eligible(values):
            return all(after < start-eps for after,start,eps in zip(values,before,thresholds))

        native_ok=eligible(native_cost)
        fit_ok=fit['status']=='CONVERGED' and eligible(fitted)
        if fit_ok and (not native_ok or sum(fitted)<sum(native_cost)):
            selected,final='joint_fit',fitted
        elif native_ok:
            selected,final='native_gan',native_cost
            for p,saved in zip(parameters,native):p.copy_(saved)
        else:
            selected,final='rest',before
            for p,saved in zip(parameters,self.g_base):p.copy_(saved)
        if rng!=_sha(_rng(local)):
            raise RuntimeError('cross-fit consumed training randomness')
        self.rng_checks+=1
        if owner!=_sha(dict(d=local['critic'].state_dict(),od=local['opt_d'].state_dict(),og=optimizer.state_dict())):
            raise RuntimeError('cross-fit changed critic or Adam state')
        self.owner_checks+=1
        actual=costs(clean(prior.z).detach())
        if any(abs(a-b)>1e-10*max(1.,abs(b)) for a,b in zip(actual,final)):
            raise RuntimeError('cross-fit selection restoration mismatch')
        if selected!='rest' and not eligible(actual):
            raise RuntimeError('accepted whole G/prior move lacks two-half improvement')
        row=dict(step=local['step']+1,selected=selected,pre_cost=sum(before),
            native_cost=sum(native_cost),target_cost=sum(target_cost),fitted_cost=sum(fitted),
            final_cost=sum(final),allocation=allocation,target_assigned_counts=counts.tolist(),fit=fit,
            halves=dict(before=before,native=native_cost,target=target_cost,fitted=fitted,final=final),
            native_eligible=native_ok,fit_eligible=fit_ok,training_samples=len(training),validation_samples=len(validation))
        self.corrections.append(row)
        self.row['reallocation']={key:row[key] for key in
            ('selected','pre_cost','native_cost','target_cost','fitted_cost','final_cost')}

    def receipt(self):
        result=super().receipt()
        result.update(method=METHOD,scratch_optimizer_policy=METHOD,
            acceptance='both existing real-batch halves strictly improve C+Q relative to pre-G; among eligible native/fit choose lower summed cost; otherwise rest',
            output_allocation='only first half proposes global donor/sample replacements and C+Q target',
            scalar_cost='sum of the two half-batch C+Q costs; each half checked separately',
            validation='second half has no role in output target construction or nonlinear fitting',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return result


@contextmanager
def crossfit_reallocation(*,task='mode_hold',start_step=0,correction=True):
    ordinary=mode_hold.sample_ring
    with patch.object(frozen,'SmoothedBothBoundRecorder',
        lambda *,start_step=0:CrossfitRecorder(start_step=start_step,task=task,correction=correction)):
        with frozen.pr84_smoothed_candidate(task=task,start_step=start_step) as value:
            recorder,_=value
            def sampled(*args,**kwargs):
                result=ordinary(*args,**kwargs)
                recorder.capture_sample(result)
                return result
            with patch.object(mode_hold,'sample_ring',sampled):
                yield value
