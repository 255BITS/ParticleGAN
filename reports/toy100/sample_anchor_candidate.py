"""Sample-group anchor MM with joint neural fitting and whole-map acceptance.

This adds the distinct-anchor objective explicitly. Its group count and
centers come only from the current real minibatch MST. One active quadratic
is minimized in output space, then realized jointly through G and the prior.
The fixed-center free-output proof does not certify group estimation or the
stochastic neural host. Conditional hosts retain the original PR84 update.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from unittest.mock import patch
import torch

from benchmarks.locked_shared import mode_hold
from reports.toy100 import pr84_smoothed_candidate as frozen
from reports.toy100.reallocation_smoothed_candidate import ReallocationRecorder
from reports.toy100.sample_group_anchor import mst_groups, output_mm_step, field
from reports.toy100.joint_output_pullback import fit_output_targets
from reports.toy100.pr84_critic_refinement_capture import _sha, _rng

METHOD='pr84_sample_group_distinct_anchor_joint_output_fit'


class SampleAnchorRecorder(ReallocationRecorder):
    @torch.no_grad()
    def correct(self,optimizer):
        local=self._local
        clean=getattr(local['generator'],'model',local['generator'])
        prior=local['prior']
        params=self._params(optimizer)
        native=[p.detach().clone() for p in params]
        rng=_sha(_rng(local))
        owner=_sha(dict(d=local['critic'].state_dict(),od=local['opt_d'].state_dict(),og=optimizer.state_dict()))
        centers,grouping=mst_groups(self.real)
        if not 0<len(centers)<len(prior.z):
            raise RuntimeError('sample-group anchor rule requires strictly more particles than inferred groups')
        with torch.enable_grad():
            mm=output_mm_step(self.pre_points.double(),centers)
            native_cost=field(clean(prior.z).detach(),centers)['total']
        before=mm['before']
        target=torch.tensor(mm['target'],dtype=prior.z.dtype)
        fit=fit_output_targets(clean,prior.z,target)
        with torch.enable_grad():
            fitted=field(clean(prior.z).detach(),centers)['total']
        eps=64*torch.finfo(torch.float64).eps*max(1.,before)
        if fit['status']=='CONVERGED' and fitted<min(before,native_cost)-eps:
            selected,final='joint_fit',fitted
        elif native_cost<before-eps:
            selected,final='native_gan',native_cost
            for p,saved in zip(params,native):p.copy_(saved)
        else:
            selected,final='rest',before
            for p,saved in zip(params,self.g_base):p.copy_(saved)
        if rng!=_sha(_rng(local)):
            raise RuntimeError('sample anchor fitting consumed training randomness')
        self.rng_checks+=1
        if owner!=_sha(dict(d=local['critic'].state_dict(),od=local['opt_d'].state_dict(),og=optimizer.state_dict())):
            raise RuntimeError('sample anchor fitting changed critic or Adam state')
        self.owner_checks+=1
        with torch.enable_grad():
            actual=field(clean(prior.z).detach(),centers)['total']
        if abs(actual-final)>1e-10*max(1.,abs(final)) or actual>before+eps:
            raise RuntimeError('whole-map anchor acceptance/restoration differs')
        row=dict(step=local['step']+1,selected=selected,pre_cost=before,native_cost=native_cost,
            target_cost=mm['after'],fitted_cost=fitted,final_cost=final,fit=fit,
            grouping=grouping,centers=centers.tolist(),mm=mm)
        self.corrections.append(row)
        self.row['reallocation']={key:row[key] for key in
            ('selected','pre_cost','native_cost','target_cost','fitted_cost','final_cost')}

    def receipt(self):
        result=super().receipt()
        result.update(method=METHOD,scratch_optimizer_policy=METHOD,
            added_objective='unit-mean distinct group-anchor squared distance plus nearest-group squared distance',
            output_allocation='one exact active-quadratic MM target; no C+Q sampled-point reallocation',
            groups='current native real minibatch; Prim MST largest additive edge gap; no configured count',
            acceptance='lowest actual current-group anchor loss among pre-G, native GAN, and converged fitted proposals',
            proof_scope='fixed correctly inferred distinct centers, N>K, freely movable outputs only',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return result


@contextmanager
def sample_anchor_candidate(*,task='mode_hold',start_step=0,correction=True):
    ordinary=mode_hold.sample_ring
    with patch.object(frozen,'SmoothedBothBoundRecorder',
        lambda *,start_step=0:SampleAnchorRecorder(start_step=start_step,task=task,correction=correction)):
        with frozen.pr84_smoothed_candidate(task=task,start_step=start_step) as value:
            recorder,_=value
            def sampled(*args,**kwargs):
                result=ordinary(*args,**kwargs)
                recorder.capture_sample(result)
                return result
            with patch.object(mode_hold,'sample_ring',sampled):
                yield value
