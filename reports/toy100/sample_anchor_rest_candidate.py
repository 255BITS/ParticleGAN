"""Require a converged anchor fit before accepting any G/prior proposal.

If the bounded numerical solve cannot certify its output target, restore the
pre-G parameters. This closes the failed-fit case of the conditional invariant
region argument. Acquisition still requires successful solves; correct sampled
groups and bounded centroid errors are separate, unproven global premises.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from unittest.mock import patch
import torch

from reports.toy100 import sample_anchor_candidate as base
from reports.toy100.sample_group_anchor import field

METHOD='pr84_sample_group_anchor_joint_fit_rest_on_nonconvergence'


class RestOnFailureRecorder(base.SampleAnchorRecorder):
    @torch.no_grad()
    def correct(self,optimizer):
        super().correct(optimizer)
        row=self.corrections[-1]
        failed=row['fit']['status']!='CONVERGED'
        row['nonconverged_fit_rested']=failed
        if failed:
            for p,saved in zip(self._params(optimizer),self.g_base):
                p.copy_(saved)
            clean=getattr(self._local['generator'],'model',self._local['generator'])
            centers=torch.tensor(row['centers'],dtype=torch.float64)
            with torch.enable_grad():
                actual=field(clean(self._local['prior'].z).detach(),centers)['total']
            if abs(actual-row['pre_cost'])>1e-10*max(1.,row['pre_cost']):
                raise RuntimeError('nonconverged anchor fit did not restore pre-G objective')
            row.update(selected='rest',final_cost=row['pre_cost'])
            self.row['reallocation'].update(selected='rest',final_cost=row['pre_cost'])

    def receipt(self):
        result=super().receipt()
        result.update(method=METHOD,scratch_optimizer_policy=METHOD,
            nonconverged_fit_policy='restore pre-G/prior; retain exactly one native Adam update per player',
            conditional_invariance='requires correct bounded-error sample groups, separation and numerical-error margins; no unconditional Gaussian pathwise theorem',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return result


@contextmanager
def sample_anchor_rest_candidate(*,task='mode_hold',start_step=0,correction=True):
    with patch.object(base,'SampleAnchorRecorder',RestOnFailureRecorder):
        with base.sample_anchor_candidate(task=task,start_step=start_step,correction=correction) as value:
            yield value
