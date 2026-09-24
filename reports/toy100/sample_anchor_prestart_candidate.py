"""Realize sampled anchor targets starting at the pre-G parameters.

The local affine model eliminates the native proposal's Jacobian-nullspace
component by fitting from the pre-G state. Nonlinear GN is only locally
minimum-increment; this is not a global nearest-parameter projection theorem.
All objective, numerical budgets, native fields and Adam counts are retained.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from unittest.mock import patch
import torch
from reports.toy100 import sample_anchor_candidate as base
from reports.toy100 import sample_anchor_rest_candidate as rest

METHOD='pr84_sample_group_anchor_prestart_joint_fit_rest_on_nonconvergence'


class PreStartRecorder(rest.RestOnFailureRecorder):
    @torch.no_grad()
    def correct(self,optimizer):
        ordinary=base.fit_output_targets
        def from_pre(generator,prior_z,target,**kwargs):
            for parameter,saved in zip(self._params(optimizer),self.g_base):
                parameter.copy_(saved)
            return ordinary(generator,prior_z,target,**kwargs)
        with patch.object(base,'fit_output_targets',from_pre):
            super().correct(optimizer)

    def receipt(self):
        result=super().receipt()
        result.update(method=METHOD,scratch_optimizer_policy=METHOD,
            fit_start='pre-G parameters; avoids carrying native Jacobian-nullspace motion in the affine model',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return result


@contextmanager
def sample_anchor_prestart_candidate(*,task='mode_hold',start_step=0,correction=True):
    with patch.object(rest,'RestOnFailureRecorder',PreStartRecorder):
        with rest.sample_anchor_rest_candidate(task=task,start_step=start_step,correction=correction) as value:
            yield value
