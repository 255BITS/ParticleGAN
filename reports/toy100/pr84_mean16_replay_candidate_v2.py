"""Isolate scratch clone constructors from host optimizer-role wrappers.

V1 remains frozen as a pre-quality harness error. Host optimizer_defaults
tracks prior parameter identities and wraps Adam construction. Scratch clones
must use the original constructors, without joining that host-owned registry.
This changes no mean16 gradient, bound, proposal, or accepted state formula.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from unittest.mock import patch

import torch
from particlegan.particle_prior import ParticlePrior
from reports.toy100 import pr84_mean16_replay_candidate as v1

METHOD = 'pr84_mean16_late_saved_continuous_replay_isolated_clones'


class IsolatedCloneRecorder(v1.MeanReplayRecorder):
    @torch.no_grad()
    def correct(self, optimizer):
        with patch.object(torch.optim.Adam,'__init__',self.scratch_adam_init), \
                patch.object(ParticlePrior,'__init__',self.scratch_prior_init):
            super().correct(optimizer)

    def receipt(self):
        result = super().receipt()
        result.update(method=METHOD,scratch_optimizer_policy=METHOD,
            clone_constructors='original Adam and ParticlePrior constructors; excluded from host role registry',
            critic_advantage_scope='native discarded-field diagnostic, not mean16 loss',
            gradient_evaluations_per_outer_step=35 if self.corrections else 3,
            gradient_evaluation_unit='per player per outer step',
            total_player_gradient_evaluations_per_outer_step=70 if self.corrections else 6,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return result


@contextmanager
def pr84_mean16_replay_candidate_v2(*,task='mode_hold',start_step=0,correction=True):
    adam_init, prior_init = torch.optim.Adam.__init__, ParticlePrior.__init__
    if (adam_init.__qualname__ != 'Adam.__init__'
            or prior_init.__qualname__ != 'ParticlePrior.__init__'):
        raise RuntimeError('construct this diagnostic before host optimizer-role wrappers')
    with patch.object(v1,'MeanReplayRecorder',IsolatedCloneRecorder):
        with v1.pr84_mean16_replay_candidate(task=task,start_step=start_step,
                                            correction=correction) as value:
            value[0].scratch_adam_init = adam_init
            value[0].scratch_prior_init = prior_init
            yield value
