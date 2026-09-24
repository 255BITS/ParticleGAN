"""Remember corroborated real support; realize it from the pre-G parameters.

The data estimator sees exactly one native D bank per outer update. After two
compatible banks establish fixed identities, absent minibatch groups cannot
delete support and isolated observations cannot create new identities. This
requires a correct complete bootstrap and a stationary separated target.
Running data estimates do not scale the generator's correction capacity.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from unittest.mock import patch

import torch

from reports.toy100 import sample_anchor_candidate as base
from reports.toy100 import sample_anchor_rest_candidate as rest
from reports.toy100.sample_anchor_prestart_candidate import PreStartRecorder
from reports.toy100.sample_group_two_bank_memory import TwoBankFixedSupportMemory
from reports.toy100.pr84_critic_refinement_capture import _sha, _rng

METHOD = 'pr84_confirmed_fixed_support_prestart_joint_fit'


class FixedSupportRecorder(PreStartRecorder):
    def __init__(self, *, start_step=0, task='mode_hold', correction=True):
        super().__init__(start_step=start_step, task=task, correction=correction)
        self.memory = (TwoBankFixedSupportMemory(expected_first_bank_id=start_step+1)
                       if correction and task == 'mode_hold' else None)
        self.learner_bank_count = 0
        self.learner_last_observed_step = None

    def learner_state_dict(self):
        return None if self.memory is None else self.memory.state_dict()

    def load_learner_state_dict(self, state):
        if self.memory is None:
            if state is not None:
                raise ValueError('inactive correction cannot restore support memory')
        else:
            if state is None:
                raise ValueError('active correction requires complete support memory')
            self.memory.load_state_dict(state)

    @torch.no_grad()
    def correct(self, optimizer):
        local = self._local
        if self.memory is None:
            raise RuntimeError('inactive support memory entered correction')
        rng = _sha(_rng(local))
        owner = _sha(dict(d=local['critic'].state_dict(),
                          od=local['opt_d'].state_dict(), og=optimizer.state_dict()))
        observation = self.memory.observe(self.real, bank_id=local['step']+1)
        self.learner_bank_count += 1
        self.learner_last_observed_step = local['step']+1
        if not self.memory.confirmed:
            for parameter, saved in zip(self._params(optimizer), self.g_base):
                parameter.copy_(saved)
            clean = getattr(local['generator'], 'model', local['generator'])
            if not torch.equal(clean(local['prior'].z), self.pre_points):
                raise RuntimeError('bootstrap rest did not restore pre-G outputs')
            row = dict(step=local['step']+1, selected='rest', pre_cost=None,
                native_cost=None, target_cost=None, fitted_cost=None, final_cost=None,
                fit=dict(status='BOOTSTRAP_PENDING', records=[]), centers=[],
                grouping=dict(source='two-bank fixed-support memory', observation=observation),
                mm=None, nonconverged_fit_rested=True)
            self.corrections.append(row)
            self.row['reallocation'] = {key: row[key] for key in
                ('selected', 'pre_cost', 'native_cost', 'target_cost', 'fitted_cost', 'final_cost')}
            self.rng_checks += 1
            self.owner_checks += 1
        else:
            centers = self.memory.centers()
            grouping = dict(source='two-bank fixed-support memory',
                observation=observation, confirmed_groups=len(centers),
                reference_centers=[x.tolist() for x in self.memory.reference_centers],
                accumulated_counts=list(self.memory.confirmed_counts))
            with patch.object(base, 'mst_groups', lambda real: (centers, grouping)):
                super().correct(optimizer)
        if rng != _sha(_rng(local)):
            raise RuntimeError('support memory or correction consumed training randomness')
        if owner != _sha(dict(d=local['critic'].state_dict(),
                             od=local['opt_d'].state_dict(), og=optimizer.state_dict())):
            raise RuntimeError('support memory or correction changed D or Adam state')
        self.corrections[-1]['learner_state_sha256'] = _sha(self.learner_state_dict())

    def receipt(self):
        result = super().receipt()
        result.update(method=METHOD, scratch_optimizer_policy=METHOD,
            groups='two consecutive compatible native D banks confirm fixed identities; then raw samples update fixed disjoint reference balls',
            acceptance='pre-G, native GAN, converged joint fit compared using remembered-center objective; failed fit rests',
            bootstrap='restore pre-G/prior while two-bank correspondence is unresolved',
            estimator='cumulative real sample sufficient statistics; absent groups retained; no births or deletions after confirmation',
            correction_capacity='same fixed numerical budget and full remembered target at every age; no count-based correction gain',
            proof_scope='stationary separated target, correct complete bootstrap, bounded estimated-center errors and successful neural target realization',
            learner_state_required=self.memory is not None,
            learner_state_sha256=None if self.memory is None else _sha(self.learner_state_dict()),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return result


@contextmanager
def sample_anchor_memory_candidate(*, task='mode_hold', start_step=0, correction=True):
    with patch.object(rest, 'RestOnFailureRecorder', FixedSupportRecorder):
        with rest.sample_anchor_rest_candidate(task=task, start_step=start_step,
                                                correction=correction) as value:
            yield value
