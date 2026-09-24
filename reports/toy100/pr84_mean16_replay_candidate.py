"""Late-state diagnostic replay of actual mean16 alternating Adam.

The original native phase callbacks execute for exact RNG/host accounting.
Their model/moment proposals are then replaced by a cloned mean16 update
from the SAME pre-step state. Both live Adam clocks end one step later, not
two. This deliberately redundant adapter is for the short continued-state
filter; a production gradient-averaging implementation would avoid wasted
native fields and repeated model construction. No LR or quality gate changes.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from unittest.mock import patch

import torch
from reports.toy100 import reallocation_smoothed_candidate as base
from reports.toy100 import pr84_finite_bank_adam_control as control
from reports.toy100 import pr84_finite_bank_vr_diagnostic as probe
from reports.toy100.pr84_critic_refinement_capture import snapshot, _sha, _rng

METHOD = 'pr84_mean16_late_saved_continuous_replay'


class MeanReplayRecorder(base.ReallocationRecorder):
    def phases(self, step, opt_d, opt_g, local):
        if self.correction and self.task == 'mode_hold' and self.enabled and step >= self.start_step:
            self.mean_pre = snapshot(local)
        yield from super().phases(step,opt_d,opt_g,local)

    @torch.no_grad()
    def step(self, optimizer, ordinary_step, closure=None):
        self.native_adam_step = ordinary_step
        result = super().step(optimizer,ordinary_step,closure)
        if (self.correction and self.task == 'mode_hold' and self.enabled
                and not self.passthrough and self.phase == 0 and optimizer is self.optimizers[0]):
            self.g_bank_origin = snapshot(self._local)
        return result

    @torch.no_grad()
    def correct(self, optimizer):
        local = self._local
        rng = _sha(_rng(local))
        before = _sha(self.mean_pre)
        step = local['step']+1
        with torch.random.fork_rng(devices=[]), torch.enable_grad(), \
                patch.object(self,'_smooth_on',False), \
                patch.object(self,'capture_sample',lambda value: None), \
                patch.object(torch.optim.Adam,'step',self.native_adam_step), \
                patch.object(probe,'STEP',step):
            generator,critic,prior = probe.fit.modules(self.mean_pre)
            a,b,_,_,native_g = probe.fit.banks(self.mean_pre,self.g_bank_origin,generator,prior)
            grows = probe.g_banks(self.g_bank_origin,generator,prior)
            if _sha(grows[0]) != _sha(native_g):
                raise RuntimeError('mean16 first G bank differs from native draw')
            drows = a+b
            if not torch.equal(drows[0]['real'],self.real):
                raise RuntimeError('mean16 first D bank differs from native draw')
            receipt,state = control.mean_adam_update(self.mean_pre,drows,grows,self.mean_recipe)
        for role in ('generator','critic'):
            getattr(local[role],'model',local[role]).load_state_dict(state[role])
        local['prior'].load_state_dict(state['prior'])
        local['opt_d'].load_state_dict(state['optimizer_d'])
        local['opt_g'].load_state_dict(state['optimizer_g'])
        if rng != _sha(_rng(local)) or before != _sha(self.mean_pre):
            raise RuntimeError('mean16 scratch update changed native randomness or pre-state')
        for role in ('d','g'):
            self.row[role] = {key:receipt['dynamics'][role][key] for key in ('rho','factor')}
        self.row['critic_sharpness'] = receipt['stencil'][-1]['sharpness']
        self.row['critic_width'] = receipt['stencil'][-1]['width']
        self.rng_checks += 1
        row = dict(step=step,selected='mean16',fit=dict(status='MEAN16_ADAM',records=[]),
            mean_update=receipt,banks_sha256=_sha(dict(d=drows,g=grows)),
            pre_state_sha256=before)
        self.corrections.append(row)

    def receipt(self):
        result = super().receipt()
        result.update(method=METHOD,scratch_optimizer_policy=METHOD,
            added_objective=None,output_allocation=None,nonlinear_fit=None,
            acceptance='original PR84 own-field bounds on mean16 native gradients',
            banks_per_role=16,mean_gradient_fields_per_player=32,
            extra_cloned_adam_updates_per_player=1,
            live_moment_updates_per_outer_step=1,
            native_fields='three native callbacks/player retained but resulting model/Adam proposals replaced',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope='late zero-input-noise/.029-output-noise ring only; short replay adapter',
            gradient_evaluations_per_outer_step=70 if self.corrections else result['gradient_evaluations_per_outer_step'])
        return result


@contextmanager
def pr84_mean16_replay_candidate(*,task='mode_hold',start_step=0,correction=True):
    import json
    recipe,_,_ = probe.declared_recipe(json.loads((Path(__file__).resolve().parents[2]/
        'configs/toy100/constraints_simple_regularization.json').read_text()))
    with patch.object(base,'ReallocationRecorder',MeanReplayRecorder):
        with base.reallocation_smoothed_candidate(task=task,start_step=start_step,correction=correction) as value:
            value[0].mean_recipe = recipe
            yield value
