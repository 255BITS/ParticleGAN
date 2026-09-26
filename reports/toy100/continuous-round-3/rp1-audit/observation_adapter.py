"""Validate rates against the controller value supplied before the same update.

Observation only: no optimizer, model, RNG, controller, or learning-rule mutation.
The original step_with_policy still owns every call and every assigned rate.
"""
import atexit, hashlib, json, math, sys, weakref
from pathlib import Path

_expected=weakref.WeakKeyDictionary()
receipt={'scope':'observation-only same-update rate validation','calls':0,'validations':0,'trace':[]}
_installed=False

def install():
    global _installed
    if _installed:return
    from benchmarks.toy100 import schedule
    original_step=schedule.step_with_policy
    original_action=schedule.policy_rate_action

    def observed_step(trainer, real, *, network_lr_horizon_cap=None, network_lr_floor=None, **step_kwargs):
        # This is the same pure controller query made immediately afterward by
        # original_step. The query does not modify the controller or any rate.
        network,prior=schedule.policy_multipliers(
            trainer.completed_steps,trainer.recipe.total_steps,
            trainer.recipe.lr_anneal_start,trainer.recipe.lr_floor,
            network_lr_horizon_cap,network_lr_floor=network_lr_floor)
        before=trainer.completed_steps
        expected={'lr_g':trainer.initial_lrs[0][0]*network,
                  'lr_prior':trainer.initial_lrs[0][1]*prior,
                  'lr_d':trainer.initial_lrs[1][0]*network}
        _expected.pop(trainer,None)
        result=original_step(trainer,real,network_lr_horizon_cap=network_lr_horizon_cap,
                             network_lr_floor=network_lr_floor,**step_kwargs)
        _expected[trainer]={'step':before+1,'network':network,'prior':prior,'rates':expected,
                            'cap':network_lr_horizon_cap,'floor':network_lr_floor}
        receipt['calls']+=1
        return result

    def observed_action(trainer, completed_step, *, network_lr_horizon_cap=None, network_lr_floor=None):
        # Preserve the original validation contract, including completed step,
        # optimizer shape, per-group exact tolerance and result structure.
        if type(completed_step) is not int or completed_step!=trainer.completed_steps or completed_step<1:
            raise ValueError('completed_step must equal the trainer\'s completed update count')
        if len(trainer.opt_g.param_groups)!=2 or len(trainer.opt_d.param_groups)!=1:
            raise RuntimeError('expected G, prior, and D optimizer groups')
        saved=_expected.get(trainer)
        if saved is None or saved['step']!=completed_step:
            raise RuntimeError('missing same-update rate observation')
        if saved['cap']!=network_lr_horizon_cap or saved['floor']!=network_lr_floor:
            raise RuntimeError('rate policy arguments changed between update and observation')
        rates={'lr_g':trainer.opt_g.param_groups[0]['lr'],
               'lr_prior':trainer.opt_g.param_groups[1]['lr'],
               'lr_d':trainer.opt_d.param_groups[0]['lr']}
        for role,actual in rates.items():
            if not math.isclose(actual,saved['rates'][role],rel_tol=1e-12,abs_tol=1e-15):
                raise RuntimeError(f'{role} diverged from declared network LR policy')
        row={**rates,'network_multiplier':saved['network'],'prior_multiplier':saved['prior'],
             'network_lr_horizon_cap':network_lr_horizon_cap}
        if network_lr_floor is not None:row['network_lr_floor']=float(network_lr_floor)
        receipt['validations']+=1
        if completed_step<=3 or completed_step%50==0:
            receipt['trace'].append({'step':completed_step,**row})
        return row

    schedule.step_with_policy=observed_step
    schedule.policy_rate_action=observed_action
    sites=[]
    for mod in list(sys.modules.values()):
        if mod is None or not getattr(mod,'__name__','').startswith(('benchmarks','particlegan')):continue
        for key,value in list(vars(mod).items()):
            if value is original_step:setattr(mod,key,observed_step);sites.append(mod.__name__+'.'+key)
            elif value is original_action:setattr(mod,key,observed_action);sites.append(mod.__name__+'.'+key)
    receipt['patched_aliases']=sites
    receipt['original_step_module']=original_step.__module__
    receipt['adapter_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    _installed=True

def save():
    if '--output' in sys.argv:
        out=Path(sys.argv[sys.argv.index('--output')+1])
        if out.is_dir():(out/'observation-adapter-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
atexit.register(save)
