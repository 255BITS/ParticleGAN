"""Eight-step exploratory alternating Rp continuation with common instance noise.

Starts from exact saved PR84 warm1325 post-D and cold1200 final model, Adam,
noise and data streams. A copied critic is fitted once as in the frozen
two-state assay; its saved D Adam moments are retained. Warm completes update
1325's G phase, then takes seven full D/G updates; cold takes eight full
D/G updates. Every D/G phase uses original paired Rp logistic losses and the
same antithetic Gaussian observation channel, with native .25/3 own-curvature
bounds and constant Adam rates. There are no subsequent inner D refits.
This is a local counterfactual with a prefitted D, not cold-from-scratch GAN.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.locked_shared import mode_hold
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
from reports.toy100 import common_instance_noise_falsifier as common
from reports.toy100 import common_instance_noise_cold_endpoint as cold
from reports.toy100 import pr84_critic_relaxation as prior_diagnostic
from reports.toy100.alternating_curvature_scratch import _metric, _rho
from reports.toy100.coverage_fixed_eval import fixed_draw, score_support


WARM_ARCHIVE = ROOT/'reports/toy100/continuous-evidence/common-instance-noise-2state/input-states.pt.gz'
WARM_RAW_SHA = '40a3e4e364a81bd8236b113286ac8873581c9ae66d4ed4c8ca28b168d0a9d564'
COLD_ARCHIVE = cold.ARCHIVE
COLD_RAW_SHA = cold.ARCHIVE_RAW_SHA
WIDTH = cold.WIDTH
OBSERVATION_SEED = cold.OBSERVATION_SEED
SOURCE = (
    'reports/toy100/common_instance_noise_short_continuation.py',
    'reports/toy100/common_instance_noise_falsifier.py',
    'reports/toy100/common_instance_noise_cold_endpoint.py',
    'reports/toy100/pr84_critic_relaxation.py',
    'reports/toy100/alternating_curvature_scratch.py',
    'reports/toy100/coverage_fixed_eval.py',
    'benchmarks/locked_shared/mode_hold.py',
    'benchmarks/locked_shared/mlp.py',
    'particlegan/gan_loss.py',
    'particlegan/grad_regularizers.py',
    'configs/toy100/constraints_simple_regularization.json',
)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def loaded(path, expected):
    compressed=path.read_bytes()
    raw=gzip.decompress(compressed)
    if sha(raw)!=expected:
        raise AssertionError(f'wrong archived state: {path}')
    with torch.random.fork_rng(devices=[]):
        return torch.load(io.BytesIO(raw),weights_only=True,map_location='cpu')


class NativeSampler:
    """Keep native data/global torch streams local; one D/G minibatch each."""

    def __init__(self, data_state, torch_state, output_sigma):
        self.data=torch.Generator().set_state(data_state)
        self.torch_state=torch_state.clone()
        self.sigma=float(output_sigma)
        if self.sigma!=.029:
            raise AssertionError('expected pinned late native output noise')

    def _draw(self, which, generator, prior):
        before=torch.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]),torch.no_grad():
            torch.set_rng_state(self.torch_state)
            if which=='d':
                real=mode_hold.sample_ring(mode_hold.ring_means(),mode_hold.BATCH,
                                           mode_hold.SIGMA,self.data)
                latent,indices=prior.sample(mode_hold.BATCH,generator=self.data)
                clean=generator(latent)
                noise=torch.randn_like(clean)
                row={'real':real,'fake':clean+self.sigma*noise,
                     'indices':indices,'noise':noise}
            else:
                latent,indices=prior.sample(mode_hold.BATCH,generator=self.data)
                clean=generator(latent)
                noise=torch.randn_like(clean)
                real=mode_hold.sample_ring(mode_hold.ring_means(),mode_hold.BATCH,
                                           mode_hold.SIGMA,self.data)
                row={'real':real,'indices':indices,'noise':noise,'sigma':self.sigma}
            self.torch_state=torch.get_rng_state().clone()
        if not torch.equal(torch.get_rng_state(),before):
            raise AssertionError('sampler changed global RNG')
        return row

    def d(self,generator,prior):
        return self._draw('d',generator,prior)

    def g(self,generator,prior):
        return self._draw('g',generator,prior)

    def receipt(self):
        return {'data_sha256':sha(self.data.get_state().numpy().tobytes()),
                'torch_sha256':sha(self.torch_state.numpy().tobytes())}


def d_optimizer(critic,saved_state):
    opt=torch.optim.Adam(critic.parameters())
    opt.load_state_dict(deepcopy(saved_state))
    return opt


def steps(opt):
    return [float(opt.state[p]['step']) for group in opt.param_groups for p in group['params']]


def d_update(critic,opt,bank,gan,regularizer,step):
    params=list(critic.parameters())
    old=[p.detach().clone() for p in params]
    before_steps=steps(opt)
    opt.zero_grad()
    before_loss,logistic,penalty=prior_diagnostic.d_loss(critic,bank,gan,regularizer,step)
    if not torch.isfinite(before_loss):
        raise FloatingPointError('nonfinite D loss at base')
    before_loss.backward()
    g0=[p.grad.detach().clone() for p in params]
    opt.step()
    proposal=[p.detach().clone() for p in params]
    metric=_metric(opt)
    after_loss,_,_=prior_diagnostic.d_loss(critic,bank,gan,regularizer,step)
    g1=torch.autograd.grad(after_loss,params)
    rho=_rho(old,proposal,g0,g1,metric)
    factor=min(1.,3./rho) if rho else 1.
    with torch.no_grad():
        for p,a,b in zip(params,old,proposal):
            p.copy_(torch.lerp(a,b,factor) if factor<1 else b)
    if steps(opt)!=[value+1 for value in before_steps]:
        raise AssertionError('D Adam moment count changed')
    return {'loss_before':float(before_loss.detach()),
            'logistic_before':float(logistic.detach()),
            'b_cap_before':float(penalty.detach()),
            'rho':rho,'factor':factor,'adam_steps':steps(opt)[0]}


def g_update(generator,prior,critic,opt,bank,gan):
    params=list(generator.parameters())+list(prior.parameters())
    old=[p.detach().clone() for p in params]
    before_steps=steps(opt)
    clean_before=generator(prior.z).detach().clone()
    opt.zero_grad()
    loss=common.common_g_loss(generator,prior,critic,bank,gan)
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite G loss at base')
    loss.backward()
    g0=[p.grad.detach().clone() for p in params]
    opt.step()
    proposal=[p.detach().clone() for p in params]
    metric=_metric(opt)
    loss_at_proposal=common.common_g_loss(generator,prior,critic,bank,gan)
    g1=torch.autograd.grad(loss_at_proposal,params)
    rho=_rho(old,proposal,g0,g1,metric)
    factor=min(1.,.25/rho) if rho else 1.
    with torch.no_grad():
        for p,a,b in zip(params,old,proposal):
            p.copy_(torch.lerp(a,b,factor) if factor<1 else b)
        clean_after=generator(prior.z).detach().clone()
    if steps(opt)!=[value+1 for value in before_steps]:
        raise AssertionError('G/prior Adam moment count changed')
    return {'loss_before':float(loss.detach()),'loss_proposal':float(loss_at_proposal.detach()),
            'rho':rho,'factor':factor,'adam_steps':steps(opt)[0],
            'clean_before':clean_before,'clean_after':clean_after}


def grade(clean,step):
    indices,noise=fixed_draw(step,clean)
    return score_support(clean,indices,noise,mode_hold.ring_means())


def geometry(clean_initial,clean_current):
    means=mode_hold.ring_means()
    owners=torch.cdist(clean_initial,means).argmin(1)
    counts=torch.bincount(owners,minlength=8)
    empty=(counts==0).nonzero().flatten().tolist()
    current_counts=torch.bincount(torch.cdist(clean_current,means).argmin(1),minlength=8)
    if empty!=[6]:
        return {'initial_missing':empty,'current_nearest_counts':current_counts.tolist()}
    target=means[6]
    rows=[]
    for i,owner in enumerate(owners.tolist()):
        rad=means[owner]/means[owner].norm()
        chord=target-means[owner]
        tangent=chord-(chord@rad)*rad
        tangent=tangent/tangent.norm() if float(tangent.norm())>1e-6 else None
        delta=clean_current[i]-clean_initial[i]
        rows.append({'particle':i,'initial_owner':owner,'surplus':bool(counts[owner]>1),
                     'cumulative_tangential_toward_missing':float(delta@tangent) if tangent is not None else None,
                     'cumulative_missing_chord':float(delta@(chord/chord.norm())),
                     'missing_center_distance':float((clean_current[i]-target).norm())})
    return {'initial_missing':[6],'current_nearest_counts':current_counts.tolist(),
            'particles':rows}


def fit_warm(states,gan,regularizer):
    pre,saved=states[1325]['pre_step'],states[1325]['post_accepted_d']
    generator,critic,prior=prior_diagnostic.modules(saved)
    _,_,train,heldout,actual_g=prior_diagnostic.banks(pre,saved,generator,prior)
    obs=torch.Generator().manual_seed(OBSERVATION_SEED)
    noisy_train=common.bank_with_instance_noise(train,WIDTH,obs)
    _=common.bank_with_instance_noise(heldout,WIDTH,obs)
    actual_aug=common.g_bank_with_instance_noise(actual_g,WIDTH,obs)
    fit=prior_diagnostic.relax(critic,noisy_train,gan,regularizer,1325,
                               prior_diagnostic.saved_metric(saved['optimizer_d']))
    h=prior_diagnostic.state_hash(critic.state_dict())
    if h!='591ebe0953fd41dbea800628e6f455d0e88ee80f3cd0094ebffeee9bbd63ee26':
        raise AssertionError('warm copied-D fit differs from frozen v1')
    fit['selected_critic_sha256']=h
    sampler=NativeSampler(saved['rng']['data'],saved['rng']['torch'],saved['noise']['output_sigma'])
    drawn=sampler.g(generator,prior)
    if prior_diagnostic.state_hash(drawn)!=prior_diagnostic.state_hash(actual_g):
        raise AssertionError('warm actual next G bank replay failed')
    return saved,generator,critic,prior,obs,sampler,fit,actual_aug


def fit_cold(snapshot,gan,regularizer):
    saved=cold.saved_view(snapshot)
    generator,critic,prior=prior_diagnostic.modules(saved)
    train,heldout,g_banks=cold.banks(snapshot,generator,prior)
    obs=torch.Generator().manual_seed(OBSERVATION_SEED)
    noisy_train=common.bank_with_instance_noise(train,WIDTH,obs)
    _=common.bank_with_instance_noise(heldout,WIDTH,obs)
    _=[common.g_bank_with_instance_noise(bank,WIDTH,obs) for bank in g_banks]
    fit=prior_diagnostic.relax(critic,noisy_train,gan,regularizer,1201,
                               prior_diagnostic.saved_metric(saved['optimizer_d']))
    h=prior_diagnostic.state_hash(critic.state_dict())
    if h!='0a3e0e24d3fe88fb133e934383e3606a3cb22f18031a49a7a62a60e6dfeffa52':
        raise AssertionError('cold copied-D fit differs from frozen cold assay')
    fit['selected_critic_sha256']=h
    sampler=NativeSampler(snapshot['data_stream'],snapshot['torch_rng'],snapshot['noise_policy']['output_sigma'])
    return saved,generator,critic,prior,obs,sampler,fit,None


def run_case(name,state,gan,regularizer,output):
    started=time.perf_counter()
    with torch.random.fork_rng(devices=[]):
        if name=='warm1325':
            saved,generator,critic,prior,obs,sampler,fit,prelude_bank=fit_warm(state,gan,regularizer)
            first_step=1325
        else:
            saved,generator,critic,prior,obs,sampler,fit,prelude_bank=fit_cold(state,gan,regularizer)
            first_step=1201
        opt_d=d_optimizer(critic,saved['optimizer_d'])
        opt_g=prior_diagnostic.g_optimizer(generator,prior,saved['optimizer_g'])
        initial=generator(prior.z).detach().clone()
        rows=[]
        for index in range(8):
            step=first_step+index
            if prelude_bank is not None and index==0:
                d_receipt={'status':'already_completed_in_archived_warm_post_D',
                           'adam_steps':steps(opt_d)[0]}
                g_bank=prelude_bank
            else:
                d_native=sampler.d(generator,prior)
                d_bank=common.bank_with_instance_noise(d_native,WIDTH,obs)
                d_receipt=d_update(critic,opt_d,d_bank,gan,regularizer,step)
                g_native=sampler.g(generator,prior)
                g_bank=common.g_bank_with_instance_noise(g_native,WIDTH,obs)
            g_receipt=g_update(generator,prior,critic,opt_g,g_bank,gan)
            clean=g_receipt.pop('clean_after')
            before=g_receipt.pop('clean_before')
            if not torch.isfinite(clean).all() or not all(torch.isfinite(p).all() for p in critic.parameters()):
                raise FloatingPointError('nonfinite model at continued update')
            row={'step':step,'d':d_receipt,'g':g_receipt,
                 'grade_before':grade(before,step),'grade_after':grade(clean,step),
                 'clean_points':clean.tolist(),'geometry':geometry(initial,clean),
                 'native_streams_after':sampler.receipt(),
                 'observation_stream_after_sha256':sha(obs.get_state().numpy().tobytes())}
            rows.append(row)
            (output/f'{name}-step-{step}.json').write_text(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps({'event':'UPDATE','case':name,'step':step,
                  'modes':row['grade_after']['modes'],'hq':row['grade_after']['hq'],
                  'nearest_counts':row['geometry']['current_nearest_counts']}),flush=True)
        result={'case':name,'status':'COMPLETE_EIGHT_G_UPDATES',
                'fit':{k:v for k,v in fit.items() if k!='records'},
                'final_critic_sha256':prior_diagnostic.state_hash(critic.state_dict()),
                'initial_grade':grade(initial,first_step),
                'rows':rows,'seconds':time.perf_counter()-started,
                'last_model_hash':prior_diagnostic.state_hash({'generator':generator.state_dict(),
                    'prior':prior.state_dict(),'critic':critic.state_dict(),
                    'optimizer_g':opt_g.state_dict(),'optimizer_d':opt_d.state_dict()})}
        (output/f'{name}-result.json').write_text(json.dumps(result,allow_nan=False)+'\n')
        return result


def run(output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    warm=loaded(WARM_ARCHIVE,WARM_RAW_SHA)
    cold_state=loaded(COLD_ARCHIVE,COLD_RAW_SHA)
    h_warm=prior_diagnostic.state_hash(warm)
    h_cold=prior_diagnostic.state_hash(cold_state)
    outer_rng=torch.get_rng_state().clone()
    config=json.loads((ROOT/SOURCE[-1]).read_text())
    recipe,_,_=declared_recipe(config)
    gan,regularizer=recipe.make_loss(),recipe.make_gradient_penalty()
    if gan.mode!='rp' or gan.loss_type!='logistic' or regularizer.arm!='b_cap':
        raise AssertionError('wrong native objective')
    declaration={'status':'DECLARED_BEFORE_FIT_AND_UPDATES',
        'scope':'prefitted copied D, then <=8 alternating D/G updates from archived exact saved states; exploratory, not from-scratch qualification',
        'warm':'1325 post-accepted-D, complete its G then seven D/G steps 1326..1332',
        'cold':'1200 post-final-eval, eight D/G steps 1201..1208',
        'common_channel':'independent real/fake Gaussian observations, antithetic 2N by 2N paired Rp; D b_cap on same observed inputs',
        'width':WIDTH,'observation_seed':OBSERVATION_SEED,
        'D':'fit once 40/80 copied critic, then original Adam moment state and constant rate, D own-curvature bound3',
        'G':'original Adam moment state and constant rate, G own-curvature bound.25, no refit',
        'native_stream':'each state saved data/global-torch streams; native late output sigma .029, input sigma0; warm phase starts post-D',
        'fit_bank_overlap':'copied-D fit used first eight native-shaped banks from the same source states; continuation begins at saved stream, so some D fit banks overlap continuation; no quality selection',
        'grade':'fixed4096 draw after each G step; dips recorded, only nonfinite stops',
        'source_sha256':{name:sha((ROOT/name).read_bytes()) for name in SOURCE},
        'warm_state_raw_sha256':WARM_RAW_SHA,'cold_state_raw_sha256':COLD_RAW_SHA,
        'warm_state_hash':h_warm,'cold_state_hash':h_cold}
    output.mkdir(parents=True)
    for name in SOURCE:
        destination=output/'source'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
    (output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    print(json.dumps({'event':'DECLARED','source_sha256':declaration['source_sha256'][SOURCE[0]]}),flush=True)
    results=[]
    for name,state in (('warm1325',warm),('cold1200',cold_state)):
        try:
            results.append(run_case(name,state,gan,regularizer,output))
        except BaseException as error:
            failure={'case':name,'status':'NUMERICAL_OR_REPLAY_ERROR','error':repr(error)}
            (output/f'{name}-error.json').write_text(json.dumps(failure,indent=2)+'\n')
            results.append(failure)
            print(json.dumps({'event':'ERROR',**failure}),flush=True)
            break
    if h_warm!=prior_diagnostic.state_hash(warm) or h_cold!=prior_diagnostic.state_hash(cold_state):
        raise AssertionError('saved input mutated')
    if not torch.equal(torch.get_rng_state(),outer_rng):
        raise AssertionError('global RNG changed')
    summary={'status':'COMPLETE' if len(results)==2 and all(r['status']=='COMPLETE_EIGHT_G_UPDATES' for r in results) else 'ERROR',
             'declaration':declaration,'results':[{k:v for k,v in r.items() if k!='rows'} for r in results],
             'saved_state_and_global_rng_unchanged':True}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.output)


if __name__=='__main__':
    main()
