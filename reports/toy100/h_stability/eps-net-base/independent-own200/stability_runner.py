"""Bounded selected-H warm-state stability probes; never production qualification.

Control is the archived H alternating update. All policies restore exactly the
same acquired G/D/prior, Adam and RNG state before any update. The three fixed
interventions change only the update algorithm/work budget, never target-aware
losses, H noise, architecture, or optimizer hyperparameters.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

PREP_ROOT = Path(__file__).resolve().parent
REPO = PREP_ROOT.parents[2]
sys.path[:0] = [str(REPO), str(REPO/'reports/toy100')]
import torch
from particlegan import ParticlePrior
from benchmarks import learned_lr_evaluation as bridge
from benchmarks.locked_shared import mode_hold
from benchmarks.toy100.continuous_probe import _window
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy, wrap_input, wrap_output
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
from critic_signal import signal_policy
from continuous_screen import verify_receipt

SOURCE = REPO
EVIDENCE = SOURCE/'reports/toy100/critic_signal_attempt'
CANDIDATE = EVIDENCE/'batch-h/h_n05r06_mixup_c0p01_lr15'
STATE_SHA = '799181c2a68df02ee5a7a5963b4a5e4fbf23677623c4fcd9f8757460944155b2'
ARCHIVE_SHA = 'ed4a61dcfebc13f1134630db096aad043d60e4846083b4a1384e871ee9ebb334'
POLICIES = {
    'control': dict(family='archived H alternating Adam', d_draws=1, g_draws=1,
                    d_commits=1, g_commits=1, provisional_pairs=0),
    'critic_refresh2': dict(family='critic tracking', d_draws=2, g_draws=1,
                           d_commits=2, g_commits=1, provisional_pairs=0),
    'average2': dict(family='gradient variance', d_draws=2, g_draws=2,
                     d_commits=1, g_commits=1, provisional_pairs=0),
    'extra_adam': dict(family='game prediction/correction', d_draws=2, g_draws=2,
                       d_commits=1, g_commits=1, provisional_pairs=1),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def print_json(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def verify_sources():
    manifest=json.loads((EVIDENCE/'batch-h/manifest.json').read_text())
    assert sha(EVIDENCE/'batch-h/source.tar.gz') == ARCHIVE_SHA
    mismatches=[name for name,digest in manifest['source_sha256'].items()
                if sha(REPO/name) != digest]
    if mismatches:
        raise RuntimeError(f'changed archived H sources: {mismatches}')
    assert sha(CANDIDATE/'mode_hold/final-state.pt') == STATE_SHA
    status=json.loads((CANDIDATE/'status.json').read_text())
    assert len(status['stages'])==10 and all(s['status']=='PASS' for s in status['stages'])


def restore(state, base, noise):
    """Exactly the archived hold reconstruction, including policy RNG objects."""
    stream=torch.Generator().manual_seed(0)
    n,z=state['models']['prior']['z'].shape
    prior=ParticlePrior(n,z,init_std=.5,generator=stream)
    generator=mode_hold.SimpleMLPGenerator(mode_hold.Z_DIM,mode_hold.HIDDEN,mode_hold.N_HIDDEN,2)
    critic=mode_hold.SimpleMLPDiscriminator(2,mode_hold.HIDDEN,mode_hold.N_HIDDEN,mode_hold.FOURIER)
    policy=NoisePolicy(noise['output_noise_std'],noise['input_noise_std'],
                       noise['input_noise_anneal_end'],1200,
                       output_noise_warmup=noise['output_noise_warmup'])
    policy.__dict__.update(deepcopy(state['noise_policy']))
    count=policy.generator_base_parameters
    policy.generator_base_parameters=None
    generator=wrap_output(generator,policy)
    assert policy.generator_base_parameters==count
    critic=wrap_input(critic,policy)
    models=dict(generator=generator,critic=critic,prior=prior)
    for name, model in models.items():
        model.load_state_dict(state['models'][name])
    opt_g,opt_d=base.make_optimizers(generator,critic,prior)
    opt_g.load_state_dict(state['optimizers']['opt_g'])
    opt_d.load_state_dict(state['optimizers']['opt_d'])
    bridge.optimizer_role(opt_g,dict(opt_g=opt_g,opt_d=opt_d))
    bridge.optimizer_role(opt_d,dict(opt_g=opt_g,opt_d=opt_d))
    stream.set_state(state['stream_rng'])
    torch.set_rng_state(state['torch_rng'])
    assert all(int(v['step'])==1200 for opt in (opt_g,opt_d) for v in opt.state.values())
    return models,dict(opt_g=opt_g,opt_d=opt_d),stream,policy


class Updates:
    """No mode centers, labels or evaluation values enter these updates.

    sample_real is the frozen minibatch sampler. H's fake sampling and all
    observation noise remain in the archived prior/model/policy modules.
    """
    def __init__(self, models, opts, stream, policy, sample_real, base, receipt):
        self.g=models['generator']; self.d=models['critic']; self.prior=models['prior']
        self.opts=opts; self.models=models; self.stream=stream; self.policy=policy
        self.sample_real=sample_real; self.gan=base.make_loss()
        self.regularizer=base.make_gradient_penalty(); self.receipt=receipt
        self.work=dict(d_backwards=0,g_backwards=0,d_commits=0,g_commits=0,
                       d_provisional=0,g_provisional=0)

    def params(self, player):
        return [p for g in self.opts['opt_'+player].param_groups for p in g['params']]

    def gradients(self, player, completed, draws=1):
        optimizer=self.opts['opt_'+player]
        # For a single draw, preserve the original zero_grad placement exactly.
        if draws>1:
            optimizer.zero_grad()
        total=0.
        for _ in range(draws):
            if player=='d':
                real=self.sample_real()
                latent,_=self.prior.sample(mode_hold.BATCH,generator=self.stream)
                with self.policy.discriminator():
                    fake=self.g(latent).detach()
                loss=self.gan.d_loss(self.d(real),self.d(fake))
                loss=loss+self.regularizer(self.d,real,fake,step=completed+1)
            else:
                latent,_=self.prior.sample(mode_hold.BATCH,generator=self.stream)
                fake=self.g(latent)
                real=self.sample_real()
                loss=self.gan.g_loss(self.d(fake),self.d(real))
            if draws==1:
                optimizer.zero_grad()
            (loss if draws==1 else loss/draws).backward()
            total+=float(loss.detach())/draws
            self.work[player+'_backwards']+=1
        return total

    def commit(self, player, step, phase='committed'):
        begin=len(self.receipt['updates'])
        self.opts['opt_'+player].step()
        self.work[player+('_provisional' if phase=='predictor' else '_commits')]+=1
        for row in self.receipt['updates'][begin:]:
            row.update(outer_step=step,phase=phase)

    def round(self, variant, completed):
        if variant!='extra_adam':
            draws=2 if variant=='average2' else 1
            d_losses=[]
            for _ in range(2 if variant=='critic_refresh2' else 1):
                d_losses.append(self.gradients('d',completed,draws))
                self.commit('d',completed+1)
            g_loss=self.gradients('g',completed,draws)
            self.commit('g',completed+1)
            return dict(loss_d=sum(d_losses)/len(d_losses),loss_g=g_loss)
        # Predictor is one ordinary alternating H round. Its temporary Adam
        # moments are discarded. Corrector uses independent minibatches at
        # the full lookahead G/D state, then applies those gradients once to
        # the original parameters and moments. Training RNGs advance through
        # both draws; there is no noise reset or nested fitting to a target.
        anchor_models={k:deepcopy(v.state_dict()) for k,v in self.models.items()}
        anchor_opts={k:deepcopy(v.state_dict()) for k,v in self.opts.items()}
        self.gradients('d',completed)
        self.commit('d',completed+1,'predictor')
        self.gradients('g',completed)
        self.commit('g',completed+1,'predictor')
        loss_d=self.gradients('d',completed)
        grad_d=[p.grad.detach().clone() for p in self.params('d')]
        loss_g=self.gradients('g',completed)
        grad_g=[p.grad.detach().clone() for p in self.params('g')]
        for key,model in self.models.items():
            model.load_state_dict(anchor_models[key])
        for key,optimizer in self.opts.items():
            optimizer.load_state_dict(anchor_opts[key])
        for player,gradients in [('d',grad_d),('g',grad_g)]:
            for param,gradient in zip(self.params(player),gradients):
                param.grad=gradient
            self.commit(player,completed+1)
        return dict(loss_d=loss_d,loss_g=loss_g)


def run(variant, output, steps=200, full_window=False):
    verify_sources()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:
        torch.set_num_interop_threads(1)
    state=torch.load(CANDIDATE/'mode_hold/final-state.pt',map_location='cpu',weights_only=False)
    config=json.loads((CANDIDATE/'config.json').read_text())
    options=json.loads((CANDIDATE/'options.json').read_text())
    base,noise,_=declared_recipe(config)
    assert base.lr_floor==1 and base.prior_reg==0
    output.mkdir(parents=True,exist_ok=False)
    declaration=dict(candidate=CANDIDATE.name,variant=variant,policy=POLICIES[variant],
                     parent_checkpoint_sha256=STATE_SHA,parent_source_archive_sha256=ARCHIVE_SHA,
                     runner_sha256=sha(Path(__file__)),steps=steps,
                     stop_on_first_failure=not full_window,config=config,options=options,
                     scope='warm-state research probe; requires new cold acquisition and all gates before promotion',
                     shared_gate_eligible=False,seed_experiment=False,
                     actual_ring_resources=dict(particles=12,z_dim=4,batch=128,hidden=96,layers=3),
                     eligible_objectives='H logistic RpGAN; D R1+R2 and mixup consistency only',
                     fixed_lrs=dict(g=base.lr,d=base.lr*base.d_lr_mult,prior=base.lr*base.prior_lr_mult),
                     stop_rule='fail if any dense check has modes != 8 or HQ < 0.90; no feedback into updates')
    write(output/'declaration.json',declaration)
    (output/'stability_runner.py').write_bytes(Path(__file__).read_bytes())
    started=time.perf_counter(); points=[]; losses=[]
    with signal_policy(options) as receipt:
        models,opts,stream,policy=restore(state,base,noise)
        generator=models['generator']; prior=models['prior']
        means=mode_hold.ring_means()
        sample_real=lambda: mode_hold.sample_ring(means,mode_hold.BATCH,mode_hold.SIGMA,stream)
        update=Updates(models,opts,stream,policy,sample_real,base,receipt)
        ema_g=deepcopy(state['ema_g']); ema_z=state['ema_z'].clone()

        @torch.no_grad()
        def measure(step):
            saved_global=torch.get_rng_state().clone()
            saved_stream=stream.get_state().clone()
            saved_input=policy.input_stream.get_state().clone()
            with torch.random.fork_rng(devices=[]),policy.evaluation(step):
                latent,_=prior.sample(mode_hold.EVAL_N,generator=torch.Generator().manual_seed(9))
                samples=generator(latent)
                metrics=mode_hold.diversity(samples,means,detailed=True)
            assert torch.equal(saved_global,torch.get_rng_state())
            assert torch.equal(saved_stream,stream.get_state())
            assert torch.equal(saved_input,policy.input_stream.get_state())
            return dict(step=step,**metrics),samples.detach().clone()

        initial,draw=measure(state['step'])
        if not torch.equal(draw,state['samples']):
            raise RuntimeError('H initial evaluation is not bitwise equal to retained cold state')
        # Match the archived continuation's initial and opened observations.
        opened,_=measure(state['step'])
        assert opened==initial
        print_json(dict(event='RESTORED',variant=variant,step=1200,
                        modes=initial['modes'],hq=initial['hq']))
        with (output/'metrics.jsonl').open('w',buffering=1) as metrics_log:
            for completed in range(1200,1200+steps):
                policy.set_step(completed)
                loss=update.round(variant,completed)
                loss=dict(step=completed+1,**loss); losses.append(loss)
                with torch.no_grad():
                    for ema,param in zip(ema_g,generator.parameters()):
                        ema.mul_(base.ema_decay).add_(param,alpha=1-base.ema_decay)
                    ema_z.mul_(base.ema_decay).add_(prior.z,alpha=1-base.ema_decay)
                point,draw=measure(completed+1); points.append(point)
                passing=point['modes']==8 and point['hq']>=.9
                metrics_log.write(json.dumps(dict(**point,**{k:v for k,v in loss.items() if k!='step'}))+'\n')
                if (completed+1)%25==0 or not passing:
                    print_json(dict(event='CHECK',variant=variant,step=completed+1,
                                    modes=point['modes'],hq=point['hq'],passing=passing,**{k:v for k,v in loss.items() if k!='step'}))
                if not passing and not full_window:
                    break
        final_step=points[-1]['step']
        torch.save(dict(step=final_step,task='mode_hold',torch_rng=torch.get_rng_state(),
                        stream_rng=stream.get_state(),noise_policy=deepcopy(policy.__dict__),
                        models={k:deepcopy(v.state_dict()) for k,v in models.items()},
                        optimizers={k:deepcopy(v.state_dict()) for k,v in opts.items()},
                        ema_g=ema_g,ema_z=ema_z,samples=draw),output/'final-state.pt')
        optimizer_steps={key:sorted({int(s['step']) for s in opt.state.values()}) for key,opt in opts.items()}
        verify_receipt(receipt, config, task='warm_probe')
        assert len(receipt['updates']) == 2 * len(points)
    window=_window(points)
    passed=len(points)==steps and window['pass_all']
    result=dict(candidate=CANDIDATE.name,variant=variant,gate='selected_h_warm_stability_probe',
                status=('PASS' if steps==1200 else 'SHORT_PASS') if passed else 'FAIL',initial=initial,window=window,
                final=points[-1],work=update.work,optimizer_steps=optimizer_steps,
                elapsed_seconds=time.perf_counter()-started,
                full_budget=len(points)==steps,hold_budget_complete=len(points)==1200,shared_gate_eligible=False)
    write(output/'metrics.json',points); write(output/'losses.json',losses)
    write(output/'summary.json',result)
    (output/'optimizer-receipt.json.gz').write_bytes(gzip.compress(json.dumps(receipt).encode(),mtime=0))
    print_json(dict(event='DONE',variant=variant,status=result['status'],checks=len(points),
                    passing_checks=window['passing_checks'],first_failure=next(iter(window['failing_steps']),None),
                    final_modes=points[-1]['modes'],final_hq=points[-1]['hq'],work=update.work,
                    seconds=result['elapsed_seconds']))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=POLICIES,default='control')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,choices=(200,1200),default=200,
                        help='200-update screen or complete 1200-update warm hold')
    parser.add_argument('--diagnostic-full-window',action='store_true',
                        help='record the complete declared window after failures; failure verdict is unchanged')
    args=parser.parse_args()
    run(args.variant,args.output,args.steps,args.diagnostic_full_window)
