"""Bounded selected-H warm-state stability probes; never production qualification.

Control is the archived H alternating update. All policies restore exactly the
same acquired G/D/prior, Adam and RNG state before any update. The declared
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
import traceback

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

SOURCE = REPO
EVIDENCE = SOURCE/'reports/toy100/critic_signal_attempt'
CANDIDATE = EVIDENCE/'batch-h/h_n05r06_mixup_c0p01_lr15'
STATE_SHA = '799181c2a68df02ee5a7a5963b4a5e4fbf23677623c4fcd9f8757460944155b2'
ARCHIVE_SHA = 'ed4a61dcfebc13f1134630db096aad043d60e4846083b4a1384e871ee9ebb334'
from dynamics import Updates, POLICIES


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
    declaration=dict(candidate=variant,parent_candidate=CANDIDATE.name,variant=variant,policy=POLICIES[variant],
                     parent_checkpoint_sha256=STATE_SHA,parent_source_archive_sha256=ARCHIVE_SHA,
                     runner_sha256=sha(Path(__file__)),steps=steps,
                     stop_on_first_failure=not full_window,config=config,options=options,
                     scope='warm-state research probe; requires new cold acquisition and all gates before promotion',
                     shared_gate_eligible=False,seed_experiment=False,
                     actual_ring_resources=dict(particles=12,z_dim=4,batch=128,hidden=96,layers=3),
                     eligible_objectives='H logistic RpGAN; D R1+R2 and mixup consistency only',
                     fixed_lrs=dict(g=.0015*POLICIES[variant].get('g_rate_mult',1.),d=.0015,prior=.003*POLICIES[variant].get('prior_rate_mult',POLICIES[variant].get('g_rate_mult',1.))),
                     stop_rule='fail if any dense check has modes != 8 or HQ < 0.90; no feedback into updates')
    write(output/'declaration.json',declaration)
    for name in ('dynamics_runner.py', 'dynamics.py', 'stability_runner.py'):
        (output/name).write_bytes((PREP_ROOT/name).read_bytes())
    declaration['actual_candidate_sources'] = {name: sha(output/name) for name in
        ('dynamics_runner.py', 'dynamics.py', 'stability_runner.py')}
    write(output/'declaration.json', declaration)
    started=time.perf_counter(); points=[]; losses=[]
    with signal_policy(options) as receipt:
        models,opts,stream,policy=restore(state,base,noise)
        generator=models['generator']; prior=models['prior']
        # Fixed declared override; applied once, before any update, never scheduled.
        for group in opts['opt_g'].param_groups:
            group['lr'] = declaration['fixed_lrs']['prior' if group.get('_comparison_prior',False) else 'g']
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
                        ema_g=ema_g,ema_z=ema_z,samples=draw,update_state=update.state_dict()),output/'final-state.pt')
        optimizer_steps={key:sorted({int(s['step']) for s in opt.state.values()}) for key,opt in opts.items()}
        for row in receipt['updates']:
            for group in row['groups']:
                assert group['lr']==declaration['fixed_lrs'][group['role']]
                assert group['betas']==[0.,.999] and group['eps']==1e-8
    window=_window(points)
    passed=len(points)==steps and window['pass_all']
    result=dict(candidate=variant,parent_candidate=CANDIDATE.name,variant=variant,gate='warm_probe_200',
                status='PASS' if passed else 'FAIL',initial=initial,window=window,
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
    parser.add_argument('--ledger',type=Path)
    args=parser.parse_args()
    started=time.perf_counter()
    try:
        result=run(args.variant,args.output,args.steps,args.diagnostic_full_window)
    except Exception:
        error=traceback.format_exc()
        print(error,flush=True)
        if args.ledger:
            with args.ledger.open('a') as ledger:
                ledger.write(json.dumps(dict(candidate=args.variant,gate='warm_probe_200',status='ERROR',
                    seconds=time.perf_counter()-started,metrics={'error':error},artifact=str(args.output.resolve())))+'\n')
        raise
    if args.ledger:
        with args.ledger.open('a') as ledger:
            ledger.write(json.dumps(dict(candidate=args.variant,gate=result['gate'],status=result['status'],
                seconds=result['elapsed_seconds'],metrics={k:result[k] for k in
                    ('window','final','work','optimizer_steps')},artifact=str(args.output.resolve())))+'\n')
