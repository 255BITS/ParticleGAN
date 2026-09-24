"""Continue only a ten-gate survivor from its own live cold ring state.

Reconstruct frozen modules, then restore weights, Adam states, and both RNGs.
No cold fitting is repeated. An optional fixed output-bias perturbation creates
an error in the generator while leaving the data distribution unchanged.
"""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from particlegan import ParticlePrior
from benchmarks import learned_lr_evaluation as bridge
from benchmarks.locked_shared import mode_hold
from benchmarks.toy100.continuous_probe import _window, RECOVERY_DEADLINE
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy, wrap_input, wrap_output
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
from critic_signal import signal_policy
from critic_signal_screen import ORDER, append


def run(candidate, output, steps, resume=None, perturb=False, verify_only=False):
    status=json.loads((candidate/'status.json').read_text())
    stages={r['gate']:r['status'] for r in status['stages']}
    required=['trajectory','mode_hold'] if verify_only else ORDER
    if any(stages.get(gate)!='PASS' for gate in required):
        raise ValueError('own-state continuation requires all ten cheap gates to pass')
    if steps!=1200:
        raise ValueError('the declared continuation budget is 1200 updates')
    if perturb and resume is None:
        raise ValueError('error recovery must continue the qualified hold checkpoint')
    path=resume or candidate/'mode_hold'/'final-state.pt'
    cold=next(r for r in status['stages'] if r['gate']=='mode_hold')
    if resume is None and cold.get('checkpoint_sha256'):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=cold['checkpoint_sha256']:
            raise ValueError('own cold checkpoint differs from the completed gate receipt')
    state=torch.load(path,map_location='cpu',weights_only=False)
    if perturb:
        previous=json.loads(path.with_name('summary.json').read_text())
        if previous['status']!='PASS' or previous['candidate']!=candidate.name:
            raise ValueError('error recovery cannot follow a failed hold')
    config=json.loads((candidate/'config.json').read_text())
    options=json.loads((candidate/'options.json').read_text())
    base,noise,_=declared_recipe(config)
    output.mkdir(parents=True,exist_ok=False)
    declaration=dict(candidate=candidate.name,parent_state=str(path.resolve()),
                     parent_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     config=config,options=options,steps=steps,perturb=perturb,
                     perturbation=[.35,0.] if perturb else None,
                     eligibility='adversarial; fixed rates; same data; live state and moments retained')
    declaration['parent_manifest_sha256']=hashlib.sha256((candidate.parent/'manifest.json').read_bytes()).hexdigest()
    declaration['parent_source_archive']=str((candidate.parent/'source.tar.gz').resolve())
    declaration['source_sha256']={}
    for name in ('critic_signal_continue.py','critic_signal.py','continuous_candidates.py'):
        source=Path(__file__).with_name(name)
        declaration['source_sha256'][name]=hashlib.sha256(source.read_bytes()).hexdigest()
        (output/name).write_bytes(source.read_bytes())
    (output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    torch.set_num_threads(1)
    points=[]
    started=time.perf_counter()
    with signal_policy(options) as receipt:
        stream=torch.Generator().manual_seed(0)
        n,z=state['models']['prior']['z'].shape
        prior=ParticlePrior(n,z,init_std=.5,generator=stream)
        generator=mode_hold.SimpleMLPGenerator(mode_hold.Z_DIM,mode_hold.HIDDEN,mode_hold.N_HIDDEN,2)
        critic=mode_hold.SimpleMLPDiscriminator(2,mode_hold.HIDDEN,mode_hold.N_HIDDEN,mode_hold.FOURIER)
        policy=NoisePolicy(noise['output_noise_std'],noise['input_noise_std'],
                           noise['input_noise_anneal_end'],1200,output_noise_warmup=noise['output_noise_warmup'])
        policy.__dict__.update(deepcopy(state['noise_policy']))
        # Registration was already performed in the original frozen host.
        count=policy.generator_base_parameters
        policy.generator_base_parameters=None
        generator=wrap_output(generator,policy)
        assert policy.generator_base_parameters==count
        critic=wrap_input(critic,policy)
        models=dict(generator=generator,critic=critic,prior=prior)
        for name,model in models.items():
            model.load_state_dict(state['models'][name])
        opt_g,opt_d=base.make_optimizers(generator,critic,prior)
        opt_g.load_state_dict(state['optimizers']['opt_g'])
        opt_d.load_state_dict(state['optimizers']['opt_d'])
        bridge.optimizer_role(opt_g,dict(opt_g=opt_g,opt_d=opt_d))
        bridge.optimizer_role(opt_d,dict(opt_g=opt_g,opt_d=opt_d))
        stream.set_state(state['stream_rng'])
        torch.set_rng_state(state['torch_rng'])
        means=mode_hold.ring_means()  # Frozen data sampler and evaluator only.
        gan=base.make_loss()
        regularizer=base.make_gradient_penalty()
        ema_g=deepcopy(state['ema_g'])
        ema_z=state['ema_z'].clone()

        @torch.no_grad()
        def measure(step):
            with torch.random.fork_rng(devices=[]),policy.evaluation(step):
                latent,_=prior.sample(mode_hold.EVAL_N,generator=torch.Generator().manual_seed(9))
                samples=generator(latent)
                metrics=mode_hold.diversity(samples,means,detailed=True)
            return dict(step=step,**metrics),samples.detach().clone()

        initial,draw=measure(state['step'])
        if not torch.equal(draw,state['samples']):
            raise RuntimeError('restored generator/evaluation state differs from its own saved sample')
        if verify_only:
            assert all(int(v['step'])==state['step'] for opt in (opt_g,opt_d) for v in opt.state.values())
            result=dict(candidate='regression',gate='own_state_checkpoint_restore',status='PASS',
                        seconds=time.perf_counter()-started,metrics={'passed':1,'candidate':candidate.name,
                        'restored_steps':state['step'],'samples_bitwise_identical':True,'training_updates':0},
                        artifact=str((output/'declaration.json').resolve()))
            (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
            return result
        if perturb:
            with torch.no_grad():
                generator.model.net[-1].bias.add_(torch.tensor([.35,0.]))
        opened,_=measure(state['step'])
        print(json.dumps(dict(event='RESTORED',candidate=candidate.name,initial=initial,opened=opened)),flush=True)
        for completed in range(state['step'],state['step']+steps):
            policy.set_step(completed)
            real=mode_hold.sample_ring(means,mode_hold.BATCH,mode_hold.SIGMA,stream)
            latent,_=prior.sample(mode_hold.BATCH,generator=stream)
            with policy.discriminator():
                fake=generator(latent).detach()
            loss_d=gan.d_loss(critic(real),critic(fake))
            loss_d=loss_d+regularizer(critic,real,fake,step=completed+1)
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
            latent,_=prior.sample(mode_hold.BATCH,generator=stream)
            fake=generator(latent)
            if gan.mode in ('rp','ra'):
                real_g=mode_hold.sample_ring(means,mode_hold.BATCH,mode_hold.SIGMA,stream)
                loss_g=gan.g_loss(critic(fake),critic(real_g))
            else:
                loss_g=gan.g_loss(critic(fake))
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()
            with torch.no_grad():
                for ema,param in zip(ema_g,generator.parameters()):
                    ema.mul_(base.ema_decay).add_(param,alpha=1-base.ema_decay)
                ema_z.mul_(base.ema_decay).add_(prior.z,alpha=1-base.ema_decay)
            point,draw=measure(completed+1)
            points.append(point)
            if (completed+1)%100==0:
                print(json.dumps(dict(event='CONTINUE',step=completed+1,modes=point['modes'],hq=point['hq'])),flush=True)
        final_step=state['step']+steps
        torch.save(dict(step=final_step,task='mode_hold',torch_rng=torch.get_rng_state(),
                        stream_rng=stream.get_state(),noise_policy=deepcopy(policy.__dict__),
                        models={k:deepcopy(v.state_dict()) for k,v in models.items()},
                        optimizers={'opt_g':deepcopy(opt_g.state_dict()),'opt_d':deepcopy(opt_d.state_dict())},
                        ema_g=ema_g,ema_z=ema_z,samples=draw),output/'final-state.pt')
    for update in receipt['updates']:
        for group in update['groups']:
            expected=base.lr*{'g':1.,'d':base.d_lr_mult,'prior':base.prior_lr_mult}[group['role']]
            assert group['lr']==expected
    (output/'optimizer-receipt.json.gz').write_bytes(gzip.compress(json.dumps(receipt).encode(),mtime=0))
    (output/'metrics.json').write_text(json.dumps(points,indent=2)+'\n')
    window=_window(points)
    late=_window([p for p in points if p['step']>=state['step']+RECOVERY_DEADLINE])
    okay=late['pass_all'] if perturb else window['pass_all']
    result=dict(candidate=candidate.name,gate='own_state_recovery' if perturb else 'own_state_hold',
                status='PASS' if okay else 'FAIL',seconds=time.perf_counter()-started,
                metrics=dict(initial=initial,opened=opened,window=window,recovery_deadline_window=late,final=points[-1]),
                artifact=str((output/'metrics.json').resolve()))
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ledger',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=1200)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--perturb',action='store_true')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    result=run(args.candidate,args.output,args.steps,args.resume,args.perturb,args.verify_only)
    append(args.ledger,result)
    print(json.dumps(result),flush=True)
