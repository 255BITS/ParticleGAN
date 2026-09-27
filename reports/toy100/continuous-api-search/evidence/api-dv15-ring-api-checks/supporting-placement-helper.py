"""Frozen vector hosts/scorers, with the unchanged experimental public learner."""
from worker import ROOT, digest, rates
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile
import traceback

import torch
from particlegan import GANTrainer, get_recipe
from particlegan.training import input_noise_std, output_noise_std
import frozen_vector_tasks as vector_tasks
from benchmarks.transfer_suite.public_default_verification import vector_discriminator
from benchmarks.transfer_suite.protocol import test_verdict
from benchmarks.toy100.models import OUTPUT_NOISE_SEED_OFFSET
from lib.toy_models import SimpleMLPGenerator


def placement(value):
    if isinstance(value, torch.Tensor):
        return dict(device=str(value.device), dtype=str(value.dtype), shape=list(value.shape), storage_device=str(value.untyped_storage().device))
    if isinstance(value,dict): return {str(k):placement(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [placement(v) for v in value]
    return None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',default='dv15')
    parser.add_argument('--resume',type=Path,required=True)
    parser.add_argument('--end',type=int,default=2400)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--task',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    plan=ROOT/'benchmarks/transfer_suite/plans/default_comparison.json'
    profile=ROOT/'reports/transfer_suite/unadjusted/leading_profile.json'
    spec=next(job['spec'] for job in json.loads(plan.read_text()) if job['spec']['name']==args.task)
    assert spec['runner']=='vector'
    cfg=vector_tasks.resolve(spec)
    card=json.loads(profile.read_text())['discriminators'].get(args.task)
    recipe=get_recipe(total_steps=None,continuous_policy=args.candidate,input_noise_std=0.,output_noise_warmup=0.,
                      num_particles=cfg['particles'],z_dim=cfg['z_dim'],batch_size=cfg['batch'])
    ring=json.loads((ROOT/('reports/data-drift-api/runs/'+args.candidate+'-single/declaration.json')).read_text())
    for name,expected in ring['source_sha256'].items():
        if name.startswith('particlegan/'):
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    sources=[*sorted((ROOT/'particlegan').glob('*.py')),Path(__file__),plan,profile,
             ROOT/'reports/data-drift-api/vector-protocol.json',ROOT/('reports/data-drift-api/'+args.candidate+'.json')]
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and Path(file).is_file() and Path(file).suffix=='.py' and Path(file).is_relative_to(ROOT):
            sources.append(Path(file))
    fixture_info=json.loads((ROOT/'reports/data-drift-api/vector-fixtures.json').read_text())['fixtures'][args.task]
    fixture_path=ROOT/fixture_info['local_path']
    assert hashlib.sha256(fixture_path.read_bytes()).hexdigest()==fixture_info['sha256']
    sources += [fixture_path, ROOT/'reports/data-drift-api/vector-fixtures.json', ROOT/'reports/data-drift-api/vector-host-receipt.json', ROOT/'reports/data-drift-api/frozen_vector_tasks.py']
    sources += list((ROOT/'benchmarks/transfer_suite').glob('*.py')) + [ROOT/'lib/toy_models.py',ROOT/'lib/toy_metrics.py']
    sources=sorted(set(sources))
    manifest=dict(supplementary=True, resume=str(args.resume.resolve()), resume_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest(), endpoint=args.end, learner_horizon=None, frozen_cfg_steps=cfg['steps'], cadence=50, placement='preserve native checkpoint devices; no map_location', candidate='API-'+args.candidate.upper(),gate='broader_'+args.task,spec=spec,discriminator_card=card,
                  recipe=recipe.to_dict(),serial_backward=True,seed=0,
                  training='actual get_recipe and GANTrainer.step; only host resource dimensions differ from ring',
                  initialization='prior first on CPU using independent seed0/std.5; CPU G then promoted-card D; canonical CPU fixture copied before CUDA/trainer; skipped unused CUDA constructor draws cannot affect independent streams',
                  fixture=fixture_info,evaluation_noise_difference='Explicit isolated2303 branch; historicalK3P402 is not matched evidence',
                  streams='CUDA data0/latent1/penalty2; separate D-real and G-real batches in original order; CUDA frozen scorer target991/directions992',
                  observations='24 original checks, 4096 samples; CUDA latent990, isolated output noise402+1901; live primary, EMA diagnostic',
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.output/'declaration.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(args.output/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in manifest['source_sha256']:
            z.write(ROOT/name,name)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(0)
    prior=recipe.make_prior(init_std=.5,generator=torch.Generator(device='cpu').manual_seed(0))
    generator=SimpleMLPGenerator(cfg['z_dim'],cfg['hidden'],cfg['layers'],2)
    critic=vector_discriminator(spec,card)
    assert all(p.device.type=='cpu' for m in (prior,generator,critic) for p in m.parameters())
    fixture=torch.load(fixture_path,map_location='cpu',weights_only=False)
    groups=[list(generator.parameters())+[prior.z],list(critic.parameters())]
    with torch.no_grad():
        for params,values in zip(groups,fixture):
            assert len(params)==len(values)
            for parameter,value in zip(params,values):
                assert parameter.shape==value.shape
                parameter.copy_(value)
    actual=[[{'shape':list(p.shape),'sha256':hashlib.sha256(p.detach().contiguous().numpy().tobytes()).hexdigest()} for p in group] for group in groups]
    assert actual==fixture_info['initial_parameter_groups']
    (args.output/'canonical-initialization.json').write_text(json.dumps(dict(all_parameter_hashes_match=True,parameters=actual),indent=2)+'\n')
    cpu_initial={k:digest(m.state_dict()) for k,m in [('G',generator),('D',critic),('prior',prior)]}
    data=torch.Generator(device='cuda:0').manual_seed(0)
    trainer=GANTrainer(recipe,generator.cuda(),critic.cuda(),prior=prior.cuda(),seed=0,
                       latent_generator=torch.Generator(device='cuda:0').manual_seed(1),
                       penalty_generator=torch.Generator(device='cuda:0').manual_seed(2),
                       optimizer_options={'foreach':False,'fused':False},serial_backward=True)
    manifest['runtime']=dict(torch=str(torch.__version__),cuda_version=torch.version.cuda,
                             gpu=torch.cuda.get_device_name(0),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                             fp32=True,tf32=False,threads=1,deterministic=True)
    (args.output/'declaration.json').write_text(json.dumps(manifest,indent=2)+'\n')
    torch.save({'trainer':trainer.state_dict(),'data_rng':data.get_state()},args.output/'initial-state.pt')
    (args.output/'initial.json').write_text(json.dumps(dict(cpu_models=cpu_initial,
        cuda_models={k:digest(v) for k,v in trainer.state_dict()['models'].items()},
        trainer=digest(trainer.state_dict()),data_rng=digest(data.get_state())),indent=2)+'\n')

    # Construct all modules/streams before restoring every checkpoint stream.
    checkpoint=torch.load(args.resume,weights_only=False)
    trainer.load_state_dict(checkpoint['trainer'])
    data.set_state(checkpoint['data_rng'])
    restored={'trainer':trainer.state_dict(),'data_rng':data.get_state()}
    assert digest(restored)==digest(checkpoint), 'restore value mismatch'
    assert placement(restored)==placement(checkpoint), 'restore placement mismatch'
    assert trainer.recipe.total_steps is None and cfg['steps']==1200
    assert all(p.device.type=='cuda' for m in (trainer.G,trainer.D,trainer.prior) for p in m.parameters())
    (args.output/'restore.json').write_text(json.dumps(dict(value_exact=True,placement_exact=True,state_sha256=digest(restored),placement=placement(restored)),indent=2)+'\n')
    torch.save(restored,args.output/'restored-state.pt')
    start=trainer.completed_steps

    def real(completed):
        # Device context affects only the frozen sampler, never learner initialization.
        with torch.device('cuda:0'):
            return vector_tasks.sample_target(cfg,cfg['batch'],data,completed)

    @torch.no_grad()
    def measure(completed,ema=False):
        model,prior=(trainer.ema_G,trainer.ema_prior) if ema else (trainer.G,trainer.prior)
        with torch.random.fork_rng(devices=[0]):
            latent,indices=prior.sample(4096,generator=torch.Generator(device='cuda:0').manual_seed(990))
            stream=torch.Generator(device='cuda:0').manual_seed(402+OUTPUT_NOISE_SEED_OFFSET)
            fake=trainer._generate(model,latent,output_noise_std(recipe,completed),stream,indices)
            with torch.device('cuda:0'):
                return vector_tasks.score_samples(fake,cfg,completed)

    started=time.monotonic()
    expected=set(range(start+50,args.end+1,50))
    observations=[]
    try:
        with (args.output/'metrics.jsonl').open('w',buffering=1) as obs, (args.output/'learning-rates.jsonl').open('w',buffering=1) as log:
            for step in range(start+1,args.end+1):
                stats=trainer.step(real(step),generator_real=lambda:real(step),collect_stats=step in expected)
                log.write(json.dumps(dict(step=step,**rates(trainer),policy=trainer.controller.diagnostics(),
                                         critic=trainer.penalty.diagnostics(),input_noise=input_noise_std(recipe,step-1),
                                         output_noise=output_noise_std(recipe,step-1)))+'\n')
                if step in expected:
                    before=digest([trainer.state_dict(),data.get_state()])
                    row=dict(step=step,**measure(step),ema=measure(step,True))
                    assert digest([trainer.state_dict(),data.get_state()])==before
                    observations.append(row)
                    obs.write(json.dumps(row)+'\n')
                    print(json.dumps(row),flush=True)
                    state={'trainer':trainer.state_dict(),'data_rng':data.get_state()}
                    if step in (1250,1300,args.end):
                        torch.save(state,args.output/f'state-{step}.pt')
                        (args.output/f'placement-{step}.json').write_text(json.dumps(placement(state),sort_keys=True)+'\n')
        from benchmarks.transfer_suite.protocol import requirements
        thresholds=requirements(spec)
        def assess(row):
            checks=[dict(metric=k,value=row[k],op=op,threshold=bound,
                         margin=(row[k]-bound if op=='>=' else bound-row[k])) for k,op,bound in thresholds]
            return dict(step=row['step'],passed=all(c['margin']>=0 for c in checks),checks=checks)
        assessments=[assess(row) for row in observations]
        suffix=0
        for row in reversed(assessments):
            if not row['passed']: break
            suffix+=1
        status='PASS' if suffix>=5 else 'FAIL'
        metrics=dict(supplementary_only=True,original_frozen_gate='FAIL4/24/suffix2 unchanged',
                     passing=sum(x['passed'] for x in assessments),total=len(assessments),
                     final_suffix=suffix,failures=[x['step'] for x in assessments if not x['passed']],
                     assessments=assessments,final=observations[-1],policy=trainer.controller.diagnostics())
        if args.reference:
            comparisons=[]
            reference_rows={r['step']:r for r in map(json.loads,(args.reference/'metrics.jsonl').read_text().splitlines())}
            for row in observations:
                reference=torch.load(args.reference/f"state-{row['step']}.pt",weights_only=False)
                state=torch.load(args.output/f"state-{row['step']}.pt",weights_only=False)
                comparisons.append(dict(step=row['step'],values=digest(reference)==digest(state),placement=placement(reference)==placement(state),observation=row==reference_rows[row['step']]))
            metrics['replay']=comparisons
            status='PASS' if all(all(x[k] for k in ('values','placement','observation')) for x in comparisons) else 'FAIL'
        torch.save({'trainer':trainer.state_dict(),'data_rng':data.get_state()},args.output/'final-state.pt')
    except Exception as error:
        status='ERROR'
        metrics={'error':repr(error),'traceback':traceback.format_exc()}
    row=dict(candidate='API-'+args.candidate.upper(),gate=('fresh_process_replay_'+str(start)+'_'+str(args.end) if args.reference else 'supplementary_resumed_1200_2400_'+args.task),status=status,seconds=time.monotonic()-started,
             metrics=metrics,artifact=str(args.output.resolve()))
    (args.output/'result.json').write_text(json.dumps(row,indent=2)+'\n')
    with (ROOT.parent/'tests.jsonl').open('a') as log:
        log.write(json.dumps(row)+'\n')
    (args.output/'artifact-sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.output.iterdir() if p.is_file() and p.name!='artifact-sha256.json'},indent=2)+'\n')
    print(json.dumps(row),flush=True)


if __name__=='__main__':
    main()
