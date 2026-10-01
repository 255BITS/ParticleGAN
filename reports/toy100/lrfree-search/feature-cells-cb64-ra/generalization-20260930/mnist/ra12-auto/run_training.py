"""CUDA port of the frozen matched-init CB64-RA/E22 learned-model acceptance."""
import common  # Runtime environment must be installed before scientific imports.
from common import (ROOT, PREV, SEED, STEPS, CHECKPOINTS, DEVICE, VARIANTS, PROBLEMS,
                    sha, require, log, write_json, verify_inputs, configure_cuda,
                    select_package, exclusive_learned_gpu, execution_receipt, rng_cpu_buffers)
import argparse
import json
from pathlib import Path
import resource
import sys
import time
import traceback

sys.path.insert(0, str(PREV))
import torch
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from models_metrics import networks, Evaluator, encode, model_hash, tensor_hash, manifold_precision_recall
from adapter import draw_samples, toy_metrics, initialize_fixture_models, selection_diagnostics
from contracts import validate_checkpoint_state, validate_initial_streams, verify_fixture_sources, toy_gate
import numpy as np


def frechet(a,b):
    a,b=a.detach().cpu().double().numpy(),b.detach().cpu().double().numpy()
    ca,cb=np.cov(a,rowvar=False),np.cov(b,rowvar=False)
    w,v=np.linalg.eigh(ca);root=(v*np.sqrt(w.clip(0))[None,:])@v.T
    mid=root@cb@root
    return float(max(0.,np.sum((a.mean(0)-b.mean(0))**2)+np.trace(ca)+np.trace(cb)-2*np.sqrt(np.linalg.eigvalsh((mid+mid.T)/2).clip(0)).sum()))


def embedding_score(real,fake):
    return dict(embedding_frechet=frechet(real,fake),**manifold_precision_recall(real,fake))


class ImageEvaluation:
    def __init__(self,device):
        self.model=Evaluator().to(device).eval()
        self.model.load_state_dict(torch.load(PREV/'evaluator.pt',map_location=device,weights_only=True)['model'])
        train=MNIST(PREV/'data',train=True,download=False);test=MNIST(PREV/'data',train=False,download=False)
        _,features=encode(self.model,train.data[:5000].to(device).float().unsqueeze(1)/127.5-1)
        mean,std=features.mean(0),features.std(0)
        self.active=std>float(std.max())*1e-6
        self.mean,self.std=mean[self.active],std[self.active]
        prob,f=encode(self.model,test.data.to(device).float().unsqueeze(1)/127.5-1)
        self.ref=f[:5000]
        self.active_ref=(self.ref[:,self.active]-self.mean)/self.std
        self.mass=torch.bincount(test.targets.to(device),minlength=10).double()/len(test)
        self.accuracy=float((prob.argmax(1)==test.targets.to(device)).float().mean())
        self.control=self.score(prob[5000:],f[5000:])
        self.train=train.data.to(device)

    def score(self,p,f):
        confidence,classes=p.max(1)
        mass=torch.bincount(classes,minlength=10).double()/len(p)
        confident_mass=torch.bincount(classes[confidence>=.9],minlength=10).double()/len(p)
        return dict(class_mass_tv=float((mass-self.mass).abs().sum()/2),predicted_class_mass=mass.tolist(),
                    confident_class_mass=confident_mass.tolist(),confident_class_coverage=int((confident_mass>=.01).sum()),
                    confident_fraction=float((confidence>=.9).float().mean()),mean_classifier_confidence=float(confidence.mean()),
                    raw_embedding=embedding_score(self.ref,f),
                    active_embedding=embedding_score(self.active_ref,(f[:,self.active]-self.mean)/self.std))

    @torch.no_grad()
    def evaluate(self,trainer):
        images=draw_samples(trainer,4096,SEED+100)
        p,f=encode(self.model,images)
        return dict(**self.score(p,f),pixel_clipping_fraction=float((images.abs()>1).float().mean()))


def diagnostics(trainer):
    out=dict(output_sigma=trainer.output_sigma(),lr=[[g['lr'] for g in opt.param_groups] for opt in (trainer.opt_g,trainer.opt_d)])
    for name in ('controller','lr_settle','birth_death'):
        obj=getattr(trainer,name,None)
        if obj is not None:out[name]=obj.diagnostics()
    ev=getattr(trainer,'row_evidence',None)
    if ev is not None:
        neff=ev.W.square()/ev.S.clamp_min(1e-30)
        out['row_evidence']=dict(counters=dict(ev.counters),fraction=ev.fraction,n_eff_max=float(neff.max()),
                                 asymptotic_cap=99.,required=384.)
    out['backend_selection']=selection_diagnostics(trainer)
    if trainer.policy.surprise is not None:out['surprise']=trainer.policy.surprise.diagnostics()
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=VARIANTS,required=True)
    parser.add_argument('--problem',choices=PROBLEMS,required=True)
    args=parser.parse_args()
    outdir=ROOT/'training'/args.problem/args.variant
    # No append/rerun can silently replace a completed or partial attempt.
    require(not outdir.exists() or not any(outdir.iterdir()), f'run directory already contains evidence: {outdir}')
    outdir.mkdir(parents=True,exist_ok=True)
    trainer=None; phase='verify_frozen_inputs'; elapsed=0.; curves=[]
    process_start=time.perf_counter()
    try:
        with exclusive_learned_gpu():
            inputs=verify_inputs()
            selected=select_package(inputs,args.variant)
            from particlegan.recipes import Recipe
            from particlegan.particle_prior import ParticlePrior
            from particlegan.training import GANTrainer
            phase='configure_cuda'
            runtime=configure_cuda(torch)
            device=torch.device(DEVICE)
            config=json.loads(Path(selected['config_path']).read_text())
            config.update(z_dim=128,num_particles=1024,batch_size=128)
            require('initialization' not in config, 'current public Recipe owns no initialization')
            fixture_sources=verify_fixture_sources()
            recipe=Recipe(**config)
            phase='matched_initialization'
            G,D=networks(args.problem)
            prior=ParticlePrior(1024,128,generator=torch.Generator().manual_seed(SEED+1))
            initialize_fixture_models(G,D)
            trainer=GANTrainer(recipe,G.to(device),D.to(device),prior=prior.to(device),seed=SEED,serial_backward=True)
            initial=dict(initial_generator_sha256=model_hash(G),initial_critic_sha256=model_hash(D),initial_prior_sha256=tensor_hash(prior.z))
            previous=inputs['expected_initial_hashes'][args.problem]
            require(initial==previous, f'initial G/D/prior differ from previous GPU E22: {initial}')
            initial_streams=validate_initial_streams(torch,trainer)
            receipt=dict(problem=args.problem,variant=args.variant,seed=SEED,steps=STEPS,recipe=recipe.to_dict(),**initial,
                         package=selected,serial_backward=True,prior_learnable=prior.z.requires_grad,
                         generator_parameters=sum(p.numel() for p in G.parameters()),critic_parameters=sum(p.numel() for p in D.parameters()),
                         previous_gpu_initialization_verified=True,device=DEVICE,
                         fixture_sources=fixture_sources,initial_streams=initial_streams,
                         primary_sampling={'output_noise':True,'scorer':'unchanged original'},
                         **execution_receipt(inputs,runtime))
            write_json(outdir/'config.json',receipt)
            log('training_start',problem=args.problem,variant=args.variant,initial_hashes=initial,runtime=runtime,
                package_sha256=selected['package_sha256'],config_sha256=selected['config_sha256'],steps=STEPS)
            phase='read_only_data_and_evaluator'
            if args.problem=='toy':
                data=torch.load(PREV/'data/toy-stream.pt',map_location='cpu',weights_only=True)['points'].to(device)
                require(len(data)>=2*STEPS*128,'toy stream shorter than the fixed budget')
                def batch(step,role):return data[(2*step+role)*128:(2*step+role+1)*128]
                evaluate=lambda:toy_metrics(trainer,SEED+100)
            else:
                evaluator=ImageEvaluation(device)
                idx=torch.load(PREV/'data/image-stream.pt',map_location='cpu',weights_only=True)['indices'].to(device)
                require(len(idx)>=2*STEPS*128,'MNIST index stream shorter than the fixed budget')
                def batch(step,role):
                    pos=(2*step+role)*128
                    return evaluator.train[idx[pos:pos+128]].float().unsqueeze(1)/127.5-1
                info=dict(accuracy=evaluator.accuracy,active_dimensions=int(evaluator.active.sum()),real_vs_real=evaluator.control,
                          normalization_rule='real train first 5000; std > 1e-6 * maximum training std',
                          active_mask_sha256=tensor_hash(evaluator.active),active_mean_sha256=tensor_hash(evaluator.mean),
                          active_std_sha256=tensor_hash(evaluator.std),raw_reference_sha256=tensor_hash(evaluator.ref),
                          active_reference_sha256=tensor_hash(evaluator.active_ref),evaluator_model_sha256=sha(PREV/'evaluator.pt'))
                write_json(outdir/'evaluator.json',info)
                log('evaluator_ready',problem=args.problem,variant=args.variant,**info)
                evaluate=lambda:evaluator.evaluate(trainer)
            wall=time.perf_counter()
            def checkpoint():
                nonlocal phase
                phase=f'evaluate_checkpoint_{trainer.completed_steps}'
                torch.cuda.synchronize(0); start=time.perf_counter()
                metrics=evaluate()
                torch.cuda.synchronize(0)
                record=dict(step=trainer.completed_steps,training_seconds=elapsed,wall_seconds=time.perf_counter()-wall,
                            evaluation_seconds=time.perf_counter()-start,metrics=metrics,diagnostics=diagnostics(trainer),
                            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(0),
                            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(0))
                curves.append(record)
                with (outdir/'metrics.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=True)+'\n')
                state=trainer.state_dict()
                record['backend_selection']=validate_checkpoint_state(state,trainer.policy.roles,args.problem,recipe)
                require(record['backend_selection']==record['diagnostics']['backend_selection'],'record/checkpoint selection disagrees')
                record['rng_buffer_placement']=rng_cpu_buffers(torch,state)
                phase=f'save_checkpoint_{trainer.completed_steps}'
                torch.save(dict(trainer=state,data_position=2*trainer.completed_steps*128,record=record,
                                receipt_sha256=sha(outdir/'config.json')),
                           outdir/f'checkpoint-{trainer.completed_steps:04d}.pt')
                if args.problem=='mnist' and trainer.completed_steps in (0,1000,2000):
                    images=draw_samples(trainer,100,SEED+101)
                    save_image(images.clamp(-1,1).add(1).div(2),outdir/f'samples-{trainer.completed_steps:04d}.png',nrow=10)
                log('training_checkpoint',problem=args.problem,variant=args.variant,**record)
            checkpoint();last_progress=time.perf_counter()
            for step in range(STEPS):
                phase=f'update_{step+1}'
                real_d,real_g=batch(step,0),batch(step,1)
                torch.cuda.synchronize(0);start=time.perf_counter()
                result=trainer.step(real_d,generator_real=real_g,collect_stats=(step+1)%100==0)
                torch.cuda.synchronize(0);elapsed+=time.perf_counter()-start
                if (step+1)%100==0 or time.perf_counter()-last_progress>=30:
                    log('training_progress',problem=args.problem,variant=args.variant,step=step+1,training_seconds=elapsed,
                        updates_per_second=(step+1)/elapsed,loss_d=float(result['loss_d']),loss_g=float(result['loss_g']),
                        penalty=float(result['penalty']),gpu_allocated_bytes=torch.cuda.memory_allocated(0))
                    last_progress=time.perf_counter()
                if step+1 in CHECKPOINTS:checkpoint()
            phase='final_receipt'
            torch.cuda.synchronize(0)
            result=dict(status='COMPLETE',problem=args.problem,variant=args.variant,steps=STEPS,training_seconds=elapsed,
                        whole_run_seconds=time.perf_counter()-wall,process_seconds=time.perf_counter()-process_start,
                        updates_per_second=STEPS/elapsed,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(0),
                        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(0),
                        peak_cpu_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                        device=DEVICE,final=curves[-1],receipt=receipt,
                        checkpoint_sha256={f'checkpoint-{step:04d}.pt':sha(outdir/f'checkpoint-{step:04d}.pt') for step in CHECKPOINTS})
            result['original_quality_gate']=('PASS' if toy_gate(curves[-1]['metrics']) else 'FAIL') if args.problem=='toy' else None
            write_json(outdir/'result.json',result)
            log('training_complete',problem=args.problem,variant=args.variant,steps=STEPS,training_seconds=elapsed,
                updates_per_second=result['updates_per_second'],peak_gpu_allocated_bytes=result['peak_gpu_allocated_bytes'],
                peak_gpu_reserved_bytes=result['peak_gpu_reserved_bytes'],final=curves[-1])
    except Exception as error:
        failure=dict(status='ERROR',problem=args.problem,variant=args.variant,phase=phase,
                     error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
                     completed_steps=None if trainer is None else trainer.completed_steps,
                     training_seconds=elapsed,process_seconds=time.perf_counter()-process_start,
                     device=DEVICE,command=[sys.executable,*sys.argv],time=common.utc_now())
        write_json(outdir/'error.json',failure);log('training_error',**failure)
        raise


if __name__=='__main__':main()
