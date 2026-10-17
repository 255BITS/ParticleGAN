"""Two independent ten-update CUDA continuations of each saved 1000-step state."""
import common
from common import (ROOT, PREV, SEED, DEVICE, PROBLEMS, VARIANTS, require, sha,
                    log, write_json, verify_inputs, configure_cuda, select_package,
                    exclusive_learned_gpu, execution_receipt, rng_cpu_buffers)
import argparse
from copy import deepcopy
import gc
import hashlib
import json
import struct
import sys
import time
import traceback

sys.path.insert(0, str(PREV))
import torch
from torchvision.datasets import MNIST
from models_metrics import networks
from contracts import validate_checkpoint_state
from adapter import draw_samples

START = 1000
COUNT = 10
EXCLUDED = ['birth_death.last.eval_seconds']
LOSS_KEYS = ('loss_d', 'loss_g', 'loss_gan', 'prior_regularization', 'penalty')


def tensor_placements(value):
    """Read all tensor locations, including CPU RNG/Adam step buffers."""
    rows=[]
    def visit(item,path):
        if isinstance(item,torch.Tensor):
            rows.append(dict(path=path,device=str(item.device),dtype=str(item.dtype),shape=list(item.shape)))
        elif isinstance(item,dict):
            for key,child in item.items():visit(child,path+'.'+str(key))
        elif isinstance(item,(list,tuple)):
            for index,child in enumerate(item):visit(child,path+'.'+str(index))
    visit(value,'trainer')
    return rows


def digest(value):
    """Typed, delimited state fingerprint; tensor shape/dtype/device and bits count."""
    h=hashlib.sha256()
    def token(x):
        b=x if isinstance(x,bytes) else str(x).encode()
        h.update(str(len(b)).encode()+b':'+b)
    def add(x):
        if isinstance(x,torch.Tensor):
            token('tensor');token(tuple(x.shape));token(x.dtype);token(x.device)
            token(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            token('dict');token(len(x))
            for k in sorted(x,key=lambda v:(type(v).__name__,repr(v))):add(k);add(x[k])
        elif isinstance(x,(list,tuple)):
            token(type(x).__name__);token(len(x))
            for v in x:add(v)
        elif isinstance(x,float):token('float64');token(struct.pack('!d',x))
        else:token(type(x).__name__);token(repr(x))
    add(value)
    return h.hexdigest()


def semantic_state(state):
    """Only this birth/death observational duration is removed."""
    state=deepcopy(state)
    state.get('birth_death',{}).get('last',{}).pop('eval_seconds',None)
    return state


def counter_delta(before,after):
    before=before.get('birth_death',{}).get('counters',{})
    after=after.get('birth_death',{}).get('counters',{})
    return {k:after.get(k,0)-before.get(k,0) for k in sorted(set(before)|set(after))}


def replay_problem(problem,variant,inputs,runtime,Recipe,ParticlePrior,GANTrainer):
    path=ROOT/'training'/problem/variant/'checkpoint-1000.pt'
    outdir=ROOT/'replay'/problem/variant
    require(not outdir.exists() or not any(outdir.iterdir()),f'replay directory already contains evidence: {outdir}')
    outdir.mkdir(parents=True,exist_ok=True)
    phase='load_frozen_checkpoint';outputs=[]
    try:
        require(path.exists(),f'required saved checkpoint absent: {path}')
        config_path=path.parent/'config.json'
        run_receipt=json.loads(config_path.read_text())
        require(run_receipt['package']==inputs['variants'][variant],'checkpoint source/config receipt differs from frozen input')
        require(run_receipt['source_freeze_sha256']==sha(ROOT/'SOURCE-FREEZE.json'),'checkpoint lane source freeze differs')
        # No map_location: saved CPU buffers stay on CPU; saved cuda:0 tensors
        # return to the same sole visible physical GPU. A blanket CUDA map
        # would incorrectly move CPU RNG buffers and is deliberately absent.
        checkpoint=torch.load(path,weights_only=False)
        saved=checkpoint['trainer']
        require(saved['device']==DEVICE and saved['completed_steps']==START,'wrong device or checkpoint cursor')
        require(saved['serial_backward'] is True,'saved serial backward execution mode is required')
        require(checkpoint['data_position']==2*START*128,'saved real-stream cursor differs')
        require(checkpoint['receipt_sha256']==sha(config_path),'checkpoint run-receipt hash differs')
        rng_placement=rng_cpu_buffers(torch,saved)
        saved_semantic_sha=digest(semantic_state(saved))
        if problem=='toy':
            data=torch.load(PREV/'data/toy-stream.pt',map_location='cpu',weights_only=True)['points'].to(DEVICE)
            def real(step,role):return data[(2*step+role)*128:(2*step+role+1)*128]
        else:
            data=MNIST(PREV/'data',train=True,download=False).data.to(DEVICE)
            indices=torch.load(PREV/'data/image-stream.pt',map_location='cpu',weights_only=True)['indices'].to(DEVICE)
            def real(step,role):
                pos=(2*step+role)*128
                return data[indices[pos:pos+128]].float().unsqueeze(1)/127.5-1
        for branch in range(2):
            phase=f'construct_branch_{branch}'
            G,D=networks(problem)
            prior=ParticlePrior(1024,128)
            trainer=GANTrainer(Recipe(**saved['recipe']),G.to(DEVICE),D.to(DEVICE),prior=prior.to(DEVICE),
                               seed=SEED,serial_backward=True)
            restore_input=(saved if branch==0 else torch.load(path,map_location='cpu',weights_only=False)['trainer'])
            trainer.load_state_dict(restore_input)
            restored=trainer.state_dict()
            validate_checkpoint_state(restored,trainer.policy.roles,problem,trainer.recipe)
            rng_cpu_buffers(torch,restored)
            restored_sections={k:digest(semantic_state(restored)[k]) for k in restored}
            saved_sections={k:digest(semantic_state(saved)[k]) for k in saved}
            restored_exact={k:restored_sections[k]==saved_sections[k] for k in saved_sections}
            restored_semantic_sha=digest(semantic_state(restored))
            restored_exact['whole_state']=restored_semantic_sha==saved_semantic_sha
            # Record restoration mismatches rather than changing checkpoint fields.
            losses=[];loss_bits=[];update_fingerprints=[];training_seconds=0.
            for step in range(START,START+COUNT):
                phase=f'branch_{branch}_update_{step+1}'
                real_d,real_g=real(step,0),real(step,1)
                torch.cuda.synchronize(0);started=time.perf_counter()
                result=trainer.step(real_d,generator_real=real_g)
                torch.cuda.synchronize(0);training_seconds+=time.perf_counter()-started
                tensors={k:result[k].detach().cpu().clone() for k in LOSS_KEYS}
                loss_bits.append(tensors)
                losses.append({k:float(tensors[k]) for k in LOSS_KEYS})
                state=trainer.state_dict()
                semantic=semantic_state(state)
                update_fingerprints.append(dict(step=step+1,loss_sha256=digest(tensors),
                    whole_state_sha256=digest(state),semantic_state_sha256=digest(semantic),
                    semantic_sections={k:digest(semantic[k]) for k in semantic}))
            require(trainer.completed_steps==START+COUNT,'replay did not complete the exact continuation budget')
            state=trainer.state_dict();semantic=semantic_state(state)
            rng_cpu_buffers(torch,state)
            state_sha=digest(state);semantic_sha=digest(semantic)
            sections={k:digest(state[k]) for k in state}
            semantic_sections={k:digest(semantic[k]) for k in semantic}
            phase=f'primary_sample_branch_{branch}'
            sample_count=8192 if problem=='toy' else 4096
            samples=draw_samples(trainer,sample_count,SEED+100).detach().cpu().clone()
            samples_sha=digest(samples)
            sample_state_unchanged=digest(semantic_state(trainer.state_dict()))==semantic_sha
            phase=f'save_branch_{branch}'
            branch_path=outdir/f'branch-{branch}.pt'
            torch.save(dict(trainer=state,loss_tensors=loss_bits,primary_samples=samples,data_position=2*(START+COUNT)*128,
                            source_checkpoint_sha256=sha(path),update_fingerprints=update_fingerprints),branch_path)
            output=dict(branch=branch,checkpoint_load_map_location='native' if branch==0 else 'cpu',
                        restore_input_tensor_placement=tensor_placements(restore_input),
                        restored_tensor_placement=tensor_placements(restored),
                        primary_samples_sha256=samples_sha,primary_sample_count=sample_count,
                        primary_sample_seed=SEED+100,primary_sampling_output_noise=True,
                        sample_preserves_semantic_training_state=sample_state_unchanged,losses=losses,losses_sha256=digest(loss_bits),sections=sections,
                        semantic_sections=semantic_sections,whole_state_sha256=state_sha,
                        semantic_state_sha256=semantic_sha,restored_semantic_sections_identical=restored_exact,
                        restored_state_semantic_sha256=restored_semantic_sha,
                        birth_death_counter_delta=counter_delta(saved,state),training_seconds=training_seconds,
                        update_fingerprints=update_fingerprints,endpoint=str(branch_path),endpoint_sha256=sha(branch_path))
            outputs.append(output)
            log('replay_branch_complete',problem=problem,variant=variant,branch=branch,
                steps_replayed=COUNT,training_seconds=training_seconds,
                semantic_state_sha256=semantic_sha,losses_sha256=output['losses_sha256'],
                birth_death_counter_delta=output['birth_death_counter_delta'],restored_semantic_identical=all(restored_exact.values()))
            del trainer,G,D,prior,state,semantic,restored,loss_bits
            gc.collect();torch.cuda.empty_cache()
        same_sections={k:outputs[0]['sections'][k]==outputs[1]['sections'][k] for k in outputs[0]['sections']}
        semantic_same={k:outputs[0]['semantic_sections'][k]==outputs[1]['semantic_sections'][k] for k in outputs[0]['semantic_sections']}
        step_matches=[dict(step=a['step'],losses_bit_identical=a['loss_sha256']==b['loss_sha256'],
                           semantic_state_bit_identical=a['semantic_state_sha256']==b['semantic_state_sha256'],
                           semantic_sections_bit_identical={k:a['semantic_sections'][k]==b['semantic_sections'][k]
                                                            for k in a['semantic_sections']})
                      for a,b in zip(outputs[0]['update_fingerprints'],outputs[1]['update_fingerprints'])]
        losses_same=outputs[0]['losses_sha256']==outputs[1]['losses_sha256']
        restoration_exact=all(all(o['restored_semantic_sections_identical'].values()) for o in outputs)
        samples_same=outputs[0]['primary_samples_sha256']==outputs[1]['primary_samples_sha256']
        samples_preserve_state=all(o['sample_preserves_semantic_training_state'] for o in outputs)
        required=(samples_same and samples_preserve_state and all(semantic_same.values()) and losses_same and restoration_exact
                  and all(s['losses_bit_identical'] and s['semantic_state_bit_identical'] for s in step_matches))
        result=dict(status='PASS' if required else 'FAIL',problem=problem,variant=variant,
                    checkpoint=str(path),checkpoint_sha256=sha(path),run_config_sha256=sha(config_path),
                    start_step=START,steps_replayed=COUNT,device=DEVICE,loss_keys=list(LOSS_KEYS),
                    primary_sample_bytes_bit_identical=samples_same,primary_sampling_preserves_training_state=samples_preserve_state,
                    losses_bit_identical=losses_same,sections_bit_identical=same_sections,
                    semantic_sections_bit_identical=semantic_same,restoration_semantic_bit_identical=restoration_exact,
                    excluded_observational_fields=EXCLUDED,
                    semantic_state_bit_identical=outputs[0]['semantic_state_sha256']==outputs[1]['semantic_state_sha256'],
                    whole_state_bit_identical=outputs[0]['whole_state_sha256']==outputs[1]['whole_state_sha256'],
                    rng_buffer_placement=rng_placement,per_update_comparison=step_matches,branches=outputs,
                    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(0),
                    peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(0),
                    receipt=execution_receipt(inputs,runtime))
        write_json(outdir/'result.json',result)
        log('checkpoint_replay',problem=problem,variant=variant,status=result['status'],steps_replayed=COUNT,
            losses_bit_identical=losses_same,semantic_sections_bit_identical=semantic_same,
            restoration_semantic_bit_identical=restoration_exact,excluded_observational_fields=EXCLUDED)
        return result
    except Exception as error:
        failure=dict(status='ERROR',problem=problem,variant=variant,phase=phase,error_type=type(error).__name__,
                     error=str(error),traceback=traceback.format_exc(),device=DEVICE,checkpoint=str(path),
                     completed_branches=len(outputs),branches=outputs,excluded_observational_fields=EXCLUDED,
                     command=[sys.executable,*sys.argv],time=common.utc_now())
        write_json(outdir/'error.json',failure);log('replay_error',**failure)
        return failure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant',choices=VARIANTS)
    args=parser.parse_args();summary_path=ROOT/f'replay-{args.variant}.json'
    require(not summary_path.exists(),f'replay aggregate already exists: {summary_path}')
    receipts={}
    try:
        with exclusive_learned_gpu():
            inputs=verify_inputs();select_package(inputs,args.variant)
            from particlegan.recipes import Recipe
            from particlegan.particle_prior import ParticlePrior
            from particlegan.training import GANTrainer
            runtime=configure_cuda(torch)
            for problem in PROBLEMS:
                receipts[problem]=replay_problem(problem,args.variant,inputs,runtime,Recipe,ParticlePrior,GANTrainer)
                write_json(summary_path,receipts)
    except Exception as error:
        for problem in PROBLEMS:
            if problem not in receipts:
                receipts[problem]=dict(status='ERROR',problem=problem,variant=args.variant,error_type=type(error).__name__,
                    error=str(error),traceback=traceback.format_exc(),device=DEVICE,phase='setup',command=[sys.executable,*sys.argv])
        write_json(summary_path,receipts);log('replay_setup_error',variant=args.variant,error=str(error))
    write_json(summary_path,receipts)
    return 0 if len(receipts)==2 and all(r['status']=='PASS' for r in receipts.values()) else 1


if __name__=='__main__':raise SystemExit(main())
