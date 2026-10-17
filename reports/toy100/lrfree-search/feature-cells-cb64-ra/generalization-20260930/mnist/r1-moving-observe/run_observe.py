"""Observation-only original moving-quarter Grid100 500->535 and1000->1035."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
FROOT = ROOT.parents[1]
BASE = ROOT.parent/'ra12-auto'
ORIGINAL = FROOT/'moving-quarter/grid100-r1-on'
PACKAGE = FROOT/'pkg-RA11-R1'
HOSTS = Path('/ml2/hypergan/lrfree-20260926/harness/hosts')
sys.path.insert(0, str(BASE))
import common
from common import require, sha, write_json, log
sys.path.insert(0, str(PACKAGE))
sys.path.insert(1, str(HOSTS))
import gc
import json
import math
import time
import traceback
from types import SimpleNamespace
import torch
from torch import nn
from particlegan import GANTrainer
from particlegan.recipes import Recipe
from particlegan.particle_prior import ParticlePrior
from particlegan.continuous import OptimizerSurprise
from native100 import problems, toy_models
from capture import plain, digest, semantic_state, rng_digest, context
import frozen_data

DEVICE = 'cuda:0'
SEED = 1234
BATCH = 2048
WINDOWS = ((500,535),(1000,1035))


def all_rng_digest(trainer, stream):
    return digest({'training': rng_digest(trainer), 'real_stream': stream.get_state()})


def make_trainer(recipe):
    # Exactly the frozen original host constructor. Model/optimizer/RNG state
    # is subsequently loaded from the native original saved checkpoint.
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        with torch.device(DEVICE):
            prior = ParticlePrior(20000,2)
            with torch.no_grad():
                prior.z.uniform_(-5.0,5.0)
            G = nn.Linear(2,2)
            with torch.no_grad():
                G.weight.copy_(torch.eye(2))
                G.bias.zero_()
            D = toy_models.SimpleMLPDiscriminator(in_dim=2, hidden_dim=128, n_hidden=3, fourier=3)
            for layer in D.modules():
                if isinstance(layer,nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        trainer = GANTrainer(recipe,G,D,prior=prior,seed=SEED,serial_backward=True,
                             optimizer_options={'foreach':False,'fused':False})
    torch.manual_seed(SEED)
    return trainer


def observe_window(start,end,runtime):
    folder = ROOT/f'window-{start}'
    require(not folder.exists(), 'positive observation already exists')
    folder.mkdir()
    checkpoint = ORIGINAL/f'checkpoint-{start:06d}.pt'
    saved = torch.load(checkpoint,weights_only=False)
    require(saved['completed_steps']==start and saved['serial_backward'] is True,'checkpoint cursor/execution differs')
    recipe = Recipe(**saved['recipe'])
    trainer = make_trainer(recipe)
    stream = torch.Generator(device=DEVICE).manual_seed(SEED)
    global_before = digest({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(0)})
    for _ in range(2*start):
        problems.sample_real('grid100',BATCH,device=DEVICE,generator=stream)
    require(digest({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(0)})==global_before,
            'original real-stream reconstruction consumed global RNG')
    real_stream_at_start = stream.get_state()
    trainer.load_state_dict(saved)
    require(torch.equal(stream.get_state(),real_stream_at_start),'checkpoint restore changed external stream')
    restored = trainer.state_dict()
    require(digest(semantic_state(restored))==digest(semantic_state(saved)), 'noncanonical original checkpoint restoration')
    common.rng_cpu_buffers(torch,restored)
    require(trainer.opt_g.param_groups[1]['params'][0] is trainer.prior.z,'table optimizer owner differs')
    require(trainer.opt_g.param_groups[2]['params'][0] is trainer.log_output_sigma,'noise optimizer owner differs')
    torch.save(dict(real_stream=real_stream_at_start,seed=SEED,original_sample_calls=2*start,
                    batch_size=BATCH,checkpoint_sha256=sha(checkpoint)),folder/'real-stream-start.pt')
    frozen_data.args = SimpleNamespace(task='grid100',rotate_every=500,rotate_deg=30.)
    frozen_data.device, frozen_data.batch, frozen_data.stream = DEVICE, BATCH, stream
    frozen_data.problems = problems
    real_batch = frozen_data.real_batch
    original_decide = OptimizerSurprise.decide
    instance_keys = set(trainer.surprise.__dict__)
    rows = []
    active_row = None
    def observed_decide(self,step=None):
        nonlocal active_row
        require(self is trainer.surprise and active_row is not None,'unexpected detector owner/call')
        before_rng = all_rng_digest(trainer,stream)
        pending = {key:float(value.detach().cpu()) for key,value in self.pending.items()}
        active_row['before_decide'] = dict(step_arg=step,pending_q=pending,
            kept_keys=[key for key in sorted(pending) if math.log(max(pending[key],1e-30))>math.log(1e-30)+1.],
            fast=dict(self.fast),slow=dict(self.slow),armed=self.armed,streak=self.streak,
            since_calm=self.since_calm,since_fire=self.since_fire,
            last_ratio=self.last_ratio,last_ratios=dict(self.last_ratios),context=context(trainer),
            real_stream_sha256=digest(stream.get_state()))
        require(all_rng_digest(trainer,stream)==before_rng,'pre-decision capture consumed RNG')
        fire = original_decide(self,step)
        active_row['after_decide'] = dict(fire=fire,fast=dict(self.fast),slow=dict(self.slow),
            last_ratio=self.last_ratio,last_ratios=dict(self.last_ratios),armed=self.armed,streak=self.streak,
            since_calm=self.since_calm,since_fire=self.since_fire,fires=self.fires,log=list(self.log),
            anchor_event=self.anchor_event,anchor_events=self.anchor_events)
        require(all_rng_digest(trainer,stream)==before_rng,'detector capture consumed RNG')
        if fire:
            log('observed_moving_fire',start=start,next_update=step+1,before=active_row['before_decide'],after=active_row['after_decide'])
        return fire
    OptimizerSurprise.decide = observed_decide
    started = time.perf_counter()
    try:
        with (folder/'trace.jsonl').open('x') as output:
            for step in range(start+1,end+1):
                active_row = dict(update=step,target_degrees=math.degrees(frozen_data.angle(step)))
                real_d = real_batch(step)
                cache = []
                def generator_real():
                    if not cache:
                        cache.append(real_batch(step))
                    return cache[0]
                with torch.autograd.set_multithreading_enabled(False):
                    result = trainer.step(real_d,generator_real=generator_real)
                if not cache:
                    generator_real()
                active_row['losses'] = {key:float(result[key].detach()) for key in
                    ('loss_d','loss_g','loss_gan','prior_regularization','penalty')}
                before_rng = all_rng_digest(trainer,stream)
                active_row['after_update'] = context(trainer)
                active_row['pending_after_update'] = {k:float(v.detach().cpu()) for k,v in trainer.surprise.pending.items()}
                active_row['real_stream_sha256_after_update'] = digest(stream.get_state())
                require(all_rng_digest(trainer,stream)==before_rng,'post-update observation consumed RNG')
                output.write(json.dumps(active_row,allow_nan=True)+'\n')
                output.flush()
                rows.append(active_row)
        torch.cuda.synchronize(0)
    finally:
        OptimizerSurprise.decide = original_decide
    require(trainer.completed_steps==end and len(rows)==end-start,'replay update budget differs')
    require(set(trainer.surprise.__dict__)==instance_keys,'observation polluted detector serialization')
    state = trainer.state_dict()
    endpoint = folder/'endpoint.pt'
    torch.save(dict(trainer=state,real_stream=stream.get_state(),original_sample_calls=2*end,
                    checkpoint_sha256=sha(checkpoint)),endpoint)
    events=[]
    for r in rows:
        if r['after_decide']['fire']:
            c=r['before_decide']['context']
            events.append(dict(update=r['update'],step_arg=r['before_decide']['step_arg'],
                pending_q=r['before_decide']['pending_q'],kept_keys=r['before_decide']['kept_keys'],
                ratios=r['after_decide']['last_ratios'],ratio=r['after_decide']['last_ratio'],
                tester_scales={k:t['s'] for k,t in c['testers'].items()},ka2=c['ka2'],noise=c['noise']))
    result = dict(status='COMPLETE_ORIGINAL_MOVING_OBSERVATION_ONLY',start=start,end=end,updates=end-start,
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),canonical_semantic_restoration=True,
        initial_semantic_sha256=digest(semantic_state(saved)),endpoint_semantic_sha256=digest(semantic_state(state)),
        real_stream_restore=dict(seed=SEED,device=DEVICE,original_sample_calls=2*start,batch_size=BATCH,
            mechanism='one supplied private CUDA generator; native sample_real twice per original update; rotations consume no RNG',
            checkpoint_loading_preserves_external_stream=True,reconstruction_preserves_global_rng=True,
            start_stream_sha256=digest(real_stream_at_start),end_stream_sha256=digest(stream.get_state())),
        fires=events,scoring_calls=0,sample_calls=0,source_mutations=0,
        instrumentation_preserves_all_rng_each_capture=True,detector_instance_schema_unchanged=True,
        class_wrapper_restored=OptimizerSurprise.decide is original_decide,
        trace_sha256=sha(folder/'trace.jsonl'),endpoint_sha256=sha(endpoint),wall_seconds=time.perf_counter()-started,runtime=runtime)
    write_json(folder/'result.json',result)
    log('moving_observation_complete',start=start,end=end,updates=end-start,fires=events,wall_seconds=result['wall_seconds'])
    del trainer,saved,state,restored
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    require(not (ROOT/'result.json').exists(),'observation attempt already completed')
    for path,expected in json.loads((ROOT/'INPUTS.json').read_text())['read_only_sha256'].items():
        require(sha(path)==expected,f'frozen moving input changed: {path}')
    for name,expected in json.loads((ROOT/'SOURCE-FREEZE.json').read_text())['source_sha256'].items():
        require(sha(ROOT/name)==expected,f'moving observation source changed: {name}')
    runtime=common.configure_cuda(torch)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(0)
    runtime.update(cpu_threads=torch.get_num_threads(),float32_matmul_precision=torch.get_float32_matmul_precision(),original_seed=SEED)
    results={}
    try:
        with common.exclusive_learned_gpu():
            for start,end in WINDOWS:
                results[str(start)]=observe_window(start,end,runtime)
    except Exception as error:
        write_json(ROOT/'error.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc()))
        raise
    for path,expected in json.loads((ROOT/'INPUTS.json').read_text())['read_only_sha256'].items():
        require(sha(path)==expected,f'frozen moving input changed after replay: {path}')
    write_json(ROOT/'result.json',dict(status='COMPLETE_ORIGINAL_MOVING_OBSERVATION_ONLY',total_updates=70,windows=results))


if __name__=='__main__':
    main()
