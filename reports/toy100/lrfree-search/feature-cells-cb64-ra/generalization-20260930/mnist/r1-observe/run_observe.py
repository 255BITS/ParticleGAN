"""Observation-only exact replay: Toy 750->835 and MNIST 100->215."""
from pathlib import Path
import sys
BASE = Path('/ml2/hypergan/gan-attempts/feature-cells-generalization-20260930/mnist/ra12-auto')
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import common
from common import require, sha, write_json, log, PREV, DEVICE, SEED
import gc
import hashlib
import json
import math
import time
import traceback
import torch
from torchvision.datasets import MNIST
sys.path.insert(0, str(PREV))
from models_metrics import networks
from replay import digest, semantic_state
from contracts import validate_checkpoint_state

WINDOWS = {'toy': (750, 835), 'mnist': (100, 215)}


def plain(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        flat = value.reshape(-1)
        return {'shape': list(value.shape), 'numel': value.numel(), 'first_64': flat[:64].tolist()}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 64:
            return {'length': len(value), 'first_64': [plain(v) for v in value[:64]]}
        return [plain(v) for v in value]
    return value


def rng_digest(trainer):
    values = {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(0)}
    for name in trainer._STREAMS:
        values[name] = getattr(trainer, name).get_state()
    if trainer.policy.birth_death is not None:
        values['birth'] = trainer.policy.birth_death.stream.get_state()
    return digest(values)


def context(trainer):
    policy = trainer.policy
    record = trainer.opt_d.record
    sigma = policy.log_output_sigma
    scales = [t.s for i, row in enumerate(policy.lr_settle.testers) if i != 1 for t in row]
    settle = policy.controller.mobility
    if scales and all(s is not None for s in scales):
        settle = 1.0 if any(s > 1. / 64. for s in scales) else settle
    from particlegan.policy import output_noise_std
    floor = output_noise_std(trainer.recipe, trainer.completed_steps) * settle
    testers = {}
    for i, row in enumerate(policy.lr_settle.testers):
        for j, tester in enumerate(row):
            if tester is not None:
                testers[f'{i}.{j}'] = plain({key: getattr(tester, key) for key in
                    ('s', 'b', 'tau', 'windows', 'last', 'counts', 'log', 'blocks_in_window')})
    birth = policy.birth_death
    return dict(completed_steps=trainer.completed_steps,
        ka2=record.state_dict(), ka2_pure_a_next_call=record.calls + 1 < 800,
        clipped_tensors=trainer.opt_d.guard.clipped_tensors,
        controller=plain({key: getattr(policy.controller, key) for key in
                         ('mobility', 'data_drive')}),
        roles=policy.roles, lrs=[[group['lr'] for group in opt.param_groups] for opt in policy.optimizers],
        noise=dict(log_sigma=float(sigma.detach()), unconstrained_sigma=float(sigma.detach().exp()),
                   last_output_sigma=policy.last_output_sigma, floor=floor, settle=settle,
                   grad=None if sigma.grad is None else float(sigma.grad.detach()),
                   adam_state=plain(policy.opt_g.state.get(sigma, {}))),
        testers=testers,
        birth=dict(counters=dict(birth.counters), last=plain(birth.last),
                   moved_rows=plain(birth.moved_rows)))


def observe_problem(problem, inputs, runtime, Recipe, ParticlePrior, GANTrainer, OptimizerSurprise):
    start, end = WINDOWS[problem]
    path = BASE/'training'/problem/'RA12-auto'/f'checkpoint-{start:04d}.pt'
    outdir = ROOT/problem
    require(not outdir.exists(), f'observation window already exists: {outdir}')
    outdir.mkdir()
    checkpoint = torch.load(path, weights_only=False)
    saved = checkpoint['trainer']
    require(saved['completed_steps'] == start and checkpoint['data_position'] == 2*start*128,
            'original checkpoint/data cursor differs')
    require(saved['serial_backward'] is True, 'serial backward required')
    if problem == 'toy':
        data = torch.load(PREV/'data/toy-stream.pt', map_location='cpu', weights_only=True)['points'].to(DEVICE)
        def real(step, role):
            return data[(2*step+role)*128:(2*step+role+1)*128]
    else:
        data = MNIST(PREV/'data', train=True, download=False).data.to(DEVICE)
        indices = torch.load(PREV/'data/image-stream.pt', map_location='cpu', weights_only=True)['indices'].to(DEVICE)
        def real(step, role):
            pos = (2*step+role)*128
            return data[indices[pos:pos+128]].float().unsqueeze(1)/127.5-1
    G, D = networks(problem)
    prior = ParticlePrior(1024, 128)
    trainer = GANTrainer(Recipe(**saved['recipe']), G.to(DEVICE), D.to(DEVICE), prior=prior.to(DEVICE),
                         seed=SEED, serial_backward=True)
    trainer.load_state_dict(saved)
    restored = trainer.state_dict()
    validate_checkpoint_state(restored, trainer.policy.roles, problem, trainer.recipe)
    require(digest(semantic_state(restored)) == digest(semantic_state(saved)), 'noncanonical restoration')
    require(set(trainer.policy.surprise.__dict__) == set(saved['surprise']), 'detector instance schema differs')
    initial_rng = rng_digest(trainer)
    initial_context = context(trainer)
    require(rng_digest(trainer) == initial_rng, 'context observation consumed RNG')
    original_decide = OptimizerSurprise.decide
    original_instance_keys = set(trainer.policy.surprise.__dict__)
    rows = []
    active_row = None
    def observed_decide(self, step=None):
        nonlocal active_row
        require(self is trainer.policy.surprise and active_row is not None, 'unexpected detector owner/call')
        before_rng = rng_digest(trainer)
        pending = {key: float(value.detach().cpu()) for key, value in self.pending.items()}
        active_row['before_decide'] = dict(
            step_arg=step, pending_q=pending,
            kept_keys=[key for key in sorted(pending) if math.log(max(pending[key], 1e-30)) > math.log(1e-30)+1.],
            fast=dict(self.fast), slow=dict(self.slow), armed=self.armed, streak=self.streak,
            since_calm=self.since_calm, since_fire=self.since_fire,
            last_ratio=self.last_ratio, last_ratios=dict(self.last_ratios), context=context(trainer))
        require(rng_digest(trainer) == before_rng, 'pre-decision observation consumed RNG')
        fire = original_decide(self, step)
        active_row['after_decide'] = dict(fire=fire, fast=dict(self.fast), slow=dict(self.slow),
            last_ratio=self.last_ratio, last_ratios=dict(self.last_ratios), armed=self.armed,
            streak=self.streak, since_calm=self.since_calm, since_fire=self.since_fire,
            fires=self.fires, log=list(self.log), anchor_event=self.anchor_event, anchor_events=self.anchor_events)
        require(rng_digest(trainer) == before_rng, 'detector wrapper consumed RNG')
        if fire:
            log('observed_r1_fire', problem=problem, next_update=step+1,
                before=active_row['before_decide'], after=active_row['after_decide'])
        return fire
    OptimizerSurprise.decide = observed_decide
    started = time.perf_counter()
    try:
        with (outdir/'trace.jsonl').open('x') as output:
            for step in range(start, end):
                active_row = dict(update=step+1)
                result = trainer.step(real(step, 0), generator_real=real(step, 1), collect_stats=(step+1)%100==0)
                active_row['losses'] = {key: float(result[key].detach()) for key in
                    ('loss_d', 'loss_g', 'loss_gan', 'prior_regularization', 'penalty')}
                before_rng = rng_digest(trainer)
                active_row['after_update'] = context(trainer)
                active_row['pending_after_update'] = {k: float(v.detach().cpu()) for k, v in trainer.policy.surprise.pending.items()}
                require(rng_digest(trainer) == before_rng, 'post-update observation consumed RNG')
                output.write(json.dumps(active_row, allow_nan=True)+'\n')
                output.flush()
                rows.append(active_row)
        torch.cuda.synchronize(0)
    finally:
        OptimizerSurprise.decide = original_decide
    require(trainer.completed_steps == end and len(rows) == end-start, 'replay budget differs')
    require(set(trainer.policy.surprise.__dict__) == original_instance_keys, 'observation polluted detector serialization')
    state = trainer.state_dict()
    common.rng_cpu_buffers(torch, state)
    endpoint = outdir/'endpoint.pt'
    torch.save(dict(trainer=state, data_position=2*end*128, source_checkpoint_sha256=sha(path)), endpoint)
    result = dict(status='COMPLETE_OBSERVATION_ONLY', problem=problem, start=start, end=end,
        updates=end-start, scoring_calls=0, primary_sample_calls=0, source_changes=0,
        class_wrapper_restored=OptimizerSurprise.decide is original_decide,
        instrumentation_preserves_rng_each_capture=True, detector_instance_schema_unchanged=True,
        checkpoint=str(path), checkpoint_sha256=sha(path),
        initial_semantic_sha256=digest(semantic_state(saved)), initial_context=initial_context,
        endpoint=str(endpoint), endpoint_sha256=sha(endpoint), endpoint_semantic_sha256=digest(semantic_state(state)),
        trace=str(outdir/'trace.jsonl'), trace_sha256=sha(outdir/'trace.jsonl'),
        fires=[dict(update=r['update'], step_arg=r['before_decide']['step_arg'],
                    pending_q=r['before_decide']['pending_q'], kept_keys=r['before_decide']['kept_keys'],
                    ratios=r['after_decide']['last_ratios'], ratio=r['after_decide']['last_ratio'],
                    ka2=r['before_decide']['context']['ka2'], noise=r['before_decide']['context']['noise'])
               for r in rows if r['after_decide']['fire']],
        wall_seconds=time.perf_counter()-started, execution=common.execution_receipt(inputs, runtime))
    write_json(outdir/'result.json', result)
    log('observation_complete', problem=problem, updates=end-start, fires=result['fires'], wall_seconds=result['wall_seconds'])
    del trainer, G, D, prior, state, restored, saved, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    require(not (ROOT/'result.json').exists(), 'closed result already exists')
    inputs = common.verify_inputs()
    manifest = json.loads((ROOT/'INPUTS.json').read_text())
    for path, expected in manifest['read_only_sha256'].items():
        require(sha(path) == expected, f'observation input changed: {path}')
    for name, expected in json.loads((ROOT/'SOURCE-FREEZE.json').read_text())['source_sha256'].items():
        require(sha(ROOT/name) == expected, f'observation source changed: {name}')
    common.select_package(inputs, 'RA12-auto')
    from particlegan.recipes import Recipe
    from particlegan.particle_prior import ParticlePrior
    from particlegan.training import GANTrainer
    from particlegan.continuous import OptimizerSurprise
    runtime = common.configure_cuda(torch)
    results = {}
    try:
        with common.exclusive_learned_gpu():
            for problem in WINDOWS:
                results[problem] = observe_problem(problem, inputs, runtime, Recipe, ParticlePrior, GANTrainer, OptimizerSurprise)
    except Exception as error:
        write_json(ROOT/'error.json', dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
        raise
    common.verify_inputs()
    write_json(ROOT/'result.json', dict(status='COMPLETE_OBSERVATION_ONLY', total_updates=200, problems=results))


if __name__ == '__main__':
    main()
