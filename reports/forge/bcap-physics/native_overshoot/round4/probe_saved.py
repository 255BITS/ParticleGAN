"""Actual finite, reversible proposal probes using the ORIGINAL trained API.

One discarded next-step reconstruction, not a continuation or qualification.
All scientific imports use the verified original source worktree. Fractions
share the actual G/prior proposal, post-D critic, training rows and kernel draws.
Evaluation geometry is an oracle diagnostic and never enters training.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

RECEIPTS = Path('/home/martyn/dev/ParticleGAN-bcap-tier2-search')
ORIGINAL = Path('/mnt/ml7tb/ParticleGAN-forge/bcap-tier2-search-v1/snapshots/2e1d0e2704f3e8cff0845f46fe66e8fb641c32fd32b7d1929f05a680b4c3bbed')
sys.path.insert(0, str(ORIGINAL))
import torch
from experiments.forge.api import task_formulation_context
from experiments.forge.adapters import _models
from experiments.forge.contracts import file_hash, atomic_json, read_json
from experiments.forge.state import state_digest
from benchmarks.toy100.problems import sample_real, evaluation_geometry
from benchmarks.toy100.metrics import evaluate_samples
from benchmarks.toy100.accuracy import evaluate_accuracy


def main(output):
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    aid = 'db79021404c14c91aa5f38f9f1be40fb'
    base = RECEIPTS / 'reports/forge/attempts' / aid
    envelope, result, cert = [read_json(base / f'{n}.json') for n in ('request','result','evidence')]
    receipt = next(r for r in read_json(RECEIPTS/'reports/forge/bcap-tier2-search/receipts.json') if r['attempt_id']==aid)
    for name, descriptor in receipt['provenance']['original_files'].items():
        assert file_hash(base/f'{name}.json') == descriptor['sha256']
    # Audit every Python scientific source used from the original tree.
    source_proofs = {}
    for name, expected in cert['source']['files'].items():
        if name.endswith('.py'):
            assert file_hash(ORIGINAL/name) == expected, name
            source_proofs[name] = expected
    request = envelope['request']
    task = request['tasks']['grid100']
    row = result['task_results'][0]
    descriptor = row['evidence']['checkpoint']
    checkpoint = Path(row['evidence']['artifact_root'])/descriptor['path']
    assert file_hash(checkpoint)==descriptor['sha256']
    saved = torch.load(checkpoint, map_location='cuda:0', weights_only=True)
    assert state_digest(saved)==descriptor['state_sha256']
    caller_cpu = torch.get_rng_state().clone()
    caller_cuda = torch.cuda.get_rng_state(0).clone()
    context = task_formulation_context(request['candidate'],task,request['protocol'],device='cuda:0',root=ORIGINAL)
    g,d = _models(context,task['execution']['model'])
    trainer = context.build_trainer(g,d,max_steps=task['execution']['steps'])
    context.load_state_dict(saved)
    initial_digest = state_digest(context.state_dict())
    assert initial_digest==descriptor['state_sha256']
    captured = {}
    original_sample = trainer._sample_training_prior
    calls = 0
    def sample(n):
        nonlocal calls
        latent, ids = original_sample(n)
        calls += 1
        if calls==2:
            captured['ids']=ids.detach().clone()
            captured['offset']=latent.detach()-trainer.prior.z[ids].detach()
        return latent,ids
    trainer._sample_training_prior = sample
    original_step = trainer.opt_g.step
    def proposal(*args, **kwargs):
        parameters = [p for group in trainer.opt_g.param_groups for p in group['params']]
        old = [p.detach().clone() for p in parameters]
        gradients = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in parameters]
        opt = deepcopy(trainer.opt_g.state_dict())
        buffers = [(b,b.detach().clone()) for m in (g,d,trainer.prior) for b in m.buffers()]
        rng = context.streams.state_dict()
        cpu, cuda = torch.get_rng_state().clone(),torch.cuda.get_rng_state(0).clone()
        def objective():
            z = trainer.prior.z[captured['ids']] + captured['offset']
            return trainer.loss.g_loss(d(g(z)))
        with torch.no_grad():
            before = float(objective())
        original_step(*args, **kwargs)
        new = [p.detach().clone() for p in parameters]
        delta = [b-a for a,b in zip(old,new)]
        slope = sum(float((a.double()*b.double()).sum()) for a,b in zip(gradients,delta))
        evaluation_stream = context.streams.generator('eval',component='live',purpose='samples')
        evaluation_rng = evaluation_stream.get_state().clone()
        centers,sigma=evaluation_geometry('grid100',device='cuda:0')
        curves=[]
        original_outputs = None
        for fraction in (0.,1.,.5,.25,.125,.0625,.03125,.015625,.0078125):
            with torch.no_grad():
                for p,a,change in zip(parameters,old,delta): p.copy_(a+fraction*change)
                loss=float(objective())
                evaluation_stream.set_state(evaluation_rng)
                points=trainer.sample(20000,ema=False,generator=evaluation_stream)
                if original_outputs is None: original_outputs=points.clone()
                distance,labels=torch.cdist(points,centers).min(1)
                motion=(points-original_outputs).norm(dim=1)/sigma
                # Full nearest-cell covariance includes spill; no radius censor.
                counts=torch.bincount(labels,minlength=100).double()
                residual=((points-centers[labels])/sigma).double()
                sums=torch.zeros((100,2),device=points.device,dtype=torch.float64).index_add_(0,labels,residual)
                outer=torch.zeros((100,4),device=points.device,dtype=torch.float64).index_add_(0,labels,(residual[:,:,None]*residual[:,None,:]).reshape(-1,4))
                mean=sums/counts.clamp_min(1)[:,None]
                cov=outer.reshape(100,2,2)/counts.clamp_min(1)[:,None,None]-mean[:,:,None]*mean[:,None,:]
                eig=torch.linalg.eigvalsh(cov[counts>=2])
                curves.append(dict(fraction=fraction,loss=loss,loss_change=loss-before,
                    armijo_bound=before+.1*fraction*slope,armijo_satisfied=loss<=before+.1*fraction*slope,
                    coverage=evaluate_samples(points,'grid100'),accuracy=evaluate_accuracy(points,'grid100'),
                    finite_motion_sigma=dict(median=float(motion.median()),p90=float(motion.quantile(.9)),max=float(motion.max())),
                    full_cell_covariance=dict(occupied_cells=int((counts>0).sum()),minimum_eigen=float(eig[:,0].min()),maximum_eigen=float(eig[:,1].max()),mean_trace=float(eig.sum(1).mean())),
                    mean_nearest_distance_sigma=float((distance/sigma).mean())))
        with torch.no_grad():
            for p,a in zip(parameters,old): p.copy_(a)
            for b,a in buffers:b.copy_(a)
        trainer.opt_g.load_state_dict(opt)
        context.streams.load_state_dict(rng)
        torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda,0)
        captured.update(curves=curves,first_order_loss_change=slope,
                        locally_downhill=slope<0,full_step_overshoots_batch_loss=curves[1]['loss']>before)
    trainer.opt_g.step=proposal
    # Public execution extension permits ONLY this discarded diagnostic update.
    # It does not alter recipe horizon or admitted 7k training.
    trainer.extend_execution(7001)
    data=context.streams.generator('data',component='target',purpose='training')
    real=sample_real('grid100',context.recipe.batch_size,device='cuda:0',generator=data)
    real_hash=hashlib.sha256(real.detach().cpu().numpy().tobytes()).hexdigest()
    trainer.step(real)
    trainer._sample_training_prior=original_sample;trainer.opt_g.step=original_step
    trainer.max_steps=7000
    context.load_state_dict(saved)
    restored_digest=state_digest(context.state_dict())
    assert restored_digest==initial_digest
    assert file_hash(checkpoint)==descriptor['sha256']
    torch.set_rng_state(caller_cpu);torch.cuda.set_rng_state(caller_cuda,0)
    for key in ('ids','offset'):captured.pop(key)
    atomic_json(output,dict(schema_version=1,qualification_input=False,
        scope='original-source reversible actual next D/G proposal at saved grid100 endpoint; oracle geometry only',
        source_commit=cert['source']['origin_commit'],source_digest=cert['source']['digest'],attempt_id=aid,
        checkpoint=dict(**descriptor,artifact_path=str(checkpoint)),initial_state_sha256=initial_digest,
        restored_state_sha256=restored_digest,all_params_optimizer_buffers_named_and_global_rng_restored=True,
        initial_recipe=saved['recipe'],real_batch_sha256=real_hash,
        hypothetical_discarded_updates=1,committed_training_updates=0,elapsed_seconds=time.monotonic()-started,
        original_receipts={n:file_hash(base/f'{n}.json') for n in ('request','result','evidence')},
        verified_scientific_python_sources=len(source_proofs),**captured))
    print(json.dumps({k:captured[k] for k in ('first_order_loss_change','locally_downhill','full_step_overshoots_batch_loss')}),flush=True)
    for point in captured['curves']:
        print(json.dumps(dict(fraction=point['fraction'],loss=point['loss'],precision=point['coverage']['precision'],motion=point['finite_motion_sigma'])),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
