"""Independent software cohort: analytic law and B/current default API parity.

Exactly two actual native API updates in each isolated default worker (four
total). They have no quality/GIF/clock/qualification or paid-budget credit.
No queue access, samplers outside the original public API, or source mutations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT = Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-tier1-search-20261012/queue/snapshots/2b04f2853c511c9bcd417ef41e8fd773e97e8c67c9e057da8abdd4fef2de5f12')
TASKS = ('gaussian1d_smoke', 'two_pole', 'unused_token_hold', 'ae_gan_hold',
         'ring16_acquisition', 'five_word_joint_smoke', 'clockfree_audit_measurement_v1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_request():
    path = ROOT/'reports/forge/r1r2-energy-distance-repair/publication/results.json'
    assert sha(path) == 'cb3f7b89d1677591a12dd86377d072b55cba1b95e99060b2841cdf42c418b75c'
    published = json.loads(path.read_text())
    row = next(r for r in published['certified_results'] if r['role']=='non_saturating'
               and r['task_id']=='ring16_acquisition')
    return json.loads((Path(row['checkpoint']['artifact_root']).parent/'request.json').read_text())['request']


def default_worker(source_root):
    import torch
    import particlegan.training
    from benchmarks.toy_audit.gaussian1d_quality import sample_target
    from experiments.forge.api import task_formulation_context
    from experiments.forge.state import state_digest
    from experiments.forge.vectorprofiles import build_vector_models, resolve_vector_spec
    assert Path(particlegan.training.__file__).resolve() == source_root/'particlegan/training.py'
    # Match the saved ambient CPU RNG as well as protocol-owned named streams.
    # Separate subprocesses otherwise begin with different OS-seeded states.
    torch.manual_seed(0)
    request = original_request()
    task = request['tasks']['gaussian1d_smoke']
    context = task_formulation_context(request['candidate'],task,request['protocol'],device='cpu',root=ROOT)
    spec = resolve_vector_spec(task,root=ROOT)
    trainer = context.build_trainer(*build_vector_models(context,spec),max_steps=task['execution']['steps'])
    assert getattr(trainer.recipe,'output_mmd_weight',0) == 0
    initial_state=context.state_dict()
    initial = state_digest(initial_state)
    generator = context.streams.generator('data',component='target',purpose='training',device='cpu')
    results,targets = [],[]
    for step in range(2):
        batch=sample_target(spec,trainer.recipe.batch_size,generator,step)
        targets.append(state_digest(batch))
        result=trainer.step(batch,collect_stats=True)
        assert 'output_mmd' not in result
        results.append(state_digest(result))
    assert trainer.completed_steps == 2
    final_state=context.state_dict()
    print(json.dumps(dict(initial_state_sha256=initial,final_state_sha256=state_digest(final_state),
        initial_fields_sha256={k:state_digest(v) for k,v in initial_state.items()},
        final_fields_sha256={k:state_digest(v) for k,v in final_state.items()},
        trainer_fields_sha256={k:state_digest(v) for k,v in final_state['trainer'].items()},
        results_sha256=results,target_batches_sha256=targets,actual_software_updates=2),sort_keys=True))


def independent_gradient():
    import numpy as np
    import torch
    from particlegan.kernel_mmd import kernel_mmd_u
    # Fixed arithmetic, unequal row counts and full 168-dimensional word space.
    x=np.sin(np.arange(4*168).reshape(4,168)*.017)
    y=np.cos(np.arange(7*168).reshape(7,168)*.013+.2)
    xx,xy,yy=x[:,None]-x[None],x[:,None]-y[None],y[:,None]-y[None]
    widths=(.0625,.125,.25,.5,1.,2.,4.)
    value,gradient=0.,np.zeros_like(x)
    for width in widths:
        kxx,kxy,kyy=(np.exp(-np.square(delta).sum(-1)/(2*width**2)) for delta in (xx,xy,yy))
        value+=kxx[~np.eye(4,dtype=bool)].mean()+kyy[~np.eye(7,dtype=bool)].mean()-2*kxy.mean()
        gradient+=(-2*(xx*kxx[:,:,None]).sum(1)/(4*3)+2*(xy*kxy[:,:,None]).sum(1)/(4*7))/width**2
    value/=len(widths);gradient/=len(widths)
    fake=torch.from_numpy(x).requires_grad_(True)
    real=torch.from_numpy(y).requires_grad_(True)
    actual=kernel_mmd_u(fake,real)
    fake_gradient,target_gradient=torch.autograd.grad(actual,(fake,real),allow_unused=True)
    value_error=abs(float(actual.detach())-value)
    gradient_error=float(abs(fake_gradient.numpy()-gradient).max())
    assert value_error<1e-12 and gradient_error<1e-12 and target_gradient is None
    return dict(sample_shape=[4,168],target_shape=[7,168],value_error=value_error,
        gradient_max_error=gradient_error,error_bound=1e-12,targets_detached=True,optimizer_updates=0)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--default-worker',action='store_true')
    parser.add_argument('--source-root',type=Path)
    args=parser.parse_args()
    if args.default_worker:
        return default_worker(args.source_root.resolve())
    request=original_request()
    all_pins={}
    for name in TASKS:
        pins=request['tasks'][name]['evaluation'].get('sources',{})
        all_pins.update(pins)
        assert all(sha(ROOT/path)==sha(SNAPSHOT/path)==digest for path,digest in pins.items())
    workers=[]
    for source in (SNAPSHOT,ROOT):
        env=dict(os.environ,PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1',
                 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        completed=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--default-worker',
                                  '--source-root',str(source)],cwd=source,env=env,
                                 capture_output=True,text=True,check=True)
        workers.append(json.loads(completed.stdout))
    if workers[0]!=workers[1]:
        print(json.dumps(dict(default_worker_mismatch=workers),sort_keys=True,indent=2))
        raise AssertionError('Original B and current default API states differ')
    print(json.dumps(dict(schema_version=1,scope='independent_cpu_software_prerequisite_not_qualification',
        original_unique_evaluator_pins=len(all_pins),original_task_count=7,
        default_original_snapshot=str(SNAPSHOT),default_api_exact_parity=True,
        default_worker_receipts=workers,independent_mmd=independent_gradient(),
        actual_native_software_updates=4,paid_jobs=0,queue_mutations=0,
        qualifications_added=0,reproduction_source_sha256=sha(Path(__file__))),sort_keys=True,indent=2))


if __name__=='__main__':
    main()
