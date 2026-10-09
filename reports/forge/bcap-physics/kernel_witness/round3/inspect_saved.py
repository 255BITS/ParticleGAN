"""Saved-state force audit before admission; no optimizer steps or seed study."""
from pathlib import Path
import json
import math
import time

import numpy as np
from scipy.special import ndtri
import torch

from particlegan.kernel_witness import kernel_witness_loss
from experiments.forge.api import task_formulation_context
from experiments.forge.contracts import atomic_json, file_hash, read_json
from experiments.forge.rng import NamedStreams
from experiments.forge.vectorprofiles import build_vector_models
from benchmarks.transfer_suite.vector_tasks import sample_target

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
ALCHEMY = Path('/tmp/bcap-physics-round2-20261009/alchemy/done.json')


def teacher(spec, n=256, *, width=1., missing=False):
    masses=np.array(spec['masses'])
    if missing:
        masses[-1]=0; masses/=masses.sum()
    group=np.searchsorted(np.cumsum(masses), (np.arange(n)+.5)/n)
    quantiles=(np.arange(n)[:,None]+.5)*np.array([math.sqrt(2),math.sqrt(3)]) % 1
    normal=ndtri(np.clip(quantiles,1e-8,1-1e-8))
    means=np.array(spec['means']); covs=np.array(spec['covariances'])
    return torch.from_numpy(means[group]+width*np.einsum('nij,nj->ni',np.linalg.cholesky(covs)[group],normal))


def cosine(a,b):
    denominator=a.norm()*b.norm()
    return float(a.dot(b)/denominator) if denominator>0 else None


def gradients(value, params):
    values=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p,g in zip(params,values)])


def audit_state(task, binding, source_name):
    path=Path(binding['path']); assert file_hash(path)==binding['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=True)
    winner=read_json(ROOT/'configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json')
    context=task_formulation_context(winner,task,dict(seed=0),device='cpu',root=ROOT)
    spec=task['execution']['host_definition'];g,d=build_vector_models(context,spec)
    g.load_state_dict(saved['trainer']['models']['G']);d.load_state_dict(saved['trainer']['models']['D'])
    g.double().eval();d.double().eval().requires_grad_(False)
    z=saved['trainer']['models']['prior']['z'].double().clone().requires_grad_()
    # Explicit deterministic antithetic cubature over every saved row. This is
    # a derivative cohort, not public served-law evidence or the last G batch.
    offsets=task['execution']['prior']['sigma']*math.sqrt(z.shape[1])*torch.cat((torch.eye(z.shape[1]),-torch.eye(z.shape[1]))).double()
    latent=(z[None]+offsets[:,None]).flatten(0,1)
    fake=g(latent)
    streams=NamedStreams(0);data=streams.generator('data',component='target',purpose='training',device='cpu')
    for step in range(task['execution']['steps']):
        real=sample_target(spec,spec['batch'],data,step)
    data_key=next(key for key,v in saved['streams']['manifest']['bindings'].items()
                  if v['family']=='data' and v['component']=='target' and v['purpose']=='training')
    assert torch.equal(data.get_state(),saved['streams']['states'][data_key])
    real=real.double();loss,parts=kernel_witness_loss(fake,real,return_parts=True)
    critic=context.recipe.make_loss().g_loss(d(fake),d(real))
    groups={'generator':list(g.parameters()),'prior':[z]};forces={}
    for role,params in groups.items():
        c=gradients(critic,params);w=gradients(loss,params)
        a=gradients(parts['attraction'],params);r=gradients(parts['repulsion'],params)
        forces[role]=dict(critic_norm=float(c.norm()),witness_norm=float(w.norm()),
            attraction_norm=float(a.norm()),repulsion_norm=float(r.norm()),
            witness_critic_cosine=cosine(w,c),attraction_repulsion_cosine=cosine(a,r),
            witness_over_critic_norm=float(w.norm()/c.norm()),
            residual_over_term_norms=float(w.norm()/(a.norm()+r.norm())),
            finite=bool(torch.isfinite(w).all()))
    return dict(source=source_name,checkpoint=binding,task_id=task['id'],
                generated_cubature_count=len(fake),actual_last_real_batch_count=len(real),
                data_stream_replay_matches=True,loss=float(loss.detach()),
                parts={k:float(v.detach()) for k,v in parts.items()},parameter_forces=forces)


def main():
    started=time.monotonic();torch.set_num_threads(1)
    before=torch.get_rng_state().clone()
    original_path=ROOT/'reports/forge/bcap-tier2-search/failure-state-analysis.json'
    original=read_json(original_path);alchemy=read_json(ALCHEMY)
    diagnostics=[];controls=[]
    for tid in ('vector_unequal_mass','vector_unequal_width','vector_two_broad'):
        task=read_json(ROOT/'configs/forge/tasks'/f'{tid}.json')
        if tid!='vector_two_broad':
            binding=next(b for b in original['artifact_proofs'] if b['task']==tid and b['path'].endswith('provenance-state.pt'))
            diagnostics.append(audit_state(task,binding,'original_winner'))
        for arm in ('candidate', 'control'):
            row=next(r for r in alchemy['per_task_attempts'] if r['arm']==arm and r['task_id']==tid)
            result=read_json(Path(row['artifact_root'])/'result.json')['task_results'][0]
            descriptor=result['evidence']['provenance_checkpoint']
            binding=dict(path=str(Path(descriptor['artifact_root'])/descriptor['path']),sha256=descriptor['sha256'],
                         original_source_digest=result['evidence']['provenance_checkpoint'].get('source_digest',
                             '587006be44148b1a2adfd905bd01d5631061ed4206c27c85e7751cc2a3f29d5a'),original_arm=arm)
            diagnostics.append(audit_state(task,binding,f'alchemy_round2_{arm}'))
        spec=task['execution']['host_definition'];real=teacher(spec)
        for name,fake in [('empirical_null',real),('collapse',teacher(spec,width=0)),
                          ('double_width',teacher(spec,width=2)),('missing_last_component',teacher(spec,missing=True))]:
            fake=fake.clone().requires_grad_();value=kernel_witness_loss(fake,real)
            grad,=torch.autograd.grad(value,fake)
            controls.append(dict(task_id=tid,control=name,loss=float(value.detach()),gradient_rms=float(grad.square().mean().sqrt())))
        print(tid,[d['parameter_forces'] for d in diagnostics if d['task_id']==tid],flush=True)
    atomic_json(OUT/'saved-evidence.json',dict(schema_version=1,qualification_input=False,optimizer_updates_added=0,
        new_random_sampling_draws=0,replayed_archived_data_batches=True,scope='CPU_float64_frozen_state_parameter_force_audit',
        original_source=original['source_digest'],original_report_sha256=file_hash(original_path),
        alchemy_done_sha256=file_hash(ALCHEMY),diagnostics=diagnostics,controls=controls,
        limitations='Cubature derivatives and raw gradients, not exact applied DualNorm steps, independent heldout prediction, or qualification. Real minibatch replay matches saved data stream. Adaptive kernels and biased V-statistic are not an unbiased fixed-kernel population gradient.',
        global_rng_unchanged=torch.equal(before,torch.get_rng_state()),cpu_wall_seconds=time.monotonic()-started))


if __name__=='__main__':main()
