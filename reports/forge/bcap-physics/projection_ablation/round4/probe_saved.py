"""Read-only endpoint decomposition; no optimizer clocks, sampler or training.

CPU FP64 gradients, public FP32 polar factor, the saved rates and smoothing.
This is a separately labelled diagnostic, never a trained gate or timestep.
"""
from pathlib import Path
import importlib.util
import json
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import atomic_json, file_hash, read_json
from experiments.forge.state import state_digest
from benchmarks.locked_shared import trajectory
from benchmarks.locked_shared.hosts import residual_student
from particlegan import ParticleRegularizer
from particlegan.gan_loss import GANLoss

helper = ROOT / 'reports/forge/bcap-tier2-search/probe_failure_states.py'
spec = importlib.util.spec_from_file_location('saved_helpers', helper)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
OUT = Path(__file__).resolve().parent


def analyze(saved, task):
    before = state_digest(saved)
    recipe = saved['applied']['recipe']
    assert recipe['optimizer_family'] == 'dualnorm' and recipe['optimizer_momentum'] == 0
    assert recipe['optimizer_smoothing'] == .001
    g, d = h.restore(saved['models'], task)
    z = saved['models']['prior0']['z'].double().detach().requires_grad_(True)
    slow, target = (x.double() for x in trajectory.trajectories())
    fake = g(slow, z)
    identity = (fake-target).square().mean()
    losses = dict(adversarial=GANLoss('non_saturating').g_loss(d(slow, fake)),
        set_coverage=trajectory.PROTOCOL['cover_weight'] * trajectory._cover(fake, target),
        latent_l2=trajectory.PROTOCOL['particle_l2'] * z.square().mean(),
        latent_spread=ParticleRegularizer(weight=trajectory.PROTOCOL['vicreg_weight'])(z))
    mask = residual_student.both_land_mask(slow, target, torch.arange(len(slow)))
    if task == 'residual_student' and bool(mask.any()):
        losses['paired_residual'] = residual_student.RESIDUAL_WEIGHT * (fake[mask]-target[mask]).square().mean()
    parameters = (*g.parameters(), z)
    gradients = {key:h.grads(loss,parameters) for key,loss in losses.items()}
    summed = tuple(sum(parts) for parts in zip(*gradients.values()))
    direct = h.grads(sum(losses.values()),parameters)
    assert all(torch.allclose(a,b,rtol=1e-10,atol=1e-10) for a,b in zip(summed,direct))
    identity_grad = h.flatten(h.grads(identity,parameters))
    normals = {key:h.flatten(gradients[key]) for key in ('adversarial','paired_residual') if key in gradients}
    def normalized(gs):
        return h.flatten([h.direction(grad,.03 if i==len(gs)-1 else .012,prior=i==len(gs)-1)
                          for i,grad in enumerate(gs)])
    total = normalized(summed)
    normalized_terms = {key:normalized(gs) for key,gs in gradients.items()}
    naive_sum = sum(normalized_terms.values())
    def motion(vec):
        return dict(norm=float(vec.norm()), identity_derivative=float(identity_grad.dot(vec)),
                    protected_derivatives={key:float(n.dot(vec)) for key,n in normals.items()})
    leaveouts = {key:motion(normalized(tuple(a-b for a,b in zip(summed,gs))))
                 for key,gs in gradients.items()}
    assert state_digest(saved) == before
    return dict(identity_mse=float(identity.detach()), both_land_rows=int(mask.sum()),
        loss_values={key:float(value.detach()) for key,value in losses.items()},
        component_gradient_norms={key:dict(network=float(h.flatten(gs[:-1]).norm()),prior=float(gs[-1].norm()))
                                  for key,gs in gradients.items()},
        gradient_cosines={f'{a}__{b}':h.cosine(h.flatten(gradients[a]),h.flatten(gradients[b]))
                          for i,a in enumerate(losses) for b in list(losses)[i+1:]},
        normalized_combined=motion(total), individually_normalized={key:motion(v) for key,v in normalized_terms.items()},
        leave_one_out=leaveouts, normalized_sum=motion(naive_sum),
        normalized_nonadditivity_ratio=float((total-naive_sum).norm()/total.norm()),
        checkpoint_unchanged=True, caveat='Endpoint derivatives in a distinct CPU FP64 diagnostic; not actual rounded training steps, finite loss guarantees or causal training ablations.')


def main():
    torch.set_num_threads(1)
    start=time.monotonic();rng=torch.get_rng_state().clone()
    inputs=[]
    original=read_json(ROOT/'reports/forge/bcap-tier2-search/failure-state-analysis.json')
    for task in ('trajectory','residual_student'):
        entry=next(x for x in original['artifact_proofs'] if x['task']==task and x['path'].endswith('provenance-state.pt'))
        inputs.append(dict(cohort='original_winner',task=task,path=entry['path'],sha256=entry['sha256'],source_digest=original['source_digest']))
    prior=read_json(ROOT/'reports/forge/bcap-physics/constraint_geometry/round3/provenance.json')
    for p in prior['proofs']:
        if p['task_id'] in ('trajectory','residual_student'):
            descriptor=p['provenance_checkpoint']
            inputs.append(dict(cohort='round3_'+p['role'],task=p['task_id'],path=str(Path(descriptor['artifact_root'])/descriptor['path']),
                               sha256=descriptor['sha256'],source_digest=p['source_digest']))
    rows=[]
    for item in inputs:
        assert file_hash(Path(item['path']))==item['sha256']
        saved=torch.load(item['path'],map_location='cpu',weights_only=False)
        rows.append(dict(**item,diagnostics=analyze(saved,item['task'])))
    assert torch.equal(rng,torch.get_rng_state())
    result=dict(schema_version=1,qualification_input=False,scope='saved_endpoint_decomposition',
        optimizer_updates_added=0,sampling_draws_added=0,analysis_seconds=time.monotonic()-start,
        helper_sha256=file_hash(helper),rows=rows,torch_rng_unchanged=True)
    atomic_json(OUT/'saved-attribution.json',result)
    print(json.dumps(dict(rows=len(rows),analysis_seconds=result['analysis_seconds'])))

if __name__=='__main__':main()
