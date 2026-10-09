"""Frozen center population and nonlinear Gaussian-kernel quadrature; no training.

The population of G(c_i) is exhaustive. Conditional kernel moments use a
deterministic five-node-per-axis Gauss-Hermite rule, not exact Gaussian integration. Component
assignments are fixed from G(c_i) for this diagnostic only; served-law grading
retains its original assignments and sampling. No global or checkpoint RNG moves.
"""
from pathlib import Path
import itertools
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments.forge.contracts import atomic_json, file_hash, read_json
from experiments.forge.state import state_digest
from lib.toy_models import SimpleMLPGenerator

OUT = Path(__file__).resolve().parent
ROLE = Path('/home/martyn/dev/ParticleGAN-bcap-r4-role_motion/reports/forge/bcap-physics/role_motion/round4')


def covariance(x):
    delta = x-x.mean(0)
    return delta.T @ delta / len(x)


def diagnose(source, receipt):
    desc = receipt['provenance_checkpoint']
    path = Path(desc['artifact_root'])/desc['path']
    assert file_hash(path) == desc['sha256']
    saved = torch.load(path, map_location='cpu', weights_only=False)
    digest = state_digest(saved)
    task_id = receipt['task_id']
    directory = Path(receipt.get('artifact_root') or receipt['local_artifact_root'])
    if not (directory/'request.json').is_file():
        directory = directory.parent
    request = read_json(directory/'request.json')['request']
    task = request['tasks'][task_id]
    gs = saved['trainer']['models']['G']
    hidden = gs['net.0.weight'].shape[0]
    layers = (len([k for k in gs if k.endswith('weight')])-1)
    with torch.device('meta'):
        model = SimpleMLPGenerator(gs['net.0.weight'].shape[1], hidden, layers,
                                   gs[f'net.{2*layers}.weight'].shape[0])
    model.load_state_dict(gs, assign=True)
    model = model.to('cuda:0').eval()
    centers = saved['trainer']['models']['prior']['z'].to('cuda:0')
    sigma = task['execution']['prior']['sigma']
    if task_id == 'grid100':
        from benchmarks.toy100.problems import evaluation_geometry
        means, target_sigma = evaluation_geometry(task_id, device='cuda:0')
        cov = torch.eye(2,device='cuda:0')[None].expand(len(means),2,2)*target_sigma**2
    else:
        spec = task['execution']['host_definition']
        means = centers.new_tensor(spec['means'])
        cov = centers.new_tensor(spec['covariances'])
    nodes, weights = np.polynomial.hermite.hermgauss(5)
    dimension = centers.shape[1]
    pairs = list(itertools.product(range(5), repeat=dimension))
    count = len(pairs)
    jitter = centers.new_tensor([[nodes[i] for i in pair] for pair in pairs]) * (2**.5*sigma)
    weight = torch.tensor([np.prod([weights[i] for i in pair])/np.pi**(dimension/2) for pair in pairs],
                          dtype=torch.float64, device='cuda:0')
    with torch.no_grad():
        y0 = torch.cat([model(c) for c in centers.split(2048)]).double()
        ys = torch.cat([model((c[:, None]+jitter).flatten(0,1)).reshape(len(c),count,-1)
                        for c in centers.split(max(1,4096//count))]).double()
        conditional_mean = (ys*weight[None,:,None]).sum(1)
        conditional_delta = ys-conditional_mean[:,None]
        within = torch.einsum('nki,nkj,k->nij',conditional_delta,conditional_delta,weight)
        assigned = torch.cdist(y0,means.double()).argmin(1)
        sampled_assign = torch.cat([torch.cdist(block,means.double()).argmin(1)
                                    for block in ys.flatten(0,1).split(4096)]).reshape(len(ys),count)
        migration = (sampled_assign != assigned[:,None]).double() @ weight
        rows = []
        for k in range(len(means)):
            mask = assigned == k
            n = int(mask.sum())
            if n < 2:
                rows.append({'component':k,'centers':n,'missing_shape':True})
                continue
            target = cov[k].double()
            c0 = covariance(y0[mask]); between = covariance(conditional_mean[mask])
            w = within[mask].mean(0)
            total = between+w
            trace = float(target.trace())
            all_y = ys[mask]; mean = (all_y*weight[None,:,None]).sum((0,1))/n
            delta = all_y-mean
            direct = torch.einsum('nki,nkj,k->ij',delta,delta,weight)/n
            rows.append({'component':k,'centers':n,
                'center_output_covariance':c0.tolist(),
                'center_output_trace_over_target':float(c0.trace())/trace,
                'conditional_mean_covariance':between.tolist(),
                'conditional_mean_trace_over_target':float(between.trace())/trace,
                'conditional_jitter_covariance':w.tolist(),
                'conditional_jitter_trace_over_target':float(w.trace())/trace,
                'between_fraction_of_fixed_assignment_trace':float(between.trace()/total.trace()),
                'finite_quadrature_total_covariance_error':float((total-target).norm()/target.norm()),
                'center_only_covariance_error':float((c0-target).norm()/target.norm()),
                'quadrature_decomposition_max_abs_error':float((direct-total).abs().max()),
                'center_to_conditional_mean_rms':float((conditional_mean[mask]-y0[mask]).square().sum(1).mean().sqrt()),
                'weighted_assignment_migration_fraction':float(migration[mask].mean())})
    assert file_hash(path) == desc['sha256'] and state_digest(saved) == digest
    torch.cuda.synchronize()
    return {'task_id':task_id,'source':source,'candidate_id':request['candidate']['id'],
        'source_commit':request['source']['origin_commit'],'source_digest':request['source']['digest'],
        'request_path':str(directory/'request.json'),'request_sha256':file_hash(directory/'request.json'),
        'checkpoint_path':str(path),'checkpoint_sha256':desc['sha256'],
        'checkpoint_state_sha256':digest,'prior_sigma':sigma,'all_centers':len(centers),
        'center_counts':torch.bincount(assigned,minlength=len(means)).tolist(),
        'kernel_forward_evaluations':len(centers)*count,'quadrature_nodes_per_kernel':count,
        'component_summary':rows,
        'average_between_fraction':sum(v['between_fraction_of_fixed_assignment_trace'] for v in rows if not v.get('missing_shape'))/sum(not v.get('missing_shape') for v in rows)}


def main():
    start = time.monotonic()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    rng = torch.get_rng_state().clone()
    local_path = ROOT/'reports/forge/bcap-physics/transport_tails/round4/provenance.json'
    local = read_json(local_path)
    rows = [diagnose('round4-local-v2',r) for r in local['attempts']
            if r['arm']=='control' and r['task_id'].startswith('vector_')]
    role_path = ROLE/'provenance.json'
    role = read_json(role_path)
    native = next(r for r in role['receipts'] if r['task_id']=='grid100'
                  and r['candidate_id'].startswith('bcap-dualnorm--'))
    rows.append(diagnose('round4-native-winner',native))
    assert torch.equal(rng,torch.get_rng_state())
    finite = ROOT/'reports/forge/bcap-physics/transport_tails/round4/saved-tail-diagnostics.json'
    result = {
        'schema_version':1,'qualification_input':False,
        'scope':'exhaustive_center_outputs_and_fixed_assignment_nonlinear_kernel_quadrature',
        'optimizer_updates_added':0,'sampling_draws_added':0,'global_rng_unchanged':True,
        'quadrature_rule':'5 nodes per latent axis, Gauss-Hermite for independent Gaussian jitter; approximate nonlinear conditional moments',
        'training_oracles':'none; target labels/moments are diagnostic only',
        'full_allowance_seconds':600,'analysis_seconds':time.monotonic()-start,
        'source_receipts':{str(local_path):file_hash(local_path),str(role_path):file_hash(role_path)},
        'retained_finite_G_prior_probe':{'path':str(finite.relative_to(ROOT)),'sha256':file_hash(finite),
            'scope':'original six frozen-critic proposals using retained last target batch and next latent draw; not actual past-update attribution'},
        'diagnostics':rows}
    archive = Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/component_tails/saved-component-full.json')
    atomic_json(archive,result)
    for row in rows:
        if row['task_id'] != 'grid100':
            continue
        components = row.pop('component_summary')
        metrics = [key for key,value in components[0].items() if isinstance(value,float)]
        row['component_summary_aggregate'] = {key:{
            'mean':float(np.mean([v[key] for v in components if key in v])),
            'median':float(np.median([v[key] for v in components if key in v])),
            'min':float(np.min([v[key] for v in components if key in v])),
            'max':float(np.max([v[key] for v in components if key in v]))} for key in metrics}
        row['missing_or_single_center_components'] = [v['component'] for v in components if v.get('missing_shape')]
    result['full_component_archive'] = {'path':str(archive),'sha256':file_hash(archive)}
    atomic_json(OUT/'saved-component-diagnostics.json',result)
    print({'event':'component_diagnostic_complete','seconds':time.monotonic()-start,
           'between_fraction':{r['task_id']:r['average_between_fraction'] for r in rows}},flush=True)


if __name__ == '__main__':
    main()
