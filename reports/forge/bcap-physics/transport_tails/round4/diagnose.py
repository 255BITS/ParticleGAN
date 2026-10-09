"""Frozen v2/v3 tail fields and finite G/prior role probes; no training."""
from pathlib import Path
import sys, time, argparse
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from experiments.forge.contracts import read_json, atomic_json, file_hash
from experiments.forge.api import task_formulation_context
from experiments.forge.vectorprofiles import resolve_vector_spec, build_vector_models
from experiments.forge.rng import NamedStreams
from experiments.forge import tier1_media
from benchmarks.transfer_suite.vector_tasks import sample_target


def census(points, spec, vectors):
    means = points.new_tensor(spec['means'])
    cov = points.new_tensor(spec['covariances'])
    assigned = torch.cdist(points, means).argmin(1)
    delta = points - means[assigned]
    radius = torch.einsum('ni,nij,nj->n', delta, torch.linalg.inv(cov)[assigned], delta).sqrt()
    rows = []
    for mode in range(len(means)):
        for scope, mask in [('core', radius <= 3), ('tail', radius > 3)]:
            selected = (assigned == mode) & mask
            if not selected.any():
                continue
            radial = delta[selected] / delta[selected].norm(dim=1, keepdim=True).clamp_min(1e-12)
            row = {'component': mode, 'scope': scope, 'count': int(selected.sum())}
            for name, vector in vectors.items():
                v = vector[selected]
                dot = (v * radial).sum(1)
                row[name] = {'mean_norm': float(v.norm(dim=1).mean()),
                             'mean_outward_radial': float(dot.mean()),
                             'inward_fraction': float((dot < 0).float().mean())}
            rows.append(row)
    return rows


def main():
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--current',action='store_true');args=parser.parse_args()
    old = Path(__file__).resolve().parent if args.current else ROOT/'reports/forge/bcap-physics/kinetic_transport/round3'
    provenance = read_json(old/'provenance.json')
    rows = []
    for source in provenance['attempts']:
        if not source['task_id'].startswith('vector_'):
            continue
        directory = Path(source['artifact_root'])
        for name, digest in source['inputs'].items():
            assert file_hash(directory/name) == digest
        request = read_json(directory/'request.json')['request']
        task = request['tasks'][source['task_id']]
        result = read_json(directory/'result.json')['task_results'][0]
        evidence = result['evidence']
        records, _ = tier1_media._scored_outputs(task, evidence, directory)
        outputs = records[-1]['samples']
        descriptor = source['provenance_checkpoint']
        checkpoint = Path(descriptor['artifact_root'])/descriptor['path']
        assert file_hash(checkpoint) == descriptor['sha256']
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        spec = resolve_vector_spec(task)
        data = NamedStreams(0).generator('data', component='target', purpose='training', device='cpu')
        for step in range(task['execution']['steps']):
            real = sample_target(spec, spec['batch'], data, step)
        key = next(k for k,b in saved['streams']['manifest']['bindings'].items()
                   if b['family']=='data' and b['component']=='target' and b['purpose']=='training')
        assert torch.equal(data.get_state(), saved['streams']['states'][key])
        context = task_formulation_context(request['candidate'], task, request['protocol'], device='cuda:0', root=ROOT)
        g,d = build_vector_models(context, spec)
        trainer = context.build_trainer(g,d,max_steps=saved['trainer']['completed_steps'])
        trainer.load_state_dict(saved['trainer'])
        trainer.G.train(); trainer.D.eval(); trainer.D.requires_grad_(False)
        real = real.to(trainer.device); outputs = outputs.to(trainer.device)
        def losses(x):
            values = {'adversarial':trainer.loss.g_loss(trainer.D(x),trainer.D(real)),
                    'sliced':trainer.recipe.kinetic_transport_loss(x,real),
                    'local':trainer.recipe.kinetic_transport_local_loss(x,real)}
            if args.current:values['tail']=trainer.recipe.kinetic_transport_tail_loss(x,real)
            return values
        fields = {k:[] for k in (('adversarial','sliced','local','tail') if args.current else ('adversarial','sliced','local'))}
        for block in outputs.split(len(real)):
            fake = block.detach().clone().requires_grad_(True)
            for name, loss in losses(fake).items():
                fields[name].append(-torch.autograd.grad(loss, fake, retain_graph=True)[0].detach())
        fields = {k:torch.cat(v) for k,v in fields.items()}
        fields['total'] = sum(fields.values())
        latent_rng = torch.Generator(device=trainer.device); latent_rng.set_state(saved['trainer']['streams']['latent_generator'])
        noise_rng = torch.Generator(device=trainer.device); noise_rng.set_state(saved['trainer']['streams']['prior_noise_generator'])
        latent, indices = trainer.prior.sample(len(real), generator=latent_rng, noise_generator=noise_rng)
        jitter = latent.detach() - trainer.prior.z[indices].detach()
        fake = trainer.G(latent); initial = sum(losses(fake).values())
        params = [p for group in trainer.opt_g.param_groups for p in group['params']]
        prior_ids = {id(p) for p in trainer.prior.parameters()}
        before = [p.detach().clone() for p in params]
        trainer.opt_g.zero_grad(); initial.backward()
        gradient = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in params]
        trainer.opt_g.set_sampled_rows(trainer.prior.z,indices); trainer.opt_g.step()
        delta = [p.detach().clone()-b for p,b in zip(params,before)]
        motions = {}; finite = []
        with torch.no_grad():
            for role in ('generator','prior','joint'):
                role_delta = [u if role=='joint' or (id(p) in prior_ids)==(role=='prior') else torch.zeros_like(u)
                              for p,u in zip(params,delta)]
                slope = float(sum((v*u).sum() for v,u in zip(gradient,role_delta)))
                trials=[]
                for alpha in (.5,1.):
                    for p,b,u in zip(params,before,role_delta):p.copy_(b+alpha*u)
                    trial = trainer.G(trainer.prior.z[indices]+jitter)
                    values = {k:float(v) for k,v in losses(trial).items()}
                    change = sum(values.values())-float(initial.detach())
                    trials.append({'scale':alpha,'loss_change':change,'losses':values,
                                   'finite_curvature_remainder':change-alpha*slope})
                    if alpha==1.: motions[role]=trial-fake.detach()
                finite.append({'role':role,'directional_derivative':slope,'trials':trials})
            for p,b in zip(params,before):p.copy_(b)
            centers = trainer.G(trainer.prior.z)
            assigned = torch.cdist(centers,centers.new_tensor(spec['means'])).argmin(1)
        rows.append({'archived_arm':source['arm'],'mechanism':'local-v2' if source['arm']=='control' else ('tail-moments-r4' if args.current else 'backtrack-v3'),
                     'task_id':task['id'],'attempt_id':source['attempt_id'],'source_digest':source['source_digest'],
                     'checkpoint_sha256':file_hash(checkpoint),'metric_endpoint':evidence['live'],
                     'center_counts':torch.bincount(assigned,minlength=len(spec['means'])).tolist(),
                     'served_output_fields':census(outputs,spec,fields),
                     'finite_probe_role_motions':census(fake.detach(),spec,motions),
                     'finite_probe':finite,'probe_generated_draws':len(real),'exact_final_data_stream_verified':True})
        print({'event':'saved_tail_probe','arm':source['arm'],'task':task['id']},flush=True)
    torch.cuda.synchronize()
    receipt={'schema_version':1,'scope':'saved_endpoint_diagnostic_separate_from_training_steps',
             'target_labels_covariances':'posthoc decomposition only; never fed to any training objective',
             'optimizer_training_updates_added':0,'in_memory_optimizer_proposals':len(rows),
             'generated_probe_draws':sum(r['probe_generated_draws'] for r in rows),
             'archived_checkpoints_streams_mutated':False,'wall_seconds':time.monotonic()-started,
             'device':'physical GPU1','diagnostics':rows}
    atomic_json(Path(__file__).with_name('current-state-diagnostics.json' if args.current else 'saved-tail-diagnostics.json'),receipt)
    print({'event':'saved_tail_probe_complete','seconds':receipt['wall_seconds']},flush=True)

if __name__=='__main__':main()
