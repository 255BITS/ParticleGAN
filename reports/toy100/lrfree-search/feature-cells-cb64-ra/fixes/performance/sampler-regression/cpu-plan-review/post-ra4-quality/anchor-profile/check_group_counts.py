"""Integer/fallback/shape and complete saved reaction+row-state equivalence."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from fixture_utils import convert, load_package, nested_equal, network, sha

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'pkg-CB64-RA6'
PROPOSAL = HERE / 'pkg-GROUP-COUNT'
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
old_module, old_birth = load_package(BASE, 'group_reference')
new_module, new_birth = load_package(PROPOSAL, 'group_proposal')


def method_text(source):
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'FeatureCellSnapshot')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_group_counts')
    return '\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])


def bits(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and torch.equal(
        left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def planner(module, birth, case):
    value = convert(case, 'cpu')
    snapshot = module.FeatureCellSnapshot.__new__(module.FeatureCellSnapshot)
    snapshot.__dict__.update(value['snapshot'])
    G, D, ema_G = (network(value['models'][key]) for key in ('G', 'D', 'ema_G'))
    trainer = SimpleNamespace(G=G, D=D)
    ctrl = SimpleNamespace(_heads=[D[4]], sample_shape=(2,))
    current = birth.learned_latent_features(ctrl, trainer, G)
    average = birth.learned_latent_features(ctrl, trainer, ema_G)
    path = ROOT / f'validation-ra4/learned/training/toy/CB64-RA4/checkpoint-{case["step"]:04d}.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False)['trainer']
    stream = torch.Generator().set_state(saved['cpu_rng'])
    def birth_planner(children, parents, supported, budget):
        return birth.plan_real_anchor_births(snapshot, value['q'], value['flags'], value['pvalues'],
            value['comparison'], value['latents'], current, ema_latents=value['ema_latents'],
            ema_feature_of_latent=average, previous_children=children, previous_copy_parents=parents,
            supported_counts=supported, max_moves=budget)
    child, parent, ordinary = snapshot.ordinary_transport(value['q'], value['flags'], value['comparison'],
        generator=stream, pvalues=value['pvalues'], birth_planner=birth_planner)
    plan = ordinary['novel_birth_plan']
    ic, ip, isolation = birth.plan_residual_isolation(snapshot, value['q'], value['flags'], value['pvalues'],
        plan, generator=stream, copy_children=child, copy_parents=parent,
        supported_counts=ordinary['planned_supported_counts'])
    actions = dict(child=child, parent=parent, isolation_child=ic, isolation_parent=ip,
        ordinary=ordinary, isolation=isolation, work=deepcopy(snapshot.work), planning_rng=stream.get_state().clone())

    # Exact saved row moments/history and graph, mechanically copied/reset without an optimizer step.
    z, ez = saved['models']['prior']['z'], saved['models']['ema_prior']['z']
    prior = SimpleNamespace(z=torch.nn.Parameter(z.clone()), num_particles=len(z))
    ema = SimpleNamespace(z=torch.nn.Parameter(ez.clone()), num_particles=len(z))
    candidates = [v for v in saved['optimizers'][0]['state'].values()
        if isinstance(v.get('exp_avg'), torch.Tensor) and v['exp_avg'].shape == z.shape]
    assert len(candidates) == 1
    moments = deepcopy(candidates[0])
    history = saved['optimizers'][0]['regularizer']['latent']['history'].clone()
    opt = SimpleNamespace(state={prior.z:moments}, latent_history=history)
    controller = SimpleNamespace(latent_bandwidth=saved['controller']['latent_bandwidth'])
    trainer = SimpleNamespace(prior=prior, ema_prior=ema, opt_g=opt, controller=controller)
    backend = module.FeatureCellBirthDeath.__new__(module.FeatureCellBirthDeath)
    degree = saved['birth_death']['lineage_neighbors'].shape[1]
    backend.lineage = module.LatentLineage(len(z), degree, torch.device('cpu'))
    backend.lineage.neighbors = saved['birth_death']['lineage_neighbors'].clone()
    backend.latent_geometry = module.BoundedLatentGeometry(rank=8, neighbors=64, chunk=256, lineage=backend.lineage)
    backend._copy_parent_rows = None
    backend.stream = stream
    for key in ('S', 'W', 'n', 'pending', 'radius', 'anchor'):
        setattr(backend, key, deepcopy(saved['birth_death'][key]))
    if len(child):
        backend._move(trainer, child, parent)
    if len(ic):
        backend._move(trainer, ic, ip)
    birth.apply_anchor_births(trainer, backend, plan)
    backend.lineage.validate(backend.lineage.neighbors)
    final = dict(live=prior.z.detach().clone(), ema=ema.z.detach().clone(), moments=moments,
        history=history, evidence={key:getattr(backend,key) for key in ('S','W','n','pending','radius','anchor')},
        lineage=backend.lineage.neighbors, graph_work=backend.lineage.work, final_rng=stream.get_state().clone())
    return actions, final


def main():
    base_sources = {p.name: sha(p) for p in sorted((BASE / 'particlegan').glob('*.py'))}
    proposed_sources = {p.name: sha(p) for p in sorted((PROPOSAL / 'particlegan').glob('*.py'))}
    assert set(base_sources) == set(proposed_sources) and len(base_sources) == 29
    assert [k for k in base_sources if base_sources[k] != proposed_sources[k]] == ['feature_cells.py']
    old = (BASE / 'particlegan/feature_cells.py').read_text()
    new = (PROPOSAL / 'particlegan/feature_cells.py').read_text()
    old_method, new_method = method_text(old), method_text(new)
    assert new.count(new_method) == 1 and new.replace(new_method, old_method, 1) == old
    cases = []
    global_rng = torch.get_rng_state().clone()
    for k in (1, 6, 64):
        for g in sorted({1, min(3,k), min(25,k), k}):
            groups = torch.arange(k) % g
            obj = SimpleNamespace(_mass_topology=lambda:groups, mass_groups=g)
            for dtype in (torch.int32, torch.int64, torch.float32, torch.float64, torch.bool, torch.int16):
                values = [torch.zeros(k,dtype=dtype), (torch.arange(k)%3).to(dtype),
                    (torch.arange(2*k)[::2]+20000).to(dtype)]
                if dtype in (torch.int32, torch.int64):
                    values += [torch.full((k,),torch.iinfo(dtype).max,dtype=dtype),
                        torch.full((k,),torch.iinfo(dtype).min,dtype=dtype)]
                if dtype in (torch.float32, torch.float64):
                    values += [torch.full((k,),float('nan'),dtype=dtype)]
                for value in values:
                    left = old_module.FeatureCellSnapshot._group_counts(obj,value)
                    right = new_module.FeatureCellSnapshot._group_counts(obj,value)
                    assert bits(left,right),(k,g,dtype,value)
                    cases.append(dict(cells=k,groups=g,dtype=str(dtype),exact=True))
            # Original failure types/messages and multidimensional fallback semantics remain exact.
            for value in (torch.tensor(1), torch.zeros(k+1,dtype=torch.long),
                    torch.arange(k*2).reshape(k,2), torch.zeros(1,k,dtype=torch.long)):
                outcomes = []
                for module in (old_module,new_module):
                    try:
                        outcomes.append(('value',module.FeatureCellSnapshot._group_counts(obj,value)))
                    except Exception as error:
                        outcomes.append(('error',(type(error).__name__,str(error))))
                assert outcomes[0][0] == outcomes[1][0]
                assert bits(outcomes[0][1],outcomes[1][1]) if outcomes[0][0]=='value' else outcomes[0][1]==outcomes[1][1]
                cases.append(dict(cells=k,groups=g,shape=list(value.shape),exact=True))
    inputs = torch.load(HERE / 'inputs.pt',map_location='cpu',weights_only=False)
    reactions = []
    for case in inputs['cases']:
        reference_actions, reference_final = planner(old_module,old_birth,case)
        proposed_actions, proposed_final = planner(new_module,new_birth,case)
        assert nested_equal(reference_actions,proposed_actions)
        assert nested_equal(reference_final,proposed_final)
        reactions.append(dict(step=case['step'], ordinary_copies=len(reference_actions['child']),
            novel_births=reference_actions['ordinary']['novel_birth_moves'], isolation_copies=len(reference_actions['isolation_child']),
            complete_plans_certificates_ledgers_work_RNG_exact=True, live_ema_moments_history_evidence_graph_RNG_exact=True))
    assert torch.equal(global_rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
    assert base_sources == {p.name:sha(p) for p in sorted((BASE/'particlegan').glob('*.py'))}
    assert proposed_sources == {p.name:sha(p) for p in sorted((PROPOSAL/'particlegan').glob('*.py'))}
    record = dict(status='PASS', scalar_group_cases=len(cases), cases=cases, saved_reactions=reactions,
        full_inverse_feature_cells_bytes_exact=True, changed_files=['feature_cells.py'],
        base_source_sha256=base_sources, proposal_source_sha256=proposed_sources,
        optimizer_updates=0, new_seeds=0, global_rng_unchanged=True, cuda_initialized=False, quality_verdict=None)
    target = HERE / 'cpu-group-contract.json'
    assert not target.exists()
    target.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(status='PASS', scalar_group_cases=len(cases), saved_reactions=reactions)))


if __name__ == '__main__':
    main()
