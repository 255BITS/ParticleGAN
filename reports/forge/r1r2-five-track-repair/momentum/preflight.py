"""Read-only admission and constructor ownership check; no training or sampling."""
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
import json
import torch
from experiments.forge.planning import resolve_idea
from experiments.forge.behavior_adapters import BehaviorComponents
from experiments.forge.word_adapter import word_context
from experiments.forge.views import load_tasks
from benchmarks.locked_shared.two_pole import HostCritic
from benchmarks.legacy.locked_shared import LOCKED_SHARED
from benchmarks.toy_audit.api_images import WordFixture

root = Path.cwd()
identifier = 'r1r2-five-track-momentum-v1'
request = resolve_idea(root, identifier, study=identifier,
    queue_root=Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-five-track-repair-20261012/queue'))
spec = spec_from_file_location('r1r2workflow', root / 'reports/forge/r1r2-tier1-search/workflow.py')
workflow = module_from_spec(spec)
spec.loader.exec_module(workflow)
projection = workflow.request_projection(request)
original = json.loads((root / 'configs/forge/configurations/r1r2--169e9539b2dcfaab6b51e17f02907f28ea5b894f77155f7e688ab07a789de2b9.json').read_text())
actual = request['candidate']['recipe_overrides']
assert [key for key, value in original['recipe_overrides'].items() if actual[key] != value] == ['betas']
proof = []
before = torch.get_rng_state().clone()
with torch.random.fork_rng(devices=[]):
    tasks = load_tasks(root)
    components = BehaviorComponents(request, tasks['two_pole'], device='cpu')
    particles = torch.nn.Parameter(torch.zeros(LOCKED_SHARED.n_particles, 1))
    critic = HostCritic()
    opt_g, opt_d, _, _ = components.bind(generator=None, critic=critic, opt_g=None,
        opt_d=None, direct_particles=[particles])
    bundles = [('two_pole_direct_generated_coordinates', opt_g), ('two_pole_critic', opt_d)]
    context = word_context(request, tasks['five_word_joint_smoke'], 'cpu', root=root)
    fixture = WordFixture(device='cpu', seed=0, recipe_name=None,
        max_steps=tasks['five_word_joint_smoke']['execution']['steps'], components=context)
    for role, group in zip(('word_G', 'word_E', 'word_prior'), fixture.opt_g.param_groups):
        assert tuple(group['betas']) == (.5, .99)
        proof.append(dict(role=role, betas=list(group['betas']), lr=group['lr'],
            amsgrad=group['amsgrad'], optimizer_class=type(fixture.opt_g).__name__))
    bundles.append(('word_D', fixture.opt_d))
    for role, optimizer in bundles:
        for group in optimizer.param_groups:
            assert tuple(group['betas']) == (.5, .99)
            proof.append(dict(role=role, betas=list(group['betas']), lr=group['lr'],
                amsgrad=group['amsgrad'], optimizer_class=type(optimizer).__name__,
                owned_parameter_shapes=[list(parameter.shape) for parameter in group['params']]))
    assert not fixture.opt_g.state and not fixture.opt_d.state and not opt_d.state
assert torch.equal(before, torch.get_rng_state())
print(json.dumps(dict(schema_version=1, candidate=identifier, study=identifier, status='READY',
    preflight_blockers=request['preflight_blockers'], projection=projection,
    public_constructor_ownership=proof, training_updates=0, training_or_evaluation_sampling_draws=0,
    caller_cpu_rng_restored=True,
    constructor_cohort='explicit isolated ownership diagnostic; no scientific task execution or qualification',
    direct_particle_betas_inactive_reason='Native Adam returns before the formulation direct-particle response; Recipe.betas owns the direct-coordinate group.'), indent=2, sort_keys=True))
