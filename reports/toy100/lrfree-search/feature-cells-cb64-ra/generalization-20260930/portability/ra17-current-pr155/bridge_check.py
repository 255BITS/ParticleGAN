"""RA16/current-PR155 valid-law bridge on fixed CPU inputs."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parents[1]
OLD = STUDY / 'pkg-RA16-portability'
NEW = STUDY / 'pkg-RA17-current-pr155'
REPO = Path('/ml2/hypergan/ParticleGAN-ra11-pr155')
FIXTURE = STUDY / 'integration-prep/ra16-portability/tests/fixtures/feature-auto-base.json'
BASE = json.loads(FIXTURE.read_text())
SEED = 1234


def package(path, name):
    spec = importlib.util.spec_from_file_location(name, path / 'particlegan/__init__.py',
                                                 submodule_search_locations=[str(path / 'particlegan')])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.device == b.device and a.dtype == b.dtype
        assert torch.allclose(a, b, rtol=0, atol=0, equal_nan=True)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):equal(x, y)
    elif isinstance(a, float) and math.isnan(a):assert math.isnan(b)
    else:assert a == b, (a, b)


def make(api, width=2):
    torch.manual_seed(SEED)
    G = torch.nn.Sequential(torch.nn.Linear(2, 8, device='cpu'), torch.nn.Tanh(),
                            torch.nn.Linear(8, width, device='cpu'))
    D = torch.nn.Sequential(torch.nn.Linear(width, 8, device='cpu'), torch.nn.Tanh(),
                            torch.nn.Linear(8, 1, device='cpu'))
    prior = api.ParticlePrior(1024, 2, device='cpu',
                             generator=torch.Generator(device='cpu').manual_seed(SEED + 1))
    recipe = api.Recipe(**dict(BASE, z_dim=2, num_particles=1024, batch_size=128))
    return api.GANTrainer(recipe, G, D, prior=prior, seed=SEED, serial_backward=True)


assert not torch.cuda.is_initialized()
torch.set_num_threads(1)
torch.set_default_device('cpu')
torch.use_deterministic_algorithms(True)
old, new = package(OLD, 'ra16_reference'), package(NEW, 'ra17_combined')
for alias in ('ra16_reference', 'ra17_combined'):
    __import__(alias + '.feature_cells')
    sys.modules[alias + '.feature_cells'].time = SimpleNamespace(perf_counter=lambda: 0.)
sources = sorted((NEW / 'particlegan').glob('*.py'))
assert len(sources) == 28
assert all(p.read_bytes() == (REPO / 'particlegan' / p.name).read_bytes() for p in sources)
changed = [p.name for p in sources if p.read_bytes() != (OLD / 'particlegan' / p.name).read_bytes()]
assert changed == ['continuous.py', 'output_moments.py', 'policy.py', 'routing.py']
assert (NEW / 'particlegan/output_moments.py').read_bytes() + b'\n' == (OLD / 'particlegan/output_moments.py').read_bytes()

traces = []
for width, steps in ((2, 18), (9, 2)):
    batch = torch.arange(128 * width, dtype=torch.float32, device='cpu').reshape(128, width).sin()
    reference = make(old, width)
    states, losses = [], []
    for step in range(steps):
        losses.append(reference.step(batch + step * .001))
        states.append(reference.state_dict())
    reference_sample = reference.sample(64, output_noise=True,
        generator=torch.Generator(device='cpu').manual_seed(SEED + 100))
    candidate = make(new, width)
    for step in range(steps):
        equal(losses[step], candidate.step(batch + step * .001))
        equal(states[step], candidate.state_dict())
    equal(reference_sample, candidate.sample(64, output_noise=True,
        generator=torch.Generator(device='cpu').manual_seed(SEED + 100)))
    saved = reference.state_dict()
    restored = make(new, width)
    restored.load_state_dict(saved)
    equal(saved, restored.state_dict())
    equal(candidate.step(batch + steps * .001), restored.step(batch + steps * .001))
    equal(candidate.state_dict(), restored.state_dict())
    back = make(old, width)
    back.load_state_dict(candidate.state_dict())
    equal(candidate.state_dict(), back.state_dict())
    traces.append(dict(width=width, updates=steps,
        actual_backend=saved['backend_selection']['actual_backend'],
        feature_reactions=getattr(candidate.birth_death, 'snapshot_serial', None),
        mean_forward_rows=candidate.birth_death.counters.get('mean_forward_rows'),
        exact_losses_and_all_checkpoint_state_every_step=True, exact_samples=True,
        bidirectional_valid_checkpoint_roundtrip=True, resumed_next_update_exact=True))
assert traces[0]['feature_reactions'] >= 2 and traces[0]['mean_forward_rows'] > 0

modules = {}
for alias in ('ra16_reference', 'ra17_combined'):
    modules[alias] = dict(controller=__import__(alias + '.continuous', fromlist=['DataDriftController']),
                         routing=__import__(alias + '.routing', fromlist=['RoutedExecution']))
lazy_cases = []
for dtype in (torch.float32, torch.float64):
    for autocast in (False, True):
        table = torch.arange(32 * 3, dtype=dtype, device='cpu').reshape(32, 3).sin()
        latent = table[:7].clone().requires_grad_()
        controllers = [modules[a]['controller'].DataDriftController('dv12') for a in modules]
        streams = [torch.Generator(device='cpu').manual_seed(SEED) for _ in controllers]
        prior = SimpleNamespace(z=table)
        outputs = [[], []]
        for controller in controllers:controller.observe_prior(prior)
        for index in range(3):
            for which, (controller, stream) in enumerate(zip(controllers, streams)):
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=autocast):
                    outputs[which].append(controller.perturb_latent(latent + index * .001, stream,
                                                                   prior=prior, record=True))
        assert len(controllers[1]._latent_application_records) == 2
        assert all(isinstance(x, modules['ra17_combined']['controller']._LatentApplication)
                   for x in controllers[1]._latent_application_records)
        equal(outputs[0], outputs[1])
        grad_a = torch.autograd.grad(sum(x.square().sum() for x in outputs[0]), latent)[0]
        grad_b = torch.autograd.grad(sum(x.square().sum() for x in outputs[1]), latent)[0]
        equal(grad_a, grad_b)
        equal(streams[0].get_state(), streams[1].get_state())
        public_a, public_b = controllers[0].state_dict(), controllers[1].state_dict()
        equal(public_a, public_b)
        assert 'latent_applications' in public_b and '_latent_application_records' not in public_b
        assert all(isinstance(x, dict) and all(type(v) is float for v in x.values())
                   for x in controllers[1].latent_applications)
        equal(controllers[0].diagnostics(), controllers[1].diagnostics())
        repeated = controllers[1].latent_applications
        assert repeated is controllers[1].latent_applications
        controllers[1].load_state_dict(public_a)
        equal(public_a, controllers[1].state_dict())
        controllers[0].load_state_dict(public_b)
        equal(public_b, controllers[0].state_dict())
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=autocast):
            a = controllers[0].perturb_latent(latent, streams[0], prior=prior, record=True)
            b = controllers[1].perturb_latent(latent, streams[1], prior=prior, record=True)
        equal(a, b)
        equal(controllers[0].state_dict(), controllers[1].state_dict())
        lazy_cases.append(dict(dtype=str(dtype), autocast_bfloat16=autocast,
            exact_outputs_gradients_and_private_stream=True, last_two_detached_records_retained=True,
            exact_public_float_diagnostics_and_serialized_state=True,
            public_checkpoint_key_preserved=True, valid_load_both_directions=True))


def routing_run(module, dtype):
    table = (torch.arange(8 * 3, dtype=dtype, device='cpu').reshape(8, 3).cos() / 10).requires_grad_()
    masses = torch.tensor([0., .1, -.2, .3, -.1, .2, -math.inf, -math.inf],
                          dtype=dtype, device='cpu', requires_grad=True)
    candidate = module.RoutedCandidate(table, masses, {})
    sites = tuple(f'site{i}' for i in range(71))
    execution = module.RoutedExecution(sites, candidate, 2, lambda x: x * .97)
    outputs, logits_used = [], []
    previous = torch.zeros(2, 1, 1, dtype=dtype, device='cpu')
    for i, site in enumerate(sites):
        tokens = i % 3 + 1
        base = torch.arange(2 * tokens * 8, dtype=dtype, device='cpu').reshape(2, tokens, 8).sin()
        logits = (base + previous * .01).detach().requires_grad_()
        # A learned downstream term keeps each full forward dependent on the
        # preceding mixed code, while each site's logits remain measurable.
        code = execution.mix(site, logits + previous * .01)
        logits_used.append(logits)
        outputs.append(code)
        previous = code.mean((1, 2), keepdim=True)
    usage = execution.finish()
    objective = sum(x.square().mean() for x in outputs) + .03 * usage.square().sum()
    gradients = torch.autograd.grad(objective, (table, masses, *logits_used))
    assert all(bool(torch.isfinite(x).all()) for x in gradients)
    execution.close()
    context = torch.arange(2 * 3, dtype=dtype, device='cpu').reshape(2, 3).sin()
    spec = module.RoutedRows(route=lambda models, c, bank: (c @ bank.table.T + bank.log_mass).softmax(-1),
        generate=lambda *args: None, features=lambda *args: None)
    weights = spec.weights_for({}, context, candidate)
    return outputs, usage, gradients, weights


routing_cases = []
for dtype in (torch.float32, torch.float64):
    a = routing_run(modules['ra16_reference']['routing'], dtype)
    b = routing_run(modules['ra17_combined']['routing'], dtype)
    equal(a, b)
    routing_cases.append(dict(dtype=str(dtype), sites=71, correlated_tokens_per_site=[1, 2, 3],
        inactive_mass_rows=2, exact_codes_usage_table_mass_logit_gradients=True,
        exact_functional_callback_weights=True, complete_full_forward=True))

proof = json.loads((STUDY / 'diagnostics/upstream-noise-floor-applicability/receipt.json').read_text())
assert proof['original_quality_tasks_proven_unaffected'] == 19
assert proof['learned_fixture_prefixes_proven_unaffected'] == 2
assert proof['tasks_requiring_fresh_training_for_noise_floor_change'] == []
assert not torch.cuda.is_initialized()
result = dict(status='CPU_PASS_LATEST_COMBINED_SOURCE_GPU_GATES_PENDING', seed=SEED,
    no_seed_search=True, candidate_byte_exact_current_repository=True, changed_modules=changed,
    EOF_trim_is_exactly_one_newline=True,
    training_forward_law=traces, lazy_DV12_diagnostics=lazy_cases, valid_many_site_routing=routing_cases,
    original_21_fixture_noise_floor_proof_sha256=hashlib.sha256(
        (STUDY / 'diagnostics/upstream-noise-floor-applicability/receipt.json').read_bytes()).hexdigest(),
    original_19_quality_and_2_learned_fixture_noise_math_unchanged=True,
    original_40_replay_ranges_inside_proved_prefix=True,
    history_relabelled=False, CUDA_initialized=False, actual_GPU_gate_executed=False)
(ROOT / 'CPU-BRIDGE.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps(result, sort_keys=True), flush=True)
