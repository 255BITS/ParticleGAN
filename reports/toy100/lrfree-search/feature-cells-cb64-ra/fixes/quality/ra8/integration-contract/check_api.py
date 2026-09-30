"""Actual composed trainer API/state/serving CPU contracts; no CUDA."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)

AREA = Path(__file__).resolve().parent
ROOT = AREA.parents[2]
PACKAGE = ROOT / 'pkg-CB64-RA8'
BASE = ROOT / 'pkg-CB64-RA7'
OWNER = ROOT / 'performance/sampler-regression/cpu-plan-review/post-ra7-quality/paired-average'
HARNESS = Path('/ml2/hypergan/lrfree-20260926/harness/screen.py')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fingerprint(value):
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            h.update(f'tensor:{v.dtype}:{tuple(v.shape)}:{v.device}:'.encode())
            h.update(v.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            h.update(b'dict')
            for k in sorted(v, key=repr): visit(k); visit(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(type(v).__name__.encode())
            for child in v: visit(child)
        else: h.update(f'{type(v).__name__}:{v!r}'.encode())
    visit(value)
    return h.hexdigest()


def load_package(path, name):
    spec = importlib.util.spec_from_file_location(name, path / 'particlegan/__init__.py',
        submodule_search_locations=[str(path / 'particlegan')])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return package


def linear(weight, bias):
    module = nn.Linear.__new__(nn.Linear); nn.Module.__init__(module)
    module.in_features, module.out_features = weight.shape[1], weight.shape[0]
    module.weight = nn.Parameter(weight.clone()); module.bias = nn.Parameter(bias.clone())
    return module


def network(weights):
    return nn.Sequential(linear(weights['0.weight'], weights['0.bias']), nn.LeakyReLU(.2),
        linear(weights['2.weight'], weights['2.bias']), nn.LeakyReLU(.2),
        linear(weights['4.weight'], weights['4.bias']))


def construct(package, saved, *, particles=None):
    values = deepcopy(saved['recipe'])
    if particles is not None: values.update(num_particles=particles, row_evidence_gate=False)
    recipe = package.Recipe(**values)
    prior = package.ParticlePrior.__new__(package.ParticlePrior); nn.Module.__init__(prior)
    prior.z = nn.Parameter(saved['models']['prior']['z'][:recipe.num_particles].clone())
    # Native constructor, existing declared seed only; no randomized model/prior
    # initialization and no global RNG consumption from fixture construction.
    with torch.random.fork_rng(devices=[]):
        trainer = package.GANTrainer(recipe, network(saved['models']['G']), network(saved['models']['D']),
            prior=prior, seed=314159, optimizer_options=deepcopy(saved['optimizer_options']),
            penalty_options=deepcopy(saved['penalty_options']), serial_backward=saved.get('serial_backward', False))
    return trainer


def copied_cpu_streams(saved):
    return {name: saved['cpu_rng'].clone() for name in saved['streams']}


def cpu_fixture(package, saved, *, measured_stamp=False):
    trainer = construct(package, saved)
    fixture = deepcopy(saved)
    fixture.update(device='cpu', cuda_rng=None, streams=copied_cpu_streams(saved))
    fixture['birth_death']['stream'] = saved['cpu_rng'].clone()
    if measured_stamp:
        # Build the new semantic measurement from current copied tensors,
        # rather than treating an old backend6 stamp as a new-law checkpoint.
        for name, values in fixture['models'].items(): getattr(trainer, name).load_state_dict(values)
        bd = trainer.birth_death
        bd.sample_shape = tuple(fixture['birth_death']['sample_shape'])
        bd.snapshot_serial = fixture['birth_death']['snapshot_serial']
        bd.fill = bd.N; bd.rows_since_eval = 0
        bd.stream.set_state(saved['cpu_rng'])
        modes = [(m, m.training) for root in (trainer.G, trainer.D) for m in root.modules()]
        try:
            trainer.G.eval(); trainer.D.eval()
            q = bd._capture_generated(trainer, trainer.prior.z.detach())
            real = bd._features(trainer, fixture['birth_death']['reservoir'], chunk=bd.settings['chunk'])
            snapshot = package.FeatureCellSnapshot.fit(real,
                generator=torch.Generator().set_state(saved['cpu_rng']),
                cells=bd.settings['cells'], rank=bd.settings['rank'], chunk=bd.settings['chunk'])
            snapshot.cache_queries(q)
            trainer.completed_steps = saved['completed_steps'] - 1
            bd._record_paired_average(trainer, snapshot)
        finally:
            for m, mode in modes: m.training = mode
        fixture['birth_death'].update(backend_schema=7, settings=dict(bd.settings),
            paired_average=dict(bd.paired_average))
        fixture['birth_death']['last'].update(paired_average=dict(bd.paired_average),
            paired_average_forward_rows=bd.N)
        trainer.completed_steps = saved['completed_steps']
    trainer.load_state_dict(fixture)
    return trainer, fixture


def all_streams(trainer):
    return {name: getattr(trainer, name).get_state().clone() for name in trainer._STREAMS} | {
        'birth_death': trainer.birth_death.stream.get_state().clone()}


def served_view(trainer):
    return dict(state=trainer.state_dict(), fast=deepcopy(trainer._fast),
        served_G=deepcopy(trainer.G.state_dict()), served_prior=deepcopy(trainer.prior.state_dict()),
        grads=[[None if p.grad is None else p.grad.clone() for p in model.parameters()]
            for model in (trainer.G, trainer.D, trainer.prior, trainer.ema_G, trainer.ema_prior)],
        modes=[m.training for model in (trainer.G, trainer.D, trainer.prior, trainer.ema_G, trainer.ema_prior)
            for m in model.modules()], cpu_rng=torch.get_rng_state().clone(), streams=all_streams(trainer))


def checkpoint_served_view(trainer):
    result = served_view(trainer)
    # The unchanged public checkpoint does not serialize transient .grad or
    # mode flags. The next _step establishes its required modes and gradients;
    # compare those after the actual resumed update, not immediately on load.
    result.pop('grads'); result.pop('modes')
    return result


def reject_atomic(trainer, state, label):
    before = fingerprint(served_view(trainer))
    try: trainer.load_state_dict(state)
    except ValueError as error: message = str(error)
    else: raise AssertionError(f'{label}: accepted invalid checkpoint')
    assert before == fingerprint(served_view(trainer)), label
    return dict(case=label, rejected=True, persistent_state_rng_and_served_view_exact=True, reason=message)


def stamp_counts(state, coherent):
    result = deepcopy(state)
    bd = result['birth_death']; stamp = bd['paired_average']
    stamp.update(finite_rows=stamp['rows'], same_group_rows=stamp['rows'],
        ema_eligible_rows=coherent, coherent_rows=coherent,
        chart_valid=True, duplicate_ok=True, eligible=coherent >= stamp['required'])
    bd['last']['paired_average'] = dict(stamp)
    return result


def comparable(state):
    result = deepcopy(state)
    bd = result['birth_death']
    bd.pop('paired_average', None)
    bd['backend_schema'] = 6
    for name in ('paired_average_policy', 'paired_average_expiry'): bd['settings'].pop(name, None)
    for name in ('paired_average', 'paired_average_forward_rows'): bd['last'].pop(name, None)
    return result


def extra_pass_control(trainer, package, saved):
    trainer._serve_release()
    bd = trainer.birth_death
    modes = [(m, m.training) for root in (trainer.G, trainer.D, trainer.ema_G) for m in root.modules()]
    try:
        trainer.G.eval(); trainer.D.eval()
        q = bd._capture_generated(trainer, trainer.prior.z.detach())
        real = bd._features(trainer, saved['birth_death']['reservoir'], chunk=bd.settings['chunk'])
        snapshot = package.FeatureCellSnapshot.fit(real,
            generator=torch.Generator().set_state(saved['cpu_rng']), cells=bd.settings['cells'],
            rank=bd.settings['rank'], chunk=bd.settings['chunk'])
        snapshot.cache_queries(q)
    finally:
        for module, mode in modes: module.training = mode
    observed = (trainer.ema_G, trainer.D)
    for model in observed:
        model.register_buffer('api_marker', torch.arange(3).float())
        model.register_buffer('api_nonpersistent', torch.ones(2), persistent=False)
        for p in model.parameters(): p.grad = torch.ones_like(p)
    trainer.ema_G.train(); trainer.ema_G[1].eval(); trainer.D.train(); trainer.D[1].eval()
    before_modes = [(m, m.training) for model in observed for m in model.modules()]
    before_buffers = [(m, dict(m._buffers), {k: None if v is None else v.clone() for k, v in m._buffers.items()},
        set(m._non_persistent_buffers_set)) for model in observed for m in model.modules()]
    before_gradients = [(p, p.grad, p.grad.clone()) for model in observed for p in model.parameters()]
    before_parameters = fingerprint([m.state_dict() for m in observed])
    before_rng = torch.get_rng_state().clone(); before_streams = all_streams(trainer)
    def perturb(model, args):
        model.api_marker.add_(1.)
        model._buffers['api_nonpersistent'] = model.api_nonpersistent + 7.
        model.register_buffer('api_temporary', torch.ones(1), persistent=False)
        for p in model.parameters(): p.grad = torch.zeros_like(p)
        torch.rand(1)
        for stream in [*[getattr(trainer, n) for n in trainer._STREAMS], bd.stream]:
            torch.rand(1, generator=stream)
    handles = [model.register_forward_pre_hook(perturb) for model in observed]
    completed = trainer.completed_steps
    trainer.completed_steps = completed - 1  # Actual hook's pre-increment phase.
    try: bd._record_paired_average(trainer, snapshot)
    finally:
        trainer.completed_steps = completed
        for handle in handles: handle.remove()
    assert before_parameters == fingerprint([m.state_dict() for m in observed])
    assert torch.equal(before_rng, torch.get_rng_state()) and fingerprint(before_streams) == fingerprint(all_streams(trainer))
    assert all(m.training == mode for m, mode in before_modes)
    assert all(m._buffers.keys() == mapping.keys() and m._non_persistent_buffers_set == nonpersistent
        and all(m._buffers[k] is v and (v is None or torch.equal(v, values[k])) for k, v in mapping.items())
        for m, mapping, values, nonpersistent in before_buffers)
    assert all(p.grad is original and torch.equal(p.grad, value) for p, original, value in before_gradients)
    for model in observed:
        model._buffers.pop('api_marker'); model._buffers.pop('api_nonpersistent')
        model._non_persistent_buffers_set.discard('api_nonpersistent')
        for p in model.parameters(): p.grad = None
    assert not trainer.ema_G._forward_pre_hooks and not trainer.D._forward_pre_hooks
    trainer._serve_apply()
    return dict(global_cpu_rng=True, all_owned_streams=True, registered_buffer_mappings_identities_values=True,
        nonpersistent_buffer_registration_metadata=True, gradient_objects_values=True,
        individual_module_modes=True, eval_draw_and_increment_replacement_registration_controls=True)


if __name__ == '__main__':
    output = AREA / 'receipt.json'
    if output.exists(): raise SystemExit('Preserve existing receipt; use a new attempt area.')
    composition = json.loads((ROOT / 'quality/ra8/COMPOSITION.json').read_text())
    candidate_map = {str(p.relative_to(PACKAGE / 'particlegan')): sha(p)
        for p in sorted((PACKAGE / 'particlegan').glob('*.py'))}
    assert candidate_map == composition['source_sha256'] and len(candidate_map) == 29
    assert candidate_map == {str(p.relative_to(OWNER / 'pkg-PAIR-AVERAGE/particlegan')): sha(p)
        for p in sorted((OWNER / 'pkg-PAIR-AVERAGE/particlegan').glob('*.py'))}
    assert sha(ROOT / 'configs/overrides-CB64-RA8.json') == sha(ROOT / 'configs/overrides-CB64-RA7.json')
    paths = [ROOT / f'validation-cb64-ra7/learned/training/toy/CB64-RA7/checkpoint-{step:04d}.pt'
        for step in (1000, 2000)]
    source_paths = [*sorted((PACKAGE / 'particlegan').glob('*.py')), *sorted((BASE / 'particlegan').glob('*.py')),
        Path(__file__), AREA / 'PROTOCOL.md', ROOT / 'quality/ra8/COMPOSITION.json', OWNER / 'READY.json',
        ROOT / 'quality/ra7/READY.json', ROOT / 'configs/overrides-CB64-RA8.json', HARNESS]
    before_files = {str(p): sha(p) for p in source_paths + paths}
    initial_rng = torch.get_rng_state().clone()
    package = load_package(PACKAGE, 'actual_ra8_api')
    baseline_package = load_package(BASE, 'actual_ra7_api')
    records, rejected, fixtures = [], [], {}
    for path in paths:
        saved = torch.load(path, map_location='cpu', weights_only=False)['trainer']
        source_state_before = fingerprint(saved)
        rng_before_load = torch.get_rng_state().clone()
        trainer, fixture = cpu_fixture(package, saved, measured_stamp=True)
        torch.set_rng_state(rng_before_load)
        assert fingerprint(saved) == source_state_before
        stamp = dict(trainer.birth_death.paired_average)
        assert trainer.birth_death.snapshot is None and trainer._serve_settled() == stamp['eligible']
        assert (trainer._fast is not None) == stamp['eligible']
        fixtures[saved['completed_steps']] = (saved, fixture)
        records.append(dict(step=saved['completed_steps'], stamp=stamp,
            eligible_dispatch_correct=True, load_without_derived_chart_correct=True,
            config_rates=trainer.initial_lrs, roles=trainer.roles))
    saved, qualified = fixtures[2000]
    subject = construct(package, saved); subject.load_state_dict(qualified)
    assert subject._fast is not None, 'Measured final saved geometry must qualify the tested positive view'
    fast = deepcopy(qualified['models'])
    assert fingerprint(subject.G.state_dict()) == fingerprint(fast['ema_G'])
    assert fingerprint(subject.prior.state_dict()) == fingerprint(fast['ema_prior'])
    checkpoint = subject.state_dict()
    assert fingerprint(checkpoint['models']) == fingerprint(fast) and subject._fast is not None
    independence = deepcopy(checkpoint)
    independence['models']['G']['0.weight'].add_(1.)
    assert fingerprint(subject.G.state_dict()) == fingerprint(fast['ema_G'])
    source_state = subject.state_dict()
    clone = construct(package, saved); clone.load_state_dict(source_state)
    assert fingerprint(served_view(clone)) == fingerprint(served_view(subject))
    subject._serve_release()
    assert fingerprint(subject.G.state_dict()) == fingerprint(fast['G'])
    assert fingerprint(subject.prior.state_dict()) == fingerprint(fast['prior'])
    subject._serve_apply(); assert subject._fast is not None
    # A served swap overwrites the same prior Parameter. The derived sorted
    # geometry must rebuild on its tensor version, with unchanged lineage.
    geometry = subject.birth_death.latent_geometry
    cache_ids = torch.arange(13, dtype=torch.long)
    graph_before = subject.birth_death.lineage.neighbors.clone()
    subject._serve_release()
    fast_geometry = geometry._local_geometry(subject.prior.z.detach()[cache_ids], subject.prior, rows=cache_ids)
    fast_version = subject.prior.z._version
    first_builds = geometry.work['builds']
    subject._serve_apply()
    assert subject.prior.z._version != fast_version
    averaged_geometry = geometry._local_geometry(subject.prior.z.detach()[cache_ids], subject.prior, rows=cache_ids)
    assert geometry.work['builds'] == first_builds + 1
    cold = type(geometry)(rank=geometry.rank, neighbors=geometry.neighbors,
        chunk=geometry.chunk, lineage=subject.birth_death.lineage)
    cold_average = cold._local_geometry(subject.prior.z.detach()[cache_ids], subject.prior, rows=cache_ids)
    assert all(torch.equal(a, b) for a, b in zip(averaged_geometry, cold_average))
    subject._serve_release()
    released_geometry = geometry._local_geometry(subject.prior.z.detach()[cache_ids], subject.prior, rows=cache_ids)
    assert geometry.work['builds'] == first_builds + 2
    assert all(torch.equal(a, b) for a, b in zip(fast_geometry, released_geometry))
    assert torch.equal(graph_before, subject.birth_death.lineage.neighbors)
    subject._serve_apply()
    before_sampling = served_view(subject)
    stream_a = torch.Generator().set_state(source_state['cpu_rng'])
    stream_b = torch.Generator().set_state(source_state['cpu_rng'])
    served = subject.sample(17, generator=stream_a)
    explicit_ema = subject.sample(17, ema=True, generator=stream_b)
    assert torch.equal(served, explicit_ema) and torch.equal(stream_a.get_state(), stream_b.get_state())
    assert fingerprint(before_sampling) == fingerprint(served_view(subject))
    forwarded = []
    perturb = subject.birth_death.perturb_latent
    def traced(latent, stream, controller=None, record=False, *, prior=None, rows=None):
        forwarded.append(None if rows is None else rows.clone())
        return perturb(latent, stream, controller, record, prior=prior, rows=rows)
    subject.birth_death.perturb_latent = traced
    ids = torch.arange(11, dtype=torch.long)
    latent = subject.prior.z.detach()[ids]
    try:
        subject._generate(subject.G, latent, 0., torch.Generator().set_state(source_state['cpu_rng']))
        positional = subject._generate(subject.G, latent, .029, torch.Generator().set_state(source_state['cpu_rng']), ids)
        keyword = subject._generate(subject.G, latent, .029, torch.Generator().set_state(source_state['cpu_rng']), rows=ids)
        subject.sample(11, generator=torch.Generator().set_state(source_state['cpu_rng']))
    finally: subject.birth_death.perturb_latent = perturb
    assert forwarded[0] is None and torch.equal(forwarded[1], ids) and torch.equal(forwarded[2], ids)
    assert forwarded[3] is not None and torch.equal(positional, keyword)
    harness_tree = ast.parse(HARNESS.read_text())
    defaults = next(node for node in harness_tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'DEFAULT_OPTIONS' for target in node.targets))
    resolver = next(node for node in harness_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'resolve_options')
    host = dict(inspect=inspect)
    exec(compile(ast.Module(body=[defaults, resolver], type_ignores=[]), str(HARNESS), 'exec'), host)
    resolved, detected = host['resolve_options'](package, {}, {})
    assert resolved['evaluation_generate'] == detected['evaluation_generate'] == 'indexed'
    assert 'indices' in inspect.signature(package.GANTrainer._generate).parameters
    assert package.GANTrainer._generate.__code__.co_argcount == baseline_package.GANTrainer._generate.__code__.co_argcount
    assert subject.output_sigma() == float(baseline_package.GANTrainer._output_sigma(subject,
        subject.recipe.output_noise_std))
    stamp = subject.birth_death.paired_average
    n = subject.birth_death.N
    subject.birth_death.rows_since_eval = n - 1
    assert subject._serve_settled()
    subject._serve_apply(); assert subject._fast is not None
    subject.birth_death.rows_since_eval = n
    assert not subject._serve_settled()
    subject._serve_apply(); assert subject._fast is None
    assert fingerprint(subject.G.state_dict()) == fingerprint(fast['G'])
    subject.load_state_dict(qualified)
    expired = deepcopy(qualified); expired['birth_death']['rows_since_eval'] = n
    clone.load_state_dict(expired)
    assert clone._fast is None and clone.birth_death.snapshot is None
    assert fingerprint(clone.G.state_dict()) == fingerprint(fast['G'])
    old, old_fixture = cpu_fixture(baseline_package, saved)
    rejected.append(reject_atomic(subject, old_fixture, 'old_backend6'))
    bad = deepcopy(qualified); bad['birth_death']['paired_average']['coherent_rows'] = n + 1
    bad['birth_death']['last']['paired_average'] = deepcopy(bad['birth_death']['paired_average'])
    rejected.append(reject_atomic(subject, bad, 'inconsistent_intersection'))
    future = deepcopy(qualified); future['birth_death']['paired_average']['step'] += 1
    future['birth_death']['last']['step'] += 1
    future['birth_death']['last']['paired_average'] = deepcopy(future['birth_death']['paired_average'])
    rejected.append(reject_atomic(subject, future, 'future_reaction_step'))
    veto = stamp_counts(qualified, stamp['required'] - 1)
    clone.load_state_dict(veto); assert not clone._serve_settled() and clone._fast is None
    exactly = stamp_counts(qualified, stamp['required'])
    clone.load_state_dict(exactly); assert clone._serve_settled() and clone._fast is not None
    neutral = extra_pass_control(subject, package, saved)
    # Actual next-update and save/load continuation, with the fixed positive
    # typed stamp declared solely as an API control on a nonterminal fixture.
    early_saved, early_fixture = fixtures[1000]
    positive = stamp_counts(early_fixture, len(early_saved['models']['prior']['z']))
    continued = construct(package, early_saved); continued.load_state_dict(positive)
    reference, reference_state = cpu_fixture(baseline_package, early_saved)
    assert continued._fast is not None
    batch = early_saved['birth_death']['reservoir'][:128]
    torch.set_rng_state(positive['cpu_rng'])
    first = continued.step(batch)
    torch.set_rng_state(reference_state['cpu_rng'])
    reference_first = reference.step(batch)
    assert all(torch.equal(first[k], reference_first[k]) for k in first if isinstance(first[k], torch.Tensor))
    after_first = continued.state_dict()
    assert fingerprint(comparable(after_first)) == fingerprint(reference.state_dict())
    assert continued._fast is not None and continued.completed_steps == 1001
    resumed = construct(package, early_saved); resumed.load_state_dict(after_first)
    assert fingerprint(checkpoint_served_view(resumed)) == fingerprint(checkpoint_served_view(continued))
    torch.set_rng_state(after_first['cpu_rng']); second = continued.step(batch)
    torch.set_rng_state(after_first['cpu_rng']); resumed_second = resumed.step(batch)
    assert all(torch.equal(second[k], resumed_second[k]) for k in second if isinstance(second[k], torch.Tensor))
    assert fingerprint(served_view(continued)) == fingerprint(served_view(resumed))
    small = construct(package, early_saved, particles=12)
    assert not hasattr(small.birth_death, 'paired_average_eligible')
    small._table_tester().last_decisive = -1
    assert small._serve_settled() and baseline_package.GANTrainer._serve_settled(small)
    small._table_tester().last_decisive = 0
    assert not small._serve_settled() and not baseline_package.GANTrainer._serve_settled(small)
    torch.set_rng_state(initial_rng)
    assert not torch.cuda.is_initialized()
    assert before_files == {str(p): sha(p) for p in source_paths + paths}
    receipt = dict(status='PASS', utc=datetime.now(timezone.utc).isoformat(), records=records,
        composition_sha256=sha(ROOT / 'quality/ra8/COMPOSITION.json'), package_sha256=composition['package_sha256'],
        source_and_input_sha256=before_files, all29_package_files_owner_exact=True, config_bytes_RA7_exact=True,
        controls=dict(swap_release_and_independent_state=True, measured_positive_roundtrip_no_chart=True,
            served_sample_equals_explicit_ema_with_same_noise_and_rng=True, named_positional_ids_and_sample_forwarded=True,
            exact_frozen_native_resolver_indexed=True, served_swap_release_geometry_version_rebuild_and_lineage_exact=True,
            strict_N_minus_one_vs_N_expiry=True, expired_load_fast=True, threshold_minus_one_and_exactly_required=True,
            next_optimizer_update_RA7_exact_after_release=True, served_checkpoint_continuation_exact=True,
            N12_reference_fallback_legacy=True, extra_pass_neutral=neutral), rejected_controls=rejected,
        old_backend6_atomic_rejection=True, all_sources_inputs_checkpoint_tensors_unchanged=True,
        cpu_only=True, cuda_initialized=False, global_cpu_rng_restored=True, new_seed_experiments=0,
        private_CPU_trainer_updates_this_attempt=4, retained_attempt1_private_CPU_updates=2,
        new_copy_birth_reaction_suites=0, new_evaluator_emissions=0,
        quality_verdict=None, historical_GPU_replay=False, production_sources_modified=False,
        limits=['The CPU fixture uses copied saved weights/history/FIFO and CPU streams; it is not cross-device historical replay.',
            'The paired-average stamp is an empirical clean anti-blur lease, not distribution equivalence, stationarity or emitted quality.',
            'The lease is stale for less than one FIFO turnover; D/G can move within it.',
            'Continuation uses a clearly declared typed positive API control on saved1000, not claimed measured eligibility there.',
            'CPU neutral controls cover registered model/gradient state and owned generators, not arbitrary Python/external RNG effects.',
            'The unchanged checkpoint does not serialize transient parameter gradients or module modes; these are compared after the next resumed update.'])
    output.write_text(json.dumps(receipt, indent=2, allow_nan=True) + '\n')
    print(json.dumps(dict(status='PASS', output=str(output), source_count=len(before_files),
        records=len(records), measured_final_stamp=records[-1]['stamp'])), flush=True)
