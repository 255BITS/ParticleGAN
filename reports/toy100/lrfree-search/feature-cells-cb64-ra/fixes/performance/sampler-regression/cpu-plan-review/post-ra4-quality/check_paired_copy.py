"""Paired copy mechanical contracts; GPU execution belongs to root only."""
import argparse
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--package-root', type=Path, default=HERE / 'pkg-PAIR-EMA')
parser.add_argument('--reference-package-root', type=Path, default=ROOT / 'pkg-CB64-RA4')
parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
os.environ.update(CUDA_VISIBLE_DEVICES='' if args.device == 'cpu' else '0',
    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
    PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
sys.dont_write_bytecode = True

import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from types import SimpleNamespace
import torch

sys.path.insert(0, str(args.package_root))
from particlegan.feature_cells import FeatureCellBirthDeath
from particlegan.recipes import Recipe


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            h.update(str((str(v.dtype), tuple(v.shape))).encode())
            h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            for key in sorted(v, key=repr):
                h.update(repr(key).encode()); visit(v[key])
        elif isinstance(v, (tuple, list)):
            h.update(type(v).__name__.encode())
            for x in v: visit(x)
        else:
            h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def source_proof():
    old_path = args.reference_package_root / 'particlegan/feature_cells.py'
    new_path = args.package_root / 'particlegan/feature_cells.py'
    old, new = ast.parse(old_path.read_text()), ast.parse(new_path.read_text())
    old_class = next(x for x in old.body if isinstance(x, ast.ClassDef) and x.name == 'FeatureCellBirthDeath')
    new_class = next(x for x in new.body if isinstance(x, ast.ClassDef) and x.name == old_class.name)
    old_move = next(x for x in old_class.body if isinstance(x, ast.FunctionDef) and x.name == '_move')
    for i, node in enumerate(new_class.body):
        if isinstance(node, ast.FunctionDef) and node.name == '_move': new_class.body[i] = deepcopy(old_move)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'BACKEND_SCHEMA' for t in node.targets):
            assert node.value.value == 5; node.value.value = 4
        if isinstance(node, ast.FunctionDef) and node.name == '__init__':
            extra = [n for n in node.body if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and t.slice.value == 'copy_noise_policy'
                for t in n.targets)]
            assert len(extra) == 1
            assert extra[0].value.value == 'shared_noise_separate_live_ema_current_geometry_v1'
            node.body.remove(extra[0])
    assert ast.dump(new, include_attributes=False) == ast.dump(old, include_attributes=False)
    differing = []
    for p in sorted((args.reference_package_root / 'particlegan').glob('*.py')):
        if sha(p) != sha(args.package_root / 'particlegan' / p.name): differing.append(p.name)
    assert differing == ['feature_cells.py']
    return dict(only_changed_module=differing, exact_ast_changes=['FeatureCellBirthDeath._move',
        'FeatureCellBirthDeath.BACKEND_SCHEMA 4→5', 'FeatureCellBirthDeath.settings.copy_noise_policy'],
        restoring_three_changes_reconstructs_entire_ra4_module=True,
        training_api_sampling_serving_output_noise_count_law_unchanged=True)


def reference():
    spec = importlib.util.spec_from_file_location('particlegan._paired_copy_reference',
        args.reference_package_root / 'particlegan/feature_cells.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.FeatureCellBirthDeath


def make_case(state, cls, seed):
    points = state['models']['prior']['z'].to(args.device)
    ema = state['models']['ema_prior']['z'].to(args.device)
    prior = SimpleNamespace(z=torch.nn.Parameter(points.clone()))
    prior_state = next(v for v in state['optimizers'][0]['state'].values() if any(
        isinstance(t, torch.Tensor) and t.shape == points.shape for t in v.values()))
    moment = {k:v.to(args.device).clone() if isinstance(v, torch.Tensor) else deepcopy(v)
              for k,v in prior_state.items()}
    history = state['optimizers'][0]['regularizer']['latent']['history']
    if history is not None: history = history.to(args.device).clone()
    with torch.random.fork_rng(devices=[]):
        critic = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)).to(args.device)
    options = dict(state['recipe'])
    options.update(birth_death_cells=64, birth_death_metric_rank=8, birth_death_chunk=256,
                   birth_death_parent_policy='real_anchor', birth_death_backend='feature_cells')
    trainer = SimpleNamespace(prior=prior, ema_prior=SimpleNamespace(z=torch.nn.Parameter(ema.clone())),
        device=points.device, dtype=points.dtype, recipe=Recipe(**options), D=critic,
        controller=SimpleNamespace(latent_bandwidth=state['controller']['latent_bandwidth'].to(args.device).clone()),
        opt_g=SimpleNamespace(state={prior.z:moment}, latent_history=history))
    backend = cls(trainer, seed + 6)
    if args.device == 'cpu': backend.stream.set_state(state['cpu_rng'])
    if 'lineage_neighbors' in state['birth_death']:
        backend.lineage.neighbors.copy_(state['birth_death']['lineage_neighbors'].to(args.device))
    backend.lineage.validate(backend.lineage.neighbors)
    return trainer, backend


def semantics(trainer, backend):
    return dict(prior=trainer.prior.z, ema_prior=trainer.ema_prior.z,
        moments=trainer.opt_g.state[trainer.prior.z], history=trainer.opt_g.latent_history,
        backend=backend.state_dict())


def copy_contract(state, seed, ref_class, child, parent, label):
    old_t, old = make_case(state, ref_class, seed)
    new_t, new = make_case(state, FeatureCellBirthDeath, seed)
    child, parent = child.to(args.device), parent.to(args.device)
    prior_before, ema_before = new_t.prior.z.detach().clone(), new_t.ema_prior.z.detach().clone()
    probe = torch.Generator(device=args.device).set_state(new.stream.get_state())
    noise = torch.randn((len(parent), prior_before.shape[1]), device=args.device,
                        dtype=prior_before.dtype, generator=probe)
    live_delta = new.latent_geometry.displacement(prior_before[parent], new_t.prior,
        new_t.controller.latent_bandwidth, noise, rows=parent)
    ema_delta = new.latent_geometry.displacement(ema_before[parent], new_t.ema_prior,
        new_t.controller.latent_bandwidth, noise, rows=parent)
    ema_radius = new.latent_geometry.radius(ema_before[parent], new_t.ema_prior, rows=parent)
    assert bool((ema_delta.norm(dim=1) <= ema_radius * 1.00001 + 1e-12).all())
    old._move(old_t, child, parent); new._move(new_t, child, parent)
    assert torch.equal(old_t.prior.z, new_t.prior.z)
    assert torch.equal(new_t.prior.z.detach()[child], prior_before[parent] + live_delta)
    assert torch.equal(new_t.ema_prior.z.detach()[child], ema_before[parent] + ema_delta)
    assert fingerprint(old_t.opt_g.state[old_t.prior.z]) == fingerprint(new_t.opt_g.state[new_t.prior.z])
    assert fingerprint(old_t.opt_g.latent_history) == fingerprint(new_t.opt_g.latent_history)
    assert torch.equal(old.stream.get_state(), new.stream.get_state())
    assert torch.equal(new.stream.get_state(), probe.get_state())
    assert torch.equal(old.lineage.neighbors, new.lineage.neighbors)
    new.lineage.validate(new.lineage.neighbors)
    assert new._copy_parent_rows is None
    assert new.latent_geometry.work['max_candidates'] <= 72
    # A mutation makes both caches stale; the next query must rebuild them.
    before_builds = new.latent_geometry.work['builds']
    new.latent_geometry.radius(new_t.prior.z[parent], new_t.prior, rows=parent)
    new.latent_geometry.radius(new_t.ema_prior.z[parent], new_t.ema_prior, rows=parent)
    assert new.latent_geometry.work['builds'] == before_builds + 2
    ratio = live_delta.norm(dim=1) / ema_radius.clamp_min(1e-20)
    return dict(case=label, rows=len(parent), fast_z_moments_history_graph_rng_bit_identical=True,
        single_noise_draw_accounting=True, ema_exact_own_geometry_placement=True,
        old_ema_radius_violation_rows=int((ratio > 1.00001).sum()), new_ema_radius_violation_rows=0,
        old_ema_radius_max_ratio=float(ratio.max()), both_versioned_caches_rebuilt=True)


def resume_contract(state, seed, ref_class):
    trainer, backend = make_case(state, FeatureCellBirthDeath, seed)
    child, parent = torch.arange(32, device=args.device), torch.arange(128, 160, device=args.device)
    backend._move(trainer, child, parent)
    saved = deepcopy(backend.state_dict())
    resumed_t = deepcopy(trainer)
    resumed = FeatureCellBirthDeath(resumed_t, seed + 6)
    resumed.load_state_dict(saved)
    assert not resumed.latent_geometry._entries
    assert fingerprint(saved) == fingerprint(resumed.state_dict())
    child, parent = torch.arange(32, 64, device=args.device), torch.arange(160, 192, device=args.device)
    backend._move(trainer, child, parent); resumed._move(resumed_t, child, parent)
    assert fingerprint(semantics(trainer, backend)) == fingerprint(semantics(resumed_t, resumed))
    old_t, old = make_case(state, ref_class, seed)
    incompatible = old.state_dict()
    before = fingerprint(semantics(resumed_t, resumed))
    for incompatible_state in (incompatible, dict(incompatible, backend_schema=5)):
        try: resumed.load_state_dict(incompatible_state)
        except ValueError: pass
        else: raise AssertionError('old copy law accepted')
        assert before == fingerprint(semantics(resumed_t, resumed))
    return dict(new_backend_schema=5, old_schema_and_forged_schema_missing_policy_rejected_before_mutation=True,
        post_copy_save_load_and_two_copy_continuation_bit_identical=True,
        lineage_exact_and_derived_cache_discarded=True)


def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.device == 'cuda':
        torch.cuda.set_device(0); torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-').lower() == '72c1b506-891d-b8bc-b353-e020585e1c47'
        torch.cuda.set_per_process_memory_fraction(.2, 0)
    ref_class = reference()
    fixtures = [(ROOT / f'validation-ra4/learned/training/toy/CB64-RA4/checkpoint-{step:04d}.pt',
                 314159, f'ra4-toy-{step}') for step in (1250, 2000)]
    fixtures.append((ROOT / 'validation/screens/runs/grid100/final-state.pt', 1234, 'ra2-grid100-final'))
    sources = [Path(__file__), HERE / 'diagnose_sampler.py',
        args.package_root / 'particlegan/feature_cells.py', args.reference_package_root / 'particlegan/feature_cells.py'] + [p for p, _, _ in fixtures]
    hashes = {str(p):sha(p) for p in sources}
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='PASS', device=args.device,
        scope='saved-input mechanical contracts; possible copy parents, not historical actions or quality reruns',
        new_seeds=0, optimizer_updates=0, source_sha256=hashes, source_proof=source_proof(), cases=[])
    for path, seed, label in fixtures:
        value = torch.load(path, map_location='cpu', weights_only=False)
        state = value.get('trainer', value)
        patterns = [(torch.arange(128), torch.arange(256, 384), 'unique'),
                    (torch.arange(10, 19), torch.full((9,), 256, dtype=torch.long), 'repeated-parent'),
                    (torch.tensor([20, 21]), torch.tensor([21, 22]), 'simultaneous-parent-overwrite')]
        for child, parent, pattern in patterns:
            row = copy_contract(state, seed, ref_class, child, parent, label + '/' + pattern)
            result['cases'].append(row); print(json.dumps(row), flush=True)
        if label == 'ra4-toy-2000': result['resume'] = resume_contract(state, seed, ref_class)
    result['sources_unchanged'] = hashes == {str(p):sha(p) for p in sources}
    result['cuda_initialized'] = torch.cuda.is_initialized()
    assert result['sources_unchanged']
    assert result['cuda_initialized'] == (args.device == 'cuda')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.exists(), 'keep prior receipts'
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='PASS', cases=len(result['cases']), output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
