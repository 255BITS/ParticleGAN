"""CPU-only integration of a declared rank-capped chart law, on saved states."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
OWNER = ROOT / 'performance/sampler-regression/cpu-plan-review/post-ra8-quality/resolution'
BASE = ROOT / 'pkg-CB64-RA8'
API = ROOT / 'quality/ra8/integration-contract/check_api.py'
HARNESS = Path('/ml2/hypergan/lrfree-20260926/harness/screen.py')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def guard_map(mapping):
    for name, expected in mapping.items():
        assert sha(Path(name)) == expected, name


def ast_method(tree, owner, name):
    parent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    return ast.dump(next(node for node in parent.body if isinstance(node, ast.FunctionDef) and node.name == name),
                    include_attributes=False)


def adapted_fixture(saved):
    # This private fixture declares the new recipe/law. It is not a production
    # checkpoint converter; loading the original old law is tested to reject.
    result = deepcopy(saved)
    result['recipe']['birth_death_cells'] = 128
    bd = result['birth_death']
    bd['backend_schema'] = 8
    bd['settings'].update(cells=128,
        resolution_policy='even_fit_average_rows_per_effective_rank_floor1_v1')
    return result


def checkpoint_roundtrip(api, package, fixture, trainer):
    saved = trainer.state_dict()
    clone = api.construct(package, fixture)
    clone.load_state_dict(saved)
    assert api.fingerprint(api.checkpoint_served_view(clone)) == api.fingerprint(api.checkpoint_served_view(trainer))
    assert clone.birth_death.snapshot is None
    assert not clone.birth_death.latent_geometry._entries
    return clone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--composition', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'receipt.json').exists(), 'Preserve prior attempts'
    assert sha(API) == '36ab993ffcd9408ae84f9c35a1d35c41cb6d40176e42f54121878d416cddac61'
    owner_seal = json.loads((OWNER / 'SOURCE-FROZEN.json').read_text())
    guard_map(owner_seal['source_and_input_sha256'])
    composition = json.loads(args.composition.read_text())
    sources = {str(p.relative_to(args.package_root / 'particlegan')): sha(p)
        for p in sorted((args.package_root / 'particlegan').glob('*.py'))}
    owner_sources = {str(p.relative_to(OWNER / 'pkg-RESOLUTION/particlegan')): sha(p)
        for p in sorted((OWNER / 'pkg-RESOLUTION/particlegan').glob('*.py'))}
    assert sources == owner_sources == composition['source_sha256'] and len(sources) == 29
    assert sources['feature_cells.py'] == '39558fb3839090eb9d933b8b37e4c73b7dfc3cf123b24c2ef58d818cc4052fcc'
    old_sources = {str(p.relative_to(BASE / 'particlegan')): sha(p)
        for p in sorted((BASE / 'particlegan').glob('*.py'))}
    assert [name for name in sources if sources[name] != old_sources[name]] == ['feature_cells.py']
    config, base_config = json.loads(args.config.read_text()), json.loads((ROOT / 'configs/overrides-CB64-RA8.json').read_text())
    assert {name:(base_config.get(name),value) for name,value in config.items()
        if value != base_config.get(name)} == {'birth_death_cells':(64,128)}
    assert set(config) == set(base_config)
    assert sha(args.config) == composition['config_sha256']
    old_tree = ast.parse((BASE / 'particlegan/feature_cells.py').read_text())
    new_tree = ast.parse((args.package_root / 'particlegan/feature_cells.py').read_text())
    unchanged = []
    for owner, names in {
        'FeatureCellBirthDeath':('_record_paired_average','paired_average_eligible','_move','perturb_latent',
                                 'maybe_apply','load_state_dict'),
        'BoundedLatentGeometry':('_local_geometry','displacement'),
        'FeatureCellSnapshot':('ordinary_transport','select_parents','cell_comparison')}.items():
        for name in names:
            assert ast_method(old_tree,owner,name) == ast_method(new_tree,owner,name), (owner,name)
            unchanged.append(f'{owner}.{name}')
    for name in ('paired_average_geometry','population_policy'):
        left=next(node for node in old_tree.body if isinstance(node,ast.FunctionDef) and node.name==name)
        right=next(node for node in new_tree.body if isinstance(node,ast.FunctionDef) and node.name==name)
        assert ast.dump(left,include_attributes=False)==ast.dump(right,include_attributes=False),name
        unchanged.append(name)
    paths = [ROOT / f'validation-cb64-ra8/learned/training/toy/CB64-RA8/checkpoint-{step:04d}.pt'
        for step in (1000,2000)]
    protected_paths = [*sorted((args.package_root / 'particlegan').glob('*.py')),
        *sorted((BASE / 'particlegan').glob('*.py')), API, HARNESS, Path(__file__), HERE / 'PROTOCOL.md',
        args.config,args.composition,OWNER / 'SOURCE-FROZEN.json',OWNER / 'READY.json',
        ROOT / 'quality/ra8/READY.json',ROOT / 'quality/ra8/integration-contract/FROZEN.json',*paths]
    protected = {str(p):sha(p) for p in protected_paths}
    # Import an already frozen fixture utility, including its one-time CPU
    # thread setup. No quality runner, scorer, CUDA or randomized model init.
    spec = importlib.util.spec_from_file_location('frozen_ra8_api_util',API)
    api = importlib.util.module_from_spec(spec); spec.loader.exec_module(api)
    torch = api.torch
    initial_rng = torch.get_rng_state().clone()
    package = api.load_package(args.package_root,'rank_cap_actual_api')
    baseline = api.load_package(BASE,'rank_cap_baseline_api')
    fixtures,records = {},[]
    for path in paths:
        original = torch.load(path,map_location='cpu',weights_only=False)['trainer']
        original_fingerprint = api.fingerprint(original)
        declared = adapted_fixture(original)
        trainer,fixture = api.cpu_fixture(package,declared)
        assert api.fingerprint(original) == original_fingerprint
        stamp = dict(trainer.birth_death.paired_average)
        assert trainer.birth_death.settings['cells'] == 128 and stamp['cells'] == 64 and stamp['rank'] == 8
        assert trainer.birth_death.last['count_multiplicity'] == 194
        assert trainer.birth_death.last['count_categories'] == 128
        assert trainer.birth_death.snapshot is None
        assert trainer._serve_settled() == stamp['eligible']
        assert (trainer._fast is not None) == stamp['eligible']
        fixtures[original['completed_steps']] = (original,declared,fixture,trainer)
        records.append(dict(step=original['completed_steps'],requested_cells=128,actual_cells=64,
            stamp=stamp,same_scalar_served_dispatch=True,load_without_chart=True))
    original,declared,qualified,subject = fixtures[2000]
    assert subject._fast is not None, 'Saved final RA8 positive view must exercise serving API'
    clone = checkpoint_roundtrip(api,package,declared,subject)
    models = deepcopy(qualified['models'])
    assert api.fingerprint(subject.G.state_dict()) == api.fingerprint(models['ema_G'])
    assert api.fingerprint(subject.prior.state_dict()) == api.fingerprint(models['ema_prior'])
    independent = subject.state_dict(); independent['models']['G']['0.weight'].add_(1.)
    assert api.fingerprint(subject.G.state_dict()) == api.fingerprint(models['ema_G'])
    graph = subject.birth_death.lineage.neighbors.clone()
    subject._serve_release()
    assert api.fingerprint(subject.G.state_dict()) == api.fingerprint(models['G'])
    assert api.fingerprint(subject.prior.state_dict()) == api.fingerprint(models['prior'])
    # Actual current-D, even-only fit. Requested128 fits64 on this fixture;
    # the entire chart, query cache, count evidence, topology and RNG match64.
    modes = [(m,m.training) for root in (subject.G,subject.D) for m in root.modules()]
    try:
        subject.G.eval(); subject.D.eval()
        with torch.no_grad():
            real = subject.birth_death._features(subject,qualified['birth_death']['reservoir'],chunk=256)
            query = subject.birth_death._capture_generated(subject,subject.prior.z.detach())
            stream1 = torch.Generator().set_state(qualified['cpu_rng'])
            stream2 = torch.Generator().set_state(qualified['cpu_rng'])
            old_chart = baseline.FeatureCellSnapshot.fit(real,generator=stream1,cells=64,rank=8,chunk=256)
            new_chart = package.FeatureCellSnapshot.fit(real,generator=stream2,cells=128,rank=8,chunk=256)
            old_chart.cache_queries(query); new_chart.cache_queries(query)
            old_comparison = old_chart.cell_comparison(query)
            new_comparison = new_chart.cell_comparison(query)
        left,right = deepcopy(old_chart.__dict__),deepcopy(new_chart.__dict__)
        assert left['requested_cells'] == 64 and right['requested_cells'] == 128
        right['requested_cells'] = 64
        assert api.fingerprint(left) == api.fingerprint(right)
        assert api.fingerprint(old_comparison) == api.fingerprint(new_comparison)
        assert torch.equal(stream1.get_state(),stream2.get_state())
    finally:
        for module,mode in modes: module.training = mode
    geometry = subject.birth_death.latent_geometry
    ids = torch.arange(13,dtype=torch.long)
    fast_geometry = geometry._local_geometry(subject.prior.z.detach()[ids],subject.prior,rows=ids)
    builds = geometry.work['builds']; version = subject.prior.z._version
    subject._serve_apply()
    assert subject.prior.z._version != version
    averaged_geometry = geometry._local_geometry(subject.prior.z.detach()[ids],subject.prior,rows=ids)
    assert geometry.work['builds'] == builds + 1
    cold = type(geometry)(rank=geometry.rank,neighbors=geometry.neighbors,chunk=geometry.chunk,
                          lineage=subject.birth_death.lineage)
    assert all(torch.equal(a,b) for a,b in zip(averaged_geometry,
        cold._local_geometry(subject.prior.z.detach()[ids],subject.prior,rows=ids)))
    subject._serve_release()
    assert all(torch.equal(a,b) for a,b in zip(fast_geometry,
        geometry._local_geometry(subject.prior.z.detach()[ids],subject.prior,rows=ids)))
    assert geometry.work['builds'] == builds + 2 and torch.equal(graph,subject.birth_death.lineage.neighbors)
    subject._serve_apply()
    before_sample = api.fingerprint(api.served_view(subject))
    a = torch.Generator().set_state(qualified['cpu_rng']); b = torch.Generator().set_state(qualified['cpu_rng'])
    assert torch.equal(subject.sample(17,generator=a),subject.sample(17,ema=True,generator=b))
    assert torch.equal(a.get_state(),b.get_state())
    assert before_sample == api.fingerprint(api.served_view(subject))
    forwarded=[]; perturb=subject.birth_death.perturb_latent
    def trace(latent,stream,controller=None,record=False,*,prior=None,rows=None):
        forwarded.append(None if rows is None else rows.clone())
        return perturb(latent,stream,controller,record,prior=prior,rows=rows)
    subject.birth_death.perturb_latent=trace
    try:
        latent=subject.prior.z.detach()[ids]
        subject._generate(subject.G,latent,0.,torch.Generator().set_state(qualified['cpu_rng']))
        positional=subject._generate(subject.G,latent,.029,torch.Generator().set_state(qualified['cpu_rng']),ids)
        keyword=subject._generate(subject.G,latent,.029,torch.Generator().set_state(qualified['cpu_rng']),rows=ids)
        subject.sample(len(ids),generator=torch.Generator().set_state(qualified['cpu_rng']))
    finally: subject.birth_death.perturb_latent=perturb
    assert forwarded[0] is None and torch.equal(forwarded[1],ids) and torch.equal(forwarded[2],ids)
    assert forwarded[3] is not None and torch.equal(positional,keyword)
    tree=ast.parse(HARNESS.read_text())
    defaults=next(n for n in tree.body if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Name) and t.id=='DEFAULT_OPTIONS' for t in n.targets))
    resolver=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='resolve_options')
    host=dict(inspect=inspect)
    exec(compile(ast.Module(body=[defaults,resolver],type_ignores=[]),str(HARNESS),'exec'),host)
    options,detected=host['resolve_options'](package,{}, {})
    assert options['evaluation_generate'] == detected['evaluation_generate'] == 'indexed'
    assert subject.output_sigma() == float(baseline.GANTrainer._output_sigma(subject,subject.recipe.output_noise_std))
    n=subject.birth_death.N
    subject.birth_death.rows_since_eval=n-1; assert subject._serve_settled()
    subject.birth_death.rows_since_eval=n; assert not subject._serve_settled()
    subject._serve_apply(); assert subject._fast is None
    expired=deepcopy(qualified); expired['birth_death']['rows_since_eval']=n
    clone.load_state_dict(expired); assert clone._fast is None and clone.birth_death.snapshot is None
    subject.load_state_dict(qualified)
    rejected=[]
    def reject(label,mutator):
        bad=deepcopy(qualified); mutator(bad)
        rejected.append(api.reject_atomic(subject,bad,label))
    reject('old_backend7_same_new_recipe',lambda s:s['birth_death'].update(backend_schema=7))
    reject('missing_resolution_policy',lambda s:s['birth_death']['settings'].pop('resolution_policy'))
    reject('requested_settings_mismatch',lambda s:s['birth_death']['settings'].update(cells=64))
    def bad_cells(s):
        bd=s['birth_death']; bd['paired_average']['cells']=63
        bd['last'].update(cells=63,paired_average=deepcopy(bd['paired_average']))
    reject('actual_cells_below_formula',bad_cells)
    reject('categories_use_requested_not_actual',lambda s:s['birth_death']['last'].update(count_categories=256))
    reject('category_integer_required',lambda s:s['birth_death']['last'].update(count_categories=True))
    reject('multiplicity_use_requested_not_actual',lambda s:s['birth_death']['last'].update(count_multiplicity=386))
    reject('cutoff_use_requested_not_actual',lambda s:s['birth_death']['last'].update(count_cutoff=.05/386))
    reject('partition_wrong_fit_rows',lambda s:s['birth_death']['last']['count_partition'].update(fitted_rows=513))
    reject('partition_fit_rows_strict_int',lambda s:s['birth_death']['last']['count_partition'].update(fitted_rows=512.))
    reject('partition_wrong_categories',lambda s:s['birth_death']['last']['count_partition'].update(categories=256))
    def future(s):
        bd=s['birth_death']; bd['paired_average']['step']+=1
        bd['last'].update(step=bd['paired_average']['step'],paired_average=deepcopy(bd['paired_average']))
    reject('future_reaction_stamp',future)
    assert torch.equal(graph,subject.birth_death.lineage.neighbors)
    # Actual next-update resume on a nonterminal saved input, with a declared
    # positive API stamp. No model motion law or reaction plan is retested.
    _,early_declared,early_fixture,_=fixtures[1000]
    positive=api.stamp_counts(early_fixture,len(early_declared['models']['prior']['z']))
    continued=api.construct(package,early_declared); continued.load_state_dict(positive)
    resumed=checkpoint_roundtrip(api,package,early_declared,continued)
    batch=early_declared['birth_death']['reservoir'][:128]
    torch.set_rng_state(positive['cpu_rng']); first=continued.step(batch)
    torch.set_rng_state(positive['cpu_rng']); second=resumed.step(batch)
    assert all(torch.equal(first[k],second[k]) for k in first if isinstance(first[k],torch.Tensor))
    assert api.fingerprint(api.served_view(continued)) == api.fingerprint(api.served_view(resumed))
    assert continued.completed_steps == resumed.completed_steps == 1001
    small=api.construct(package,early_declared,particles=12)
    assert not hasattr(small.birth_death,'paired_average_eligible')
    small._table_tester().last_decisive=-1
    assert small._serve_settled() and baseline.GANTrainer._serve_settled(small)
    small._table_tester().last_decisive=0
    assert not small._serve_settled() and not baseline.GANTrainer._serve_settled(small)
    fresh=api.construct(package,early_declared)
    initial=fresh.state_dict(); assert initial['birth_death']['paired_average']['snapshot']==0
    fresh.load_state_dict(initial); assert fresh.birth_death.snapshot is None and fresh._fast is None
    torch.set_rng_state(initial_rng)
    assert not torch.cuda.is_initialized()
    assert protected == {str(p):sha(p) for p in protected_paths}
    guard_map(owner_seal['source_and_input_sha256'])
    value=dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),
        composition_sha256=sha(args.composition),package_sha256=composition['package_sha256'],
        config_sha256=sha(args.config),backend_schema=8,trainer_schema=5,
        records=records,unchanged_method_asts=unchanged,
        source_and_input_sha256=protected,all29_modules_exact_owner=True,
        exactly_one_config_value_changed=True,
        controls=dict(actual_initial_and_noninitial_load=True,old7_and_malformed_state_atomic_rejection=True,
            same_actual64_chart_cache_comparison_topology_and_rng=True,independent_served_checkpoint=True,
            served_swap_release_version_rebuild_and_lineage_exact=True,load_discards_derived_caches=True,
            default_versus_explicit_EMA_sample_noise_rng_exact=True,named5th_positional_and_public_sample_ids=True,
            original_native_resolver_indexed=True,strict_turnover_expiry_and_expired_load_fast=True,
            served_checkpoint_next_update_continuation_exact=True,N12_legacy_fallback=True),
        rejected_controls=rejected,cpu_only=True,cuda_initialized=False,private_CPU_trainer_updates=2,
        new_copy_birth_reaction_suites=0,extra_neutrality_suites=0,new_evaluator_emissions=0,
        new_seed_experiments=0,global_cpu_rng_restored=True,production_sources_modified=False,quality_verdict=None,
        limits=['Old RA8 saved tensors are copied into an explicitly declared new-law CPU fixture, never migrated by production loading.',
            'CPU streams use copied saved CPU RNG bytes; this is not historical CUDA replay or cross-device resume.',
            'Requested128 is metadata; actual64 is shown on saved toy. Native actual128 and action parity belong to owner contracts.',
            'The unchanged empirical serving lease is not stationarity, equivalence or emitted quality; it can age for less than one real FIFO turnover.',
            'A synthetic positive stamp at1000 is an API control, not a measured eligibility or production override.',
            'Unchanged extra-pass neutrality and numerical actions are covered by prior frozen and owner proofs, not duplicated here.'])
    (args.output/'receipt.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',output=str(args.output/'receipt.json'),rejections=len(rejected),
        measured_chart_actual_K=new_chart.cells,package_sha256=composition['package_sha256'])),flush=True)


if __name__ == '__main__':
    main()
