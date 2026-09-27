#!/usr/bin/env python3
"""Freeze existing RP12/RP14/RP15 recovery-ring followups; standard library only."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import shutil
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
BASE = HERE / 'dv23-single-shift-preparation/api-dv2'
OUTPUT = HERE / 'precision-three-single-shift-preparation'
EVIDENCE = HERE.parent / 'continuous-api-search/evidence'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def replace(source, old, new):
    assert source.count(old) == 1, (old[:100], source.count(old))
    return source.replace(old, new)


def spans(source):
    return {n.name: ast.get_source_segment(source, n) for n in ast.parse(source).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def literal_defaults(source):
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Recipe')
    return json.loads(json.dumps({n.target.id: ast.literal_eval(n.value) for n in cls.body
                                 if isinstance(n, ast.AnnAssign)}))


def main():
    if OUTPUT.exists():
        raise FileExistsError('Never overwrite sealed preparations')
    base_manifest = read(BASE / 'bundle-sha256.json')
    for name, wanted in base_manifest['files'].items():
        assert sha(BASE / name) == wanted
    OUTPUT.mkdir()
    (OUTPUT / 'originals').mkdir()
    (OUTPUT / 'prepared-bundles').mkdir()
    original_worker = (BASE / 'worker.py').read_text()
    worker = replace(original_worker, 'DV2/DV3 original recovery-ring retest', 'RP12/RP14/RP15 original recovery-ring followup')
    worker = replace(worker, '            assert not value.opt_g.state and not value.opt_d.state, "release Adam must begin lazy/empty"', '''            assert value.serial_backward is True and value.precision is not None
            for optimizer in (value.opt_g,value.opt_d):
                for group in optimizer.param_groups:
                    for parameter in group['params']:
                        state=optimizer.state[parameter]
                        assert {'step','exp_avg','exp_avg_sq'} <= state.keys()
                        assert state['step'].shape==() and float(state['step'])==0
                        assert all(state[key].device==parameter.device for key in ('step','exp_avg','exp_avg_sq'))
                        assert not bool(state['exp_avg'].count_nonzero()) and not bool(state['exp_avg_sq'].count_nonzero())''')
    worker = replace(worker, 'r["step_device"] == "cpu"', 'r["step_device"] == r["parameter_device"]')
    worker = replace(worker, 'unexpected native Adam placement; no relocation or repair allowed', 'unexpected declared package-owned eager Adam placement; no relocation or repair allowed')
    worker = replace(worker, '        initial = host.state_receipt(trainer, stream, means)', '        optimizer_proof(trainer, "main-0000")\n        initial = host.state_receipt(trainer, stream, means)')
    worker = replace(worker, 'adam_state_empty=True', 'adam_state_empty=False, adam_eager_state=recipe.adam_eager_state, counter_ownership="unchanged public candidate recipe"')
    worker = replace(worker, '                assert trainer.completed_steps == step', '''                assert trainer.completed_steps == step
                assert trainer.precision.state['updates']==step
                assert trainer.game_stats['accepted_updates']==step
                assert trainer.game_stats['field_evaluations']==2''')
    assert worker.count('policy=trainer.controller.diagnostics()') == 2
    worker = worker.replace('policy=trainer.controller.diagnostics()', 'precision=dict(trainer.precision.state), game=trainer.game_stats')
    worker = replace(worker, '                                 losses={k:float(v) for k,v in stats.items() if isinstance(v,torch.Tensor)},', '''                                 losses={k:float(v) for k,v in stats.items() if isinstance(v,torch.Tensor)},
                                 trainer_stats=serializable(stats),''')
    worker = replace(worker, '        frozen_digest = None', '''        def serializable(value):
            if isinstance(value,torch.Tensor):
                value=value.detach().cpu()
                return value.item() if value.numel()==1 else value.tolist()
            if isinstance(value,dict):return {str(k):serializable(v) for k,v in value.items()}
            if isinstance(value,(list,tuple)):return [serializable(v) for v in value]
            if value is None or isinstance(value,(str,int,float,bool)):return value
            raise TypeError(type(value))

        frozen_digest = None''')
    worker = replace(worker, '                       completed_updates=trainer.completed_steps,observations=len(observations),', '                       completed_updates=trainer.completed_steps,field_evaluations=2*trainer.completed_steps,observations=len(observations),')
    contract = (BASE / 'ring_contract.py').read_text()
    contract = replace(contract, "optimizer_options={'foreach':False,'fused':False})", "optimizer_options={'foreach':False,'fused':False},serial_backward=True)")
    execution = (BASE / 'execution_contract.py').read_text()
    execution = replace(execution, 'requires exact DV2/DV3 trainer schema4', 'requires exact precision-candidate trainer schema4')
    execution = replace(execution, '\n\ndef validate_bundle', '''
    if envelope['trainer'].get('serial_backward') is not True or 'precision' not in envelope['trainer']:
        raise ValueError('precision checkpoint must retain serial execution and controller state')


def validate_bundle''')
    preflight = (BASE / 'preflight.py').read_text()
    preflight = replace(preflight, "recipe['output_noise_warmup']==0", "recipe['noise_policy']=='constant'")
    preflight = replace(preflight, "    assert recipe['continuous_policy'] in ('dv2','dv3')", "    assert recipe['continuous_precision']=='rp5' and recipe['game_update']=='secant_resolvent'\n    assert recipe['adam_eager_state'] is True\n    assert declaration['initial_optimizer_state']=='declared_eager'\n    assert declaration['optimizer_step_devices']=={'D':'parameter','G':'parameter'}")
    for source in (worker, contract, execution, preflight):
        ast.parse(source)
    (OUTPUT / 'worker-changes.patch').write_text(''.join(difflib.unified_diff(original_worker.splitlines(True), worker.splitlines(True), fromfile='reviewed-dv23-ring/worker.py', tofile='precision-three-ring/worker.py')))
    for file in ('worker.py', 'ring_contract.py', 'execution_contract.py', 'declaration.json'):
        shutil.copyfile(BASE / file, OUTPUT / 'originals' / ('dv2-' + file))
    baseline = read(BASE / 'declaration.json')
    rng_keys = ('streams_sha256', 'cpu_rng_sha256', 'cuda_rng_sha256', 'real_stream_sha256', 'means_sha256')
    canonical = read(EVIDENCE / 'api-rp5-single/initial.json')
    for name in ('api-rp7-single', 'api-dv2-single'):
        another = read(EVIDENCE / name / 'initial.json')
        assert all(canonical[k] == another[k] for k in rng_keys)
    shutil.copyfile(EVIDENCE / 'api-rp5-single/initial.json', OUTPUT / 'originals/canonical-rp5-ring-initial.json')
    selected = spans((BASE / 'source/frozen_host.py').read_text())
    rows, archives = [], []
    for name in ('api-rp12', 'api-rp14', 'api-rp15'):
        port = HERE / 'port-source' / name
        declaration = read(port / 'candidate-declaration.json')
        assert declaration['serial_backward_argument'] is True
        assert declaration['initial_optimizer_state'] == 'declared_eager'
        assert declaration['optimizer_step_devices'] == {'D': 'parameter', 'G': 'parameter'}
        original_zip = Path(declaration['algorithm_source']['source_zip'])
        assert sha(original_zip) == declaration['algorithm_source']['source_zip_sha256']
        target = OUTPUT / name
        (target / 'source').mkdir(parents=True)
        source_receipts = {}
        with zipfile.ZipFile(original_zip) as archive:
            old_worker = archive.read('reports/reversible-precision/public_worker.py').decode()
            old_spans = spans(old_worker)
            call = ast.parse(old_spans['make_recipe']).body[0].body[0].value
            assert isinstance(call, ast.Call) and call.func.id == 'get_recipe' and not call.args
            overrides = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
            old_resolved = literal_defaults(archive.read('particlegan/recipes.py').decode()) | overrides
            expected_recipe = old_resolved | {'initialization': 'batch_feature_zero'}
            recipe = declaration['resolved_recipe'] | {'num_particles': 20000, 'z_dim': 2, 'batch_size': 2048}
            assert recipe == expected_recipe, (name, {k:(recipe.get(k), expected_recipe.get(k)) for k in recipe.keys() | expected_recipe.keys() if recipe.get(k)!=expected_recipe.get(k)})
            for source, symbols in {
                    'benchmarks/locked_shared/mlp.py': ('SimpleMLPGenerator', 'SimpleMLPDiscriminator'),
                    'benchmarks/locked_shared/mode_hold.py': ('ring_means', 'diversity'),
                    'reports/reversible-precision/public_worker.py': ('digest', 'rates', 'state_receipt')}.items():
                old = spans(archive.read(source).decode())
                for symbol in symbols:
                    assert selected[symbol] == old[symbol], (name, symbol)
                    source_receipts[symbol] = dict(original_source=source, sha256=hashlib.sha256(old[symbol].encode()).hexdigest())
            (OUTPUT / 'originals' / (name + '-public_worker.py')).write_text(old_worker)
            (OUTPUT / 'originals' / (name + '-recipes.py')).write_bytes(archive.read('particlegan/recipes.py'))
        with zipfile.ZipFile(port / 'package.zip') as archive:
            assert {n: hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist()} == declaration['package_sha256']
            for rel in archive.namelist():
                assert rel.startswith('particlegan/') and '..' not in Path(rel).parts
                path = target / 'source' / rel
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(archive.read(rel))
        for file, source in {'worker.py': worker, 'ring_contract.py': contract, 'execution_contract.py': execution, 'preflight.py': preflight}.items():
            (target / file).write_text(source)
        shutil.copyfile(BASE / 'source/frozen_host.py', target / 'source/frozen_host.py')
        prepared = dict(baseline)
        for key in ('historical_declaration_sha256', 'mechanism'):
            prepared.pop(key, None)
        prepared.update(candidate=declaration['candidate'], initializer_commit=declaration['initializer_commit'],
            package_sha256=declaration['package_sha256'], port_manifest_sha256=sha(port / 'port-manifest.json'),
            port_package_sha256=sha(port / 'package.zip'), candidate_declaration_sha256=sha(port / 'candidate-declaration.json'),
            historical_source_zip_sha256=sha(original_zip), historical_ring_quality='NOT_RUN; this is the first own ring measurement for this exact candidate',
            recipe=recipe, execution=baseline['execution'] | {'adam_state': 'Unchanged public candidate recipe eagerly initializes Adam state; scalar steps, parameters, gradients and moments use the parameter device (CUDA during training).'}, initial_optimizer_state=declaration['initial_optimizer_state'], optimizer_step_devices=declaration['optimizer_step_devices'],
            serial_backward_argument=True, host_source_receipt=source_receipts,
            expected_initial_fixture={k: canonical[k] for k in rng_keys},
            initial_comparison_scope='Canonical RP5/RP7/DV2 ring RNG and means match; no historical weight or score inheritance. Every fresh initial non-RNG state tensor and named parameter/buffer must match own independent CPU proof.',
            recipe_binding=dict(original_source='reports/reversible-precision/public_worker.py:make_recipe', original_source_zip=str(original_zip), original_overrides=overrides,
                method='AST literal Recipe defaults plus original get_recipe overrides; equals port resolved tiny recipe with only original ring resources restored and new initializer enabled.'),
            mechanism=dict(continuous_precision=recipe['continuous_precision'], game_update=recipe['game_update'], noise_policy=recipe['noise_policy'],
                optional_variant={k:recipe[k] for k in ('generator_update','relativistic_pairing','particle_loss_weighting') if k in recipe},
                field_evaluations_per_accepted_update=2, rates='Autonomous reversible precision; no evaluator horizon or target-change hint.'),
            checkpoint='Schema4 trainer, precision/reference, optimizer/KA2/EMA/RNG plus caller data stream, means and serial execution. Copy full state at2400 into frozen control; never reset main. Initial/first/final and frozen optimizer receipts preserve package-owned parameter-device clocks.')
        dump(target / 'declaration.json', prepared)
        files = {str(p.relative_to(target)): sha(p) for p in sorted(target.rglob('*')) if p.is_file()}
        dump(target / 'bundle-sha256.json', dict(schema=1, files=files))
        manifest_sha = sha(target / 'bundle-sha256.json')
        rows.append(dict(candidate=prepared['candidate'], directory=str(target), manifest_sha256=manifest_sha,
            required_review=str(OUTPUT / 'reviews' / name / 'cpu-constructor-proof.json'), quality='NOT_RUN', source_status='PREPARED_REQUIRES_INDEPENDENT_CPU_REVIEW'))
        compact_files = files | {'bundle-sha256.json': manifest_sha}
        archive_path = OUTPUT / 'prepared-bundles' / (name + '.zip')
        with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for rel in sorted(compact_files):
                info = zipfile.ZipInfo(rel, (2026, 9, 27, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, (target / rel).read_bytes())
        archives.append(dict(candidate=prepared['candidate'], directory_relative=name,
            archive=str(archive_path.relative_to(OUTPUT)), archive_sha256=sha(archive_path), manifest_sha256=manifest_sha, files=compact_files))
    dump(OUTPUT / 'prepared-index.json', dict(rows=rows, preparer_sha256=sha(Path(__file__)), template_manifest_sha256=sha(BASE / 'bundle-sha256.json')))
    dump(OUTPUT / 'prepared-bundles/index.json', dict(schema=1, rows=archives))
    shutil.copyfile(BASE.parent / 'restore_prepared.py', OUTPUT / 'restore_prepared.py')
    (OUTPUT / '.gitignore').write_text('/api-rp12/\n/api-rp14/\n/api-rp15/\n__pycache__/\n')
    results = [json.loads(subprocess.run(['python', str(Path(r['directory']) / 'preflight.py')], capture_output=True, text=True, check=True).stdout) for r in rows]
    dump(OUTPUT / 'source-preflight-summary.json', dict(results=results, torch_imported=False, training=False))
    subprocess.run(['python', str(OUTPUT / 'restore_prepared.py')], check=True)
    print(json.dumps(dict(rows=rows, training=False), indent=2))


if __name__ == '__main__':
    main()
