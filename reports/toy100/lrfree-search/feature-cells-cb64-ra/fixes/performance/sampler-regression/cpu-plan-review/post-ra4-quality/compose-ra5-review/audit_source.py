"""Independent inverse composition proof; stdlib only, no package import."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import textwrap

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
PAIR = ROOT / 'performance/sampler-regression/cpu-plan-review/post-ra4-quality'
BIRTH = ROOT / 'integration/review/training-regression/post-ra4-quality'
POPULATION = ROOT / 'performance/training-regression/count-review/population-scheduler'
PACKAGE = ROOT / 'pkg-CB64-RA5'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(node):
    return ast.dump(node, include_attributes=False)


def digest(package):
    value = hashlib.sha256()
    for p in sorted((package / 'particlegan').rglob('*.py')):
        value.update(str(p.relative_to(package / 'particlegan')).encode() + b'\0' + p.read_bytes() + b'\0')
    return value.hexdigest()


def source_map(package):
    return {str(p.relative_to(package / 'particlegan')): sha(p)
        for p in sorted((package / 'particlegan').rglob('*.py'))}


def method_source(source, owner, name):
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == owner)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
    return '\n'.join(source.splitlines()[start:node.end_lineno])


def once(source, before, after):
    assert source.count(before) == 1, before[:120]
    return source.replace(before, after, 1)


def owner_files(ready):
    value = json.loads(ready.read_text())
    frozen = {}
    def absolute_maps(item):
        if isinstance(item, dict):
            for name, entry in item.items():
                if isinstance(name, str) and name.startswith('/') and isinstance(entry, str) and len(entry) == 64:
                    frozen[name] = entry
                absolute_maps(entry)
        elif isinstance(item, list):
            for entry in item:
                absolute_maps(entry)
    absolute_maps(value)
    for field in ('numerical_source_sha256', 'local_source_sha256', 'evidence_sha256', 'helper_source_sha256'):
        for name, expected in value.get(field, {}).items():
            path = Path(name) if name.startswith('/') else ready.parent / name
            frozen[str(path)] = expected
    package = Path(value['package_root'])
    expected_map = value.get('package_source_sha256')
    if expected_map is not None:
        prefix = package if all(name.startswith('particlegan/') for name in expected_map) else package / 'particlegan'
        actual = {str(p.relative_to(prefix)): sha(p) for p in sorted((package / 'particlegan').rglob('*.py'))}
        assert actual == expected_map, str(ready)
        frozen.update({str(prefix / name): expected for name, expected in expected_map.items()})
    if 'package_sha256' in value:
        if ready.parent == POPULATION:
            declared = json.loads((POPULATION / 'IDENTITY-DECLARATION.json').read_text())
            assert declared['frozen_ready_sha256'] == sha(ready)
            assert hashlib.sha256(json.dumps(expected_map, sort_keys=True).encode()).hexdigest() == value['package_sha256']
            assert digest(package) == declared['canonical_package_sha256']
            assert digest(ROOT / 'pkg-CB64-RA4') == declared['canonical_base_package_sha256']
            frozen[str(POPULATION / 'IDENTITY-DECLARATION.json')] = sha(POPULATION / 'IDENTITY-DECLARATION.json')
            identity_freeze = POPULATION / 'IDENTITY-FROZEN.json'
            identity = json.loads(identity_freeze.read_text())
            assert identity['files'][str(POPULATION / 'IDENTITY-DECLARATION.json')] == frozen[str(POPULATION / 'IDENTITY-DECLARATION.json')]
            for path, expected in identity['files'].items():
                assert sha(path) == expected
                frozen[path] = expected
            frozen[str(identity_freeze)] = sha(identity_freeze)
        else:
            assert digest(package) == value['package_sha256']
    for path, expected in frozen.items():
        assert sha(path) == expected, path
    return value, frozen


def undo_transport(source):
    source = once(source, 'pvalues=None, max_moves=None, birth_planner=None):', 'pvalues=None, max_moves=None):')
    source = once(source,
        '        birth_plan = None if birth_planner is None else birth_planner(child,parent,after_local,budget)\n'
        '        birth_rows = empty if birth_plan is None else birth_plan["children"]\n'
        '        remaining = budget-len(child)-len(birth_rows)\n',
        '        remaining = budget-len(child)\n')
    source = once(source,
        '        if birth_plan is not None:\n'
        '            global_child,global_parent,global_phase = plan_residual_global_copies(\n'
        '                self,query_features,flags,pvalues,comparison,birth_plan,generator=generator,\n'
        '                previous_children=child,previous_copy_parents=parent)\n'
        '        elif possible:\n            global_child,global_parent,global_phase =',
        '        if possible:\n            global_child,global_parent,global_phase =')
    source = once(source,
        '        supported_before_global = after_local if birth_plan is None else birth_plan["planned_supported_counts"]\n'
        '        planned = supported_before_global+torch.bincount(ids[global_parent],minlength=self.cells)\n',
        '        planned = after_local+torch.bincount(ids[global_parent],minlength=self.cells)\n')
    source = once(source,
        '        if birth_plan is not None:\n'
        '            source_ids = ids[birth_rows]\n'
        '            destination_ids = birth_plan["target_cell_ids"]\n'
        '            same_group = self._mass_topology()[source_ids] == self._mass_topology()[destination_ids]\n'
        '            detail.update(novel_birth_plan=birth_plan,novel_birth_moves=len(birth_rows),\n'
        '                moves=len(child)+len(birth_rows),copy_moves=len(child),\n'
        '                death_policy="paired_copy_and_even_real_anchor_birth_shared_3K_plus_2",\n'
        '                within_group_moves=detail["within_group_moves"]+int(same_group.sum()),\n'
        '                between_group_moves=detail["between_group_moves"]+len(birth_rows)-int(same_group.sum()),\n'
        '                ordinary_flagged_deaths=detail["ordinary_flagged_deaths"]+len(birth_rows),\n'
        '                death_allocation=detail["death_allocation"]+torch.bincount(source_ids,minlength=self.cells),\n'
        '                birth_allocation=detail["birth_allocation"]+torch.bincount(destination_ids,minlength=self.cells),\n'
        '                action_children=torch.cat((child,birth_rows)),\n'
        '                action_destination_category_ids=torch.cat((categories[parent],birth_plan["destination_category_ids"])),\n'
        '                action_kinds=torch.cat((detail["action_kinds"],torch.full_like(birth_rows,3))),\n'
        '                supported_after_births=supported_before_global)\n'
        '        return child,parent,detail',
        '        return child,parent,detail')
    return source


def undo_reaction(source):
    source = once(source,
        '            current_features = learned_latent_features(self,trainer,trainer.G)\n'
        '            average_features = learned_latent_features(self,trainer,trainer.ema_G)\n'
        '            def birth_planner(copy_children,copy_parents,supported,budget):\n'
        '                return plan_real_anchor_births(snapshot,q,flags,pvalues,comparison,z,current_features,\n'
        '                    ema_latents=trainer.ema_prior.z.detach(),ema_feature_of_latent=average_features,\n'
        '                    previous_children=copy_children,previous_copy_parents=copy_parents,\n'
        '                    supported_counts=supported,max_moves=budget)\n'
        '            child,parent,ordinary = snapshot.ordinary_transport(q,flags,comparison,\n'
        '                generator=self.stream,pvalues=pvalues,birth_planner=birth_planner)\n'
        '            birth_plan = ordinary["novel_birth_plan"]\n'
        '            newborn = birth_plan["children"]\n'
        '            iso_child,iso_parent,anchored = plan_residual_isolation(snapshot,q,flags,pvalues,birth_plan,\n'
        '                generator=self.stream,copy_children=child,copy_parents=parent,\n'
        '                supported_counts=ordinary["planned_supported_counts"])',
        '            child,parent,ordinary = snapshot.ordinary_transport(q,flags,comparison,generator=self.stream,pvalues=pvalues)\n'
        '            iso_child,iso_parent,anchored = snapshot.select_parents(q,flags,ordinary_children=child,ordinary_parents=parent,\n'
        '                                                                 generator=self.stream,pvalues=pvalues)')
    source = once(source,
        '            c["realised_deaths"] += len(child)+len(newborn); c["realised_births"] += len(parent)+len(newborn)',
        '            c["realised_deaths"] += len(child); c["realised_births"] += len(parent)')
    source = once(source, "ordinary_moves=0 if self.dry_run else ordinary['moves'],", "ordinary_moves=0 if self.dry_run else len(child),")
    source = once(source,
        '            last.update(ordinary_copy_moves=0 if self.dry_run else len(child),\n'
        '                ordinary_novel_birth_moves=0 if self.dry_run else len(newborn),\n'
        '                novel_birth_attempts=birth_plan["attempted_cells"],\n'
        '                novel_birth_target_cells=birth_plan["target_cell_ids"].clone(),\n'
        '                novel_birth_children=newborn.clone(),novel_birth_seed_rows=birth_plan["source_seed_rows"].clone(),\n'
        '                novel_birth=novel_birth_diagnostics(birth_plan),\n'
        '                novel_birth_work_bound=dict(birth_plan["work_bound"]))\n'
        '            if not snapshot.valid_metric:',
        '            if not snapshot.valid_metric:')
    source = once(source,
        '                apply_anchor_births(trainer,self,birth_plan)\n'
        '                moved = torch.cat((child,iso_child,newborn))',
        '                moved = torch.cat((child,iso_child))')
    source = once(source,
        '                c["ordinary_moves"] += len(child)+len(newborn); c["moves"] += len(child)+len(newborn)\n'
        '                c["novel_birth_moves"] += len(newborn)\n'
        '                c["novel_birth_attempts"] += birth_plan["attempted_cells"]',
        '                c["ordinary_moves"] += len(child); c["moves"] += len(child)')
    source = once(source, '                last["moves"] = len(child)+len(iso_child)+len(newborn)',
        '                last["moves"] = len(child)+len(iso_child)')
    source = once(source, '                c["matched"] += len(child)  # Copy matches; novel births have their own counter.',
        '                c["matched"] += len(child)')
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert (ROOT / 'quality/ra5/COMPOSITION.json').exists()
    assert args.output.exists() and not (args.output / 'receipt.json').exists()
    all_inputs = {}
    owner_records = {}
    for name, directory in (('PAIR', PAIR), ('BIRTH', BIRTH), ('POPULATION', POPULATION)):
        ready = directory / 'READY.json'
        value, files = owner_files(ready)
        owner_records[name] = dict(ready_path=str(ready), ready_sha256=sha(ready), status=value['status'], verified_files=len(files))
        all_inputs.update(files)
        all_inputs[str(ready)] = sha(ready)
    composition_path = ROOT / 'quality/ra5/COMPOSITION.json'
    composition = json.loads(composition_path.read_text())
    candidate_map = source_map(PACKAGE)
    assert candidate_map == composition['source_sha256']
    for path, expected in composition['composed_from'].items():
        assert sha(path) == expected
        all_inputs[path] = expected
    pair_package = PAIR / 'pkg-PAIR-EMA'
    pair_map = source_map(pair_package)
    assert len(pair_map) == 27 and len(candidate_map) == 29
    assert set(candidate_map) == set(pair_map) | {'anchor_birth.py', 'birth_phase.py'}
    changed = []
    for name, expected in pair_map.items():
        if candidate_map[name] != expected:
            changed.append(name)
            assert name in ('feature_cells.py', 'continuous.py', 'training.py')
    assert set(changed) == {'feature_cells.py', 'continuous.py', 'training.py'}
    for name in ('continuous.py', 'training.py'):
        assert (PACKAGE / 'particlegan' / name).read_bytes() == (POPULATION / 'pkg-POPULATION/particlegan' / name).read_bytes()
    for name in ('anchor_birth.py', 'birth_phase.py'):
        assert (PACKAGE / 'particlegan' / name).read_bytes() == (BIRTH / name).read_bytes()
        assert (BIRTH / name).read_bytes() == (BIRTH / 'pkg-ANCHOR-CONTRACT/particlegan' / name).read_bytes()
    for p in sorted((PACKAGE / 'particlegan').rglob('*.py')):
        compile(p.read_text(), str(p), 'exec')

    candidate = (PACKAGE / 'particlegan/feature_cells.py').read_text()
    pair = (pair_package / 'particlegan/feature_cells.py').read_text()
    base = (ROOT / 'pkg-CB64-RA4/particlegan/feature_cells.py').read_text()
    contract = (BIRTH / 'pkg-ANCHOR-CONTRACT/particlegan/feature_cells.py').read_text()
    selected = method_source(candidate, 'FeatureCellSnapshot', 'select_parents')
    assert selected == method_source(contract, 'FeatureCellSnapshot', 'select_parents')
    move = method_source(candidate, 'FeatureCellBirthDeath', '_move')
    assert move == method_source(pair, 'FeatureCellBirthDeath', '_move')
    assert composition['exact_move_ast'] == dump(ast.parse(textwrap.dedent(move)))
    for name in ('_ordinary_mass_transport', '_ordinary_support_transport', '_ordinary_global_transport'):
        assert method_source(candidate, 'FeatureCellSnapshot', name) == method_source(pair, 'FeatureCellSnapshot', name)

    # Explicit inverse strings are independent of root compose_ra5.py functions.
    restored = undo_reaction(undo_transport(candidate))
    restored = once(restored, selected, method_source(pair, 'FeatureCellSnapshot', 'select_parents'))
    restored = once(restored,
        'from .birth_death import ParticleBirthDeath\n'
        'from .birth_phase import (plan_real_anchor_births,learned_latent_features,\n'
        '    plan_residual_global_copies,plan_residual_isolation,apply_anchor_births,novel_birth_diagnostics)',
        'from .birth_death import ParticleBirthDeath')
    restored = once(restored, '    BACKEND_SCHEMA = 6', '    BACKEND_SCHEMA = 5')
    restored = once(restored,
        '        self.settings.update(novel_birth_policy="paired_even_real_anchor_shared_3K_plus_2_v1",\n'
        '            novel_birth_cells=4,novel_birth_linearizations=4,\n'
        '            novel_birth_trust="per_model_prior_rms_coordinate_spread")\n', '')
    restored = once(restored,
        '"feature_forward_rows","invalidated_cells","novel_birth_moves","novel_birth_attempts")',
        '"feature_forward_rows","invalidated_cells")')
    assert restored == pair
    assert dump(ast.parse(restored)) == dump(ast.parse(pair))
    # Reverse the independently frozen PAIR ownership delta to the original RA4 module.
    restored_base = once(restored, move, method_source(base, 'FeatureCellBirthDeath', '_move'))
    restored_base = once(restored_base, '    BACKEND_SCHEMA = 5', '    BACKEND_SCHEMA = 4')
    restored_base = once(restored_base,
        '        self.settings["copy_noise_policy"] = "shared_noise_separate_live_ema_current_geometry_v1"\n', '')
    assert dump(ast.parse(restored_base)) == dump(ast.parse(base))

    config = ROOT / 'configs/overrides-CB64-RA5.json'
    original_config = ROOT / 'configs/overrides-CB64-RA4.json'
    assert config.read_bytes() == original_config.read_bytes()
    assert sha(config) == composition['config_sha256']
    training = (PACKAGE / 'particlegan/training.py').read_text()
    original_training = (ROOT / 'pkg-CB64-RA4/particlegan/training.py').read_text()
    assert method_source(training, 'GANTrainer', '_generate') == method_source(original_training, 'GANTrainer', '_generate')
    assert '"schema": 5' in training and 'BACKEND_SCHEMA = 6' in candidate
    assert 'copy_noise_policy' in candidate and 'novel_birth_policy' in candidate
    assert 'tester.rebase(group["params"], self.birth_death.moved_rows)' in training
    assert 'self.row_evidence.reset(self.birth_death.moved_rows)' in training
    # Explicit inspection of generic helper dependencies, excluding documentation.
    helper_imports = {}
    for name in ('anchor_birth.py', 'birth_phase.py'):
        tree = ast.parse((PACKAGE / 'particlegan' / name).read_text())
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module)
            elif isinstance(node, ast.Name):
                assert node.id not in {'grid100', 'toy', 'mode_ids', 'holdout', 'benchmark', 'evaluation_geometry', 'score_samples', 'score_metrics'}
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {'read_text', 'read_bytes', 'load', 'save', 'evaluation_geometry', 'score_samples', 'score_metrics'}
        assert imports <= {'math', 'time', 'torch', 'anchor_birth'}
        helper_imports[name] = sorted(imports)
    screen = Path('/ml2/hypergan/lrfree-20260926/harness/screen.py')
    assert sha(screen) == 'ee8193adbdf09e93511befae7b6491143c26de88612eddf065cbb92eb2153c3c'
    all_inputs[str(screen)] = sha(screen)
    all_inputs.update({str(PACKAGE / 'particlegan' / p): h for p, h in candidate_map.items()})
    all_inputs[str(config)] = sha(config)
    all_inputs[str(original_config)] = sha(original_config)
    all_inputs[str(composition_path)] = sha(composition_path)
    for path, expected in all_inputs.items():
        assert sha(path) == expected, path
    receipt = dict(status='PASS', utc=datetime.now(timezone.utc).isoformat(),
        scope='CPU-only stdlib composition/source audit; no CUDA context, numerical job, training, new seed or quality claim',
        package_root=str(PACKAGE), package_sha256=digest(PACKAGE), package_source_sha256=candidate_map,
        config_sha256=sha(config), owner_records=owner_records, verified_input_sha256=all_inputs,
        helper_imports=helper_imports,
        checks=dict(exact_29_module_file_set=True, all_other_24_modules_PAIR_byte_exact=True,
            population_continuous_training_byte_exact=True, birth_helpers_byte_exact=True,
            select_parents_decorated_owner_method_exact=True, pair_move_decorated_method_exact=True,
            three_count_planners_exact=True, inverse_all_declared_root_splices_reconstructs_PAIR_bytes=True,
            inverse_PAIR_delta_reconstructs_RA4_full_AST=True, configuration_bytes_unchanged=True,
            indexed_generate_API_unchanged=True, trainer_schema_5_backend_schema_6_and_both_policies=True,
            population_rows_include_all_moved_actions=True, complete_COMPOSITION_hash_maps_exact=True,
            generic_birth_helpers_no_oracle_or_evaluator_dependencies=True, canonical_harness_gates_unchanged=True,
            owner_freeze_maps_verified=True, all_reviewed_inputs_unchanged=True),
        changed_original_modules=changed, added_modules=['anchor_birth.py', 'birth_phase.py'], defects=[],
        qualification='Saved-state/action/replay mechanics are qualified by owner and root CPU contracts; strict learned toy/grid quality remains pending.')
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(status=receipt['status'], package_sha256=receipt['package_sha256'], modules=len(candidate_map))))


if __name__ == '__main__':
    main()
