"""Compose separate reviewed mechanisms; no prior package or evidence changes."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import textwrap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'quality/ra5'
PACKAGE = ROOT / 'pkg-CB64-RA5'
PAIR = ROOT / 'performance/sampler-regression/cpu-plan-review/post-ra4-quality'
BIRTH = ROOT / 'integration/review/training-regression/post-ra4-quality'
POPULATION = ROOT / 'performance/training-regression/count-review/population-scheduler'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(source, before, after):
    assert source.count(before) == 1, before[:120]
    return source.replace(before, after, 1)


def method_source(source, owner, name):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    start = min([method.lineno] + [n.lineno for n in method.decorator_list]) - 1
    return '\n'.join(source.splitlines()[start:method.end_lineno])


def frozen_absolute_files(value):
    found = {}
    if isinstance(value, dict):
        for name, item in value.items():
            if isinstance(name, str) and name.startswith('/') and isinstance(item, str) and len(item) == 64:
                found[name] = item
            found.update(frozen_absolute_files(item))
    elif isinstance(value, list):
        for item in value:
            found.update(frozen_absolute_files(item))
    return found


def verify_owner(ready, required_files):
    value = json.loads(ready.read_text())
    frozen = frozen_absolute_files(value)
    for field in ('numerical_source_sha256','local_source_sha256','evidence_sha256','helper_source_sha256'):
        for name, expected in value.get(field,{}).items():
            path = Path(name) if name.startswith('/') else ready.parent/name
            frozen[str(path)] = expected
    assert frozen, f'No absolute frozen source map: {ready}'
    for name, expected in frozen.items():
        assert sha(name) == expected, f'Owner frozen input changed: {name}'
    if 'package_source_sha256' in value:
        package_root = Path(value['package_root'])
        package = package_root/'particlegan'
        prefix = package_root if all(name.startswith('particlegan/') for name in value['package_source_sha256']) else package
        actual = {str(p.relative_to(prefix)):sha(p) for p in sorted(package.rglob('*.py'))}
        assert actual == value['package_source_sha256'], f'Owner package source map changed: {ready}'
        frozen.update({str(prefix/name):expected for name,expected in value['package_source_sha256'].items()})
        digest = hashlib.sha256()
        for p in sorted(package.rglob('*.py')):
            digest.update(str(p.relative_to(package)).encode()+b'\0'+p.read_bytes()+b'\0')
        if 'package_sha256' in value:
            # Population READY declares the hash of its full-root relative
            # source manifest; PAIR READY uses the canonical byte digest.
            aggregate = (hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest()
                if prefix == package_root else digest.hexdigest())
            assert aggregate == value['package_sha256'], f'Owner package digest changed: {ready}'
    for path in required_files:
        assert str(path) in frozen, f'Required owner input not frozen: {path}'
    return value


def integrate_transport(source):
    old = method_source(source, 'FeatureCellSnapshot', 'ordinary_transport')
    new = replace_once(old, 'pvalues=None, max_moves=None):',
        'pvalues=None, max_moves=None, birth_planner=None):')
    new = replace_once(new, '        remaining = budget-len(child)\n',
        '        birth_plan = None if birth_planner is None else birth_planner(child,parent,after_local,budget)\n'
        '        birth_rows = empty if birth_plan is None else birth_plan["children"]\n'
        '        remaining = budget-len(child)-len(birth_rows)\n')
    new = replace_once(new, '        if possible:\n            global_child,global_parent,global_phase =',
        '        if birth_plan is not None:\n'
        '            global_child,global_parent,global_phase = plan_residual_global_copies(\n'
        '                self,query_features,flags,pvalues,comparison,birth_plan,generator=generator,\n'
        '                previous_children=child,previous_copy_parents=parent)\n'
        '        elif possible:\n            global_child,global_parent,global_phase =')
    new = replace_once(new, '        planned = after_local+torch.bincount(ids[global_parent],minlength=self.cells)\n',
        '        supported_before_global = after_local if birth_plan is None else birth_plan["planned_supported_counts"]\n'
        '        planned = supported_before_global+torch.bincount(ids[global_parent],minlength=self.cells)\n')
    new = replace_once(new, '        return child,parent,detail',
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
        '        return child,parent,detail')
    return replace_once(source, old, new)


def integrate_reaction(source):
    old = method_source(source, 'FeatureCellBirthDeath', 'maybe_apply')
    new = replace_once(old,
        '            child,parent,ordinary = snapshot.ordinary_transport(q,flags,comparison,generator=self.stream,pvalues=pvalues)\n'
        '            iso_child,iso_parent,anchored = snapshot.select_parents(q,flags,ordinary_children=child,ordinary_parents=parent,\n'
        '                                                                 generator=self.stream,pvalues=pvalues)',
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
        '                supported_counts=ordinary["planned_supported_counts"])')
    new = replace_once(new,
        '            c["realised_deaths"] += len(child); c["realised_births"] += len(parent)',
        '            c["realised_deaths"] += len(child)+len(newborn); c["realised_births"] += len(parent)+len(newborn)')
    new = replace_once(new, "ordinary_moves=0 if self.dry_run else len(child),",
        "ordinary_moves=0 if self.dry_run else ordinary['moves'],")
    new = replace_once(new, '            if not snapshot.valid_metric:',
        '            last.update(ordinary_copy_moves=0 if self.dry_run else len(child),\n'
        '                ordinary_novel_birth_moves=0 if self.dry_run else len(newborn),\n'
        '                novel_birth_attempts=birth_plan["attempted_cells"],\n'
        '                novel_birth_target_cells=birth_plan["target_cell_ids"].clone(),\n'
        '                novel_birth_children=newborn.clone(),novel_birth_seed_rows=birth_plan["source_seed_rows"].clone(),\n'
        '                novel_birth=novel_birth_diagnostics(birth_plan),\n'
        '                novel_birth_work_bound=dict(birth_plan["work_bound"]))\n'
        '            if not snapshot.valid_metric:')
    new = replace_once(new, '                moved = torch.cat((child,iso_child))',
        '                apply_anchor_births(trainer,self,birth_plan)\n'
        '                moved = torch.cat((child,iso_child,newborn))')
    new = replace_once(new, '                c["ordinary_moves"] += len(child); c["moves"] += len(child)',
        '                c["ordinary_moves"] += len(child)+len(newborn); c["moves"] += len(child)+len(newborn)\n'
        '                c["novel_birth_moves"] += len(newborn)\n'
        '                c["novel_birth_attempts"] += birth_plan["attempted_cells"]')
    new = replace_once(new, '                last["moves"] = len(child)+len(iso_child)',
        '                last["moves"] = len(child)+len(iso_child)+len(newborn)')
    new = replace_once(new, '                c["matched"] += len(child)',
        '                c["matched"] += len(child)  # Copy matches; novel births have their own counter.')
    return replace_once(source, old, new)


def main():
    assert not PACKAGE.exists() and not OUT.exists()
    assert not (ROOT/'configs/overrides-CB64-RA5.json').exists()
    for receipt in (PAIR/'READY.json', BIRTH/'READY.json', POPULATION/'READY.json'):
        assert receipt.exists(), receipt
    verify_owner(PAIR/'READY.json',[PAIR/'pkg-PAIR-EMA/particlegan/feature_cells.py'])
    verify_owner(BIRTH/'READY.json',[BIRTH/'anchor_birth.py',BIRTH/'birth_phase.py',
        BIRTH/'pkg-ANCHOR-CONTRACT/particlegan/feature_cells.py'])
    verify_owner(POPULATION/'READY.json',[POPULATION/'pkg-POPULATION/particlegan/continuous.py',
        POPULATION/'pkg-POPULATION/particlegan/training.py'])
    shutil.copytree(PAIR/'pkg-PAIR-EMA', PACKAGE, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    path = PACKAGE/'particlegan/feature_cells.py'
    source = path.read_text()
    contract = (BIRTH/'pkg-ANCHOR-CONTRACT/particlegan/feature_cells.py').read_text()
    source = replace_once(source, method_source(source,'FeatureCellSnapshot','select_parents'),
        method_source(contract,'FeatureCellSnapshot','select_parents'))
    source = replace_once(source, 'from .birth_death import ParticleBirthDeath',
        'from .birth_death import ParticleBirthDeath\n'
        'from .birth_phase import (plan_real_anchor_births,learned_latent_features,\n'
        '    plan_residual_global_copies,plan_residual_isolation,apply_anchor_births,novel_birth_diagnostics)')
    source = integrate_transport(source)
    source = integrate_reaction(source)
    source = replace_once(source, '    BACKEND_SCHEMA = 5', '    BACKEND_SCHEMA = 6')
    source = replace_once(source, '        self.lineage = LatentLineage(',
        '        self.settings.update(novel_birth_policy="paired_even_real_anchor_shared_3K_plus_2_v1",\n'
        '            novel_birth_cells=4,novel_birth_linearizations=4,\n'
        '            novel_birth_trust="per_model_prior_rms_coordinate_spread")\n'
        '        self.lineage = LatentLineage(')
    source = replace_once(source, '"feature_forward_rows","invalidated_cells")',
        '"feature_forward_rows","invalidated_cells","novel_birth_moves","novel_birth_attempts")')
    path.write_text(source)
    for name in ('anchor_birth.py','birth_phase.py'):
        shutil.copyfile(BIRTH/name,PACKAGE/'particlegan'/name)
    for name in ('continuous.py','training.py'):
        shutil.copyfile(POPULATION/'pkg-POPULATION/particlegan'/name,PACKAGE/'particlegan'/name)
    for path in PACKAGE.rglob('*.py'):
        compile(path.read_text(),str(path),'exec')
    shutil.copyfile(ROOT/'configs/overrides-CB64-RA4.json',ROOT/'configs/overrides-CB64-RA5.json')
    OUT.mkdir()
    sources = {str(p.relative_to(PACKAGE/'particlegan')):sha(p) for p in sorted(PACKAGE.rglob('*.py'))}
    inputs = {str(p):sha(p) for p in (PAIR/'READY.json',BIRTH/'READY.json',POPULATION/'READY.json',Path(__file__))}
    receipt = dict(status='COMPOSED_CPU_REVIEW_PENDING_NO_QUALITY_RUN',package_root=str(PACKAGE),
        source_sha256=sources,composed_from=inputs,config_sha256=sha(ROOT/'configs/overrides-CB64-RA5.json'),
        exact_move_ast=ast.dump(ast.parse(textwrap.dedent(method_source(source,'FeatureCellBirthDeath','_move'))),include_attributes=False))
    (OUT/'COMPOSITION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'],package=str(PACKAGE))))


if __name__ == '__main__':
    main()
