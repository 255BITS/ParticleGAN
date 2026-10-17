"""Compose independently frozen count and exact performance proposals."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
HERE = ROOT/'integration/iteration-4'
OUT = ROOT/'pkg-CB64-RA4'
AXIS = ROOT/'performance/sampler-regression/cpu-plan-review/pkg-AXIS-ID'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def node(source, name, owner=None):
    tree = ast.parse(source)
    entries = tree.body if owner is None else next(n for n in tree.body
        if isinstance(n,ast.ClassDef) and n.name==owner).body
    return next((n for n in entries if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name),None)


def ast_sha(n):
    return hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()


def extract(source, n):
    return ''.join(source.splitlines(keepends=True)[min([n.lineno]+[d.lineno for d in getattr(n,'decorator_list',[])])-1:n.end_lineno])


def replace(source, old, addition):
    lines=source.splitlines(keepends=True)
    start=min([old.lineno]+[d.lineno for d in getattr(old,'decorator_list',[])])-1
    lines[start:old.end_lineno]=[addition]
    return ''.join(lines)


def verify_package(package, ready):
    sources=ready.get('package_source_sha256',ready.get('source_sha256'))
    assert sources, 'Package source map required'
    for name, expected in sources.items():
        assert sha(package/name)==expected, name


def settings(source):
    init=node(source,'__init__','FeatureCellBirthDeath')
    return next(n for n in ast.walk(init) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Attribute) and t.attr=='settings' for t in n.targets))


def last_diagnostics(source):
    method=node(source,'maybe_apply','FeatureCellBirthDeath')
    return next(n for n in ast.walk(method) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='last' for t in n.targets))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--count-package',type=Path,required=True)
    parser.add_argument('--count-ready',type=Path,required=True)
    parser.add_argument('--performance-package',type=Path,required=True)
    parser.add_argument('--performance-ready',type=Path,required=True)
    args=parser.parse_args()
    assert not OUT.exists(), 'Refusing to replace a candidate'
    count_ready=json.loads(args.count_ready.read_text())
    perf_ready=json.loads(args.performance_ready.read_text())
    axis_ready_path=AXIS.parent/'AXIS-ID-READY.json'
    verify_package(AXIS,json.loads(axis_ready_path.read_text()))
    verify_package(args.count_package,count_ready)
    verify_package(args.performance_package,perf_ready)
    axis=(AXIS/'particlegan/feature_cells.py').read_text()
    count=(args.count_package/'particlegan/feature_cells.py').read_text()
    perf=(args.performance_package/'particlegan/feature_cells.py').read_text()
    # Count regions and planning replace the snapshot only. Copy topology,
    # sampler and canonical trainer API are inherited from the AXIS package.
    source=replace(axis,node(axis,'FeatureCellSnapshot'),extract(count,node(count,'FeatureCellSnapshot')))
    count_options={k.arg:k for k in settings(count).value.keywords
                   if k.arg=='mass_policy' or k.arg.startswith('count_')}
    setting_node=settings(source)
    existing={k.arg:k for k in setting_node.value.keywords}
    for name, keyword in count_options.items():
        if name in existing:
            setting_node.value.keywords[setting_node.value.keywords.index(existing[name])]=keyword
        else:
            setting_node.value.keywords.append(keyword)
    # The assignment is inside an __init__ block rather than at module level.
    source=replace(source,settings(source),'        '+ast.unparse(setting_node)+'\n')
    diagnostic_node=last_diagnostics(source)
    count_diagnostics={k.arg:k for k in last_diagnostics(count).value.keywords
        if k.arg.startswith('count_') or k.arg in
        ('ordinary_death_policy','mass_moves','support_moves','global_moves','count_cutoff',
         'ordinary_mass_moves','ordinary_support_moves','ordinary_global_moves')}
    original_keys={k.arg for k in diagnostic_node.value.keywords}
    for name,keyword in count_diagnostics.items():
        assert name not in original_keys, name
        diagnostic_node.value.keywords.append(keyword)
    source=replace(source,last_diagnostics(source),'            '+ast.unparse(diagnostic_node)+'\n')
    proofs=[]
    for splice in perf_ready['ast_splices']:
        owner=splice.get('owner')
        old=node(source,splice['name'],owner)
        new=node(perf,splice['name'],owner)
        assert new is not None
        assert ast_sha(new)==splice['proposal_ast_sha256']
        if old is None:
            assert owner is None and splice['base_ast_sha256'] is None
            snapshot=node(source,'FeatureCellSnapshot')
            source=replace(source,snapshot,extract(perf,new)+'\n\n'+extract(source,snapshot))
        else:
            assert ast_sha(old)==splice['base_ast_sha256'], splice['name']
            source=replace(source,old,extract(perf,new))
        assert ast_sha(node(source,splice['name'],owner))==ast_sha(new)
        proofs.append(splice)
    compile(source,str(OUT/'particlegan/feature_cells.py'),'exec')
    for name in ('LatentLineage','BoundedLatentGeometry'):
        assert ast_sha(node(source,name))==ast_sha(node(axis,name))
    changed_methods={s['name'] for s in proofs if s.get('owner')=='FeatureCellSnapshot'}
    count_class=node(count,'FeatureCellSnapshot')
    for member in count_class.body:
        if isinstance(member,ast.FunctionDef) and member.name not in changed_methods:
            assert ast_sha(node(source,member.name,'FeatureCellSnapshot'))==ast_sha(member),member.name
    composed_settings={k.arg:ast.dump(k.value,include_attributes=False) for k in settings(source).value.keywords}
    assert all(composed_settings[name]==ast.dump(value.value,include_attributes=False)
               for name,value in count_options.items())
    shutil.copytree(AXIS,OUT,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (OUT/'particlegan/feature_cells.py').write_text(source)
    config=ROOT/'configs/overrides-CB64-RA4.json'
    config.write_bytes((ROOT/'configs/overrides-CB64-RA2.json').read_bytes())
    HERE.mkdir(parents=True,exist_ok=True)
    proof=dict(status='COMPOSED_FOR_MERGED_CONTRACTS',candidate=str(OUT),
        axis_ready_sha256=sha(axis_ready_path),count_ready_sha256=sha(args.count_ready),
        performance_ready_sha256=sha(args.performance_ready),
        lineage_classes_ast_exact=True,training_source_exact=sha(OUT/'particlegan/training.py')==sha(AXIS/'particlegan/training.py'),
        count_methods_ast_exact_except_proven_splices=True,count_settings_ast_exact=True,
        count_diagnostic_fields=list(count_diagnostics),
        performance_splices=proofs,config_unchanged=True,quality_execution_started=False,
        package_source_sha256={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*.py'))})
    (HERE/'COMPOSITION.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if k!='package_source_sha256' and k!='performance_splices'}))


if __name__=='__main__':
    main()
