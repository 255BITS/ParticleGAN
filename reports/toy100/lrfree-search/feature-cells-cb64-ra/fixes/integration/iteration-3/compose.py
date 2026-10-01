"""Compose independently tested lineage and ordinary accounting corrections."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
LINEAGE = ROOT / 'geometry/training-regression/pkg-LINEAGE'
COUNT = ROOT / 'integration/review/training-regression/pkg-count-recovery'
OUT = ROOT / 'pkg-CB64-RA3'
HERE = Path(__file__).resolve().parent
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def method(source, cls, name):
    owner = next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name==cls)
    return next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name==name)


def definition(source, name):
    return next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name==name)


def main():
    assert not OUT.exists(), 'Refusing to overwrite a candidate.'
    assert sha(LINEAGE/'particlegan/feature_cells.py') == '23c70cd51c7171a272f1b21fa2310c7a15a670b5bf9777c8029af55a5ba80622'
    assert sha(LINEAGE/'particlegan/training.py') == '7edd9cb522357c49021719f8ce63e2cc9eec3a3b41c0f89219eab172e13b1c58'
    ready = json.loads((COUNT.parent/'READY.json').read_text())
    for name, expected in ready['package_source_sha256'].items():
        assert sha(COUNT/name) == expected, name
    source = (LINEAGE/'particlegan/feature_cells.py').read_text()
    count = (COUNT/'particlegan/feature_cells.py').read_text()
    old = method(source, 'FeatureCellSnapshot', 'ordinary_transport')
    new = method(count, 'FeatureCellSnapshot', 'ordinary_transport')
    lines, additions = source.splitlines(keepends=True), count.splitlines(keepends=True)
    lines[old.lineno-1:old.end_lineno] = additions[new.lineno-1:new.end_lineno]
    composed = ''.join(lines)
    policy = 'reference_topology_vacancies_unique_parents_v3'
    assert composed.count(policy) == 1
    composed = composed.replace(policy, 'reference_topology_vacancies_unique_parents_v4')
    compile(composed, 'composed-feature-cells.py', 'exec')
    compare = lambda n: ast.dump(n,include_attributes=False)
    assert compare(method(composed,'FeatureCellSnapshot','ordinary_transport')) == compare(new)
    for name in ('LatentLineage','BoundedLatentGeometry'):
        assert compare(definition(composed,name)) == compare(definition(source,name))
    shutil.copytree(LINEAGE, OUT, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (OUT/'particlegan/feature_cells.py').write_text(composed)
    (ROOT/'configs/overrides-CB64-RA3.json').write_bytes((ROOT/'configs/overrides-CB64-RA2.json').read_bytes())
    proof = dict(status='COMPOSED_FOR_FOCUSED_REVIEW', candidate=str(OUT),
                 lineage_sources={name:sha(LINEAGE/'particlegan'/name) for name in ('feature_cells.py','training.py')},
                 count_ready_sha256=sha(COUNT.parent/'READY.json'),
                 count_method_ast_exact=True,lineage_classes_ast_exact=True,
                 training_source_exact=sha(OUT/'particlegan/training.py')==sha(LINEAGE/'particlegan/training.py'),
                 config_unchanged=True, quality_execution_started=False,
                 composed_source_sha256={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*.py'))})
    (HERE/'COMPOSITION.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({k:proof[k] for k in ('status','candidate','count_method_ast_exact','lineage_classes_ast_exact','training_source_exact')}))


if __name__ == '__main__':
    main()
