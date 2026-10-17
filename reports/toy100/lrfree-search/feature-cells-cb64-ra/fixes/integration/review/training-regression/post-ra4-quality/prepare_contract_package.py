"""Prepare a private RA4 copy with only the required isolation ledger API."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
base = ROOT / 'pkg-CB64-RA4'
target = HERE / 'pkg-ANCHOR-CONTRACT'
if target.exists():
    raise SystemExit('private contract package exists')
shutil.copytree(base, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
path = target / 'particlegan/feature_cells.py'
source = path.read_text()
tree = ast.parse(source)
method = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == 'FeatureCellSnapshot'
    for n in c.body if isinstance(n, ast.FunctionDef) and n.name == 'select_parents')
old = '\n'.join(source.splitlines()[method.lineno - 2:method.end_lineno])
new = old.replace('generator, pvalues=None):', 'generator, pvalues=None, supported_counts=None, reserved_rows=None):', 1)
new = new.replace('        eligible = ~flags & (pvalues>Q)\n',
    '        eligible = ~flags & (pvalues>Q)\n'
    '        if reserved_rows is not None:\n'
    '            eligible[reserved_rows] = False\n'
    '            allowed = torch.ones(len(features),device=self.device,dtype=torch.bool)\n'
    '            allowed[reserved_rows] = False\n'
    '            dead = dead[allowed[dead]]\n', 1)
new = new.replace('        if ordinary_parents is not None:\n            supported_children =',
    '        if supported_counts is not None:\n'
    '            if (supported_counts.shape != (self.cells,) or supported_counts.device != self.device\n'
    '                    or supported_counts.dtype != torch.long or bool((supported_counts<0).any())):\n'
    '                raise ValueError("invalid explicit isolation supported ledger")\n'
    '            kept = supported_counts.clone()\n'
    '        elif ordinary_parents is not None:\n            supported_children =', 1)
if old == new or 'supported_counts.clone()' not in new or 'dead = dead[allowed[dead]]' not in new:
    raise RuntimeError('isolation API splice failed')
source = source.replace(old, new, 1)
path.write_text(source)
for name in ('anchor_birth.py', 'birth_phase.py'):
    shutil.copyfile(HERE / name, target / 'particlegan' / name)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
receipt = dict(scope='private contract package, frozen RA4 unchanged; only isolation API extension plus novel helper modules',
    base_source_sha256={str(p.relative_to(base)): sha(p) for p in sorted(base.rglob('*.py'))},
    package_source_sha256={str(p.relative_to(target)): sha(p) for p in sorted(target.rglob('*.py'))},
    local_helper_sha256={name: sha(HERE / name) for name in ('anchor_birth.py', 'birth_phase.py')})
(HERE / 'prepare-contract-package.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(dict(event='private_contract_package_prepared',path=str(target),feature_cells_sha256=sha(path))),flush=True)
