"""Separate exact integer group-sum proposal; frozen RA6 is read-only."""
import ast
import difflib
import json
from pathlib import Path
import shutil
import hashlib
import textwrap

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'pkg-CB64-RA6'
TARGET = HERE / 'pkg-GROUP-COUNT'


def method(source):
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'FeatureCellSnapshot')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_group_counts')
    return '\n'.join(source.splitlines()[node.lineno - 1:node.end_lineno])


def main():
    assert not TARGET.exists()
    original = (BASE / 'particlegan/feature_cells.py').read_text()
    old = method(original)
    new = '''    def _group_counts(self, counts):
        groups = self._mass_topology()
        if counts.dtype in (torch.int32, torch.int64) and counts.ndim == 1 and len(counts) == len(groups):
            members = groups[None] == torch.arange(self.mass_groups,device=groups.device)[:,None]
            # Integer sums retain exact quotas; the matrix is bounded by K*K.
            return counts[None].expand(self.mass_groups,-1).masked_fill(~members,0).sum(1)
        return torch.stack([counts[groups==group].sum() for group in range(self.mass_groups)])'''
    assert original.count(old) == 1
    source = original.replace(old, new, 1)
    compile(source, str(TARGET / 'particlegan/feature_cells.py'), 'exec')
    shutil.copytree(BASE, TARGET, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (TARGET / 'particlegan/feature_cells.py').write_text(source)
    patch = ''.join(difflib.unified_diff(original.splitlines(True), source.splitlines(True),
        fromfile='a/particlegan/feature_cells.py', tofile='b/particlegan/feature_cells.py'))
    (HERE / 'GROUP-COUNT.patch').write_text(patch)
    sha = lambda text: hashlib.sha256(text.encode()).hexdigest()
    receipt = dict(status='PRIVATE_PROPOSAL_UNQUALIFIED',
        base_package=str(BASE), package_root=str(TARGET),
        ast_splices=[dict(kind='method', owner='FeatureCellSnapshot', name='_group_counts',
            base_ast_sha256=sha(ast.dump(ast.parse(textwrap.dedent(old)).body[0], include_attributes=False)),
            proposal_ast_sha256=sha(ast.dump(ast.parse(textwrap.dedent(new)).body[0], include_attributes=False)))],
        maximum_matrix_entries=4096, state_or_law_changes=False)
    (HERE / 'GROUP-PREPARATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(status=receipt['status'], package_root=str(TARGET))))


if __name__ == '__main__':
    main()
