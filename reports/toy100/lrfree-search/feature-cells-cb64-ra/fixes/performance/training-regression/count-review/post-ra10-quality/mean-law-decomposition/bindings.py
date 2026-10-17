"""Raw-byte bindings for one final RA10 descriptive diagnostic; stdlib only."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
RUN = ROOT / 'validation-cb64-ra10/screens/runs/grid100'
PACKAGE = ROOT / 'pkg-CB64-RA10'
PLAN = HERE.parent / 'mean-law-source-plan'
GRID_AUDIT = HERE.parent / 'grid-canonical-review'
HEAD = ROOT / 'performance/training-regression/count-review/post-ra8-quality/grid-covariance/diagnose.py'
AFFINE = ROOT / 'integration/review/training-regression/post-ra9-quality/mean-category-transport/run_prototype.py'
STATE_HASH = ROOT / 'integration/review/training-regression/post-ra4-quality/measure_saved_utils.py'
SCREEN = Path('/ml2/hypergan/lrfree-20260926/harness/screen.py')
EXPECTED = {
    PLAN / 'FROZEN.json': '744dc8717f010ba4de7de494d6ad78e8b168f111ed8dfbc743b3ffe4868e7733',
    PLAN / 'DESIGN.md': '8067dff5cb789c8b2135398a3a04a2540aafc8c3a82f089afb423bbb39f5c6bf',
    GRID_AUDIT / 'FROZEN.json': 'dadcb05b464192bd0b7eb859dfa565387922f9847eca90cea4657419d8e3ac0f',
    ROOT / 'quality/ra10/READY.json': '9f29f98761b136055ef5bfce3697de0cd45cb8d12c2d65fee01e5f5281edf6fd',
    ROOT / 'validation-cb64-ra10/source-freeze.json': '4cd1be039f2d6fb9bd12fe3d93bae60ca331bb68d77d365c9eadb4019bd8b856',
    HEAD: '1634bda4c9e94b8e752b347dfca3e3a02740f5c82653fe1d5ad0a59980ddf79e',
    AFFINE: '86e61f4bacd69f38efc538013139ddf03c1ab5dd779cf394b0bb95f2b145e0b5',
    STATE_HASH: '9f6b4dc2517b642ced51c2c0cb197cff3e1fd4d42efcf03ae8e40e1b2e3077f2',
    SCREEN: 'ee8193adbdf09e93511befae7b6491143c26de88612eddf065cbb92eb2153c3c',
}
NUMERIC_INPUTS = (RUN / 'final-state.pt', RUN / 'native-clean/holdout_samples.npz',
                  RUN / 'native-noisy/holdout_samples.npz')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def merge_maps(*maps):
    result = {}
    for mapping in maps:
        for path, digest in mapping.items():
            if path in result and result[path] != digest:
                raise ValueError('conflicting immutable binding: ' + path)
            result[path] = digest
    return result


def verify(mapping):
    for path, digest in mapping.items():
        if sha(path) != digest:
            raise ValueError('raw-byte guard mismatch: ' + str(path))


def function_node(path, name, affine_only=False):
    nodes = [node for node in ast.walk(ast.parse(Path(path).read_text()))
             if isinstance(node, ast.FunctionDef) and node.name == name]
    if affine_only:
        nodes = [node for node in nodes if any(isinstance(child, ast.Call)
                 and isinstance(child.func, ast.Attribute)
                 and isinstance(child.func.value, ast.Name)
                 and child.func.value.id == 'F' and child.func.attr == 'linear'
                 for child in ast.walk(node))]
    if len(nodes) != 1:
        raise ValueError('ambiguous immutable function: ' + name)
    return nodes[0]


def ast_sha(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
