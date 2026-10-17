"""Bind the unchanged frozen equations to saved RA8 artifacts only."""
from pathlib import Path
import hashlib
import importlib.util
import json
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
ORIGINAL = ROOT/'performance/sampler-regression/cpu-plan-review/post-ra7-quality/grid-saved-diagnostic/diagnose.py'
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
ready=json.loads((HERE/'PREPARATION.json').read_text())
for path,digest in ready['source_and_input_sha256'].items():
    assert sha(path)==digest,path
assert sha(ORIGINAL)=='ab083335ea1df0a2eddd9fe574e657484da8795dfb74dc53377a3584e1ab0a05'
spec=importlib.util.spec_from_file_location('frozen_ra4_grid_equations_bound_ra8',ORIGINAL)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.HERE=HERE
module.RUN=ROOT/'validation-cb64-ra8/screens/runs/grid100'


def default(value):
    if isinstance(value,module.torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f'unsupported JSON context: {type(value).__name__}')


def dumps(value,**kwargs):
    return json.dumps(value,default=default,**kwargs)


module.json=SimpleNamespace(loads=json.loads,dumps=dumps)
module.main()
for path,digest in ready['source_and_input_sha256'].items():
    assert sha(path)==digest,path
