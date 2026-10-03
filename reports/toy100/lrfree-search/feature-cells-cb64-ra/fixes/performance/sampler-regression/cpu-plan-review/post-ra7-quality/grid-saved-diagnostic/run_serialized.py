"""Output-only adapter for saved scalar Tensor context; frozen math unchanged."""
from pathlib import Path
import hashlib
import importlib.util
import json
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
ready = json.loads((HERE / 'SERIALIZATION-READY.json').read_text())
for path, digest in ready['source_sha256'].items():
    assert sha(path) == digest, path
spec = importlib.util.spec_from_file_location('frozen_saved_grid_diagnostic', HERE / 'diagnose.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def default(value):
    if isinstance(value, module.torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f'unsupported JSON context: {type(value).__name__}')


def dumps(value, **kwargs):
    return json.dumps(value, default=default, **kwargs)


module.json = SimpleNamespace(loads=json.loads, dumps=dumps)
module.main()
for path, digest in ready['source_sha256'].items():
    assert sha(path) == digest, path
