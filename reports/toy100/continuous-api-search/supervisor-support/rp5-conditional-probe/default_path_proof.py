"""CPU-only frozen-RP5 comparison; no optimizer or training execution."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "snapshot"))
from particlegan.precision import ReversiblePrecision

baseline = json.loads((ROOT / "baseline.json").read_text())
frozen = zipfile.ZipFile(baseline["source"]).read("particlegan/precision.py")
assert hashlib.sha256(frozen).hexdigest() == baseline["package_sha256"]["particlegan/precision.py"]
namespace = {}
exec(compile(frozen, "immutable-rp5/particlegan/precision.py", "exec"), namespace)
OriginalPrecision = namespace["ReversiblePrecision"]


def fingerprint(value):
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous()
        return [str(array.dtype), list(array.shape),
                hashlib.sha256(array.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()]
    if isinstance(value, (tuple, list)):
        return [fingerprint(v) for v in value]
    if isinstance(value, dict):
        return {k: fingerprint(v) for k, v in value.items()}
    return value


class Operations(TorchDispatchMode):
    def __init__(self):
        self.trace = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        value = func(*args, **(kwargs or {}))
        self.trace.append([str(func), fingerprint(value)])
        return value


critic = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1)).double()
with torch.no_grad():
    for i, parameter in enumerate(critic.parameters()):
        parameter.fill_((i + 1) / 10)
old = OriginalPrecision(critic, variant="rp5")
new = ReversiblePrecision(critic, variant="rp5")
observations = []
for i, activity in enumerate((.3, .2, .6)):
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.add_(.03)
    real = torch.tensor([[.1, -.2], [.6, .8]], dtype=torch.float64) + i / 10
    a, b = Operations(), Operations()
    with a:
        old.observe(critic, real, activity)
    with b:
        new.observe(critic, real, activity)
    assert a.trace == b.trace
    assert fingerprint(old.state_dict()) == fingerprint(new.state_dict())
    observations.append({"observation": i + 1, "aten_operations": len(a.trace),
                         "operation_names_and_output_bytes_equal": True,
                         "full_controller_and_reference_state_equal": True,
                         "trace_sha256": hashlib.sha256(json.dumps(a.trace, sort_keys=True).encode()).hexdigest()})

result = {"status": "PASS", "scope": "Three tiny CPU observations of frozen and proposed default paths; no training or optimizer steps",
          "torch": str(torch.__version__), "python": sys.version.split()[0],
          "baseline_precision_sha256": hashlib.sha256(frozen).hexdigest(),
          "observations": observations}
(ROOT / "default-path-proof.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
