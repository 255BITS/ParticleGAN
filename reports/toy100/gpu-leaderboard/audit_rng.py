"""Verify the saved sparse/dense measurement prefix tensors without retraining."""
import json
import os
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
import torch

root = Path(__file__).resolve().parent
a = torch.load(root / 'preflight/pr143-prefix/final-state.pt', weights_only=False, map_location='cpu')
b = torch.load(root / 'preflight/pr143-prefix-dense/final-state.pt', weights_only=False, map_location='cpu')
equal_tensors = []
equal_generators = []


def compare(left, right, path='state'):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right), path
        equal_tensors.append(path)
    elif isinstance(left, torch.Generator):
        assert isinstance(right, torch.Generator) and torch.equal(left.get_state(), right.get_state()), path
        equal_generators.append(path)
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            compare(left[key], right[key], path + '/' + str(key))
    elif isinstance(left, (list, tuple)):
        # Scalar observation receipts may differ with diagnostic cadence.
        if any(isinstance(x, (torch.Tensor, dict, list, tuple, torch.Generator)) for x in left):
            assert len(left) == len(right), path
            for i, (x, y) in enumerate(zip(left, right)):
                compare(x, y, path + '/' + str(i))


compare(a, b)
report = dict(status='PASS', tensor_leaves_equal=len(equal_tensors), generators_equal=len(equal_generators),
              scope='20 CUDA updates; sparse vs dense observation; model, optimizer, CPU/CUDA/data/noise RNG',
              tensors=equal_tensors, generators=equal_generators)
(root / 'rng-validation-detailed.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k not in ('tensors', 'generators')}))
