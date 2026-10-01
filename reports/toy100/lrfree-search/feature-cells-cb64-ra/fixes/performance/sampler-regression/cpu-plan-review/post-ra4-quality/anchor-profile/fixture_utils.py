"""Saved-input helpers; no random model initialization or training."""
from copy import deepcopy
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_package(root, alias):
    package = ModuleType(alias)
    package.__path__ = [str(Path(root) / 'particlegan')]
    sys.modules[alias] = package
    return importlib.import_module(alias + '.feature_cells'), importlib.import_module(alias + '.birth_phase')


def import_file(path, alias):
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def network(weights):
    layers = []
    for i in (0, 2, 4):
        weight, bias = weights[f'{i}.weight'], weights[f'{i}.bias']
        layer = torch.nn.Linear.__new__(torch.nn.Linear)
        torch.nn.Module.__init__(layer)
        layer.in_features, layer.out_features = weight.shape[1], weight.shape[0]
        layer.weight = torch.nn.Parameter(weight.detach().clone())
        layer.bias = torch.nn.Parameter(bias.detach().clone())
        layers.append(layer)
        if i != 4:
            layers.append(torch.nn.LeakyReLU(.2))
    return torch.nn.Sequential(*layers)


def convert(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device).clone()
    if isinstance(value, torch.device):
        return torch.device(device)
    if isinstance(value, dict):
        return {k: convert(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [convert(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(convert(v, device) for v in value)
    return deepcopy(value)


def nested_equal(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(nested_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def output_hash(value):
    result = hashlib.sha256()
    def visit(part):
        if isinstance(part, torch.Tensor):
            result.update(str((part.dtype, tuple(part.shape))).encode())
            result.update(part.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(part, dict):
            for key in sorted(part):
                result.update(key.encode())
                visit(part[key])
        elif isinstance(part, (tuple, list)):
            for item in part:
                visit(item)
        else:
            result.update(repr(part).encode())
    visit(value)
    return result.hexdigest()
