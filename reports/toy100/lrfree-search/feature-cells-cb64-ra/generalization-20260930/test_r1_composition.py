"""Focused CPU checks before the unchanged moving-target quality test."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'pkg-RA11-R1'))
from particlegan import GANTrainer, get_recipe
from particlegan.continuous import OptimizerSurprise

torch.set_num_threads(1)


def trainer(reopen=True):
    torch.manual_seed(0)
    config = json.loads((ROOT / 'configs/RA11-R1-historical-rates.json').read_text())
    config.update(num_particles=32, z_dim=2, batch_size=16, initialization=None)
    if not reopen:
        config.update(reopen_signal='none', reopen_anchor='hold')
    return GANTrainer(get_recipe(**config),
        nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(.2), nn.Linear(16, 2)),
        nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(.2), nn.Linear(16, 1)),
        seed=0, serial_backward=True, optimizer_options={'foreach': False, 'fused': False})


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(
            a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
            b.detach().cpu().contiguous().reshape(-1).view(torch.uint8))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) == type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b or (isinstance(a, float) and isinstance(b, float) and a != a and b != b)


def differences(a, b, path='root'):
    if same(a, b):
        return []
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        return [p for k in a for p in differences(a[k], b[k], path + '.' + str(k))]
    if isinstance(a, (list, tuple)) and isinstance(b, type(a)) and len(a) == len(b):
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in differences(x, y, path + '.' + str(i))]
    return [path]


def test_detector_jump_ramp_and_state():
    import math
    detector = OptimizerSurprise()
    fires = []
    for step, value in enumerate([1.] * 200 + [5.] * 100):
        detector.pending = {'0.0': torch.tensor(value)}
        if detector.decide(step):
            fires.append(step)
    assert len(fires) == 1 and 212 < fires[0] <= 224
    restored = OptimizerSurprise()
    restored.load_state_dict(detector.state_dict())
    assert same(restored.state_dict(), detector.state_dict())
    ramp = OptimizerSurprise()
    for step, value in enumerate([1.] * 200 + [math.exp(math.log(5.) * min(1., i/400)) for i in range(600)]):
        ramp.pending = {'0.0': torch.tensor(value)}
        assert not ramp.decide(step)


def test_training_observations_reopen_and_exact_replay():
    active = trainer()
    batches = [torch.randn(16, 2) for _ in range(20)]
    for real in batches[:5]:
        active.step(real)
    assert active.surprise.pending
    # Isolate trainer reaction wiring from the detector's already-tested
    # gradient-shock detection. A fixed pending shock must reopen all ladders.
    active.surprise.fast = {'0.0': 0.}
    active.surprise.slow = {'0.0': 0.}
    active.surprise.pending = {'0.0': torch.tensor(1e6)}
    active.surprise.streak = OptimizerSurprise.K - 1
    active.surprise.since_calm = 0
    active.step(batches[5])
    assert active.surprise.fires == 1
    assert any(t.counts.get('reopens', 0) for row in active.lr_settle.testers for t in row if t is not None)
    state = active.state_dict()
    uninterrupted = [active.step(real) for real in batches[6:12]]
    endpoint = active.state_dict()
    resumed = trainer()
    resumed.load_state_dict(state)
    continued = [resumed.step(real) for real in batches[6:12]]
    assert same(uninterrupted, continued)
    assert same(endpoint, resumed.state_dict()), differences(endpoint, resumed.state_dict())


def test_disabled_recipe_and_checkpoint_have_no_new_semantic_fields():
    disabled = trainer(False)
    assert disabled.surprise is None
    assert 'surprise' not in disabled.state_dict()
    assert 'reopen_anchor' not in disabled.recipe.to_dict()
