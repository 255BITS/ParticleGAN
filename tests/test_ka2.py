"""KA2 promotion: exact frozen parity, independent critics and checkpoint state."""
import copy
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ka2_scenarios as sc  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DRIVER = r'''
import importlib.util, sys
from pathlib import Path
repo, output, lazy = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
sys.path[:0] = [str(repo), str(repo / "tests")]
import benchmarks.legacy.grad_regularizers as legacy
sys.modules["particlegan.grad_regularizers"] = legacy
import torch
torch.set_num_threads(1)
import ka2_scenarios as sc
spec = importlib.util.spec_from_file_location("frozen_ka2", repo / sc.FROZEN_SOURCE)
mechanism = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mechanism)
torch.save(sc.frozen(mechanism, lazy), output)
'''


@pytest.fixture(scope="module", params=[1, 2], ids=["every_step", "lazy2"])
def frozen(request, tmp_path_factory):
    source = REPO / sc.FROZEN_SOURCE
    assert hashlib.sha256(source.read_bytes()).hexdigest() == sc.FROZEN_SHA256
    tmp = tmp_path_factory.mktemp("frozen_ka2")
    driver, result = tmp / "driver.py", tmp / "trace.pt"
    driver.write_text(DRIVER)
    subprocess.run([sys.executable, str(driver), str(REPO), str(result), str(request.param)],
                   cwd=tmp, check=True)
    return request.param, torch.load(result, weights_only=False)


def assert_trace_equal(expected, actual):
    assert len(expected) == len(actual)
    for reference, new in zip(expected, actual):
        assert reference.keys() == new.keys()
        for key in reference:
            left, right = reference[key], new[key]
            if key in ("params", "raw_grads", "ema") and left is not None:
                assert all(torch.equal(a, b) for a, b in zip(left, right)), (reference["t"], key)
            elif key == "penalty":
                assert torch.equal(left, right), (reference["t"], key)
            elif key == "control" and reference["calls"] == 0:
                # The research global hook first discovers its optimizer on
                # the first *applied* call. The package owns it from creation.
                assert left["last_sur"] is None
                assert {k: v for k, v in left.items() if k != "last_sur"} == {
                    k: v for k, v in right.items() if k != "last_sur"}
            else:
                assert left == right, (reference["t"], key, left, right)


def test_default_matches_exact_frozen_ka2_through_release_reseed_and_reanchor(frozen):
    lazy_k, reference = frozen
    objects = sc.build(lazy_k)
    trace = sc.package_trace(objects, range(1, sc.CALLS * lazy_k + 1), lazy_k)
    assert_trace_equal(reference["trace"], trace)
    record = objects[1].record
    assert reference["first_blend"] == 800
    assert all(row["ema"] is None for row in trace[:800 * lazy_k - 1])
    assert trace[800 * lazy_k - 1]["ema"] is not None
    assert objects[1].guard.clipped_tensors == reference["clipped"] > 0
    assert record.observed_steps == sc.CALLS * lazy_k
    assert record.calls == sc.CALLS and len(record.sur_hist) == 400
    assert record.ema_reseeds >= 1 and record.ema_updates > 0 and record.ema_skips > 0
    assert any(row["control"]["w"] == 0 for row in trace)
    assert trace[-1]["control"]["w"] == 1 and max(r["control"]["alpha"] for r in trace) > .5
    assert objects[1].param_groups[0]["lr"] == sc.LR  # transitions at constant LR
    if lazy_k == 2:
        assert all(not row["applied"] and float(row["penalty"]) == 0 for row in trace[::2])
        assert any(b["ema_updates"] > a["ema_updates"] for a, b in zip(trace, trace[1:])
                   if not b["applied"])  # EMA advances on skipped penalty steps too.


def test_optimizer_checkpoint_continues_active_history_bit_exactly(frozen):
    lazy_k, reference = frozen
    split = 900 * lazy_k  # released, alpha active, baseline and history established
    first = sc.build(lazy_k)
    part = sc.package_trace(first, range(1, split + 1), lazy_k)
    checkpoint = copy.deepcopy(dict(critic=first[0].state_dict(), optimizer=first[1].state_dict()))
    assert first[1].record.w == 0 and first[1].record.alpha > 0 and first[1].record.sur_base is not None
    resumed = sc.build(lazy_k)
    resumed[0].load_state_dict(checkpoint["critic"])
    resumed[1].load_state_dict(checkpoint["optimizer"])
    rest = sc.package_trace(resumed, range(split + 1, sc.CALLS * lazy_k + 1), lazy_k)
    assert_trace_equal(reference["trace"], part + rest)
    assert resumed[1].guard.clipped_tensors == reference["clipped"]


def test_surprise_uses_post_adam_moments_and_upper_median():
    _, optimizer, _ = sc.build()
    for i, p in enumerate(optimizer.critic.parameters(), start=1):
        optimizer.state[p].update(step=torch.tensor(1.), exp_avg=torch.zeros_like(p),
                                  exp_avg_sq=torch.full_like(p, .001))
        p.grad = torch.full_like(p, float(i))
    optimizer.record.record_step(optimizer)
    expected = float(torch.tensor(3., dtype=sc.DT) /
                     (torch.tensor(.001, dtype=sc.DT) / (1 - .999 ** torch.tensor(1.))).sqrt())
    assert optimizer.record.last_sur == expected  # upper median of 1, 2, 3, 4, not 2.5


def test_two_default_critics_own_their_controller_state_without_global_hooks(monkeypatch):
    import particlegan.ka2 as ka2
    from torch.optim import optimizer as optimizer_module

    monkeypatch.setattr(ka2, "WARMUP_CALLS", 4)
    hooks = (len(optimizer_module._global_optimizer_pre_hooks), len(optimizer_module._global_optimizer_post_hooks))
    standalone = sc.build()
    expected = sc.package_trace(standalone, range(1, 50))
    a, b = sc.build(), sc.build()
    actual = []
    for t in range(1, 50):
        actual += sc.package_trace(a, [t])
        real, fake = sc.batch(t + 17)
        b[1].zero_grad(set_to_none=True)
        (b[2](b[0], real, fake) + b[0](fake).square().mean()).backward()
        b[1].step()
    assert_trace_equal(expected, actual)
    assert a[1].record is not b[1].record
    assert a[1].record.sur_hist != b[1].record.sur_hist
    assert hooks == (len(optimizer_module._global_optimizer_pre_hooks),
                     len(optimizer_module._global_optimizer_post_hooks))


class ConditionalCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2, 4), nn.BatchNorm1d(4), nn.Tanh(), nn.Linear(4, 2)).double()

    def forward(self, x, labels, *, shift):
        logits = self.network(x + shift)
        return logits.gather(1, labels[:, None]), logits


def test_conditional_critic_and_ema_receive_identical_conditioning(monkeypatch):
    import particlegan.ka2 as ka2
    from particlegan import get_recipe

    monkeypatch.setattr(ka2, "WARMUP_CALLS", 3)
    critic = ConditionalCritic()
    recipe = get_recipe()
    optimizer = recipe.make_critic_optimizer(critic, ema_critic=copy.deepcopy(critic))
    penalty = recipe.make_critic_penalty(optimizer, collect_stats=True)
    labels, shift = torch.tensor([0, 1, 1, 0]), torch.tensor([.4, -.2], dtype=sc.DT)
    seen = []
    hook = optimizer.ema_critic.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append((args[1].clone(), kwargs["shift"].clone())), with_kwargs=True)
    for t in range(1, 6):
        real, fake = sc.batch(t)
        before = {key: value.clone() for key, value in optimizer.ema_critic.named_buffers()}
        started = optimizer.record.anchor_started
        optimizer.zero_grad(set_to_none=True)
        value = penalty(critic, real, fake, labels, shift=shift)
        if started:
            assert all(torch.equal(before[k], v) for k, v in optimizer.ema_critic.named_buffers())
        (value + critic(fake, labels, shift=shift)[0].mean()).backward()
        optimizer.step()
    hook.remove()
    assert penalty.last_stats["phase"] == "blend" and optimizer.record.calls == 5
    assert seen and all(torch.equal(y, labels) and torch.equal(s, shift) for y, s in seen)


class FourierCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.3, dtype=torch.float32))
        self.register_buffer("freqs", torch.pi * (2.0 ** torch.arange(3, dtype=torch.float32)))

    def forward(self, x):
        return torch.sin(x * self.freqs).sum(-1) * self.weight


def test_adaptive_ema_keeps_fixed_fourier_buffers_and_anchor_gradients_exact():
    from particlegan import get_recipe
    from particlegan.k3p import CriticAnchor

    critic = FourierCritic()
    optimizer = get_recipe().make_critic_optimizer(critic, ema_critic=copy.deepcopy(critic))
    # The frozen mechanism averages parameters only; structural Fourier
    # frequencies stay equal to the live critic. Compare that exact reference.
    reference = CriticAnchor(critic, copy.deepcopy(critic))
    optimizer.anchor.start_()
    reference.start_()
    record = optimizer.record
    record.anchor_started = True
    record.sur_base = 1.0
    record.sur_hist = [4.0] * 24
    record.last_sur = 4.0
    decay_trace = []
    for step in range(8):
        if step == 4:
            # A settled window makes the real KA2 controller release alpha.
            record.sur_hist = [.5] * 24
            record.last_sur = .5
        record.advance_blend()
        record.record_step(optimizer)
        reference.decay = optimizer.anchor.decay
        reference.update_()
        decay_trace.append(reference.decay)
        assert torch.equal(optimizer.ema_critic.freqs, critic.freqs)
        assert all(torch.equal(a, b) for a, b in zip(
            optimizer.ema_critic.parameters(), reference.ema_critic.parameters()))
        x = torch.tensor([[.21], [.71], [1.31], [2.72]], requires_grad=True)
        expected, actual = reference(x), optimizer.anchor(x)
        assert torch.equal(actual, expected)
        assert torch.equal(torch.autograd.grad(actual.sum(), x)[0],
                           torch.autograd.grad(expected.sum(), x)[0])
    assert decay_trace[3] < decay_trace[0] and decay_trace[4] > decay_trace[3]
    assert record.ema_updates == 8


def test_buffer_ema_still_averages_changing_batchnorm_statistics_and_copies_counts():
    from particlegan.k3p import RobustCriticAnchor

    critic = nn.BatchNorm1d(3)
    ema = copy.deepcopy(critic)
    anchor = RobustCriticAnchor(critic, ema)
    anchor.start_()
    for step, decay in enumerate((.999, .9934981558641975, .97, .91), start=1):
        before_mean, before_var = ema.running_mean.clone(), ema.running_var.clone()
        with torch.no_grad():
            critic.running_mean.copy_(torch.tensor([.1, .7, -1.2]) * step)
            critic.running_var.copy_(torch.tensor([.3, 1.8, 2.1]) * step)
            critic.num_batches_tracked.fill_(step)
        anchor.decay = decay
        anchor.update_()
        torch.testing.assert_close(ema.running_mean,
                                   before_mean * decay + critic.running_mean * (1.0 - decay))
        torch.testing.assert_close(ema.running_var,
                                   before_var * decay + critic.running_var * (1.0 - decay))
        assert not torch.equal(ema.running_mean, critic.running_mean)
        assert not torch.equal(ema.running_var, critic.running_var)
        assert torch.equal(ema.num_batches_tracked, critic.num_batches_tracked)
