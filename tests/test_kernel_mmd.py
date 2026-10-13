"""Bounded MMD software contracts; eight native updates, no qualification."""
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
from types import ModuleType

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, Recipe, get_recipe
from particlegan.init import deterministic_orthogonal_
from particlegan.kernel_mmd import DEFAULT_OUTPUT_MMD_BANDWIDTHS, OutputMMDU, kernel_mmd_u
from particlegan.recipe_compat import without_default_additions


ROOT = Path(__file__).resolve().parents[1]
WIDTHS = DEFAULT_OUTPUT_MMD_BANDWIDTHS
# Exact pre-MMD scientific blobs; assertions never compare to a moving HEAD.
OLD_TRAINING_BLOB = "dab0c31c684d09ff68eb4433be4c73f43f2db4c0"
OLD_RECIPES_BLOB = "2d7497c98ff6b2112d663b4d78fb306d8c4cae34"


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def scalar_kernel(x, y, widths=WIDTHS):
    squared = sum((a - b) ** 2 for a, b in zip(x, y))
    return sum(math.exp(-squared / (2 * width ** 2)) for width in widths) / len(widths)


def explicit_u(fake, real, widths=WIDTHS):
    n, m = len(fake), len(real)
    return (sum(scalar_kernel(a, b, widths) for i, a in enumerate(fake)
                for j, b in enumerate(fake) if i != j) / (n * (n - 1))
            + sum(scalar_kernel(a, b, widths) for i, a in enumerate(real)
                  for j, b in enumerate(real) if i != j) / (m * (m - 1))
            - 2 * sum(scalar_kernel(a, b, widths) for a in fake for b in real) / (n * m))


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def old_module(blob, name):
    source = subprocess.check_output(["git", "cat-file", "blob", blob], cwd=ROOT)
    module = ModuleType("particlegan." + name)
    module.__package__ = "particlegan"
    exec(compile(source, "archived-git-blob:" + blob, "exec"), module.__dict__)
    return module


def trainer(weight=1., *, energy=0., widths=WIDTHS, cls=GANTrainer):
    with torch.random.fork_rng(devices=[]):
        recipe = get_recipe("k3p", loss="non_saturating", reg_coeff=.1, reg_every=1,
            lr=.0085, betas=(0., .99), amsgrad=True, batch_size=4, z_dim=2,
            num_particles=8, total_steps=4, prior_kind="mog", sigma_rel=.025,
            standardize=False, input_noise_std=0., output_noise_std=0., prior_reg=0.,
            energy_distance_weight=energy, output_mmd_weight=weight, output_mmd_bandwidths=widths)
        g = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))
        d = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
        prior = recipe.make_prior(sigma=.025)
        for model in (g, d, prior):
            deterministic_orthogonal_(model, seed=0)
        return cls(recipe, g, d, prior=prior, seed=0)


PANEL = torch.tensor([[-1., -.5], [-.2, .1], [.4, .8], [1., 1.5]])


def test_fixed_mixture_matches_independent_scalar_u_with_unequal_panels():
    fake = torch.tensor([[-.7, .2], [.1, -.6], [.8, .4], [-.1, .9]], dtype=torch.float64)
    real = torch.tensor([[-.4, -.8], [.6, -.2], [.1, .5]], dtype=torch.float64)
    before = torch.get_rng_state().clone()
    assert kernel_mmd_u(fake, real).item() == pytest.approx(explicit_u(fake.tolist(), real.tolist()), abs=1e-15)
    assert torch.equal(before, torch.get_rng_state())
    torch.testing.assert_close(kernel_mmd_u(fake.reshape(4, 1, 2), real.reshape(3, 1, 2)),
                               kernel_mmd_u(fake, real))
    assert WIDTHS == (.0625, .125, .25, .5, 1., 2., 4.)


def test_both_fake_arguments_gradient_and_target_detachment():
    fake = torch.tensor([[-.73, .21], [.19, -.61], [.82, .48], [-.14, .92]],
                        dtype=torch.float64, requires_grad=True)
    real = torch.tensor([[-.42, -.83], [.64, -.19], [.08, .57]],
                        dtype=torch.float64, requires_grad=True)
    value = kernel_mmd_u(fake, real)
    gradient, target = torch.autograd.grad(value, (fake, real), allow_unused=True)
    assert target is None
    # Closed first derivative includes both ordered fake arguments explicitly.
    n, m = len(fake), len(real)
    expected = torch.zeros_like(fake)
    for i, x in enumerate(fake.detach()):
        for width in WIDTHS:
            for j, z in enumerate(fake.detach()):
                if i != j:
                    k = math.exp(-float((x - z).square().sum()) / (2 * width ** 2))
                    expected[i] -= 2 * k * (x - z) / (len(WIDTHS) * width ** 2 * n * (n - 1))
            for y in real.detach():
                k = math.exp(-float((x - y).square().sum()) / (2 * width ** 2))
                expected[i] += 2 * k * (x - y) / (len(WIDTHS) * width ** 2 * n * m)
    torch.testing.assert_close(gradient, expected, rtol=1e-12, atol=1e-14)
    assert torch.autograd.gradcheck(lambda x: kernel_mmd_u(x, real), (fake,))
    order = torch.tensor([2, 0, 3, 1])
    reordered = fake.detach()[order].requires_grad_(True)
    permuted = kernel_mmd_u(reordered, real.flip(0))
    torch.testing.assert_close(permuted, value)
    torch.testing.assert_close(torch.autograd.grad(permuted, reordered)[0], gradient[order])


def test_signed_matched_u_retains_duplicate_rows_and_zero_collision_gradient():
    panel = torch.tensor([[-1.], [1.]], dtype=torch.float64, requires_grad=True)
    k = scalar_kernel([-1.], [1.])
    assert kernel_mmd_u(panel, panel).item() == pytest.approx(k - 1)
    duplicate = panel.detach().repeat_interleave(2, 0)
    assert kernel_mmd_u(duplicate, duplicate).item() == pytest.approx((k - 1) / 3)
    collapsed = torch.zeros(8, 2, dtype=torch.float64, requires_grad=True)
    value = kernel_mmd_u(collapsed, collapsed)
    assert value.item() == 0.
    assert torch.equal(torch.autograd.grad(value, collapsed)[0], torch.zeros_like(collapsed))
    pole = torch.zeros(12, 1, dtype=torch.float64, requires_grad=True)
    offsets = torch.linspace(-.05, .05, 6, dtype=torch.float64)
    target = torch.cat((-1 + offsets, 1 + offsets)).unsqueeze(1)
    torch.testing.assert_close(torch.autograd.grad(kernel_mmd_u(pole, target), pole)[0],
                               torch.zeros_like(pole), rtol=0, atol=1e-17)


@pytest.mark.parametrize("weight", [True, -1., float("nan"), float("inf"), "1", 10 ** 400])
def test_invalid_weight_rejected(weight):
    with pytest.raises(ValueError, match="output_mmd_weight"):
        Recipe(output_mmd_weight=weight)


@pytest.mark.parametrize("widths", [(), [], .5, "1", (True,), (0.,), (-1.,),
                                    (float("nan"),), (float("inf"),), ("1",), (10 ** 400,)])
def test_invalid_widths_rejected(widths):
    with pytest.raises(ValueError, match="output_mmd_bandwidths"):
        Recipe(output_mmd_bandwidths=widths)


@pytest.mark.parametrize("dtype,width", [(torch.float32, 1e-30), (torch.float32, 1e30),
                                        (torch.float64, 1e-200), (torch.float64, 1e200)])
def test_squared_bandwidth_underflow_or_overflow_fails_explicitly(dtype, width):
    panel = torch.zeros(2, 1, dtype=dtype)
    with pytest.raises(ValueError, match="squared.*representable"):
        kernel_mmd_u(panel, panel, bandwidths=(width,))


@pytest.mark.parametrize("bad", [torch.zeros(1, 2), torch.zeros(3, 0),
    torch.full((3, 2), float("nan")), torch.zeros(3, 2, dtype=torch.int64)])
def test_invalid_panels_rejected(bad):
    with pytest.raises(ValueError, match="MMD U"):
        kernel_mmd_u(bad, bad)


def test_mismatched_space_dtype_and_nonrepresentable_distances_rejected():
    good = torch.zeros(3, 2)
    for bad in (torch.zeros(3, 3), good.double()):
        with pytest.raises(ValueError, match="MMD U"):
            kernel_mmd_u(good, bad)
    huge = torch.tensor([[1e30], [-1e30]])
    with pytest.raises(ValueError, match="distances.*representable"):
        kernel_mmd_u(huge, huge)


def test_default_packet_matches_archived_recipe_and_custom_inactive_widths_remain_bound():
    original = old_module(OLD_RECIPES_BLOB, "_mmd_old_recipes")
    assert json.dumps(Recipe().to_dict(), sort_keys=True) == json.dumps(original.Recipe().to_dict(), sort_keys=True)
    assert not {"output_mmd_weight", "output_mmd_bandwidths"} & Recipe().to_dict().keys()
    recipe = Recipe(output_mmd_weight=1., output_mmd_bandwidths=list(WIDTHS))
    assert recipe.output_mmd_bandwidths == WIDTHS
    assert recipe.to_dict()["output_mmd_bandwidths"] == WIDTHS
    assert Recipe(**json.loads(json.dumps(recipe.to_dict()))) == recipe
    custom = Recipe(output_mmd_bandwidths=[.25, 1.])
    assert custom.to_dict()["output_mmd_bandwidths"] == (.25, 1.)
    assert Recipe(**custom.to_dict()) == custom
    assert without_default_additions(asdict(Recipe())) == without_default_additions(original.asdict(original.Recipe()))
    for bad in ({"output_mmd_weight": True}, {"output_mmd_weight": 10 ** 400},
                {"output_mmd_weight": 1.},
                {"output_mmd_weight": 1., "output_mmd_bandwidths": [True]}):
        with pytest.raises(ValueError):
            without_default_additions(bad)


def test_signed_consumer_resume_json_binding_and_conditioning_exclusion():
    recipe = Recipe(output_mmd_weight=2.)
    consumer = OutputMMDU(recipe)
    panel = torch.tensor([[-1.], [1.]], requires_grad=True)
    value = consumer.add(panel.new_zeros(()), panel, panel,
        conditioning=torch.tensor([[1000.], [-2000.]]))
    assert value.item() == consumer.loss_sum == consumer.minimum_loss == consumer.maximum_loss < 0
    value.backward()
    assert torch.isfinite(panel.grad).all()
    saved = json.loads(json.dumps(consumer.state_dict()))
    restored = OutputMMDU(recipe)
    restored.load_state_dict(saved)
    assert restored.state_dict() == consumer.state_dict()
    restored.add(value.detach(), panel, panel)
    assert restored.calls == restored.active_calls == 2
    assert restored.loss_sum == 2 * consumer.loss_sum


@pytest.mark.parametrize("field,value", [("weight", .5), ("weight", True),
    ("bandwidths", (.25, 1.)), ("bandwidths", [True]),
    ("consumer", "output_energy_u_v1"), ("schema_version", True),
    ("loss_sum", float("nan")), ("loss_sum", 4.), ("calls", -1),
    ("calls", True), ("active_calls", 0), ("minimum_loss", 4.),
    ("maximum_loss", float("inf")), ("loss_sum", 10 ** 400),
    ("minimum_loss", 10 ** 400), ("maximum_loss", 10 ** 400)])
def test_corrupt_signed_state_rejected_atomically(field, value):
    consumer = OutputMMDU(Recipe(output_mmd_weight=1.))
    panel = torch.tensor([[-1.], [1.]])
    consumer.add(torch.tensor(0.), panel, panel)
    original = consumer.state_dict()
    bad = deepcopy(original)
    bad[field] = value
    with pytest.raises(ValueError, match="checkpoint"):
        consumer.load_state_dict(bad)
    assert consumer.state_dict() == original


def test_missing_extra_and_inactive_active_state_rejected():
    consumer = OutputMMDU(Recipe(output_mmd_weight=1.))
    for bad in ({}, {**consumer.state_dict(), "extra": 0},
                {**consumer.state_dict(), "calls": 1, "active_calls": 1, "loss_sum": -1.}):
        with pytest.raises(ValueError, match="checkpoint"):
            consumer.load_state_dict(bad)
    disabled = OutputMMDU(Recipe())
    with pytest.raises(ValueError, match="checkpoint"):
        disabled.load_state_dict({**disabled.state_dict(), "calls": 1, "active_calls": 1})


def test_disabled_consumer_never_inspects_panels_or_calls_loss(monkeypatch):
    import particlegan.kernel_mmd as module
    def forbidden(*args):
        raise AssertionError("disabled MMD consumed")
    monkeypatch.setattr(module, "kernel_mmd_u", forbidden)
    assert Recipe().output_mmd_loss(torch.zeros(0, 1), None).item() == 0.
    monkeypatch.setattr(Recipe, "output_mmd_loss", forbidden)
    consumer = OutputMMDU(Recipe())
    total = torch.tensor(3.)
    assert consumer.add(total, None, None, conditioning=object()) is total
    assert consumer.calls == consumer.active_calls == 0


def test_inactive_native_update_matches_exact_pre_mmd_source_and_rng(monkeypatch):
    original = old_module(OLD_TRAINING_BLOB, "_mmd_old_training")
    control, current = trainer(0., cls=original.GANTrainer), trainer(0.)
    def forbidden(*args):
        raise AssertionError("zero-weight MMD call")
    monkeypatch.setattr(Recipe, "output_mmd_loss", forbidden)
    equal(control.state_dict(), current.state_dict())
    equal(control.step(PANEL), current.step(PANEL))
    equal(control.state_dict(), current.state_dict())


def test_native_mmd_and_composed_energy_use_same_existing_panels_and_no_extra_calls(monkeypatch):
    instances = [trainer(0.), trainer(1.), trainer(1., energy=1.)]
    captures = {"mmd": [], "energy": []}
    for method, key in (("output_mmd_loss", "mmd"), ("energy_distance_loss", "energy")):
        original = getattr(Recipe, method)
        def capture(self, fake, real, *, original=original, key=key):
            captures[key].append((fake, real))
            return original(self, fake, real)
        monkeypatch.setattr(Recipe, method, capture)
    counts, targets, results = [], [], []
    for instance in instances:
        bucket = dict(G=0, D=0, sample=0)
        for key, model in (("G", instance.G), ("D", instance.D)):
            def observe(_module, _inputs, _output, *, key=key, bucket=bucket):
                bucket[key] += 1
            model.register_forward_hook(observe)
        sampler = instance._sample_training_prior
        def sampled(n, *, sampler=sampler, bucket=bucket):
            bucket["sample"] += 1
            return sampler(n)
        monkeypatch.setattr(instance, "_sample_training_prior", sampled)
        def target():
            targets.append(1)
            return PANEL + .2
        results.append(instance.step(PANEL, generator_real=target))
        counts.append(bucket)
    assert counts[0] == counts[1] == counts[2] and counts[0]["sample"] == 2
    assert len(targets) == 3 and len(captures["mmd"]) == 2 and len(captures["energy"]) == 1
    assert captures["mmd"][1][0] is captures["energy"][0][0]
    assert captures["mmd"][1][1] is captures["energy"][0][1]
    assert "output_mmd" not in results[0]
    for i, (fake, real) in enumerate(captures["mmd"], start=1):
        assert fake.requires_grad and torch.equal(real, PANEL + .2)
        assert results[i]["output_mmd"] == kernel_mmd_u(fake, real).detach()
    for other in instances[1:]:
        equal(instances[0].state_dict()["streams"], other.state_dict()["streams"])
        assert any(not torch.equal(a, b) for a, b in zip(instances[0].G.parameters(), other.G.parameters()))
    assert instances[1].opt_g.state[instances[1].prior.z]["step"] == 1
    assert all(instances[1].opt_d.state[p]["step"] == 1 for p in instances[1].D.parameters())


def test_active_native_resume_binds_weight_widths_and_rejects_legacy_alias_substitution():
    active = trainer()
    active.step(PANEL)
    saved = active.state_dict()
    expected = active.step(PANEL)
    restored = trainer()
    restored.load_state_dict(saved)
    equal(expected, restored.step(PANEL))
    equal(active.state_dict(), restored.state_dict())
    for incompatible in (trainer(0.), trainer(.5), trainer(1., widths=(.25, 1.))):
        before = incompatible.state_dict()
        with pytest.raises(ValueError, match="recipe"):
            incompatible.load_state_dict(saved)
        equal(before, incompatible.state_dict())
    for mutate in (lambda r: r.pop("output_mmd_bandwidths"),
                   lambda r: r.update(output_mmd_weight=True),
                   lambda r: r.update(loss_type="logistic"),
                   lambda r: r.update(gan_mode="rp"),
                   lambda r: r.update(reg_method="autograd"),
                   lambda r: r.update(loss_type="logistic", gan_mode="ls", reg_method="autograd")):
        bad = deepcopy(saved)
        mutate(bad["recipe"])
        before = restored.state_dict()
        with pytest.raises(ValueError, match="recipe"):
            restored.load_state_dict(bad)
        equal(before, restored.state_dict())
