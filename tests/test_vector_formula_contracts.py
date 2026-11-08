"""Software controls for formula-faithful vector execution, without quality credit."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from benchmarks.toy_audit import api_vectors as api
from tests.test_toy_api_vectors import assert_nested_equal


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_case(identifier):
    case = deepcopy(api._registry()[identifier])
    case.update(particles=32, batch_size=16)
    return case


@pytest.mark.parametrize("import_first", [False, True])
def test_shared_generator_architecture_does_not_depend_on_vector_import(import_first):
    source = f'''
import json
import torch
if {import_first!r}:
    import benchmarks.toy_audit.api_vectors
import lib.toy_models as models
original = models.SimpleMLPGenerator
torch.manual_seed(0)
before = original(2, 32, 16, 2)
import benchmarks.toy_audit.api_vectors as api
torch.manual_seed(0)
after = models.SimpleMLPGenerator(2, 32, 16, 2)
print(json.dumps({{
    "same_class": original is models.SimpleMLPGenerator is api.SimpleMLPGenerator,
    "normalizations": [sum(isinstance(m, torch.nn.LayerNorm) for m in g.modules()) for g in (before, after)],
    "same_initial_state": all(torch.equal(v, after.state_dict()[k]) for k, v in before.state_dict().items()),
}}))
'''
    result = subprocess.run([sys.executable, "-c", source], text=True, capture_output=True,
                            cwd=Path(__file__).resolve().parents[1], check=True)
    assert json.loads(result.stdout) == {
        "same_class": True, "normalizations": [0, 0], "same_initial_state": True}


@pytest.mark.parametrize("identifier,expected_class,normalizations,residual_scale", [
    ("api-grid100-mlp16", api.SimpleMLPGenerator, 0, None),
    ("api-grid100-res16-tanh", api.ResidualMLPGenerator, 0, 1.),
    ("api-grid100-res16-scaled-tanh", api.ScaledResidualMLPGenerator, 1, 1 / 14 ** .5),
])
def test_deep_tasks_construct_distinct_declared_architectures(identifier, expected_class,
                                                            normalizations, residual_scale):
    fixture = api.VectorFixture(small_case(identifier), recipe_name="k3p", max_steps=1)
    generator = fixture.trainer.G
    assert type(generator) is expected_class
    assert sum(isinstance(m, torch.nn.LayerNorm) for m in generator.modules()) == normalizations
    if residual_scale is not None:
        assert all(block.scale == pytest.approx(residual_scale) for block in generator.blocks)
    assert fixture.metadata["architecture"]["generator"] == expected_class.__name__


def test_scaled_task_has_no_model_or_prior_writes_outside_the_trainer(monkeypatch):
    """Cross all former helper deadlines with the optimizer deliberately inactive.

    This is a writer-ownership check, not a learned-quality experiment. The
    former empirical helper reset the perturbed generator and rewrote rows.
    """
    fixture = api.VectorFixture(small_case("api-grid100-res16-scaled-tanh"),
                                recipe_name="k3p", max_steps=100)
    with torch.no_grad():
        fixture.trainer.G.head.bias.add_(.3)
        fixture.trainer.prior.z.add_(2.)
    before = deepcopy(fixture.trainer.state_dict())
    calls = []

    def inactive_step(real, *, generator_real):
        calls.append((real.clone(), generator_real.clone()))
        return {"step": len(calls)}

    monkeypatch.setattr(fixture.trainer, "step", inactive_step)
    for _ in range(100):
        fixture.step()
    assert len(calls) == fixture.completed_steps == 100
    assert_nested_equal(before, fixture.trainer.state_dict())
    assert not hasattr(fixture.trainer, "_scaled_res16_emp_state")


def test_scaled_fixture_fresh_object_continuation_matches_uninterrupted_state():
    case = small_case("api-grid100-res16-scaled-tanh")
    source = api.VectorFixture(case, recipe_name="k3p", max_steps=3)
    source.step()
    saved = source.state_dict()
    restored = api.VectorFixture(case, recipe_name="k3p", max_steps=3)
    restored.load_state_dict(saved)
    assert_nested_equal(saved, restored.state_dict())
    source.step()
    restored.step()
    assert_nested_equal(source.state_dict(), restored.state_dict())


def test_pre_repair_vector_checkpoint_requires_its_original_source():
    fixture = api.VectorFixture(small_case("api-grid100-res16-scaled-tanh"),
                                recipe_name="k3p", max_steps=1)
    before = fixture.state_dict()
    historical = deepcopy(before)
    historical["version"] = "toy-public-api-vectors-v2"
    with pytest.raises(ValueError, match="identity/recipe"):
        fixture.load_state_dict(historical)
    assert_nested_equal(before, fixture.state_dict())


def test_soft_bounded_tanh_is_the_derivative_of_its_forward():
    u = torch.tensor([-15., -1., 0., 1., 15.], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(api.TanhRailGuard.apply, (u,))
    assert torch.autograd.gradgradcheck(api.TanhRailGuard.apply, (u,))
    zero = torch.autograd.grad(api.TanhRailGuard.apply(u), u, torch.zeros_like(u))[0]
    assert torch.equal(zero, torch.zeros_like(u))
    one = torch.tensor([1.], dtype=torch.float64, requires_grad=True)
    observed = torch.autograd.grad(api.TanhRailGuard.apply(one), one)[0]
    expected = 1 - torch.tanh(one / 5.25).square()
    torch.testing.assert_close(observed, expected)
