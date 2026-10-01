"""Routed DV12 sees represented atoms, rather than physical row slots."""
from copy import deepcopy
import importlib
import io
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from particlegan import RoutedRows
from particlegan.continuous import DataDriftController


DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))]


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            assert_tree_equal(left[name], right[name])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    elif isinstance(left, float) and math.isnan(left):
        assert math.isnan(right)
    else:
        assert left == right


def cpu_roundtrip(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=True)


@pytest.fixture(autouse=True)
def single_threaded():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.autograd.set_multithreading_enabled(False):
        yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def api():
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
        sites = importlib.import_module("e22_routed_sites")
        replay = importlib.import_module("e22_routed_replay")
    return sites, replay.checkpointed_generate


def make_mass_loop(api, device):
    sites, _ = api
    loop = sites.make_loop(mode="no_rows", device=device, tokens=2, particles=4, batch_size=4)
    p = loop.policy
    with torch.no_grad():
        p.table.copy_(p.table.new_tensor([[1., 0.], [-1., 0.], [20., 30.], [0., 2.]]))
        p.averaged_table.copy_(2 * p.table)
        p.router.log_mass.copy_(p.table.new_tensor([0., 0., -torch.inf, 0.]))
        p.ema_router.log_mass.copy_(p.router.log_mass)
    p.controller.latent_bandwidth = None
    p.controller.observe_prior(p._prior_view())
    return loop


def duplicate_parent(policy):
    with torch.no_grad():
        for table, router in ((policy.table, policy.router), (policy.averaged_table, policy.ema_router)):
            table[2].copy_(table[0])
            router.log_mass[0].sub_(math.log(2))
            router.log_mass[2].copy_(router.log_mass[0])


@pytest.mark.parametrize("device", DEVICES)
def test_mass_weighted_spread_and_effective_unique_atom_count(device):
    table = torch.tensor([[1., 0.], [-1., 0.], [1e6, -1e6], [0., 2.]], device=device)
    mass = table.new_tensor([0., math.log(2), -torch.inf, 0.])
    centers, width = DataDriftController.routed_geometry(table, mass)
    assert_tree_equal(centers, table.new_tensor([[-1., 0.], [0., 2.], [1., 0.]]))
    # Atom weights (.25, .5, .25) give variance (.6875, .75), N_eff=8/3.
    expected = table.new_tensor([.6875, .75]).sqrt() * (8 / 3) ** -.5
    torch.testing.assert_close(width, expected, rtol=2e-7, atol=0)
    # The prior depends on normalized mass, not its arbitrary log offset.
    torch.testing.assert_close(width, DataDriftController.routed_geometry(table, mass + 16)[1],
                               rtol=2e-6, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_inactive_row_changes_neither_bandwidth_nor_support_clipping(device):
    table = torch.tensor([[0., 0.], [2., 0.], [0., 2.], [.4001, .4001]], device=device)
    mass = table.new_tensor([0., 0., 0., -torch.inf])
    far = table.clone()
    far[-1] = far.new_tensor([1e6, -1e6])
    controllers = [DataDriftController("dv12") for _ in range(2)]
    latent = table.new_tensor([[.4, .4], [.7, .6]])
    outputs, states = [], []
    for controller, centers in zip(controllers, (table, far)):
        # A plain representation-bound view also works for caller diagnostics.
        prior = SimpleNamespace(z=centers, log_mass=mass)
        controller.observe_prior(prior)
        outputs.append(controller.perturb_latent(
            latent, torch.Generator(device=device).manual_seed(23), prior, record=True))
        states.append(controller.state_dict())
    assert_tree_equal(outputs[0], outputs[1])
    assert_tree_equal(states[0], states[1])
    assert (outputs[0] - latent).norm(dim=-1).max() > .01
    # Even invalid inactive storage cannot enter the geometric calculations.
    far[-1] = torch.nan
    assert_tree_equal(DataDriftController.routed_geometry(table, mass),
                      DataDriftController.routed_geometry(far, mass))


@pytest.mark.parametrize("device", DEVICES)
def test_exact_half_mass_duplicate_preserves_geometry_and_bandwidth_refresh(api, device):
    original, duplicate = [make_mass_loop(api, device) for _ in range(2)]
    duplicate_parent(duplicate.policy)
    for averaged in (False, True):
        priors = [loop.policy._prior_view(averaged) for loop in (original, duplicate)]
        assert_tree_equal(priors[0]._mass_support, priors[1]._mass_support)
        assert_tree_equal(priors[0]._mass_width, priors[1]._mass_width)
        controllers = [DataDriftController("dv12") for _ in range(2)]
        for controller in controllers:
            controller.latent_bandwidth = original.policy.table.new_tensor([.3, .7])
        for _ in range(3):
            for controller, prior in zip(controllers, priors):
                controller.observe_prior(prior)
        assert_tree_equal(controllers[0].state_dict(), controllers[1].state_dict())


@pytest.mark.parametrize("device", DEVICES)
def test_two_site_clean_noisy_outputs_and_coupled_gradients_survive_half_duplicate(api, device, monkeypatch):
    original, duplicate = [make_mass_loop(api, device) for _ in range(2)]
    duplicate_parent(duplicate.policy)
    context = original.fit_context[:3]
    captures = {id(loop.policy.controller): [] for loop in (original, duplicate)}
    perturb = DataDriftController.perturb_latent

    def captured(controller, latent, stream, prior=None, record=False):
        output = perturb(controller, latent, stream, prior, record=record)
        captures[id(controller)].append(output.detach().clone())
        return output

    monkeypatch.setattr(DataDriftController, "perturb_latent", captured)
    for noisy in (False, True):
        outputs, gradients = [], []
        for loop in (original, duplicate):
            p = loop.policy
            output = p.routed_generate(context, sigma=0, perturb=noisy,
                                       stream=torch.Generator(device=device).manual_seed(19))
            probe = torch.linspace(-.3, .7, output.numel(), device=device).reshape_as(output)
            outputs.append(output.detach())
            gradients.append(torch.autograd.grad((output * probe).sum(), p.table)[0])
        torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-6, atol=2e-7)
        before, after = gradients
        assert before[2].eq(0).all() and before[0].norm() > 0
        torch.testing.assert_close(after[0], before[0] / 2, rtol=2e-5, atol=2e-7)
        torch.testing.assert_close(after[2], before[0] / 2, rtol=2e-5, atol=2e-7)
        torch.testing.assert_close(after[0] + after[2], before[0], rtol=2e-5, atol=2e-7)
        torch.testing.assert_close(after[[1, 3]], before[[1, 3]], rtol=2e-5, atol=2e-7)
    for calls in captures.values():
        assert len(calls) == 2 and all(value.shape == (6, 2) for value in calls)
    for before, after in zip(*captures.values()):
        torch.testing.assert_close(before, after, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_tied_key_and_value_gradient_paths_each_transport_half_parent_mass(api, device):
    original, duplicate = [make_mass_loop(api, device) for _ in range(2)]
    duplicate_parent(duplicate.policy)
    parts = []
    for loop in (original, duplicate):
        p, context = loop.policy, loop.fit_context[:3]
        logits = p.router.first_query(p.encoder(context)) @ p.table.T / math.sqrt(p.table.shape[1])
        weights = (logits + p.router.log_mass).softmax(-1)
        probe = torch.linspace(-.4, .8, len(context) * 2 * 2, device=device).reshape(len(context), 2, 2)
        key = torch.autograd.grad(((weights @ p.table.detach()) * probe).sum(), p.table,
                                  retain_graph=True)[0]
        value = torch.autograd.grad(((weights.detach() @ p.table) * probe).sum(), p.table)[0]
        assert key[0].norm() > 0 and value[0].norm() > 0
        parts.append((key, value))
    for before, after in zip(*parts):
        torch.testing.assert_close(after[0], before[0] / 2, rtol=2e-5, atol=1e-7)
        torch.testing.assert_close(after[2], before[0] / 2, rtol=2e-5, atol=1e-7)
        torch.testing.assert_close(after[[1, 3]], before[[1, 3]], rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("decisive, averaged", [(1, False), (-1, True)])
def test_serving_noise_binds_selected_table_and_mass_and_preserves_duplicate(api, device, decisive, averaged):
    original, duplicate = [make_mass_loop(api, device) for _ in range(2)]
    for loop in (original, duplicate):
        loop.policy.ema_router.log_mass[1] = -torch.inf
        loop.policy._table_tester().last_decisive = decisive
    duplicate_parent(duplicate.policy)
    context, outputs = original.test_context[:3], []
    for loop in (original, duplicate):
        p = loop.policy
        before = p.state_dict()
        snapshot = p.served_model()
        assert snapshot.source == ("averaged" if averaged else "fast")
        actual = snapshot.routed_forward(context, perturb=True,
                                         generator=torch.Generator(device=device).manual_seed(29))
        candidate = p.routed_control.candidate(averaged=averaged)
        prior = DataDriftController.routed_prior(candidate.table, candidate.log_mass)
        controller, stream = deepcopy(p.controller), torch.Generator(device=device).manual_seed(29)
        expected = p.routed_control.generate(context, candidate=candidate,
            perturb_fn=lambda codes: controller.perturb_latent(codes, stream, prior))
        assert_tree_equal(actual, expected)
        assert_tree_equal(p.state_dict(), before)
        outputs.append(actual)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_mass_geometry_exact_resume_and_activation_replay_after_cpu_load(api, device):
    sites, checkpointed_generate = api
    loop = make_mass_loop(api, device)
    duplicate_parent(loop.policy)
    sites.update(loop)
    saved = cpu_roundtrip(sites.checkpoint(loop))
    native, replay = [make_mass_loop(api, device) for _ in range(2)]
    for restored in (native, replay):
        sites.restore(restored, saved)
        assert_tree_equal(restored.policy.served_snapshot(), loop.policy.served_snapshot())
    for _ in range(3):
        expected = sites.update(loop)
        assert_tree_equal(sites.update(native), expected)
        assert_tree_equal(sites.update(replay, generator_forward=checkpointed_generate), expected)
        assert_tree_equal(sites.checkpoint(native), sites.checkpoint(loop))
        assert_tree_equal(sites.checkpoint(replay), sites.checkpoint(loop))
        assert_tree_equal(replay.policy.table.grad, native.policy.table.grad)
    expected = loop.policy.served_model().routed_forward(loop.test_context[:3], perturb=True,
                              generator=torch.Generator(device=device).manual_seed(31))
    for restored in (native, replay):
        assert_tree_equal(restored.policy.served_model().routed_forward(restored.test_context[:3], perturb=True,
                          generator=torch.Generator(device=device).manual_seed(31)), expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("marker", [None, "rows_v0"])
def test_old_routed_noise_law_is_rejected_before_checkpoint_mutation(api, device, marker):
    p = make_mass_loop(api, device).policy
    p._table_tester().last_decisive = -1
    before, served = p.state_dict(), p.served_snapshot()
    assert before["routing"]["config"]["routed_geometry"] == "mass_atoms_v1"
    changed = deepcopy(before)
    if marker is None:
        del changed["routing"]["config"]["routed_geometry"]
    else:
        changed["routing"]["config"]["routed_geometry"] = marker
    changed["models"]["generator"]["first_adapter.weight"].zero_()
    with pytest.raises(ValueError, match="routed DV12 law changed.*prior release.*migration"):
        p.load_state_dict(changed)
    assert_tree_equal(p.state_dict(), before)
    assert_tree_equal(p.served_snapshot(), served)


def test_geometry_version_configuration_roundtrip_preserves_public_spec_reuse():
    def unused(*args):
        raise AssertionError("specification construction must not execute callbacks")

    original = RoutedRows(route=unused, generate=unused, features=unused)
    restored = RoutedRows(route=unused, generate=unused, features=unused, **original.to_dict())
    assert restored.to_dict() == original.to_dict()
    before = original.to_dict()
    for version in (None, "rows_v0", True, 1):
        with pytest.raises(ValueError, match="unsupported routed_geometry.*prior release.*migration"):
            RoutedRows(route=unused, generate=unused, features=unused,
                       **{**before, "routed_geometry": version})
        assert original.to_dict() == before
