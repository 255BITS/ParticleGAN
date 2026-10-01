"""Independent particle ownership, callbacks, row transport and exact recovery."""
from copy import deepcopy
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from particlegan.birth_death import ParticleBirthDeath, ParticleRows, ScalarHeadFeatures
from particlegan.row_evidence import RowEvidence


def _table(rows=32, width=2):
    t = torch.arange(rows * width, dtype=torch.float64).reshape(rows, width)
    return nn.Parameter(torch.sin(t * .71) + t * .03)


def _rows(table=None, **overrides):
    table = _table() if table is None else table
    return ParticleRows(**{"table": table, "optimizer": torch.optim.Adam([table], lr=.01, amsgrad=True),
                           "averaged_table": table.detach().clone(), "generate": lambda z: z,
                           **overrides})


def _same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _same(a, b)
    elif isinstance(left, float) and math.isnan(left):
        assert math.isnan(right)
    else:
        assert left == right


@pytest.mark.parametrize("semantics", ["conditional", "soft", "dense_soft", "routed", "weighted"])
def test_non_independent_banks_reject_row_controls(semantics):
    with pytest.raises(ValueError, match="independent"):
        _rows(semantics=semantics)
    with pytest.raises(ValueError, match="independent"):
        RowEvidence(_table(), semantics=semantics)


def test_table_optimizer_and_average_ownership_are_explicit():
    table = _table()
    other = _table()
    with pytest.raises(ValueError, match="optimizer exactly once"):
        _rows(table, optimizer=torch.optim.Adam([other]))
    with pytest.raises(ValueError, match="separate storage"):
        _rows(table, averaged_table=table.detach())
    with pytest.raises(ValueError, match="frozen"):
        _rows(table, averaged_table=table.detach().clone().requires_grad_())
    with pytest.raises(ValueError, match="leaf"):
        _rows(table.detach())
    with pytest.raises(ValueError, match="critic-feature callback"):
        ParticleBirthDeath(_rows(table), seed=0, space="critic")


def test_callback_sampling_matches_uniform_law_and_private_draw_order():
    table = _table(rows=64)
    model = nn.Sequential(nn.Linear(2, 8), nn.BatchNorm1d(8), nn.Dropout(.8), nn.Linear(8, 2)).double()
    model.train()
    model[2].eval()                 # mixed flags must be preserved
    modes = [module.training for module in model.modules()]
    buffers = {key: value.clone() for key, value in model.named_buffers()}
    calls = []

    def generate(z):
        calls.append((z.clone(), torch.is_grad_enabled(), [module.training for module in model.modules()]))
        return model(z)

    rows = _rows(table, generate=generate, evaluation_modules=(model,))
    birth_death = ParticleBirthDeath(rows, seed=7)
    birth_death.dry_run = True
    birth_death.observe_real(table.detach() + .037)
    expected_stream = torch.Generator().set_state(birth_death.stream.get_state())
    pick = torch.randint(len(table), (len(table),), generator=expected_stream)
    torch.randn(table.shape, dtype=table.dtype, generator=expected_stream)
    torch.randn(table.shape, dtype=table.dtype, generator=expected_stream)
    torch.rand(len(table), dtype=torch.float64, generator=expected_stream)
    training_rng = torch.get_rng_state().clone()

    event = birth_death.maybe_apply(.029)

    assert event is not None and len(calls) == 2
    assert torch.equal(calls[0][0], table.detach()[pick])
    assert torch.equal(calls[1][0], table.detach())
    assert len(torch.unique(pick)) < len(table)  # iid draws with replacement
    assert all(not grad and not any(flags) for _, grad, flags in calls)
    assert modes == [module.training for module in model.modules()]
    for key, value in model.named_buffers():
        assert torch.equal(value, buffers[key])
    assert torch.equal(birth_death.stream.get_state(), expected_stream.get_state())
    assert torch.equal(torch.get_rng_state(), training_rng)


def test_moves_copy_fast_average_moments_and_history_with_original_jitter():
    table = _table()
    optimizer = torch.optim.Adam([table], lr=.01, amsgrad=True)
    table.grad = torch.arange(table.numel(), dtype=table.dtype).reshape_as(table) * .1
    optimizer.step()
    optimizer.latent_history = table.detach().clone() + 20
    average = table.detach().clone() + 100
    controller = SimpleNamespace(variant="dv12", latent_bandwidth=.05)
    rows = _rows(table, optimizer=optimizer, averaged_table=average, controller=controller)
    birth_death = ParticleBirthDeath(rows, seed=11)
    child, parent = torch.tensor([0, 1, 7]), torch.tensor([1, 0, 14])
    before, before_average = table.detach().clone(), average.clone()
    before_state = deepcopy(optimizer.state[table])
    before_history = optimizer.latent_history.clone()
    expected_stream = torch.Generator().set_state(birth_death.stream.get_state())
    noise = torch.randn(before[parent].shape, dtype=table.dtype, generator=expected_stream)
    # Independent analytical DV12 half-neighbour bound (not the sampler's kNN implementation).
    distances = (before[parent][:, None, :] - before[None, :, :]).norm(dim=2)
    radius = distances.masked_fill(distances == 0, float("inf")).min(1).values * .5
    displacement = controller.latent_bandwidth * noise
    delta = displacement * (radius / displacement.norm(dim=1).clamp_min(1e-20)).clamp_max(1)[:, None]

    birth_death._move(child, parent)

    torch.testing.assert_close(table[child], before[parent] + delta, rtol=0, atol=1e-15)
    torch.testing.assert_close(average[child], before_average[parent] + delta, rtol=0, atol=1e-14)
    untouched = torch.ones(len(table), dtype=torch.bool)
    untouched[child] = False
    assert torch.equal(table[untouched], before[untouched])
    assert torch.equal(average[untouched], before_average[untouched])
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        assert torch.equal(optimizer.state[table][key][child], before_state[key][parent])
        assert torch.equal(optimizer.state[table][key][untouched], before_state[key][untouched])
    assert torch.equal(optimizer.state[table]["step"], before_state["step"])
    assert torch.equal(optimizer.latent_history[child], before_history[parent])
    assert torch.equal(optimizer.latent_history[untouched], before_history[untouched])
    assert torch.equal(birth_death.stream.get_state(), expected_stream.get_state())


def test_moves_support_generic_optimizer_without_latent_history():
    rows = _rows()                 # ordinary torch Adam has no A2 history
    birth_death = ParticleBirthDeath(rows, seed=0)
    parent = rows.table.detach()[[8]].clone()
    birth_death._move(torch.tensor([0]), torch.tensor([8]))
    assert torch.equal(rows.table[0], parent[0])
    assert not hasattr(rows.optimizer, "latent_history")


class _SkipCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.unused = nn.Linear(2, 1)
        self.hidden = nn.Linear(2, 4)
        self.head = nn.Linear(4, 1)
        self.raw_skip = nn.Linear(2, 1)

    def forward(self, sample):
        return self.head(self.hidden(sample).tanh()) + self.raw_skip(sample)


def test_native_features_exclude_raw_and_unused_heads_and_resume_discovery(monkeypatch):
    critic = _SkipCritic().double().train()
    critic.raw_skip.eval()
    modes = [module.training for module in critic.modules()]
    features = ScalarHeadFeatures(critic)
    sample = _table().detach()
    expected = critic.hidden(sample).tanh()
    actual = features(sample)
    assert torch.equal(actual, expected)
    assert features.state_dict() == {"heads": ["head"]}
    assert modes == [module.training for module in critic.modules()]
    assert not any(module._forward_pre_hooks for module in critic.modules())

    restored = ScalarHeadFeatures(deepcopy(critic))
    restored.load_state_dict(features.state_dict())
    monkeypatch.setattr(restored, "_is_raw", lambda *args: pytest.fail("restored callback rediscovered heads"))
    assert torch.equal(restored(sample), expected)
    with pytest.raises(ValueError, match="feature heads"):
        restored.load_state_dict({"heads": ["missing"]})


def test_raw_only_critic_is_rejected_and_flags_hooks_are_restored():
    critic = nn.Linear(2, 1).double().train()
    features = ScalarHeadFeatures(critic)
    with pytest.raises(ValueError, match="no feature space"):
        features(_table().detach())
    assert critic.training and not critic._forward_pre_hooks


def test_custom_feature_callback_receives_image_shape_and_resumes_before_eval():
    seen_shapes = []

    def features(sample):
        seen_shapes.append(tuple(sample.shape))
        flat = sample.flatten(1)
        return torch.cat((flat, flat.square()), 1)

    table = _table(width=4)
    rows = _rows(table, generate=lambda z: z.reshape(-1, 1, 2, 2), critic_features=features,
                 completed_steps=lambda: 13)
    birth_death = ParticleBirthDeath(rows, seed=3, space="critic", feature_scale="std")
    birth_death.observe_real((table.detach() + .07).reshape(-1, 1, 2, 2))
    checkpoint = birth_death.state_dict()
    restored = ParticleBirthDeath(_rows(nn.Parameter(table.detach().clone()), generate=rows.generate,
                                        critic_features=features, completed_steps=lambda: 13),
                                  seed=0, space="critic", feature_scale="std")
    restored.load_state_dict(checkpoint)

    event = birth_death.maybe_apply(.029)
    restored_event = restored.maybe_apply(.029)

    _same(event, restored_event)
    _same(birth_death.state_dict(), restored.state_dict())
    assert event["step"] == 14
    assert seen_shapes == [(32, 1, 2, 2)] * 6
    assert torch.equal(birth_death.rows.table, restored.rows.table)


def test_birth_death_checkpoint_snapshots_do_not_alias_live_evidence_and_legacy_loads():
    rows = _rows()
    features = ScalarHeadFeatures(_SkipCritic().double())
    rows.critic_features = features
    birth_death = ParticleBirthDeath(rows, seed=2, space="critic")
    birth_death.observe_real(rows.table.detach() + .05)
    birth_death.maybe_apply(.029)
    checkpoint = birth_death.state_dict()
    frozen = deepcopy(checkpoint)
    birth_death.S.fill_(8)
    birth_death.reservoir.fill_(3)
    birth_death.iso_log.append([1, 2, 1])
    _same(checkpoint, frozen)
    birth_death.load_state_dict(checkpoint)
    _same(birth_death.state_dict(), checkpoint)

    legacy_keys = set(birth_death._TENSORS) | {"fill", "cursor", "rows_since_eval", "counters", "last", "stream"}
    legacy = {key: value for key, value in checkpoint.items() if key in legacy_keys}
    birth_death.load_state_dict(legacy)
    assert birth_death.sample_shape is None
    birth_death.observe_real(rows.table.detach() + .05)
    assert birth_death.sample_shape == (2,)
    birth_death.maybe_apply(.029)


def test_scaled_row_evidence_serializes_scale_and_continues_exactly():
    table = _table()
    evidence = RowEvidence(table, null="scaled")
    base = torch.arange(table.numel(), dtype=table.dtype).reshape_as(table) * .1 + 1
    gradients = [base + torch.sin(base + step) for step in range(14)]
    for grad in gradients:
        evidence.update(grad)
    assert evidence.scale_c > 1
    checkpoint = evidence.state_dict()
    restored = RowEvidence(table, null="scaled")
    restored.load_state_dict(checkpoint)
    _same(evidence.state_dict(), restored.state_dict())
    evidence.update(gradients[-1] * .7)
    restored.update(gradients[-1] * .7)
    _same(evidence.state_dict(), restored.state_dict())
    assert not torch.equal(checkpoint["M"], evidence.M)

    legacy = {key: value for key, value in checkpoint.items() if key not in ("config", "scale_c")}
    restored.load_state_dict(legacy)
    assert restored.scale_c == 1 and torch.equal(restored.M, checkpoint["M"])
    with pytest.raises(ValueError, match="configuration"):
        RowEvidence(table, window=25, null="scaled").load_state_dict(checkpoint)
    with pytest.raises(ValueError, match="full floating table gradient"):
        evidence.update(torch.ones(1, 2))


def test_row_evidence_counts_aggregated_duplicate_draws_once_and_rejects_missing_sparse_gradients():
    table = _table()
    table[torch.tensor([0, 0, 3])].sum().backward()
    evidence = RowEvidence(table)
    evidence.update(table.grad)
    assert evidence.W[0] == evidence.W[3] == 1
    assert torch.equal(evidence.M[0], torch.full((2,), 2., dtype=table.dtype))
    assert torch.equal(evidence.M[3], torch.ones(2, dtype=table.dtype))
    assert evidence.W.sum() == 2
    for invalid in (None, table.grad.to_sparse()):
        with pytest.raises(ValueError, match="full floating table gradient"):
            evidence.update(invalid)
