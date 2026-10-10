"""Frozen BF16 features retain their precision and exact values in E22."""
from copy import deepcopy
import io
import math

import pytest
import torch
from torch import nn

from particlegan import E22Policy, GANTrainer, get_recipe


class FrozenFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 4)).bfloat16()
        self.layers.requires_grad_(False)
        self.offset = nn.Parameter(torch.zeros(4), requires_grad=False)
        self.register_buffer("gain", torch.ones(4, dtype=torch.bfloat16))
        self.register_buffer("reference", torch.tensor([.123456789], dtype=torch.float64))
        self.register_buffer("counter", torch.tensor(7, dtype=torch.int64))

    def forward(self, x):
        features = self.layers(x.to(torch.bfloat16)) * self.gain
        return features.to(x.dtype) + self.offset.to(x.dtype)


class MixedModule(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.nested = nn.ModuleDict({"pretrained": FrozenFeatures()})
        self.head = nn.Linear(4, output)

    def forward(self, x):
        return self.head(self.nested["pretrained"](x))


def make_policy(*, serve_average=5., wrong_role=None, trainable_buffer=False, frozen_table=False):
    recipe = get_recipe("e22", num_particles=16, z_dim=2, batch_size=8,
                        particle_birth_death=False, birth_death_isolation=False,
                        birth_death_feature_scale="none", row_evidence_gate=False,
                        output_noise_std=.04, serve_average=serve_average)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(717)
        modules = {name: MixedModule(1 if name == "critic" else 2)
                   for name in ("generator", "critic", "encoder", "router")}
        table = nn.Parameter(torch.randn(16, 2), requires_grad=not frozen_table)
    if wrong_role is not None:
        modules[wrong_role].head.bfloat16()
    if trainable_buffer:
        modules["generator"].register_buffer(
            "differentiable", torch.ones(2, dtype=torch.bfloat16, requires_grad=True))
    opt_g = recipe.make_generator_optimizer([
        {"params": [p for p in modules[name].parameters() if p.requires_grad]}
        for name in ("generator", "encoder", "router")
    ] + [{"params": [table], "lr": recipe.lr * recipe.prior_lr_mult}],
        latent_table=table, foreach=False)
    opt_d = recipe.make_critic_optimizer(modules["critic"], ema_critic=deepcopy(modules["critic"]), foreach=False)
    policy = E22Policy(recipe, modules["generator"], modules["critic"], table=table,
                       encoder=modules["encoder"], router=modules["router"],
                       generator_optimizer=opt_g, critic_optimizer=opt_d,
                       roles=[["generator", "encoder", "router", "table"], ["critic"]], seed=19)
    tester = policy._table_tester()
    if tester is not None:
        tester.s, tester.b = .5, 4.
    return policy


def update(policy):
    real = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 16
    noise = policy.begin_step(real)
    policy.D.train()
    policy.G.eval()
    with torch.no_grad():
        fake = policy.generate(policy.table[:8], sigma=noise.output_sigma)
    policy.observe_critic_pair(real, fake)
    loss = policy.recipe.make_loss()
    penalty = policy.penalty(policy.D, real, fake)
    loss_d = loss.d_loss(policy.D(real), policy.D(fake)) + penalty
    policy.opt_d.zero_grad()
    loss_d.backward()
    policy.opt_d.step()
    policy.after_critic_step()
    policy.D.eval()
    policy.G.train()
    flags = [p.requires_grad for p in policy.D.parameters()]
    try:
        policy.D.requires_grad_(False)
        policy.opt_g.zero_grad()
        routed = policy.router(policy.encoder(policy.table[:8]))
        fake = policy.generate(routed, sigma=noise.output_sigma)
        loss_g = loss.g_loss(policy.D(fake), policy.D(real))
        loss_g.backward()
        policy.after_generator_backward(loss_gan=loss_g, loss_critic=loss_d - penalty)
        policy.opt_g.step()
        policy.after_generator_step()
    finally:
        for parameter, flag in zip(policy.D.parameters(), flags):
            parameter.requires_grad_(flag)
    policy.finish_step()
    return loss_d.detach(), loss_g.detach()


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    elif isinstance(left, float) and math.isnan(left):
        assert math.isnan(right)
    else:
        assert left == right


def assert_frozen_copies(policy):
    for name, averaged in policy._average_modules().items():
        source = policy._training_modules()[name]
        for key, parameter in source.named_parameters():
            if not parameter.requires_grad:
                assert_tree_equal(parameter, dict(averaged.named_parameters())[key])
        for key, buffer in source.named_buffers():
            assert_tree_equal(buffer, dict(averaged.named_buffers())[key])
    for key, parameter in policy.D.named_parameters():
        if not parameter.requires_grad:
            assert_tree_equal(parameter, dict(policy.opt_d.ema_critic.named_parameters())[key])
    for key, buffer in policy.D.named_buffers():
        assert_tree_equal(buffer, dict(policy.opt_d.ema_critic.named_buffers())[key])


@pytest.mark.parametrize("serve_average", [0., 5.])
def test_nested_mixed_precision_freezing_is_exact_in_averages_serving_and_resume(serve_average):
    policy = make_policy(serve_average=serve_average)
    # Simulate loading new frozen weights into caller-owned modules after binding.
    # The first average must copy them, and subsequent updates must never round them.
    with torch.no_grad():
        for index, module in enumerate(policy._training_modules().values()):
            for parameter in module.parameters():
                if not parameter.requires_grad:
                    parameter.add_((index + 1) / 16.)
            for buffer in module.buffers():
                if buffer.is_floating_point():
                    buffer.add_((index + 1) / 16.)
    policy.opt_d.record.calls = 799  # Include the real KA2 anchor start and update.
    old = deepcopy(policy.ema_G.head.state_dict())
    for _ in range(3):
        update(policy)
        assert_frozen_copies(policy)
    assert not torch.equal(old["weight"], policy.ema_G.head.weight)
    policy._table_tester().last_decisive = -1
    snapshot = policy.served_snapshot()
    served = policy.served_model(generation_factory=lambda models: lambda G, z: G(
        models["router"](models["encoder"](z))))
    assert snapshot["source"] == ("averaged" if serve_average else "fast")
    for name, module in policy._training_modules().items():
        for key, tensor in module.state_dict().items():
            assert served.models[name].state_dict()[key].dtype == tensor.dtype
        for key, parameter in module.named_parameters():
            if not parameter.requires_grad:
                assert_tree_equal(parameter, snapshot["models"][name][key])
    eval_rng = torch.Generator().manual_seed(23)
    served_output = served.sample(9, generator=eval_rng, output_noise=True)
    buffer = io.BytesIO()
    torch.save(policy.state_dict(), buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)
    expected_losses = update(policy)
    expected = policy.state_dict()
    restored = make_policy(serve_average=serve_average)
    identities = {name: {key: id(p) for key, p in module.named_parameters()}
                  for name, module in restored._training_modules().items()}
    restored.load_state_dict(checkpoint)
    assert_tree_equal(checkpoint, restored.state_dict())
    restored_served = restored.served_model(generation_factory=lambda models: lambda G, z: G(
        models["router"](models["encoder"](z))))
    actual_output = restored_served.sample(9, generator=torch.Generator().manual_seed(23), output_noise=True)
    assert_tree_equal(served_output, actual_output)
    assert_tree_equal(expected_losses, update(restored))
    assert_tree_equal(expected, restored.state_dict())
    assert_frozen_copies(restored)
    assert identities == {name: {key: id(p) for key, p in module.named_parameters()}
                          for name, module in restored._training_modules().items()}


@pytest.mark.parametrize("role", ["generator", "critic", "encoder", "router"])
def test_trainable_bf16_heads_still_require_table_precision(role):
    with pytest.raises(ValueError, match="trainable.*dtype"):
        make_policy(wrong_role=role)


def test_trainable_bf16_buffer_is_not_treated_as_frozen():
    with pytest.raises(ValueError, match="trainable.*dtype"):
        make_policy(trainable_buffer=True)


@pytest.mark.parametrize("serve_average", [0., 5.])
def test_standalone_frozen_table_is_copied_without_ema_arithmetic(serve_average):
    policy = make_policy(serve_average=serve_average, frozen_table=True)
    with torch.no_grad():
        policy.table.add_(.1234567)
        policy.averaged_table.zero_()
    update(policy)
    assert_tree_equal(policy.table, policy.averaged_table)


def test_native_trainer_selects_trainable_precision_and_restores_frozen_bf16():
    recipe = get_recipe("e22", num_particles=16, z_dim=2, batch_size=8,
                        output_noise_std=.04)

    def make_trainer():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(717)
            generator = MixedModule(2)
            features = generator.nested["pretrained"]
            features.offset = nn.Parameter(features.offset.bfloat16(), requires_grad=False)
            return GANTrainer(recipe, generator, MixedModule(1), seed=19, serial_backward=True)

    trainer = make_trainer()
    assert next(trainer.G.parameters()).dtype == torch.bfloat16
    assert trainer.dtype == trainer.prior.z.dtype == torch.float32
    trainer.opt_d.record.calls = 799
    real = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 16
    trainer.step(real)
    trainer.step(real)
    assert trainer.row_evidence is not None
    assert trainer.birth_death.counters["evals"] == 1
    assert trainer.birth_death.rows.critic_features.state_dict()["heads"] == ["head"]
    assert trainer.birth_death.space == "critic"
    assert trainer.birth_death.isolation and trainer.birth_death.feature_scale == "std"
    assert_frozen_copies(trainer.policy)

    # Native consumers read the served parameters in G/prior; public consumers
    # can freeze the same bank without replacing the caller's training weights.
    trainer._table_tester().last_decisive = -1
    trainer._serve_apply()
    snapshot = trainer.served_snapshot()
    assert snapshot["source"] == "averaged"
    served = trainer.served_model()
    for noisy in (False, True):
        actual = trainer.sample(9, generator=torch.Generator().manual_seed(23), output_noise=noisy)
        exported = served.sample(9, generator=torch.Generator().manual_seed(23), output_noise=noisy)
        assert_tree_equal(actual, exported)

    buffer = io.BytesIO()
    torch.save(trainer.state_dict(), buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)
    expected_losses = [trainer.step(real), trainer.step(real)]
    expected = trainer.policy.state_dict()
    assert trainer.birth_death.counters["evals"] == 2
    restored = make_trainer()
    restored.load_state_dict(checkpoint)
    assert_tree_equal(snapshot, restored.served_snapshot())
    assert_tree_equal(served.sample(9, generator=torch.Generator().manual_seed(23)),
                      restored.served_model().sample(9, generator=torch.Generator().manual_seed(23)))
    assert_tree_equal(expected_losses, [restored.step(real), restored.step(real)])
    assert_tree_equal(expected, restored.policy.state_dict())
    assert_frozen_copies(restored.policy)


@pytest.mark.parametrize("family", ["models", "averages", "critic_anchor"])
def test_wrong_frozen_checkpoint_dtype_rejects_before_live_mutation(family):
    policy = make_policy()
    update(policy)
    policy._table_tester().last_decisive = -1
    before = policy.state_dict()
    bad = deepcopy(before)
    key = "nested.pretrained.layers.0.weight"
    if family == "critic_anchor":
        anchor = bad["optimizers"][1]["regularizer"]["ema"]
        anchor[key] = anchor[key].float()
    else:
        bad[family]["generator"][key] = bad[family]["generator"][key].float()
    bad["models"]["generator"]["head.weight"].zero_()
    with pytest.raises(ValueError, match="tensor|optimizer"):
        policy.load_state_dict(bad)
    assert_tree_equal(before, policy.state_dict())
