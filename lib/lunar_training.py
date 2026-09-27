"""Learn Lunar Lander dynamics and a conditional RpGAN controller.

All optimization, checkpoints, and evaluation use live weights; there is no EMA.
The policy's adversarial phase has no action-cloning term. Its frozen learned
world model supplies a differentiable successor target from expert transitions.
Rollout success and speed must be measured separately in the actual simulator.

Each training stage is a problem declaration (``LunarWorld``, ``LunarPolicy``)
run by ``benchmarks.toy_runner``: optimizers, their LR schedule, the loss and
the critic penalty all come from the stage's recipe.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from particlegan import get_recipe, init
from benchmarks.toy_runner import Networks, Sample, ToyProblem, View, run


STATE_DIM = 8
ACTION_DIM = 2
MAIN_DEADBAND = .12


def _records(records):
    out = {}
    for key, width in (("states", 8), ("actions", 2), ("next_states", 8)):
        if key not in records:
            raise ValueError(f"Missing {key}")
        value = np.asarray(records[key], dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != width or not np.isfinite(value).all():
            raise ValueError(f"{key} must be finite [N,{width}]")
        out[key] = value
    if len(out["states"]) < 2 or len({len(value) for value in out.values()}) != 1:
        raise ValueError("Records must contain at least two aligned transitions")
    return out


def _mlp(input_dim, output_dim, width=128):
    return nn.Sequential(nn.Linear(input_dim, width), nn.SiLU(),
                         nn.Linear(width, width), nn.SiLU(), nn.Linear(width, output_dim))


def engine_power(actions):
    """Known actuator semantics; dynamics still learns the resulting motion.

    A positive main command ignites at half power. Off and down are distinct
    from that jump. Side commands have a half-power dead zone as in Box2D.
    """
    main = torch.where(actions[:, :1] > 0, .5 + .5 * actions[:, :1], actions[:, :1])
    side = torch.where(actions[:, 1:].abs() > .5, actions[:, 1:], torch.zeros_like(actions[:, 1:]))
    return torch.cat([main, side], dim=-1)


class DynamicsModel(nn.Module):
    """Predict standardized next-state delta from state and continuous action."""

    def __init__(self, state_mean, state_scale, delta_mean, delta_scale, width=128,
                 action_features="engine_power"):
        super().__init__()
        if action_features not in ("raw", "engine_power"):
            raise ValueError("Unknown world-model action features")
        for name, value in (("state_mean", state_mean), ("state_scale", state_scale),
                            ("delta_mean", delta_mean), ("delta_scale", delta_scale)):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32).clone())
        self.net = _mlp(STATE_DIM + ACTION_DIM, STATE_DIM, width)
        self.width = width
        self.action_features = action_features

    def normalized_delta(self, states, actions):
        standardized = (states - self.state_mean) / self.state_scale
        if self.action_features == "engine_power":
            actions = engine_power(actions)
        return self.net(torch.cat([standardized, actions], dim=-1))

    def forward(self, states, actions):
        delta = self.normalized_delta(states, actions)
        return states + delta * self.delta_scale + self.delta_mean


class FastPolicy(nn.Module):
    """Deterministic learned controller; no expert call at inference."""

    def __init__(self, state_mean, state_scale, width=128, main_deadband=MAIN_DEADBAND):
        super().__init__()
        if not 0 <= main_deadband < 1:
            raise ValueError("main_deadband must be in [0,1)")
        self.register_buffer("state_mean", torch.as_tensor(state_mean, dtype=torch.float32).clone())
        self.register_buffer("state_scale", torch.as_tensor(state_scale, dtype=torch.float32).clone())
        self.net = _mlp(STATE_DIM, ACTION_DIM, width)
        self.width = width
        self.main_deadband = float(main_deadband)

    def forward(self, states):
        raw = self.net((states - self.state_mean) / self.state_scale).tanh()
        # Gym's upward engine fires at at least half power for *any* positive
        # command. Exact zero matters, and a continuous network rarely emits
        # it unaided. The straight-through path makes the executed action enter
        # both critic and frozen dynamics losses while retaining gradients.
        main = torch.where(raw[:, :1].abs() < self.main_deadband,
                           torch.zeros_like(raw[:, :1]), raw[:, :1])
        executed = torch.cat([main, raw[:, 1:]], dim=-1)
        return raw + (executed - raw).detach()

    @torch.no_grad()
    def act(self, state):
        value = np.asarray(state, dtype=np.float32)
        if value.shape != (STATE_DIM,):
            raise ValueError("state must have shape [8]")
        tensor = torch.as_tensor(value, device=self.state_mean.device).unsqueeze(0)
        return self(tensor)[0].cpu().numpy()


class ConditionalCritic(nn.Module):
    """Scores an action given the state it answers: ``critic(actions, states)``.

    States are standardized with the policy's own statistics, so the critic
    reads the same state-action features as before; the action is the scored
    sample (``x``) and the state its condition.
    """

    def __init__(self, state_mean, state_scale, width=128):
        super().__init__()
        self.register_buffer("state_mean", torch.as_tensor(state_mean, dtype=torch.float32).clone())
        self.register_buffer("state_scale", torch.as_tensor(state_scale, dtype=torch.float32).clone())
        self.net = _mlp(STATE_DIM + ACTION_DIM, 1, width)

    def forward(self, actions, states):
        standardized = (states - self.state_mean) / self.state_scale
        return self.net(torch.cat([standardized, actions], dim=-1)).squeeze(-1)


def _device(device):
    value = torch.device(device)
    if value.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return value


def _tensor_records(records, device):
    return {key: torch.as_tensor(value, device=device) for key, value in records.items()}


def _rows(values, n, stream):
    return torch.randint(len(values["states"]), (n,), generator=stream, device=values["states"].device)


def _state_statistics(records):
    states = records["states"]
    mean = states.mean(axis=0)
    scale = np.maximum(states.std(axis=0), .05)
    return mean, scale


def _dynamics_statistics(records):
    delta = records["next_states"] - records["states"]
    return delta.mean(axis=0), np.maximum(delta.std(axis=0), .01)


@torch.no_grad()
def _world_metrics(model, values):
    predicted = model(values["states"], values["actions"])
    error = predicted - values["next_states"]
    return {"next_state_mse": float(error.square().mean().cpu()),
            "next_state_mae": float(error.abs().mean().cpu()),
            "next_state_normalized_mse": float((error / model.delta_scale).square().mean().cpu())}


def _lunar_recipe(*, total_steps, batch_size, supervised, **rates):
    """The shipped recipe at this stage's shape, with the lunar family's rates.

    Effective rates match develop's pipeline stage by stage. There the world
    fit (plain Adam) and the BC warm start (recipe G optimizer, never
    rescaled) ran at a constant rate, so supervised stages get network floor
    1.0 (a flat schedule). The RpGAN stage set G and D by hand to the recipe
    cosine over its own ``steps``; here it is a fresh run whose optimizers
    apply that same schedule. (Unmigrated code on the optimizer-owned-schedule
    core instead pushed BC through the GAN horizon, leaving G at the 0.01
    floor for all of RpGAN; that side effect is deliberately not kept.) No stage uses
    critic input or generator output noise: the policy's executed actions have
    exact-zero engine semantics that output noise would destroy. ema_decay=0:
    checkpoints and evaluation use live weights.
    """
    schedule = dict(network_lr_floor=1.) if supervised else {}
    return get_recipe(total_steps=total_steps, batch_size=batch_size, ema_decay=0.,
                      input_noise_std=0., output_noise_std=0., **schedule, **rates)


_WORLD_RATES = dict(lr=1e-3, betas=(.9, .999))
_POLICY_RATES = dict(lr=3e-4, d_lr_mult=2 / 3, betas=(.5, .99))


class LunarWorld(ToyProblem):
    """Supervised dynamics fit: predict the standardized next-state delta.

    A student-only problem (no critic): the generator side is the dynamics
    model and its only loss is the regression below.
    """

    name = "lunar_world"

    def __init__(self, records, validation, *, steps, batch_size, width, action_features, device):
        self.values = _tensor_records(records, device)
        self.scored = _tensor_records(validation, device) if validation is not None else self.values
        self.statistics = (*_state_statistics(records), *_dynamics_statistics(records))
        self.steps, self.batch_size, self.width, self.action_features = steps, batch_size, width, action_features
        self.model = None

    def recipe(self):
        return _lunar_recipe(total_steps=self.steps, batch_size=self.batch_size, supervised=True, **_WORLD_RATES)

    def networks(self, recipe, seed):
        model = DynamicsModel(*self.statistics, self.width, self.action_features)
        self.model = init.deterministic_orthogonal_(model, seed=seed)
        return Networks(generator=self.model, critics={}, prior=None)

    def real(self, n, stream):
        ids = _rows(self.values, n, stream)
        states, actions, successor = (self.values[key][ids] for key in ("states", "actions", "next_states"))
        target = (successor - states - self.model.delta_mean) / self.model.delta_scale
        return Sample(target, condition=(states, actions))

    def fake(self, nets, n, stream, real):
        if real is None:
            ids = _rows(self.scored, n, stream)
            real = Sample(None, condition=(self.scored["states"][ids], self.scored["actions"][ids]))
        return Sample(nets.generator.normalized_delta(*real.condition), condition=real.condition)

    def losses(self, role, nets, real, fake):
        return {"normalized_mse": F.mse_loss(fake.x, real.x)} if role == "generator" else {}

    def metrics(self, model):
        return _world_metrics(model.nets.generator, self.scored)

    def verdict(self, metrics):
        # A dynamics fit has no pass bar of its own; the pipeline's verdict is
        # measured by flying the resulting policy in the real simulator.
        return "UNSCORED"


def train_world_model(records, checkpoint_path, *, validation_records=None, steps=1500,
                      batch_size=256, device="cpu", seed=0, log=None, width=128,
                      action_features="engine_power"):
    """Train on real transitions; return final train and held-out errors."""
    records = _records(records)
    validation = _records(validation_records) if validation_records is not None else None
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    device = _device(device)
    problem = LunarWorld(records, validation, steps=steps, batch_size=batch_size, width=width,
                         action_features=action_features, device=device)
    result = run(problem, seed=seed, device=device, log=log)
    model = problem.model.eval()
    val_values = _tensor_records(validation, device) if validation is not None else None
    metrics = {"train": _world_metrics(model, problem.values),
               "validation": _world_metrics(model, val_values) if val_values is not None else None,
               "updates": steps, "records": len(records["states"]),
               "action_features": action_features, "weight_kind": "live",
               "recipe": result["recipe"]}
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "lunar_world_v2", "weight_kind": "live", "state_dict": model.state_dict(),
                "action_features": action_features,
                "width": width, "metrics": metrics}, path)
    return metrics


def load_world_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    legacy = checkpoint.get("format") == "lunar_world_v1"
    if checkpoint.get("weight_kind", "live" if legacy else None) != "live":
        raise ValueError("Lunar world checkpoints must use live weights")
    if checkpoint.get("format") not in ("lunar_world_v1", "lunar_world_v2"):
        raise ValueError("Expected a lunar_world_v1 or v2 checkpoint")
    if checkpoint["format"] == "lunar_world_v2" and "action_features" not in checkpoint:
        raise ValueError("v2 world checkpoint must declare action features")
    state = checkpoint["state_dict"]
    model = DynamicsModel(state["state_mean"], state["state_scale"],
                          state["delta_mean"], state["delta_scale"], checkpoint["width"],
                          checkpoint.get("action_features", "raw"))
    model.load_state_dict(state)
    return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def _policy_metrics(policy, world, values):
    states, actions, successor = (values[key] for key in ("states", "actions", "next_states"))
    output = policy(states)
    target_delta = successor - states
    predicted_delta = world(states, output) - states
    return {"action_mse": float(F.mse_loss(output, actions).cpu()),
            "world_successor_mse": float(F.mse_loss(predicted_delta, target_delta).cpu()),
            "world_successor_normalized_mse": float(F.mse_loss(predicted_delta / world.delta_scale,
                                                              target_delta / world.delta_scale).cpu())}


def _fresh_policy(statistics, width):
    return init.deterministic_orthogonal_(FastPolicy(*statistics, width), seed=0)


class LunarPolicy(ToyProblem):
    """The landing policy on expert transitions, as one of two stages.

    ``adversarial=False``: behaviour-cloning warm start (student-only, action
    MSE). ``adversarial=True``: conditional RpGAN -- the critic scores actions
    given their state -- plus the frozen world model's successor loss, with no
    action-cloning term. ``policy`` is trained in place; when None the first
    stage builds a fresh one.
    """

    def __init__(self, records, validation, world, policy, *, adversarial, steps, batch_size, width,
                 world_weight, adversarial_weight, device):
        self.name = "lunar_policy_rpgan" if adversarial else "lunar_policy_bc"
        self.values = _tensor_records(records, device)
        self.scored = _tensor_records(validation, device) if validation is not None else self.values
        self.statistics = _state_statistics(records)
        self.world, self.policy, self.adversarial = world, policy, adversarial
        self.steps, self.batch_size, self.width = steps, batch_size, width
        self.world_weight, self.adversarial_weight = world_weight, adversarial_weight

    def recipe(self):
        return _lunar_recipe(total_steps=self.steps, batch_size=self.batch_size,
                             supervised=not self.adversarial, **_POLICY_RATES)

    def networks(self, recipe, seed):
        if self.policy is None:
            # A loaded initial policy keeps its trained weights; a fresh one is initialized here.
            self.policy = _fresh_policy(self.statistics, self.width)
        critics = {}
        if self.adversarial:
            critic = ConditionalCritic(self.policy.state_mean, self.policy.state_scale, self.width)
            critics = init.deterministic_orthogonal_(critic, seed=1)
        return Networks(generator=self.policy, critics=critics, prior=None)

    def real(self, n, stream):
        ids = _rows(self.values, n, stream)
        return Sample(self.values["actions"][ids], condition=(self.values["states"][ids],), indices=ids)

    def fake(self, nets, n, stream, real):
        states = real.condition[0] if real is not None else self.scored["states"][_rows(self.scored, n, stream)]
        return Sample(nets.generator(states), condition=(states,))

    def views(self, nets, real, fake):
        return [View("critic", real.x, fake.x, real.condition, self.adversarial_weight)] if self.adversarial else []

    def losses(self, role, nets, real, fake):
        if role != "generator":
            return {}
        if not self.adversarial:
            return {"action_mse": F.mse_loss(fake.x, real.x)}
        states, successor = real.condition[0], self.values["next_states"][real.indices]
        predicted = self.world(states, fake.x)
        # Continuous coordinates have useful action gradients. Contact bits are
        # discontinuous and their rare flips should not dominate this signal.
        scale = self.world.delta_scale[:6]
        target_delta = (successor[:, :6] - states[:, :6]) / scale
        predicted_delta = (predicted[:, :6] - states[:, :6]) / scale
        return {"world": self.world_weight * F.mse_loss(predicted_delta, target_delta)}

    def metrics(self, model):
        return _policy_metrics(model.nets.generator, self.world, self.scored)

    def verdict(self, metrics):
        # Landing success and speed are measured by flying in the real simulator.
        return "UNSCORED"


def train_fast_policy(records, world_checkpoint, checkpoint_path, *, validation_records=None,
                      initial_policy=None, warmup_steps=8000, steps=400, batch_size=256,
                      device="cpu", seed=0, log=None, width=128, world_weight=2.5,
                      adversarial_weight=1.):
    """BC initialization followed by conditional paired RpGAN and world gradients.

    `steps` is the number of adversarial updates after `warmup_steps`. The
    successor loss compares world-model predictions for generated actions with
    expert successors, without directly penalizing action error in that phase.
    Both stages run on the shared toy runner under their own recipe.
    """
    records = _records(records)
    validation = _records(validation_records) if validation_records is not None else None
    if warmup_steps < 0 or steps < 1 or batch_size < 1:
        raise ValueError("invalid training step or batch size")
    if world_weight <= 0 or adversarial_weight <= 0:
        raise ValueError("world and adversarial weights must be positive")
    device = _device(device)
    world = load_world_model(world_checkpoint, device)
    policy = None
    if initial_policy is not None:
        policy = load_fast_policy(initial_policy, device)
        if policy.width != width:
            raise ValueError("initial policy width differs from requested width")
        policy.requires_grad_(True)
    shared = dict(steps=warmup_steps, batch_size=batch_size, width=width, world_weight=world_weight,
                  adversarial_weight=adversarial_weight, device=device)
    if warmup_steps:
        stage = LunarPolicy(records, validation, world, policy, adversarial=False, **shared)
        run(stage, seed=seed, device=device, log=log)
        policy = stage.policy
    if policy is None:
        # No warm start: the RpGAN stage starts from (and reports) a fresh policy.
        policy = _fresh_policy(_state_statistics(records), width).to(device)
    shared["steps"] = steps
    stage = LunarPolicy(records, validation, world, policy, adversarial=True, **shared)
    before_adversarial = _policy_metrics(policy, world, stage.scored)
    run(stage, seed=seed, device=device, log=log)
    policy = stage.policy.eval()
    train_values = stage.values
    val_values = _tensor_records(validation, device) if validation is not None else None
    recipe = stage.recipe()
    metrics = {"train": _policy_metrics(policy, world, train_values),
               "validation": _policy_metrics(policy, world, val_values) if val_values is not None else None,
               "before_adversarial": before_adversarial,
               "warmup_updates": warmup_steps, "rpgan_updates": steps,
               "world_weight": world_weight, "adversarial_weight": adversarial_weight,
               "records": len(records["states"]), "recipe": recipe.to_dict(), "weight_kind": "live",
               "optimizers": {"generator": {"name": "recipe.make_generator_optimizer", "lr": recipe.lr,
                                            "betas": recipe.betas},
                              "discriminator": {"name": "recipe.make_critic_optimizer",
                                                "lr": recipe.lr * recipe.d_lr_mult, "betas": recipe.betas}},
               "main_deadband": policy.main_deadband,
               "initial_policy": str(initial_policy) if initial_policy is not None else None,
               "world_checkpoint": str(world_checkpoint)}
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "lunar_policy_rpgan_v3", "weight_kind": "live", "state_dict": policy.state_dict(),
                "width": width, "main_deadband": policy.main_deadband, "metrics": metrics}, path)
    return metrics


train_policy = train_fast_policy


def load_fast_policy(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    legacy = checkpoint.get("format") in ("lunar_policy_rpgan_v1", "lunar_policy_rpgan_v2")
    if checkpoint.get("weight_kind", "live" if legacy else None) != "live":
        raise ValueError("Lunar policy checkpoints must use live weights")
    if checkpoint.get("format") not in ("lunar_policy_rpgan_v1", "lunar_policy_rpgan_v2", "lunar_policy_rpgan_v3"):
        raise ValueError("Expected a lunar_policy_rpgan_v1, v2 or v3 checkpoint")
    if checkpoint["format"] != "lunar_policy_rpgan_v1" and "main_deadband" not in checkpoint:
        raise ValueError("v2/v3 policy checkpoint must declare its main-engine deadband")
    state = checkpoint["state_dict"]
    # Early v1 pilots emitted raw main commands. Later v1 checkpoints included
    # an explicit deadband; v2 requires it, so neither silently changes action
    # semantics when the module default changes.
    deadband = checkpoint.get("main_deadband", 0.)
    policy = FastPolicy(state["state_mean"], state["state_scale"], checkpoint["width"], deadband)
    policy.load_state_dict(state)
    return policy.to(device).eval().requires_grad_(False)
