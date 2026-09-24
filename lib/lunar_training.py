"""Learn Lunar Lander dynamics and a conditional RpGAN controller.

All optimization, checkpoints, and evaluation use live weights; there is no EMA.
The policy's adversarial phase has no action-cloning term. Its frozen learned
world model supplies a differentiable successor target from expert transitions.
Rollout success and speed must be measured separately in the actual simulator.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from particlegan import get_recipe


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
    def __init__(self, width=128):
        super().__init__()
        self.net = _mlp(STATE_DIM + ACTION_DIM, 1, width)

    def forward(self, state_action):
        return self.net(state_action).squeeze(-1)


def _device(device):
    value = torch.device(device)
    if value.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return value


def _tensor_records(records, device):
    return {key: torch.as_tensor(value, device=device) for key, value in records.items()}


def _emit(log, message):
    if log is not None:
        log(message)


def _sample(values, size, rng):
    ids = torch.randint(len(values["states"]), (size,), generator=rng, device=values["states"].device)
    return {key: value[ids] for key, value in values.items()}


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


def train_world_model(records, checkpoint_path, *, validation_records=None, steps=1500,
                      batch_size=256, device="cpu", seed=0, log=None, width=128,
                      action_features="engine_power"):
    """Train on real transitions; return final train and held-out errors."""
    records = _records(records)
    validation = _records(validation_records) if validation_records is not None else None
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    device = _device(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        mean, scale = _state_statistics(records)
        delta_mean, delta_scale = _dynamics_statistics(records)
        model = DynamicsModel(mean, scale, delta_mean, delta_scale, width, action_features).to(device)
    train_values = _tensor_records(records, device)
    val_values = _tensor_records(validation, device) if validation is not None else None
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = torch.Generator(device=device).manual_seed(seed + 11)
    for step in range(1, steps + 1):
        batch = _sample(train_values, batch_size, rng)
        target = (batch["next_states"] - batch["states"] - model.delta_mean) / model.delta_scale
        prediction = model.normalized_delta(batch["states"], batch["actions"])
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step == 1 or step == steps or step % max(steps // 10, 1) == 0:
            _emit(log, f"world step={step}/{steps} normalized_mse={loss.item():.5f}")
    model.eval()
    metrics = {"train": _world_metrics(model, train_values),
               "validation": _world_metrics(model, val_values) if val_values is not None else None,
               "updates": steps, "records": len(records["states"]),
               "action_features": action_features, "weight_kind": "live"}
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


def _critic_input(policy, states, actions):
    return torch.cat([(states - policy.state_mean) / policy.state_scale, actions], dim=-1)


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


def train_fast_policy(records, world_checkpoint, checkpoint_path, *, validation_records=None,
                      initial_policy=None, warmup_steps=8000, steps=400, batch_size=256,
                      device="cpu", seed=0, log=None, width=128, world_weight=2.5,
                      adversarial_weight=1.):
    """BC initialization followed by conditional paired RpGAN and world gradients.

    `steps` is the number of adversarial updates after `warmup_steps`. The
    successor loss compares world-model predictions for generated actions with
    expert successors, without directly penalizing action error in that phase.
    """
    records = _records(records)
    validation = _records(validation_records) if validation_records is not None else None
    if warmup_steps < 0 or steps < 1 or batch_size < 1:
        raise ValueError("invalid training step or batch size")
    if world_weight <= 0 or adversarial_weight <= 0:
        raise ValueError("world and adversarial weights must be positive")
    device = _device(device)
    world = load_world_model(world_checkpoint, device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if initial_policy is not None:
            policy = load_fast_policy(initial_policy, device)
            if policy.width != width:
                raise ValueError("initial policy width differs from requested width")
            policy.requires_grad_(True)
        else:
            mean, scale = _state_statistics(records)
            policy = FastPolicy(mean, scale, width).to(device)
        critic = ConditionalCritic(width).to(device)
    train_values = _tensor_records(records, device)
    val_values = _tensor_records(validation, device) if validation is not None else None
    optimizer_g = torch.optim.Adam(policy.parameters(), lr=3e-4, betas=(.5, .99))
    optimizer_d = torch.optim.Adam(critic.parameters(), lr=2e-4, betas=(.5, .99))
    # Only loss/cap factories are used, but also disable the unused recipe EMA
    # setting explicitly so checkpoint metadata cannot suggest averaged weights.
    recipe = get_recipe(total_steps=steps, batch_size=batch_size, reg_coeff=1., reg_every=4,
                        ema_decay=0.)
    gan = recipe.make_loss()
    penalty = recipe.make_gradient_penalty()
    rng = torch.Generator(device=device).manual_seed(seed + 29)
    for step in range(1, warmup_steps + 1):
        batch = _sample(train_values, batch_size, rng)
        loss = F.mse_loss(policy(batch["states"]), batch["actions"])
        optimizer_g.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 5.)
        optimizer_g.step()
        if step == 1 or step == warmup_steps or step % max(warmup_steps // 10, 1) == 0:
            _emit(log, f"policy BC step={step}/{warmup_steps} action_mse={loss.item():.5f}")
    before_adversarial = _policy_metrics(policy, world, val_values or train_values)
    for step in range(1, steps + 1):
        batch = _sample(train_values, batch_size, rng)
        states, actions, successor = (batch[key] for key in ("states", "actions", "next_states"))
        real = _critic_input(policy, states, actions)
        with torch.no_grad():
            fake_actions = policy(states)
        fake = _critic_input(policy, states, fake_actions)
        d_loss = gan.d_loss(critic(real), critic(fake))
        d_loss = d_loss + penalty(critic, real, fake, step=step)
        optimizer_d.zero_grad(set_to_none=True)
        d_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 5.)
        optimizer_d.step()

        critic.requires_grad_(False)
        generated = policy(states)
        fake = _critic_input(policy, states, generated)
        with torch.no_grad():
            real_logits = critic(real)
        adversarial = gan.g_loss(critic(fake), real_logits)
        predicted = world(states, generated)
        # Continuous coordinates have useful action gradients. Contact bits are
        # discontinuous and their rare flips should not dominate this signal.
        target_delta = (successor[:, :6] - states[:, :6]) / world.delta_scale[:6]
        predicted_delta = (predicted[:, :6] - states[:, :6]) / world.delta_scale[:6]
        successor_loss = F.mse_loss(predicted_delta, target_delta)
        g_loss = adversarial_weight * adversarial + world_weight * successor_loss
        optimizer_g.zero_grad(set_to_none=True)
        g_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 5.)
        optimizer_g.step()
        critic.requires_grad_(True)
        if step == 1 or step == steps or step % max(steps // 10, 1) == 0:
            _emit(log, f"policy RpGAN step={step}/{steps} d={d_loss.item():.5f} "
                       f"g_adv={adversarial.item():.5f} world={successor_loss.item():.5f}")
    policy.eval()
    metrics = {"train": _policy_metrics(policy, world, train_values),
               "validation": _policy_metrics(policy, world, val_values) if val_values is not None else None,
               "before_adversarial": before_adversarial,
               "warmup_updates": warmup_steps, "rpgan_updates": steps,
               "world_weight": world_weight, "adversarial_weight": adversarial_weight,
               "records": len(records["states"]), "recipe": recipe.to_dict(), "weight_kind": "live",
               "optimizers": {"generator": {"name": "Adam", "lr": 3e-4, "betas": (.5, .99)},
                              "discriminator": {"name": "Adam", "lr": 2e-4, "betas": (.5, .99)}},
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
