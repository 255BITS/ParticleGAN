"""Reproducible LunarLanderContinuous-v3 flights with an optional down thruster.

The bidirectional variant keeps Gymnasium's Box2D terrain, contact physics,
observation, lateral engine, and positive main engine. Negative main commands
apply an additional *downward* impulse to the lander before the simulator step.
This is an explicit environment variant; use ``bidirectional=False`` for the
unchanged stock LunarLanderContinuous-v3 action semantics.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

VARIANT = "LunarLanderContinuous-v3-bidirectional-main-v1"
STOCK_VARIANT = "LunarLanderContinuous-v3"
GYM_VERSION = "1.2.3"


def make_lunar_env(*, bidirectional: bool = True, render_mode: str | None = None):
    """Create the real Gymnasium Box2D simulator with declared action semantics."""
    import gymnasium as gym
    from gymnasium.envs.box2d.lunar_lander import (MAIN_ENGINE_POWER,
                                                   MAIN_ENGINE_Y_LOCATION, SCALE)

    if gym.__version__ != GYM_VERSION:
        raise RuntimeError(f"Lunar flight adapter requires gymnasium=={GYM_VERSION}; got {gym.__version__}")
    env = gym.make("LunarLander-v3", continuous=True, enable_wind=False,
                   render_mode=render_mode)
    if not bidirectional:
        return env

    class BidirectionalMain(gym.Wrapper):
        variant = VARIANT
        action_semantics = ("main > 0: stock upward engine; main = 0: off; "
                            "main < 0: downward center-of-mass impulse with "
                            "impulse = abs(main) * MAIN_ENGINE_POWER * "
                            "MAIN_ENGINE_Y_LOCATION / SCALE; side: stock engine")

        def step(self, action):
            command = np.asarray(action, dtype=np.float32).reshape(2)
            if not np.all(np.isfinite(command)):
                raise ValueError("Lunar actions must be finite")
            command = np.clip(command, -1., 1.)
            power = max(0., -float(command[0]))
            if power:
                lander = self.unwrapped.lander
                impulse = MAIN_ENGINE_POWER * MAIN_ENGINE_Y_LOCATION / SCALE * power
                lander.ApplyLinearImpulse((0., -impulse),
                                          lander.worldCenter, True)
                command = command.copy()
                command[0] = -1.  # Stock simulator sees main engine off.
            state, reward, terminated, truncated, info = self.env.step(command)
            if power:
                reward -= 0.30 * power  # Match stock engine fuel cost at full power.
            info = dict(info)
            info["downward_main_power"] = power
            info["lunar_variant"] = VARIANT
            return state, float(reward), terminated, truncated, info

    return BidirectionalMain(env)


def _expert(state, *, height_gain: float, velocity_gain: float, down_power: float):
    """Deterministic guidance using normalized Lunar observation coordinates."""
    x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = map(float, state)
    target_angle = np.clip(0.5 * x + vx, -0.4, 0.4)
    target_height = 0.55 * abs(x)
    angle_error = (target_angle - angle) * 0.5 - angular_velocity
    vertical_error = (target_height - y) * height_gain - vy * velocity_gain
    if left_leg or right_leg:
        angle_error = 0.
        vertical_error = -vy * 0.5
    # Match the stock heuristic's stable upward control. Its negative region
    # means "off" under stock semantics; in this variant, only deliberately
    # command down while high above the pad and calling for a strong descent.
    main = float(np.clip(vertical_error * 20. - 1., -1., 1.))
    if main < 0.:
        main = -down_power if not (left_leg or right_leg) and y > 0.4 and vertical_error < -0.05 else 0.
    side = np.clip(-angle_error * 20., -1., 1.)
    return np.asarray([main, side], dtype=np.float32)


def slow_expert(state):
    """Conservative pad guidance used as expert trajectory source."""
    return _expert(state, height_gain=0.5, velocity_gain=0.65, down_power=0.)


def fast_expert(state):
    """Faster descent with a near-pad flare and active downward thrust."""
    return _expert(state, height_gain=0.5, velocity_gain=0.5, down_power=0.3)


def _outcome(env, state, terminated, truncated):
    if terminated:
        base = env.unwrapped
        if base.game_over:
            return "crash"
        if abs(float(state[0])) >= 1.:
            return "out_of_bounds"
        on_pad = base.helipad_x1 <= base.lander.position.x <= base.helipad_x2
        both_legs = all(leg.ground_contact for leg in base.legs)
        if not base.lander.awake and both_legs and on_pad:
            return "successful_landing"
        return "crash"
    return "time_limit" if truncated else None


def rollout_episode(seed: int, policy: Callable, *, bidirectional: bool = True,
                    render: bool = False, frame_stride: int = 3,
                    max_steps: int = 1000) -> dict:
    """Record one actual Box2D flight; ``policy(state)`` returns a 2-vector.

    The first observed leg contact and final sleep are reported separately.
    Arrays include every transition, including the terminal transition.
    """
    if max_steps < 1 or frame_stride < 1:
        raise ValueError("max_steps and frame_stride must be positive")
    env = make_lunar_env(bidirectional=bidirectional,
                         render_mode="rgb_array" if render else None)
    frames, rows = [], []
    first_contact = None
    try:
        state, _ = env.reset(seed=int(seed))
        if render:
            frames.append(env.render())
        for step in range(1, max_steps + 1):
            action = np.asarray(policy(state), dtype=np.float32).reshape(2)
            following, reward, terminated, truncated, info = env.step(action)
            truncated = bool(truncated or (step == max_steps and not terminated))
            rows.append((state, action, following, reward, terminated, truncated,
                         info.get("downward_main_power", 0.)))
            if first_contact is None and np.any(following[6:8] > 0.5):
                first_contact = step
            if render and (step % frame_stride == 0 or terminated or truncated):
                frames.append(env.render())
            state = following
            if terminated or truncated:
                break
        outcome = _outcome(env, state, terminated, truncated)
        if outcome is None:
            outcome = "time_limit"
        data = dict(seed=int(seed), outcome=outcome, steps=len(rows),
                    contact_step=first_contact, return_=float(sum(r[3] for r in rows)),
                    variant=VARIANT if bidirectional else STOCK_VARIANT,
                    states=np.asarray([r[0] for r in rows], np.float32),
                    actions=np.asarray([r[1] for r in rows], np.float32),
                    next_states=np.asarray([r[2] for r in rows], np.float32),
                    rewards=np.asarray([r[3] for r in rows], np.float32),
                    terminated=np.asarray([r[4] for r in rows], bool),
                    truncated=np.asarray([r[5] for r in rows], bool),
                    downward_main_power=np.asarray([r[6] for r in rows], np.float32))
        if render:
            data["frames"] = np.asarray(frames, dtype=np.uint8)
            data["frame_stride"] = int(frame_stride)
        data["return"] = data["return_"]
        return data
    finally:
        env.close()


def collect_expert_episodes(seeds: Iterable[int], behavior: str = "slow",
                            *, bidirectional: bool = True) -> list[dict]:
    """Collect a fixed seed cohort for one named deterministic expert."""
    policies = {"slow": slow_expert, "fast": fast_expert}
    if behavior not in policies:
        raise ValueError("behavior must be 'slow' or 'fast'")
    return [dict(rollout_episode(seed, policies[behavior],
                                 bidirectional=bidirectional), behavior=behavior)
            for seed in seeds]


def summarize_episodes(episodes: Iterable[dict]) -> dict:
    """Success and speed metrics; successful means simulator sleep on the pad."""
    episodes = list(episodes)
    if not episodes:
        raise ValueError("Expected at least one episode")
    successes = [e for e in episodes if e["outcome"] == "successful_landing"]
    contacts = [e["contact_step"] for e in episodes if e["contact_step"] is not None]
    return dict(episodes=len(episodes), success_count=len(successes),
                success_rate=len(successes) / len(episodes),
                mean_steps=float(np.mean([e["steps"] for e in episodes])),
                mean_success_steps=float(np.mean([e["steps"] for e in successes])) if successes else None,
                mean_contact_step=float(np.mean(contacts)) if contacts else None,
                mean_return=float(np.mean([e["return_"] for e in episodes])),
                outcomes={outcome: sum(e["outcome"] == outcome for e in episodes)
                          for outcome in ("successful_landing", "crash", "out_of_bounds", "time_limit", "other_termination")})
