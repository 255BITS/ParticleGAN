"""Slow→fast Lunar pairs and the speed score that refuses crash shortcuts.

Successful landings from one frozen controller are split by steps-to-land.
Crashes, timeouts, and flyaways never enter either pool. Each kept pair is a
slow landing and a strictly faster landing whose start and terrain are close.
Training rows sit on the slow trajectory: neutral is the slow action, target
is the fast action at the nearest state. There is no second policy to query,
and there is no kinematic safe-fast cost in this file.
"""
import json
from pathlib import Path

import numpy as np

from lib.gym_control_evaluation import OFF_ACTION

# Same reset seeds as lib.gym_control_evaluation.freeze_protocol.
VALIDATION_SEEDS = tuple(range(391000, 391020))
TEST_SEEDS = tuple(range(491000, 491050))
EVAL_SEEDS = frozenset(VALIDATION_SEEDS + TEST_SEEDS)
COLLECTION_SEED_START = 591000
SUCCESS = "successful_landing"
OUTCOMES = ("successful_landing", "crash", "out_of_bounds", "time_limit")
# Continuous Lunar state: x, y, vx, vy, angle, angular velocity.
STATE_WEIGHT = np.array([1., 1., .5, .5, .25, .25], dtype=np.float32)
PAIR_ARRAYS = (
    "states", "previous_actions", "neutral_actions", "target_actions", "terrain",
    "slow_steps", "fast_steps", "slow_seed", "fast_seed", "slow_index", "fast_index",
    "state_distance",
)


def protocol_seeds(split):
    if split == "validation":
        return VALIDATION_SEEDS
    if split == "test":
        return TEST_SEEDS
    raise ValueError("split must be validation or test")


def assert_collection_seeds(seeds):
    seeds = [int(seed) for seed in seeds]
    if len(seeds) != len(set(seeds)):
        raise ValueError("collection seeds must be unique")
    overlap = EVAL_SEEDS.intersection(seeds)
    if overlap:
        shown = ", ".join(str(seed) for seed in sorted(overlap)[:5])
        raise ValueError(f"collection seeds overlap the shared eval protocol: {shown}")
    return seeds


def is_success(episode):
    """Fail closed. A short crash cannot be relabeled into the fast set."""
    if episode.get("outcome") != SUCCESS:
        return False
    if episode.get("game_over", True) or episode.get("lander_awake", True):
        return False
    if not episode.get("terminated", False) or episode.get("truncated", False):
        return False
    return True


def _finite_vector(value, width, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (width,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector of length {width}")
    return array


def _trajectory(episode):
    states = np.asarray(episode["states"], dtype=np.float32)
    actions = np.asarray(episode["actions"], dtype=np.float32)
    steps = int(episode["steps"])
    if states.ndim != 2 or states.shape[1] != 8 or actions.shape != (len(states), 2):
        raise ValueError(f"seed {episode.get('seed')} needs states [T,8] and actions [T,2]")
    if len(states) != steps or steps < 1:
        raise ValueError(f"seed {episode.get('seed')} trajectory length must equal steps-to-land")
    if not np.isfinite(states).all() or not np.isfinite(actions).all() or (np.abs(actions) > 1).any():
        raise ValueError(f"seed {episode.get('seed')} has a nonfinite or out-of-range action")
    initial = _finite_vector(episode["initial_state"], 8, "initial_state")
    terrain = _finite_vector(episode["terrain"], 11, "terrain")
    return states, actions, initial, terrain


def start_distance(slow, fast):
    left = np.asarray(slow["initial_state"], dtype=np.float32)[:6]
    right = np.asarray(fast["initial_state"], dtype=np.float32)[:6]
    return float(np.linalg.norm((left - right) * STATE_WEIGHT))


def terrain_distance(slow, fast):
    left = np.asarray(slow["terrain"], dtype=np.float32)
    right = np.asarray(fast["terrain"], dtype=np.float32)
    return float(np.linalg.norm(left - right))


def _quantile_cut(steps, quantile):
    return float(np.quantile(np.asarray(steps, dtype=np.float64), quantile))


def split_success_pools(episodes, fast_quantile=0.30, slow_quantile=0.70):
    """Successes only. The fast pool is strictly sooner than the slow pool."""
    if not 0 < float(fast_quantile) < float(slow_quantile) < 1:
        raise ValueError("need 0 < fast_quantile < slow_quantile < 1")
    dropped = {name: 0 for name in OUTCOMES if name != SUCCESS}
    dropped["other"] = 0
    successes = []
    seen = set()
    for episode in episodes:
        seed = int(episode["seed"])
        if seed in seen:
            raise ValueError(f"duplicate rollout seed {seed}")
        seen.add(seed)
        if is_success(episode):
            _trajectory(episode)
            successes.append(episode)
            continue
        outcome = episode.get("outcome")
        dropped[outcome if outcome in dropped else "other"] += 1
    if len(successes) < 4:
        raise ValueError(f"need at least 4 successful landings to split pools, got {len(successes)}")
    steps = [int(episode["steps"]) for episode in successes]
    fast_cut = _quantile_cut(steps, fast_quantile)
    slow_cut = _quantile_cut(steps, slow_quantile)
    fast = [episode for episode in successes if int(episode["steps"]) <= fast_cut]
    slow = [episode for episode in successes if int(episode["steps"]) >= slow_cut]
    fast_seeds = {int(episode["seed"]) for episode in fast}
    slow = [episode for episode in slow if int(episode["seed"]) not in fast_seeds]
    if not fast or not slow:
        raise ValueError("quantile split produced an empty slow or fast pool; collect more landings")
    if max(int(episode["steps"]) for episode in fast) >= min(int(episode["steps"]) for episode in slow):
        raise ValueError("fast pool is not strictly sooner than the slow pool")
    for pool in (fast, slow):
        if any(not is_success(episode) for episode in pool):
            raise RuntimeError("a non-success entered a slow/fast pool")
    return fast, slow, dropped, successes


def match_episodes(slow, fast, max_start_distance, max_terrain_distance):
    """One-to-one greedy match, slowest first. Far starts and terrain stay unpaired."""
    unused = set(range(len(fast)))
    order = sorted(range(len(slow)), key=lambda index: int(slow[index]["steps"]), reverse=True)
    pairs, unmatched = [], []
    nearest = []
    for index in order:
        best = None
        best_key = None
        closest = None
        for other in unused:
            start = start_distance(slow[index], fast[other])
            terrain = terrain_distance(slow[index], fast[other])
            key = (start + terrain, start, terrain, other)
            if closest is None or key < closest:
                closest = key
            if start > max_start_distance or terrain > max_terrain_distance:
                continue
            if best_key is None or key < best_key:
                best_key = key
                best = other
        nearest.append(dict(slow_seed=int(slow[index]["seed"]),
                            nearest_start=None if closest is None else closest[1],
                            nearest_terrain=None if closest is None else closest[2]))
        if best is None:
            unmatched.append(int(slow[index]["seed"]))
            continue
        unused.remove(best)
        pairs.append(dict(slow=slow[index], fast=fast[best], start_distance=best_key[1],
                          terrain_distance=best_key[2]))
    return pairs, unmatched, nearest


def _nearest_states(slow_states, fast_states):
    left = slow_states[:, None, :6] * STATE_WEIGHT
    right = fast_states[None, :, :6] * STATE_WEIGHT
    distance = np.linalg.norm(left - right, axis=-1)
    distance += 0.1 * np.abs(slow_states[:, None, 6:8] - fast_states[None, :, 6:8]).sum(-1)
    index = distance.argmin(axis=1)
    rows = np.arange(len(slow_states))
    return index, distance[rows, index]


def build_slow_fast_pairs(episodes, fast_quantile=0.30, slow_quantile=0.70,
                          max_start_distance=0.50, max_terrain_distance=0.50,
                          max_state_distance=0.75):
    """Matched (slow, fast) rows. Crashes never become targets."""
    for name, value in (("max_start_distance", max_start_distance),
                        ("max_terrain_distance", max_terrain_distance),
                        ("max_state_distance", max_state_distance)):
        if type(value) is bool or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative number")
    fast, slow, dropped, successes = split_success_pools(episodes, fast_quantile, slow_quantile)
    matched, unmatched, nearest = match_episodes(slow, fast, float(max_start_distance),
                                                 float(max_terrain_distance))
    columns = {name: [] for name in PAIR_ARRAYS}
    pair_rows = []
    dropped_states = 0
    for pair_id, pair in enumerate(matched):
        slow_states, slow_actions, _, terrain = _trajectory(pair["slow"])
        fast_states, fast_actions, _, _ = _trajectory(pair["fast"])
        if int(pair["fast"]["steps"]) >= int(pair["slow"]["steps"]):
            raise RuntimeError("matched pair is not strictly faster")
        choice, distance = _nearest_states(slow_states, fast_states)
        previous = np.empty_like(slow_actions)
        previous[0] = OFF_ACTION
        if len(slow_actions) > 1:
            previous[1:] = slow_actions[:-1]
        kept = 0
        for row in range(len(slow_states)):
            if float(distance[row]) > max_state_distance:
                dropped_states += 1
                continue
            fast_row = int(choice[row])
            columns["states"].append(slow_states[row])
            columns["previous_actions"].append(previous[row])
            columns["neutral_actions"].append(slow_actions[row])
            columns["target_actions"].append(fast_actions[fast_row])
            columns["terrain"].append(terrain)
            columns["slow_steps"].append(int(pair["slow"]["steps"]))
            columns["fast_steps"].append(int(pair["fast"]["steps"]))
            columns["slow_seed"].append(int(pair["slow"]["seed"]))
            columns["fast_seed"].append(int(pair["fast"]["seed"]))
            columns["slow_index"].append(row)
            columns["fast_index"].append(fast_row)
            columns["state_distance"].append(float(distance[row]))
            kept += 1
        if kept:
            pair_rows.append(dict(slow_seed=int(pair["slow"]["seed"]), fast_seed=int(pair["fast"]["seed"]),
                                  slow_steps=int(pair["slow"]["steps"]), fast_steps=int(pair["fast"]["steps"]),
                                  start_distance=pair["start_distance"], terrain_distance=pair["terrain_distance"],
                                  rows=kept))
    if not columns["states"]:
        raise ValueError("no matched slow/fast rows; loosen the distance caps or collect more landings. "
                         f"nearest={_distance_summary(nearest)} unmatched_slow={len(unmatched)} "
                         f"dropped_states={dropped_states}")
    arrays = _stack_columns(columns)
    if np.any(arrays["fast_steps"] >= arrays["slow_steps"]):
        raise RuntimeError("a fast row is not strictly sooner than its slow partner")
    edit = np.abs(arrays["target_actions"] - arrays["neutral_actions"]).mean()
    manifest = dict(
        format="slow_fast_pairs_v1",
        successes=len(successes),
        fast_pool=len(fast), slow_pool=len(slow), pairs=len(pair_rows), rows=int(len(arrays["states"])),
        crashes_excluded=dropped["crash"], timeouts_excluded=dropped["time_limit"],
        oob_excluded=dropped["out_of_bounds"], other_excluded=dropped["other"],
        fast_steps_max=int(arrays["fast_steps"].max()), slow_steps_min=int(arrays["slow_steps"].min()),
        mean_action_edit=float(edit), dropped_state_rows=dropped_states, unmatched_slow=len(unmatched),
        nearest=_distance_summary(nearest), pairs_detail=pair_rows,
        fast_quantile=float(fast_quantile), slow_quantile=float(slow_quantile),
        max_start_distance=float(max_start_distance), max_terrain_distance=float(max_terrain_distance),
        max_state_distance=float(max_state_distance),
        outcomes_in_pools=[SUCCESS],
        neutral="slow action on the slow trajectory",
        target="fast action at the nearest state on the matched faster landing",
        speed_mechanism="paired successful actions; not the safe-fast kinematic cost",
    )
    return dict(arrays=arrays, manifest=manifest)


def _stack_columns(columns):
    arrays = dict(
        states=np.stack(columns["states"]).astype(np.float32),
        previous_actions=np.stack(columns["previous_actions"]).astype(np.float32),
        neutral_actions=np.stack(columns["neutral_actions"]).astype(np.float32),
        target_actions=np.stack(columns["target_actions"]).astype(np.float32),
        terrain=np.stack(columns["terrain"]).astype(np.float32),
        slow_steps=np.asarray(columns["slow_steps"], dtype=np.int32),
        fast_steps=np.asarray(columns["fast_steps"], dtype=np.int32),
        slow_seed=np.asarray(columns["slow_seed"], dtype=np.int64),
        fast_seed=np.asarray(columns["fast_seed"], dtype=np.int64),
        slow_index=np.asarray(columns["slow_index"], dtype=np.int32),
        fast_index=np.asarray(columns["fast_index"], dtype=np.int32),
        state_distance=np.asarray(columns["state_distance"], dtype=np.float32),
    )
    _validate_pairs(arrays)
    return arrays


def _distance_summary(nearest):
    if not nearest:
        return dict(count=0)
    start = np.asarray([row["nearest_start"] for row in nearest], dtype=np.float64)
    terrain = np.asarray([row["nearest_terrain"] for row in nearest], dtype=np.float64)
    return dict(count=len(nearest), start_p50=float(np.median(start)), start_p90=float(np.quantile(start, 0.9)),
                terrain_p50=float(np.median(terrain)), terrain_p90=float(np.quantile(terrain, 0.9)))


def _validate_pairs(arrays):
    missing = [name for name in PAIR_ARRAYS if name not in arrays]
    if missing:
        raise ValueError(f"pairs file is missing {missing}")
    rows = len(arrays["states"])
    if rows < 2:
        raise ValueError("pairs need at least 2 rows so the edit scale is defined")
    expected = {
        "states": (rows, 8), "previous_actions": (rows, 2), "neutral_actions": (rows, 2),
        "target_actions": (rows, 2), "terrain": (rows, 11), "slow_steps": (rows,),
        "fast_steps": (rows,), "slow_seed": (rows,), "fast_seed": (rows,),
        "slow_index": (rows,), "fast_index": (rows,), "state_distance": (rows,),
    }
    for name, shape in expected.items():
        if tuple(arrays[name].shape) != shape:
            raise ValueError(f"{name} has shape {arrays[name].shape}, expected {shape}")
    finite = ("states", "previous_actions", "neutral_actions", "target_actions", "terrain", "state_distance")
    if any(not np.isfinite(arrays[name]).all() for name in finite):
        raise ValueError("pairs contain a nonfinite value")
    for name in ("previous_actions", "neutral_actions", "target_actions"):
        if (np.abs(arrays[name]) > 1).any():
            raise ValueError(f"{name} must stay inside [-1, 1]")
    if np.any(arrays["fast_steps"] >= arrays["slow_steps"]):
        raise ValueError("refusing pairs whose fast member is not strictly sooner")
    if np.any(arrays["slow_seed"] == arrays["fast_seed"]):
        raise ValueError("a pair must come from two rollouts")


def save_pairs(path, built):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **built["arrays"])
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(built["manifest"], indent=2) + "\n")
    return path, sidecar


def load_pairs(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    _validate_pairs(arrays)
    sidecar = path.with_suffix(".json")
    manifest = json.loads(sidecar.read_text()) if sidecar.is_file() else None
    if manifest is not None:
        if manifest.get("format") != "slow_fast_pairs_v1":
            raise ValueError("pairs manifest is not slow_fast_pairs_v1")
        if manifest.get("outcomes_in_pools") != [SUCCESS]:
            raise ValueError("pairs manifest must list only successful landings")
        if int(manifest["crashes_excluded"]) < 0:
            raise ValueError("crashes_excluded must be nonnegative")
    return arrays, manifest


def mean_success_steps(episodes):
    """Mean steps among successful landings. Crashes do not make this smaller."""
    steps = [int(episode["steps"]) for episode in episodes if episode.get("outcome") == SUCCESS]
    if not steps:
        return None
    return float(np.mean(steps))


def speed_stats(episodes):
    if not episodes:
        raise ValueError("no episodes")
    counts = {name: sum(episode.get("outcome") == name for episode in episodes) for name in OUTCOMES}
    successes = counts[SUCCESS]
    return dict(
        episodes=len(episodes),
        landing_count=successes,
        mean_success_steps=mean_success_steps(episodes),
        crash_count=counts["crash"],
        timeout_count=counts["time_limit"],
        oob_count=counts["out_of_bounds"],
        mean_episode_steps=float(np.mean([int(episode["steps"]) for episode in episodes])),
    )


def speed_decision(candidate, baseline):
    """A shorter contact that crashes, or that lands less often, is not a speed win."""
    reasons = []
    if candidate["landing_count"] < baseline["landing_count"]:
        reasons.append("landing count fell")
    if candidate["crash_count"] > baseline["crash_count"]:
        reasons.append("crash count rose")
    if candidate["oob_count"] > baseline["oob_count"]:
        reasons.append("out-of-bounds count rose")
    if candidate["mean_success_steps"] is None:
        reasons.append("no successful landings")
    elif baseline["mean_success_steps"] is None:
        reasons.append("baseline has no successful landings")
    elif candidate["mean_success_steps"] >= baseline["mean_success_steps"]:
        reasons.append("not faster among successes")
    eligible = not reasons
    return dict(eligible=eligible, reasons=reasons or
                ["faster among successes without lost landings or extra crashes"])


def synthetic_episodes():
    """Tiny successful and crashing rollouts for the CPU smoke. Not Lunar physics."""
    def episode(seed, steps, outcome, x, action, terrain=0., success=False):
        states = np.zeros((steps, 8), dtype=np.float32)
        states[:, 0] = x
        states[:, 1] = np.linspace(1.2, 0.05, steps)
        actions = np.tile(np.asarray(action, dtype=np.float32), (steps, 1))
        return dict(seed=seed, steps=steps, outcome=outcome, states=states, actions=actions,
                    initial_state=states[0].tolist(), terrain=[terrain] * 11,
                    game_over=not success, lander_awake=not success, terminated=outcome != "time_limit",
                    truncated=outcome == "time_limit")

    return [
        episode(1, 10, SUCCESS, 0.00, [0.2, 0.0], success=True),
        episode(2, 12, SUCCESS, 0.02, [0.4, 0.1], success=True),
        episode(3, 40, SUCCESS, 0.01, [-0.2, 0.0], success=True),
        episode(4, 48, SUCCESS, 0.03, [-0.4, -0.1], success=True),
        episode(5, 6, "crash", 0.00, [1.0, 1.0], success=False),
        episode(6, 7, "time_limit", 0.00, [0.0, 0.0], success=False),
        episode(7, 8, SUCCESS, 3.0, [0.9, 0.9], terrain=1.0, success=True),
    ]


def read_jsonl(path):
    episodes = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    if not episodes:
        raise ValueError(f"no rollouts in {path}")
    return episodes
