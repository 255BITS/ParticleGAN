"""Same-seed slow→fast Lunar pairs and the speed score that refuses crash shortcuts.

The safe teacher and the fast teacher each roll a seed. A pair is kept only
when both land and the fast landing is strictly sooner. Rows are aligned by
progress ``t/T`` on that shared landing. Cross-episode nearest-state matching
is disabled. Crashes, timeouts, and flyaways never enter the fast set.
There is no kinematic safe-fast cost in this file.
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
# Fast-teacher probes. Disjoint from validation, test, and collection.
PROBE_SEED_START = 581000
STRANGER_ARCHIVE_NAME = "pairs_stranger_do_not_train.npz"
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


def progress_index(slow_len, fast_len, index):
    """Fast-trajectory index at the same fraction of the landing."""
    if slow_len < 1 or fast_len < 1:
        raise ValueError("progress alignment needs a nonempty trajectory")
    if slow_len == 1:
        return 0
    fraction = index / (slow_len - 1)
    return int(round(fraction * (fast_len - 1)))


def _index_successes(episodes, role):
    """One record per seed. Non-successes are counted and dropped."""
    dropped = {name: 0 for name in OUTCOMES if name != SUCCESS}
    dropped["other"] = 0
    by_seed = {}
    for episode in episodes:
        seed = int(episode["seed"])
        if seed in by_seed:
            raise ValueError(f"duplicate {role} rollout seed {seed}")
        if is_success(episode):
            _trajectory(episode)
            by_seed[seed] = episode
            continue
        outcome = episode.get("outcome")
        dropped[outcome if outcome in dropped else "other"] += 1
    return by_seed, dropped


def _aligned_distance(slow_state, fast_state):
    gap = (slow_state[:6] - fast_state[:6]) * STATE_WEIGHT
    return float(np.linalg.norm(gap) + 0.1 * np.abs(slow_state[6:8] - fast_state[6:8]).sum())


def build_progress_pairs(safe_episodes, fast_episodes):
    """Same seed, both land, align by t/T. Crashed fast episodes are not targets."""
    safe_by, safe_dropped = _index_successes(safe_episodes, "safe")
    fast_by, fast_dropped = _index_successes(fast_episodes, "fast")
    columns = {name: [] for name in PAIR_ARRAYS}
    pair_rows = []
    not_faster = 0
    safe_only = sorted(set(safe_by) - set(fast_by))
    fast_only = sorted(set(fast_by) - set(safe_by))
    for seed in sorted(set(safe_by) & set(fast_by)):
        safe, fast = safe_by[seed], fast_by[seed]
        if int(fast["steps"]) >= int(safe["steps"]):
            not_faster += 1
            continue
        slow_states, slow_actions, _, terrain = _trajectory(safe)
        fast_states, fast_actions, _, _ = _trajectory(fast)
        previous = np.empty_like(slow_actions)
        previous[0] = OFF_ACTION
        if len(slow_actions) > 1:
            previous[1:] = slow_actions[:-1]
        for row in range(len(slow_states)):
            fast_row = progress_index(len(slow_states), len(fast_states), row)
            columns["states"].append(slow_states[row])
            columns["previous_actions"].append(previous[row])
            columns["neutral_actions"].append(slow_actions[row])
            columns["target_actions"].append(fast_actions[fast_row])
            columns["terrain"].append(terrain)
            columns["slow_steps"].append(int(safe["steps"]))
            columns["fast_steps"].append(int(fast["steps"]))
            columns["slow_seed"].append(seed)
            columns["fast_seed"].append(seed)
            columns["slow_index"].append(row)
            columns["fast_index"].append(fast_row)
            columns["state_distance"].append(_aligned_distance(slow_states[row], fast_states[fast_row]))
        pair_rows.append(dict(slow_seed=seed, fast_seed=seed, slow_steps=int(safe["steps"]),
                              fast_steps=int(fast["steps"]), rows=len(slow_states)))
    if not columns["states"]:
        raise ValueError("no same-seed both-land progress pairs; the fast teacher did not "
                         "land sooner on any shared seed")
    arrays = _stack_columns(columns)
    if np.any(arrays["slow_seed"] != arrays["fast_seed"]):
        raise RuntimeError("progress pairs must share a seed")
    crashes = safe_dropped["crash"] + fast_dropped["crash"]
    timeouts = safe_dropped["time_limit"] + fast_dropped["time_limit"]
    oob = safe_dropped["out_of_bounds"] + fast_dropped["out_of_bounds"]
    other = safe_dropped["other"] + fast_dropped["other"]
    edit = float(np.abs(arrays["target_actions"] - arrays["neutral_actions"]).mean())
    manifest = dict(
        format="slow_fast_pairs_v1",
        pairing="progress_same_seed",
        alignment="t/T",
        successes=len(pair_rows),
        fast_pool=len(pair_rows), slow_pool=len(pair_rows), pairs=len(pair_rows),
        rows=int(len(arrays["states"])),
        crashes_excluded=int(crashes), timeouts_excluded=int(timeouts),
        oob_excluded=int(oob), other_excluded=int(other),
        not_faster=int(not_faster), safe_only=len(safe_only), fast_only=len(fast_only),
        fast_steps_max=int(arrays["fast_steps"].max()), slow_steps_min=int(arrays["slow_steps"].min()),
        mean_action_edit=edit, dropped_state_rows=0, unmatched_slow=len(safe_only),
        nearest=dict(count=0, note="nearest-state matching is disabled"),
        pairs_detail=pair_rows,
        outcomes_in_pools=[SUCCESS],
        neutral="safe teacher action on the safe trajectory",
        target="fast teacher action at the same progress t/T on this seed",
        speed_mechanism="paired successful actions; not the safe-fast kinematic cost",
    )
    return dict(arrays=arrays, manifest=manifest)


def build_slow_fast_pairs(*_args, **_kwargs):
    """Disabled. Cross-episode nearest matching has no shared landing."""
    raise RuntimeError(
        "stranger nearest pairing is disabled. "
        "Use build_progress_pairs: same seed, both land, progress t/T.")


def archive_stranger_pairs(directory):
    """Rename a legacy pairs.npz so the student trainer cannot load it by habit."""
    directory = Path(directory)
    legacy = directory / "pairs.npz"
    if not legacy.is_file():
        return None
    dest = directory / STRANGER_ARCHIVE_NAME
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite {dest} while archiving {legacy}")
    legacy.rename(dest)
    sidecar = legacy.with_suffix(".json")
    archived_sidecar = dest.with_suffix(".json")
    if sidecar.is_file():
        if archived_sidecar.exists():
            raise FileExistsError(f"Refusing to overwrite {archived_sidecar}")
        sidecar.rename(archived_sidecar)
    return dest


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
    if np.any(arrays["slow_seed"] != arrays["fast_seed"]):
        raise ValueError(
            "stranger pairs: slow_seed and fast_seed differ. "
            "Nearest-episode matching has no shared landing.")


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


def _synthetic_episode(seed, steps, outcome, action, success):
    """One fake rollout. Not Lunar physics."""
    states = np.zeros((steps, 8), dtype=np.float32)
    states[:, 0] = 0.01 * seed
    states[:, 1] = np.linspace(1.2, 0.05, steps)
    actions = np.tile(np.asarray(action, dtype=np.float32), (steps, 1))
    return dict(seed=seed, steps=steps, outcome=outcome, states=states, actions=actions,
                initial_state=states[0].tolist(), terrain=[0.] * 11,
                game_over=not success, lander_awake=not success, terminated=outcome != "time_limit",
                truncated=outcome == "time_limit", teacher="synthetic")


def synthetic_teacher_rollouts():
    """Same seeds for both teachers. Seed 4 crashes on the fast teacher and stays out."""
    safe = [
        _synthetic_episode(1, 40, SUCCESS, [-0.2, 0.0], success=True),
        _synthetic_episode(2, 48, SUCCESS, [-0.3, 0.1], success=True),
        _synthetic_episode(3, 36, SUCCESS, [-0.1, -0.1], success=True),
        _synthetic_episode(4, 40, SUCCESS, [-0.2, 0.0], success=True),
    ]
    fast = [
        _synthetic_episode(1, 12, SUCCESS, [0.4, 0.2], success=True),
        _synthetic_episode(2, 16, SUCCESS, [0.5, 0.0], success=True),
        _synthetic_episode(3, 14, SUCCESS, [0.3, -0.2], success=True),
        _synthetic_episode(4, 8, "crash", [1.0, 1.0], success=False),
    ]
    return safe, fast


def synthetic_episodes():
    """Flat list kept for outcome checks. Pairing uses synthetic_teacher_rollouts."""
    safe, fast = synthetic_teacher_rollouts()
    return safe + fast


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
