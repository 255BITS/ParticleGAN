#!/usr/bin/env python
"""Train the Lunar fast teacher from a frozen #18 spine.

The student does not see this loss. Every stage descends

    speed_bias * (main_engine * altitude) + anchor_weight * MSE(action, frozen safe action)

on states from the safe teacher's own successful probe landings. Positive main
engine fires up, so the speed term cuts hover. The anchor is only a stabilizer
so the first step does not erase the lander. It is not a fractional shrink of
the action, and it is not the safe-fast kinematic cost (`safe_fast_weight` stays
0). The speed term keeps growing through every stage. A crashy probe is
expected. `held.pt` is the fastest stage that still has successful landings,
not the last stage that matched the safe landing rate. Collect drops crashes.
Crashed episodes are never a training set.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.config import read_config
from experiments.train_gym_transition import sha256, write_json
from lib.gym_control import predict_control
from lib.gym_particle_finetune import MODULE_KEYS, configure_control_scope, load_paired_controller
from lib.slow_fast_lunar import (COLLECTION_SEED_START, EVAL_SEEDS, PROBE_SEED_START,
    is_success, speed_stats)

DEFAULTS = dict(
    checkpoint="results/gym/lunar_lander_particle_finetune/particle/best.pt",
    out_dir="results/gym/lunar_lander_slow_fast/fast_teacher",
    live_log="results/gym/lunar_lander_slow_fast/fast_teacher.log",
    device="cuda:1",
    speed_bias=0.05,
    anchor_weight=1.0,
    speed_growth=2.0,
    steps_per_stage=40,
    max_stages=8,
    min_held_landings=1,
    lr=1e-4,
    batch_size=256,
    seed=24011,
    probe_seed_start=PROBE_SEED_START,
    probe_episodes=20,
)
FORMAT = "gym_fast_teacher_v1"
GRAD_CLIP = 1.0
# A stage that still matches the safe landing count must be at least this much
# faster. A 20/20 probe that is one step quicker is not a hold. A crashy stage
# is not subject to this bar; fewer success steps win.
FULL_LANDING_MIN_DROP = 0.10
LOCKED = dict(
    arm="fast_teacher",
    imitation_weight=0.0,
    real_encoding_weight=0.0,
    synthetic_reconstruction_weight=0.0,
    adv_weight=1.0,
    safe_fast_weight=0.0,
    speed_term="main_engine * altitude",
    train_scope="control",
)


def _positive_number(value, key):
    if isinstance(value, bool) or type(value) not in (int, float) or not value > 0:
        raise ValueError(f"{key} must be a positive number")


def validate(cfg):
    if set(cfg) != set(DEFAULTS):
        raise ValueError(f"Unexpected config keys: {set(cfg) ^ set(DEFAULTS)}")
    for key in ("checkpoint", "out_dir", "live_log", "device"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if str(cfg["device"]).startswith("cuda") and cfg["device"] != "cuda:1":
        raise ValueError("Experiments must use cuda:1; GPU 0 belongs to the user")
    for key in ("speed_bias", "anchor_weight", "lr"):
        _positive_number(cfg[key], key)
    _positive_number(cfg["speed_growth"], "speed_growth")
    if cfg["speed_growth"] <= 1:
        raise ValueError("speed_growth must be greater than 1 so each stage is faster")
    for key in ("steps_per_stage", "max_stages", "batch_size", "probe_episodes", "min_held_landings"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["min_held_landings"] > cfg["probe_episodes"]:
        raise ValueError("min_held_landings cannot exceed probe_episodes")
    if type(cfg["seed"]) is not int or type(cfg["probe_seed_start"]) is not int:
        raise ValueError("seed and probe_seed_start must be integers")
    if cfg["batch_size"] < 2:
        raise ValueError("batch_size must be at least 2")
    probe_seeds(cfg["probe_seed_start"], cfg["probe_episodes"])


def probe_seeds(start, count):
    seeds = list(range(int(start), int(start) + int(count)))
    overlap = EVAL_SEEDS.intersection(seeds)
    if overlap:
        shown = ", ".join(str(seed) for seed in sorted(overlap)[:5])
        raise ValueError(f"probe seeds overlap the shared eval protocol: {shown}")
    if any(seed >= COLLECTION_SEED_START for seed in seeds):
        raise ValueError("probe seeds overlap collection seeds")
    return seeds


def select_held_stage(stages, min_landings, safe=None, full_landing_min_drop=FULL_LANDING_MIN_DROP):
    """Fastest stage that still has landings. Do not stop because the probe crashed.

    Eligible stages have ``landing_count >= min_landings`` and a finite mean
    success-step count. The hold minimizes that mean. Ties go to the later
    stage. A stage that still matches the safe landing count must also be
    ``full_landing_min_drop`` faster than the safe probe, so a 20/20 result
    one step quicker is not kept. A crashy stage has no such bar.
    """
    if type(min_landings) is not int or min_landings < 1:
        raise ValueError("min_landings must be a positive integer")
    eligible = []
    for row in stages:
        steps = row.get("mean_success_steps")
        if steps is None:
            continue
        if int(row["landing_count"]) < min_landings:
            continue
        eligible.append(row)
    if not eligible:
        return None
    chosen = min(eligible, key=lambda row: (float(row["mean_success_steps"]), -int(row["stage"])))
    if safe is None or safe.get("mean_success_steps") is None:
        return chosen
    if int(chosen["landing_count"]) < int(safe["landing_count"]):
        return chosen
    safe_steps = float(safe["mean_success_steps"])
    if safe_steps <= 0:
        return chosen
    drop = (safe_steps - float(chosen["mean_success_steps"])) / safe_steps
    if drop < full_landing_min_drop:
        return None
    return chosen


def _speed_loss(action, altitude, safe_action, speed_bias, anchor_weight):
    speed = (action[:, 0] * altitude.clamp(min=0)).mean()
    anchor = torch.nn.functional.mse_loss(action, safe_action)
    return speed_bias * speed + anchor_weight * anchor, float(speed.detach()), float(anchor.detach())


def _saved_config(cfg, speed_bias):
    return {**cfg, **LOCKED, "speed_bias": float(speed_bias)}


def _save_checkpoint(path, bundle, cfg, speed_bias, step, provenance, validation):
    saved = dict(
        format=FORMAT,
        config=_saved_config(cfg, speed_bias),
        world_config=bundle["world_config"],
        scaler=bundle["scaler"].state_dict(),
        step=int(step),
        provenance=provenance,
        validation=validation,
    )
    saved.update({key: bundle[key].state_dict() for key in MODULE_KEYS})
    if "residual" in saved or "residual_spec" in saved:
        raise RuntimeError("fast teacher checkpoints do not carry a residual")
    torch.save(saved, path)
    return path


def _open_spine(cfg, device):
    bundle = load_paired_controller(cfg["checkpoint"], device)
    if bundle.get("residual") is not None:
        raise ValueError("fast teacher starts from a frozen #18 checkpoint, not a residual student")
    if bundle.get("format") == FORMAT:
        raise ValueError("fast teacher starts from the frozen #18 safe teacher, not another fast teacher")
    configure_control_scope(bundle)
    bundle["E_control"].train()
    bundle["G"].branches[1].train()
    return bundle


def _log_to(live, logfile, message):
    text = f"[fast-teacher] {message}"
    print(text, flush=True)
    logfile.write(text + "\n")
    logfile.flush()
    live.write(text + "\n")
    live.flush()


def _prepare_out(cfg):
    out = Path(cfg["out_dir"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a fresh empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    live = Path(cfg["live_log"])
    live.parent.mkdir(parents=True, exist_ok=True)
    return out, live


def _random_batch(cfg, device):
    generator = torch.Generator(device=device).manual_seed(cfg["seed"])
    states = torch.zeros(cfg["batch_size"], 8, device=device)
    states[:, 1] = 1.
    previous = torch.tensor([-1., 0.], device=device).expand(cfg["batch_size"], -1).contiguous()
    terrain = torch.zeros(cfg["batch_size"], 11, device=device)
    noise = torch.randn(states.shape, generator=generator, device=device, dtype=torch.float32)
    states = states + 0.01 * noise
    return states, previous, terrain


def _train_smoke(cfg, smoke_steps):
    if type(smoke_steps) is not int or smoke_steps < 1:
        raise ValueError("smoke_steps must be a positive integer")
    device = torch.device(cfg["device"])
    torch.set_num_threads(1)
    out, live_path = _prepare_out(cfg)
    bundle = _open_spine(cfg, device)
    states, previous, terrain = _random_batch(cfg, device)
    with torch.no_grad():
        safe_action, _ = predict_control(bundle, states, previous, terrain)
        safe_action = safe_action.detach()
    before = [p.detach().clone() for p in list(bundle["E_control"].parameters())
              + list(bundle["G"].branches[1].parameters())]
    opt = torch.optim.Adam(
        [p for p in list(bundle["E_control"].parameters()) + list(bundle["G"].branches[1].parameters())
         if p.requires_grad],
        lr=cfg["lr"])
    provenance = dict(smoke=True, parent=cfg["checkpoint"], note="not a Lunar landing")
    with (out / "log.txt").open("w", buffering=1) as logfile, live_path.open("a", buffering=1) as live:
        def log(message):
            _log_to(live, logfile, message)

        log("SMOKE random states. Not a Lunar landing and not a closed-loop probe.")
        log(f"START parent={cfg['checkpoint']} speed_bias={cfg['speed_bias']} "
            f"anchor_weight={cfg['anchor_weight']} safe_fast_weight=0 adv_weight=1 device={device}")
        for step in range(1, smoke_steps + 1):
            action, _ = predict_control(bundle, states, previous, terrain)
            loss, speed, anchor = _speed_loss(
                action, states[:, 1], safe_action, cfg["speed_bias"], cfg["anchor_weight"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite fast-teacher loss at smoke step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in opt.param_groups[0]["params"]], GRAD_CLIP)
            opt.step()
            log(f"smoke step={step}/{smoke_steps} loss={float(loss.detach()):.5f} "
                f"speed={speed:.5f} anchor={anchor:.5f}")
        moved = [p.detach() for p in list(bundle["E_control"].parameters())
                 + list(bundle["G"].branches[1].parameters())]
        if not any(not torch.equal(left, right) for left, right in zip(before, moved)):
            raise RuntimeError("smoke speed step did not move E_control or G2")
        held = out / "held.pt"
        _save_checkpoint(held, bundle, cfg, cfg["speed_bias"], smoke_steps, provenance,
                         dict(status="smoke", lunar_landing=False))
        log(f"WROTE {held} format={FORMAT}. SMOKE is not a Lunar landing.")
    reloaded = load_paired_controller(held, device)
    if reloaded.get("residual") is not None or reloaded.get("format") != FORMAT:
        raise RuntimeError("smoke held.pt did not reload as a residual-free fast teacher")
    if float(reloaded["config"]["adv_weight"]) != 1. or float(reloaded["config"]["safe_fast_weight"]) != 0.:
        raise RuntimeError("smoke held.pt left the adv_weight=1 safe_fast_weight=0 contract")
    return dict(held=str(held), smoke=True, weights_moved=True, format=FORMAT)


def _roll(bundle, seeds, live, logfile, label):
    from lib.gym_control_evaluation import OFF_ACTION, terminal_reason
    from lib.gym_data import make_env, terrain_context
    from lib.gym_particle_finetune import playback_action

    env = make_env()
    episodes = []
    columns = {key: [] for key in ("states", "previous", "terrain")}
    try:
        for seed in seeds:
            state, _ = env.reset(seed=int(seed))
            terrain = terrain_context(env)
            previous = OFF_ACTION.copy()
            states, actions = [], []
            while True:
                action, _meta = playback_action(bundle, state, previous, terrain)
                action = action.astype("float32", copy=False)
                following, _reward, terminated, truncated, _ = env.step(action)
                states.append(state.copy())
                actions.append(action.copy())
                state, previous = following, action
                if terminated or truncated:
                    outcome = terminal_reason(env, state, terminated, truncated)
                    episode = dict(
                        seed=int(seed), steps=len(actions), outcome=outcome,
                        game_over=bool(env.unwrapped.game_over),
                        lander_awake=bool(env.unwrapped.lander.awake),
                        terminated=bool(terminated), truncated=bool(truncated))
                    episodes.append(episode)
                    if is_success(episode):
                        prev = OFF_ACTION.copy()
                        for row_state, row_action in zip(states, actions):
                            columns["states"].append(row_state)
                            columns["previous"].append(prev.copy())
                            columns["terrain"].append(terrain.copy())
                            prev = row_action
                    _log_to(live, logfile,
                            f"probe {label} seed={seed} outcome={outcome} steps={episode['steps']} "
                            f"success={outcome == 'successful_landing'}")
                    break
                if len(actions) >= 1000:
                    raise RuntimeError("Pinned TimeLimit failed to end episode")
    finally:
        env.close()
    return episodes, columns


def _as_batch(columns, device):
    if not columns["states"]:
        return None
    return (
        torch.as_tensor(np_stack(columns["states"]), device=device),
        torch.as_tensor(np_stack(columns["previous"]), device=device),
        torch.as_tensor(np_stack(columns["terrain"]), device=device),
    )


def np_stack(rows):
    import numpy as np
    return np.stack(rows).astype("float32")


def _snapshot_safe_action(bundle, states, previous, terrain, batch_size):
    pieces = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            action, _ = predict_control(
                bundle, states[start:start + batch_size], previous[start:start + batch_size],
                terrain[start:start + batch_size])
            pieces.append(action.detach())
    return torch.cat(pieces, 0)


def _train_curriculum(cfg):
    device = torch.device(cfg["device"])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out, live_path = _prepare_out(cfg)
    seeds = probe_seeds(cfg["probe_seed_start"], cfg["probe_episodes"])
    bundle = _open_spine(cfg, device)
    provenance = dict(
        parent=dict(path=cfg["checkpoint"], sha256=sha256(cfg["checkpoint"])),
        probe_seeds=[seeds[0], seeds[-1]],
        speed_term=LOCKED["speed_term"],
        safe_fast_weight=0.0,
        adv_weight=1.0,
        note="Held checkpoint is the fastest stage that still has successful landings. "
             "The probe may be crashy. Crashed episodes are not training rows.",
    )
    held_path = None
    held_stage = None
    with (out / "log.txt").open("w", buffering=1) as logfile, live_path.open("a", buffering=1) as live, \
         (out / "metrics.jsonl").open("w", buffering=1) as metrics:
        def log(message):
            _log_to(live, logfile, message)

        log(f"START parent={cfg['checkpoint']} device={device} probe={seeds[0]}-{seeds[-1]} "
            f"speed_bias={cfg['speed_bias']} growth={cfg['speed_growth']} "
            f"anchor_weight={cfg['anchor_weight']} safe_fast_weight=0 adv_weight=1")
        log("LOSS main_engine * altitude, plus an anchor to the frozen safe action. "
            "Not the safe-fast kinematic cost. Not the student paired-error step.")
        log(f"HOLD after all {cfg['max_stages']} stages. min_held_landings={cfg['min_held_landings']}. "
            "A crashy probe is expected. Do not stop at the first unsafe stage.")
        bundle["E_control"].eval()
        bundle["G"].branches[1].eval()
        safe_episodes, safe_columns = _roll(bundle, seeds, live, logfile, "safe")
        safe = speed_stats(safe_episodes)
        log(f"SAFE landings={safe['landing_count']}/{safe['episodes']} crashes={safe['crash_count']} "
            f"oob={safe['oob_count']} success_steps={safe['mean_success_steps']}")
        batch = _as_batch(safe_columns, device)
        if safe["landing_count"] < 1 or batch is None:
            log("STOP safe probe did not land. No held checkpoint. Do not collect.")
            return dict(held=None, reason="safe probe did not land")
        states, previous, terrain = batch
        if len(states) < cfg["batch_size"]:
            raise ValueError(
                f"safe probe produced {len(states)} rows, smaller than batch_size {cfg['batch_size']}")
        safe_action = _snapshot_safe_action(bundle, states, previous, terrain, cfg["batch_size"])
        opt = torch.optim.Adam(
            [p for p in list(bundle["E_control"].parameters()) + list(bundle["G"].branches[1].parameters())
             if p.requires_grad],
            lr=cfg["lr"])
        generator = torch.Generator(device=device).manual_seed(cfg["seed"])
        history = []
        for stage in range(1, cfg["max_stages"] + 1):
            bias = float(cfg["speed_bias"]) * float(cfg["speed_growth"]) ** (stage - 1)
            log(f"STAGE {stage}/{cfg['max_stages']} speed_bias={bias:.6g} "
                f"steps={cfg['steps_per_stage']} anchor_weight={cfg['anchor_weight']}")
            bundle["E_control"].train()
            bundle["G"].branches[1].train()
            last = None
            for step in range(1, cfg["steps_per_stage"] + 1):
                ids = torch.randint(len(states), (cfg["batch_size"],), device=device, generator=generator)
                action, _ = predict_control(bundle, states[ids], previous[ids], terrain[ids])
                loss, speed, anchor = _speed_loss(
                    action, states[ids, 1], safe_action[ids], bias, cfg["anchor_weight"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite fast-teacher loss at stage {stage} step {step}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in opt.param_groups[0]["params"]], GRAD_CLIP)
                opt.step()
                last = (float(loss.detach()), speed, anchor)
                if step == 1 or step == cfg["steps_per_stage"]:
                    log(f"stage={stage} step={step}/{cfg['steps_per_stage']} "
                        f"loss={last[0]:.5f} speed={last[1]:.5f} anchor={last[2]:.5f}")
            bundle["E_control"].eval()
            bundle["G"].branches[1].eval()
            stage_episodes, _ignored = _roll(bundle, seeds, live, logfile, f"stage={stage}")
            stage_stats = speed_stats(stage_episodes)
            steps_text = "none" if stage_stats["mean_success_steps"] is None else \
                f"{stage_stats['mean_success_steps']:.2f}"
            eligible = (stage_stats["mean_success_steps"] is not None
                        and stage_stats["landing_count"] >= cfg["min_held_landings"])
            log(f"PROBE stage={stage} landings={stage_stats['landing_count']}/{stage_stats['episodes']} "
                f"crashes={stage_stats['crash_count']} oob={stage_stats['oob_count']} "
                f"success_steps={steps_text} eligible={eligible}")
            record = dict(stage=stage, speed_bias=bias, eligible=eligible,
                          loss=last[0], speed=last[1], anchor=last[2],
                          safe=safe, probe=stage_stats)
            metrics.write(json_line(record))
            total_step = stage * cfg["steps_per_stage"]
            stage_path = out / f"stage_{stage}.pt"
            _save_checkpoint(
                stage_path, bundle, cfg, bias, total_step, provenance,
                dict(status="recorded", probe=stage_stats, safe=safe, held=False))
            history.append(dict(
                stage=stage, path=stage_path, speed_bias=bias,
                landing_count=stage_stats["landing_count"],
                crash_count=stage_stats["crash_count"],
                oob_count=stage_stats["oob_count"],
                mean_success_steps=stage_stats["mean_success_steps"],
                episodes=stage_stats["episodes"]))
        chosen = select_held_stage(history, cfg["min_held_landings"], safe=safe)
        if chosen is None:
            log("STOP no stage was a faster landing worth holding. "
                "A 20/20 probe a step quicker than safe is not kept. Do not collect.")
        else:
            held_stage = int(chosen["stage"])
            held_path = out / "held.pt"
            saved = torch.load(chosen["path"], map_location="cpu", weights_only=False)
            saved["validation"] = dict(
                status="held", held=True, stage=held_stage,
                probe=dict(landing_count=chosen["landing_count"],
                           crash_count=chosen["crash_count"],
                           oob_count=chosen["oob_count"],
                           mean_success_steps=chosen["mean_success_steps"],
                           episodes=chosen["episodes"]),
                safe=safe,
                note="Fastest stage with successful landings. Probe crashes are expected. "
                     "Collect keeps a pair only when both teachers land.")
            torch.save(saved, held_path)
            steps_text = f"{chosen['mean_success_steps']:.2f}"
            log(f"HELD stage={held_stage} landings={chosen['landing_count']}/{chosen['episodes']} "
                f"crashes={chosen['crash_count']} oob={chosen['oob_count']} "
                f"success_steps={steps_text} -> {held_path}")
            log("HELD may be crashy on the probe. Collect drops every seed this teacher misses. "
                "Do not train crash actions.")
    write_json(out / "provenance.json", provenance)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    return dict(held=None if held_path is None else str(held_path), held_stage=held_stage, smoke=False)


def json_line(record):
    import json
    return json.dumps(record) + "\n"


def train(cfg, smoke=False, smoke_steps=2):
    validate(cfg)
    if smoke:
        return _train_smoke(cfg, smoke_steps)
    started = time.perf_counter()
    result = _train_curriculum(cfg)
    result["elapsed_s"] = time.perf_counter() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--out-dir")
    parser.add_argument("--live-log")
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true",
                        help="A few optimizer steps on random states. Not a Lunar landing.")
    parser.add_argument("--smoke-steps", type=int, default=2)
    args = parser.parse_args()
    cfg = {**DEFAULTS, **(read_config(args.config) if args.config else {})}
    cli = dict(checkpoint=args.checkpoint, out_dir=args.out_dir, live_log=args.live_log, device=args.device)
    for key, value in cli.items():
        if value is not None:
            cfg[key] = value
    result = train(cfg, smoke=args.smoke, smoke_steps=args.smoke_steps)
    if result.get("held") is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
