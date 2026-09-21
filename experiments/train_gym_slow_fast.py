#!/usr/bin/env python
"""Finetune a frozen #18 Lunar controller with a bounded action residual.

E_control and G2 stay at the loaded lander. The only trained module is an
action residual of at most 0.15 per channel. The controller step is still
`controller_objective`: paired-error RpGAN, adv_weight 1, sample-point b_cap
every fourth update. Neutral is the slow action. Target is the fast action.
Diagnostic action MSE is logged under no_grad and is not in the loss. The
safe-fast kinematic cost is not this trainer.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.config import read_config
from experiments.train_gym_particle_finetune import link_live_log
from experiments.train_gym_transition import parameter_count, sha256, training_recipe, write_json
from lib.gym_particle_finetune import (EDIT_CAP_EVERY, MODULE_KEYS, RESIDUAL_IN_DIM,
    RESIDUAL_SCALE, ActionResidual, build_edit_critic, controller_objective,
    discriminator_objective, edit_cap, load_paired_controller, require_live_adversary,
    residual_physical_action)

DEFAULTS = dict(
    arm="slow_fast", steps=2500, batch_size=256, checkpoints=[250, 1000, 2500],
    log_interval=250, seed=24002, device="cuda:1",
    imitation_weight=0., real_encoding_weight=0., synthetic_reconstruction_weight=0.,
    adv_weight=1., safe_fast_weight=0., train_scope="residual",
    residual_scale=RESIDUAL_SCALE, residual_lr=0.01, residual_width=32,
    error_tokens=8, error_width=48, error_heads=4,
    checkpoint="results/gym/lunar_lander_particle_finetune/particle/best.pt",
    pairs="results/gym/lunar_lander_slow_fast/pairs.npz",
    out_dir="results/gym/lunar_lander_slow_fast/particle",
    live_log="results/gym/lunar_lander_slow_fast/live.log")
L2_KEYS = ("imitation_weight", "real_encoding_weight", "synthetic_reconstruction_weight")
FORMAT = "gym_slow_fast_finetune_v1"
CRITIC_LR = 1e-3


def validate(cfg):
    if set(cfg) != set(DEFAULTS):
        raise ValueError(f"Unexpected config keys: {set(cfg) ^ set(DEFAULTS)}")
    if cfg["arm"] != "slow_fast":
        raise ValueError("arm must be slow_fast")
    for key in ("steps", "batch_size", "log_interval"):
        if type(cfg[key]) is not int or cfg[key] < (2 if key == "batch_size" else 1):
            raise ValueError(f"Invalid {key}")
    if type(cfg["seed"]) is not int:
        raise ValueError("seed must be an integer")
    for key in L2_KEYS:
        if cfg[key] != 0:
            raise ValueError(f"{key} is removed; slow-fast does not train action MSE")
    require_live_adversary(cfg["adv_weight"])
    if cfg["safe_fast_weight"] != 0:
        raise ValueError("safe_fast_weight stays 0; slow-fast does not use the kinematic plant cost")
    if cfg["train_scope"] != "residual":
        raise ValueError("train_scope stays residual: freeze #18 and train the action residual")
    if float(cfg["residual_scale"]) != RESIDUAL_SCALE:
        raise ValueError("residual_scale stays 0.15; 0.20 dipped landings on the toy")
    if float(cfg["residual_lr"]) != 0.01:
        raise ValueError("residual_lr stays 0.01")
    if type(cfg["residual_width"]) is not int or cfg["residual_width"] < 1:
        raise ValueError("residual_width must be a positive integer")
    for key in ("error_tokens", "error_width", "error_heads"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["error_width"] % cfg["error_heads"] != 0:
        raise ValueError("error_width must divide across error_heads")
    if not isinstance(cfg["checkpoints"], list) or any(type(step) is not int or step <= 0
                                                       for step in cfg["checkpoints"]):
        raise ValueError("checkpoints must be positive integer steps")
    for key in ("device", "checkpoint", "pairs", "out_dir", "live_log"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if str(cfg["device"]).startswith("cuda") and cfg["device"] != "cuda:1":
        raise ValueError("Experiments must use cuda:1; GPU 0 belongs to the user")


def _columns(arrays, device):
    keys = ("states", "previous_actions", "neutral_actions", "target_actions", "terrain")
    return [torch.as_tensor(arrays[key], device=device) for key in keys]


def train(cfg):
    from lib.slow_fast_lunar import load_pairs

    validate(cfg)
    device = torch.device(cfg["device"])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    arrays, manifest = load_pairs(cfg["pairs"])
    if (arrays["slow_seed"] != arrays["fast_seed"]).any():
        raise ValueError(
            "stranger pairs: slow_seed and fast_seed differ. "
            "Nearest-episode matching has no shared landing. "
            "Rework collect to connected pairs before training.")
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Use a fresh empty output directory: {out}")
    live = Path(cfg["live_log"])
    link_live_log(out, live)
    if len(arrays["states"]) < cfg["batch_size"]:
        raise ValueError("batch_size is larger than the paired rows")
    bundle = load_paired_controller(cfg["checkpoint"], device)
    if bundle.get("residual") is not None:
        raise ValueError("start from a frozen #18 checkpoint, not a residual continuation")
    for key in MODULE_KEYS:
        bundle[key].eval().requires_grad_(False)
    residual = ActionResidual(scale=cfg["residual_scale"], width=cfg["residual_width"],
                              in_dim=RESIDUAL_IN_DIM).to(device)
    residual.train()
    bundle["residual"] = residual
    frozen_ec = [p.detach().clone() for p in bundle["E_control"].parameters()]
    frozen_g2 = [p.detach().clone() for p in bundle["G"].branches[1].parameters()]
    world = bundle["world_config"]
    columns = _columns(arrays, device)
    with torch.no_grad():
        neutrals = bundle["scaler"].action(columns[2])
        targets = bundle["scaler"].action(columns[3])
    if torch.allclose(neutrals, targets):
        raise ValueError("slow and fast actions are identical; there is no paired edit")
    critic = build_edit_critic(targets, neutrals, cfg).to(device)
    if critic.normalization != "paired_edit_per_coordinate_std_median_rms_gain":
        raise RuntimeError("Edit critic must whiten fast-minus-slow, not absolute actions")
    reg = edit_cap()
    recipe = training_recipe({**world, "steps": cfg["steps"], "batch_size": cfg["batch_size"]})
    adam_kwargs = dict(fused=True) if device.type == "cuda" else {}
    opt_g = torch.optim.Adam(residual.parameters(), lr=cfg["residual_lr"], **adam_kwargs)
    opt_r = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR, betas=(0., 0.999), **adam_kwargs)
    rng = {name: torch.Generator(device=device).manual_seed(cfg["seed"] + offset)
           for name, offset in dict(data=11, d_data=21, edit_d=41, edit_g=42).items()}
    objective = dict(
        loss_type="logistic", gan_mode="rp", reg_arm="b_cap", reg_method="autograd",
        reg_every=EDIT_CAP_EVERY, adv_weight=1., safe_fast_weight=0., train_scope="residual",
        residual_scale=RESIDUAL_SCALE, residual_lr=cfg["residual_lr"], critic_lr=CRITIC_LR,
        l2_aux_weight=0., diagnostic_mse="logged under no_grad; not added to the controller loss",
        neutral="recorded slow action", target="recorded fast action at the nearest matched state",
        speed_mechanism="bounded residual on a frozen lander; not the safe-fast kinematic cost",
        critic="gmix_t8_w48_l1", normalization=critic.normalization,
        initialization_recipe=recipe.to_dict())
    provenance = dict(
        initial_checkpoint=dict(path=cfg["checkpoint"], sha256=sha256(cfg["checkpoint"]),
                                step=int(bundle["step"])),
        pairs=dict(path=cfg["pairs"], sha256=sha256(cfg["pairs"]),
                   manifest=None if manifest is None else {key: manifest[key] for key in manifest
                                                           if key != "pairs_detail"}),
        removed_l2=list(L2_KEYS), adv_weight=1., safe_fast_weight=0.)
    write_json(out / "provenance.json", provenance)
    write_json(out / "recipe.json", objective)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    checkpoints = sorted({step for step in cfg["checkpoints"] if step <= cfg["steps"]} | {cfg["steps"]})

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def sample(name):
        ids = torch.randint(len(columns[0]), (cfg["batch_size"],), device=device, generator=rng[name])
        return [column[ids] for column in columns]

    def frozen_still_matches():
        for saved_p, live in zip(frozen_ec, bundle["E_control"].parameters()):
            if not torch.equal(saved_p, live):
                raise RuntimeError("E_control changed during residual training")
        for saved_p, live in zip(frozen_g2, bundle["G"].branches[1].parameters()):
            if not torch.equal(saved_p, live):
                raise RuntimeError("G2 changed during residual training")

    def save(step):
        frozen_still_matches()
        saved = dict(format=FORMAT, config=cfg, world_config=world, recipe=objective,
                     scaler=bundle["scaler"].state_dict(), step=step, provenance=provenance,
                     validation=dict(status="Awaiting shared-seed slow-fast rollout evaluation"),
                     residual=residual.state_dict(),
                     residual_spec=dict(scale=RESIDUAL_SCALE, width=cfg["residual_width"],
                                        in_dim=RESIDUAL_IN_DIM))
        saved.update({key: bundle[key].state_dict() for key in MODULE_KEYS})
        torch.save(saved, out / f"checkpoint_{step}.pt")

    with (out / "log.txt").open("w", buffering=1) as logfile, live.open("a", buffering=1) as livefile, \
         (out / "metrics.jsonl").open("w", buffering=1) as metrics:
        def log(message):
            text = f"[slow-fast] {message}"
            print(text, flush=True)
            logfile.write(text + "\n")
            livefile.write(text + "\n")

        log(f"START rows={len(columns[0])} pairs={0 if manifest is None else manifest.get('pairs')} "
            f"steps={cfg['steps']} device={device} adv_weight=1 safe_fast_weight=0")
        log(f"RESIDUAL frozen=#18 scale={RESIDUAL_SCALE} width={cfg['residual_width']} "
            f"lr={cfg['residual_lr']} critic_lr={CRITIC_LR} "
            "E_control and G2 are not updated.")
        log("CONTROLLER STEP paired-error RpGAN. neutral=slow action. target=fast action. "
            f"sample-point b_cap every {EDIT_CAP_EVERY} updates. diag_action_mse is outside the loss.")
        log("REFUSE adv_weight=0, action MSE in the loss, crash rows in the fast set, "
            "train_scope=control, and the safe-fast kinematic cost.")
        log("PLAYBACK clamp(tanh(G2) + residual, -1, 1). TRAIN the residual only.")
        started = time.perf_counter()
        sync()
        segment = time.perf_counter()
        optimization_seconds = 0.
        b_cap_applications = 0
        for step in range(1, cfg["steps"] + 1):
            d_rows = sample("d_data")
            with torch.no_grad():
                predicted_d = bundle["scaler"].action(
                    residual_physical_action(bundle, d_rows[0], d_rows[1], d_rows[4]))
            ld, d_terms = discriminator_objective(
                critic, predicted_d, bundle["scaler"].action(d_rows[3]), step, rng["edit_d"], reg, cfg["steps"])
            if not torch.isfinite(ld):
                raise FloatingPointError(f"Nonfinite discriminator loss at step {step}")
            opt_r.zero_grad(set_to_none=True)
            ld.backward()
            opt_r.step()
            if d_terms["b_cap_applied"]:
                b_cap_applications += 1
            g_rows = sample("data")
            predicted = bundle["scaler"].action(
                residual_physical_action(bundle, g_rows[0], g_rows[1], g_rows[4]))
            target = bundle["scaler"].action(g_rows[3])
            lg, g_terms = controller_objective(
                critic, predicted, target, step, rng["edit_g"], cfg["steps"], cfg["adv_weight"])
            if not torch.isfinite(lg):
                raise FloatingPointError(f"Nonfinite controller loss at step {step}")
            if g_terms["adv_weight"] != 1. or g_terms["safe_fast_weight"] != 0.:
                raise RuntimeError("controller step left the paired-error graph")
            opt_g.zero_grad(set_to_none=True)
            lg.backward()
            opt_g.step()
            with torch.no_grad():
                action_mse = torch.nn.functional.mse_loss(predicted.detach(), target.detach())
            if step == 1 or step % cfg["log_interval"] == 0 or step in checkpoints:
                sync()
                row = dict(step=step, loss=float(lg.detach()), d_loss=float(ld.detach()),
                           g_loss=float(lg.detach()), adv_weight=1., safe_fast_weight=0.,
                           b_cap_applied=d_terms["b_cap_applied"], lr=cfg["residual_lr"],
                           elapsed_seconds=time.perf_counter() - started,
                           action_mse=float(action_mse), l2_aux_weight=0.,
                           **{key: float(value) for key, value in {**d_terms, **g_terms}.items()
                              if key not in ("b_cap_applied", "adv_weight", "safe_fast_weight")})
                metrics.write(json.dumps(row, allow_nan=False) + "\n")
                log(f"step={step}/{cfg['steps']} loss={row['loss']:.5f} D={row['d_loss']:.5f} "
                    f"G={row['g_loss']:.5f} adv_weight=1 safe_fast_weight=0 "
                    f"b_cap_applied={int(row['b_cap_applied'])} diag_action_mse={row['action_mse']:.5f} "
                    f"l2_aux=0 elapsed_s={row['elapsed_seconds']:.1f}")
            if step in checkpoints:
                sync()
                optimization_seconds += time.perf_counter() - segment
                if any(not torch.isfinite(p).all() for p in bundle["E_control"].parameters()):
                    raise FloatingPointError(f"Nonfinite E_control at checkpoint {step}")
                if any(not torch.isfinite(p).all() for p in bundle["G"].branches[1].parameters()):
                    raise FloatingPointError(f"Nonfinite G2 at checkpoint {step}")
                save(step)
                log(f"CHECKPOINT step={step}; landing selection is not part of this train step")
                sync()
                segment = time.perf_counter()
        final = out / "final.pt"
        final.write_bytes((out / f"checkpoint_{cfg['steps']}.pt").read_bytes())
        summary = dict(
            config=cfg, recipe=objective, provenance=provenance, format=FORMAT,
            parameters={key: parameter_count(bundle[key]) for key in MODULE_KEYS},
            paired_rows=len(columns[0]), train_seconds=optimization_seconds,
            total_seconds=time.perf_counter() - started, simulator_calls=0,
            checkpoints={path.name: sha256(path) for path in sorted(out.glob("*.pt"))},
            l2_aux_weight=0., adv_weight=1., safe_fast_weight=0.,
            b_cap_applications=b_cap_applications,
            diagnostic_mse="outside the loss",
            selection="Deferred: shared-seed rollout has not been run",
            train_scope="residual", residual_scale=RESIDUAL_SCALE, residual_lr=cfg["residual_lr"],
            critic_lr=CRITIC_LR,
            residual_parameters=parameter_count(residual),
            initialization="Loaded the paired-error controller. Froze E_control and G2. "
                           "Trained only the bounded action residual. "
                           "Neutral is the slow action and target is the fast action.")
        write_json(out / "summary.json", summary)
        log(f"COMPLETE train_seconds={optimization_seconds:.1f} b_cap_applications={b_cap_applications} "
            "awaiting shared-seed eval")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--out-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--pairs")
    parser.add_argument("--error-tokens", type=int)
    parser.add_argument("--error-width", type=int)
    parser.add_argument("--error-heads", type=int)
    args = parser.parse_args()
    cfg = {**DEFAULTS, **(read_config(args.config) if args.config else {})}
    for key in ("steps", "batch_size", "device", "out_dir", "checkpoint", "pairs",
                "error_tokens", "error_width", "error_heads"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    train(cfg)


if __name__ == "__main__":
    main()
