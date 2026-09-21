#!/usr/bin/env python
"""Circle transition encoder, then paired-error RpGAN controller finetune.

Stage 1 trains E_pair, G1/G2/G3, the MoG prior, and transition D.
Stage 2 trains E_control and G2 only, with paired-error RpGAN at adv_weight 1.
Diagnostic action MSE is logged and is not part of the controller loss.
Rollouts run after training, on frozen weights.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.config import read_config
from lib.circle_transition import (HORIZONS, PANEL_SEEDS, TANGENT_REFINE_FRACTION, WIDE_RHO_RANGE,
    CircleDiscriminator, CircleEncoder, CircleGenerator, CircleScaler, TangentResidual, assert_aligned,
    assert_feedforward, cap_radial_edit_scale, cap_tangent_edit_scale, closed_loop_fidelity, compose_pair,
    encode_pair, evaluation_panel, isolate_tangent_pair, learned_policy, local_errors,
    normalized_control_action, paired_edit_actions, protocol, radial_channel_diagnostics, radius_hold_gate,
    sample_rows, tangent_hold_tolerance, zero_policy, reversed_policy, expert_policy, evaluate_panel)
from lib.gym_particle_finetune import (EDIT_NOISE_HOLD, build_edit_critic, configure_control_scope,
    controller_objective, discriminator_objective, edit_cap, require_live_adversary)
from particlegan import get_recipe, learning_rate_scale


DEFAULTS = dict(
    seed=24002, device="cpu", width=128, encoder_width=128, d_width=128, z_dim=8,
    num_particles=64, sigma_rel=0.5, pretrain_steps=300, finetune_steps=2000, batch_size=128,
    log_interval=25, adv_weight=1.0, error_tokens=8, error_width=48, error_heads=4,
    normalization_samples=8192, eval_episodes=128, recovery_window=64, edit_frame="radial_tangent",
    rho_low=0.8, rho_high=1.2, on_circle_rate=0.5, rho_curriculum=False, tangent_refine=False,
    out_dir="results/circle_transition/radial_hold",
    live_log="results/circle_transition/live.log", save_checkpoint=True)


def validate(cfg):
    if set(cfg) != set(DEFAULTS):
        raise ValueError(f"Unknown or missing keys: {sorted(set(cfg) ^ set(DEFAULTS))}")
    for key in ("width", "encoder_width", "d_width", "z_dim", "num_particles", "pretrain_steps",
                "finetune_steps", "batch_size", "log_interval", "error_tokens", "error_width",
                "error_heads", "normalization_samples", "eval_episodes", "recovery_window"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["batch_size"] % 2 or cfg["eval_episodes"] % 4 or cfg["normalization_samples"] < 32:
        raise ValueError("batch_size must be even, eval_episodes a multiple of 4, normalization_samples >= 32")
    if cfg["num_particles"] < 2 or cfg["z_dim"] < 1:
        raise ValueError("particle table must contain at least two codes")
    if cfg["error_width"] % cfg["error_heads"]:
        raise ValueError("error_width must be divisible by error_heads")
    if not math.isfinite(cfg["sigma_rel"]) or cfg["sigma_rel"] <= 0:
        raise ValueError("sigma_rel must be finite and positive")
    if type(cfg["seed"]) is not int or type(cfg["save_checkpoint"]) is not bool:
        raise ValueError("invalid seed or save_checkpoint")
    for key in ("out_dir", "live_log", "device"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    require_live_adversary(cfg["adv_weight"])
    if cfg["edit_frame"] not in ("cartesian", "radial_tangent"):
        raise ValueError("edit_frame must be cartesian or radial_tangent")
    for key in ("rho_low", "rho_high"):
        if type(cfg[key]) is not float or not math.isfinite(cfg[key]):
            raise ValueError(f"{key} must be a finite float")
    if not 0.2 <= cfg["rho_low"] < 1 < cfg["rho_high"] <= 2.5:
        raise ValueError("training rho range must sit inside (0.2, 2.5) and contain 1")
    if type(cfg["on_circle_rate"]) is not float or not 0 <= cfg["on_circle_rate"] <= 1:
        raise ValueError("on_circle_rate must be a float in [0, 1]")
    if type(cfg["rho_curriculum"]) is not bool or type(cfg["tangent_refine"]) is not bool:
        raise ValueError("rho_curriculum and tangent_refine must be bools")
    if cfg["rho_curriculum"] and cfg["tangent_refine"]:
        raise ValueError("tangent refine replaces the rho curriculum; enable only one")
    if cfg["recovery_window"] >= min(HORIZONS):
        raise ValueError("recovery window must be shorter than every horizon")


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def link_live_log(out, live):
    live.parent.mkdir(parents=True, exist_ok=True)
    live.touch(exist_ok=True)
    alias = out / "live.log"
    if alias.is_symlink() or alias.exists():
        alias.unlink()
    alias.symlink_to(os.path.relpath(live, out))


def _snapshot(module):
    """Tensor parameters and buffers only. Prior extra-state dicts are not weights."""
    state = {f"p:{name}": parameter.detach().clone() for name, parameter in module.named_parameters()}
    state.update({f"b:{name}": buffer.detach().clone() for name, buffer in module.named_buffers()})
    return state


def _unchanged(module, saved, label):
    current = _snapshot(module)
    if current.keys() != saved.keys():
        raise RuntimeError(f"{label} tensor state changed shape while frozen")
    for name, value in current.items():
        if not torch.equal(value, saved[name]):
            raise RuntimeError(f"{label} changed while it was supposed to stay frozen ({name})")


def _grad_norm(parameters):
    total = 0.
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().square().sum())
    return math.sqrt(total)


def _prepare_dirs(cfg):
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {out}")
    live = Path(cfg["live_log"])
    link_live_log(out, live)
    return out, live


def _build(cfg, device):
    torch.manual_seed(cfg["seed"])
    generator = CircleGenerator(cfg["z_dim"], cfg["width"]).to(device)
    torch.manual_seed(cfg["seed"] + 100)
    discriminator = CircleDiscriminator(cfg["d_width"]).to(device)
    torch.manual_seed(cfg["seed"] + 102)
    encoder = CircleEncoder(8, cfg["z_dim"], cfg["encoder_width"]).to(device)
    torch.manual_seed(cfg["seed"] + 202)
    control = CircleEncoder(6, cfg["z_dim"], cfg["encoder_width"]).to(device)
    recipe = get_recipe("mog", z_dim=cfg["z_dim"], num_particles=cfg["num_particles"], sigma_rel=cfg["sigma_rel"],
                        total_steps=max(cfg["pretrain_steps"], 1), batch_size=cfg["batch_size"])
    prior = recipe.make_prior(device=device, generator=torch.Generator(device=device).manual_seed(cfg["seed"] + 101))
    assert_feedforward(generator, discriminator, encoder, control)
    if control.in_dim != 6 or encoder.in_dim != 8:
        raise RuntimeError("E_control must not receive the current action; E_pair may")
    return dict(G=generator, E=encoder, E_control=control, D=discriminator, prior=prior, recipe=recipe)


def _normalized(bundle, batch):
    scaler = bundle["scaler"]
    return scaler.triple(batch.position, batch.action, batch.next_position), scaler.context(batch.context())


def _transition_d_loss(discriminator, real, fake, context, gan, reg, step, rng):
    score_real = discriminator(real, context)
    score_fake = discriminator(fake.detach(), context)
    penalty, stats = reg.penalty(lambda sample, held=context: discriminator(sample, held), real, fake.detach(),
                                 step, rng, collect_stats=True)
    return gan.d_loss(score_real, score_fake) + penalty, stats


def _transition_g_loss(discriminator, real, fake, context, gan):
    with torch.no_grad():
        score_real = discriminator(real, context)
    return gan.g_loss(discriminator(fake, context), score_real)


def _apply_lr(optimizers, base_rates, step, total_steps, recipe):
    scale = learning_rate_scale(step - 1, total_steps, recipe.lr_anneal_start, recipe.lr_floor)
    for optimizer, rates in zip(optimizers, base_rates):
        for group, rate in zip(optimizer.param_groups, rates):
            group["lr"] = rate * scale
    return scale


def train(cfg):
    validate(cfg)
    device = torch.device(cfg["device"])
    torch.set_num_threads(1)
    out, live = _prepare_dirs(cfg)
    bundle = _build(cfg, device)
    bundle["edit_frame"] = cfg["edit_frame"]
    recipe = bundle["recipe"]
    data_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 11)
    prior_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 12)
    penalty_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 13)
    fit_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 14)
    calibration = sample_rows(cfg["normalization_samples"], fit_rng, "train", "mixed", device)
    bundle["scaler"] = CircleScaler.fit(calibration).to(device)
    control_init = _snapshot(bundle["E_control"])
    bundle["E_control"].requires_grad_(False)
    for module in (bundle["G"], bundle["E"], bundle["D"], bundle["prior"]):
        module.train()
    gan = recipe.make_loss()
    regularizer = recipe.make_gradient_penalty()
    spread = recipe.make_prior_regularizer()
    opt_g, opt_d = recipe.make_optimizers(bundle["G"], bundle["D"], bundle["prior"], encoder=bundle["E"],
                                          fused=device.type == "cuda")
    pre_rates = [[group["lr"] for group in opt.param_groups] for opt in (opt_g, opt_d)]
    write_json(out / "protocol.json", protocol())
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))

    started = time.perf_counter()
    with (out / "log.txt").open("w", buffering=1) as logfile, live.open("a", buffering=1) as livefile, \
            (out / "metrics.jsonl").open("w", buffering=1) as metrics:
        def log(message):
            line = f"[{out.name}] {message}"
            print(line, flush=True)
            logfile.write(line + "\n")
            livefile.write(line + "\n")

        log(f"START stage=pretrain device={device} steps={cfg['pretrain_steps']} seed={cfg['seed']} "
            f"particles={cfg['num_particles']} z_dim={cfg['z_dim']} sigma_rel={cfg['sigma_rel']}")
        log("PRETRAIN E_pair(st, at, context) -> z -> G1/G2/G3. Joint Rp logistic + sample-point b_cap. "
            "Real triple MSE and synthetic state/action MSE. E_control is not updated.")
        pretrain_draws = 0
        for step in range(1, cfg["pretrain_steps"] + 1):
            scale = _apply_lr((opt_g, opt_d), pre_rates, step, cfg["pretrain_steps"], recipe)
            bundle["D"].requires_grad_(True)
            batch = sample_rows(cfg["batch_size"], data_rng, "train", "mixed", device)
            real, context = _normalized(bundle, batch)
            with torch.no_grad():
                fake = bundle["G"](bundle["prior"].sample(len(batch), prior_rng)[0], context)
                composed = compose_pair(bundle["E"], bundle["G"], bundle["prior"], fake, context)[0]
                fake_d = torch.cat([fake[:len(batch) // 2], composed[len(batch) // 2:]], 0)
            d_loss, d_stats = _transition_d_loss(bundle["D"], real, fake_d, context, gan, regularizer, step, penalty_rng)
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
            bundle["D"].requires_grad_(False)
            batch = sample_rows(cfg["batch_size"], data_rng, "train", "mixed", device)
            real, context = _normalized(bundle, batch)
            latent, _ = bundle["prior"].sample(len(batch), prior_rng)
            fake = bundle["G"](latent, context)
            composed, decoded_fake, _ = compose_pair(bundle["E"], bundle["G"], bundle["prior"], fake, context)
            g_loss = (_transition_g_loss(bundle["D"], real, fake, context, gan)
                      + _transition_g_loss(bundle["D"], real, composed, context, gan)) / 2
            decoded_real, _ = encode_pair(bundle["E"], bundle["G"], bundle["prior"], real[:, :4], context)
            real_mse = (decoded_real - real).square().mean()
            synthetic_mse = (decoded_fake[:, :4] - fake[:, :4].detach()).square().mean()
            prior_loss = spread(bundle["prior"].z)
            opt_g.zero_grad(set_to_none=True)
            (g_loss + real_mse + synthetic_mse + prior_loss).backward()
            opt_g.step()
            pretrain_draws += 2 * cfg["batch_size"]
            if step == 1 or step % cfg["log_interval"] == 0 or step == cfg["pretrain_steps"]:
                with torch.no_grad():
                    action_mse = (decoded_real[:, 2:4] - real[:, 2:4]).square().mean()
                row = dict(stage="pretrain", step=step, d_loss=float(d_loss.detach()), g_loss=float(g_loss.detach()),
                           prior_loss=float(prior_loss.detach()), real_mse=float(real_mse.detach()),
                           synthetic_mse=float(synthetic_mse.detach()), pretrain_action_mse=float(action_mse),
                           b_cap_applied=bool(d_stats.get("applied", False)), lr_scale=scale,
                           elapsed_seconds=time.perf_counter() - started)
                metrics.write(json.dumps(row, allow_nan=False) + "\n")
                log(f"step={step}/{cfg['pretrain_steps']} stage=pretrain D={row['d_loss']:.4f} "
                    f"G={row['g_loss']:.4f} real_mse={row['real_mse']:.4f} "
                    f"pretrain_action_mse={row['pretrain_action_mse']:.4f} elapsed_s={row['elapsed_seconds']:.1f}")
        _unchanged(bundle["E_control"], control_init, "E_control")
        configure_control_scope(bundle)
        finetune_recipe = recipe.replace(total_steps=cfg["finetune_steps"])
        reg = edit_cap()
        with torch.no_grad():
            neutral_rows = sample_rows(cfg["normalization_samples"], fit_rng, "train", "mixed", device)
            neutrals, targets = [], []
            for start in range(0, len(neutral_rows), 1024):
                chunk = neutral_rows.index(slice(start, start + 1024))
                neutral, target = paired_edit_actions(
                    bundle, normalized_control_action(bundle, chunk.position, chunk.context()), chunk)
                neutrals.append(neutral)
                targets.append(target)
            neutrals, targets = torch.cat(neutrals), torch.cat(targets)
        critic = build_edit_critic(targets, neutrals, cfg).to(device)
        trainable = [parameter for parameter in list(bundle["E_control"].parameters())
                     + list(bundle["G"].branches[1].parameters()) if parameter.requires_grad]
        opt_c = torch.optim.Adam(trainable, lr=finetune_recipe.lr, betas=finetune_recipe.betas)
        opt_r = torch.optim.Adam(critic.parameters(), lr=finetune_recipe.lr * finetune_recipe.d_lr_mult,
                                 betas=finetune_recipe.betas)
        fine_rates = [[group["lr"] for group in opt.param_groups] for opt in (opt_c, opt_r)]
        frozen = {name: _snapshot(module) for name, module in
                  (("G1", bundle["G"].branches[0]), ("G3", bundle["G"].branches[2]), ("E_pair", bundle["E"]),
                   ("prior", bundle["prior"]), ("D", bundle["D"]))}
        edit_rng = {name: torch.Generator(device=device).manual_seed(cfg["seed"] + offset)
                    for name, offset in (("d", 41), ("g", 42))}
        b_cap_applications = 0
        gan_grad_abs = 0.
        finetune_draws = 0
        log(f"START stage=finetune steps={cfg['finetune_steps']} adv_weight=1 "
            f"normalization={critic.normalization} noise_start={float(critic.noise_start):.4f} "
            f"edit_rms={float(critic.edit_rms):.4f}")
        log("CONTROLLER STEP: real=noise, fake=noise+(pred-target)/scale. Rp logistic, adv_weight=1. "
            "sample-point b_cap every 4 updates, coeff times 4. TRAIN E_control+G2. "
            "FROZEN G1 G3 E_pair prior transition D. diag_action_mse is outside the loss. "
            f"edit_frame={cfg['edit_frame']} train_rho={cfg['rho_low']:.2f},{cfg['rho_high']:.2f} "
            f"on_circle_rate={cfg['on_circle_rate']:.2f} rho_curriculum={int(cfg['rho_curriculum'])} "
            f"tangent_refine={int(cfg['tangent_refine'])}.")
        if cfg["edit_frame"] == "radial_tangent":
            cap_info = cap_radial_edit_scale(critic, targets)
            log(f"RADIAL HOLD scale tangent={float(critic.target_std[0]):.5f} "
                f"radial={float(critic.target_std[1]):.5f} capped={cap_info['capped']} "
                f"signal_radial={cap_info['signal_scale']:.5f}")
        else:
            log(f"CARTESIAN edit scale={float(critic.target_std[0]):.5f},{float(critic.target_std[1]):.5f}")
        on_circle_rate = cfg["on_circle_rate"]
        curriculum_until = cfg["finetune_steps"] // 2 if cfg["rho_curriculum"] else 0
        refine_at = int(cfg["finetune_steps"] * TANGENT_REFINE_FRACTION) if cfg["tangent_refine"] else 0
        if cfg["tangent_refine"]:
            if cfg["edit_frame"] != "radial_tangent":
                raise RuntimeError("tangent refine requires the radial_tangent edit frame")
            bundle["tangent_head"] = TangentResidual(cfg["width"]).to(device)
            bundle["tangent_head_active"] = False
            log(f"TANGENT REFINE scheduled after step {refine_at}: freeze E_control+G2, "
                f"train zero-init tangent residual, wide_rho={WIDE_RHO_RANGE[0]:.2f},{WIDE_RHO_RANGE[1]:.2f}")
        head_parameters = []
        opt_h = None
        head_rates = None
        refine_base = None
        bundle["E_control"].train()
        bundle["G"].train()
        for step in range(1, cfg["finetune_steps"] + 1):
            refining = bool(refine_at) and step > refine_at
            if refining and opt_h is None:
                for parameter in trainable:
                    parameter.requires_grad_(False)
                bundle["tangent_head_active"] = True
                refine_base = (_snapshot(bundle["E_control"]), _snapshot(bundle["G"]))
                # Head is still zero, so this fit is the frozen policy's on-circle residual.
                fit_rows = sample_rows(cfg["normalization_samples"], torch.Generator(device=device).manual_seed(
                    cfg["seed"] + 15), "train", "on", device)
                with torch.no_grad():
                    fit_neutral, fit_target = [], []
                    for start in range(0, len(fit_rows), 1024):
                        chunk = fit_rows.index(slice(start, start + 1024))
                        neutral, target = paired_edit_actions(
                            bundle, normalized_control_action(bundle, chunk.position, chunk.context()), chunk)
                        fit_neutral.append(neutral)
                        fit_target.append(target)
                    fit_neutral, fit_target = torch.cat(fit_neutral), torch.cat(fit_target)
                critic = build_edit_critic(fit_target, fit_neutral, cfg).to(device)
                cap_radial_edit_scale(critic, fit_target)
                tangent_cap = cap_tangent_edit_scale(critic, EDIT_NOISE_HOLD, tangent_hold_tolerance())
                head_parameters = [parameter for parameter in bundle["tangent_head"].parameters()
                                   if parameter.requires_grad]
                opt_h = torch.optim.Adam(head_parameters, lr=finetune_recipe.lr, betas=finetune_recipe.betas)
                opt_r = torch.optim.Adam(critic.parameters(), lr=finetune_recipe.lr * finetune_recipe.d_lr_mult,
                                         betas=finetune_recipe.betas)
                head_rates = [[group["lr"] for group in opt.param_groups] for opt in (opt_h, opt_r)]
                log(f"TANGENT REFINE start at step {step}: freeze radius pathway, "
                    f"tangent_scale={tangent_cap['tangent_scale']:.5f} capped={tangent_cap['capped']} "
                    f"noise_limited={tangent_cap['noise_limited']:.5f} "
                    f"tolerance={tangent_cap['tolerance']:.5f}")
            if refining:
                phase_total = cfg["finetune_steps"] - refine_at
                scale = _apply_lr((opt_h, opt_r), head_rates, step - refine_at, phase_total, finetune_recipe)
                train_rho = (cfg["rho_low"], cfg["rho_high"])
                step_on_circle = 1.0
            else:
                scale = _apply_lr((opt_c, opt_r), fine_rates, step, cfg["finetune_steps"], finetune_recipe)
                if cfg["tangent_refine"]:
                    train_rho = WIDE_RHO_RANGE
                else:
                    train_rho = WIDE_RHO_RANGE if step <= curriculum_until else (cfg["rho_low"], cfg["rho_high"])
                step_on_circle = on_circle_rate
                if step == curriculum_until + 1 and curriculum_until:
                    log(f"RHO CURRICULUM switch at step {step}: train_rho={train_rho[0]:.2f},{train_rho[1]:.2f}")
            held = sample_rows(cfg["batch_size"], data_rng, "train", "mixed", device, rho_range=train_rho,
                               on_circle_rate=step_on_circle)
            with torch.no_grad():
                predicted_d = normalized_control_action(bundle, held.position, held.context())
                predicted_d, target_d = paired_edit_actions(bundle, predicted_d, held)
                if refining:
                    predicted_d, target_d = isolate_tangent_pair(predicted_d, target_d)
            d_loss, d_terms = discriminator_objective(
                critic, predicted_d, target_d, step, edit_rng["d"], reg, cfg["finetune_steps"])
            opt_r.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_r.step()
            if d_terms["b_cap_applied"]:
                b_cap_applications += 1
            batch = sample_rows(cfg["batch_size"], data_rng, "train", "mixed", device, rho_range=train_rho,
                                on_circle_rate=step_on_circle)
            assert_aligned(batch)
            normalized = normalized_control_action(bundle, batch.position, batch.context())
            predicted, target = paired_edit_actions(bundle, normalized, batch)
            scored_predicted, scored_target = isolate_tangent_pair(predicted, target) if refining else (predicted, target)
            g_loss, g_terms = controller_objective(
                critic, scored_predicted, scored_target, step, edit_rng["g"], cfg["finetune_steps"], cfg["adv_weight"])
            if not torch.allclose(g_loss.detach(), g_terms["error_g"] * cfg["adv_weight"], rtol=1e-4, atol=1e-5):
                raise RuntimeError("controller loss is not adv_weight times the paired-error RpGAN term")
            step_parameters = head_parameters if refining else trainable
            (opt_h if refining else opt_c).zero_grad(set_to_none=True)
            g_loss.backward()
            gan_grad_abs += _grad_norm(step_parameters)
            (opt_h if refining else opt_c).step()
            with torch.no_grad():
                diagnostic = torch.nn.functional.mse_loss(predicted.detach(), target.detach())
                physical = bundle["scaler"].inverse_action(normalized.detach())
                radial_l1, radial_corr, tangent_l1 = radial_channel_diagnostics(physical, batch)
            finetune_draws += 2 * cfg["batch_size"]
            if not torch.isfinite(g_loss) or not torch.isfinite(d_loss):
                raise FloatingPointError(f"nonfinite controller loss at step {step}")
            if step == 1 or step % cfg["log_interval"] == 0 or step == cfg["finetune_steps"]:
                row = dict(stage="finetune", step=step, loss=float(g_loss.detach()), d_loss=float(d_loss.detach()),
                           g_loss=float(g_loss.detach()), prior_loss=0., l2_aux_weight=0., adv_weight=1.,
                           b_cap_applied=d_terms["b_cap_applied"], diag_action_mse=float(diagnostic),
                           diag_radial_l1=float(radial_l1), diag_radial_rho_corr=float(radial_corr),
                           diag_tangent_l1=float(tangent_l1), tangent_refine_active=refining,
                           edit_frame=cfg["edit_frame"],
                           error_g=float(g_terms["error_g"]), error_d=float(d_terms["error_d"]),
                           b_cap=float(d_terms["b_cap"]), gan_grad_abs=gan_grad_abs, lr_scale=scale,
                           elapsed_seconds=time.perf_counter() - started)
                metrics.write(json.dumps(row, allow_nan=False) + "\n")
                log(f"step={step}/{cfg['finetune_steps']} stage=finetune loss={row['loss']:.5f} "
                    f"D={row['d_loss']:.5f} G={row['g_loss']:.5f} adv_weight=1 "
                    f"b_cap_applied={int(row['b_cap_applied'])} diag_action_mse={row['diag_action_mse']:.5f} "
                    f"diag_radial_l1={row['diag_radial_l1']:.5f} diag_radial_rho_corr={row['diag_radial_rho_corr']:.3f} "
                    f"diag_tangent_l1={row['diag_tangent_l1']:.5f} "
                    f"l2_aux=0 elapsed_s={row['elapsed_seconds']:.1f}")
        for name, saved in frozen.items():
            module = {"G1": bundle["G"].branches[0], "G3": bundle["G"].branches[2], "E_pair": bundle["E"],
                      "prior": bundle["prior"], "D": bundle["D"]}[name]
            _unchanged(module, saved, name)
        if refine_base is not None:
            _unchanged(bundle["E_control"], refine_base[0], "E_control during tangent refine")
            _unchanged(bundle["G"], refine_base[1], "G during tangent refine")
        if b_cap_applications < 1 or gan_grad_abs <= 0:
            raise RuntimeError("paired-error finetune did not apply b_cap or an adversarial gradient")
        modules = [bundle["G"], bundle["E"], bundle["E_control"], bundle["D"], bundle["prior"]]
        if bundle.get("tangent_head") is not None:
            modules.append(bundle["tangent_head"])
        for module in modules:
            module.eval()
            module.requires_grad_(False)
        if cfg["save_checkpoint"]:
            payload = dict(format="circle_transition_v1", config=cfg, protocol=protocol(),
                           G=bundle["G"].state_dict(), E=bundle["E"].state_dict(),
                           E_control=bundle["E_control"].state_dict(), D=bundle["D"].state_dict(),
                           prior=bundle["prior"].state_dict(), scaler=bundle["scaler"].state_dict(),
                           tangent_head_active=bool(bundle.get("tangent_head_active", False)))
            if bundle.get("tangent_head") is not None:
                payload["tangent_head"] = bundle["tangent_head"].state_dict()
            torch.save(payload, out / "final.pt")
            saved = torch.load(out / "final.pt", map_location=device, weights_only=False)
            bundle["G"].load_state_dict(saved["G"])
            bundle["E"].load_state_dict(saved["E"])
            bundle["E_control"].load_state_dict(saved["E_control"])
            bundle["prior"].load_state_dict(saved["prior"])
            bundle["scaler"].load_state_dict(saved["scaler"])
            if "tangent_head" in saved:
                bundle["tangent_head"].load_state_dict(saved["tangent_head"])
                bundle["tangent_head_active"] = bool(saved["tangent_head_active"])
        log("EVAL frozen final weights. No expert correction, no circle projection, G3 is not the environment.")
        evaluation = _evaluate(bundle, cfg, device)
        train_seconds = time.perf_counter() - started
        learned = evaluation["test"]["main"]["1024"]
        learned_val = evaluation["val"]["main"]["1024"]
        zero = evaluation["controls"]["test"]["zero"]["main"]["1024"]
        reversed_row = evaluation["controls"]["test"]["reversed"]["main"]["1024"]
        expert = evaluation["controls"]["test"]["expert"]["main"]["1024"]
        learned_fidelity = closed_loop_fidelity(learned)
        zero_fidelity = closed_loop_fidelity(zero)
        reversed_fidelity = closed_loop_fidelity(reversed_row)
        hold_val = radius_hold_gate(learned_val)
        hold_test = radius_hold_gate(learned)
        passed = bool(
            hold_val["passed"] and hold_test["passed"] and b_cap_applications > 0 and gan_grad_abs > 0
            and cfg["adv_weight"] == 1 and expert["success"] >= 0.99 and zero["success"] == 0
            and reversed_row["direction_agreement"] < 0.05)
        gate = dict(passed=passed, radius_hold_val=hold_val, radius_hold_test=hold_test,
                    legacy_fidelity=learned_fidelity, legacy_fidelity_would_pass=bool(
                        learned_fidelity > zero_fidelity + 0.05 and learned_fidelity > reversed_fidelity + 0.05),
                    zero_fidelity=zero_fidelity, reversed_fidelity=reversed_fidelity, adv_weight=1.,
                    b_cap_applications=b_cap_applications, gan_grad_abs=gan_grad_abs,
                    edit_frame=cfg["edit_frame"])
        counts = {name: sum(parameter.numel() for parameter in module.parameters())
                  for name, module in (("G", bundle["G"]), ("E_pair", bundle["E"]),
                                       ("E_control", bundle["E_control"]), ("D", bundle["D"]),
                                       ("prior", bundle["prior"]), ("R", critic))}
        summary = dict(
            config=cfg, protocol=protocol(), gate=gate, evaluation=evaluation,
            parameters=counts, trainable_controller=sum(parameter.numel() for parameter in trainable),
            pretrain_examples=pretrain_draws, finetune_examples=finetune_draws,
            simulator_calls_during_training=0, b_cap_applications=b_cap_applications,
            gan_grad_abs=gan_grad_abs, adv_weight=1., l2_aux_weight=0.,
            train_seconds=train_seconds, wall_seconds=time.perf_counter() - started)
        write_json(out / "summary.json", summary)
        write_json(out / "evaluation.json", evaluation)
        log(f"EVAL val main 1024 learned success={learned_val['success']:.3f} "
            f"worst_dir={learned_val['worst_direction_success']:.3f} radial={learned_val['radial_rmse']:.4f} "
            f"speed_err={learned_val['signed_speed_error']:.4f} hold={'PASS' if hold_val['passed'] else 'FAIL'}")
        log(f"EVAL test main 1024 learned success={learned['success']:.3f} "
            f"worst_dir={learned['worst_direction_success']:.3f} radial={learned['radial_rmse']:.4f} "
            f"speed_err={learned['signed_speed_error']:.4f} turns={learned['completed_turns']:.3f} "
            f"dir={learned['direction_agreement']:.3f} fidelity={learned_fidelity:.3f} "
            f"hold={'PASS' if hold_test['passed'] else 'FAIL'}")
        log(f"EVAL controls zero_fidelity={zero_fidelity:.3f} reversed_fidelity={reversed_fidelity:.3f} "
            f"expert_success={expert['success']:.3f}")
        log(f"LOCAL test action_l2={evaluation['test']['local']['action_l2']:.4f} "
            f"g3_l2={evaluation['test']['local']['g3_next_l2']:.4f} "
            f"persistence_l2={evaluation['test']['local']['persistence_next_l2']:.4f}")
        log(f"GATE {'PASS' if passed else 'FAIL'} train_s={train_seconds:.1f} "
            f"finetune_examples={finetune_draws} b_cap_applications={b_cap_applications}")
    return summary


def _evaluate(bundle, cfg, device):
    policies = dict(learned=learned_policy(bundle), zero=zero_policy, reversed=reversed_policy, expert=expert_policy)
    evaluation = {"controls": {}}
    with torch.no_grad():
        for split in ("val", "test"):
            evaluation[split] = {}
            for kind, window in (("main", 0), ("recovery", cfg["recovery_window"])):
                panel = evaluation_panel(split, kind, cfg["eval_episodes"], device)
                report = evaluate_panel(policies["learned"], panel, HORIZONS, window)
                evaluation[split][kind] = {str(horizon): row for horizon, row in report.items()}
            local_panel = sample_rows(cfg["eval_episodes"], torch.Generator(device=device).manual_seed(
                PANEL_SEEDS[(split, "main")] + 1000), split, "mixed", device)
            evaluation[split]["local"] = local_errors(bundle, local_panel)
            evaluation["controls"][split] = {}
            for name in ("zero", "reversed", "expert"):
                evaluation["controls"][split][name] = {}
                for kind, window in (("main", 0), ("recovery", cfg["recovery_window"])):
                    panel = evaluation_panel(split, kind, cfg["eval_episodes"], device)
                    report = evaluate_panel(policies[name], panel, HORIZONS, window)
                    evaluation["controls"][split][name][kind] = {str(horizon): row for horizon, row in report.items()}
    return evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/circle/paired_error.yaml"))
    parser.add_argument("--pretrain-steps", type=int)
    parser.add_argument("--finetune-steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--out-dir")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-episodes", type=int)
    args = parser.parse_args()
    cfg = {**DEFAULTS, **read_config(args.config)}
    for key, value in (("pretrain_steps", args.pretrain_steps), ("finetune_steps", args.finetune_steps),
                       ("device", args.device), ("out_dir", args.out_dir), ("batch_size", args.batch_size),
                       ("eval_episodes", args.eval_episodes)):
        if value is not None:
            cfg[key] = value
    summary = train(cfg)
    raise SystemExit(0 if summary["gate"]["passed"] else 1)


if __name__ == "__main__":
    main()
