"""Compare D-memory handoff and expert-free feedback using identical GRU32 readers."""
import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.autonomous_memory import frozen
from experiments.memory_path import circles, mlp
from experiments import memory_scout as base
from particlegan import get_recipe, learning_rate_scale


@dataclass
class Config(base.Config):
    name: str = "handoff_cold"
    writer: str = "gru"
    handoff_weight: float = 1.
    cold_weight: float = 1.
    warm_weight: float = 0.
    warm_prefixes: list = field(default_factory=lambda: [0, 8, 32])
    handoff_zero_fraction: float = .125
    eval_prefixes: list = field(default_factory=lambda: [8, 32])
    evaluate_checkpoint: str | None = None

    def __post_init__(self):
        super().__post_init__()
        assert self.writer == "gru" and self.memory_dim == 32
        assert not self.g_private and not self.g_film and self.read_memory
        assert self.critic == "flat" and not self.differences and self.geometry == "none"
        assert not self.frozen_writer and self.freeze_writer_at is None and self.writer_lr_mult == 1
        assert min(self.handoff_weight, self.cold_weight, self.warm_weight) >= 0
        assert self.handoff_weight+self.cold_weight+self.warm_weight > 0
        assert self.warm_prefixes and all(0 <= n < self.train_length for n in self.warm_prefixes)
        assert 0 < self.handoff_zero_fraction < 1
        assert self.eval_prefixes and all(n >= 3 for n in self.eval_prefixes)
        assert not (self.resume and self.evaluate_checkpoint)


def context(writer, prefix):
    memory = writer.initial(prefix)
    for point in prefix.unbind(1):
        memory = writer.write(memory, point)
    return memory


def selected_context(writer, real, positions):
    """Each row contains only real points strictly before its target."""
    memory = writer.initial(real)
    selected = memory
    for t, point in enumerate(real.unbind(1)):
        selected = torch.where((positions == t)[:, None, None], memory, selected)
        memory = writer.write(memory, point)
    return selected


class Critic(base.Critic):
    def __init__(self, cfg):
        super().__init__(cfg)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(46)
            self.handoff_head = mlp(cfg.memory_dim+2, 1, cfg.d_width)

    def score_candidate(self, candidate, memory):
        # Scoring is read-only. This method never calls write().
        return self.handoff_head(torch.cat((candidate, memory.flatten(1)), -1)).squeeze(-1)


class HandoffView(nn.Module):
    def __init__(self, critic, real, positions):
        super().__init__()
        self.critic, self.real, self.positions = critic, real, positions

    def forward(self, candidate):
        return self.critic.score_candidate(candidate,
                    selected_context(self.critic.writer, self.real, self.positions))


class PrefixView(nn.Module):
    def __init__(self, critic, prefix):
        super().__init__()
        self.critic, self.prefix = critic, prefix

    def forward(self, suffix):
        return self.critic(torch.cat((self.prefix, suffix), 1))


def continuation(generator, writer, z, prefix, steps, intervention=None, states=False):
    """X is disconnected after prefix. Only generated points are written next."""
    path, history = [], []
    with frozen(writer):
        memory = context(writer, prefix)
        for t in range(steps):
            read = memory
            if intervention == "zero":
                read = torch.zeros_like(read)
            elif intervention == "shuffle":
                read = read.roll(1, 0)
            time_args = {'time_index': prefix.shape[1]+t} if getattr(generator, 'clock_bands', 0) else {}
            point, _ = generator(z, read, **time_args)
            path.append(point)
            memory = writer.write(memory, point)
            if states:
                history.append(memory.flatten(1))
    return torch.stack(path, 1), torch.stack(history, 1) if states else memory


def build(cfg, device):
    torch.manual_seed(42)
    generator = base.Reader(cfg).to(device)
    torch.manual_seed(44)
    critic = Critic(cfg).to(device)
    recipe = get_recipe(total_steps=cfg.schedule_steps, batch_size=cfg.batch_size,
                        num_particles=512, **cfg.recipe)
    torch.manual_seed(45)
    prior = recipe.make_prior().to(device)
    return generator, critic, prior, recipe


def fidelity(generated, clean, cutoff):
    """Offline known-circle reference, never provided to the rollout."""
    centers, radii = [], []
    for path in clean[:, :32].astype(np.float64):
        origin = path.mean(0)
        xy = path-origin
        fit = np.linalg.lstsq(np.column_stack((2*xy, np.ones(len(xy)))),
                              (xy*xy).sum(1), rcond=None)[0]
        centers.append(origin+fit[:2])
        radii.append(np.sqrt(max(1e-12, fit[2]+np.square(fit[:2]).sum())))
    center, radius = np.asarray(centers), np.asarray(radii)
    offsets = generated-center[:, None]
    # Include the transition out of the final expert point in all speed tests.
    offsets = np.concatenate((clean[:, cutoff-1:cutoff]-center[:, None], offsets), 1)
    angle = np.arctan2(offsets[:, :-1, 0]*offsets[:, 1:, 1]-offsets[:, :-1, 1]*offsets[:, 1:, 0],
                       (offsets[:, :-1]*offsets[:, 1:]).sum(-1))
    ref = clean-center[:, None]
    omega = np.arctan2(ref[:, 0, 0]*ref[:, 1, 1]-ref[:, 0, 1]*ref[:, 1, 0],
                       (ref[:, 0]*ref[:, 1]).sum(-1))
    radial = np.linalg.norm(offsets[:, 1:], axis=-1)/radius[:, None]-1
    target = clean[:, cutoff:cutoff+generated.shape[1]]
    errors = np.linalg.norm(generated-target, axis=-1)/radius[:, None]
    rmse = np.sqrt(np.square(radial).mean(1))
    direction = (angle*np.sign(omega[:, None])>0).mean(1)
    speed_error = np.abs(angle.mean(1)-omega)
    startup = errors[:, 0]
    passed = (rmse < .1) & (direction > .95) & (speed_error < .03) & (startup < .2)
    return {"reference_orbit_fraction": float(passed.mean()),
            "relative_radial_rmse": float(rmse.mean()),
            "signed_speed_mae": float(speed_error.mean()),
            "reference_direction_consistency": float(direction.mean()),
            "startup_error_relative": float(startup.mean()),
            "position_error_first32": float(errors[:, :32].mean()),
            "position_error_last128": float(errors[:, -128:].mean()),
            "passing_cw": int((passed & (omega<0)).sum()),
            "passing_ccw": int((passed & (omega>0)).sum())}


def state_metrics(states):
    return {"norm": states.norm(dim=-1).mean().item(),
            "saturation": (states.abs()>.95).float().mean().item()}


@torch.no_grad()
def evaluate(generator, critic, prior, cfg, device):
    metrics, paths = base.evaluate(generator, critic, prior, cfg, device)
    z = prior(torch.arange(cfg.eval_batch, device=device))
    observed, clean = circles(cfg.eval_batch, max(cfg.eval_prefixes)+cfg.eval_steps,
        torch.Generator(device=device).manual_seed(20260917), device, noise=cfg.noise)
    paths["continuation_reference"] = clean.cpu().numpy()
    for n in cfg.eval_prefixes:
        prefix = observed[:, :n]
        generated, states = continuation(generator, critic.writer, z, prefix, cfg.eval_steps, states=True)
        array = generated.cpu().numpy()
        paths[f"prefix{n}"] = array
        paths[f"observed_prefix{n}"] = prefix.cpu().numpy()
        prefix_states = []
        memory = critic.writer.initial(z)
        for point in prefix.unbind(1):
            memory = critic.writer.write(memory, point)
            prefix_states.append(memory.flatten(1))
        row = {"generated_256": base.diagnostics(array[:, :256]),
               "generated_long": base.diagnostics(array),
               "fidelity_256": fidelity(array[:, :256], paths["continuation_reference"], n),
               "fidelity_long": fidelity(array, paths["continuation_reference"], n),
               "real_prefix_state": state_metrics(torch.stack(prefix_states, 1)),
               "early_generated_state": state_metrics(states[:, :64]),
               "late_generated_state": state_metrics(states[:, -256:])}
        for intervention in ("zero", "shuffle"):
            altered, _ = continuation(generator, critic.writer, z, prefix, 256, intervention)
            row[intervention] = fidelity(altered.cpu().numpy(), paths["continuation_reference"], n)
        metrics[f"prefix{n}"] = row
    return metrics, paths


def objectives(cfg, generator, critic, z, real, positions, prefix_length):
    """Return separate read-only conditional and explicit feedback objectives."""
    items = []
    if cfg.handoff_weight:
        memory = selected_context(critic.writer, real, positions)
        fake, _ = generator(z, memory)
        target = real[torch.arange(len(real), device=real.device), positions]
        items.append(("handoff", cfg.handoff_weight, HandoffView(critic, real, positions), target, fake))
    if cfg.cold_weight:
        fake, _ = continuation(generator, critic.writer, z, real[:, :0], cfg.train_length)
        items.append(("cold", cfg.cold_weight, critic, real, fake))
    if cfg.warm_weight:
        prefix = real[:, :prefix_length]
        fake, _ = continuation(generator, critic.writer, z, prefix, cfg.train_length-prefix_length)
        items.append(("warm", cfg.warm_weight, PrefixView(critic, prefix), real[:, prefix_length:], fake))
    return items


def train(cfg, out, device, log):
    generator, critic, prior, recipe = build(cfg, device)
    opt_g, opt_d = recipe.make_optimizers(generator, critic, prior)
    gan, penalty, spread = recipe.make_loss(), recipe.make_gradient_penalty(), recipe.make_prior_regularizer()
    rates = [[group["lr"] for group in opt.param_groups] for opt in (opt_g, opt_d)]
    rngs = {"data": torch.Generator(device=device).manual_seed(31415),
            "latent": torch.Generator(device=device).manual_seed(27182),
            "context": torch.Generator(device=device).manual_seed(16180)}
    start = 0
    if cfg.resume or cfg.evaluate_checkpoint:
        saved = torch.load(cfg.resume or cfg.evaluate_checkpoint, map_location=device, weights_only=False)
        if cfg.resume:
            old = asdict(Config(**saved["config"]))
            mutable = {"name", "steps", "resume", "eval_steps", "eval_batch", "log_every"}
            assert all(old[k] == v for k, v in asdict(cfg).items() if k not in mutable), "resume changes training"
        else:
            # Reuse completed autonomous control, adding only a dormant head.
            old = saved["config"]
            assert not old.get("g_private", False)
            for key in ("writer", "memory_dim", "g_width", "d_width", "train_length", "recipe"):
                assert getattr(cfg, key) == getattr(base.Config(**old), key), key
        generator.load_state_dict(saved["generator"])
        missing, unexpected = critic.load_state_dict(saved["critic"], strict=False)
        assert not unexpected and (not missing or (cfg.evaluate_checkpoint and all(k.startswith("handoff_head.") for k in missing)))
        prior.load_state_dict(saved["prior"])
        start = saved["step"]
        if cfg.resume:
            opt_g.load_state_dict(saved["opt_g"])
            opt_d.load_state_dict(saved["opt_d"])
            for name, rng in rngs.items():
                rng.set_state(saved["rngs"][name].cpu())
            torch.set_rng_state(saved["torch_rng"].cpu())
            if device.startswith("cuda"):
                torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), device)
            assert cfg.steps > start
    resolved = {**asdict(cfg), "resolved_recipe": recipe.to_dict(), "device": device,
                "gradient_clipping": None, "ema": False, "writer_training": "D only",
                "particle_policy": "fixed per trajectory; independent of sampled real episode",
                "memory_timing": "write real prefix strictly before target; candidate scoring read-only",
                "feedback": "generated writes with full BPTT; writer weights frozen in G updates",
                "loss_normalization": "weighted mean of active adversarial losses and penalties",
                "penalty_domain": "handoff: candidate; cold: full path; warm: generated suffix with real prefix fixed",
                "zero_prefix_training": bool(cfg.handoff_weight or cfg.cold_weight or 0 in cfg.warm_prefixes),
                "parameters": {"g": sum(p.numel() for p in generator.parameters()),
                               "d": sum(p.numel() for p in critic.parameters())}}
    (out/"config.json").write_text(json.dumps(resolved, indent=2))
    log(event="start", start_step=start, config=resolved)
    started = time.monotonic()
    end = start if cfg.evaluate_checkpoint else cfg.steps
    weight = cfg.handoff_weight+cfg.cold_weight+cfg.warm_weight
    for step in range(start+1, end+1):
        scale = learning_rate_scale(step-1, cfg.schedule_steps, recipe.lr_anneal_start, recipe.lr_floor)
        for opt, rates_ in zip((opt_g, opt_d), rates):
            for group, rate in zip(opt.param_groups, rates_):
                group["lr"] = rate*scale
        real, _ = circles(cfg.batch_size, cfg.train_length, rngs["data"], device, noise=cfg.noise)
        z, indices = prior.sample(cfg.batch_size, generator=rngs["latent"])
        positions = torch.randint(1, cfg.train_length, (cfg.batch_size,), device=device, generator=rngs["context"])
        positions[torch.rand(cfg.batch_size, device=device, generator=rngs["context"]) < cfg.handoff_zero_fraction] = 0
        choice = torch.randint(len(cfg.warm_prefixes), (), generator=rngs["context"], device=device).item()
        prefix_length = cfg.warm_prefixes[choice]
        with torch.no_grad():
            items = objectives(cfg, generator, critic, z, real, positions, prefix_length)
        opt_d.zero_grad(set_to_none=True)
        d_adv, d_reg = 0., 0.
        for _, w, view, target, fake in items:
            d_adv = d_adv+w*gan.d_loss(view(target), view(fake.detach()))/weight
            d_reg = d_reg+w*penalty(view, target, fake.detach(), step=step)/weight
        (d_adv+d_reg).backward()
        opt_d.step()
        opt_d.zero_grad(set_to_none=True)
        with frozen(critic):
            opt_g.zero_grad(set_to_none=True)
            items = objectives(cfg, generator, critic, z, real, positions, prefix_length)
            g_adv = 0.
            for _, w, view, target, fake in items:
                with torch.no_grad():
                    real_scores = view(target)
                g_adv = g_adv+w*gan.g_loss(view(fake), real_scores)/weight
            (g_adv+spread(prior(indices.unique()))).backward()
            opt_g.step()
        if step == start+1 or step % cfg.log_every == 0 or step == end:
            losses = {"d": d_adv.item(), "g": g_adv.item(), "penalty": d_reg.item()}
            if not all(np.isfinite(v) for v in losses.values()):
                raise RuntimeError(f"Nonfinite losses: {losses}")
            log(event="train", step=step, seconds=round(time.monotonic()-started, 2), **losses)
    if not cfg.evaluate_checkpoint:
        torch.save({"generator": generator.state_dict(), "critic": critic.state_dict(),
                    "writer": critic.writer.state_dict(), "prior": prior.state_dict(),
                    "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                    "rngs": {key: rng.get_state() for key, rng in rngs.items()},
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if device.startswith("cuda") else None,
                    "step": end, "config": asdict(cfg)}, out/"model.pt")
    generator.eval(), critic.eval(), prior.eval()
    metrics, paths = evaluate(generator, critic, prior, cfg, device)
    np.savez_compressed(out/"trajectories.npz", **paths)
    result = {"name": cfg.name, "steps": end, "seconds": time.monotonic()-started,
              "metrics": metrics, "config": resolved}
    (out/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    log(event="complete", steps=end, seconds=round(result["seconds"], 2), metrics=metrics)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = Config(**json.loads(args.config.read_text()))
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    provenance = {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "torch": torch.__version__, "argv": sys.argv, "sources": {}}
    for name in ("memory_core_scout.py", "memory_scout.py", "autonomous_memory.py", "memory_path.py"):
        source = Path(__file__).with_name(name).read_bytes()
        (args.out/name).write_bytes(source)
        provenance["sources"][name] = hashlib.sha256(source).hexdigest()
    if cfg.evaluate_checkpoint:
        provenance["checkpoint_sha256"] = hashlib.sha256(Path(cfg.evaluate_checkpoint).read_bytes()).hexdigest()
    (args.out/"provenance.json").write_text(json.dumps(provenance, indent=2))
    (args.out/"input.json").write_bytes(args.config.read_bytes())
    with (args.out/"experiment.log").open("w", buffering=1) as stream:
        def log(**row):
            line = json.dumps({"name": cfg.name, **row}, allow_nan=False)
            stream.write(line+"\n")
            print(line, flush=True)
        train(cfg, args.out, args.device, log)


if __name__ == "__main__":
    main()
