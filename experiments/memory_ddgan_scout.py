"""Four-step DDGAN with one D-owned trajectory memory and autonomous path loss.

The diffusion state is local to one emitted point. Both heads belong to D;
only D trains the memory writer. Real-prefix transition supervision is paired
with full-BPTT autonomous path supervision; no expert appears at deployment.
"""
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
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.autonomous_memory import frozen
from experiments.memory_path import circles, mlp
from experiments.memory_scout import Config as BaseConfig, Critic as PathCritic, diagnostics
from particlegan import DDGAN, get_recipe, learning_rate_scale


@dataclass
class Config(BaseConfig):
    name: str = "gru_ddgan4"
    writer: str = "gru"
    alpha_bar: list = field(default_factory=lambda: [1., .9, .5, .05, .0001])
    path_weight: float = 1.
    transition_weight: float = 1.

    def __post_init__(self):
        super().__post_init__()
        assert self.writer == "gru" and not self.g_private and self.read_memory
        assert not self.g_film and self.g_activation == "leaky_relu"
        assert self.g_memory_concat and self.g_film_source == "particle"
        assert self.critic == "flat" and not self.differences and self.geometry == "none"
        assert not self.frozen_writer and self.freeze_writer_at is None and self.writer_lr_mult == 1
        assert getattr(self, "memory_dim", 32) == 32
        assert self.path_weight > 0 and self.transition_weight > 0
        assert len(self.alpha_bar) == 5, "this scout uses four reverse transitions"
        DDGAN(self.alpha_bar)


class Denoiser(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = mlp(cfg.recipe.get("z_dim", 4)+32+2+4, 2, cfg.g_width)

    def forward(self, z, memory, noisy, t):
        time_code = F.one_hot(t-1, 4).to(noisy.dtype)
        return self.net(torch.cat((z, memory.flatten(1), noisy, time_code), -1))


def initial_memory(like):
    return like.new_zeros(len(like), 8, 4)


def prefix_memory(writer, path, positions):
    """Memory BEFORE each selected real point; later real points cannot leak."""
    memory = initial_memory(path)
    history = []
    for point in path.unbind(1):
        history.append(memory.flatten(1))
        memory = writer.write(memory, point)
    states = torch.stack(history, 1)
    return states[torch.arange(len(path), device=path.device), positions].reshape(-1, 8, 4)


class Critic(PathCritic):
    def __init__(self, cfg):
        super().__init__(cfg)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(46)
            self.transition_head = mlp(32+2+2+4, 1, cfg.d_width)

    def transition(self, candidate, real, noisy, positions, t):
        # The conditional penalty differentiates only the candidate coordinate.
        # Recomputing memory preserves D gradients to the writer through context.
        memory = prefix_memory(self.writer, real, positions)
        features = torch.cat((memory.flatten(1), candidate, noisy,
                              F.one_hot(t-1, 4).to(candidate.dtype)), -1)
        return self.transition_head(features).squeeze(-1)


class TransitionView(nn.Module):
    def __init__(self, critic, real, noisy, positions, t):
        super().__init__()
        self.critic, self.real, self.noisy = critic, real, noisy
        self.positions, self.t = positions, t

    def forward(self, candidate):
        return self.critic.transition(candidate, self.real, self.noisy, self.positions, self.t)


def rollout(generator, writer, diffusion, z, steps, rng, intervention=None, states=False):
    memory = initial_memory(z)
    path, history = [], []
    times = [torch.full((len(z),), k, device=z.device, dtype=torch.long)
             for k in range(diffusion.steps, 0, -1)]
    with frozen(writer):
        for _ in range(steps):
            read = memory
            if intervention == "zero":
                read = torch.zeros_like(read)
            elif intervention == "shuffle":
                read = read.roll(1, 0)
            noisy = torch.randn((len(z), 2), device=z.device, dtype=z.dtype, generator=rng)
            for t in times:
                clean = generator(z, read, noisy, t)
                eta = torch.randn(noisy.shape, device=z.device, dtype=z.dtype, generator=rng)
                noisy = diffusion.reverse(clean, noisy, t, eta)
            path.append(noisy)
            memory = writer.write(memory, noisy)
            if states:
                history.append(memory.flatten(1))
    return torch.stack(path, 1), torch.stack(history, 1) if states else memory


def build(cfg, device):
    torch.manual_seed(42)
    generator = Denoiser(cfg).to(device)
    torch.manual_seed(44)
    critic = Critic(cfg).to(device)
    recipe = get_recipe("ddgan", total_steps=cfg.schedule_steps, batch_size=cfg.batch_size,
                        conditioning="scalar", num_classes=None, alpha_bar=cfg.alpha_bar,
                        num_particles=512, **cfg.recipe)
    torch.manual_seed(45)
    prior = recipe.make_prior().to(device)
    diffusion = DDGAN(cfg.alpha_bar, device=device, validate_args=False)
    return generator, critic, prior, recipe, diffusion


def transition_fake(generator, critic, diffusion, z, real, positions, noisy, t, eta):
    with frozen(critic.writer):
        memory = prefix_memory(critic.writer, real, positions)
        clean = generator(z, memory, noisy, t)
        return diffusion.reverse(clean, noisy, t, eta)


@torch.no_grad()
def evaluate(generator, critic, prior, diffusion, cfg, device):
    z = prior(torch.arange(cfg.eval_batch, device=device))
    # Identical noise stream for baseline/interventions, independent of training.
    def evaluate_rollout(steps, intervention=None, states=False):
        return rollout(generator, critic.writer, diffusion, z, steps,
                       torch.Generator(device=device).manual_seed(99173), intervention, states)
    generated, states = evaluate_rollout(cfg.eval_steps, states=True)
    real, clean = circles(cfg.eval_batch, 256, torch.Generator(device=device).manual_seed(20260916), device)
    paths = {"generated": generated.cpu().numpy(), "real_noisy": real.cpu().numpy(), "real_clean": clean.cpu().numpy()}
    metrics = {"generated_256": diagnostics(paths["generated"][:, :256]),
               "generated_long": diagnostics(paths["generated"]),
               "real_noisy": diagnostics(paths["real_noisy"])}
    for intervention in ("zero", "shuffle"):
        altered, _ = evaluate_rollout(256, intervention)
        metrics[intervention] = diagnostics(altered.cpu().numpy())
    norm = states.norm(dim=-1)
    metrics["state"] = {"early_norm": norm[:, :cfg.train_length].mean().item(),
                        "late_norm": norm[:, -256:].mean().item(),
                        "early_saturation": (states[:, :cfg.train_length].abs()>.95).float().mean().item(),
                        "late_saturation": (states[:, -256:].abs()>.95).float().mean().item()}
    return metrics, paths


def train(cfg, out, device, log):
    generator, critic, prior, recipe, diffusion = build(cfg, device)
    opt_g, opt_d = recipe.make_optimizers(generator, critic, prior)
    gan, penalty, spread = recipe.make_loss(), recipe.make_gradient_penalty(), recipe.make_prior_regularizer()
    rates = [[group["lr"] for group in opt.param_groups] for opt in (opt_g, opt_d)]
    rngs = {"data": torch.Generator(device=device).manual_seed(31415),
            "latent": torch.Generator(device=device).manual_seed(27182),
            "diffusion": torch.Generator(device=device).manual_seed(16180)}
    start = 0
    if cfg.resume:
        saved = torch.load(cfg.resume, map_location=device, weights_only=False)
        mutable = {"name", "steps", "resume", "eval_steps", "eval_batch", "log_every"}
        old = asdict(Config(**saved["config"]))
        assert all(old[key] == value for key, value in asdict(cfg).items() if key not in mutable)
        for name, module in (("generator", generator), ("critic", critic), ("prior", prior)):
            module.load_state_dict(saved[name])
        opt_g.load_state_dict(saved["opt_g"])
        opt_d.load_state_dict(saved["opt_d"])
        for name, rng in rngs.items():
            rng.set_state(saved["rngs"][name].cpu())
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.startswith("cuda"):
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), device)
        start = saved["step"]
        assert cfg.steps > start
    resolved = {**asdict(cfg), "resolved_recipe": recipe.to_dict(), "device": device,
                "gradient_clipping": None, "ema": False,
                "g_state_gradient": "full BPTT; writer weights frozen",
                "writer_training": "D only", "particle_policy": "fixed across trajectory and diffusion steps",
                "initial_memory": "zeros", "diffusion_noise": "fresh Gaussian initial and posterior noise per emitted point",
                "transition_condition": "same real prefix, noisy point, timestep for each real/fake pair",
                "transition_sampling": "one uniform trajectory position and diffusion timestep per batch entry per update",
                "penalty_domain": "path: full trajectory; transition: candidate only, conditioning held fixed",
                "loss_normalization": "weighted mean of path and transition losses and penalties",
                "parameters": {"g": sum(p.numel() for p in generator.parameters()),
                               "d": sum(p.numel() for p in critic.parameters())}}
    (out/"config.json").write_text(json.dumps(resolved, indent=2))
    log(event="start", start_step=start, config=resolved)
    started = time.monotonic()
    wp, wt = cfg.path_weight, cfg.transition_weight
    for step in range(start+1, cfg.steps+1):
        scale = learning_rate_scale(step-1, cfg.schedule_steps, recipe.lr_anneal_start, recipe.lr_floor)
        for opt, base in zip((opt_g, opt_d), rates):
            for group, rate in zip(opt.param_groups, base):
                group["lr"] = rate*scale
        real, _ = circles(cfg.batch_size, cfg.train_length, rngs["data"], device, noise=cfg.noise)
        z, indices = prior.sample(cfg.batch_size, generator=rngs["latent"])
        positions = torch.randint(cfg.train_length, (cfg.batch_size,), device=device, generator=rngs["diffusion"])
        t = torch.randint(1, 5, (cfg.batch_size,), device=device, generator=rngs["diffusion"])
        clean = real[torch.arange(cfg.batch_size, device=device), positions]
        previous, noisy = diffusion.forward_pair(clean, t, generator=rngs["diffusion"])
        eta = torch.randn(noisy.shape, device=device, generator=rngs["diffusion"])
        transition = TransitionView(critic, real, noisy, positions, t)
        with torch.no_grad():
            fake, _ = rollout(generator, critic.writer, diffusion, z, cfg.train_length, rngs["diffusion"])
            fake_previous = transition_fake(generator, critic, diffusion, z, real, positions, noisy, t, eta)
        opt_d.zero_grad(set_to_none=True)
        d_path = gan.d_loss(critic(real), critic(fake))
        d_transition = gan.d_loss(transition(previous), transition(fake_previous))
        p_path = penalty(critic, real, fake, step=step)
        p_transition = penalty(transition, previous, fake_previous, step=step)
        d_adv = (wp*d_path+wt*d_transition)/(wp+wt)
        d_reg = (wp*p_path+wt*p_transition)/(wp+wt)
        (d_adv+d_reg).backward()
        opt_d.step()
        opt_d.zero_grad(set_to_none=True)
        with frozen(critic):
            opt_g.zero_grad(set_to_none=True)
            fake, _ = rollout(generator, critic.writer, diffusion, z, cfg.train_length, rngs["diffusion"])
            fake_previous = transition_fake(generator, critic, diffusion, z, real, positions, noisy, t, eta)
            with torch.no_grad():
                real_path_scores, real_transition_scores = critic(real), transition(previous)
            g_path = gan.g_loss(critic(fake), real_path_scores)
            g_transition = gan.g_loss(transition(fake_previous), real_transition_scores)
            g_adv = (wp*g_path+wt*g_transition)/(wp+wt)
            (g_adv+spread(prior(indices.unique()))).backward()
            opt_g.step()
        if step == start+1 or step % cfg.log_every == 0 or step == cfg.steps:
            losses = {"d": d_adv.item(), "g": g_adv.item(), "penalty": d_reg.item(),
                      "d_path": d_path.item(), "d_transition": d_transition.item(),
                      "g_path": g_path.item(), "g_transition": g_transition.item(),
                      "penalty_path": p_path.item(), "penalty_transition": p_transition.item()}
            if not all(np.isfinite(value) for value in losses.values()):
                raise RuntimeError(f"nonfinite losses: {losses}")
            log(event="train", step=step, seconds=round(time.monotonic()-started, 2), **losses)
    torch.save({"generator": generator.state_dict(), "critic": critic.state_dict(),
                "writer": critic.writer.state_dict(), "prior": prior.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                "rngs": {name: rng.get_state() for name, rng in rngs.items()},
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.startswith("cuda") else None,
                "step": cfg.steps, "config": asdict(cfg)}, out/"model.pt")
    generator.eval(), critic.eval(), prior.eval()
    metrics, paths = evaluate(generator, critic, prior, diffusion, cfg, device)
    np.savez_compressed(out/"trajectories.npz", **paths)
    result = {"name": cfg.name, "steps": cfg.steps, "seconds": time.monotonic()-started,
              "metrics": metrics, "config": resolved}
    (out/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    log(event="complete", steps=cfg.steps, seconds=round(result["seconds"], 2), metrics=metrics)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    cfg = Config(**json.loads(args.config.read_text()))
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    provenance = {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "torch": torch.__version__, "argv": sys.argv, "sources": {}}
    for path in [Path(__file__), Path(__file__).with_name("memory_scout.py"),
                 Path(__file__).with_name("autonomous_memory.py"), Path(__file__).with_name("memory_path.py"),
                 Path(__file__).resolve().parents[1]/"particlegan/diffusion.py"]:
        source = path.read_bytes()
        (args.out/path.name).write_bytes(source)
        provenance["sources"][str(path)] = hashlib.sha256(source).hexdigest()
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
