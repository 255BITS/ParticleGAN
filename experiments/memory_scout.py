"""Config-driven autonomous particle/memory GAN scouts. No expert at runtime."""
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
from experiments.autonomous_memory import frozen, trajectory_metrics
from experiments.memory_path import FastMemory, circles, mlp
from particlegan import get_recipe, learning_rate_scale


@dataclass
class Config:
    name: str = "trace_flat"
    writer: str = "trace"
    memory_dim: int = 32
    frozen_writer: bool = False
    critic: str = "flat"
    g_width: int = 64
    d_width: int = 128
    g_activation: str = "leaky_relu"
    g_film: bool = False
    g_film_source: str = "particle"
    g_film_mapping: bool = False
    g_film_w_dim: int = 32
    g_film_fourier: int = 0
    g_memory_concat: bool = True
    writer_lr_mult: float = 1.0
    freeze_writer_at: int | None = None
    g_private: bool = False
    read_memory: bool = True
    differences: bool = False
    geometry: str = "none"
    train_length: int = 64
    steps: int = 2000
    schedule_steps: int = 10000
    batch_size: int = 128
    eval_batch: int = 128
    eval_steps: int = 1024
    noise: float = .03
    log_every: int = 100
    recipe: dict = field(default_factory=dict)
    resume: str | None = None

    def __post_init__(self):
        assert self.writer in ("trace", "gru", "delay")
        assert self.memory_dim > 0 and self.memory_dim % 4 == 0
        assert self.writer == "gru" or self.memory_dim == 32, "size scouts currently use GRU memory"
        assert self.g_film_source in ("particle", "memory")
        assert self.g_film_w_dim > 0 and 0 <= self.g_film_fourier <= 8
        assert not (self.g_film_mapping or self.g_film_fourier) or (self.g_film and self.g_film_source == "memory")
        assert self.g_memory_concat or (self.g_film and self.g_film_source == "memory")
        assert self.critic in ("flat", "multiscale")
        assert self.geometry in ("none", "distances", "oriented")
        assert self.g_activation in ("leaky_relu", "tanh", "silu")
        assert self.writer_lr_mult > 0
        assert self.freeze_writer_at is None or self.freeze_writer_at >= 0
        assert self.train_length >= 64 and self.steps <= self.schedule_steps
        assert self.eval_steps >= 256 and 4 <= self.eval_batch <= 512
        assert min(self.steps, self.batch_size, self.log_every, self.g_width, self.d_width) > 0
        forbidden = {"reg_arm", "reg_coeff", "reg_kappa", "reg_every", "reg_method"}
        if forbidden & self.recipe.keys():
            raise ValueError("Keep public API B-cap defaults for this study")
        supported = {"z_dim", "lr", "d_lr_mult", "prior_lr_mult", "betas", "loss_type", "gan_mode",
                     "prior_reg", "lr_anneal_start", "lr_floor"}
        if self.recipe.keys()-supported:
            raise ValueError(f"Unsupported recipe overrides: {self.recipe.keys()-supported}")


class Writer(nn.Module):
    def __init__(self, kind, memory_dim=32):
        super().__init__()
        self.kind, self.memory_dim = kind, memory_dim
        assert memory_dim > 0 and memory_dim % 4 == 0
        assert kind == "gru" or memory_dim == 32
        if kind == "gru":
            # Explicit GRU equations support exact second derivatives for B-cap.
            self.input = nn.Linear(2, 3*memory_dim)
            self.hidden = nn.Linear(memory_dim, 3*memory_dim)
        else:
            self.trace = FastMemory()

    def initial(self, batch_like):
        return batch_like.new_zeros(len(batch_like), self.memory_dim//4, 4)

    def write(self, memory, point):
        if self.kind == "gru":
            ir, iz, inn = self.input(point).chunk(3, -1)
            hr, hz, hn = self.hidden(memory.flatten(1)).chunk(3, -1)
            reset, update = (ir + hr).sigmoid(), (iz + hz).sigmoid()
            candidate = (inn + reset * hn).tanh()
            return ((1-update)*candidate + update*memory.flatten(1)).reshape(-1, self.memory_dim//4, 4)
        if self.kind == "delay":
            value = self.trace.values(point)
            return torch.cat((value[:, :, None], memory[:, :, :3]), -1)
        return self.trace.write(memory, point)


class Reader(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.private, self.read_memory = cfg.g_private, cfg.read_memory
        self.memory_concat, self.film_source = cfg.g_memory_concat, cfg.g_film_source
        inputs = cfg.recipe.get("z_dim", 4)+(cfg.memory_dim if self.memory_concat else 0)
        if self.private:
            self.cell = nn.GRUCell(inputs, cfg.g_width)
            self.net = mlp(cfg.g_width, 2, cfg.g_width)
        else:
            self.net = mlp(inputs, 2, cfg.g_width)
        if cfg.g_activation != "leaky_relu":
            activation = nn.Tanh if cfg.g_activation == "tanh" else nn.SiLU
            self.net[1], self.net[3] = activation(), activation()
        self.film = nn.ModuleList()
        self.film_mapping = nn.Identity()
        self.film_fourier = cfg.g_film_fourier
        if cfg.g_film:
            condition_dim = cfg.recipe.get("z_dim", 4) if self.film_source == "particle" else cfg.memory_dim
            if cfg.g_film_mapping:
                self.film_mapping = nn.Sequential(nn.Linear(condition_dim, cfg.g_film_w_dim),
                                                  nn.LeakyReLU(.2),
                                                  nn.Linear(cfg.g_film_w_dim, cfg.g_film_w_dim))
                condition_dim = cfg.g_film_w_dim
            condition_dim *= 1+2*self.film_fourier
            for _ in range(2):
                modulation = nn.Linear(condition_dim, 2*cfg.g_width)
                nn.init.zeros_(modulation.weight)
                nn.init.zeros_(modulation.bias)
                self.film.append(modulation)

    def forward(self, z, memory, hidden=None):
        if not self.read_memory:
            memory = torch.zeros_like(memory)
        inputs = torch.cat((z, memory.flatten(1)), -1) if self.memory_concat else z
        if self.private:
            hidden = self.cell(inputs, hidden)
            inputs = hidden
        if self.film:
            condition = z if self.film_source == "particle" else memory.flatten(1)
            condition = self.film_mapping(condition)
            if self.film_fourier:
                frequencies = 2.**torch.arange(self.film_fourier, device=condition.device, dtype=condition.dtype)
                angles = torch.pi*condition[:, :, None]*frequencies
                # Keep raw w as well as periodic features, avoiding forced aliasing.
                condition = torch.cat((condition, angles.sin().flatten(1), angles.cos().flatten(1)), -1)
            for i, modulation in enumerate(self.film):
                scale, shift = modulation(condition).chunk(2, -1)
                inputs = self.net[2*i+1](self.net[2*i](inputs)*(1+scale)+shift)
            return self.net[-1](inputs), hidden
        return self.net(inputs), hidden


def rollout(generator, writer, z, steps, intervention=None, states=False):
    memory, hidden = writer.initial(z), None
    path, history = [], []
    with frozen(writer):
        for _ in range(steps):
            read = memory
            if intervention == "zero":
                read = torch.zeros_like(memory)
            elif intervention == "shuffle":
                read = memory.roll(1, 0)
            point, hidden = generator(z, read, hidden)
            path.append(point)
            memory = writer.write(memory, point)
            if states:
                history.append(memory.flatten(1))
    return torch.stack(path, 1), torch.stack(history, 1) if states else memory


class Critic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Initialize shared components independently of head architecture.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(43)
            self.writer = Writer(cfg.writer, cfg.memory_dim)
        self.writer.requires_grad_(not cfg.frozen_writer)
        inputs = cfg.memory_dim+(6 if cfg.differences else 2)
        sizes = [cfg.train_length] if cfg.critic == "flat" else [8, 32, 64]
        self.heads = nn.ModuleDict({str(n): mlp(n*inputs, 1, cfg.d_width) for n in sizes})
        self.geometry_head = None
        if cfg.geometry != "none":
            channels = 2 if cfg.geometry == "oriented" else 1
            self.geometry_head = mlp(channels*cfg.train_length**2, 1, cfg.d_width)

    def forward(self, path):
        memory = self.writer.initial(path)
        features = []
        for point in path.unbind(1):
            features.append(memory.flatten(1))
            memory = self.writer.write(memory, point)
        parts = [path, torch.stack(features, 1)]
        if self.cfg.differences:
            delta = F.pad(path[:, 1:]-path[:, :-1], (0, 0, 1, 0))
            accel = F.pad(delta[:, 1:]-delta[:, :-1], (0, 0, 1, 0))
            parts += [delta*4, accel*8]
        features = torch.cat(parts, -1)
        if self.cfg.critic == "flat":
            score = self.heads[str(self.cfg.train_length)](features.flatten(1)).squeeze(-1)
        else:
            scores = []
            for length, head in self.heads.items():
                length = int(length)
                # Preserve full-prefix memory; overlapping windows share each head.
                windows = features.unfold(1, length, length//2).transpose(-1, -2)
                batch, count = windows.shape[:2]
                scores.append(head(windows.reshape(batch*count, -1)).reshape(batch, count).mean(1))
            scores.append(self.heads["8"](features[:, :8].flatten(1)).squeeze(-1))
            score = torch.stack(scores).mean(0)
        if self.geometry_head is not None:
            centered = path-path.mean(1, keepdim=True)
            distances = (centered[:, :, None]-centered[:, None, :]).square().sum(-1)/4
            geometry = [distances.flatten(1)]
            if self.cfg.geometry == "oriented":
                x, y = centered.unbind(-1)
                cross = (x[:, :, None]*y[:, None, :]-y[:, :, None]*x[:, None, :])/4
                geometry.append(cross.flatten(1))
            score = (score+self.geometry_head(torch.cat(geometry, -1)).squeeze(-1))/2
        # One normalized score, one public API adversarial loss and B-cap.
        return score


def build(cfg, device):
    torch.manual_seed(42)
    generator = Reader(cfg).to(device)
    torch.manual_seed(44)
    critic = Critic(cfg).to(device)
    recipe = get_recipe(total_steps=cfg.schedule_steps, batch_size=cfg.batch_size,
                        num_particles=512, **cfg.recipe)
    torch.manual_seed(45)
    prior = recipe.make_prior().to(device)
    return generator, critic, prior, recipe


def diagnostics(paths):
    full = trajectory_metrics(paths)
    late = trajectory_metrics(paths[:, -128:])
    individuals = [trajectory_metrics(p[None]) for p in paths]
    passes = [m for m in individuals if m["circle_like_fraction"]]
    full.update(late_circle_fraction=late["circle_like_fraction"],
                late_stopped_fraction=float((np.linalg.norm(np.diff(paths[:, -65:], axis=1), axis=-1).mean(1)<.01).mean()),
                passing_cw=sum(m["signed_angular_speed"]<0 for m in passes),
                passing_ccw=sum(m["signed_angular_speed"]>0 for m in passes))
    return full


@torch.no_grad()
def evaluate(generator, critic, prior, cfg, device):
    z = prior(torch.arange(cfg.eval_batch, device=device))
    generated, states = rollout(generator, critic.writer, z, cfg.eval_steps, states=True)
    # Standard reference identical to earlier 256-step studies.
    real, clean = circles(cfg.eval_batch, 256, torch.Generator(device=device).manual_seed(20260916), device)
    paths = {"generated": generated.cpu().numpy(), "real_noisy": real.cpu().numpy(), "real_clean": clean.cpu().numpy()}
    metrics = {"generated_256": diagnostics(paths["generated"][:, :256]),
               "generated_long": diagnostics(paths["generated"]),
               "real_noisy": diagnostics(paths["real_noisy"])}
    for intervention in ("zero", "shuffle"):
        altered, _ = rollout(generator, critic.writer, z, 256, intervention=intervention)
        metrics[intervention] = diagnostics(altered.cpu().numpy())
    norm = states.norm(dim=-1)
    metrics["state"] = {"early_norm": norm[:, :cfg.train_length].mean().item(),
                        "late_norm": norm[:, -256:].mean().item(),
                        "early_saturation": (states[:, :cfg.train_length].abs()>.95).float().mean().item(),
                        "late_saturation": (states[:, -256:].abs()>.95).float().mean().item()}
    return metrics, paths


def train(cfg, out, device, log):
    generator, critic, prior, recipe = build(cfg, device)
    opt_g, opt_d = recipe.make_optimizers(generator, critic, prior)
    if cfg.writer_lr_mult != 1.0 and not cfg.frozen_writer:
        writer_parameters = list(critic.writer.parameters())
        writer_ids = {id(p) for p in writer_parameters}
        group = opt_d.param_groups[0]
        group["params"] = [p for p in group["params"] if id(p) not in writer_ids]
        opt_d.add_param_group({"params": writer_parameters, "lr": group["lr"]*cfg.writer_lr_mult})
    gan, penalty, spread = recipe.make_loss(), recipe.make_gradient_penalty(), recipe.make_prior_regularizer()
    rates = [[g["lr"] for g in opt.param_groups] for opt in (opt_g, opt_d)]
    data_rng = torch.Generator(device=device).manual_seed(31415)
    latent_rng = torch.Generator(device=device).manual_seed(27182)
    start = 0
    if cfg.resume:
        saved = torch.load(cfg.resume, map_location=device, weights_only=False)
        old = asdict(Config(**saved["config"]))
        mutable = {"name", "steps", "resume", "eval_steps", "eval_batch", "log_every"}
        if cfg.freeze_writer_at != old["freeze_writer_at"]:
            assert old["freeze_writer_at"] is None and cfg.freeze_writer_at is not None
            assert cfg.freeze_writer_at >= saved["step"], "cannot change past writer updates"
            mutable.add("freeze_writer_at")
        assert all(old[k] == v for k, v in asdict(cfg).items() if k not in mutable), "resume config changes training"
        generator.load_state_dict(saved["generator"])
        critic.load_state_dict(saved["critic"])
        prior.load_state_dict(saved["prior"])
        opt_g.load_state_dict(saved["opt_g"])
        opt_d.load_state_dict(saved["opt_d"])
        data_rng.set_state(saved["data_rng"].cpu())
        latent_rng.set_state(saved["latent_rng"].cpu())
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.startswith("cuda"):
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), device)
        start = saved["step"]
        assert cfg.steps > start
    resolved = {**asdict(cfg), "resolved_recipe": recipe.to_dict(), "device": device,
                "gradient_clipping": None,
                "ema": False, "g_state_gradient": "full BPTT; writer weights frozen",
                "writer_training": ("frozen initialization" if cfg.frozen_writer else
                                    f"D only; frozen after update {cfg.freeze_writer_at}" if cfg.freeze_writer_at is not None else "D only"),
                "particle_policy": "fixed_per_trajectory", "initial_memory": "zeros",
                "parameters": {"g": sum(p.numel() for p in generator.parameters()),
                               "d": sum(p.numel() for p in critic.parameters())}}
    (out/"config.json").write_text(json.dumps(resolved, indent=2))
    log(event="start", start_step=start, config=resolved)
    started = time.monotonic()
    for step in range(start+1, cfg.steps+1):
        if cfg.freeze_writer_at is not None and step > cfg.freeze_writer_at:
            critic.writer.requires_grad_(False)
        scale = learning_rate_scale(step-1, cfg.schedule_steps, recipe.lr_anneal_start, recipe.lr_floor)
        for opt, base in zip((opt_g, opt_d), rates):
            for group, rate in zip(opt.param_groups, base):
                group["lr"] = rate*scale
        real, _ = circles(cfg.batch_size, cfg.train_length, data_rng, device, noise=cfg.noise)
        z, indices = prior.sample(cfg.batch_size, generator=latent_rng)
        with torch.no_grad():
            fake, _ = rollout(generator, critic.writer, z, cfg.train_length)
        opt_d.zero_grad(set_to_none=True)
        d_adv = gan.d_loss(critic(real), critic(fake))
        d_reg = penalty(critic, real, fake, step=step)
        (d_adv+d_reg).backward()
        opt_d.step()
        opt_d.zero_grad(set_to_none=True)
        with frozen(critic):
            opt_g.zero_grad(set_to_none=True)
            fake, _ = rollout(generator, critic.writer, z, cfg.train_length)
            with torch.no_grad():
                real_scores = critic(real)
            g_adv = gan.g_loss(critic(fake), real_scores)
            (g_adv+spread(prior(indices.unique()))).backward()
            opt_g.step()
        if step == start+1 or step % cfg.log_every == 0 or step == cfg.steps:
            losses = {"d": d_adv.item(), "g": g_adv.item(), "penalty": d_reg.item()}
            if not all(np.isfinite(v) for v in losses.values()):
                raise RuntimeError(f"nonfinite losses: {losses}")
            log(event="train", step=step, seconds=round(time.monotonic()-started, 2), **losses)
    # Save training state BEFORE evaluation. A promoted run resumes exactly.
    torch.save({"generator": generator.state_dict(), "critic": critic.state_dict(),
                "writer": critic.writer.state_dict(), "prior": prior.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                "data_rng": data_rng.get_state(), "latent_rng": latent_rng.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.startswith("cuda") else None,
                "step": cfg.steps, "config": asdict(cfg)}, out/"model.pt")
    generator.eval(), critic.eval(), prior.eval()
    metrics, paths = evaluate(generator, critic, prior, cfg, device)
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
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = Config(**json.loads(args.config.read_text()))
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    provenance = {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "torch": torch.__version__, "argv": sys.argv, "sources": {}}
    for name in ("memory_scout.py", "autonomous_memory.py", "memory_path.py"):
        source = Path(__file__).with_name(name).read_bytes()
        (args.out/name).write_bytes(source)
        provenance["sources"][name] = hashlib.sha256(source).hexdigest()
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
