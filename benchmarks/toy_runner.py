"""One shared runner for problem-only toys.

A toy declares only its problem -- data sampler(s), networks, metrics and a
verdict -- by subclassing ``ToyProblem``. ``run`` (or ``ToyRun``) builds
everything else from the training recipe: the recipe-built optimizers (which
own the LR schedule), the loss, the critic penalty, the prior, critic input
noise, generator output noise, EMA, observation logging, checkpoints, and
the extended / hold / shift protocols. The loop inside is the ordinary
``zero_grad`` / ``backward`` / ``opt.step()``.

Minimal declaration::

    class Ring(ToyProblem):
        name = "ring"
        def recipe(self):                     # the shipped recipe at this task's shape
            return get_recipe(z_dim=4, num_particles=12, batch_size=128, total_steps=1200)
        def networks(self, recipe, seed):     # explicit init is particlegan.init tooling
            return Networks(generator=G, critics=D)   # prior defaults to recipe.make_prior()
        def real(self, n, stream):
            return sample(n, stream)
        def metrics(self, model):
            return score(model.sample(4096).x)
        def verdict(self, metrics):
            return "PASS" if ... else "FAIL"

    result = run(Ring(), log_path="runs/toy-refactor/ring.log")

Extension points (all optional; the defaults are the 1G + prior + 1D game):

* ``Networks.critics``: one module or ``{name: module}``. Each critic gets its
  own ``make_critic_optimizer`` + ``make_critic_penalty``; ``{}`` trains the
  generator side on ``losses`` alone (student-only / supervised arms).
* ``Networks.encoder`` / ``Networks.extra``: more generator-side modules
  (AE + D, several generators); they share the generator optimizer, EMA and
  ``losses``.
* ``Networks.prior``: ``RECIPE_PRIOR`` (default), a prior module, a tuple of
  priors (each gets a ``recipe.prior_param_group``), or None (student-only,
  particles-only: override ``fake``).
* ``Networks.direct_particles``: parameters that are the samples themselves
  (particles-only toys); they form the recipe's direct-particle group.
* ``Networks.recipes``: per-role recipes, ``{"generator" | critic name: Recipe}``;
  each role's optimizer/penalty comes from that recipe's factories.
* ``ToyProblem.fake``: how generator-side modules produce a batch (default:
  ``generator(prior.sample(n))``), e.g. conditional or residual students.
* ``ToyProblem.views``: which critic scores which (real, fake, condition)
  with what weight (default: every critic sees the pair once), e.g. several
  scales of one conditional critic, or a critic per generator.
* ``ToyProblem.losses``: extra named loss terms per role ("generator" or a
  critic name), e.g. reconstruction or a problem-defined constraint.
* ``ToyProblem.shift``: mutate the target in place for the shift protocol.

Not expressible on purpose: formulation classes, caller-set learning rates,
freezing optimizer updates mid-run, and host-local optimizers.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from particlegan import ParticlePrior, Recipe
from particlegan.training import InputNoise, input_noise_std, output_noise_std

__all__ = ["Sample", "View", "Networks", "Model", "ToyProblem", "ToyRun", "run", "RECIPE_PRIOR", "main"]


class _RecipePrior:
    def __repr__(self):
        return "RECIPE_PRIOR"


RECIPE_PRIOR = _RecipePrior()
"""Default ``Networks.prior``: ``recipe.make_prior()`` built by the runner."""


@dataclass(frozen=True)
class Sample:
    """A batch plus the conditioning its critic reads: ``critic(x, *condition)``."""
    x: torch.Tensor
    condition: tuple = ()
    indices: torch.Tensor | None = None  # prior rows used, when drawn from a particle table


@dataclass(frozen=True)
class View:
    """One critic evaluation: ``critic(real, *condition)`` vs ``critic(fake, *condition)``."""
    critic: str
    real: torch.Tensor
    fake: torch.Tensor
    condition: tuple = ()
    weight: float = 1.0


@dataclass
class Networks:
    """The problem's modules. Everything that trains adversarially or on
    ``losses`` on the generator side shares one recipe generator optimizer."""
    generator: nn.Module | None
    critics: nn.Module | Mapping[str, nn.Module]
    prior: Any = RECIPE_PRIOR
    encoder: nn.Module | None = None
    extra: Sequence[nn.Module] = ()
    direct_particles: Sequence[nn.Parameter] | None = None
    recipes: Mapping[str, Recipe] = field(default_factory=dict)

    @property
    def priors(self) -> tuple:
        if self.prior is None:
            return ()
        return tuple(self.prior) if isinstance(self.prior, (tuple, list)) else (self.prior,)

    def generator_side(self) -> list[nn.Module]:
        modules = [self.generator, self.encoder, *self.extra, *self.priors]
        return [m for m in modules if m is not None]


def _as_sample(value) -> Sample:
    if isinstance(value, Sample):
        return value
    if isinstance(value, torch.Tensor):
        return Sample(value)
    raise TypeError("real()/fake() must return a Tensor or a Sample")


class Model:
    """What ``ToyProblem.metrics`` sees: live or EMA networks and a sampler.

    ``sample(n)`` draws through ``ToyProblem.fake`` without training noise,
    from an evaluation stream that restarts at every measurement.
    """

    def __init__(self, problem, nets: Networks, stream: torch.Generator, *, ema: bool):
        self.problem, self.nets, self.stream, self.ema = problem, nets, stream, ema

    def sample(self, n: int) -> Sample:
        return _as_sample(self.problem.fake(self.nets, n, self.stream, None))


class ToyProblem:
    """Declare a toy's problem only; see the module docstring."""

    name = "toy"

    def recipe(self) -> Recipe:
        raise NotImplementedError

    def networks(self, recipe: Recipe, seed: int) -> Networks:
        raise NotImplementedError

    def real(self, n: int, stream: torch.Generator) -> torch.Tensor | Sample:
        raise NotImplementedError

    def metrics(self, model: Model) -> dict:
        raise NotImplementedError

    def verdict(self, metrics: dict) -> str:
        raise NotImplementedError

    def fake(self, nets: Networks, n: int, stream: torch.Generator, real: Sample | None) -> torch.Tensor | Sample:
        """Default: ``generator(prior.sample(n))``. ``real`` is the batch it is paired with
        (None when sampling for metrics)."""
        if len(nets.priors) != 1 or nets.generator is None:
            raise NotImplementedError("override fake() for toys without exactly one prior and one generator")
        latent, indices = nets.priors[0].sample(n, generator=stream)
        return Sample(nets.generator(latent), indices=indices)

    def views(self, nets: Networks, real: Sample, fake: Sample) -> list[View]:
        """Default: every critic scores (real, fake) once under real's condition."""
        return [View(name, real.x, fake.x, real.condition) for name in _critics(nets)]

    def losses(self, role: str, nets: Networks, real: Sample, fake: Sample) -> Mapping[str, torch.Tensor]:
        """Extra named loss terms for ``role`` ("generator" or a critic name)."""
        return {}

    def shift(self) -> None:
        """Move the target in place (shift protocol)."""
        raise NotImplementedError(f"{self.name} declares no target shift")


def _critics(nets: Networks) -> dict[str, nn.Module]:
    critics = nets.critics
    if isinstance(critics, nn.Module) and not isinstance(critics, nn.ModuleDict):
        return {"critic": critics}
    return dict(critics)


class ToyRun:
    """One training run of a ``ToyProblem`` under its recipe.

    ``step()`` performs one critic update per critic and one generator-side
    update; ``measure(ema=...)`` scores the problem's metrics; ``state_dict``
    / ``load_state_dict`` checkpoint everything (the optimizers carry the
    schedule, K3P and EMA-critic state).
    """

    def __init__(self, problem: ToyProblem, *, recipe: Recipe | None = None, seed: int = 0,
                 device: str | torch.device = "cpu"):
        self.problem = problem
        self.recipe = problem.recipe() if recipe is None else recipe
        if not isinstance(self.recipe, Recipe):
            raise TypeError("the problem's recipe must be a particlegan Recipe")
        self.seed, self.device = seed, torch.device(device)
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            nets = problem.networks(self.recipe, seed)
            if nets.prior is RECIPE_PRIOR:
                nets.prior = self.recipe.make_prior()
        self.nets = nets
        for module in (*nets.generator_side(), *_critics(nets).values()):
            module.to(self.device)
        self.critics = _critics(nets)
        self.role_recipes = {role: nets.recipes.get(role, self.recipe) for role in ("generator", *self.critics)}
        unknown = set(nets.recipes) - set(self.role_recipes)
        if unknown:
            raise ValueError(f"recipes for unknown roles: {sorted(unknown)}")
        self.opt_g = self._generator_optimizer()
        self.opt_d, self.penalties, self.noisy = {}, {}, {}
        self.noise_stream = self._stream(5)
        for name, critic in self.critics.items():
            recipe = self.role_recipes[name]
            self.opt_d[name] = recipe.make_critic_optimizer(critic, ema_critic=deepcopy(critic))
            self.penalties[name] = recipe.make_critic_penalty(self.opt_d[name])
            self.noisy[name] = InputNoise(critic, 0.0, self.noise_stream)
        self.loss = self.recipe.make_loss()
        self.prior_regularizer = self.recipe.make_prior_regularizer(weight=1.0)
        self.ema = [deepcopy(m).requires_grad_(False) for m in nets.generator_side()]
        self.ema_nets = self._ema_networks()
        self.data_stream, self.latent_stream = self._stream(0), self._stream(2)
        self.completed_steps = 0

    # -- construction ---------------------------------------------------------
    def _stream(self, offset):
        return torch.Generator(device=self.device).manual_seed(self.seed + offset)

    def _generator_optimizer(self):
        nets, recipe = self.nets, self.role_recipes["generator"]
        prior_ids = {id(p) for prior in nets.priors for p in prior.parameters()}
        direct = [p for p in (nets.direct_particles or ()) if p.requires_grad]
        seen = prior_ids | {id(p) for p in direct}
        network = []
        for module in (nets.generator, nets.encoder, *nets.extra):
            for p in (() if module is None else module.parameters()):
                if p.requires_grad and id(p) not in seen:
                    network.append(p)
                    seen.add(id(p))
        groups = [{"params": network}] if network else []
        if direct:
            groups.append({"params": direct})
        for prior in nets.priors:
            params = [p for p in prior.parameters() if p.requires_grad]
            if params:
                groups.append(recipe.prior_param_group(params))
        if not groups:
            raise ValueError("the problem declares no trainable generator-side parameters")
        tables = [prior.z for prior in nets.priors if type(prior) is ParticlePrior and prior.z.requires_grad]
        return recipe.make_generator_optimizer(groups, latent_table=tables[0] if len(tables) == 1 else None,
                                               direct_particles=direct or None)

    def _ema_networks(self):
        nets, copies = self.nets, dict(zip(map(id, self.nets.generator_side()), self.ema))

        def swap(module):
            return None if module is None else copies[id(module)]
        prior = nets.prior
        if isinstance(prior, (tuple, list)):
            prior = type(prior)(swap(p) for p in prior)
        elif prior is not None:
            prior = swap(prior)
        return Networks(generator=swap(nets.generator), critics=nets.critics, prior=prior,
                        encoder=swap(nets.encoder), extra=tuple(swap(m) for m in nets.extra),
                        direct_particles=nets.direct_particles, recipes=nets.recipes)

    # -- training -------------------------------------------------------------
    def _noisy_fake(self, stream, real):
        sample = _as_sample(self.problem.fake(self.nets, self.recipe.batch_size, stream, real))
        sigma = output_noise_std(self.recipe, self.completed_steps)
        if sigma == 0:
            return sample
        x = sample.x
        noise = torch.randn(x.shape, generator=self.noise_stream, device=x.device, dtype=x.dtype)
        return Sample(x + sigma * noise, sample.condition, sample.indices)

    def _set_modes(self, *, critics_train):
        for critic in self.critics.values():
            critic.train(critics_train)
        for module in self.nets.generator_side():
            module.train(not critics_train)

    def step(self) -> dict:
        """One update of every critic, then of the generator side; returns detached losses."""
        recipe, problem, nets, batch = self.recipe, self.problem, self.nets, self.recipe.batch_size
        sigma_in = input_noise_std(recipe, self.completed_steps)
        for wrapper in self.noisy.values():
            wrapper.std = sigma_in
        out = {}
        if self.critics:
            self._set_modes(critics_train=True)
            real = _as_sample(problem.real(batch, self.data_stream))
            with torch.no_grad():
                fake = self._noisy_fake(self.latent_stream, real)
            d_total = 0.0
            for view in problem.views(nets, real, fake):
                critic = self.noisy[view.critic]
                d = self.loss.d_loss(critic(view.real, *view.condition), critic(view.fake, *view.condition))
                d = d + self.penalties[view.critic](critic, view.real, view.fake, *view.condition)
                d_total = d_total + view.weight * d
            for name in self.critics:
                for term in problem.losses(name, nets, real, fake).values():
                    d_total = d_total + term
            for opt in self.opt_d.values():
                opt.zero_grad()
            d_total.backward()
            for opt in self.opt_d.values():
                opt.step()
            out["loss_d"] = d_total.detach()
        self._set_modes(critics_train=False)
        flags = [(p, p.requires_grad) for c in self.critics.values() for p in c.parameters()]
        try:
            for critic in self.critics.values():
                critic.requires_grad_(False)
            real = _as_sample(problem.real(batch, self.data_stream))
            fake = self._noisy_fake(self.latent_stream, real)
            g_total = fake.x.new_zeros(())
            for view in problem.views(nets, real, fake):
                critic = self.noisy[view.critic]
                g = self.loss.g_loss(critic(view.fake, *view.condition), critic(view.real, *view.condition))
                g_total = g_total + view.weight * g
            out["loss_gan"] = g_total.detach()
            prior_reg = self.recipe.prior_reg
            for prior in nets.priors:
                if prior_reg > 0 and type(prior) is ParticlePrior and prior.z.requires_grad:
                    # As GANTrainer: the whole table when small, else the rows drawn.
                    rows = prior.z
                    if len(rows) > 1024 and len(nets.priors) == 1 and fake.indices is not None:
                        rows = rows[torch.unique(fake.indices)]
                    g_total = g_total + prior_reg * self.prior_regularizer(rows)
            for key, term in problem.losses("generator", nets, real, fake).items():
                g_total = g_total + term
                out[key] = term.detach()
            self.opt_g.zero_grad()
            g_total.backward()
            self.opt_g.step()
        finally:
            for p, flag in flags:
                p.requires_grad_(flag)
        out["loss_g"] = g_total.detach()
        decay = recipe.ema_decay
        with torch.no_grad():
            for averaged, live in zip(self.ema, nets.generator_side()):
                for a, p in zip(averaged.parameters(), live.parameters()):
                    a.mul_(decay).add_(p, alpha=1 - decay)
                for a, b in zip(averaged.buffers(), live.buffers()):
                    a.copy_(b)
        self.completed_steps += 1
        out["step"] = self.completed_steps
        return out

    # -- measurement ----------------------------------------------------------
    @torch.no_grad()
    def measure(self, *, ema: bool = False) -> dict:
        """The problem's metrics plus its verdict, without touching training RNGs or modes."""
        nets = self.ema_nets if ema else self.nets
        modules = [*nets.generator_side(), *self.critics.values()]
        modes = [(m, m.training) for root in modules for m in root.modules()]
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        try:
            for module in modules:
                module.eval()
            with torch.random.fork_rng(devices=devices):
                metrics = dict(self.problem.metrics(Model(self.problem, nets, self._stream(9), ema=ema)))
        finally:
            for module, flag in modes:
                module.training = flag
        metrics["verdict"] = self.problem.verdict(metrics)
        return metrics

    # -- checkpoints ----------------------------------------------------------
    def state_dict(self) -> dict:
        streams = {"data": self.data_stream, "latent": self.latent_stream, "noise": self.noise_stream}
        return deepcopy({
            "problem": self.problem.name, "recipe": self.recipe.to_dict(), "completed_steps": self.completed_steps,
            "generator_side": [m.state_dict() for m in self.nets.generator_side()],
            "ema": [m.state_dict() for m in self.ema],
            "critics": {k: c.state_dict() for k, c in self.critics.items()},
            "opt_g": self.opt_g.state_dict(), "opt_d": {k: o.state_dict() for k, o in self.opt_d.items()},
            "streams": {k: s.get_state() for k, s in streams.items()},
        })

    def load_state_dict(self, state: dict) -> None:
        if state["problem"] != self.problem.name or state["recipe"] != self.recipe.to_dict():
            raise ValueError("checkpoint belongs to another problem or recipe")
        for module, values in zip(self.nets.generator_side(), state["generator_side"]):
            module.load_state_dict(values)
        for module, values in zip(self.ema, state["ema"]):
            module.load_state_dict(values)
        for name, critic in self.critics.items():
            critic.load_state_dict(state["critics"][name])
            self.opt_d[name].load_state_dict(state["opt_d"][name])
        self.opt_g.load_state_dict(state["opt_g"])
        for name, stream in (("data", self.data_stream), ("latent", self.latent_stream), ("noise", self.noise_stream)):
            stream.set_state(state["streams"][name])
        self.completed_steps = state["completed_steps"]


def _observation_steps(steps, every):
    if every is None:
        every = max(1, steps // 24)
    return set(range(every, steps + 1, every)) | {steps}


def _hold(curve):
    """Hold protocol: the first PASS, and the step from which every later observation passes."""
    passing = [point["verdict"] == "PASS" for point in curve]
    start = len(curve)
    while start and passing[start - 1]:
        start -= 1
    return {"observations": len(curve), "passing": sum(passing),
            "first_pass_step": next((p["step"] for p, ok in zip(curve, passing) if ok), None),
            "stable_from_step": curve[start]["step"] if start < len(curve) else None}


def run(problem: ToyProblem, *, recipe: Recipe | None = None, steps: int | None = None, seed: int = 0,
        device: str | torch.device = "cpu", observe_every: int | None = None, shift_step: int | None = None,
        log: Callable[[dict], None] | None = None, log_path: str | Path | None = None,
        observer: Callable[[int, Callable[[], dict]], None] | None = None) -> dict:
    """Train ``problem`` under its recipe and score it.

    ``steps`` defaults to ``recipe.total_steps``; a longer run is the extended
    protocol (the recipe schedule and noise keep their ``total_steps`` horizon
    and hold). Live metrics are observed every ``observe_every`` updates
    (default: 24 observations) for the hold summary; ``shift_step`` calls
    ``problem.shift()`` after that many updates and reports recovery. Each
    observation goes to ``log`` and, as one JSON line, to ``log_path`` (easy
    to ``tail -f``). ``observer(step, measure)`` is called after every update
    with a lazy live measurement (e.g. ``benchmarks.locked_shared.observation.checkpoint``).
    """
    toy = ToyRun(problem, recipe=recipe, seed=seed, device=device)
    steps = toy.recipe.total_steps if steps is None else steps
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if shift_step is not None and not 0 < shift_step < steps:
        raise ValueError("shift_step must fall inside the run")
    observe = _observation_steps(steps, observe_every)
    path = None if log_path is None else Path(log_path)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    started = time.monotonic()

    def emit(row):
        if log is not None:
            log(row)
        if path is not None:
            with path.open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False, default=float) + "\n")
    curve = []
    for _ in range(steps):
        if shift_step is not None and toy.completed_steps == shift_step:
            problem.shift()
            emit({"toy": problem.name, "event": "shift", "step": shift_step})
        losses = toy.step()
        step = toy.completed_steps
        if not all(math.isfinite(float(v)) for k, v in losses.items() if k != "step"):
            raise FloatingPointError(f"{problem.name}: non-finite loss at step {step}: {losses}")
        if observer is not None:
            observer(step, toy.measure)
        if step in observe:
            point = {"step": step, **toy.measure(), "seconds": round(time.monotonic() - started, 3)}
            clash = sorted((set(point) | {"toy"}) & (set(losses) - {"step"}))
            if clash:
                raise ValueError(f"{problem.name}: loss names {clash} collide with metric keys; "
                                 "rename the losses so the logged row keeps both")
            curve.append(point)
            emit({"toy": problem.name, **point, **{k: float(v) for k, v in losses.items() if k != "step"}})
    live, ema = toy.measure(), toy.measure(ema=True)
    result = {"problem": problem.name, "steps": steps, "seed": seed, "recipe": toy.recipe.to_dict(),
              "live": live, "ema": ema, "verdict": live["verdict"], "curve": curve, "hold": _hold(curve),
              "seconds": time.monotonic() - started}
    if steps > toy.recipe.total_steps:
        result["extended_from"] = toy.recipe.total_steps
    if shift_step is not None:
        after = [p for p in curve if p["step"] > shift_step]
        result["shift"] = {"step": shift_step, **_hold(after),
                           "recovery_updates": next((p["step"] - shift_step for p in after
                                                     if p["verdict"] == "PASS"), None)}
    emit({"toy": problem.name, "event": "final", "verdict": result["verdict"],
          "live": live, "ema": ema, "hold": result["hold"], "seconds": round(result["seconds"], 3)})
    return result


def main(problem: ToyProblem, argv=None) -> int:
    """Shared CLI: ``python -m <toy module> [--steps N] [--shift-step N] [--log runs/toy-refactor/<name>.log]``."""
    import argparse
    parser = argparse.ArgumentParser(description=f"Train and score the {problem.name} toy on its recipe.")
    parser.add_argument("--steps", type=int, help="default: the recipe budget; more is the extended protocol")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shift-step", type=int)
    parser.add_argument("--observe-every", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log", type=Path, default=Path(f"runs/toy-refactor/{problem.name}.log"))
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    result = run(problem, steps=args.steps, seed=args.seed, device=args.device, shift_step=args.shift_step,
                 observe_every=args.observe_every, log_path=args.log)
    print(json.dumps({k: result[k] for k in ("problem", "steps", "verdict", "live", "ema", "hold")}, default=float))
    return 0 if result["verdict"] == "PASS" else 1
