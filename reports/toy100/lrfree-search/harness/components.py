"""Candidate-policy engine for the custom22 hosts (see reports/custom22-design.md, "Implementation").

The learner is the candidate package's ``GANTrainer`` policy, re-expressed as components that fire in
the order of ``GANTrainer._step``; the task (data, architecture, auxiliary losses, scorers) is the frozen
host. ``Engine.build`` copies ``GANTrainer.__init__`` (same package factories, same order); an update is

    with eng.update(real) as u:          # P0-P4  budget, observe_prior/observe_game, LRs, critic scale, sigma
        u.d_phase()                      # P5     D.train(), G-side .eval()
        fake = u.sample(fn, latent, table) / u.perturb(...) / u.noise(...)       # P6 (no grad, sigma detached)
        u.observe_pair(real_all, fake_all)                                      # P7
        pen = u.penalty(view, x_real, x_fake, *cond)                            # P8 (exactly once)
        u.d_step(loss_d_adv, pen)        # P9     opt_d step (+ settle observe)
        u.g_phase()                      # P10    D.eval(), G-side .train(), D frozen
        fake = u.generate(fn, latent, table) / u.noise(x)                        # P11 (sigma attached)
        u.g_step(loss_g, loss_gan)       # P12    observe_generator between backward and opt_g.step
                                         # P13    (exit) restore D flags, EMA, completed += 1

``scalar_step`` drives these phases exactly like ``GANTrainer._step`` on an ordinary (G, D, prior) triple;
``components_parity.py`` checks it bitwise against the package's own ``GANTrainer.step`` (the L1 gate).
Learning rates are written only in P2/P3 (asserted before every optimizer step). Unsupported policy is
refused (``EngineRefusal``), never silently dropped. No module-level ``particlegan`` import: the package is
passed in (whatever the job imported).
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from copy import deepcopy
import dataclasses
import hashlib
import importlib
import inspect
import math
from types import SimpleNamespace

import torch
from torch import nn

ENGINE_VERSION = 'custom22-engine-1'
# GANTrainer methods/properties the engine re-expresses (union of the parity-checked dv12-ams / dv12-st
# packages). A package whose GANTrainer defines anything else has a hook the engine does not know: refused.
KNOWN_TRAINER_METHODS = frozenset({
    '__init__', '_batch', '_generate', '_output_sigma', '_settle_observe', '_step', '_stream', 'ema_D',
    'latent_damping', 'latent_history', 'load_state_dict', 'output_sigma', 'sample', 'state_dict', 'step'})
# Recipe fields the engine handles (dv12-st Recipe). A field outside this set is refused unless it holds
# its dataclass default (a new policy knob the engine was never parity-checked for).
KNOWN_RECIPE_FIELDS = frozenset((
    'name model z_dim num_particles prior_kind sigma_rel standardize num_classes conditioning ucd_target '
    'ucd_weight alpha_bar batch_size total_steps continuous_policy lr d_lr_mult prior_lr_mult betas prior_betas '
    'reg_coeff reg_kappa reg_every prior_reg ema_decay lr_anneal_start lr_floor network_lr_floor '
    'network_lr_horizon_cap reg_anchor_min_decay reg_anchor_weight direct_particle_gain d_guard_ratio '
    'd_guard_min_steps latent_damping_max_rate direct_particle_betas input_noise_std input_noise_anneal_end '
    'output_noise_std output_noise_warmup encoder_mode routing_temperature distance_reduction observation_sigma '
    'reconstruction_weight initialization amsgrad critic_r1_real critic_payoff_damping output_noise_mode '
    'lr_control particle_birth_death').split())
CONTINUOUS_PENALTY_LINK = ('dv2', 'dv3', 'dv4', 'dv5', 'dv6', 'dv7', 'dv8', 'dv9', 'dv10', 'dv11', 'dv12')
EVAL_SEED_OFFSET = 402            # frozen NoisePolicy.evaluation: global seed + 402 + step
OUTPUT_NOISE_SEED_OFFSET = 1901   # frozen benchmarks/toy100/models.py OUTPUT_NOISE_SEED_OFFSET


class EngineRefusal(RuntimeError):
    """The candidate asks for policy the engine does not bind (never silently dropped)."""


def source_sha256():
    from pathlib import Path
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def trainer_methods(package):
    return sorted(k for k, v in vars(package.GANTrainer).items()
                  if inspect.isfunction(v) or isinstance(v, property))


def check_package(package, recipe, tables=None, *, allow_dv11=False):
    """Refusals that do not depend on a host (see design §2.1).

    ``tables`` is the host binding (available only at build time); the BD
    table check therefore runs in ``build`` via ``check_tables``, while the
    rest stays here so scalar parity (no host tables) keeps working.
    """
    unknown = set(trainer_methods(package)) - KNOWN_TRAINER_METHODS
    if unknown:
        raise EngineRefusal(f'GANTrainer defines hooks the engine does not re-express: {sorted(unknown)}')
    for field in dataclasses.fields(type(recipe)):
        if field.name in KNOWN_RECIPE_FIELDS:
            continue
        default = field.default if field.default is not dataclasses.MISSING else (
            field.default_factory() if field.default_factory is not dataclasses.MISSING else dataclasses.MISSING)
        if getattr(recipe, field.name) != default:
            raise EngineRefusal(f'recipe field {field.name}={getattr(recipe, field.name)!r} is not bound by the engine')
    if tables is not None:
        check_tables(package, recipe, tables)
    if recipe.model != 'gan':
        raise EngineRefusal(f'model={recipe.model!r} (GANTrainer route supports gan only)')
    if recipe.conditioning != 'scalar':
        raise EngineRefusal(f'conditioning={recipe.conditioning!r} (GANTrainer route supports scalar only)')
    if getattr(recipe, 'continuous_policy', None) == 'dv11' and not allow_dv11:
        raise EngineRefusal('dv11 observe_support needs a latent->sample module; the custom hosts bind maps, not G')


def check_tables(package, recipe, tables):
    """BD table check (needs the host binding): bound only on plain trainable tables."""
    if getattr(recipe, 'particle_birth_death', False):
        if not tables:
            raise EngineRefusal('particle_birth_death needs a prior table (no-table hosts)')
        for table in tables:
            if not _is_plain_table(package, table) or not table.z.requires_grad:
                raise EngineRefusal('particle_birth_death needs plain trainable ParticlePrior tables')


def _is_plain_table(package, table):
    return type(table) is package.ParticlePrior


class _View(SimpleNamespace):
    """An object with ``.z`` for the controller's latent-geometry hooks."""


class RoleView(nn.Module):
    """Row-block view of a scale-conditioned critic: ``forward(z) = cat_r critic.score(z[block_r], scale_r)``.

    One KA2 call on the role-union batch (design §5.5). The critic is a child, so the penalty's paired
    EMA view (``CriticPenalty._ema_view``) evaluates the same blocks on the EMA critic.
    """

    def __init__(self, critic, scales, rows):
        super().__init__()
        self.critic = critic
        self.scales = tuple(float(s) for s in scales)
        self.rows = int(rows)

    def forward(self, z):
        if len(z) != self.rows * len(self.scales):
            raise ValueError('RoleView needs equal role blocks in scale order')
        return torch.cat([self.critic.score(z[i * self.rows:(i + 1) * self.rows], s)
                          for i, s in enumerate(self.scales)])


class Engine:
    """The candidate's GANTrainer policy for a host-supplied role registry (design §2)."""

    legacy = False

    def __init__(self, package, overrides, resources=None, *, seed=0, serial_backward=True,
                 optimizer_options=None, penalty_options=None, allow_dv11=False):
        self.package = package
        self.training = importlib.import_module(package.__name__ + '.training')
        self.T = package.GANTrainer
        options = dict(overrides)
        options.update(resources or {})
        self.recipe = package.get_recipe(**options)
        self.resources = dict(resources or {})
        check_package(package, self.recipe, allow_dv11=allow_dv11)
        self.seed = int(seed)
        self.serial_backward = bool(serial_backward)
        self.optimizer_options = dict({'foreach': False, 'fused': False} if optimizer_options is None
                                      else optimizer_options)
        self.penalty_options = dict(penalty_options or {})
        self.built = False
        self.completed_steps = 0
        self.receipts = {}

    # ------------------------------------------------------------------ build (GANTrainer.__init__ order)
    def build(self, *, generator=None, critic, tables=(), encoder=None, latent_generator=None,
              penalty_generator=None, g_container=None, legacy=None, direct_particles=None):
        """``legacy`` (the host's own optimizer/loss/penalty recipe) is used only by the test-only host
        policy (harness/tests/custom22_host_policy.py); the candidate engine ignores it.
        ``direct_particles``: parameters that ARE the samples (two_pole), bound through the package's
        direct-particle API (``_direct_optimizers``) instead of ``make_optimizers``; None everywhere else."""
        package, recipe = self.package, self.recipe
        cpu_before = torch.get_rng_state()
        self.G = generator
        self.encoder = encoder
        self.G_side = [m for m in (generator, encoder) if m is not None]
        self.D = critic
        self.tables = list(tables)
        parameters = list(critic.parameters())
        self.device, self.dtype = parameters[0].device, parameters[0].dtype
        if not any(p.requires_grad for p in parameters):
            raise ValueError('critic must have trainable parameters')
        seen = set()
        for module in (*self.G_side, critic, *self.tables):
            for value in (*module.parameters(), *module.buffers()):
                if value.device != self.device or (value.is_floating_point() and value.dtype != self.dtype):
                    raise ValueError('host modules must share one device and floating dtype')
            for parameter in module.parameters():
                if id(parameter) in seen:
                    raise ValueError('host modules must not share parameters')
                seen.add(id(parameter))
        self.direct = list(direct_particles or ())
        for parameter in self.direct:
            if parameter.device != self.device or parameter.dtype != self.dtype or id(parameter) in seen \
                    or not parameter.requires_grad:
                raise ValueError('direct particles must be trainable, unshared, on the model device and dtype')
            seen.add(id(parameter))
        prior0 = self.tables[0] if self.tables else None
        check_tables(package, recipe, self.tables)
        # 2. make_optimizers: role groups [g(+E), prior], betas, amsgrad, A2 (plain table), KA2 critic Adam
        #    with spike guard and EMA critic, batch_feature_zero on fresh modules (GANTrainer line order).
        if self.direct:
            self.opt_g, self.opt_d = self._direct_optimizers(critic)
        else:
            self.opt_g, self.opt_d = recipe.make_optimizers(
                generator, critic, prior0, encoder=encoder, ema_critic=deepcopy(critic), **self.optimizer_options)
        # 3. further tables: own prior group + own A2 (package LatentRowDamping around opt_g.step, §5.4).
        self.extra_damping = []
        for table in self.tables[1:]:
            betas = recipe.prior_betas if recipe.prior_betas is not None else recipe.betas
            self.opt_g.add_param_group({'params': [p for p in table.parameters() if p.requires_grad],
                                        'lr': recipe.lr * recipe.prior_lr_mult, 'betas': betas})
            if (_is_plain_table(package, table) and table.z.requires_grad
                    and recipe.latent_damping_max_rate > 0):
                if betas[0] != 0.0:
                    raise EngineRefusal('A2 latent damping needs beta1 == 0 for every table group')
                from importlib import import_module
                k3p = import_module(package.__name__ + '.k3p')
                history = torch.zeros_like(table.z, requires_grad=False)
                self.extra_damping.append((k3p.LatentRowDamping(table.z, history,
                                                                max_rate=recipe.latent_damping_max_rate), history))
        # 4. learnable output sigma (GANTrainer: its own opt_g group at the G base LR).
        self._sigma_api = hasattr(self.T, '_output_sigma')
        self.log_output_sigma = None
        self.last_output_sigma = None
        mode = getattr(recipe, 'output_noise_mode', 'fixed')
        if mode == 'learnable':
            if not self._sigma_api:
                raise EngineRefusal('output_noise_mode=learnable without GANTrainer._output_sigma')
            self.log_output_sigma = nn.Parameter(torch.full(
                (), math.log(recipe.output_noise_std), device=self.device, dtype=self.dtype))
            self.opt_g.add_param_group({'params': [self.log_output_sigma], 'lr': recipe.lr})
        elif mode != 'fixed' and not self._sigma_api:
            raise EngineRefusal(f'output_noise_mode={mode!r} without GANTrainer._output_sigma')
        # 5. initial LRs, roles, loss, prior regularizer, EMA copies, streams.
        self.initial_lrs = [[group['lr'] for group in opt.param_groups] for opt in (self.opt_g, self.opt_d)]
        prior_ids = {id(p) for t in self.tables for p in t.parameters()} | {id(p) for p in self.direct}
        self.roles = [['prior' if any(id(p) in prior_ids for p in group['params']) else 'generator'
                       for group in self.opt_g.param_groups], ['critic'] * len(self.opt_d.param_groups)]
        self.loss = recipe.make_loss()
        self.prior_regularizer = recipe.make_prior_regularizer(weight=1.0)
        self.ema_G_side = [deepcopy(m).eval() for m in self.G_side]
        self.ema_tables = [deepcopy(t).eval() for t in self.tables]
        for module in (*self.ema_G_side, *self.ema_tables):
            module.requires_grad_(False)
        self.ema_of = {id(m): e for m, e in zip((*self.G_side, *self.tables), (*self.ema_G_side, *self.ema_tables))}
        self.ema_direct = [p.detach().clone() for p in self.direct]   # EMA of direct particles (reported only)
        self.ema_of.update({id(p): e for p, e in zip(self.direct, self.ema_direct)})
        self.g_container =g_container if g_container is not None else (
            self.G_side[0] if len(self.G_side) == 1 else nn.ModuleList(self.G_side))
        stream = lambda g, s: torch.Generator(device=self.device).manual_seed(self.seed + s) if g is None else g
        self.latent_generator = stream(latent_generator, 2)
        self.penalty_generator = stream(penalty_generator, 3)
        self.eval_generator = stream(None, 4)
        self.noise_generator = stream(None, 5)
        # 6. input-noise wrapper, KA2 penalty, controller (+ penalty link), 7. stationarity LR.
        self._noisy_D = self.training.InputNoise(self.D, 0.0, self.noise_generator)
        self.penalty = recipe.make_critic_penalty(self.opt_d, **self.penalty_options)
        self.completed_steps = 0
        # Packages without the continuous controller (develop K3P: no Recipe.continuous_policy, no
        # particlegan.continuous) run their own horizon schedule through learning_rate_scales.
        policy = getattr(recipe, 'continuous_policy', None)
        stationarity = getattr(recipe, 'lr_control', 'mobility') == 'stationarity'
        continuous = (importlib.import_module(self.package.__name__ + '.continuous')
                      if policy is not None or stationarity else None)
        self.controller = (None if policy is None else continuous.DataDriftController(policy))
        if self.controller is not None and self.tables:
            self.controller.observe_prior(self.table_view())
        if policy in CONTINUOUS_PENALTY_LINK:
            self.penalty.regularizer.continuous_controller = self.controller
        self.lr_settle = None
        if stationarity:
            self.lr_settle = continuous.StationarityLR(
                (self.opt_g, self.opt_d), prior_param=self.tables[0].z if self.tables else None)
        # Birth-death (GANTrainer.__init__ order: after lr_settle, before built).
        # Bound only on plain trainable tables (check_tables); refused elsewhere.
        # The package module is constructed with the engine as ``trainer`` so
        # every attribute it reads exists: G/G_side (generator or module list),
        # prior (tables[0]), ema_prior (ema_tables[0]), opt_g (Adam state +
        # A2 latent_history via the package's own generator optimizer),
        # controller, completed_steps, device/dtype. Private stream seed + 6,
        # exactly as GANTrainer.
        self.birth_death = None
        if getattr(recipe, 'particle_birth_death', False):
            from importlib import import_module
            self.G = self.g_container
            self.ema_G = self.ema_G_side[0] if len(self.ema_G_side) == 1 else nn.ModuleList(self.ema_G_side)
            self.prior = self.tables[0]
            self.ema_prior = self.ema_tables[0]
            birth_death = import_module(package.__name__ + '.birth_death')
            self.birth_death = birth_death.ParticleBirthDeath(self, self.seed + 6)
        self.built = True
        self.penalty_calls = 0
        self.last_penalty_stats = {}
        # Construction must not consume the host's global RNG (design §5.2 / L3c receipt).
        self.receipts['construction_global_rng_untouched'] = bool(torch.equal(cpu_before, torch.get_rng_state()))
        if not self.receipts['construction_global_rng_untouched']:
            raise RuntimeError('engine construction consumed the global RNG')
        self.receipts['groups'] = self.group_table()
        return self

    def _direct_optimizers(self, critic):
        """Direct sample particles (the host's samples are its parameters; two_pole) through the package's own
        API: "Direct sample-particle groups use ``recipe.make_generator_optimizer(params, direct_particles=[...])``"
        (docs/k3p.md; recipe fields ``direct_particle_betas`` / ``direct_particle_gain``). The group is built as
        ``make_optimizers`` builds its prior group (``lr * prior_lr_mult``, ``prior_betas or betas``; prior role,
        as the frozen compare route keeps opt_p "direct particles prior-owned"); inside its step the package's
        ``DirectParticleResponse`` applies ``direct_particle_betas`` and the centered-gradient LR gain, as the
        frozen K3P route's response hook did for this group. Not a latent table: no A2, no controller latent
        hooks. The critic optimizer is ``make_optimizers``' own ``make_critic_optimizer`` call."""
        recipe = self.recipe
        if self.G_side or self.tables:
            raise EngineRefusal('direct sample particles are bound only on hosts without G/E/tables')
        if recipe.initialization is not None:
            raise EngineRefusal('direct particles need initialization=None (make_optimizers owns fresh-module '
                                'initialization and has no direct-particle path)')
        betas = recipe.prior_betas if recipe.prior_betas is not None else recipe.betas
        groups = [{'params': self.direct, 'lr': recipe.lr * recipe.prior_lr_mult, 'betas': betas}]
        return (recipe.make_generator_optimizer(groups, direct_particles=self.direct, **self.optimizer_options),
                recipe.make_critic_optimizer(critic, ema_critic=deepcopy(critic), **self.optimizer_options))

    # ------------------------------------------------------------------ views, sigma, GANTrainer delegates
    def table_view(self, tables=None):
        """``observe_prior`` view: the table itself (one plain table, bitwise GANTrainer), the union of
        several tables (§5.3), or MoG component centres ``means()`` (§8 B1)."""
        tables = self.tables if tables is None else tables
        if not tables:
            return None
        views = [self.perturb_prior(t) for t in tables]
        if len(views) == 1:
            return views[0]
        return _View(z=torch.cat([v.z.detach() for v in views], 0))

    def perturb_prior(self, table):
        """The ``prior`` argument of ``perturb_latent`` for a latent drawn from ``table``."""
        if table is None:
            return None
        if _is_plain_table(self.package, table):
            return table
        if hasattr(table, 'means'):
            return _View(z=table.means().detach())   # B1: MoG samples live in means() coordinates
        raise EngineRefusal(f'no latent-geometry view for {type(table).__name__}')

    def _output_sigma(self, base, detach=True):
        return self.T._output_sigma(self, base, detach)

    def output_sigma(self):
        if self._sigma_api:
            return float(self.T.output_sigma(self))
        return float(self.training.output_noise_std(self.recipe, self.completed_steps))

    def _settle_observe(self, index):
        return self.T._settle_observe(self, index)

    def group_table(self):
        rows = []
        for name, opt, roles, rates in (('opt_g', self.opt_g, self.roles[0], self.initial_lrs[0]),
                                        ('opt_d', self.opt_d, self.roles[1], self.initial_lrs[1])):
            for index, (group, role, rate) in enumerate(zip(opt.param_groups, roles, rates)):
                owner = 'log_output_sigma' if self.log_output_sigma is not None and any(
                    p is self.log_output_sigma for p in group['params']) else role
                rows.append(dict(optimizer=name, group=index, role=role, owner=owner, initial_lr=rate,
                                 betas=list(group['betas']), amsgrad=group.get('amsgrad'),
                                 parameters=sum(p.numel() for p in group['params'])))
        return rows

    def current_lrs(self):
        return [[g['lr'] for g in o.param_groups] for o in (self.opt_g, self.opt_d)]

    def add_prior_reg(self, loss, z):
        """GANTrainer's ``loss_gan + recipe.prior_reg * prior_regularizer(z)`` for a table without a host
        VICReg site (ae_gan_hold; two_pole's direct particles are no prior table); ``prior_regularizer`` is ``make_prior_regularizer(weight=1)``."""
        if not z.requires_grad:
            return loss
        return loss + self.recipe.prior_reg * self.prior_regularizer(z)

    # ------------------------------------------------------------------ one update
    @contextmanager
    def update(self, real, *, collect_stats=False):
        if not self.built:
            raise RuntimeError('build() first')
        with ExitStack() as stack:
            if self.serial_backward:
                stack.enter_context(torch.autograd.set_multithreading_enabled(False))
            u = Update(self, real, collect_stats)
            u._enter()
            try:
                yield u
                u._check_complete()
            finally:
                u._restore_flags()
            u._exit()

    # ------------------------------------------------------------------ evaluation scope (design §2.3)
    @contextmanager
    def evaluate(self, step, *, noisy=True):
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] \
            if self.device.type == 'cuda' else []
        streams = [g.get_state().clone() for g in (self.latent_generator, self.penalty_generator,
                                                   self.eval_generator, self.noise_generator)]
        modules = [*self.G_side, *self.ema_G_side, *self.tables, *self.ema_tables]
        modes = [(m, m.training) for root in modules for m in root.modules()]
        try:
            for root in modules:
                root.eval()
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(self.seed + EVAL_SEED_OFFSET + step)
                if self.device.type == 'cuda':
                    torch.cuda.default_generators[devices[0]].manual_seed(self.seed + EVAL_SEED_OFFSET + step)
                stream = torch.Generator(device=self.device).manual_seed(
                    self.seed + EVAL_SEED_OFFSET + step + OUTPUT_NOISE_SEED_OFFSET)
                yield Evaluation(self, stream, self.output_sigma() if noisy else 0.)
        finally:
            for module, flag in modes:
                module.training = flag
        after = [g.get_state() for g in (self.latent_generator, self.penalty_generator,
                                         self.eval_generator, self.noise_generator)]
        if not all(torch.equal(a, b) for a, b in zip(streams, after)):
            raise RuntimeError('evaluation advanced a training stream')

    # ------------------------------------------------------------------ export
    def state_dict(self):
        return deepcopy({
            'engine': ENGINE_VERSION, 'recipe': self.recipe.to_dict(), 'completed_steps': self.completed_steps,
            'G_side': [m.state_dict() for m in self.G_side], 'D': self.D.state_dict(),
            'tables': [t.state_dict() for t in self.tables],
            'ema_G_side': [m.state_dict() for m in self.ema_G_side],
            'ema_tables': [t.state_dict() for t in self.ema_tables],
            'optimizers': [self.opt_g.state_dict(), self.opt_d.state_dict()],
            'extra_damping': [dict(state=d.state_dict(), history=h) for d, h in self.extra_damping],
            'initial_lrs': self.initial_lrs,
            'streams': {n: getattr(self, n).get_state() for n in ('latent_generator', 'penalty_generator',
                                                                  'eval_generator', 'noise_generator')},
            'controller': None if self.controller is None else self.controller.state_dict(),
            'lr_settle': None if self.lr_settle is None else self.lr_settle.state_dict(),
            'birth_death': None if self.birth_death is None else self.birth_death.state_dict(),
            'log_output_sigma': None if self.log_output_sigma is None else self.log_output_sigma.detach(),
            'cpu_rng': torch.get_rng_state(),
            **({'direct': [p.detach() for p in self.direct], 'ema_direct': self.ema_direct} if self.direct else {})})


class Update:
    """Phase object of one update (design §2.2). Order is asserted."""

    def __init__(self, eng, real, collect_stats):
        self.eng, self.collect_stats = eng, bool(collect_stats)
        if not isinstance(real, torch.Tensor) or real.ndim < 2 or not len(real) \
                or real.device != eng.device or real.dtype != eng.dtype:
            raise ValueError('real must be a nonempty batch on the model device and dtype')
        self.real = real.detach()
        self.stage = 'new'
        self.counts = dict(observe_pair=0, penalty=0)
        self.flags = None
        self.penalty_value = self.penalty_stats = self.loss_d = None

    # P0-P4
    def _enter(self):
        eng, recipe = self.eng, self.eng.recipe
        if recipe.total_steps is not None and eng.completed_steps >= recipe.total_steps:
            raise RuntimeError('recipe training budget exhausted')
        c = eng.controller
        if c is not None:
            view = eng.table_view()
            if view is not None:
                c.observe_prior(view)
            c.observe_game(eng.penalty.regularizer.record)
        if eng.birth_death is not None:
            eng.birth_death.observe_real(self.real.detach().flatten(1) if self.real.ndim > 2 else self.real)
        if eng.lr_settle is None:
            network, prior_scale = (eng.package.learning_rate_scales(eng.completed_steps, recipe)
                                    if c is None else c.observe_real(self.real))
            for optimizer, rates, roles in zip((eng.opt_g, eng.opt_d), eng.initial_lrs, eng.roles):
                for group, rate, role in zip(optimizer.param_groups, rates, roles):
                    group['lr'] = rate * (prior_scale if role == 'prior' else network)
        else:
            c.observe_real(self.real)
            reopen = c.data_score > 3.
            for group, tester in eng.lr_settle.pairs((eng.opt_g, eng.opt_d)):
                if reopen:
                    tester.restart(group['params'], reopen=True)
                else:
                    tester.begin(group['params'])
            for optimizer, rates, testers in zip((eng.opt_g, eng.opt_d), eng.initial_lrs, eng.lr_settle.testers):
                for group, rate, tester in zip(optimizer.param_groups, rates, testers):
                    group['lr'] = rate * (1. if tester is None else tester.s)
        if c is not None and getattr(recipe, 'critic_payoff_damping', True):
            for group in eng.opt_d.param_groups:
                group['lr'] *= c.critic_scale()
        self.lrs = eng.current_lrs()
        self.sigma_in = eng.training.input_noise_std(recipe, eng.completed_steps)
        eng._noisy_D.std = self.sigma_in
        base = eng.training.output_noise_std(recipe, eng.completed_steps)
        if eng._sigma_api:
            self.sigma_out = eng._output_sigma(base, detach=False)
            eng.last_output_sigma = float(self.sigma_out.detach() if torch.is_tensor(self.sigma_out)
                                          else self.sigma_out)
        else:
            self.sigma_out = base
        self.stage = 'entered'

    def _require(self, *stages):
        if self.stage not in stages:
            raise AssertionError(f'engine phase order: {self.stage} not in {stages}')

    def _check_lrs(self):
        if self.eng.current_lrs() != self.lrs:
            raise AssertionError('a learning rate changed outside the engine P2/P3 phases')

    def _sigma(self):
        if self.stage == 'g':
            return self.sigma_out
        return self.sigma_out.detach() if torch.is_tensor(self.sigma_out) else self.sigma_out

    def _draw(self, y, sigma, stream):
        if sigma == 0:
            return y
        return y + sigma * torch.randn(y.shape, generator=stream, device=y.device, dtype=y.dtype)

    # P5
    def d_phase(self):
        self._require('entered')
        self.eng.D.train()
        for module in self.eng.G_side:
            module.eval()
        self.stage = 'd'

    # P6 / P11 building blocks
    def perturb(self, latent, table):
        """``controller.perturb_latent`` right after a table draw (GANTrainer ``_generate``)."""
        self._require('d', 'g')
        c = self.eng.controller
        if c is None:
            return latent
        return c.perturb_latent(latent, self.eng.noise_generator, self.eng.perturb_prior(table), record=True)

    def noise(self, x):
        """Output noise at a frozen site: detached sigma in the D phase, attached in the G phase."""
        self._require('d', 'g')
        return self._draw(x, self._sigma(), self.eng.noise_generator)

    def sample(self, fn, latent=None, table=None):
        """D-phase fake: observe_support, perturb, map, output noise; no grad (GANTrainer lines 295-299)."""
        self._require('d')
        eng = self.eng
        with torch.no_grad():
            if eng.controller is not None and latent is not None:
                eng.controller.observe_support(fn if isinstance(fn, nn.Module) else None, eng.D, latent,
                                               self.sigma_out, eng.noise_generator)
            if latent is not None:
                latent = self.perturb(latent, table)
            y = fn(latent) if latent is not None else fn()
            return self._draw(y, self.sigma_out, eng.noise_generator)

    def generate(self, fn, latent=None, table=None):
        """G-phase fake: perturb, map, output noise with the attached sigma (GANTrainer ``_generate``)."""
        self._require('g')
        if latent is not None:
            latent = self.perturb(latent, table)
        y = fn(latent) if latent is not None else fn()
        return self._draw(y, self.sigma_out, self.eng.noise_generator)

    # P7
    def observe_pair(self, real, fake):
        self._require('d')
        self.counts['observe_pair'] += 1
        if self.eng.controller is not None:
            self.eng.controller.observe_pair(real, fake)

    # P8
    def penalty(self, view, x_real, x_fake, *condition):
        self._require('d')
        self.counts['penalty'] += 1
        if self.counts['penalty'] != 1:
            raise AssertionError('exactly one critic penalty call per update')
        if self.sigma_in != 0:
            raise EngineRefusal('critic input noise > 0 is not bound for custom hosts')
        pen = self.eng.penalty
        pen.collect_stats = self.collect_stats
        self.penalty_value = pen(view, x_real, x_fake, *condition)
        self.penalty_stats = pen.last_stats
        self.eng.last_penalty_stats = pen.last_stats
        self.eng.penalty_calls += 1
        return self.penalty_value

    # P9
    def d_step(self, loss_d_adv, penalty):
        self._require('d')
        if self.counts['penalty'] != 1 or penalty is not self.penalty_value:
            raise AssertionError('d_step needs the one penalty of this update')
        if self.counts['observe_pair'] != 1:
            raise AssertionError('observe_pair must run once before the critic step')
        eng = self.eng
        self.loss_d = loss_d_adv + penalty
        eng.opt_d.zero_grad()
        self.loss_d.backward()
        self._check_lrs()
        eng.opt_d.step()
        if eng.lr_settle is not None:
            eng._settle_observe(1)
        self.stage = 'd_done'

    # P10
    def g_phase(self):
        self._require('d_done')
        eng = self.eng
        eng.D.eval()
        for module in eng.G_side:
            module.train()
        self.flags = [p.requires_grad for p in eng.D.parameters()]
        eng.D.requires_grad_(False)
        self.stage = 'g'

    # P12
    def g_step(self, loss_g, loss_gan):
        self._require('g')
        eng = self.eng
        eng.opt_g.zero_grad()
        loss_g.backward()
        if eng.controller is not None:
            eng.controller.observe_generator(eng.g_container, loss_gan.detach(),
                                             (self.loss_d - self.penalty_value).detach())
        self._check_lrs()
        with ExitStack() as stack:
            for damping, _history in eng.extra_damping:
                stack.enter_context(damping.around(eng.opt_g))
            eng.opt_g.step()
        if eng.lr_settle is not None:
            eng._settle_observe(0)
        self.loss_g, self.loss_gan = loss_g, loss_gan
        self.stage = 'g_done'

    def _check_complete(self):
        self._require('g_done')

    def _restore_flags(self):
        if self.flags is not None:
            for parameter, flag in zip(self.eng.D.parameters(), self.flags):
                parameter.requires_grad_(flag)
            self.flags = None

    # P13
    def _exit(self):
        eng, decay = self.eng, self.eng.recipe.ema_decay
        with torch.no_grad():
            for target, source in zip((*eng.ema_G_side, *eng.ema_tables), (*eng.G_side, *eng.tables)):
                for averaged, current in zip(target.parameters(), source.parameters()):
                    averaged.mul_(decay).add_(current, alpha=1 - decay)
                for averaged, current in zip(target.buffers(), source.buffers()):
                    averaged.copy_(current)
            for averaged, current in zip(eng.ema_direct, eng.direct):
                averaged.mul_(decay).add_(current, alpha=1 - decay)
        # Birth-death moves (GANTrainer._step tail: after EMA, before completed += 1).
        # maybe_apply runs its evaluation when the reservoir turns over and moves
        # matched rows (prior.z + ema.z + optimizer rows + A2 history); then the
        # moved rows' settle blocks are rebased (a teleport is not gradient
        # movement), exactly as GANTrainer lines 349-356.
        if eng.birth_death is not None:
            event = eng.birth_death.maybe_apply(eng, eng.last_output_sigma)
            if event and event.get("moves") and eng.lr_settle is not None:
                for group, tester, role in zip(eng.opt_g.param_groups, eng.lr_settle.testers[0], eng.roles[0]):
                    if role == "prior" and tester is not None:
                        tester.rebase(group["params"], eng.birth_death.moved_rows)
            self.bd_event = event
        eng.completed_steps += 1
        self.stage = 'done'


class Evaluation:
    """Seeded evaluation scope: private noise stream, candidate perturbation kept, sigma = noisy or 0."""

    def __init__(self, eng, stream, sigma):
        self.eng, self.stream, self.sigma = eng, stream, float(sigma)

    def perturb(self, latent, table):
        c = self.eng.controller
        if c is None:
            return latent
        return c.perturb_latent(latent, self.stream, self.eng.perturb_prior(table), record=False)

    @torch.no_grad()
    def generate(self, fn, latent=None, table=None):
        if latent is not None:
            latent = self.perturb(latent, table)
        y = fn(latent) if latent is not None else fn()
        return self.noise(y)

    @torch.no_grad()
    def noise(self, x):
        if self.sigma == 0:
            return x
        return x + self.sigma * torch.randn(x.shape, generator=self.stream, device=x.device, dtype=x.dtype)


# ---------------------------------------------------------------------- scalar GANTrainer re-expression (L1)
def scalar_step(eng, real, *, generator_real=None, collect_stats=False):
    """``GANTrainer._step`` through the engine phases, on (G, D, tables[0]) with the trainer's streams."""
    if generator_real is not None and not callable(generator_real):
        if (not isinstance(generator_real, torch.Tensor) or generator_real.shape[1:] != real.shape[1:]
                or len(generator_real) != len(real)):
            raise ValueError('generator_real must match the real batch')
        generator_real = generator_real.detach()
    prior, G, critic, recipe = eng.tables[0], eng.G, eng._noisy_D, eng.recipe
    with eng.update(real, collect_stats=collect_stats) as u:
        real = u.real
        u.d_phase()
        with torch.no_grad():
            latent, _ = prior.sample(len(real), generator=eng.latent_generator)
            fake = u.sample(G, latent, prior)
        u.observe_pair(real, fake)
        loss_d = eng.loss.d_loss(critic(real), critic(fake))
        penalty = u.penalty(critic, real, fake)
        penalty_stats = u.penalty_stats
        u.d_step(loss_d, penalty)
        u.g_phase()
        latent, indices = prior.sample(len(real), generator=eng.latent_generator)
        fake_logits = critic(u.generate(G, latent, prior))
        real_g = generator_real() if callable(generator_real) else generator_real
        real_g = real if real_g is None else real_g.detach()
        if real_g.shape[1:] != real.shape[1:] or len(real_g) != len(real):
            raise ValueError('generator_real must match the real batch')
        real_logits = critic(real_g)
        loss_gan = eng.loss.g_loss(fake_logits, real_logits)
        prior_reg = loss_gan.new_zeros(())
        if prior.z.requires_grad:
            raw = prior.z if recipe.num_particles <= 1024 else prior.z[torch.unique(indices)]
            prior_reg = eng.prior_regularizer(raw)
        loss_g = loss_gan + recipe.prior_reg * prior_reg
        u.g_step(loss_g, loss_gan)
        loss_d_total = u.loss_d
    result = {key: value.detach() for key, value in dict(
        loss_d=loss_d_total, loss_g=loss_g, loss_gan=loss_gan, prior_regularization=prior_reg,
        penalty=penalty).items()}
    result['step'] = eng.completed_steps
    if collect_stats:
        result['penalty_stats'] = penalty_stats
    return result
