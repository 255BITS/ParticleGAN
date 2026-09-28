"""custom22: the 8 frozen custom behavioral hosts of the 22-check suite, learner = candidate GANTrainer policy.

two_pole (80), trajectory (400), residual_student (400), unipolar (400), ae_gan_hold (250),
cover_leftover (800), unused_token_hold (200), mid_scale_identity (800). Design and every decision:
reports/custom22-design.md. Host sources are byte-identical copies under hosts/custom/benchmarks (3-line
provenance header, sha256-checked here); only each host's training loop is re-expressed below, line by
line (``# Lnnn`` = line in the ORIGINAL host file), with the host's learner lines (Adam, legacy GAN loss /
b_cap penalty, host LR schedules, schedule_optimizer, host EMA) replaced by the engine phases of
components.py. Data, models, conditioning, auxiliary losses, budgets, observation schedule, scorers and
frozen thresholds stay the host's.

Scoring of record is NOISY: output noise (the candidate's sigma) is part of the sampling law and is applied
at the frozen legacy_noise_adapters sites; the clean score is kept under ``clean``. Parameter-scored hosts
(two_pole, unipolar, cover_leftover, unused_token_hold, mid_scale_identity) have noisy == clean.
Verdict = verbatim protocol.test_verdict (24 observations at ceil(i*budget/24), passing suffix >= 5 and
every final live cell). CPU, 1 thread, deterministic (frozen baseline protocol).
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import components

HARNESS = Path(__file__).resolve().parent
CUSTOM = HARNESS / 'hosts' / 'custom'
SPECS_PATH = HARNESS / 'tasks' / 'custom22_specs.json'
PARITY_DIR = HARNESS.parent / 'runs' / '_parity'
CUSTOM_TASKS = ('two_pole', 'trajectory', 'residual_student', 'unipolar', 'ae_gan_hold', 'cover_leftover',
                'unused_token_hold', 'mid_scale_identity')
# Recipe fields forced to host values (like screen.py's host resources). Learner policy is the candidate's.
RESOURCES = {
    'two_pole': dict(num_particles=12, z_dim=1, batch_size=12, initialization=None),   # pinned HostCritic weights
    'trajectory': dict(num_particles=12, z_dim=4, batch_size=12),
    'residual_student': dict(num_particles=12, z_dim=4, batch_size=12),
    'unipolar': {},
    'ae_gan_hold': dict(num_particles=12, z_dim=2, batch_size=64, prior_kind='mog', sigma_rel=.025,
                        encoder_mode='ae'),
    'cover_leftover': dict(num_particles=12, z_dim=4, batch_size=32),
    'unused_token_hold': {},
    'mid_scale_identity': {},
}
BLOCKERS = {   # reports/custom22-design.md §8: open policy extensions applied (recommended option) per task
    'ae_gan_hold': ['B1 MoG latent kernel view z=means() (kept, labelled: perturbations clip to half of each '
                    "sample's offset from its centre, so the kernel is effectively inactive on this host; see "
                    'diag.controller.latent_applications.clipped_fraction)',
                    'B2 extended scope (encoder_mode=ae, prior_kind=mog; MoG table built by the candidate recipe '
                    '(initialization + component-width calibration), not the host make_recipe(cfg) draw)',
                    'B4 learnable sigma also trained by reconstruction/cover'],
    'unipolar': ['B3 one role-union KA2 call per update'],
    'mid_scale_identity': ['B3 one role-union KA2 call per update', 'B4 learnable sigma also trained by cover'],
    'cover_leftover': ['B5 one latent bandwidth on the union of both tables'],
    'trajectory': ['B4 learnable sigma also trained by cover'],
    'residual_student': ['B4 learnable sigma also trained by cover/residual'],
}
_HOSTS = None


# --------------------------------------------------------------------------- sources
def verify_sources():
    """Every copy is byte-identical below its 3-line header; shims and frozen specs are the recorded ones."""
    manifest = json.loads((CUSTOM / 'MANIFEST.json').read_text())
    lines = manifest['header_lines']
    out = {}
    for rel, info in manifest['files'].items():
        raw = (CUSTOM / rel).read_bytes()
        body = raw.split(b'\n', lines)[lines]
        got = hashlib.sha256(body).hexdigest()
        if got != info['sha256'] or rel.encode() not in raw.split(b'\n', 1)[0]:
            raise RuntimeError(f'custom22 host copy differs from the frozen source: {rel}')
        out[rel] = got
    for rel, info in manifest['shims'].items():
        if hashlib.sha256((CUSTOM / rel).read_bytes()).hexdigest() != info['shim_sha256']:
            raise RuntimeError(f'custom22 import shim changed: {rel}')
    specs = json.loads(SPECS_PATH.read_text())
    if specs['source_sha256'] != manifest['referenced']['benchmarks/transfer_suite/plans/default_comparison.json']:
        raise RuntimeError('custom22 specs do not come from the recorded default_comparison.json')
    return dict(source_commit=manifest['source_commit'], files=out, specs_sha256=specs['source_sha256'],
                manifest_sha256=hashlib.sha256((CUSTOM / 'MANIFEST.json').read_bytes()).hexdigest())


def import_hosts():
    global _HOSTS
    if _HOSTS is not None:
        return _HOSTS
    if str(CUSTOM) not in sys.path:
        sys.path.insert(0, str(CUSTOM))
    import importlib
    import benchmarks
    if Path(benchmarks.__file__).resolve().parent != (CUSTOM / 'benchmarks').resolve():
        raise RuntimeError(f'imported the wrong benchmarks package: {benchmarks.__file__}')
    names = dict(two_pole='benchmarks.locked_shared.two_pole', trajectory='benchmarks.locked_shared.trajectory',
                 residual_student='benchmarks.locked_shared.hosts.residual_student',
                 unipolar='benchmarks.locked_shared.hosts.unipolar',
                 ae_gan_hold='benchmarks.locked_shared.hosts.ae_gan_hold',
                 cover_leftover='benchmarks.locked_shared.hosts.cover_leftover',
                 unused_token_hold='benchmarks.locked_shared.hosts.unused_token_hold',
                 mid_scale_identity='benchmarks.locked_shared.hosts.mid_scale_identity',
                 observation='benchmarks.locked_shared.observation')
    hosts = {k: importlib.import_module(v) for k, v in names.items()}
    hosts['verdict'] = _load_verdict()
    _HOSTS = type('Hosts', (), hosts)
    return _HOSTS


def _load_verdict():
    import importlib.util
    spec = importlib.util.spec_from_file_location('lrfree_custom22_frozen_verdict', CUSTOM / 'frozen_verdict.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_aux(task, recipe):
    """Auxiliary coefficients exactly as the frozen compare route resolved them (compare_defaults.candidate +
    baseline.Candidate + host constants): particle_l2 0, VICReg spread weight = recipe.prior_reg, cover 1.5,
    feature matching 0; residual/hold/reconstruction/adversarial weights are host constants (1.0)."""
    return dict(particle_l2=0.0, vicreg_weight=float(recipe.prior_reg), cover_weight=1.5, fm_weight=0.0)


# --------------------------------------------------------------------------- observation / logging
def _round(value, digits=4):
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return float(f'{value:.{digits}g}') if abs(value) < 1e-3 and value else round(value, digits)
    return value


class Observer:
    """24 observations at ceil(i*budget/24): noisy (record) + clean, live + EMA; rates every update."""

    def __init__(self, ctx, eng, task, budget, thresholds):
        self.ctx, self.eng, self.task, self.budget = ctx, eng, task, budget
        self.expected = sorted({math.ceil(i * budget / 24) for i in range(1, 25)})
        self.steps = set(self.expected)
        self.thresholds = thresholds
        self.keys = list(dict.fromkeys(name for name, _, _ in thresholds))
        self.rows, self.alt_rows = [], []
        self.started = time.monotonic()
        self.legacy = getattr(eng, 'legacy', False)
        self.initial = None

    def collect(self, step):
        return step in self.steps

    def passes(self, point):
        cells = _HOSTS.verdict.score_metrics(point, self.thresholds)
        return all(c['status'] == 'PASS' for c in cells)

    def scores(self, fn, step):
        """{noisy, clean} x {live, ema}, each in its own seeded evaluation scope (design §2.3)."""
        eng = self.eng
        out = {}
        for kind, noisy in (('noisy', True), ('clean', False)):
            for which, ema in (('live', False), ('ema', True)):
                if self.legacy and (ema or not noisy):
                    continue
                with eng.evaluate(step, noisy=noisy) as e:
                    out[(kind, which)] = fn(e, ema)
        if self.legacy:
            live = out[('noisy', 'live')]
            out.update({('clean', 'live'): live, ('noisy', 'ema'): {}, ('clean', 'ema'): {}})
        return out

    def checkpoint(self, step, measure):
        ctx, eng = self.ctx, self.eng
        if ctx is not None and not self.legacy:
            ctx.rate_row(eng, step)
        if step not in self.steps:
            return
        s = {k: _host_pass(v) for k, v in self.scores(measure, step).items()}
        point = dict(step=step, **s[('noisy', 'live')], ema=s[('noisy', 'ema')])
        alt = dict(step=step, **s[('clean', 'live')], ema=s[('clean', 'ema')])
        ok, alt_ok = self.passes(point), self.passes(alt)
        point['pass'] = ok
        alt['pass'] = alt_ok
        point['clean'] = {k: v for k, v in alt.items() if k != 'step'}
        point['seconds'] = round(time.monotonic() - self.started, 3)
        self.alt_rows.append(alt)
        if ctx is not None and not self.legacy:
            point['lr'] = eng.current_lrs()
            diag = ctx.diagnostics(eng)
            stats = getattr(eng, 'last_penalty_stats', None)
            if stats:
                diag['penalty_stats'] = components_jsonable(ctx.torch, stats)
            diag['ka2_calls'] = eng.penalty.optimizer.record.calls
            diag['penalty_calls'] = eng.penalty_calls
            diag['a2'] = {f'table{i}': d.state_dict() for i, (d, _) in enumerate(eng.extra_damping, 1)}
            if eng.opt_g.latent_damping is not None:
                diag['a2']['table0'] = eng.opt_g.latent_damping.state_dict()
            direct = getattr(eng.opt_g, 'direct_response', None)
            if direct is not None:   # two_pole: package DirectParticleResponse (last applied LR gain)
                diag['direct'] = dict(last_gain=direct.last_gain, betas=list(direct.betas), gain=direct.gain)
            point['diag'] = diag
        self.rows.append(point)
        if ctx is not None:
            ctx.metrics.write(json.dumps(point, default=str) + '\n')
            print(json.dumps(self.compact(point, alt_ok)), flush=True)

    def compact(self, point, alt_ok):
        line = dict(step=point['step'], ok=int(point['pass']))
        line.update({k: _round(point.get(k)) for k in self.keys})
        if alt_ok != point['pass']:
            line['clean_ok'] = int(alt_ok)
        ema = point.get('ema') or {}
        if ema:
            line['ema'] = {k: _round(ema.get(k)) for k in self.keys}
        if 'lr' in point:
            line['lr'] = [float(f'{v:.3g}') for row in point['lr'] for v in row]
        sched = (point.get('diag') or {}).get('sched')
        if isinstance(sched, dict):
            line['sigma'] = _round(sched.get('output_sigma'))
            for key in ('pe', 'm', 'gt'):
                if key in sched:
                    line[key] = _round(sched[key], 3)
        return line

    def finish(self, final_fn):
        """The host's post-loop block: the same seeded evaluation as step = budget."""
        s = {k: _host_pass(v) for k, v in self.scores(final_fn, self.budget).items()}
        return s[('noisy', 'live')], s[('clean', 'live')], s[('noisy', 'ema')], s[('clean', 'ema')]


def _host_pass(metrics):
    """A host scorer's own legacy 'pass' flag (cover_leftover, mid_scale_identity) is kept as 'host_pass';
    'pass' in a harness row is the frozen-threshold pass."""
    if isinstance(metrics, dict) and 'pass' in metrics:
        metrics = dict(metrics)
        metrics['host_pass'] = metrics.pop('pass')
    return metrics


def components_jsonable(torch, value):
    import screen
    return screen.jsonable(torch, value)


# --------------------------------------------------------------------------- bindings (one per host)
def run_two_pole(eng, obs, aux, steps):
    """benchmarks/locked_shared/two_pole.py::train (pairing='live').

    The particles ARE the samples (no latent, no G): direct sample particles, bound through the package's
    direct-particle API (components.Engine._direct_optimizers: make_generator_optimizer(direct_particles=),
    prior-owned LR, direct_particle_betas + gain), as the frozen K3P route's response hook treated this opt_p
    group (reports/custom22-design.md "two_pole investigation"). Not a latent table: no A2, no dv12 latent
    kernel, no GANTrainer prior regularizer (the host has no VICReg site)."""
    import torch
    from torch import nn
    tp = _HOSTS.two_pole
    torch.manual_seed(tp.TOY_SEED)                                                   # L99
    particle_l2 = aux['particle_l2']                                                 # L100
    base_critic = tp.HostCritic()                                                    # L101
    critic = base_critic                                                             # L103 (input noise: engine)
    particles = nn.Parameter(torch.zeros(tp.LOCKED_SHARED.n_particles, 1))           # L107
    legacy = dict(optimizers=lambda: (torch.optim.Adam([particles], lr=tp.TOY_LR, betas=tp.TOY_BETAS),
                                      torch.optim.Adam(critic.parameters(), lr=tp.TOY_LR, betas=tp.TOY_BETAS)),
                  loss=lambda: tp.make_gan_loss(), penalty=lambda: _call_penalty(tp.make_b_cap()))
    eng.build(generator=None, critic=critic, direct_particles=[particles], legacy=legacy)  # L110-121 -> engine
    gan = eng.loss
    real = tp.real_batch(tp.LOCKED_SHARED.n_particles)                               # L122

    def measure(e, ema):                                                             # L147-148
        z = eng.ema_of[id(particles)] if ema else particles
        return {"mean_abs": float(z.detach().abs().mean()), "grad_med": tp._grad_median(base_critic, real, z)}

    for step in range(1, steps + 1):                                                 # L124
        with eng.update(real, collect_stats=obs.collect(step)) as u:                 # L125-126 set_step -> P0-P4
            u.d_phase()
            fake = u.noise(particles.detach())                                       # L127-130
            u.observe_pair(real, fake)
            d_loss = gan.d_loss(critic(real), critic(fake))                          # L131
            pen = u.penalty(critic, real, fake)                                      # L132
            u.d_step(d_loss, pen)                                                    # L132-134
            u.g_phase()
            d_real = critic(real).detach()                                           # L137
            generated = u.noise(particles)                                           # L138-140
            paired = critic(generated)                                               # L141
            g_loss = gan.g_loss(paired, d_real)                                      # L142
            loss_gan = g_loss
            g_loss = g_loss + particle_l2 * particles.square().mean()                # L143
            u.g_step(g_loss, loss_gan)                                               # L144-146
        obs.checkpoint(step, measure)                                                # L147

    def final(e, ema):                                                               # L150-160
        z = eng.ema_of[id(particles)] if ema else particles
        with torch.no_grad():
            mean_abs = float(z.abs().mean())
            nearest = tp._nearest(z)
        grad_med = tp._grad_median(base_critic, real, z)
        return {"mean_abs": mean_abs, "grad_med": grad_med, "nearest": nearest,
                "cover_score": tp.LOCKED_SHARED.cover_weight * (1.0 - min(nearest, 1.0))}
    return final


def _trajectory_like(eng, obs, aux, steps, residual):
    import torch
    from particlegan import ParticlePrior, ParticleRegularizer
    tr = _HOSTS.trajectory
    rs = _HOSTS.residual_student
    P = rs.PROTOCOL if residual else tr.PROTOCOL
    torch.set_num_threads(1)                                                         # tr L140 / rs L179
    torch.manual_seed(P["seed"])                                                     # tr L141 / rs L180
    slow, fast = tr.trajectories()                                                   # tr L142 / rs L181
    index = tr.pairing_index("shared", slow)                                         # tr L143 / rs L182
    paired = fast[index]                                                             # tr L144 / rs L183
    mask = rs.both_land_mask(slow, fast, index) if residual else None                # rs L184
    hidden = P["critic_hidden"]                                                      # tr L145 / rs L185
    if residual:
        model = rs.ResidualHead(slow.shape[1], P["z_dim"], hidden)                   # rs L186
        critic = rs._Critic(slow.shape[1] + fast.shape[1], hidden)                   # rs L187
        view = rs._FastView(critic)                                                  # rs L192
    else:
        model = tr._Generator(slow.shape[1], P["z_dim"], fast.shape[1], hidden)      # tr L146
        critic = tr._Critic(slow.shape[1] + fast.shape[1], hidden)                   # tr L147
        view = tr._FastView(critic)                                                  # tr L152
    prior = ParticlePrior(P["n_particles"], P["z_dim"], init_std=0.1,
                          generator=torch.Generator().manual_seed(P["seed"]))        # tr L153-156 / rs L193-196
    spread = ParticleRegularizer(weight=aux['vicreg_weight'])                        # tr L159 / rs L203
    betas = (P["beta1"], P["beta2"])
    if residual:
        legacy_loss = lambda: rs.GANLoss(P["loss_type"], P["gan_mode"])  # noqa: E731    rs L197
        legacy_pen = lambda: _call_penalty(rs.GradRegularizer(                         # rs L198-202
            P["reg_arm"], P["reg_coeff"], kappa=P["reg_kappa"], norm=P["reg_norm"], lazy_k=P["reg_lazy"],
            target_anneal=P["target_anneal"]))
    else:
        legacy_loss = lambda: tr.make_gan_loss()  # noqa: E731                         tr L157
        legacy_pen = lambda: _call_penalty(tr.make_b_cap())  # noqa: E731              tr L158
    legacy = dict(optimizers=lambda: (torch.optim.Adam(list(model.parameters()) + list(prior.parameters()),
                                                       lr=P["lr"], betas=betas),
                                      torch.optim.Adam(critic.parameters(), lr=P["lr"], betas=betas)),
                  loss=legacy_loss, penalty=legacy_pen)
    eng.build(generator=model, critic=critic, tables=[prior], legacy=legacy)        # tr L157-166 / rs L197-210
    gan = eng.loss
    both = int(mask.sum()) if residual else 0                                        # rs L237

    def measure(e, ema):                                                             # tr L202-206 / rs L275-280
        m, t = (eng.ema_of[id(model)], eng.ema_of[id(prior)]) if ema else (model, prior)
        with torch.no_grad():
            pred = e.generate(lambda z: m(slow, z), t.z, t)
        if residual:
            return {"identity_mse": tr.identity_mse(pred, fast), **rs.landing_stats(pred, fast)}
        return {"identity_mse": tr.identity_mse(pred.detach(), fast)}

    for step in range(1, steps + 1):                                                 # tr L170 / rs L238
        with eng.update(paired, collect_stats=obs.collect(step)) as u:
            u.d_phase()
            fake = u.sample(lambda z: model(slow, z), prior.z, prior)                # tr L173-175 / rs L241-243
            u.observe_pair(paired, fake)
            d_loss = gan.d_loss(critic(slow, paired), critic(slow, fake.detach()))   # tr L177 / rs L245
            view.slow = slow.detach()                                                # tr L178 / rs L246
            pen = u.penalty(view, paired, fake.detach())                             # tr L179 / rs L247
            u.d_step(d_loss, pen)                                                    # tr L179-182 / rs L247-250
            u.g_phase()                                                              # tr L184-185 / rs L252-253
            fake = u.generate(lambda z: model(slow, z), prior.z, prior)              # tr L188 / rs L256
            g_loss = gan.g_loss(critic(slow, fake), critic(slow, paired).detach())   # tr L189 / rs L257
            loss_gan = g_loss
            g_loss = g_loss + aux['cover_weight'] * tr._cover(fake, fast)            # tr L193 / rs L261
            g_loss = g_loss + aux['particle_l2'] * prior.z.square().mean()           # tr L194 / rs L262
            g_loss = g_loss + spread(prior.z)                                        # tr L195 / rs L263
            if residual:
                if both:                                                             # rs L264-268
                    residual_mse = (fake[mask] - fast[mask]).pow(2).mean()
                else:
                    residual_mse = fake.new_zeros(())
                g_loss = g_loss + rs.RESIDUAL_WEIGHT * residual_mse
            u.g_step(g_loss, loss_gan)                                               # tr L196-198 / rs L269-271
        obs.checkpoint(step, measure)                                                # tr L206 / rs L280

    def final(e, ema):                                                               # tr L208-235 / rs L300-334
        m, t = (eng.ema_of[id(model)], eng.ema_of[id(prior)]) if ema else (model, prior)
        with torch.no_grad():
            pred = e.generate(lambda z: m(slow, z), t.z, t)
            mse = tr.identity_mse(pred, fast)
            paired_mse = tr.identity_mse(pred, paired)
        if residual:
            stats = rs.landing_stats(pred, fast)
            return {"identity_mse": mse, "paired_target_mse": paired_mse, "success_rate": stats["success_rate"],
                    "wrong_pad_rate": stats["wrong_pad_rate"], "endpoint_l2": stats["endpoint_l2"],
                    "both_land_rows": both, "residual_weight": rs.RESIDUAL_WEIGHT}
        result = {"identity_mse": mse, "paired_target_mse": paired_mse}
        with torch.no_grad():                                                        # diagnostics=True block
            distances = torch.cdist(pred, fast)
            result["set_cover"] = float(tr._cover(pred, fast))
            result["own_nearest_fraction"] = float((distances.argmin(1) == torch.arange(len(fast))).float().mean())
            result["particle_mean_square"] = float(t.z.square().mean())
            result["particle_std_mean"] = float(t.z.std(0).mean())
        norms = []
        for batch in (paired, pred):
            point = batch.detach().requires_grad_(True)
            grad = torch.autograd.grad(view(point).sum(), point)[0]
            norms.append(grad.norm(dim=1).detach())
        norms = torch.cat(norms)
        result["critic_gradient_median"] = float(norms.median())
        result["critic_gradient_max"] = float(norms.max())
        return result
    return final


def run_trajectory(eng, obs, aux, steps):
    """benchmarks/locked_shared/trajectory.py::train (pairing='shared', diagnostics=True)."""
    return _trajectory_like(eng, obs, aux, steps, residual=False)


def run_residual_student(eng, obs, aux, steps):
    """benchmarks/locked_shared/hosts/residual_student.py::train (pairing='shared')."""
    return _trajectory_like(eng, obs, aux, steps, residual=True)


def _role_union(critic, scales, reals, n_rows, weights):
    """Role-union KA2 view for a scale-conditioned critic (design §5.5): valid only for uniform role
    weights 1/R and equal role sizes, where one union call equals the host's sum_r w_r * cap_r."""
    import torch
    if len(set(weights)) != 1 or abs(weights[0] * len(scales) - 1.0) > 1e-12:
        raise AssertionError('role-union penalty needs uniform role weights 1/R')
    if any(reals[s].shape[0] != n_rows for s in scales):
        raise AssertionError('role-union penalty needs equal role sizes')
    return components.RoleView(critic, scales, n_rows), torch.cat([reals[s] for s in scales])


def run_unipolar(eng, obs, aux, steps):
    """benchmarks/locked_shared/hosts/unipolar.py::run_arm('locked_rpgan') -> _fit_rpgan."""
    import torch
    up = _HOSTS.unipolar
    recipe = up.UnipolarRecipe(arm="locked_rpgan", steps=int(steps), seed=0)         # L358-371
    torch.manual_seed(recipe.seed)                                                   # L372
    student = up.FreeOriginResidual(up.DIM)                                          # L373
    target = recipe.polarity * up.PLUS                                               # L374
    teacher = up._batch(up.PLUS if recipe.polarity > 0 else -up.PLUS)                # L379
    critic = up.ScaleCritic(up.DIM, teacher, hidden=up.CRITIC_HIDDEN)                # L380
    reg_holder = {}

    def legacy_pen():
        reg_holder['reg'] = up.GradientPenalty(arm=recipe.reg_arm, coeff=recipe.reg_coeff, kappa=recipe.reg_kappa,
                                               norm=recipe.reg_norm, lazy_k=recipe.reg_lazy,
                                               target_anneal=recipe.target_anneal)   # L256-263
        return None
    legacy = dict(optimizers=lambda: (torch.optim.Adam(list(student.parameters()), lr=up.LR, betas=up.BETAS),
                                      torch.optim.Adam(critic.parameters(), lr=up.LR, betas=up.BETAS)),
                  loss=lambda: up.GANLoss(loss_type=recipe.loss_type, mode=recipe.gan_mode), penalty=legacy_pen)
    eng.build(generator=student, critic=critic, legacy=legacy)                       # L255-274 -> engine
    gan = eng.loss
    real = {0.0: up._batch(torch.zeros_like(target)), 1.0: up._batch(target)}        # L275-278
    weight = 0.5                                                                     # L299/L314 host role weight
    view, real_all = _role_union(critic, up.SCALES, real, up.N_ROWS, [weight] * len(up.SCALES))

    def measure(e, ema):                                                             # L319
        return up.score_residual(eng.ema_of[id(student)] if ema else student)

    for step in range(steps):                                                        # L280
        with eng.update(real_all, collect_stats=obs.collect(step + 1)) as u:         # L281-285 (LR/flags: engine)
            u.d_phase()
            d_loss = student.odd.new_zeros(())                                       # L287
            if eng.legacy:                                                           # verbatim host D loss
                for scale in up.SCALES:
                    fake = u.noise(student.delta(scale).unsqueeze(0).expand(up.N_ROWS, -1).detach())
                    cap, _stats = reg_holder['reg'].penalty(lambda z, scale=scale: critic.score(z, scale),
                                                            real[scale] / critic.input_scale,
                                                            fake / critic.input_scale, step=step + 1)
                    d_term = gan.d_loss(critic(real[scale], scale), critic(fake, scale))
                    d_loss = d_loss + 0.5 * (d_term + cap)
                u.d_step(d_loss, None)
            else:
                fakes = {}
                for scale in up.SCALES:                                              # L288
                    fakes[scale] = u.noise(student.delta(scale).unsqueeze(0).expand(up.N_ROWS, -1).detach())  # L289-291
                fake_all = torch.cat([fakes[s] for s in up.SCALES])
                u.observe_pair(real_all, fake_all)
                cap = u.penalty(view, real_all / critic.input_scale, fake_all / critic.input_scale)  # L292-297 (union)
                for scale in up.SCALES:
                    d_term = gan.d_loss(critic(real[scale], scale), critic(fakes[scale], scale))  # L298
                    d_loss = d_loss + weight * d_term                                # L299 (adversarial part)
                u.d_step(d_loss, cap)                                                # L300-302
            u.g_phase()                                                              # L304
            g_loss = student.odd.new_zeros(())                                       # L306
            with torch.no_grad():
                real_scores = {scale: critic(real[scale], scale) for scale in up.SCALES}  # L307-308
            for scale in up.SCALES:                                                  # L309
                fake = u.noise(student.delta(scale).unsqueeze(0).expand(up.N_ROWS, -1))  # L310-312
                g_term = gan.g_loss(critic(fake, scale), real_scores[scale])         # L313
                g_loss = g_loss + 0.5 * g_term                                       # L314
            u.g_step(g_loss, g_loss)                                                 # L315-317 (no auxiliary term)
        obs.checkpoint(step + 1, measure)                                            # L319

    def final(e, ema):                                                               # run_arm L385-406
        s = eng.ema_of[id(student)] if ema else student
        row = up.score_residual(s)
        row.update(origin_norm=float(s.origin.detach().norm()), odd_norm=float(s.odd.detach().norm()),
                   even_norm=float(s.even.detach().norm()))
        return row
    return final


def run_ae_gan_hold(eng, obs, aux, steps):
    """benchmarks/locked_shared/hosts/ae_gan_hold.py::train (extended scope: B1/B2)."""
    import torch
    ae = _HOSTS.ae_gan_hold
    cfg = ae.HoldConfig(name='custom22', steps=int(steps), cover_weight=aux['cover_weight'],
                        particle_l2=aux['particle_l2'], fm_weight=aux['fm_weight'])  # baseline.run_toy L198-199
    torch.manual_seed(cfg.seed)                                                      # L161
    recipe = ae.make_recipe(cfg) if eng.legacy else eng.recipe                       # L162 -> candidate recipe
    prior = recipe.make_prior()                                                      # L163
    encoder, decoder, critic = ae.MLP(2, 4), ae.MLP(2, 2), ae.MLP(2, 1)              # L164
    legacy = dict(optimizers=lambda: recipe.make_optimizers(decoder, critic, prior, encoder=encoder),   # L169
                  loss=lambda: recipe.make_loss(),                                                      # L172
                  penalty=lambda: _pair_penalty(recipe.make_gradient_penalty(norm=cfg.reg_norm,
                                                                             target_anneal=cfg.target_anneal)))  # L173
    eng.build(generator=decoder, encoder=encoder, critic=critic, tables=[prior], legacy=legacy)
    gan = eng.loss
    stream = torch.Generator().manual_seed(0)                                        # L178
    torch.rand(3, generator=stream)                                                  # L179
    torch.randn(8, 2, generator=stream)                                              # L180
    torch.randn(8, 2, generator=stream)                                              # L181
    torch.set_rng_state(stream.get_state())                                          # L182

    def measure(e, ema):                                                             # L184-187 -> evaluate L115-129
        if ema:
            enc, dec, pri = eng.ema_of[id(encoder)], eng.ema_of[id(decoder)], eng.ema_of[id(prior)]
        else:
            enc, dec, pri = encoder, decoder, prior
        state = torch.get_rng_state()
        try:
            with torch.no_grad():
                data = ae.sample_data(1024)
                query, offset = enc(data).chunk(2, dim=1)
                encoded = recipe.encode(query, pri, offset=offset)
                recon = e.noise(dec(encoded.codes[:, 0]))                            # noise site: decoder output
                recon_mse = float((recon - data).square().mean())
                codes, _ = pri.sample(1024)
                fake = e.generate(dec, codes, pri)                                   # perturb + decoder + noise
                return {"recon_mse": recon_mse, "hold": ae._hold_distance(fake)}
        finally:
            torch.set_rng_state(state)

    opened = obs.scores(measure, 0)[('noisy', 'live')]                                # L189
    obs.initial = opened
    penalty_applied = adv_steps = 0                                                  # L191-192
    for step in range(1, cfg.steps + 1):                                             # L193
        data = ae.sample_data(cfg.batch)                                             # L196
        with eng.update(data, collect_stats=obs.collect(step)) as u:                 # L194-195 -> P0-P4
            if not cfg.adversarial_weight > 0:                                       # L197
                raise AssertionError('custom22 binds the adversarial ae_gan_hold arm')
            u.d_phase()
            codes, _ = prior.sample(cfg.batch)                                       # L198
            fake = u.sample(decoder, codes, prior).detach()                          # L199-201
            u.observe_pair(data, fake)
            d_loss = gan.d_loss(critic(data).squeeze(-1), critic(fake).squeeze(-1))  # L203
            penalty = u.penalty(critic, data, fake)                                  # L204
            penalty_applied += 1                                                     # L205-206
            u.d_step(d_loss, penalty)                                                # L207-209
            u.g_phase()                                                              # L218-219 (critic frozen)
            query, offset = encoder(data).chunk(2, dim=1)                            # L211
            encoded = recipe.encode(query, prior, offset=offset)                     # L212
            reconstructed = u.noise(decoder(encoded.codes[:, 0]))                    # L213 (+ decoder noise site)
            recon = encoded.reconstruction_loss(reconstructed[:, None], data)        # L214
            codes, _ = prior.sample(cfg.batch)                                       # L215
            generated = u.generate(decoder, codes, prior)                            # L216
            loss = cfg.reconstruction_weight * recon + cfg.particle_l2 * prior.z.square().mean()  # L220
            real_logits = critic(data).squeeze(-1).detach()                          # L222
            fake_logits = critic(generated).squeeze(-1)                              # L223
            adv = gan.g_loss(fake_logits, real_logits)                               # L224
            anchors = ae._anchors()                                                  # L225
            cover = torch.cdist(anchors, generated).min(dim=1).values.mean()         # L226
            loss = loss + cfg.adversarial_weight * adv + cfg.cover_weight * cover    # L227
            if cfg.fm_weight > 0:                                                    # L228-231
                real_feat = critic.features(data).detach().mean(0)
                fake_feat = critic.features(generated).mean(0)
                loss = loss + cfg.fm_weight * (real_feat - fake_feat).square().mean()
            adv_steps += 1                                                           # L232
            loss = eng.add_prior_reg(loss, prior.z)                                  # GANTrainer prior_reg
            u.g_step(loss, cfg.adversarial_weight * adv)                             # L233-237
        obs.checkpoint(step, measure)                                                # L238

    def final(e, ema):                                                               # L243-253
        row = measure(e, ema)
        row.update(init_recon_mse=opened["recon_mse"], penalty_applied=penalty_applied, adv_steps=adv_steps,
                   steps=cfg.steps)
        return row
    return final


def run_cover_leftover(eng, obs, aux, steps):
    """benchmarks/locked_shared/hosts/cover_leftover.py::fit_cover_leftover(CoverRecipe())."""
    import torch
    from particlegan import ParticlePrior, ParticleRegularizer
    cl = _HOSTS.cover_leftover
    recipe = cl.CoverRecipe(steps=int(steps))
    frozen = dict(particle_l2=aux['particle_l2'], vicreg_weight=aux['vicreg_weight'], cover_weight=aux['cover_weight'])
    knob = lambda name: frozen[name] if name in frozen else recipe.knob(name)  # noqa: E731  (baseline.run_toy patch)
    field = cl.LeftoverField()                                                       # L383
    torch.manual_seed(recipe.seed)                                                   # L384
    dim = field.dim                                                                  # L385
    residual = cl._Residual(dim)                                                     # L386
    prior_p = ParticlePrior(knob("n_particles"), dim, init_std=knob("particle_init_std"))   # L387
    prior_m = ParticlePrior(knob("n_particles"), dim, init_std=knob("particle_init_std"))   # L388
    critic = cl._FourierCritic(dim, n_rand=knob("critic_n_rand"), hidden=knob("critic_hidden"),
                               seed=recipe.seed)                                     # L389-394
    spread = ParticleRegularizer(target_std=knob("vicreg_std"), weight=knob("vicreg_weight"))   # L407
    lr = float(knob("lr"))
    betas = (float(knob("beta1")), float(knob("beta2")))
    legacy = dict(
        optimizers=lambda: (torch.optim.Adam([{"params": list(residual.parameters()), "lr": lr},
                                              {"params": list(prior_p.parameters()) + list(prior_m.parameters()),
                                               "lr": lr}], lr=lr, betas=betas),
                            torch.optim.Adam(critic.parameters(), lr=lr, betas=betas)),   # L415-423
        loss=lambda: cl.GANLoss(loss_type=knob("loss_type"), mode=knob("gan_mode")),      # L398
        penalty=lambda: _call_penalty(cl.GradientPenalty(arm=knob("reg_arm"), coeff=knob("reg_coeff"),
                                                         kappa=knob("reg_kappa"), norm=knob("reg_norm"), lazy_k=1,
                                                         target_anneal="none")))          # L399-406
    eng.build(generator=residual, critic=critic, tables=[prior_p, prior_m], legacy=legacy)
    gan = eng.loss
    poles_p, poles_m, neu = cl.teacher_poles(field, knob("teacher"))                 # L427
    half = max(1, int(knob("batch")) // 2)                                           # L428
    jitter = float(knob("particle_jitter"))                                          # L429
    cover_w = float(knob("cover_weight"))                                            # L430
    particle_l2 = float(knob("particle_l2"))                                         # L431

    def particle_batch(prior, perturb):                                              # L365-369
        z, _idx = prior.sample(half)
        z = perturb(z, prior)                                                        # candidate perturbation (§5.3)
        if float(jitter) > 0.0:
            z = z + float(jitter) * torch.randn_like(z)
        return z

    def fake_batch(perturb):                                                         # L449-452
        fake_p = neu + residual.delta(1.0) + particle_batch(prior_p, perturb)
        fake_m = neu + residual.delta(-1.0) + particle_batch(prior_m, perturb)
        return fake_p, fake_m

    def measure(e, ema):                                                             # L505
        return cl.score_geometry(eng.ema_of[id(residual)] if ema else residual, field, poles_p, poles_m, neu)

    for step in range(recipe.steps):                                                 # L454
        real_p = cl._sample_real_cloud(poles_p, neu, half, cloud_std=knob("cloud_std"), span_frac=knob("span_frac"),
                                       end_margin=knob("end_margin"))                # L462-467
        real_m = cl._sample_real_cloud(poles_m, neu, half, cloud_std=knob("cloud_std"), span_frac=knob("span_frac"),
                                       end_margin=knob("end_margin"))                # L468-473
        real = torch.cat([real_p, real_m], dim=0)                                    # L474
        with eng.update(real, collect_stats=obs.collect(step + 1)) as u:             # L455-461 -> P0-P4
            u.d_phase()
            fake_p, fake_m = fake_batch(u.perturb)                                   # L475
            fake = u.noise(torch.cat([fake_p, fake_m], dim=0).detach())              # L476-478
            u.observe_pair(real, fake)
            d_loss = gan.d_loss(critic(real.detach()), critic(fake))                 # L479
            cap = u.penalty(critic, real.detach(), fake)                             # L480
            u.d_step(d_loss, cap)                                                    # L481-485
            u.g_phase()
            fake_p, fake_m = fake_batch(u.perturb)                                   # L487
            fake = u.noise(torch.cat([fake_p, fake_m], dim=0))                       # L488-490
            g_loss = gan.g_loss(critic(fake), critic(real.detach()))                 # L491
            loss_gan = g_loss
            parts = torch.cat([prior_p.z, prior_m.z], dim=0)                         # L492
            g_loss = g_loss + spread(parts)                                          # L493
            if particle_l2 > 0.0:                                                    # L494-495
                g_loss = g_loss + particle_l2 * parts.pow(2).mean()
            if cover_w > 0.0:                                                        # L496-499
                cover = (neu + residual.delta(1.0) - poles_p).pow(2).mean()
                cover = cover + (neu + residual.delta(-1.0) - poles_m).pow(2).mean()
                g_loss = g_loss + cover_w * cover
            u.g_step(g_loss, loss_gan)                                               # L500-503
        obs.checkpoint(step + 1, measure)                                            # L504 EMA -> engine; L505

    def final(e, ema):                                                               # L528-555 (live before EMA copy)
        row = measure(e, ema)
        with torch.no_grad():
            tables = (eng.ema_of[id(prior_p)], eng.ema_of[id(prior_m)]) if ema else (prior_p, prior_m)
            row["particle_rms"] = float(torch.cat([t.z for t in tables], dim=0).pow(2).mean().sqrt())
        return row
    return final


def run_unused_token_hold(eng, obs, aux, steps):
    """benchmarks/locked_shared/hosts/unused_token_hold.py::train(UnusedHoldRecipe(...))."""
    import torch
    ut = _HOSTS.unused_token_hold
    recipe = ut.UnusedHoldRecipe(name='custom22', steps=int(steps), particle_l2=aux['particle_l2'],
                                 cover_weight=aux['cover_weight'], fm_weight=aux['fm_weight'])  # baseline L206-208
    torch.manual_seed(int(recipe.seed))                                              # L223
    student = ut.SharedSlotStudent()                                                 # L224
    critic = ut.SlotCritic()                                                         # L225
    legacy = dict(optimizers=lambda: (torch.optim.Adam(list(student.parameters()), lr=ut.LR, betas=ut.BETAS),
                                      torch.optim.Adam(critic.parameters(), lr=ut.LR, betas=ut.BETAS)),
                  loss=lambda: ut.GANLoss(loss_type=recipe.loss_type, mode=recipe.gan_mode),
                  penalty=lambda: _pair_penalty(ut._make_regularizer(recipe)))
    eng.build(generator=student, critic=critic, legacy=legacy)                       # L229-239 -> engine
    gan = eng.loss
    real = ut._batch(ut.CONCEPT_DIR)                                                 # L240
    pairs = ut.hold_pairs(recipe.pairing)                                            # L241

    def measure(e, ema):                                                             # L278
        return ut.score_student(eng.ema_of[id(student)] if ema else student)

    for step in range(int(recipe.steps)):                                            # L243
        with eng.update(real, collect_stats=obs.collect(step + 1)) as u:             # L244-245
            u.d_phase()
            fake = u.noise(student.embeds(1.0)[ut.CONCEPT].unsqueeze(0).expand(ut.N_ROWS, -1).detach())  # L246-248
            u.observe_pair(real, fake)
            penalty = u.penalty(critic, real, fake)                                  # L250 (host order)
            d_adv = gan.d_loss(critic(real), critic(fake))                           # L253
            u.d_step(d_adv, penalty)                                                 # L253-256
            u.g_phase()                                                              # L258
            fake_g = u.noise(student.embeds(1.0)[ut.CONCEPT].unsqueeze(0).expand(ut.N_ROWS, -1))  # L260-262
            g_loss = gan.g_loss(critic(fake_g), critic(real).detach())               # L263
            loss_gan = g_loss
            if float(recipe.fm_weight) != 0.0:                                       # L266-269
                real_feat = critic.features(real).detach().mean(0)
                fake_feat = critic.features(fake_g).mean(0)
                g_loss = g_loss + float(recipe.fm_weight) * (real_feat - fake_feat).pow(2).mean()
            loss = g_loss                                                            # L270
            if float(recipe.hold_weight) != 0.0:                                     # L271-273
                embeds = student.embeds(1.0)
                loss = loss + float(recipe.hold_weight) * ut.unused_hold_loss(embeds, student.neu, pairs)
            u.g_step(loss, loss_gan)                                                 # L274-277
        obs.checkpoint(step + 1, measure)                                            # L278

    def final(e, ema):                                                               # L283
        return measure(e, ema)
    return final


def run_mid_scale_identity(eng, obs, aux, steps):
    """benchmarks/locked_shared/hosts/mid_scale_identity.py::run_arm('locked') -> _fit, _finish."""
    import torch
    import torch.nn.functional as F
    ms = _HOSTS.mid_scale_identity
    arm = "locked"
    teacher = ms.smile_teacher()                                                     # run_arm L569
    if not ms._has_scale(ms.EVAL_SCALES, -1.0):                                      # _fit L432-433
        raise RuntimeError("training grid must include -1")
    train_arm = ms._train_arm_name(arm)                                              # L434
    torch.manual_seed(0)                                                             # L435
    student = ms.MidScaleResidual(int(teacher.concept.numel()))                      # L436
    if any(param.is_cuda for param in student.parameters()):                         # L437-438
        raise RuntimeError("mid-scale identity toy is CPU only")
    targets = {scale: teacher.train_target(train_arm, scale) for scale in ms.EVAL_SCALES}   # L439
    cloud = torch.stack([targets[scale] for scale in ms.EVAL_SCALES], dim=0)         # L440
    critic = ms.ScaleCritic(student.odd.numel(), cloud, hidden=ms.CRITIC_HIDDEN)     # L441
    F_ = ms.FORMULATION
    reg_holder = {}

    def legacy_pen():
        reg_holder['reg'] = ms.GradientPenalty(arm=F_["reg_arm"], coeff=F_["reg_coeff"], kappa=F_["reg_kappa"],
                                               norm=F_["reg_norm"], lazy_k=F_["reg_lazy"],
                                               target_anneal=F_["target_anneal"])    # L445-452
        return None
    legacy = dict(optimizers=lambda: (torch.optim.Adam(list(student.parameters()), lr=ms.LR, betas=ms.BETAS),
                                      torch.optim.Adam(critic.parameters(), lr=ms.LR, betas=ms.BETAS)),
                  loss=lambda: ms.GANLoss(loss_type=F_["loss_type"], mode=F_["gan_mode"]), penalty=legacy_pen)
    eng.build(generator=student, critic=critic, legacy=legacy)                       # L444-463 -> engine
    gan = eng.loss
    reals = {scale: ms._batch(targets[scale]) for scale in ms.EVAL_SCALES}           # L464
    cover_w = float(aux['cover_weight'])                                             # L465
    n_scales = float(len(ms.EVAL_SCALES))                                            # L467
    view, real_all = _role_union(critic, ms.EVAL_SCALES, reals, ms.N_ROWS, [1.0 / n_scales] * len(ms.EVAL_SCALES))

    def measure(e, ema):                                                             # L512-513
        return ms.score_hold(eng.ema_of[id(student)] if ema else student, scales=ms._eval_scales(arm),
                             pairing="stranger" if arm == "stranger" else "matched", teacher=teacher)

    for step in range(int(steps)):                                                   # L469
        with eng.update(real_all, collect_stats=obs.collect(step + 1)) as u:         # L470-474
            u.d_phase()
            d_loss = student.odd.new_zeros(())                                       # L476
            if eng.legacy:                                                           # verbatim host D loss
                for scale in ms.EVAL_SCALES:
                    fake = u.noise(student.state(scale).unsqueeze(0).expand(ms.N_ROWS, -1).detach())
                    cap, _stats = reg_holder['reg'].penalty(lambda z, scale=scale: critic.score(z, scale),
                                                            reals[scale] / critic.input_scale,
                                                            fake / critic.input_scale, step=step + 1)
                    d_term = gan.d_loss(critic(reals[scale], scale), critic(fake, scale))
                    d_loss = d_loss + (d_term + cap) / n_scales
                u.d_step(d_loss, None)
            else:
                fakes = {}
                for scale in ms.EVAL_SCALES:                                         # L477
                    fakes[scale] = u.noise(student.state(scale).unsqueeze(0).expand(ms.N_ROWS, -1).detach())  # L478-480
                fake_all = torch.cat([fakes[s] for s in ms.EVAL_SCALES])
                u.observe_pair(real_all, fake_all)
                cap = u.penalty(view, real_all / critic.input_scale, fake_all / critic.input_scale)  # L481-487 (union)
                for scale in ms.EVAL_SCALES:
                    d_term = gan.d_loss(critic(reals[scale], scale), critic(fakes[scale], scale))  # L488
                    d_loss = d_loss + d_term / n_scales                              # L489 (adversarial part)
                u.d_step(d_loss, cap)                                                # L490-492
            u.g_phase()                                                              # L494
            g_loss = student.odd.new_zeros(())                                       # L496
            with torch.no_grad():
                real_scores = {scale: critic(reals[scale], scale) for scale in ms.EVAL_SCALES}   # L497-498
            for scale in ms.EVAL_SCALES:                                             # L499
                fake = u.noise(student.state(scale).unsqueeze(0).expand(ms.N_ROWS, -1))   # L500-502
                g_loss = g_loss + gan.g_loss(critic(fake, scale), real_scores[scale]) / n_scales   # L503
            loss_gan = g_loss
            cover = student.odd.new_zeros(())                                        # L504
            for scale in ms.EVAL_SCALES:                                             # L505-506
                cover = cover + F.mse_loss(student.state(scale), targets[scale])
            g_loss = g_loss + cover_w * cover / n_scales                             # L507
            u.g_step(g_loss, loss_gan)                                               # L508-511
        obs.checkpoint(step + 1, measure)                                            # L512

    def final(e, ema):                                                               # _finish L547-559
        row = measure(e, ema)
        row.update(arm=arm, steps=int(steps), seed=0, cover_weight=cover_w)
        return row
    return final


def _call_penalty(regularizer):
    """Legacy (host-policy) penalty callable: ``regularizer(view, real, fake, step=step)``."""
    return lambda view, x_real, x_fake, step: regularizer(view, x_real, x_fake, step=step)


def _pair_penalty(regularizer):
    """Legacy (host-policy) penalty callable: ``regularizer.penalty(view, real, fake, step=step)[0]``."""
    return lambda view, x_real, x_fake, step: regularizer.penalty(view, x_real, x_fake, step=step)[0]


BINDINGS = dict(two_pole=run_two_pole, trajectory=run_trajectory, residual_student=run_residual_student,
                unipolar=run_unipolar, ae_gan_hold=run_ae_gan_hold, cover_leftover=run_cover_leftover,
                unused_token_hold=run_unused_token_hold, mid_scale_identity=run_mid_scale_identity)


# --------------------------------------------------------------------------- parity gate (design §3)
PARITY_SCOPE = ('per-package gate = components_parity c1 (scalar mode_hold) + c2 (sparse table) only; host-specific '
                'engine paths (initialization=None, second table + own A2, RoleView, encoder + MoG) are covered by '
                'the L3a/L3b unit tests and HostSmokeTest in harness/tests/test_components_parity.py')


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def ensure_parity(package_root, overrides, *, timeout=3600):
    """L1-CPU (1000 updates, c1+c2) for this package + overrides + engine revision, once, under a lock."""
    import lrlib
    package_sha = lrlib.package_digest(package_root)
    blob = json.dumps(overrides, sort_keys=True, default=str)
    key = f'{package_sha[:16]}-{hashlib.sha256(blob.encode()).hexdigest()[:12]}-{components.source_sha256()[:12]}'
    key += f'-{hashlib.sha256((HARNESS / "components_parity.py").read_bytes()).hexdigest()[:12]}'   # gate revision
    path = PARITY_DIR / f'{key}.json'
    lock = path.with_suffix('.lock')
    PARITY_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    while True:
        if path.exists():
            record = json.loads(path.read_text())
            if record.get('status') != 'PASS':
                raise RuntimeError(f'engine parity {record.get("status")}: {path.name} '
                                   f'{[r.get("first_mismatch") or r.get("error") for r in record.get("results", [])]}')
            return dict(id=key, file=str(path), status='PASS', steps=record['steps'], scope=PARITY_SCOPE)
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                pid = int(lock.read_text().strip() or 0)
            except (OSError, ValueError):
                pid = 0
            if pid and not _alive(pid):
                try:
                    lock.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() - started > timeout:
                raise RuntimeError(f'engine parity lock timeout ({lock})')
            time.sleep(2.0)
            continue
        with os.fdopen(fd, 'w') as handle:
            handle.write(str(os.getpid()))
        try:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            command = [sys.executable, str(HARNESS / 'components_parity.py'), '--package-root', str(package_root),
                       '--overrides', blob, '--steps', '1000', '--device', 'cpu', '--out', str(path)]
            print(json.dumps(dict(event='parity', key=key, status='running L1-CPU 1000 updates')), flush=True)
            proc = subprocess.run(command, env=env, capture_output=True, text=True)
            if not path.exists():
                raise RuntimeError(f'engine parity run failed: {proc.stderr[-800:]}')
        finally:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass


# --------------------------------------------------------------------------- entry point (screen.py)
def cpu_deterministic(torch):
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


def run(ctx, task):
    """One custom task for screen.py: returns (engine, result dict in the screen.py schema)."""
    torch, package, options = ctx.torch, ctx.package, ctx.options
    sources = verify_sources()
    import_hosts()
    specs = json.loads(SPECS_PATH.read_text())['tasks']
    entry = specs[task]
    spec = dict(entry['spec'])
    test_steps = os.environ.get('LRFREE_CUSTOM_TEST_STEPS')   # plumbing/determinism tests only
    if test_steps:
        spec['steps'] = int(test_steps)
    budget = spec['steps']
    cpu_deterministic(torch)
    parity = ensure_parity(ctx.args.package_root.resolve(), ctx.overrides)
    eng = components.Engine(package, ctx.overrides, RESOURCES[task], seed=0,
                            serial_backward=bool(options['serial_backward_argument']))
    thresholds = [list(t) for t in spec['thresholds']]
    obs = Observer(ctx, eng, task, budget, thresholds)
    aux = frozen_aux(task, eng.recipe)
    started = time.monotonic()
    final_fn = BINDINGS[task](eng, obs, aux, budget)
    if eng.completed_steps != budget or eng.penalty_calls != budget:
        raise RuntimeError(f'update/penalty count {eng.completed_steps}/{eng.penalty_calls} != budget {budget}')
    final, final_clean, ema_final, ema_final_clean = obs.finish(final_fn)
    verdict_mod = _HOSTS.verdict
    keys = [name for name, _, _ in thresholds]
    for row, fin in ((obs.rows[-1], final), (obs.alt_rows[-1], final_clean)):
        diffs = [k for k in keys if row.get(k) != fin.get(k)]
        if diffs:
            ctx.warnings.append(f'final block differs from the step-{budget} observation on {diffs}')

    def judge(rows, live):
        verdict = verdict_mod.test_verdict(spec, dict(observations=[{k: v for k, v in r.items() if k != 'clean'}
                                                                    for r in rows], live=live))
        return ('PASS' if verdict['passed'] else 'FAIL'), verdict
    status, verdict = judge(obs.rows, final)
    alt_status, alt_verdict = judge(obs.alt_rows, final_clean)
    conv, alt_conv = verdict.get('convergence', {}), alt_verdict.get('convergence', {})
    strip = lambda d: {k: v for k, v in d.items() if k not in ('pass', 'seconds')}  # noqa: E731
    noisy_equals_clean = all(strip({k: v for k, v in a.items() if k not in ('clean', 'diag', 'lr')}) ==
                             strip({k: v for k, v in b.items()}) for a, b in zip(obs.rows, obs.alt_rows))
    record = eng.penalty.optimizer.record
    result = dict(
        status=status, verdict=verdict, passing_checks=conv.get('passing_observations'),
        observations=conv.get('observations'), first_arrival=conv.get('first_pass_step'),
        final_streak=conv.get('passing_suffix'), final=final, ema_final=ema_final,
        final_lr=obs.rows[-1].get('lr'), thresholds=thresholds,
        clean_status=alt_status, clean_passing_checks=alt_conv.get('passing_observations'),
        clean_first_arrival=alt_conv.get('first_pass_step'), clean_final_streak=alt_conv.get('passing_suffix'),
        clean_final=final_clean, clean_ema_final=ema_final_clean, clean_verdict=alt_verdict,
        primary_scoring='noisy', noisy_equals_clean=noisy_equals_clean,
        eval_output_noise=bool(options['eval_output_noise']),
        eval_scope=entry['eval_scope'], budget=budget, test_steps=int(test_steps) if test_steps else None,
        pass_rule='frozen protocol.test_verdict: 24 observations at ceil(i*budget/24), passing suffix >= 5 and '
                  'every final live threshold (noisy scoring of record)',
        custom22=dict(
            engine=components.ENGINE_VERSION, engine_sha256=components.source_sha256(),
            binding_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), sources=sources,
            spec=spec, spec_reference=entry['reference'], parity=parity, resources=RESOURCES[task],
            aux=aux, blockers_applied=[b for b in BLOCKERS.get(task, []) if not b.startswith('B4')
                                       or eng.log_output_sigma is not None],
            extended_scope=task == 'ae_gan_hold',
            groups=eng.receipts['groups'], construction_global_rng_untouched=eng.receipts[
                'construction_global_rng_untouched'],
            ka2_calls=record.calls, penalty_calls=eng.penalty_calls, completed_steps=eng.completed_steps,
            initial=getattr(obs, 'initial', None), train_seconds=round(time.monotonic() - started, 2),
            device='cpu', initialization=getattr(eng.recipe, 'initialization', None)))
    return eng, result
