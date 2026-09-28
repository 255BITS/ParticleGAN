#!/usr/bin/env python
"""Test-only: run a custom22 host loop with the ORIGINAL host learner ("engine disabled") or the original host.

  custom22_host_policy.py --which copy|original --task TASK --steps N --out FILE

``copy``: the re-expressed loop in harness/custom22.py driven by ``HostPolicy`` (the host's own Adam,
legacy GAN loss, legacy b_cap/GradRegularizer penalty, no candidate controller, no output noise, host
default auxiliary coefficients) on the verbatim host copies in harness/hosts/custom.
``original``: the untouched host function from the PR #155 checkout (read-only import, no bytecode
written) with ``schedule_optimizer`` patched to a recorder.

Both record, right before every optimizer ``step()``, a sha256 of that optimizer's group LRs, parameters
and gradients, plus the observation curve (``observation.recording``) and the final metrics. Equality
proves the copies are verbatim and the loop re-expression keeps the host's data order, RNG consumption,
models, auxiliary terms and scorers (harness/tests/test_components_parity.py::HostCopyTest). Both sides
import the PR #155 ``particlegan`` so host classes are identical.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import sys

sys.dont_write_bytecode = True
HARNESS = Path(__file__).resolve().parents[1]
CUSTOM = HARNESS / 'hosts' / 'custom'
PR155 = Path('/ml2/hypergan/ParticleGAN-k3p-continuous-search')


def digest(optimizer):
    import torch
    h = hashlib.sha256()
    for group in optimizer.param_groups:
        h.update(repr(float(group['lr'])).encode())
        for p in group['params']:
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
            h.update(b'none' if p.grad is None else p.grad.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class HostPolicy:
    """The original host learner behind the engine interface (no controller, no noise, no EMA)."""

    legacy = True

    def __init__(self):
        self.completed_steps = 0
        self.records = []
        self.penalty_calls = 0

    def build(self, *, generator=None, critic, tables=(), encoder=None, legacy, **_ignored):
        self.D, self.tables = critic, list(tables)
        self.G_side = [m for m in (generator, encoder) if m is not None]
        self.opt_g, self.opt_d = legacy['optimizers']()
        self.loss = legacy['loss']()
        self.legacy_penalty = legacy['penalty']()
        self.ema_of = {}
        return self

    def add_prior_reg(self, loss, z):
        return loss

    def record(self, optimizer, tag):
        self.records.append([self.completed_steps + 1, tag, digest(optimizer)])

    @contextmanager
    def update(self, real, *, collect_stats=False):
        u = HostUpdate(self)
        try:
            yield u
        finally:
            u.restore()
        self.completed_steps += 1

    @contextmanager
    def evaluate(self, step, *, noisy=True):
        import torch
        with torch.random.fork_rng(devices=[]):   # the original Recorder: fork only, no reseed
            yield HostEvaluation()


class HostEvaluation:
    def perturb(self, latent, table):
        return latent

    def generate(self, fn, latent=None, table=None):
        import torch
        with torch.no_grad():
            return fn(latent) if latent is not None else fn()

    def noise(self, x):
        return x


class HostUpdate:
    def __init__(self, policy):
        self.p = policy
        self.flags = None
        self.penalty_stats = None

    def d_phase(self):
        pass

    def perturb(self, latent, table):
        return latent

    def noise(self, x):
        return x

    def sample(self, fn, latent=None, table=None):
        import torch
        with torch.no_grad():
            return fn(latent) if latent is not None else fn()

    def generate(self, fn, latent=None, table=None):
        return fn(latent) if latent is not None else fn()

    def observe_pair(self, real, fake):
        pass

    def penalty(self, view, x_real, x_fake, *condition):
        self.p.penalty_calls += 1
        return self.p.legacy_penalty(view, x_real, x_fake, self.p.completed_steps + 1)

    def d_step(self, loss_d_adv, penalty):
        opt = self.p.opt_d
        opt.zero_grad(set_to_none=True)
        (loss_d_adv if penalty is None else loss_d_adv + penalty).backward()
        self.p.record(opt, 'd')
        opt.step()

    def g_phase(self):
        self.flags = [q.requires_grad for q in self.p.D.parameters()]
        self.p.D.requires_grad_(False)

    def g_step(self, loss_g, loss_gan):
        opt = self.p.opt_g
        opt.zero_grad(set_to_none=True)
        loss_g.backward()
        self.p.record(opt, 'g')
        opt.step()

    def restore(self):
        if self.flags is not None:
            for q, flag in zip(self.p.D.parameters(), self.flags):
                q.requires_grad_(flag)


def host_default_aux(task, hosts):
    """The hosts' own default coefficients (the original functions called with their defaults)."""
    if task == 'two_pole':
        return dict(particle_l2=hosts.two_pole.LOCKED_SHARED.particle_l2)
    if task in ('trajectory', 'residual_student'):
        P = hosts.residual_student.PROTOCOL if task == 'residual_student' else hosts.trajectory.PROTOCOL
        return dict(particle_l2=P['particle_l2'], vicreg_weight=P['vicreg_weight'], cover_weight=P['cover_weight'])
    if task == 'ae_gan_hold':
        cfg = hosts.ae_gan_hold.HoldConfig(name='x')
        return dict(particle_l2=cfg.particle_l2, cover_weight=cfg.cover_weight, fm_weight=cfg.fm_weight)
    if task == 'cover_leftover':
        F = hosts.cover_leftover.FORMULATION
        return dict(particle_l2=F['particle_l2'], vicreg_weight=F['vicreg_weight'], cover_weight=F['cover_weight'])
    if task == 'unused_token_hold':
        r = hosts.unused_token_hold.UnusedHoldRecipe()
        return dict(particle_l2=r.particle_l2, cover_weight=r.cover_weight, fm_weight=r.fm_weight)
    if task == 'mid_scale_identity':
        return dict(cover_weight=hosts.mid_scale_identity.FORMULATION['cover_weight'])
    return {}


def _inject_real_legacy_recipe():
    """ae_gan_hold's legacy learner needs the real gan_v3 / legacy recipe instead of the import-only shims
    (test-only; loaded from the PR #155 checkout under the copies' package names)."""
    import benchmarks
    import benchmarks.legacy
    for name, rel in (('benchmarks.legacy.recipe', 'benchmarks/legacy/recipe.py'),
                      ('benchmarks.gan_v3', 'benchmarks/gan_v3.py')):
        spec = importlib.util.spec_from_file_location(name, PR155 / rel)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        parent, _, leaf = name.rpartition('.')
        setattr(sys.modules[parent], leaf, module)


def run_copy(task, steps):
    sys.path[:0] = [str(CUSTOM), str(HARNESS), str(PR155)]
    import torch
    torch.set_num_threads(1)
    if task == 'ae_gan_hold':
        _inject_real_legacy_recipe()
    import custom22
    custom22.verify_sources()
    hosts = custom22.import_hosts()
    policy = HostPolicy()
    spec = json.loads(custom22.SPECS_PATH.read_text())['tasks'][task]['spec']
    obs = custom22.Observer(None, policy, task, steps, [list(t) for t in spec['thresholds']])
    aux = host_default_aux(task, hosts)
    final_fn = custom22.BINDINGS[task](policy, obs, aux, steps)
    unhost = lambda d: {('pass' if k == 'host_pass' else k): v for k, v in d.items()  # noqa: E731
                        if k not in ('ema', 'clean', 'pass', 'seconds')}
    final = unhost(obs.finish(final_fn)[0])
    curve = [unhost(row) for row in obs.rows]
    return dict(records=policy.records, curve=curve, final=final, penalty_calls=policy.penalty_calls,
                particlegan=str(Path(importlib.import_module('particlegan').__file__).resolve()))


def run_original(task, steps):
    sys.path.insert(0, str(PR155))
    import torch
    torch.set_num_threads(1)
    from benchmarks.locked_shared import observation, two_pole, trajectory
    from benchmarks.locked_shared.hosts import (ae_gan_hold, cover_leftover, mid_scale_identity, residual_student,
                                                unipolar, unused_token_hold)
    records = []
    counter = {'n': 0}
    module = dict(two_pole=two_pole, trajectory=trajectory, residual_student=residual_student, unipolar=unipolar,
                  ae_gan_hold=ae_gan_hold, cover_leftover=cover_leftover, unused_token_hold=unused_token_hold,
                  mid_scale_identity=mid_scale_identity)[task]

    def recorder(optimizer, completed):
        records.append([completed + 1, 'd' if counter['n'] % 2 == 0 else 'g', digest(optimizer)])
        counter['n'] += 1
    module.schedule_optimizer = recorder
    with observation.recording(steps) as rec:
        if task == 'two_pole':
            two_pole.TOY_STEPS = steps
            raw = two_pole.train()
        elif task == 'trajectory':
            trajectory.PROTOCOL['steps'] = steps
            raw = trajectory.train(diagnostics=True)
        elif task == 'residual_student':
            residual_student.PROTOCOL['steps'] = steps
            raw = residual_student.train()
        elif task == 'unipolar':
            raw = unipolar.run_arm('locked_rpgan', steps=steps)
        elif task == 'ae_gan_hold':
            raw = ae_gan_hold.train(ae_gan_hold.HoldConfig(name='custom22', steps=steps))
            raw.pop('cfg', None)
        elif task == 'cover_leftover':
            raw = cover_leftover.fit_cover_leftover(cover_leftover.CoverRecipe(steps=steps))
        elif task == 'unused_token_hold':
            raw = unused_token_hold.train(unused_token_hold.UnusedHoldRecipe(name='custom22', steps=steps))
        else:
            raw = mid_scale_identity.run_arm('locked', steps=steps)
    live = raw.get('live', raw)
    curve = [{k: v for k, v in row.items() if k != 'seconds'} for row in rec.curve]
    return dict(records=records, curve=curve, final=live,
                particlegan=str(Path(importlib.import_module('particlegan').__file__).resolve()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--which', choices=('copy', 'original'), required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = (run_copy if args.which == 'copy' else run_original)(args.task, args.steps)
    args.out.write_text(json.dumps(out, default=lambda v: v if not isinstance(v, float) or math.isfinite(v)
                                   else str(v)) + '\n')


if __name__ == '__main__':
    main()
