#!/usr/bin/env python
"""Test-only: HostCopyTest with OUTPUT NOISE ON -- regression test of the custom22 noise sites.

  custom22_host_noise.py --which copy|original --task TASK --steps N --out FILE

Same comparison as custom22_host_policy.py (engine disabled: the host's own learner, per-step optimizer
digests of LRs/params/grads, observation curve, final metrics), but both sides get the SAME duck-typed
noise policy: an isolated output-noise stream (fixed seed, fixed sigma), reseeded per evaluation the same
way on both sides. ``copy`` routes it through the binding's ``update.noise/sample/generate`` and
``evaluation.generate/noise`` sites; ``original`` passes it to the untouched PR #155 host as
``noise_policy=`` (the frozen legacy_noise_adapters route). Equality of digests, curve, final metrics and
of the sequence of TRAINING noise calls (step, input shape, input hash) proves the bindings apply output
noise at the host's frozen sites, in the host's order. The originals' extra EVAL-scope calls come from
logging-only evaluations (residual_student ``_emit``, ae_gan_hold ``_log``) inside restored scopes and are
not compared.

Test-only adjustments on the ORIGINAL side (none on the copy):
- the host LR schedule (unipolar/mid_scale_identity ``_apply_lr``, cover_leftover ``_delayed_cosine``) is
  disabled: the bindings are LR-free by construction, the schedule is not part of this comparison;
- mid_scale_identity: ``ScaleCritic.score`` without the frozen route's ``policy.input(z*s)/s`` round-trip.
  With input_std=0 that round-trip is the identity up to float rounding (input_scale is not a power of two;
  unipolar's 0.5 is, so it is exact there); the harness refuses input noise > 0, so the binding does not
  wire ``critic.noise_policy`` (reports/custom22-design.md §9).
Originally the reviewer's scratch check (noise_sites.py), promoted to a regression test.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import custom22_host_policy as hp  # noqa: E402

SIGMA = 0.05
STREAM_SEED = 1234


class NoisePolicy:
    """Minimal duck-typed legacy noise policy: constant output sigma, zero input noise."""

    output_scale = None

    def __init__(self):
        import torch
        self.torch = torch
        self.stream = torch.Generator().manual_seed(STREAM_SEED)
        self.calls = []
        self._eval = False

    def set_step(self, step):
        self.calls.append(['set', step])

    def capture_final_live(self):
        pass

    def capture_final_ema(self):
        pass

    def scale_parameters(self):
        return []

    def register_generator_base(self, module):
        pass

    def register_generator_optimizer(self, opt_g, opt_d):
        pass

    def input(self, x):
        return x

    @contextmanager
    def discriminator(self):
        yield

    def output(self, x, *, generator_step=None):
        n = self.torch.randn(x.shape, generator=self.stream)
        self.calls.append(['out', 'eval' if self._eval else 'train', list(x.shape),
                           hashlib.sha256(x.detach().contiguous().numpy().tobytes()).hexdigest()[:16]])
        return x + SIGMA * n

    @contextmanager
    def evaluation(self, step):
        torch = self.torch
        saved = self.stream.get_state()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(402 + step)
            self.stream.manual_seed(9000 + step)
            prev, self._eval = self._eval, True
            try:
                yield
            finally:
                self._eval = prev
        self.stream.set_state(saved)


POL = None


class NoisyUpdate(hp.HostUpdate):
    def noise(self, x):
        return POL.output(x, generator_step=self.flags is not None)

    def sample(self, fn, latent=None, table=None):
        import torch
        with torch.no_grad():
            return POL.output(fn(latent) if latent is not None else fn(), generator_step=False)

    def generate(self, fn, latent=None, table=None):
        return POL.output(fn(latent) if latent is not None else fn(), generator_step=True)


class NoisyEvaluation(hp.HostEvaluation):
    def generate(self, fn, latent=None, table=None):
        import torch
        with torch.no_grad():
            return POL.output(fn(latent) if latent is not None else fn())

    def noise(self, x):
        return POL.output(x)


class NoisyHostPolicy(hp.HostPolicy):
    @contextmanager
    def update(self, real, *, collect_stats=False):
        POL.set_step(self.completed_steps)
        u = NoisyUpdate(self)
        try:
            yield u
        finally:
            u.restore()
        self.completed_steps += 1

    @contextmanager
    def evaluate(self, step, *, noisy=True):
        import torch
        with torch.random.fork_rng(devices=[]), POL.evaluation(step):
            yield NoisyEvaluation()


def run_copy(task, steps):
    global POL
    sys.path[:0] = [str(hp.CUSTOM), str(hp.HARNESS), str(hp.PR155)]
    import torch
    torch.set_num_threads(1)
    if task == 'ae_gan_hold':
        hp._inject_real_legacy_recipe()
    import custom22
    custom22.verify_sources()
    hosts = custom22.import_hosts()
    POL = NoisePolicy()
    policy = NoisyHostPolicy()
    spec = json.loads(custom22.SPECS_PATH.read_text())['tasks'][task]['spec']
    obs = custom22.Observer(None, policy, task, steps, [list(t) for t in spec['thresholds']])
    final_fn = custom22.BINDINGS[task](policy, obs, hp.host_default_aux(task, hosts), steps)
    unhost = lambda d: {('pass' if k == 'host_pass' else k): v for k, v in d.items()  # noqa: E731
                        if k not in ('ema', 'clean', 'pass', 'seconds')}
    final = unhost(obs.finish(final_fn)[0])
    return dict(records=policy.records, curve=[unhost(r) for r in obs.rows], final=final,
                penalty_calls=policy.penalty_calls)


def run_original(task, steps):
    global POL
    sys.path.insert(0, str(hp.PR155))
    import torch
    torch.set_num_threads(1)
    from benchmarks.locked_shared import observation, two_pole, trajectory
    from benchmarks.locked_shared.hosts import (ae_gan_hold, cover_leftover, mid_scale_identity, residual_student,
                                                unipolar, unused_token_hold)
    POL = NoisePolicy()
    records, counter = [], {'n': 0}
    module = dict(two_pole=two_pole, trajectory=trajectory, residual_student=residual_student, unipolar=unipolar,
                  ae_gan_hold=ae_gan_hold, cover_leftover=cover_leftover, unused_token_hold=unused_token_hold,
                  mid_scale_identity=mid_scale_identity)[task]

    def recorder(optimizer, completed):
        records.append([completed + 1, 'dg'[counter['n'] % 2], hp.digest(optimizer)])
        counter['n'] += 1
    module.schedule_optimizer = recorder
    if task in ('unipolar', 'mid_scale_identity'):
        module._apply_lr = lambda *a, **k: None
    if task == 'cover_leftover':
        module._delayed_cosine = lambda *a, **k: 1.0
    if task == 'mid_scale_identity':
        def score(self, z, scale):   # frozen route minus the input_std=0 policy.input round-trip (see doc)
            label = z.new_full((z.shape[0], 1), float(scale))
            return self.net(torch.cat([z, label], dim=-1)).squeeze(-1)
        mid_scale_identity.ScaleCritic.score = score
    kw = dict(noise_policy=POL)
    with observation.recording(steps) as rec:
        if task == 'two_pole':
            two_pole.TOY_STEPS = steps
            raw = two_pole.train(**kw)
        elif task == 'trajectory':
            trajectory.PROTOCOL['steps'] = steps
            raw = trajectory.train(diagnostics=True, **kw)
        elif task == 'residual_student':
            residual_student.PROTOCOL['steps'] = steps
            raw = residual_student.train(**kw)
        elif task == 'unipolar':
            raw = unipolar.run_arm('locked_rpgan', steps=steps, **kw)
        elif task == 'ae_gan_hold':
            raw = ae_gan_hold.train(ae_gan_hold.HoldConfig(name='custom22', steps=steps), **kw)
            raw.pop('cfg', None)
        elif task == 'cover_leftover':
            raw = cover_leftover.fit_cover_leftover(cover_leftover.CoverRecipe(steps=steps), **kw)
        elif task == 'unused_token_hold':
            raw = unused_token_hold.train(unused_token_hold.UnusedHoldRecipe(name='custom22', steps=steps), **kw)
        else:
            raw = mid_scale_identity.run_arm('locked', steps=steps, **kw)
    live = raw.get('live', raw)
    return dict(records=records, curve=[{k: v for k, v in r.items() if k != 'seconds'} for r in rec.curve],
                final=live)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--which', choices=('copy', 'original'), required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = (run_copy if args.which == 'copy' else run_original)(args.task, args.steps)
    out['noise_calls'] = POL.calls
    args.out.write_text(json.dumps(out, default=lambda v: v if not isinstance(v, float) or math.isfinite(v)
                                   else str(v)) + '\n')


if __name__ == '__main__':
    main()
