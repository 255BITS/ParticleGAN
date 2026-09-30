"""Replay cross-only updates and separate three drift sources.

This is an attribution diagnostic, not a candidate and not a gate change.
At each accepted step it records:

- own-player versus cross-player field change along that step
- clean output motion of the explicit Adam step and of the accepted step
- target mismatch: mode-hold distance to the HQ ball, or trajectory identity MSE

Warm mode-hold replays the constant-rate fork after the scheduled update-1000
state. Cold trajectory replays acquisition from scratch. Neither row is a
promotion.
"""
from contextlib import contextmanager, nullcontext
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from benchmarks.locked_shared.mode_hold import EVAL_N, SIGMA, diversity
from benchmarks.locked_shared.trajectory import identity_mse
from benchmarks.toy100.continuous_probe import run_probe
from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context
from reports.toy100.cross_competitive_scratch import CrossCompetitiveRecorder


HQ_RADIUS = 3.0 * SIGMA


def radial_outward(before, after, centers):
    delta = after - before
    radial = before - centers
    unit = radial / radial.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (delta * unit).sum(-1)


class DriftReplayRecorder(CrossCompetitiveRecorder):
    def __init__(self, **options):
        super().__init__(**options)
        self.replay_rows = []
        self._local = None
        self._before = None
        self._parts = []

    def phases(self, step, opt_d, opt_g, local):
        self._local = local
        self._before = self._forward_stats()
        before_outer = self.outer_steps
        yield from super().phases(step, opt_d, opt_g, local)
        if self.outer_steps == before_outer:
            return
        after = self._forward_stats()
        part = self._parts[-1] if self._parts and self._parts[-1]['outer_step'] == self.outer_steps else None
        self.replay_rows.append(self._row(step, self._before, after, part))

    def _evaluate(self, u, kind):
        n_d = sum(p.numel() for group in self.optimizers[0].param_groups for p in group['params'])
        d_only = torch.zeros_like(u)
        d_only[:n_d] = u[:n_d]
        g_only = torch.zeros_like(u)
        g_only[n_d:] = u[n_d:]
        q_d, _ = yield from ImplicitView.evaluate(self, d_only, kind + '_d_only')
        q_g, _ = yield from ImplicitView.evaluate(self, g_only, kind + '_g_only')
        cross_q = torch.cat((q_g[:n_d], q_d[n_d:]))
        own_q = torch.cat((q_d[:n_d], q_g[n_d:]))
        actual_u = self._set_point(u)
        if kind == 'jvp':
            self.cross_products.append(dict(outer_step=self.outer_steps + 1,
                cross_field_difference_norm=float(torch.linalg.vector_norm(cross_q - self.solve_q0)),
                own_field_difference_norm=float(torch.linalg.vector_norm(own_q - self.solve_q0))))
        elif kind == 'nonlinear_residual':
            joint_q, actual_u = yield from ImplicitView.evaluate(self, u, 'joint_residual_diagnostic')
            denominator = self.solve_scale * float(torch.linalg.vector_norm(self.solve_q0))
            cross_residual = float(torch.linalg.vector_norm(actual_u + self.solve_scale * cross_q)) / denominator
            joint_residual = float(torch.linalg.vector_norm(actual_u + self.solve_scale * joint_q)) / denominator
            self.joint_residuals.append(dict(outer_step=self.outer_steps + 1, scale=self.solve_scale,
                full_joint_relative_residual=joint_residual, cross_relative_residual=cross_residual))
            explicit = -self.solve_scale * self.solve_q0
            self._parts.append(dict(outer_step=self.outer_steps + 1, scale=self.solve_scale,
                own_field_change=float(torch.linalg.vector_norm(own_q - self.solve_q0)),
                cross_field_change=float(torch.linalg.vector_norm(cross_q - self.solve_q0)),
                joint_field_change=float(torch.linalg.vector_norm(joint_q - self.solve_q0)),
                explicit_output=self._outputs_at(explicit),
                accepted_output=self._forward_stats()))
        return cross_q, actual_u

    @torch.no_grad()
    def _outputs_at(self, u):
        saved = {p: p.detach().clone() for p in self.parameters}
        self._set_point(u)
        stats = self._forward_stats()
        for parameter, value in saved.items():
            parameter.copy_(value)
        return stats

    @torch.no_grad()
    def _forward_stats(self):
        local = self._local or {}
        means, prior = local.get('means'), local.get('prior')
        fast, slow = local.get('fast'), local.get('slow')
        generator = local.get('generator')
        if means is not None and prior is not None and generator is not None:
            clean_model = getattr(generator, 'model', generator)
            clean = clean_model(prior.z).detach()
            dist = torch.cdist(clean, means)
            nearest, which = dist.min(1)
            eval_samples = self._eval_cloud(generator, prior, means)
            eval_dist = torch.cdist(eval_samples, means)
            eval_nearest, eval_which = eval_dist.min(1)
            return dict(kind='mode_hold',
                clean_hq=float((nearest <= HQ_RADIUS).float().mean()),
                mean_nearest=float(nearest.mean()), max_nearest=float(nearest.max()),
                margin_count=int(((nearest > HQ_RADIUS - .02) & (nearest <= HQ_RADIUS)).sum()),
                outside_count=int((nearest > HQ_RADIUS).sum()),
                points=clean, centers=means[which],
                eval_hq=float((eval_nearest <= HQ_RADIUS).float().mean()),
                eval_mean_nearest=float(eval_nearest.mean()),
                eval_points=eval_samples, eval_centers=means[eval_which])
        if fast is not None and slow is not None and generator is not None and prior is not None:
            pred = generator(slow, prior.z).detach()
            return dict(kind='trajectory', identity_mse=identity_mse(pred, fast),
                per_point_mse=(pred - fast).pow(2).mean(-1))
        return None

    def _eval_cloud(self, generator, prior, means):
        policy = (self._local or {}).get('noise_policy')
        context = policy.evaluation(1200) if policy is not None else nullcontext()
        with torch.random.fork_rng(devices=[]):
            with context:
                latent, _ = prior.sample(EVAL_N, generator=torch.Generator().manual_seed(9))
                return generator(latent).detach()

    def _row(self, step, before, after, part):
        update = int(step) if before and before.get('kind') == 'trajectory' else int(step) + 1
        row = dict(update=update, outer_step=self.outer_steps)
        if before and before.get('kind') == 'mode_hold':
            outward = radial_outward(before['points'], after['points'], before['centers'])
            eval_out = radial_outward(before['eval_points'], after['eval_points'], before['eval_centers'])
            row.update(target='mode_hold_ring',
                before_clean_hq=before['clean_hq'], after_clean_hq=after['clean_hq'],
                before_mean_nearest=before['mean_nearest'], after_mean_nearest=after['mean_nearest'],
                before_max_nearest=before['max_nearest'], after_max_nearest=after['max_nearest'],
                before_margin_count=before['margin_count'], before_outside_count=before['outside_count'],
                before_eval_hq=before['eval_hq'], after_eval_hq=after['eval_hq'],
                mean_outward=float(outward.mean()),
                outward_fraction=float((outward > 0).float().mean()),
                eval_mean_outward=float(eval_out.mean()))
        elif before and before.get('kind') == 'trajectory':
            row.update(target='trajectory_fast_cloud',
                before_identity_mse=before['identity_mse'], after_identity_mse=after['identity_mse'],
                mse_change=after['identity_mse'] - before['identity_mse'])
        if part:
            explicit, accepted = part['explicit_output'], part['accepted_output']
            row.update(scale=part['scale'],
                own_field_change=part['own_field_change'],
                cross_field_change=part['cross_field_change'],
                joint_field_change=part['joint_field_change'],
                own_to_cross_field_ratio=part['own_field_change'] / max(part['cross_field_change'], 1e-30))
            if explicit and explicit.get('kind') == 'mode_hold':
                row.update(
                    explicit_mean_outward=float(radial_outward(before['points'], explicit['points'], before['centers']).mean()),
                    explicit_output_rms=float((explicit['points'] - before['points']).square().sum(-1).mean().sqrt()),
                    accepted_output_rms=float((accepted['points'] - before['points']).square().sum(-1).mean().sqrt()))
            elif explicit and explicit.get('kind') == 'trajectory':
                row.update(explicit_mse_change=explicit['identity_mse'] - before['identity_mse'],
                    accepted_mse_change=accepted['identity_mse'] - before['identity_mse'])
        return row


class ImplicitView:
    evaluate = CrossCompetitiveRecorder.__bases__[0]._evaluate


def _source():
    names = ('cross_drift_replay.py', 'cross_competitive_scratch.py', 'implicit_extra_scratch.py',
             'fixed_metric_extra_scratch.py', 'extra_adam_scratch.py')
    return {name: hashlib.sha256((ROOT / 'reports/toy100' / name).read_bytes()).hexdigest() for name in names}


def _install(recorder_box):
    import reports.toy100.implicit_extra_scratch as implicit_module
    from unittest.mock import patch

    @contextmanager
    def opened(**options):
        with patch.object(implicit_module, 'ImplicitExtraRecorder', DriftReplayRecorder):
            with implicit_module.implicit_extra(**options) as (recorder, source):
                recorder_box.append(recorder)
                yield recorder, source
    return opened


def run_warm(output: Path):
    config = json.loads((ROOT / 'configs/toy100/constraints_simple_regularization.json').read_text())
    holder, rate_cm = [], []

    def hook(state):
        cm = constant_rate_context(state)
        rate_cm.append(cm)
        cm.__enter__()
        completed, target = state['completed_steps'], state['target_steps']

        def accounting(calls, outer):
            state['declare_optimizer_accounting'](
                calls=completed + calls + (target - completed - outer), moment_updates=target)
        holder[0].accounting = accounting

    opened = _install(holder)
    with opened(start_step=1000) as (recorder, _):
        try:
            evidence = run_probe(config, mode='scheduled', steps=1200, diagnostic_every=50,
                                 dense_after=1190, dense_until=1200,
                                 checkpoint_hook_step=1000, checkpoint_hook=hook)
        finally:
            if rate_cm:
                rate_cm[0].__exit__(None, None, None)
    rows = recorder.replay_rows
    focus = [row for row in rows if row['update'] in {1001, 1100, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198}]
    probe = [point for point in evidence['diagnostic'] if point['step'] in {1193, 1194, 1195, 1196, 1197, 1198}]
    return dict(phase='warm_replay', steps=1200, rows=focus, row_count=len(rows),
                probe_window=probe, final_hq=evidence['final'], status=evidence['status'])


def run_cold(output: Path):
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy, declared_recipe
    config = json.loads((ROOT / 'configs/toy100/constraints_simple_regularization.json').read_text())
    config.update(name='cross_drift_replay', lr_floor=1., lr_anneal_start=0.)
    config.pop('network_lr_horizon_cap', None)
    config.pop('network_lr_floor', None)
    recipe, noise, _ = declared_recipe(config)
    spec = next(job['spec'] for job in plan() if job['spec']['name'] == 'trajectory')
    holder = []
    opened = _install(holder)
    with opened(task='trajectory') as (recorder, _):
        result, _ = run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
    rows = recorder.replay_rows
    picks = {1, 17, 50, 100, 200, 300, 400}
    focus = [row for row in rows if row['update'] in picks]
    changes = [row['mse_change'] for row in rows if 'mse_change' in row]
    return dict(phase='cold_acquisition_replay', final_identity_mse=result['live'].get('identity_mse'),
                rows=focus, row_count=len(rows),
                mse_decreased_steps=sum(value < 0 for value in changes),
                mse_increased_steps=sum(value > 0 for value in changes),
                mean_mse_change=sum(changes) / len(changes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('warm', 'cold', 'both'), default='both')
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    declaration = dict(scope='cross_only_drift_replay', shared_gate_eligible=False,
                       promotion=False, source=_source())
    (args.output / 'declaration.json').write_text(json.dumps(declaration, indent=2) + '\n')
    payload = dict(declaration=declaration)
    if args.phase in ('warm', 'both'):
        payload['warm'] = run_warm(args.output)
        print(json.dumps(dict(event='WARM_DONE', focus=payload['warm']['rows']), default=str), flush=True)
    if args.phase in ('cold', 'both'):
        payload['cold'] = run_cold(args.output)
        print(json.dumps(dict(event='COLD_DONE', focus=payload['cold']['rows'],
                              decreased=payload['cold']['mse_decreased_steps'],
                              increased=payload['cold']['mse_increased_steps']), default=str), flush=True)
    (args.output / 'replay.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(dict(event='DONE', output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
