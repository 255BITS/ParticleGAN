"""Extra observations of the original host, isolated from its owned state.

Imported only inside a root-launched numerical worker. All observations use
the original snapshot seeds 77/78, with no new training or evaluation stream.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


def write_json(path, value):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode() + b'\0' + str(tuple(value.shape)).encode() + b'\0')
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def compare(a, b, path='', *, skip=(), limit=20):
    differences = []
    def visit(x, y, at):
        if at in skip or len(differences) >= limit:
            return
        if torch.is_tensor(x) or torch.is_tensor(y):
            same = (torch.is_tensor(x) and torch.is_tensor(y) and x.shape == y.shape
                    and x.dtype == y.dtype and torch.equal(
                        x.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                        y.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))
            if not same:
                differences.append(dict(path=at, reason='tensor bytes differ'))
        elif isinstance(x, dict) and isinstance(y, dict):
            if x.keys() != y.keys():
                differences.append(dict(path=at, reason='keys differ'))
            for key in x.keys() & y.keys():
                visit(x[key], y[key], f'{at}.{key}' if at else str(key))
        elif isinstance(x, (tuple, list)) and isinstance(y, (tuple, list)):
            if type(x) is not type(y) or len(x) != len(y):
                differences.append(dict(path=at, reason='sequence differs'))
            for i, (left, right) in enumerate(zip(x, y)):
                visit(left, right, f'{at}.{i}')
        elif type(x) is float and type(y) is float and math.isnan(x) and math.isnan(y):
            pass
        elif type(x) is not type(y) or x != y:
            differences.append(dict(path=at, reason='scalar differs', left=str(x), right=str(y)))
    visit(a, b, path)
    return differences


def _modules(trainer):
    return {name: getattr(trainer, name) for name in ('G', 'D', 'prior', 'ema_G', 'ema_prior', 'ema_D')}


def _runtime_metadata(trainer, external):
    modules = _modules(trainer)
    return dict(
        modes={name: [(key, child.training) for key, child in module.named_modules()]
               for name, module in modules.items()},
        versions={name: [(key, value._version) for key, value in
                         list(module.named_parameters()) + list(module.named_buffers())]
                  for name, module in modules.items()},
        gradients={name: {key: None if p.grad is None else p.grad.detach().clone()
                          for key, p in module.named_parameters()} for name, module in modules.items()},
        external_cursor=external.get_state().clone())


def _guard_checkpoint(trainer):
    """Read current served tensors without swapping parameters or lazy records.

This is an observation guard, not a restorable checkpoint. _state_dict reads
the current exposed modules directly; a real saved checkpoint uses state_dict.
Lazy DV12 materialization is restricted to a disposable list copy.
"""
    controller = trainer.controller
    records = getattr(controller, '_latent_application_records', None)
    if records is not None:
        controller._latent_application_records = list(records)
    try:
        return trainer._state_dict()
    finally:
        if records is not None:
            controller._latent_application_records = records


@contextmanager
def isolated_geometry(trainer):
    geometry = getattr(trainer.birth_death, 'latent_geometry', None)
    if geometry is None:
        yield
        return
    entries, work = geometry._entries, geometry.work
    geometry._entries, geometry.work = dict(entries), dict(work)
    try:
        yield
    finally:
        geometry._entries, geometry.work = entries, work


class Recorder:
    def __init__(self, trainer, external, draw, angle, rotation, centers, output,
                 variant, stage, observed, restore_checkpoint, inherited_verdict):
        self.trainer, self.external, self.draw = trainer, external, draw
        self.angle, self.rotation, self.centers = angle, rotation, centers
        self.output, self.variant, self.stage, self.observed = Path(output), variant, stage, observed
        self.restore_checkpoint, self.inherited_verdict = restore_checkpoint, inherited_verdict
        self.frames, self.steps, self.angles, self.events, self.hq, self.modes = [], [], [], [], [], []
        self.losses, self.preservation_checks = [], 0
        self.start, self.external_advance = 0, 0

    def begin(self, real_batch, gate_rows):
        if self.restore_checkpoint is not None:
            for completed in range(1, 1001):
                real_batch(completed); real_batch(completed)
                self.external_advance += 2
            cursor = self.external.get_state().clone()
            state = torch.load(self.restore_checkpoint, map_location='cpu', weights_only=False)
            assert state['completed_steps'] == 1000 and state['device'] == 'cuda:0'
            self.trainer.load_state_dict(state)
            assert torch.equal(cursor, self.external.get_state())
            assert not compare(state, self.trainer.state_dict()), 'restored checkpoint differs'
            gate_rows.extend(json.loads(Path(self.inherited_verdict).read_text())['periods'][:2])
            self.start = 1000
            write_json(self.output / 'RESTORE.json', dict(status='PASS', checkpoint=str(self.restore_checkpoint),
                completed_steps=1000, real_batches_advanced=2000,
                external_cursor_sha256=tensor_sha(cursor), all_saved_owned_state_exact=True))
        return self.start

    def capture(self, step):
        if not self.observed:
            return
        before = _guard_checkpoint(self.trainer)
        runtime_before = _runtime_metadata(self.trainer, self.external)
        controller = self.trainer.controller
        controller_records = getattr(controller, '_latent_application_records', None)
        geometry = getattr(self.trainer.birth_death, 'latent_geometry', None)
        geometry_objects = None if geometry is None else (geometry._entries, geometry.work)
        with isolated_geometry(self.trainer):
            pts = self.draw(4096, 77, 78)
        after = _guard_checkpoint(self.trainer)
        runtime_after = _runtime_metadata(self.trainer, self.external)
        assert not compare(before, after), ('observation changed owned state', compare(before, after))
        assert not compare(runtime_before, runtime_after), 'observation changed runtime flags/versions/gradients/cursor'
        assert getattr(controller, '_latent_application_records', None) is controller_records
        if geometry_objects is not None:
            assert geometry._entries is geometry_objects[0] and geometry.work is geometry_objects[1]
        self.preservation_checks += 1
        cloud = pts.detach().cpu().numpy().astype(np.float32, copy=True)
        self._append(cloud, step, self.angle(step), 'initial' if step == 0 else 'update', pts)
        if step in (500, 1000):
            # Actual unchanged generator at the instant the external target
            # shifts, before update501/1001. No new samples or particle motion.
            self._append(cloud.copy(), step, self.angle(step + 1), 'target_shift', pts)
        if step % 100 == 0 or self.stage == 'window':
            self.save_frames(self.output / 'dense-frames.partial.npz')

    @torch.no_grad()
    def _append(self, cloud, step, angle, event, pts):
        current_centers = self.centers @ self.rotation(angle).T
        distance, nearest = torch.cdist(pts.float(), current_centers).min(1)
        good = distance <= .09
        hq = float(good.float().mean())
        modes = int((torch.bincount(nearest[good], minlength=100) >= 10).sum())
        self.frames.append(cloud); self.steps.append(step); self.angles.append(angle)
        self.events.append(event); self.hq.append(hq); self.modes.append(modes)
        row = dict(variant=self.variant, step=step, event_kind=event,
                   added_target_rotation_deg=round(math.degrees(angle)),
                   capture_hq=hq, capture_modes=modes, capture_points=4096,
                   capture_mode_min_HQ_points=10, original_acceptance_draw=False,
                   observation_owned_state_exact=True)
        with (self.output / 'visualization-metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
        print('CAPTURE ' + json.dumps(row, sort_keys=True), flush=True)

    def note_update(self, step, result):
        assert self.trainer.completed_steps == step
        if self.stage == 'window':
            row = {key: float(value) if torch.is_tensor(value) else value for key, value in result.items()}
            self.losses.append(row)
        if step % 10 == 0 or self.stage == 'window':
            surprise = self.trainer.policy.surprise
            birth = self.trainer.birth_death
            row = dict(variant=self.variant, step=step,
                added_target_rotation_deg=round(math.degrees(self.angle(step))),
                surprise_fires=None if surprise is None else surprise.fires,
                served_source='averaged' if self.trainer._fast is not None else 'fast',
                output_sigma=self.trainer.output_sigma(),
                lr_g_prior_d=[[float(g['lr']) for g in opt.param_groups]
                              for opt in (self.trainer.opt_g, self.trainer.opt_d)],
                mean_transport=dict(birth.last.get('mean_transport', {})))
            with (self.output / 'training-trace.jsonl').open('a') as handle:
                handle.write(json.dumps(row, sort_keys=True) + '\n')
        if self.observed and (self.stage == 'window' or step % 10 == 0):
            self.capture(step)

    def save_frames(self, path):
        if not self.frames:
            return
        launch_path = self.output / 'LAUNCH.json'
        launch = json.loads(launch_path.read_text()) if launch_path.exists() else {}
        np.savez_compressed(path, frames=np.stack(self.frames), steps=np.asarray(self.steps),
            angles=np.asarray(self.angles), centers=self.centers.detach().cpu().numpy(),
            event_kind=np.asarray(self.events, dtype='U12'), capture_hq=np.asarray(self.hq),
            capture_modes=np.asarray(self.modes), capture_points=np.array(4096),
            capture_mode_min_HQ_points=np.array(10), capture_source=np.array(self.variant),
            metadata=np.array(json.dumps(dict(variant=self.variant, dtype='float32',
                package_sha256=launch.get('package_sha256'), config_sha256=launch.get('config_sha256'),
                frame_seed_latent=77, frame_seed_noise=78, update_cadence=10,
                target_shift_frames='unchanged observed cloud; zero intervening updates',
                original_acceptance_draw_count=20000, visualization_metrics_are_acceptance=False))))

    def finish_window(self):
        expected_end = 1010 if self.start else 20
        assert self.trainer.completed_steps == expected_end
        state = self.trainer.state_dict()
        torch.save(state, self.output / 'final-state.pt')
        torch.save(_runtime_metadata(self.trainer, self.external), self.output / 'runtime-state.pt')
        # Same original evaluator seeds, separate from the visualization draw.
        pts = self.draw(20000, 1637, 1636)
        d, nearest = torch.cdist(pts.float(), self.centers @ self.rotation(self.angle(expected_end)).T).min(1)
        good = d <= .09
        score = dict(hq=float(good.float().mean()),
                     modes=int((torch.bincount(nearest[good], minlength=100) >= 10).sum()))
        write_json(self.output / 'LOSSES.json', self.losses)
        write_json(self.output / 'WINDOW-COMPLETION.json', dict(status='COMPLETE', variant=self.variant,
            observed=self.observed, first_update=self.start + 1, last_update=expected_end,
            training_updates=expected_end-self.start, original_seed=1234,
            private_streams_in_saved_state=True, original_evaluation_score=score,
            external_cursor_sha256=tensor_sha(self.external.get_state()),
            preservation_checks=self.preservation_checks))
        if self.observed:
            self.save_frames(self.output / 'dense-frames.npz')

    def finish_full(self):
        assert self.start == 0 and self.trainer.completed_steps == 1500
        assert len(self.frames) == 153 and len(self.steps) == 153
        assert self.events.count('target_shift') == 2 and self.preservation_checks == 151
        self.save_frames(self.output / 'dense-frames.npz')
        verdict = json.loads((self.output / 'original-frames.npz.verdict.json').read_text())
        assert [row['period_end'] for row in verdict['periods']] == [500, 1000, 1500]
        write_json(self.output / 'CAPTURE-COMPLETION.json', dict(status='COMPLETE', variant=self.variant,
            original_quality_status=verdict['status'], original_verdict=verdict,
            observed_states=151, target_shift_events=2, frames=153,
            all_observations_owned_state_exact=True, preservation_checks=self.preservation_checks,
            original_snapshot_seeds=[77,78], original_gate_seeds=[1637,1636],
            original_acceptance_draws=20000, visualization_points=4096,
            visualization_metrics_are_acceptance=False, interpolation=False,
            training_updates=1500, external_cursor_sha256=tensor_sha(self.external.get_state())))


_recorder = None


def begin(trainer, external, real_batch, draw, angle, rotation, centers, gate_rows, settings):
    global _recorder
    _recorder = Recorder(trainer, external, draw, angle, rotation, centers, **settings)
    return _recorder.begin(real_batch, gate_rows)


def capture(step):
    _recorder.capture(step)


def note_update(step, result):
    _recorder.note_update(step, result)


def finish_window():
    _recorder.finish_window()


def finish_full():
    _recorder.finish_full()
