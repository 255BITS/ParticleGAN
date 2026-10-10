"""Observation and checkpoint continuation support for the original CUDA host.

Imported only by explicitly launched numerical workers. Preparation does not
import this module or torch. Observation reads already computed function locals;
it does not query charts, run models, draw random numbers, or change decisions.
"""
import functools
import hashlib
import inspect
import json
from pathlib import Path
import sys

import torch

STUDY = Path('/ml2/hypergan/gan-attempts/feature-cells-generalization-20260930')
ORIGINAL = STUDY / 'validation-ra14-r2/moving/rotated100'
CHECKPOINT = ORIGINAL / 'checkpoint-001000.pt'
_trainer = None
_output = None
_pending = []
_advances = 0


def tensor_sha(tensor):
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode() + b'\0')
    h.update(str(tuple(value.shape)).encode() + b'\0')
    h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def compare(left, right, path='', *, skip=(), limit=20):
    """Exact semantic equality; only explicitly named elapsed wall time is skipped."""
    problems = []
    def visit(a, b, at):
        if at in skip or len(problems) >= limit:
            return
        if torch.is_tensor(a) or torch.is_tensor(b):
            if not (torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype
                    and a.shape == b.shape and torch.equal(a.detach().cpu(), b.detach().cpu())):
                problems.append(dict(path=at, reason='tensor differs'))
        elif isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                problems.append(dict(path=at, reason='keys differ'))
            for key in a.keys() & b.keys():
                visit(a[key], b[key], f'{at}.{key}' if at else str(key))
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                problems.append(dict(path=at, reason='sequence differs'))
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f'{at}.{index}')
        elif type(a) is not type(b) or a != b:
            problems.append(dict(path=at, reason='scalar differs', expected=str(a), actual=str(b)))
    visit(left, right, path)
    return problems


def resume(trainer, real_batch, stream, gate_rows, output):
    """Rebuild the external data cursor, then restore every saved owned RNG."""
    global _trainer, _output, _advances
    _trainer, _output = trainer, Path(output)
    assert trainer.completed_steps == 0
    for completed in range(1, 1001):
        real_batch(completed)
        real_batch(completed)
        _advances += 2
    assert _advances == 2000
    cursor = stream.get_state().clone()
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    assert state['completed_steps'] == 1000 and state['device'] == 'cuda:0'
    assert state['serial_backward'] is True
    trainer.load_state_dict(state)
    assert trainer.completed_steps == 1000
    assert torch.equal(cursor, stream.get_state())
    assert torch.equal(state['cpu_rng'].cpu(), torch.get_rng_state())
    assert torch.equal(state['cuda_rng'].cpu(), torch.cuda.get_rng_state(trainer.device))
    for name, value in state['streams'].items():
        assert torch.equal(value.cpu(), getattr(trainer, name).get_state()), name
    restored = trainer.state_dict()
    mismatch = compare(state, restored)
    assert not mismatch, ('checkpoint restoration mismatch', mismatch)
    original = json.loads((ORIGINAL / 'COMPLETION.json').read_text())['verdict']
    gate_rows.extend(original['periods'][:2])
    assert [row['period_end'] for row in gate_rows] == [500, 1000]
    assert gate_rows[0]['hq'] == .9609 and gate_rows[1]['hq'] == .9548
    write(_output / 'RESTORE.json', dict(status='PASS', completed_steps=1000,
        external_seed=1234, external_batches_advanced=2000, batch_size=2048,
        external_cursor_sha256=tensor_sha(cursor), global_cpu_rng_sha256=tensor_sha(torch.get_rng_state()),
        global_cuda_rng_sha256=tensor_sha(torch.cuda.get_rng_state(trainer.device)),
        private_rng_sha256={name: tensor_sha(getattr(trainer, name).get_state()) for name in trainer._STREAMS},
        exact_saved_state_restored=True, fresh_training_updates_before_resume=0,
        inherited_gate_periods=gate_rows.copy(), target_degrees_during_window=60,
        checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()))
    attach_observers()
    print('RESTORE ' + json.dumps(dict(status='PASS', completed_steps=1000,
        external_batches_advanced=2000)), flush=True)


def _capture(kind, frame, returned):
    local = frame.f_locals
    snapshot = local.get('snapshot')
    record = dict(phase=kind, step=_trainer.completed_steps + 1,
                  chart_groups=None if snapshot is None else getattr(snapshot, 'mass_groups', None))
    for name in ('counts', 'ema_counts'):
        counts = local.get(name)
        if torch.is_tensor(counts):
            counts = counts.detach().cpu()
            record[name] = counts.tolist()
            record[name + '_missing_groups'] = (counts == 0).nonzero().flatten().tolist()
    fixed = local.get('fixed')
    if fixed is not None:
        record['initial_active_groups'] = (fixed.weights > 0).nonzero().flatten().tolist()
        record['initial_inactive_groups'] = (fixed.weights == 0).nonzero().flatten().tolist()
        record['retained_even_mass'] = float(fixed.weights.sum())
    if kind == 'freeze_moment' and isinstance(returned, tuple):
        record['reason'] = returned[1]
        value = returned[0]
        if value is not None:
            record['initial_active_groups'] = (value.weights > 0).nonzero().flatten().tolist()
            record['initial_inactive_groups'] = (value.weights == 0).nonzero().flatten().tolist()
            record['retained_even_mass'] = float(value.weights.sum())
    if kind == 'run_mean_phase' and isinstance(returned, dict):
        record['diagnostics'] = dict(returned['diagnostics'])
    _pending.append(record)


def _observe_return(original, kind):
    code = inspect.unwrap(original).__code__
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        previous = sys.getprofile()
        assert previous is None, 'numerical host unexpectedly has a profiler'
        def profiler(frame, event, value):
            if event == 'return' and frame.f_code is code:
                _capture(kind, frame, value)
        sys.setprofile(profiler)
        try:
            return original(*args, **kwargs)
        finally:
            sys.setprofile(previous)
    return wrapped


def attach_observers():
    from particlegan import mean_transport, feature_cells
    mean_transport.freeze_moment = _observe_return(mean_transport.freeze_moment, 'freeze_moment')
    feature_cells.run_mean_phase = _observe_return(feature_cells.run_mean_phase, 'run_mean_phase')


def note_update(trainer, step, stream):
    assert trainer is _trainer and trainer.completed_steps == step
    birth = trainer.birth_death
    paired = dict(birth.paired_average)
    row = dict(step=step, target_deg=60,
        surprise_fires=trainer.policy.surprise.fires,
        lr_g_prior_d=[[float(group['lr']) for group in opt.param_groups] for opt in (trainer.opt_g, trainer.opt_d)],
        snapshot_serial=birth.snapshot_serial, last_reaction_step=birth.last.get('step'),
        rows_since_eval=birth.rows_since_eval, paired_average=paired,
        served_source='averaged' if trainer._fast is not None else 'fast',
        mean_transport=dict(birth.last.get('mean_transport', {})),
        cumulative_mean_fires=birth.counters['mean_witness_fires'],
        cumulative_mean_moves=birth.counters['mean_moves'],
        chart_observations=list(_pending))
    _pending.clear()
    with (_output / 'steps.jsonl').open('a') as handle:
        handle.write(json.dumps(row, sort_keys=True) + '\n')
    if birth.last.get('step') == step:
        print('REACTION ' + json.dumps(dict(step=step, mean=row['mean_transport'],
            chart_observations=row['chart_observations'], paired_average=paired,
            cumulative_mean_moves=row['cumulative_mean_moves'])), flush=True)


def finish(trainer, stream, variant):
    assert trainer.completed_steps == 1500
    rows = [json.loads(line) for line in (_output / 'steps.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == list(range(1001, 1501))
    assert len([row for row in rows if row['last_reaction_step'] == row['step']]) == 50
    state = torch.load(_output / 'checkpoint-001500.pt', map_location='cpu', weights_only=False)
    assert state['completed_steps'] == 1500
    verdict = json.loads((_output / 'frames.npz.verdict.json').read_text())
    assert [r['period_end'] for r in verdict['periods']] == [500, 1000, 1500]
    assert verdict['pre_turn_hq'] == .9609
    receipt = dict(status='COMPLETE', variant=variant, training_updates=500,
        first_update=1001, last_update=1500, external_batches_advanced_before_restore=_advances,
        external_batches_in_window=1000, external_cursor_sha256=tensor_sha(stream.get_state()),
        global_cpu_rng_sha256=tensor_sha(torch.get_rng_state()),
        global_cuda_rng_sha256=tensor_sha(torch.cuda.get_rng_state(trainer.device)),
        private_rng_sha256={name: tensor_sha(getattr(trainer, name).get_state()) for name in trainer._STREAMS},
        original_quality_bar=.86481, quality_status=verdict['status'], verdict=verdict,
        elapsed_time_only_exclusions=['birth_death.last.eval_seconds'],
        scorer_changed=False, thresholds_changed=False, seeds_changed=False,
        control_reproduction=None, peak_allocated_gpu_mib=torch.cuda.max_memory_allocated(0) / 2**20,
        peak_reserved_gpu_mib=torch.cuda.max_memory_reserved(0) / 2**20)
    if variant == 'control':
        original = torch.load(ORIGINAL / 'checkpoint-001500.pt', map_location='cpu', weights_only=False)
        mismatch = compare(original, state, skip=('birth_death.last.eval_seconds',))
        retained = json.loads((ORIGINAL / 'COMPLETION.json').read_text())['verdict']
        receipt['control_reproduction'] = dict(status='PASS' if not mismatch and verdict == retained else 'FAIL',
            checkpoint_semantic_mismatches=mismatch, identical_original_verdict=verdict == retained,
            original_checkpoint_sha256=hashlib.sha256((ORIGINAL / 'checkpoint-001500.pt').read_bytes()).hexdigest())
    write(_output / 'WINDOW-COMPLETION.json', receipt)
    assert variant != 'control' or receipt['control_reproduction']['status'] == 'PASS', receipt['control_reproduction']
    print('WINDOW_COMPLETE ' + json.dumps(dict(variant=variant, quality_status=verdict['status'],
        control_reproduction=receipt['control_reproduction'])), flush=True)
