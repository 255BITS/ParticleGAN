"""Passive, source-bound first-100-update replay of the repaired cold ring.

The native host still has its 1,200-update recipe and noise horizon. Its
existing sparse checkpoint is observed at updates 50 and 100; a sentinel
stops after checkpoint 100, before any update 101. Four already reviewed
refinement-stage snapshots are saved at each predeclared selected update.
No training method, optimizer, data draw, diagnostic cadence, or gate changes.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.locked_shared import mode_hold
from benchmarks.transfer_suite.compare_defaults import plan
from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy, declared_recipe
from reports.toy100.pr84_critic_refinement_capture import STAGES, _sha, capture_refinement, snapshot
from reports.toy100.pr84_critic_refinement_finite import METHOD, pr84_critic_refinement_finite
from reports.toy100.pr84_critic_refinement_probe import untimed


SELECTED = (1, 25, 50, 100)
STOP = 100
SPARSE_EVAL = (50, 100)
RATES = {'d': (.00425,), 'g': (.00425, .0085)}


class PrefixComplete(Exception):
    """Raised only after the native host's scheduled checkpoint at update 100."""


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_gzip_json(path, value):
    with path.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as compressed:
            compressed.write((json.dumps(value, allow_nan=False) + '\n').encode())


def source_bound_reference(cold):
    declaration = json.loads((cold / 'declaration.json').read_text())
    summary = json.loads((cold / 'summary.json').read_text())
    trajectory = json.loads((cold / 'trajectory.json').read_text())
    ring = json.loads((cold / 'mode_hold.json').read_text())
    if (declaration['method'] != METHOD or summary['method'] != METHOD
            or ring['method'] != METHOD or summary['status'] != 'FAIL'
            or trajectory['verdict']['status'] != 'PASS'
            or ring['verdict']['status'] != 'FAIL'
            or ring['dynamics']['outer_steps'] != 1200
            or len(ring['dynamics']['records']) != 1200
            or len(ring['dynamics']['refinement_records']) != 1200
            or [row['step'] for row in ring['result']['observations']] != list(range(50, 1201, 50))
            or ring['source'] != declaration['source']):
        raise RuntimeError('reference is not the completed repaired cold ring')
    for name, digest in declaration['source'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f'original cold source changed: {name}')
    expected = json.loads((ROOT / 'configs/toy100/constraints_simple_regularization.json').read_text())
    expected.update(name='pr84_critic_refinement_cold', lr_floor=1., lr_anneal_start=0.)
    expected.pop('network_lr_horizon_cap')
    expected.pop('network_lr_floor')
    config = json.loads((cold / 'config.json').read_text())
    if config != expected or ring['noise']['total_steps'] != 1200:
        raise RuntimeError('cold configuration or original noise horizon changed')
    spec = next(job['spec'] for job in plan() if job['spec']['name'] == 'mode_hold')
    if spec['steps'] != 1200:
        raise RuntimeError('mode_hold host budget changed')
    return declaration, ring, config, spec


def freeze_declaration(output, cold, source, reference):
    row = dict(method=METHOD, phase='diagnostic_first100_replay',
        verdict='NO_PROMOTION_OR_QUALITY_GATE',
        cold_reference_status='FAIL_3_MODES',
        cold_reference_sha256={name: sha(cold / name) for name in
            ('declaration.json', 'config.json', 'summary.json', 'mode_hold.json')},
        source=source, selected_updates=list(SELECTED), stages=list(STAGES),
        stop_after_native_checkpoint=STOP, sparse_native_evaluations=list(SPARSE_EVAL),
        original_recipe_steps=1200, original_noise_horizon=1200,
        nominal_and_required_actual_rates=RATES, lr_decay=False,
        seed=0, no_candidate_change=True, shared_gate_eligible=False,
        comparison='all 100 untimed update and refinement records plus exact original 50/100 observations',
        reference_step100=dict(modes=reference['result']['observations'][1]['modes'],
                               hq=reference['result']['observations'][1]['hq']))
    write_json(output / 'declaration.json', row)
    with tarfile.open(output / 'source.tar.gz', 'w:gz') as archive:
        for name in source:
            archive.add(ROOT / name, arcname=name)
    print(json.dumps(dict(event='DECLARED', declaration=row)), flush=True)


def run(cold, output):
    _, reference, config, spec = source_bound_reference(cold)
    recipe, noise, _ = declared_recipe(config)
    source_names = set(json.loads((cold / 'declaration.json').read_text())['source']) | {
        'reports/toy100/pr84_finite_cold_prefix_capture.py',
        'reports/toy100/pr84_critic_refinement_probe.py',
    }
    source = {name: sha(ROOT / name) for name in sorted(source_names)}
    output.mkdir(parents=True, exist_ok=False)
    freeze_declaration(output, cold, source, reference)

    torch.set_num_threads(1)
    observations = []
    calls = {}
    after_checkpoint = None
    ordinary_adam_step = torch.optim.Adam.step
    ordinary_checkpoint = mode_hold.checkpoint

    def audited_adam_step(optimizer, closure=None):
        calls.setdefault(optimizer, []).append(tuple(group['lr'] for group in optimizer.param_groups))
        return ordinary_adam_step(optimizer, closure=closure)

    def sparse_checkpoint(step, measure):
        nonlocal after_checkpoint
        def observed_measure():
            row = measure()
            observations.append(dict(step=step, **row))
            return row
        ordinary_checkpoint(step, observed_measure)
        if step % 25 == 0:
            print(json.dumps(dict(event='PREFIX_PROGRESS', update=step,
                                  native_observations=len(observations),
                                  fit_gradient_evaluations=recorder.fit_gradient_evaluations)), flush=True)
        if step == STOP:
            after_checkpoint = snapshot(recorder._local)
            raise PrefixComplete()

    with patch.object(torch.optim.Adam, 'step', audited_adam_step), \
         pr84_critic_refinement_finite(task='mode_hold') as (recorder, _), \
         capture_refinement(recorder, task='mode_hold', steps=SELECTED) as capture, \
         patch.object(mode_hold, 'checkpoint', sparse_checkpoint):
        try:
            run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
        except PrefixComplete:
            pass
        else:
            raise RuntimeError('native 1,200-update host did not stop after checkpoint 100')

    actual = recorder.receipt()
    expected = reference['dynamics']
    checks = dict(
        update_records_exact=untimed(actual['records']) == untimed(expected['records'][:STOP]),
        refinement_records_exact=(untimed(actual['refinement_records']) ==
                                  untimed(expected['refinement_records'][:STOP])),
        native_observations_exact=(untimed(observations) ==
                                   untimed(reference['result']['observations'][:2])),
        sparse_observation_steps=[point['step'] for point in observations] == list(SPARSE_EVAL),
        selected_capture_steps=sorted(capture.saved_states) == list(SELECTED),
        all_four_stages=all(tuple(capture.saved_states[step]) == STAGES for step in SELECTED),
        capture_rng_checks=capture.rng_checks == len(SELECTED) * len(STAGES),
        after_checkpoint100=(after_checkpoint is not None and
            after_checkpoint['noise']['step_calls'] == STOP and
            after_checkpoint['noise_policy']['total_steps'] == 1200 and
            after_checkpoint['snapshot_scope']['host_loop_step'] == STOP - 1),
        outer_updates=recorder.outer_steps == STOP,
        first_bank_rng=recorder.bank_rng_verified == STOP,
        fit_rng=recorder.fit_rng_verified == STOP,
    )
    accounting = {}
    for role, optimizer in zip(('d', 'g'), recorder.optimizers):
        rates = calls.get(optimizer, [])
        moments = [int(optimizer.state[p]['step']) for group in optimizer.param_groups
                   for p in group['params']]
        accounting[role] = dict(actual_adam_calls=len(rates),
                                applied_rates=sorted({rate for row in rates for rate in row}),
                                moments_min=min(moments), moments_max=max(moments))
        checks[f'{role}_actual_rates_and_moments'] = (
            len(rates) == STOP and all(row == RATES[role] for row in rates)
            and all(moment == STOP for moment in moments))
    if not all(checks.values()):
        write_json(output / 'failed-checks.json', dict(checks=checks, accounting=accounting))
        raise RuntimeError(f'passive prefix replay differed from original: {checks}')

    states = dict(selected=capture.saved_states, after_checkpoint100=after_checkpoint)
    state_path = output / 'prefix-states.pt'
    torch.save(states, state_path)
    write_gzip_json(output / 'prefix-records.json.gz', dict(
        records=actual['records'], refinement_records=actual['refinement_records'],
        native_observations=observations))
    result = dict(status='PASS_EXACT_DIAGNOSTIC_REPLAY', method=METHOD,
        shared_gate_eligible=False, source=source, checks=checks,
        compared_update_records=STOP, compared_refinement_records=STOP,
        compared_original_native_observations=list(SPARSE_EVAL),
        selected_capture=capture.receipt(),
        after_checkpoint100_sha256=_sha(after_checkpoint),
        original_reference_mode_hold_sha256=sha(cold / 'mode_hold.json'),
        accounting=accounting,
        observations=[dict(step=row['step'], modes=row['modes'], hq=row['hq'],
                           nearest_counts=row['nearest_counts']) for row in observations],
        noise_horizon=after_checkpoint['noise_policy']['total_steps'],
        noise_step_calls=after_checkpoint['noise']['step_calls'],
        output_nonzero_steps=after_checkpoint['noise_policy']['_nonzero_output_steps'],
        input_nonzero_steps=after_checkpoint['noise_policy']['_nonzero_steps'],
        state_file=state_path.name, state_file_sha256=sha(state_path),
        records_file='prefix-records.json.gz',
        records_file_sha256=sha(output / 'prefix-records.json.gz'),
        source_archive_sha256=sha(output / 'source.tar.gz'),
        interpretation='early fixed-target acquisition diagnostic only; original cold gate remains failed')
    write_json(output / 'result.json', result)
    print(json.dumps(dict(event='PREFIX_DONE', **result)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cold', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.cold, arguments.output)


if __name__ == '__main__':
    main()
