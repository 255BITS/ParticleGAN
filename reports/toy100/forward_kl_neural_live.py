"""Live D/G+Adam host for the finite-GH9 remembered likelihood adapter.

The restoring update is ``ForwardKLV2Recorder.correct``: after the native PR84
alternating Adam step it proposes a clean 12x2 target with the frozen GH5 /
GH9 / remembered-donor operator and keeps a pre-G joint fit only when that
fit converges and the actual finite-GH9 cost falls. D, Adam moments, RNG, and
native gradients stay untouched. This is not a free-output screen.

``trajectory`` does not activate the likelihood correction. The adapter
contract is mode-hold only, same as the saved44 filter. A trajectory pass is
the underlying PR84 host, not a KL acquisition result.
"""
from contextlib import contextmanager
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reports.toy100 import pr84_prediction_state_filter as replay
from reports.toy100.forward_kl_neural_v2 import METHOD, ForwardKLV2Recorder, forward_kl_neural_v2
from reports.toy100.pr84_critic_refinement_filter import grade
from reports.toy100.sample_anchor_free1200 import load_states

WARM_START, WARM_END = 1324, 1339


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    names = (
        'reports/toy100/forward_kl_neural_live.py',
        'reports/toy100/forward_kl_neural_v2.py',
        'reports/toy100/forward_kl_gh9_remembered.py',
        'reports/toy100/forward_kl_gh9_stress.py',
        'reports/toy100/forward_kl_remembered_donor_rescue.py',
        'reports/toy100/forward_kl_chunked.py',
        'reports/toy100/forward_kl_free_filter.py',
    )
    return {name: _sha(ROOT / name) for name in names}


def _log_corrections():
    original = ForwardKLV2Recorder.correct

    def logged(self, optimizer):
        started = time.perf_counter()
        original(self, optimizer)
        row = self.corrections[-1]
        print(json.dumps(dict(event='KL_UPDATE', step=row['step'], selected=row['selected'],
            pre_gh9=row['pre_gh9'], final_gh9=row['final_gh9'],
            banks=row['bank_count'], seconds=round(time.perf_counter() - started, 3))), flush=True)

    return patch.object(ForwardKLV2Recorder, 'correct', logged)


def run_warm(output):
    config = json.loads((ROOT / 'configs/toy100/constraints_simple_regularization.json').read_text())
    _, warm, inputs = load_states()
    source_dir = output / 'source'
    source_dir.mkdir()

    @contextmanager
    def selected(*, task, prediction):
        with forward_kl_neural_v2(task=task, start_step=WARM_START - 1,
                                   correction=prediction, history_mode='activate') as value:
            yield value

    started = time.perf_counter()
    with _log_corrections(), patch.object(replay.prediction_module, 'pr84_opponent_prediction', selected):
        value, _ = replay.run_local(config, warm, start=WARM_START, end=WARM_END,
                                     opponent='predicted', source_dir=source_dir)
    local = grade(value['points'])
    summary = dict(gate='neural_warm16', status='PASS' if local['pass_all'] else 'FAIL',
        host='neural', method=METHOD, window=[WARM_START, WARM_END],
        input_hashes=inputs, local_gate=local,
        selections=[row['selected'] for row in value['dynamics']['corrections']],
        seconds=time.perf_counter() - started,
        note='live resume of archived warm1324 pre_step; fresh likelihood history at this window, same adapter as neural44')
    (output / 'warm.json').write_text(json.dumps(dict(summary=summary, points=[
        dict(step=p['step'], grade=p['grade']) for p in value['points']]), allow_nan=False) + '\n')
    print(json.dumps(dict(event='GATE', **summary)), flush=True)
    return summary


def run_host(output, task, steps):
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy, declared_recipe
    config = json.loads((ROOT / 'configs/toy100/constraints_simple_regularization.json').read_text())
    config.update(name='forward_kl_neural_live', lr_floor=1., lr_anneal_start=0.)
    config.pop('network_lr_horizon_cap', None)
    config.pop('network_lr_floor', None)
    recipe, noise, _ = declared_recipe(config)
    spec = next(job['spec'] for job in plan() if job['spec']['name'] == task)
    started = time.perf_counter()
    with _log_corrections(), forward_kl_neural_v2(task=task, start_step=0,
            correction=True, history_mode='activate') as (recorder, _):
        result, context = run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
    verdict = test_verdict(spec, result)
    corrections = recorder.corrections
    summary = dict(gate=task, status='PASS' if verdict['passed'] else 'FAIL',
        host='neural', method=METHOD, verdict=verdict, live=result.get('live'),
        seconds=result.get('seconds'), wall_seconds=time.perf_counter() - started,
        corrections=len(corrections),
        selections=[row['selected'] for row in corrections],
        likelihood_active=task == 'mode_hold',
        final_grade=None)
    if task == 'mode_hold' and result.get('observations'):
        tail = result['observations'][-5:]
        summary['final_observations'] = tail
    (output / f'{task}.json').write_text(json.dumps(dict(
        summary=summary, noise=context['noise_receipt'],
        dynamics_method=recorder.receipt().get('method')), allow_nan=False) + '\n')
    print(json.dumps(dict(event='GATE', gate=task, status=summary['status'],
        verdict=verdict, corrections=len(corrections),
        seconds=summary['seconds'])), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', choices=('warm', 'trajectory', 'ring'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    declaration = dict(method=METHOD, host='neural', gate=args.gate,
        sources=source_hashes(), seed=0, rates=dict(g=.00425, d=.00425, prior=.0085),
        objective='finite GH9 cumulative smoothed forward cross-entropy plus donor/EM; not a pure GAN fix',
        wiring='native PR84 D-then-G Adam once, then pre-G joint fit of the pure GH9 target or exact restore',
        warm='live 1324-1339 from archived warm pre_step; all 8 modes and HQ>=.9',
        trajectory='400-update identity host; likelihood correction stays off',
        ring='1200-update cold mode_hold; likelihood correction on; 8 modes HQ>=.9 required by the host verdict')
    (args.output / 'declaration.json').write_text(json.dumps(declaration, indent=2) + '\n')
    print(json.dumps(dict(event='DECLARED', gate=args.gate, method=METHOD)), flush=True)
    if args.gate == 'warm':
        run_warm(args.output)
    elif args.gate == 'trajectory':
        run_host(args.output, 'trajectory', 400)
    else:
        run_host(args.output, 'mode_hold', 1200)


if __name__ == '__main__':
    main()
