"""New source epoch: fail-fast cold gates after exact finite-trial recovery."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reports.toy100 import pr84_critic_refinement_cold_probe as original_driver
from reports.toy100.pr84_critic_refinement_finite import METHOD, pr84_critic_refinement_finite

CAPTURE_SHA = '19ceb38a92ba95ffaccd3aafbda55bbd77917765612666d923e986e5523d6965'


def require_recovery(path):
    value = json.loads(path.read_text())
    if (value['status'] != 'PASS_NUMERICAL_RECOVERY' or value['capture_sha256'] != CAPTURE_SHA
            or value['method'] != METHOD):
        raise RuntimeError('the exact failed fit has not passed the declared numerical repair gate')
    for name, digest in value['source_hashes'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'numerical recovery source changed: {name}')
    continuation = value['continuation']
    if (continuation['start'] != 472 or continuation['end'] != 491 or continuation['steps'] != 20
            or continuation['finite_state_pass'] is not True
            or [row['step'] for row in continuation['quality_observations']] != list(range(472,492))
            or value['after_set_step_parity_exact'] is not True):
        raise RuntimeError('the complete20-update numerical continuation is required')
    exact = value['exact_failed_fit']
    if (any(exact[key] is not True for key in ('original_failure_reproduced',
            'finite_records_exact', 'recovered_best_state_exact'))
            or exact['closure_calls'] != 49 or exact['finite_closure_calls'] != 48
            or exact['nonfinite_closure_calls'] != 1
            or value['finite_path_parity']['passed'] is not True):
        raise RuntimeError('failed-fit and unchanged-finite-path parity are required')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hold', type=Path, required=True)
    parser.add_argument('--recovery', type=Path, required=True)
    parser.add_argument('--finite-parity', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    previous = original_driver.require_hold(args.hold)
    recovery = require_recovery(args.recovery)
    parity = json.loads(args.finite_parity.read_text())
    if (parity['status'] != 'PASS_FINITE_PATH_PARITY' or parity['method'] != METHOD
            or parity['steps'] != 44
            or [(r['start'],r['end'],r['steps']) for r in parity['branches']] !=
               [(1324,1335,12),(1380,1395,16),(1530,1545,16)]):
        raise RuntimeError('all44 original finite-path checks are required')
    for row in parity['branches']:
        if (any(row[key] is not True for key in ('all_state_hashes_exact',
                'all_supports_and_metrics_exact','original_records_exact'))
                or row['nonfinite_rejections'] != 0):
            raise RuntimeError('finite-path behavior changed')
    for name,digest in parity['source_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'finite-path source changed: {name}')
    names = set(previous['source']) | set(recovery['source_hashes']) | set(parity['source_hashes']) | {
        'reports/toy100/pr84_critic_refinement_finite.py',
        'reports/toy100/pr84_critic_refinement_finite_probe.py',
        'reports/toy100/pr84_critic_refinement_cold_probe.py',
        'reports/toy100/pr84_critic_refinement_cold.py',
        'reports/toy100/pr84_critic_refinement_capture.py',
        'benchmarks/transfer_suite/legacy_noise_adapters.py',
        'benchmarks/transfer_suite/toy100_compatibility.py',
        'benchmarks/transfer_suite/protocol.py',
        'tests/test_pr84_critic_refinement_finite.py',
    }
    source = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(names)}
    declaration = dict(method=METHOD, phase='cold', source=source, shared_gate_eligible=False,
        source_epoch='explicit later-nonfinite-trial rejection; original fit and adapters unchanged on disk',
        previous_hold_sha256=hashlib.sha256(args.hold.read_bytes()).hexdigest(),
        numerical_recovery_sha256=hashlib.sha256(args.recovery.read_bytes()).hexdigest(),
        finite_path_parity_sha256=hashlib.sha256(args.finite_parity.read_bytes()).hexdigest(),
        hold_scope='prior original finite-path hold; equivalence checked, no new long-hold claim',
        fit_policy='one40/80 attempt; a later nonfinite trial ends fit and restores best finite training-loss point',
        initial_nonfinite_policy='ERROR', invalid_trial_gradient_policy='clear invalid D leaf grads after best restoration',
        extra_attempts=0, failed_trials_count_against_budget=True,
        order=['trajectory400', 'mode_hold1200'], stop_at_first_failed_host=True,
        rates=dict(g=.00425, d=.00425, prior=.0085), lr_decay=False, seed=0,
        noise_horizon='unchanged original host budget',
        native_fit_pairs=dict(trajectory=96, mode_hold=1024),
        inherited_config_name='pr84_critic_refinement_cold; method and source epoch above identify this repaired run',
        scope='new cold acquisition gate; own-acquired continuation and production gates remain')
    args.output.mkdir(parents=True, exist_ok=False)
    for name in source:
        target = args.output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    (args.output / 'previous-hold.json').write_bytes(args.hold.read_bytes())
    (args.output / 'recovery-gate.json').write_bytes(args.recovery.read_bytes())
    (args.output / 'finite-parity-gate.json').write_bytes(args.finite_parity.read_bytes())
    (args.output / 'declaration.json').write_text(json.dumps(declaration, indent=2)+'\n')
    print(json.dumps(dict(event='DECLARED', **declaration)), flush=True)
    torch.set_num_threads(1)
    # Reuse only the already reviewed cold ordering, strict host grader,
    # actual-Adam rate/moment audit and complete final-state serialization.
    with patch.object(original_driver, 'pr84_critic_refinement_cold', pr84_critic_refinement_finite), \
         patch.object(original_driver, 'METHOD', METHOD):
        try:
            original_driver.run(args.output, source)
        except BaseException as error:
            (args.output / 'error.json').write_text(json.dumps(dict(status='ERROR_INCOMPLETE',
                method=METHOD, error=repr(error), shared_gate_eligible=False), indent=2)+'\n')
            raise


if __name__ == '__main__':
    main()
