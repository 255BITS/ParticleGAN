"""Own-state continuation 1201-2400 for the balanced-mass anchor.

Replays the same seed-0 cold ring, then continues that snapshot with the
same factory. Scores every resumed update. This is the stay check for the
fidelity change, not a second seed.
"""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reports.toy100.pr84_critic_refinement_capture import snapshot
from reports.toy100.sample_anchor_balanced_candidate import METHOD, sample_anchor_balanced_candidate
from benchmarks.locked_shared import mode_hold
from reports.toy100.sample_anchor_balanced_probe import fidelity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy, declared_recipe
    from reports.toy100 import pr84_model_error_recovery as recovery

    config = json.loads((ROOT / 'configs/toy100/constraints_simple_regularization.json').read_text())
    config.update(name=METHOD, lr_floor=1., lr_anneal_start=0.)
    config.pop('network_lr_horizon_cap', None)
    config.pop('network_lr_floor', None)
    recipe, noise, _ = declared_recipe(config)
    spec = next(job['spec'] for job in plan() if job['spec']['name'] == 'mode_hold')
    print(json.dumps(dict(event='HOLD_DECLARED', method=METHOD, window='1201-2400')), flush=True)
    with sample_anchor_balanced_candidate(task='mode_hold', correction=True) as (recorder, _):
        run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
        saved = snapshot(recorder._local)
    if saved['noise']['step_calls'] != 1200:
        raise RuntimeError(f"ring snapshot clock is {saved['noise']['step_calls']}, expected 1200")
    torch.save(saved, args.output / 'ring1200.pt')
    print(json.dumps(dict(event='RING_SNAPSHOT', step_calls=1200)), flush=True)

    def factory():
        return sample_anchor_balanced_candidate(task='mode_hold', start_step=0, correction=True)

    def observe(row):
        if row['step'] % 50 == 0 or not row['passed']:
            print(json.dumps(dict(event='OWN_HOLD', step=row['step'], modes=row['modes'],
                                  hq=row['hq'], passed=row['passed'])), flush=True)

    with patch.object(recovery, 'pr84_critic_refinement_finite', factory):
        branch = recovery.run_continuation(
            saved, recipe, noise, completed_steps=1200, target_steps=2400,
            perturb=False, fail_fast=False, log=observe)
    rows = branch['receipt']['checkpoints']
    fails = [row['step'] for row in rows if row['modes'] != 8 or row['hq'] < .9]
    final = fidelity(branch['final_clean_support'], mode_hold.ring_means())
    summary = dict(
        method=METHOD, host='neural', window='1201-2400', updates=len(rows),
        min_modes=min(row['modes'] for row in rows),
        min_hq=min(row['hq'] for row in rows),
        failing_steps=fails, n_fail=len(fails),
        final_modes=rows[-1]['modes'], final_hq=rows[-1]['hq'],
        fidelity_2400=final,
        pass_all=len(fails) == 0 and rows[-1]['modes'] == 8)
    (args.output / 'hold.json').write_text(json.dumps(summary, allow_nan=False) + '\n')
    print(json.dumps(dict(event='HOLD_DONE', **summary)), flush=True)


if __name__ == '__main__':
    main()
