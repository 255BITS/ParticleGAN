"""Diagnostic continuation from the selected base's own cold-acquired state.

Uses the validated frozen update loop with unchanged acquired Adam/RNG state.
This measures a research blocker; it does not bypass the full toy promotion gate.
"""
import argparse
import json
from pathlib import Path
import shutil

import stability_runner as runner

ROOT = Path(__file__).resolve().parents[3]
SELECTION = Path(__file__).with_name('current-base.json')
H_CANDIDATE = runner.CANDIDATE
H_STATE_SHA = runner.STATE_SHA
VERIFY_H = runner.verify_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, choices=(200, 1200), default=200)
    parser.add_argument('--diagnostic-full-window', action='store_true')
    args = parser.parse_args()
    selection = json.loads(SELECTION.read_text())
    candidate = ROOT / selection['cold_candidate']
    state_sha = selection['cold_checkpoint_sha256']
    if selection['options'].get('adam_response'):
        from adam_response import response_policy
        runner.signal_policy = response_policy
        runner.POLICIES['control'] = dict(runner.POLICIES['control'],
            family='selected constant-rate Adam response', epsilon=selection['epsilon'])

    def verify():
        # Preserve the original H source/control checks, then verify the new state.
        runner.CANDIDATE, runner.STATE_SHA = H_CANDIDATE, H_STATE_SHA
        VERIFY_H()
        runner.CANDIDATE, runner.STATE_SHA = candidate, state_sha
        assert runner.sha(candidate/'mode_hold/final-state.pt') == state_sha
        status = json.loads((candidate/'status.json').read_text())
        ring = next(row for row in status['stages'] if row['gate'] == 'mode_hold')
        assert ring['status'] == 'PASS'
        config = json.loads((candidate/'config.json').read_text())
        actual = dict(g=config['lr'], d=config['lr']*config['d_lr_mult'],
                      prior=config['lr']*config['prior_lr_mult'])
        assert actual == selection['fixed_rates']

    runner.verify_sources = verify
    result = runner.run('control', args.output, args.steps, args.diagnostic_full_window)
    if result['status'] == 'SHORT_PASS':
        result['status'] = 'PASS'
    result.update(candidate=selection['candidate'], gate=f'own_state_diagnostic_{args.steps}',
                  qualification='UNQUALIFIED: older toys still fail',
                  borrowed_H_state=False, rates_changed_on_restore=False)
    declaration = json.loads((args.output/'declaration.json').read_text())
    declaration.update(state_origin='own cold acquisition with the same fixed rates',
                       wrapper_sha256=runner.sha(Path(__file__)),
                       qualification='diagnostic only; full cold promotion gates not passed')
    runner.write(args.output/'declaration.json', declaration)
    runner.write(args.output/'summary.json', result)
    shutil.copy2(__file__, args.output/'selected_base_probe.py')
    shutil.copy2(SELECTION, args.output/'current-base.json')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
