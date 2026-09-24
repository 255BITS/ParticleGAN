"""Fail-fast 200-update diagnostic from selected-base OWN cold state.

Changed policies borrow this base state only for warm_probe screening. They must
reacquire cold state under their declared policy before any own-state credit.
"""
import argparse
import json
from pathlib import Path
import tarfile
import time
import traceback

import stability_runner as runner
from adam_response import PROPOSALS, response_policy
from critic_signal_screen import append
from continuous_screen import source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=PROPOSALS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--cold-candidate', type=Path,
                        help='same-policy cold candidate directory for own-state diagnostic')
    args = parser.parse_args()
    selection = json.loads((runner.PREP_ROOT/'current-base.json').read_text())
    candidate = args.cold_candidate or runner.REPO/selection['cold_candidate']
    checkpoint = candidate/'mode_hold/final-state.pt'
    digest = runner.sha(checkpoint)
    own = args.cold_candidate is not None
    gate = 'own_state_diagnostic_200' if own else 'warm_probe_base_own_200'

    def verify():
        assert runner.sha(checkpoint) == digest
        if not own:
            assert digest == selection['cold_checkpoint_sha256']
        else:
            assert json.loads((candidate/'options.json').read_text())['adam_response'] == args.variant
        stages = json.loads((candidate/'status.json').read_text())['stages']
        assert next(s for s in stages if s['gate'] == 'mode_hold')['status'] == 'PASS'
        cfg = json.loads((candidate/'config.json').read_text())
        assert dict(g=cfg['lr'], d=cfg['lr']*cfg['d_lr_mult'],
                    prior=cfg['lr']*cfg['prior_lr_mult']) == selection['fixed_rates']

    runner.CANDIDATE, runner.STATE_SHA = candidate, digest
    # The checkpoint came from this candidate's cold batch, not H's archive.
    parent_archive = candidate.parent/'source.tar.gz'
    runner.ARCHIVE_SHA = runner.sha(parent_archive)
    runner.verify_sources = verify
    runner.POLICIES[args.variant] = dict(runner.POLICIES['control'],
        family='fixed Adam denominator response', epsilon=PROPOSALS[args.variant])
    runner.signal_policy = lambda options: response_policy(dict(options, adam_response=args.variant))
    started = time.perf_counter()
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = runner.run(args.variant, args.output, 200)
        result.update(candidate=args.variant, gate=gate,
                      status='PASS' if result['status'] == 'SHORT_PASS' else result['status'],
                      state_origin='same-policy own cold acquisition' if own else 'selected-base OWN acquisition; changed-policy warm diagnostic',
                      shared_gate_eligible=False)
        declaration = json.loads((args.output/'declaration.json').read_text())
        declaration.update(candidate=args.variant, gate=gate, state_origin=result['state_origin'],
                           candidate_own_state=own, borrowed_H_state=False,
                           epsilon=PROPOSALS[args.variant],
                           parent_source_archive_path=str(parent_archive.resolve()))
        runner.write(args.output/'declaration.json', declaration)
        runner.write(args.output/'summary.json', result)
        sources = source_hashes()
        for name in ('critic_signal.py', 'selected_h_extension.py', 'critic_signal_screen.py',
                     'h_stability/stability_runner.py', 'h_stability/adam_response.py',
                     'h_stability/adam_response_probe.py'):
            path = 'reports/toy100/'+name
            sources[path] = runner.sha(runner.REPO/path)
        runner.write(args.output/'source-manifest.json', sources)
        with tarfile.open(args.output/'source.tar.gz', 'w:gz') as archive:
            for name in sources:
                archive.add(runner.REPO/name, arcname=name)
        event = dict(candidate=args.variant, gate=gate, status=result['status'],
                     seconds=time.perf_counter()-started,
                     metrics={k: result[k] for k in ('window', 'final', 'state_origin', 'full_budget')},
                     artifact=str((args.output/'summary.json').resolve()))
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        error = traceback.format_exc()
        (args.output/'error.txt').write_text(error)
        event = dict(candidate=args.variant, gate=gate, status='ERROR',
                     seconds=time.perf_counter()-started, metrics={}, error=error,
                     artifact=str((args.output/'error.txt').resolve()))
    append(args.ledger, event)
    runner.print_json(event)
    if event['status'] == 'ERROR':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
