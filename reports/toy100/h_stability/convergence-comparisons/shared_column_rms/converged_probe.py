"""Bounded settling then hold, from a candidate's own unchanged cold policy."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import tarfile

import stability_runner as runner
from convergence_gate import ConvergenceGate
from continuous_screen import verify_receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = runner.REPO
    if args.candidate is None:
        selection = json.loads(Path(__file__).with_name('current-base.json').read_text())
        candidate = root / selection['cold_candidate']
    else:
        candidate = args.candidate
    candidate = candidate.resolve()
    config = json.loads((candidate/'config.json').read_text())
    options = json.loads((candidate/'options.json').read_text())
    original_status = json.loads((candidate/'status.json').read_text())
    ring = next(r for r in original_status['stages'] if r['gate'] == 'mode_hold')
    checkpoint = candidate/'mode_hold/final-state.pt'
    checkpoint_sha = runner.sha(checkpoint)
    receipt = json.loads(gzip.decompress((candidate/'mode_hold/signal-policy.json.gz').read_bytes()))

    def verify():
        if ring['status'] not in ('PASS', 'FAIL'):
            raise ValueError('requires a completed cold ring run, not an error or skipped gate')
        assert checkpoint_sha == ring['checkpoint_sha256']
        assert runner.sha(checkpoint) == checkpoint_sha
        verify_receipt(receipt, config, task='mode_hold')
        assert config['lr_floor'] == 1 and config['prior_reg'] == 0

    if 'adam_response' in options:
        from adam_response import response_policy
        runner.signal_policy = response_policy
    else:
        from critic_signal import signal_policy
        runner.signal_policy = signal_policy
    runner.CANDIDATE, runner.STATE_SHA = candidate, checkpoint_sha
    runner.ARCHIVE_SHA = runner.sha(candidate.parent/'source.tar.gz')
    runner.verify_sources = verify
    runner.POLICIES['control'] = dict(runner.POLICIES['control'],
        family='same declared cold policy during settling and hold', options=options)
    gate = ConvergenceGate()
    result = runner.run('control', args.output, gate.settling_budget + gate.hold_budget,
                        convergence_gate=gate)
    result.update(candidate=candidate.name, original_cold_verdicts=original_status['stages'],
                  borrowed_state=False, qualification='diagnostic only; all toy gates still required')
    runner.write(args.output/'summary.json', result)
    declaration = json.loads((args.output/'declaration.json').read_text())
    declaration.update(state_origin='own cold checkpoint with unchanged policy; no restore at convergence',
                       original_cold_ring_status=ring['status'], wrapper_sha256=runner.sha(Path(__file__)),
                       eligible_objectives=f"{config['loss_type']} {config['gan_mode']} discriminator objective")
    runner.write(args.output/'declaration.json', declaration)
    shutil.copy2(__file__, args.output/'converged_probe.py')
    # Archive the applied research policy and frozen sources using existing plumbing.
    from critic_signal_screen import source_hashes
    hashes = source_hashes()
    extras = list(Path(__file__).parent.glob('*.py'))
    extras += [root/'reports/toy100'/name for name in ('selected_h_extension.py', 'selected_h_remaining.py')]
    for path in extras:
        if not path.exists():
            continue
        hashes[str(path.relative_to(root))] = runner.sha(path)
    runner.write(args.output/'source-manifest.json', hashes)
    with tarfile.open(args.output/'source.tar.gz', 'w:gz') as archive:
        for name in hashes:
            archive.add(root/name, arcname=name)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
