"""Nine ordinary native updates after the archived paired update-2401 bank test.

The archived update 2401 starts from the candidate's qualified own update-2400
state. Only its D real bank is conditioned to omit component 0 in one arm.
This source adds no intervention: both saved arms resume through update 2410
with the original mode_hold source, noise horizon, optimizer and anchor method.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import torch


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def grade(row):
    return dict(step=row['step'], modes=row['modes'], hq=row['hq'],
                passed=row['modes'] == 8 and row['hq'] >= .9,
                missing_modes=row['missing_modes'])


def run(root, cold, hold, source, output):
    sys.path.insert(0, str(root))
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
    from reports.toy100 import sample_anchor_own_state_probe as own
    from reports.toy100.pr84_critic_refinement_capture import _sha

    torch.set_num_threads(1)
    factory, method, filename = own.load_factory(root, own.PRESTART_FACTORY)
    _, _, _, config = own.require_cold(cold, root, factory,
                                      own.PRESTART_FACTORY, method, filename)
    hold_declaration = json.loads((hold / 'declaration.json').read_text())
    saved, _ = own.require_hold(hold, cold, hold_declaration)
    source_result = json.loads((source / 'result.json').read_text())
    if (source_result['initial_state_sha256'] != _sha(saved)
            or source_result['scope'] !=
            'one actual native neural update2401; only its D real128 bank conditioned absent component0'):
        raise RuntimeError('archived update 2401 did not start from this qualified own state')
    for name, digest in source_result['source'].items():
        if sha((root / name).read_bytes()) != digest:
            raise RuntimeError(f'archived first-step source changed: {name}')
    recipe, noise, _ = declared_recipe(config)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='COMPLETE', scope='archived conditioned update2401 followed by nine ordinary native updates through2410; same fixed target',
                  conditioned_event='D real128 bank only at update2401 in omitted arm; no subsequent intervention',
                  source_first_step_sha256=sha((source / 'result.json').read_bytes()),
                  qualified_own2400_state_sha256=_sha(saved),
                  method=method, factory=own.PRESTART_FACTORY, branches={})
    generated_hashes = []
    for label in ('ordinary', 'omitted_D_bank'):
        prior = source_result['branches'][label]
        state_path = source / (label + '-state.pt')
        loaded = torch.load(state_path, weights_only=True, map_location='cpu')
        if (sha(state_path.read_bytes()) != prior.get('state_file_sha256', sha(state_path.read_bytes()))
                or _sha(loaded) != prior['final_state_sha256']
                or loaded['noise']['step_calls'] != 2401):
            raise RuntimeError(f'archived {label} post-2401 state changed')
        started = time.perf_counter()
        branch, generated = own.run_bound(loaded, recipe, noise, factory,
                                          completed=2401, target=2410)
        elapsed = time.perf_counter() - started
        receipt = branch['receipt']
        if (receipt['updates'] != 9 or not receipt['completed']
                or receipt['actual_adam_updates'] != {'d': 9, 'g': 9}
                or receipt['optimizer_callbacks'] != {'d': 27, 'g': 27}
                or [row['step'] for row in receipt['checkpoints']] != list(range(2402, 2411))
                or branch['state']['noise_policy']['total_steps'] != 1200):
            raise RuntimeError(f'{label} continuation changed the original clock or budget')
        first = grade(prior['receipt']['checkpoints'][0])
        suffix = [grade(row) for row in receipt['checkpoints']]
        grades = [first] + suffix
        if label == 'ordinary' and not first['passed']:
            raise RuntimeError('ordinary archived first step no longer passes')
        if label == 'omitted_D_bank' and (first['passed'] or first['missing_modes'] != [0]):
            raise RuntimeError('conditioned first-step loss differs from archive')
        full = dict(first_step=first, suffix=suffix, first_full_recovery_step=next(
            (row['step'] for row in suffix if row['passed']), None),
            last_five=suffix[-5:], failing_steps=[row['step'] for row in grades if not row['passed']],
            worst_modes=min(row['modes'] for row in grades),
            worst_hq=min(row['hq'] for row in grades),
            final=grades[-1], elapsed_seconds=elapsed,
            initial_state_sha256=_sha(loaded), final_state_sha256=_sha(branch['state']),
            final_rng_sha256=_sha(branch['state']['rng']),
            final_noise_step_calls=branch['state']['noise']['step_calls'],
            actual_adam_updates=receipt['actual_adam_updates'],
            optimizer_callbacks=receipt['optimizer_callbacks'],
            optimizer_rates=branch['dynamics'].get('optimizer_rates'),
            generated_source_sha256=sha(generated.encode()),
            selected=[row['selected'] for row in branch['dynamics']['corrections']])
        report['branches'][label] = full
        generated_hashes.append(full['generated_source_sha256'])
        torch.save(branch['state'], output / (label + '-update2410-state.pt'))
        with gzip.open(output / (label + '-raw.json.gz'), 'wt') as file:
            json.dump(dict(receipt=receipt, dynamics=branch['dynamics'], applied=branch['applied'],
                           policy=branch['policy']), file, allow_nan=False)
        print(json.dumps(dict(event='BRANCH_COMPLETE', branch=label, first_full_recovery_step=full['first_full_recovery_step'],
                              failing_steps=full['failing_steps'], final=full['final'], elapsed_seconds=elapsed)), flush=True)
    if len(set(generated_hashes)) != 1:
        raise RuntimeError('paired suffixes generated different host source')
    report['paired_final_rng_equal'] = (report['branches']['ordinary']['final_rng_sha256'] ==
                                         report['branches']['omitted_D_bank']['final_rng_sha256'])
    report['paired_generated_host_equal'] = True
    if not report['paired_final_rng_equal']:
        raise RuntimeError('paired suffixes consumed different RNG streams')
    write_json(output / 'result.json', report)
    print(json.dumps(dict(event='OMISSION_RECOVERY10_DONE',
                          conditioned_first_full_recovery=report['branches']['omitted_D_bank']['first_full_recovery_step'],
                          conditioned_failing_steps=report['branches']['omitted_D_bank']['failing_steps'])), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cold', type=Path, required=True)
    parser.add_argument('--hold', type=Path, required=True)
    parser.add_argument('--first-step', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(*(value.resolve() for value in (args.root, args.cold, args.hold,
                                        args.first_step, args.output)))


if __name__ == '__main__':
    main()
