"""Read certified Tier 1 failures before admitting the direction-only follow-up."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from experiments.forge.contracts import stable_hash


SOURCES = dict(
    optimistic='5cb95a39f799a493efd731c0c49960b8f7fa8bb8',
    confidence='4bf7d44e6a88faca6e132d19d4755370474ec1dd',
    anisotropic='ad29d3b475ec724a7f4b3c668ad54b814a00736f',
    sinkhorn='80d165fc3550625e08cfb7e696cf7e86adfa91ec',
    secant='ea194a7354c1dd2f1c86f08612cf592be9ced12a')


def inspect(main_repository, artifacts):
    failures, snapshots = {}, {}
    for track, source in SOURCES.items():
        report = Path(str(main_repository) + '-bcap-moonshot-' + track) / 'reports/forge' / ('bcap-moonshot-' + track)
        registration_bytes = (report / 'registration.json').read_bytes()
        registration = json.loads(registration_bytes)
        archive = artifacts / 'moonshots' / track
        progress = json.loads((archive / 'phase3-progress.json').read_bytes())
        assert progress['source_commit'] == source and registration['protocol_seed'] == 0
        tier1 = [item['task'] for item in registration['original_requirements'] if item['qualification_tier'] == 1]
        assert len(tier1) == 6
        request = progress['requests']['candidate']
        state_bytes = (archive / 'phase3-queue/queue/state.json').read_bytes()
        state = json.loads(state_bytes)
        snapshots[track] = state_bytes
        for task in tier1:
            matching = [job['result'] for job in state['jobs'].values()
                        if request in job['subscribers'] and job.get('result')
                        and any(row['task_id'] == task and row['gate_status'] == 'FAIL'
                                for row in job['result']['task_results'])]
            if not matching:
                continue
            assert len(matching) == 1
            result = matching[0]
            grading = result['raw']['grading']
            grade = grading['grades'][task]
            assert result['raw']['attempt_status'] == 'completed'
            assert result['candidate_revision'] == registration['arms']['candidate']['candidate_revision']
            assert grading['source_digest'] == registration['source_digest']
            assert grade['gate_status'] == 'FAIL'
            evaluator = grade['evaluator_result']
            if 'convergence' in evaluator:
                assert evaluator['convergence']['complete']
                completeness = evaluator['convergence']
            else:
                # Gaussian acquisition uses independent confirmed checkpoints,
                # rather than the sustained-suffix evaluator used by toy holds.
                assert task == 'gaussian1d_smoke' and grade['metrics']['step'] == 1000
                assert evaluator['confirmed_steps'] == [] and evaluator['first_confirmed_step'] is None
                completeness = dict(complete=True, consumed_updates=1000,
                    confirmed_steps=[], first_confirmed_step=None,
                    passing_observations=evaluator['passing_observations'])
            failures[track] = dict(source_commit=source, source_digest=registration['source_digest'],
                registration_sha256=hashlib.sha256(registration_bytes).hexdigest(),
                candidate_request=request, candidate_revision=result['candidate_revision'],
                task_id=task, original_tier=1, gate_status='FAIL', attempt_id=result['attempt_id'],
                result_hash=stable_hash(result), grading_raw_hash=grading['raw_hash'],
                metrics=grade['metrics'], completeness=completeness,
                queue_snapshot_sha256=hashlib.sha256(state_bytes).hexdigest())
            break
    missing = sorted(set(SOURCES) - set(failures))
    return dict(schema_version=1, observed_at=datetime.now(timezone.utc).isoformat(),
                status='SATISFIED' if not missing else 'NOT_READY',
                interpretation='A complete original Tier1 failure defeats the six-pass replacement rule; '
                               'all remaining study measurements and publications must still finish.',
                qualification_input=False, complete_tier1_failures=failures,
                candidates_without_defeating_failure=missing), snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-repository', type=Path, default=Path('/home/martyn/dev/ParticleGAN'))
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--record', type=Path)
    options = parser.parse_args()
    result, snapshots = inspect(options.main_repository.resolve(), options.artifacts.resolve())
    if options.record:
        assert result['status'] == 'SATISFIED', 'Conditional admission remains pending'
        assert not options.record.exists(), 'Keep the original admission receipt immutable'
        directory = options.record.parent / 'admission-queue-snapshots'
        directory.mkdir(parents=True, exist_ok=False)
        for track, content in snapshots.items():
            (directory / (track + '.json')).write_bytes(content)
        options.record.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], failures=list(result['complete_tier1_failures']),
                         missing=result['candidates_without_defeating_failure'])))


if __name__ == '__main__':
    main()
