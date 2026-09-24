"""Strict three-branch local gate for one frozen empirical-D refinement candidate.

Use the full capture-v2 sidecar, not its compact public subset: update 1380
must be present. The shared resume utility changes only host loop bounds and
restores full pre-step state. No warm/cold episode is launched by this script.
"""

from contextlib import contextmanager
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reports.toy100 import pr84_prediction_state_filter as replay
from reports.toy100.pr84_critic_refinement import METHOD, pr84_critic_refinement


BRANCHES = ((1324, 1335), (1380, 1395), (1530, 1545))
EXPECTED_CAPTURE = '37aa612bd3b3e1867a2d1508674158849ccf5940c907af76b1f8b061f4eb3e47'


def grade(points):
    okay = [point['grade']['modes'] == 8 and point['grade']['hq'] >= .9 for point in points]
    return dict(checks=len(points), passing_checks=sum(okay), pass_all=all(okay),
                failing_steps=[point['step'] for point, flag in zip(points, okay) if not flag],
                min_hq=min(point['grade']['hq'] for point in points),
                min_modes=min(point['grade']['modes'] for point in points))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    source_dir = args.output / 'source'
    source_dir.mkdir()
    diagnosis_file = args.capture / 'diagnosis.json'
    states_file = args.capture / 'selected-states.pt'
    diagnosis = json.loads(diagnosis_file.read_text())
    if (diagnosis['status'] != 'EXACT_REFERENCE_PARITY'
            or hashlib.sha256(states_file.read_bytes()).hexdigest() != EXPECTED_CAPTURE
            or diagnosis['selected_states_sha256'] != EXPECTED_CAPTURE):
        raise RuntimeError('required full capture-v2 source identity differs')
    states = torch.load(states_file, weights_only=True)
    original_rows = {row['step']: row for row in diagnosis['rows']}
    files = ('reports/toy100/pr84_critic_refinement_filter.py',
             'reports/toy100/pr84_critic_refinement.py',
             'reports/toy100/pr84_critic_relaxation.py',
             'particlegan/gan_loss.py',
             'particlegan/grad_regularizers.py',
             'reports/toy100/pr84_prediction_state_filter.py',
             'reports/toy100/pr84_opponent_prediction.py',
             'reports/toy100/pr84_smoothed_candidate.py',
             'reports/toy100/alternating_curvature_scratch.py',
             'reports/toy100/extra_adam_scratch.py',
             'benchmarks/locked_shared/mode_hold.py',
             'configs/toy100/constraints_simple_regularization.json')
    hashes = {}
    for name in files:
        raw = (ROOT/name).read_bytes()
        target = source_dir/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    declaration = dict(method=METHOD, scope='saved_state_local_diagnostic_only',
        shared_gate_eligible=False, scratch_optimizer_policy=METHOD,
        branches=BRANCHES, required_local_gate='all live G checkpoints in all three branches pass',
        live_threshold=dict(modes=8, hq=.9), variants=['original', 'refinement'],
        bank_batches=8, bank_batch_size=128, fit_max_iterations=40, fit_hard_max_closures=80,
        fit_selection='lowest finite same penalized D training loss',
        active_scope='mode_hold with zero input noise only', seed=0, noise_horizon=1200,
        nominal_rates=dict(g=.00425, d=.00425, prior=.0085),
        source=hashes, capture_sha256=hashlib.sha256(diagnosis_file.read_bytes()).hexdigest(),
        states_sha256=EXPECTED_CAPTURE, required_input='full capture-v2, including update1380',
        state_stage='phase2 after refined Dhat and bounded G, before EMA')
    (args.output/'declaration.json').write_text(json.dumps(declaration, indent=2)+'\n')
    print(json.dumps(dict(event='DECLARED', **declaration)), flush=True)
    config = json.loads((ROOT/files[-1]).read_text())

    @contextmanager
    def selected_context(*, task, prediction):
        # Reuse only the already parity-checked resume utility. Its boolean
        # selects this refinement controller, never opponent extrapolation.
        with pr84_critic_refinement(task=task, refinement=prediction) as value:
            yield value

    results = []
    with patch.object(replay.prediction_module, 'pr84_opponent_prediction', selected_context):
        for start, end in BRANCHES:
            variants = {}
            for name, switch in (('original', 'current'), ('refinement', 'predicted')):
                try:
                    value, _ = replay.run_local(config, states[start]['pre_step'], start=start, end=end,
                                                opponent=switch, source_dir=source_dir)
                except BaseException as error:
                    (args.output/f'{name}-{start}-{end}.error.json').write_text(json.dumps(
                        dict(status='ERROR', error=repr(error), fit_max_iterations=40, fit_hard_max_closures=80))+'\n')
                    raise
                value['opponent'] = name
                value['accepted_state_stage'] = declaration['state_stage']
                if name == 'original':
                    if value['accepted_states'][0]['accepted_state_sha256'] != replay.state_hash(
                            states[start]['post_bounded_g']):
                        raise RuntimeError('ordinary accepted state hash differs')
                    for point in value['points']:
                        if point['support'] != original_rows[point['step']]['stages']['bounded_joint']:
                            raise RuntimeError('ordinary continuation differs from archived original')
                    for index, record in enumerate(value['dynamics']['records']):
                        if record != dict(original_rows[start+index]['stages']['record'], outer_step=index+1):
                            raise RuntimeError('ordinary update record differs')
                else:
                    if value['rng_final_sha256'] != variants['original']['rng_final_sha256']:
                        raise RuntimeError('critic refinement advanced the training random streams')
                    if value['noise'] != variants['original']['noise']:
                        raise RuntimeError('critic refinement changed the training noise policy receipt')
                value['local_gate'] = grade(value['points'])
                variants[name] = value
                (args.output/f'{name}-{start}-{end}.json').write_text(json.dumps(value,allow_nan=False)+'\n')
                print(json.dumps(dict(event='BRANCH_DONE', variant=name,start=start,end=end,
                                      **value['local_gate'])),flush=True)
            results.append(dict(start=start,end=end,variants=variants))
    all_pass = all(row['variants']['refinement']['local_gate']['pass_all'] for row in results)
    result = dict(status='PASS' if all_pass else 'FAIL', local_gate_all_branches=all_pass,
                  warm_eligible=all_pass, shared_gate_eligible=False,
                  original_all_exact=True, declaration=declaration, branches=results)
    (args.output/'summary.json').write_text(json.dumps(result,allow_nan=False)+'\n')
    print(json.dumps(dict(event='DONE',status=result['status'],warm_eligible=all_pass)),flush=True)


if __name__ == '__main__':
    main()
