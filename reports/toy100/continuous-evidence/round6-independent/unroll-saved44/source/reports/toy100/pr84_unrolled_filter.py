"""Frozen 44-update rejection gate for the one-step total-G response."""

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
from reports.toy100.pr84_critic_refinement_filter import BRANCHES, EXPECTED_CAPTURE, grade
from reports.toy100.pr84_unrolled_candidate import METHOD, pr84_unrolled_candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    source_dir = args.output / 'source'
    source_dir.mkdir()
    states_file = args.capture / 'selected-states.pt'
    diagnosis_file = args.capture / 'diagnosis.json'
    diagnosis = json.loads(diagnosis_file.read_text())
    if (diagnosis['status'] != 'EXACT_REFERENCE_PARITY'
            or hashlib.sha256(states_file.read_bytes()).hexdigest() != EXPECTED_CAPTURE
            or diagnosis['selected_states_sha256'] != EXPECTED_CAPTURE):
        raise RuntimeError('full capture-v2 identity differs')
    states = torch.load(states_file, weights_only=True)
    rows = {row['step']: row for row in diagnosis['rows']}
    names = ('reports/toy100/pr84_unrolled_filter.py',
             'reports/toy100/pr84_unrolled_candidate.py',
             'reports/toy100/functional_b_cap.py',
             'reports/toy100/pr84_critic_refinement_filter.py',
             'reports/toy100/pr84_critic_refinement_capture.py',
             'reports/toy100/pr84_prediction_state_filter.py',
             'reports/toy100/pr84_opponent_prediction.py',
             'reports/toy100/pr84_smoothed_candidate.py',
             'reports/toy100/alternating_curvature_scratch.py',
             'reports/toy100/extra_adam_scratch.py',
             'reports/toy100/coverage_fixed_eval.py',
             'benchmarks/locked_shared/mode_hold.py',
             'benchmarks/locked_shared/mlp.py',
             'particlegan/gan_loss.py', 'particlegan/grad_regularizers.py',
             'configs/toy100/constraints_simple_regularization.json')
    hashes = {}
    for name in names:
        raw = (ROOT / name).read_bytes()
        target = source_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    declaration = dict(method=METHOD, scope='saved_state_local_diagnostic_only',
        shared_gate_eligible=False, branches=BRANCHES, variants=['original', 'unrolled'],
        required_local_gate='all 44 live checkpoints pass; no warm run on failure',
        thresholds=dict(modes=8, hq=.9), source=hashes,
        states_sha256=EXPECTED_CAPTURE,
        capture_sha256=hashlib.sha256(diagnosis_file.read_bytes()).hexdigest(),
        response='one virtual sharp penalized D step, full G derivative',
        virtual_metric='frozen post-real-D-Adam diagonal',
        persistent_d='original bounded Adam only',
        active_stencil='phase1 width frozen through phase2',
        stationary_point_scope='changes general-sum surrogate; no equilibrium-preservation theorem',
        seed=0, noise_horizon=1200, nominal_rates=dict(g=.00425,d=.00425,prior=.0085))
    (args.output / 'declaration.json').write_text(json.dumps(declaration, indent=2)+'\n')
    print(json.dumps(dict(event='DECLARED', **declaration)), flush=True)
    config = json.loads((ROOT / names[-1]).read_text())

    @contextmanager
    def selected(*, task, prediction):
        with pr84_unrolled_candidate(task=task, response='full' if prediction else 'off') as value:
            yield value

    results = []
    with patch.object(replay.prediction_module, 'pr84_opponent_prediction', selected):
        for start, end in BRANCHES:
            variants = {}
            for name, switch in (('original', 'current'), ('unrolled', 'predicted')):
                try:
                    value, _ = replay.run_local(config, states[start]['pre_step'],
                        start=start, end=end, opponent=switch, source_dir=source_dir)
                    value['opponent'] = name
                    if name == 'original':
                        assert value['accepted_states'][0]['accepted_state_sha256'] == replay.state_hash(
                            states[start]['post_bounded_g']), 'original first state mismatch'
                        for point in value['points']:
                            assert point['support'] == rows[point['step']]['stages']['bounded_joint']
                        for index, record in enumerate(value['dynamics']['records']):
                            assert record == dict(rows[start+index]['stages']['record'], outer_step=index+1)
                    else:
                        assert value['rng_final_sha256'] == variants['original']['rng_final_sha256']
                        assert value['noise'] == variants['original']['noise']
                    value['local_gate'] = grade(value['points'])
                    variants[name] = value
                    (args.output / f'{name}-{start}-{end}.json').write_text(
                        json.dumps(value, allow_nan=False)+'\n')
                    print(json.dumps(dict(event='BRANCH_DONE', variant=name, start=start, end=end,
                        **value['local_gate'])), flush=True)
                except BaseException as error:
                    (args.output / f'{name}-{start}-{end}.error.json').write_text(
                        json.dumps(dict(status='ERROR', error=repr(error)))+'\n')
                    raise
            results.append(dict(start=start, end=end, variants=variants))
    okay = all(row['variants']['unrolled']['local_gate']['pass_all'] for row in results)
    result = dict(status='PASS' if okay else 'FAIL', local_gate_all_branches=okay,
        warm_eligible=okay, shared_gate_eligible=False, original_all_exact=True,
        declaration=declaration, branches=results)
    (args.output / 'summary.json').write_text(json.dumps(result, allow_nan=False)+'\n')
    print(json.dumps(dict(event='DONE', status=result['status'], warm_eligible=okay)), flush=True)


if __name__ == '__main__':
    main()
