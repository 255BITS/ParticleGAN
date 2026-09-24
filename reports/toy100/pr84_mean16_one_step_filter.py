"""Reject mean-field averaging on independent original saved failures first.

Each point starts at the exact original archived pre-step state. This is a
44-point one-update filter, NOT a continuous 44-update candidate trajectory.
The unchanged one-bank full model/Adam update must replay exactly at every
point before its 16-bank comparison can be interpreted.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from reports.toy100 import pr84_finite_bank_adam_control as control
from reports.toy100 import pr84_finite_bank_vr_diagnostic as probe
from reports.toy100.pr84_critic_refinement_filter import BRANCHES, EXPECTED_CAPTURE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    source = set(probe.SOURCE_FILES) | {
        'reports/toy100/pr84_finite_bank_adam_control.py',
        'reports/toy100/pr84_mean16_one_step_filter.py',
        'reports/toy100/pr84_critic_refinement_filter.py'}
    hashes = {}
    for name in sorted(source):
        raw = (ROOT/name).read_bytes()
        out = args.output/'source'/name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    path = args.capture/'selected-states.pt'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED_CAPTURE
    diagnosis = json.loads((args.capture/'diagnosis.json').read_text())
    assert diagnosis['status'] == 'EXACT_REFERENCE_PARITY'
    states = torch.load(path, weights_only=True)
    steps = [step for start, end in BRANCHES for step in range(start, end+1)]
    declaration = dict(method='pr84_mean16_independent_saved_one_step_filter',
        sources=hashes, capture_sha256=EXPECTED_CAPTURE, steps=steps,
        scope='independent one-update rejection filter, not continued candidate state',
        banks_per_role=16, rates=dict(d=.00425,g=.00425,prior=.0085),
        moments='cloned original Adam advances once per role from each original state',
        extra_data='frozen scratch native-sized banks from copied original streams; no target geometry in update',
        required='all44 mean16 updates eight modes/HQ>=.9 and exact one-bank full model/Adam controls',
        stop_at_first_quality_failure=True, shared_gate_eligible=False)
    (args.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    print(json.dumps(dict(event='DECLARED',**declaration)),flush=True)
    recipe, _, _ = probe.declared_recipe(json.loads((ROOT/probe.SOURCE_FILES[-1]).read_text()))
    rows = []
    for step in steps:
        phases = states[step]
        before_hash = probe._sha(phases)
        rng = torch.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]), patch.object(probe,'STEP',step):
            pre, accepted_d, original = (phases[key] for key in
                ('pre_step','post_accepted_d','post_bounded_g'))
            generator, critic, prior = probe.fit.modules(accepted_d)
            a,b,_,_,actual_g = probe.fit.banks(pre,accepted_d,generator,prior)
            d_rows = a+b
            g_rows = probe.g_banks(accepted_d,generator,prior)
            assert probe._sha(g_rows[0]) == probe._sha(actual_g)
            native, native_state = control.mean_adam_update(pre,d_rows[:1],g_rows[:1],recipe)
            expected = {key:probe.fit.unwrapped(original[key]) if key in ('generator','critic')
                        else original[key] for key in native_state}
            if probe._sha(native_state) != probe._sha(expected):
                raise RuntimeError(f'original model/Adam replay differs at {step}')
            averaged, new_state = control.mean_adam_update(pre,d_rows,g_rows,recipe)
        if before_hash != probe._sha(phases) or not torch.equal(rng,torch.get_rng_state()):
            raise RuntimeError('one-step probe changed its inputs or caller randomness')
        passed = averaged['grade_after']['modes']==8 and averaged['grade_after']['hq']>=.9
        row = dict(step=step,native=native,mean16=averaged,pass_quality=passed,
            native_model_adam_exact=True,inputs_and_rng_unchanged=True,
            banks_sha256=probe._sha(dict(d=d_rows,g=g_rows)))
        rows.append(row)
        (args.output/f'update-{step}.json').write_text(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(dict(event='POINT_DONE',step=step,passed=passed,
            original=native['grade_after'],mean16=averaged['grade_after'])),flush=True)
        if not passed:
            torch.save(dict(phases=phases,banks=dict(d=d_rows,g=g_rows),mean16=new_state),
                       args.output/'first-failure.pt')
            break
    passed = len(rows)==len(steps) and all(row['pass_quality'] for row in rows)
    result = dict(status='PASS' if passed else 'FAIL',declaration=declaration,rows=rows,
        checks=len(rows),passing_checks=sum(row['pass_quality'] for row in rows),
        continuous_filter_eligible=passed,shared_gate_eligible=False)
    (args.output/'summary.json').write_text(json.dumps(result,allow_nan=False)+'\n')
    for name,digest in hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    print(json.dumps(dict(event='DONE',status=result['status'],checks=len(rows),
        passing=result['passing_checks'],continuous_filter_eligible=passed)),flush=True)


if __name__=='__main__':
    main()
