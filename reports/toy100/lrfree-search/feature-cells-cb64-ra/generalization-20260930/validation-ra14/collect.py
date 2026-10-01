"""Independent validity and unchanged quality verdicts for the current API."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from freeze import HARNESS, PACKAGE, ROOT, sha, verify
from run_screen import ENV, GPU_UUID, TASKS

OLD_SCREENS = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/validation-cb64-ra11/screens')


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def original_validator():
    sys.path.insert(0,str(OLD_SCREENS))
    spec = importlib.util.spec_from_file_location('_unchanged_ra11_collector',OLD_SCREENS / 'collect.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.pop(0)
    return module


def selection_contract(output, plan, reasons):
    import os
    os.environ.update(ENV)
    sys.path.insert(0,str(PACKAGE))
    import torch
    torch.set_num_threads(1)
    from particlegan.feature_cells import population_policy
    from particlegan.output_moments import MAX_RANK
    state = torch.load(output / 'final-state.pt',map_location='cpu',weights_only=False)['trainer']
    if state.get('recipe', {}).get('reopen_guard') != 'settled':
        reasons.append('final checkpoint does not use the declared settled re-open guard')
    guard = state.get('reopen_guard')
    if not isinstance(guard, dict) or guard.get('schema') != 1:
        reasons.append('final checkpoint lacks the typed settled re-open guard state')
    selected = state.get('backend_selection')
    if not isinstance(selected,dict):
        reasons.append('final checkpoint lacks backend-selection metadata')
        return None
    shape = selected.get('output_shape')
    if not isinstance(shape,list) or not shape or any(type(d) is not int or d <= 0 for d in shape):
        reasons.append('final checkpoint has no valid frozen output shape')
        return selected
    import math
    width = math.prod(shape)
    population = population_policy(plan['num_particles'])
    actual = 'feature_cells' if population['finite_resolution_feasible'] and width <= MAX_RANK else 'knn'
    factor = .25 if actual == 'feature_cells' else 1.
    if selected.get('actual_backend') != actual or selected.get('generator_noise_factor') != factor:
        reasons.append('selected backend or calibration disagrees with the declared capability rule')
    if selected.get('requested_backend') != 'auto' or selected.get('population_policy') != population:
        reasons.append('requested backend or finite-population metadata differs')
    if selected.get('raw_output_width') != width or selected.get('moment_rank_bound') != MAX_RANK:
        reasons.append('checkpoint output frame metadata is inconsistent')
    mapping = selected.get('rate_mapping')
    expected_rates = [[item['rate'] for item in row] for row in mapping]
    if state['initial_lrs'] != expected_rates:
        reasons.append('checkpoint optimizer bases disagree with selected rate mapping')
    for row in mapping:
        for item in row:
            wanted = factor if item['role'] in ('generator','noise') else 1.
            if item['factor'] != wanted or item['rate'] != item['base_rate'] * wanted:
                reasons.append('selected calibration changes the wrong optimizer role')
    if state['completed_steps'] != plan['steps']:
        reasons.append('checkpoint completed budget differs from original')
    return dict(selection=selected,surprise=state.get('surprise'),reopen_guard=guard,
        birth_death_counters=(state.get('birth_death') or {}).get('counters'))


def collect(task):
    output = ROOT / 'runs' / task
    reasons = []
    validator = original_validator()
    plan = validator.task_plan(task)
    result = read(output / 'result.json')
    execution = read(output / 'execution-receipt.json')
    integrity = verify()
    rows = jsonl(output / 'metrics.jsonl')
    if result.get('status') not in ('PASS','FAIL'):
        reasons.append('original quality runner did not complete a PASS/FAIL judgment')
    for key in ('source_integrity_before','source_integrity_after'):
        if execution.get(key,{}).get('status') != 'VALID':
            reasons.append(f'{key} is not VALID')
    if execution.get('process_exit_code') != 0 or execution.get('resources',{}).get('gpu_uuid') != GPU_UUID:
        reasons.append('execution failed or was not on the original GPU0')
    header = result.get('header',{})
    if header.get('package_sha256') != integrity['package_sha256']:
        reasons.append('quality header package digest differs from the frozen candidate')
    if header.get('device') != 'cuda:0' or header.get('cuda_visible_devices') != '0':
        reasons.append('quality header is not physical GPU0')
    expected_options = dict(eval_output_noise=True,strict_streams=True,save_final_state=True,diagnostics=True,
        evaluation_generate='indexed',serial_backward_argument=True,initialization='batch_feature_zero',
        image_prior_perturb=False,ring_frozen_control=False)
    if header.get('options') != expected_options:
        reasons.append('quality runner options differ from the original noisy-primary contract')
    if result.get('stream_deviations') != 0:
        reasons.append('original stream transaction deviated')
    if result.get('completed_steps') != plan['steps']:
        reasons.append('original full update budget was not completed')
    if [row.get('step') for row in rows] != plan['observation_steps']:
        reasons.append('observation schedule differs from the original')
    if any('construction RNG' in warning or 'initial receipt unavailable' in warning
           for warning in result.get('warnings',[])):
        reasons.append('original construction fixture warning')
    native = validator.validate_native(output,task,result,reasons) if task in validator.NATIVE else None
    mechanisms = selection_contract(output,plan,reasons) if (output / 'final-state.pt').exists() else None
    if mechanisms is None:
        reasons.append('final checkpoint missing or not auditable')
    receipt = dict(task=task,quality_status=result.get('status'),
        evidence_validity='INVALID' if reasons else 'VALID',
        acceptance_status=result.get('status') if not reasons else 'INVALID',reasons=reasons,
        original_plan=plan,source_integrity=integrity,result_sha256=sha(output / 'result.json'),
        final=result.get('final'),first_arrival=result.get('first_arrival'),final_streak=result.get('final_streak'),
        passing_checks=result.get('passing_checks'),native=native,mechanisms=mechanisms,
        peak_allocated_gpu_mib=execution.get('peak_allocated_gpu_mib'),
        peak_reserved_gpu_mib=execution.get('peak_reserved_gpu_mib'),wall_seconds=execution.get('wall_seconds'))
    path = output / 'acceptance-receipt.json'
    path.write_text(json.dumps(receipt,indent=2,sort_keys=True,default=str) + '\n')
    print(json.dumps(dict(task=task,quality=receipt['quality_status'],validity=receipt['evidence_validity'],
        acceptance=receipt['acceptance_status'],reasons=reasons,
        backend=None if mechanisms is None else mechanisms['selection']['actual_backend'])),flush=True)
    return not reasons


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,choices=TASKS)
    args = parser.parse_args()
    raise SystemExit(0 if collect(args.task) else 1)
