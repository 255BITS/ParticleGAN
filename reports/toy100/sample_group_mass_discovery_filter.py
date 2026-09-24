"""One frozen-bank, pure-data falsifier for mass-supported later discovery.

No GAN, optimizer, or noise policy runs here. The exact saved native data
stream is used to reproduce the archived false seven-group bootstrap, then
future ordinary real banks validate one region selected only from update
2403's rejected data. Other conditioned cases diagnose scope boundaries.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.mlp import SimpleMLPGenerator
from reports.toy100.sample_group_mass_discovery import (
    DELTA, AnytimeMassCertificate, propose_region)
from reports.toy100.sample_group_two_bank_memory import TwoBankFixedSupportMemory
from reports.toy100.sample_group_anchor import output_mm_step
from reports.toy100.coverage_fixed_eval import fixed_draw, score_support


SOURCES = (
    'reports/toy100/sample_group_mass_discovery.py',
    'reports/toy100/sample_group_mass_discovery_filter.py',
    'reports/toy100/sample_group_two_bank_memory.py',
    'reports/toy100/sample_group_anchor.py',
    'reports/toy100/coverage_fixed_eval.py',
    'benchmarks/locked_shared/mode_hold.py',
    'benchmarks/locked_shared/mlp.py',
    'particlegan/particle_prior.py',
)
MAX_VALIDATION_BANKS = 96


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def step_data(stream, means, *, d_source=None, n_particles=12):
    """Consume the native D-real, two prior-index, and G-real data draws."""
    if type(n_particles) is not int or n_particles < 1:
        raise ValueError('invalid particle count')
    real = mode_hold.sample_ring(means if d_source is None else d_source,
                                 mode_hold.BATCH, mode_hold.SIGMA, stream)
    torch.randint(0, n_particles, (mode_hold.BATCH,), generator=stream)
    torch.randint(0, n_particles, (mode_hold.BATCH,), generator=stream)
    mode_hold.sample_ring(means, mode_hold.BATCH, mode_hold.SIGMA, stream)
    return real


def clean_support(saved):
    with torch.random.fork_rng(devices=[]):
        generator = SimpleMLPGenerator(mode_hold.Z_DIM, mode_hold.HIDDEN,
                                       mode_hold.N_HIDDEN, 2)
        generator.load_state_dict({key.removeprefix('model.'): value
                                   for key, value in saved['generator'].items()})
        return generator(saved['prior']['z']).detach().double()


def grade_output(points, centers, means, *, step):
    target = torch.tensor(output_mm_step(points.double(), centers)['target'],
                          dtype=torch.float64)
    latent_index, output_noise = fixed_draw(step, target.float())
    grade = score_support(target.float(), latent_index, output_noise, means)
    return target, dict(modes=grade['modes'], hq=grade['hq'],
                        target=target.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--native-false', type=Path, required=True)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.state.read_bytes()
    saved = torch.load(args.state, weights_only=True, map_location='cpu')
    false = json.loads(args.native_false.read_text())
    code_root = args.code_root.resolve()
    def source_path(name):
        local = ROOT / name
        return local if local.is_file() else code_root / name
    if (sha(raw) != false['declaration']['input_file_sha256']
            or saved['noise']['step_calls'] != 2400
            or saved['noise_policy']['total_steps'] != 1200
            or len(saved['prior']['z']) != 12
            or false['confirmed_groups'] != 7
            or false['third_bank_rejected_mode0'] != false['third_bank_mode0_samples']):
        raise RuntimeError('exact archived native false confirmation fixture changed')
    for name, digest in false['declaration']['source'].items():
        if name in SOURCES and sha(source_path(name).read_bytes()) != digest:
            raise RuntimeError(f'native false-confirmation source changed: {name}')
    args.output.mkdir(parents=True)
    source = {name: sha(source_path(name).read_bytes()) for name in SOURCES}
    declaration = dict(scope='pure fixed-output/data filter; no model or optimizer updates',
        state_file_sha256=sha(raw), native_false_result_sha256=sha(args.native_false.read_bytes()),
        source=source, candidate_index=1, family_error_delta=DELTA,
        candidate_error_allocation=DELTA / 2,
        threshold='strict one equal particle mass, 1/12',
        max_validation_banks=MAX_VALIDATION_BANKS,
        validation='only fresh full-law native D banks after the proposal bank; all real rows count',
        expected_fixture='two D-only omitted-component banks confirm seven, third full bank proposes missing region',
        shared_gate_eligible=False)
    (args.output/'declaration.json').write_text(json.dumps(declaration, indent=2)+'\n')
    print(json.dumps(dict(event='DISCOVERY_DECLARED', declaration=declaration)), flush=True)

    means = mode_hold.ring_means()
    before_global = torch.random.get_rng_state().clone()
    stream = torch.Generator()
    stream.set_state(saved['rng']['data'])
    memory = TwoBankFixedSupportMemory(expected_first_bank_id=2401)
    first = step_data(stream, means, d_source=means[1:])
    second = step_data(stream, means, d_source=means[1:])
    pending = memory.observe(first, bank_id=2401)
    confirmed = memory.observe(second, bank_id=2402)
    if ([pending['status'], confirmed['status']] != ['PENDING_FIRST_BANK', 'CONFIRMED']
            or confirmed['confirmed_groups'] != 7
            or pending['bank_sha256'] != false['calls'][0]['sha256']
            or confirmed['bank_sha256'] != false['calls'][6]['sha256']):
        raise RuntimeError('conditioned two-bank native data replay differs')
    refs = torch.stack(memory.reference_centers)
    old_radius = memory.fixed_half_separation
    third = step_data(stream, means)
    third_sha = sha(third.detach().contiguous().numpy().tobytes())
    if third_sha != false['calls'][12]['sha256']:
        raise RuntimeError('third native full-law discovery bank differs')
    observed_third = memory.observe(third, bank_id=2403)
    region, proposal = propose_region(third, refs, old_radius, bank_id=2403)
    if (region is None or proposal['status'] != 'REGION_FROZEN'
            or observed_third['rejected'] != false['third_bank_rejected']):
        raise RuntimeError('known omitted support produced no disjoint discovery region')
    monitor = AnytimeMassCertificate(region, candidate_index=1,
                                     n_particles=len(saved['prior']['z']), delta=DELTA)
    validations = []
    for bank in range(1, MAX_VALIDATION_BANKS+1):
        real = step_data(stream, means)
        row = monitor.observe(real, bank_id=2403+bank)
        validations.append(row)
        if bank % 16 == 0 or row['admitted']:
            print(json.dumps(dict(event='MASS_VALIDATION', bank=bank,
                empirical=row['empirical_mass'], lower=row['lower_mass_bound'],
                threshold=row['threshold'], admitted=row['admitted'])), flush=True)
        if row['admitted']:
            break
    if not monitor.admitted:
        raise AssertionError('predeclared 96-bank false-seven fixture did not earn mass support')

    # An old-mode tail singleton is one rejected point, not a proposed region.
    tail_stream = torch.Generator()
    tail_stream.set_state(saved['rng']['data'])
    complete = TwoBankFixedSupportMemory(expected_first_bank_id=2401)
    complete.observe(step_data(tail_stream, means), bank_id=2401)
    full_confirmation = complete.observe(step_data(tail_stream, means), bank_id=2402)
    if full_confirmation['status'] != 'CONFIRMED' or full_confirmation['confirmed_groups'] != 8:
        raise RuntimeError('singleton control lacked complete old identities')
    tail_bank = step_data(tail_stream, means)
    ref8 = torch.stack(complete.reference_centers)
    outward = ref8[0] / ref8[0].norm()
    tail_bank[0] = (ref8[0] + 1.2 * complete.fixed_half_separation * outward).float()
    tail_region, tail_receipt = propose_region(tail_bank, ref8,
        complete.fixed_half_separation, bank_id=2403)
    if tail_region is not None or tail_receipt['rejected_count'] != 1:
        raise AssertionError('one conditioned old-mode tail was mistaken for new support')

    # With only opposite old centers, adjacent mode 1 is inside the old
    # radius. A rejected-only rule cannot discover it, regardless of mass.
    sparse_refs = means[[0,4]].double()
    sparse_radius = float(torch.linalg.vector_norm(sparse_refs[0]-sparse_refs[1])) / 2
    merge_stream = torch.Generator()
    merge_stream.set_state(saved['rng']['data'])
    new1 = step_data(merge_stream, means, d_source=means[1:2])
    merged_region, merged_receipt = propose_region(new1, sparse_refs, sparse_radius,
        bank_id=2401)
    diagnostic_new1 = int((torch.cdist(new1, means).argmin(1) == 1).sum())
    if (merged_region is not None or merged_receipt['rejected_count'] != 0
            or diagnostic_new1 != len(new1)):
        raise AssertionError('known 0/4→1 interior-merge falsifier changed')

    support = clean_support(saved)
    old_target, old_grade = grade_output(support, memory.centers(), means, step=2404)
    learned_centers = torch.cat((memory.centers(),
        torch.tensor(region.center, dtype=torch.float64)[None]), dim=0)
    new_target, new_grade = grade_output(old_target, learned_centers, means, step=2404)
    _, second_grade = grade_output(new_target, learned_centers, means, step=2404)
    if not torch.equal(torch.random.get_rng_state(), before_global):
        raise RuntimeError('pure discovery filter changed global training RNG')
    result = dict(status='PURE_MASS_FILTER_COMPLETE', declaration=declaration,
        fixture=dict(pending=pending, confirmed=confirmed,
                     third=observed_third, third_native_bank_sha256=third_sha),
        proposal=proposal, region=asdict(region),
        validation=dict(admitted=monitor.admitted, first_admitted_bank=monitor.banks,
                        total_real_samples=monitor.samples, rows=validations,
                        final_data_rng_sha256=sha(stream.get_state().numpy().tobytes())),
        singleton_tail=dict(status=tail_receipt['status'], rejected=tail_receipt['rejected_count'],
                            conditioned_one_old_mode_tail=True, no_birth=tail_region is None),
        old_0_4_to_new_1=dict(status=merged_receipt['status'],
                            true_mode_1_samples_diagnostic_only=diagnostic_new1,
                            rejected=merged_receipt['rejected_count'],
                            no_discovery=merged_region is None,
                            limitation='fixed old balls absorb a genuinely new interior component'),
        output_only=dict(old_seven_target=old_grade,
                         newly_admitted_eight_target_one_mm=new_grade,
                         newly_admitted_eight_target_two_mm=second_grade,
                         no_neural_or_optimizer_update=True,
                         grading_only_no_adaptive_mm_stop=True),
        statistical_scope='conditional on IID future real samples and a region frozen before validation; '
                          'one-sided Hoeffding with bankwise and candidatewise countable error spending',
        limitations=['mass-supported ball is not a mode/component theorem',
                     'rejected-only proposals cannot discover mass inside old balls',
                     'p<=1/12 components are not eligible under this conservative rule',
                     'output-space support change does not prove neural realization or stability'],
        global_training_rng_unchanged=True, shared_gate_eligible=False)
    (args.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(event='DISCOVERY_DONE', status=result['status'],
        first_admitted_bank=monitor.banks, samples=monitor.samples,
        lower=validations[-1]['lower_mass_bound'],
        output_before=(old_grade['modes'], old_grade['hq']),
        output_after_one=(new_grade['modes'], new_grade['hq']),
        output_after_two=(second_grade['modes'], second_grade['hq']),
        merge_blocked=True)), flush=True)


if __name__ == '__main__':
    main()
