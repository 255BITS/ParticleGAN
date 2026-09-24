"""Offline common-instance-noise Rp assay at the archived PR84 cold endpoint.

The endpoint is after final evaluation at update 1200. Its saved streams seed
native-shaped, disjoint D-fit, D-heldout and G assay banks. This is a copied-
critic counterfactual, not an exact native update replay or a training run.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.locked_shared import mode_hold
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
from reports.toy100 import common_instance_noise_falsifier as common
from reports.toy100 import pr84_critic_relaxation as prior_diagnostic
from reports.toy100.alternating_curvature_scratch import _metric, _rho
from reports.toy100.coverage_fixed_eval import fixed_draw, score_support


ARCHIVE = Path('/ml2/hypergan/ParticleGAN-continuous-learning/reports/toy100/continuous-evidence/pr88-cold-independent-audit/mode_hold-pr84-state.pt.gz')
ARCHIVE_RAW_SHA = 'ec212d3e4f3f03cb1f03d505014f1facd5919ca457f4b965315ba1939d866c0b'
WIDTH = 1.1279860476026753  # frozen 1325 native-real-bank rule; no refit or width search
OBSERVATION_SEED = 1058832626671652556  # frozen two-state v1 seed
STEP = 1201
SOURCE = (
    'reports/toy100/common_instance_noise_cold_endpoint.py',
    'reports/toy100/common_instance_noise_falsifier.py',
    'reports/toy100/pr84_critic_relaxation.py',
    'reports/toy100/alternating_curvature_scratch.py',
    'reports/toy100/coverage_fixed_eval.py',
    'benchmarks/locked_shared/mode_hold.py',
    'benchmarks/locked_shared/mlp.py',
    'particlegan/gan_loss.py',
    'particlegan/grad_regularizers.py',
    'configs/toy100/constraints_simple_regularization.json',
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def saved_view(snapshot):
    return {
        'generator': snapshot['generator'], 'critic': snapshot['critic'],
        'prior': snapshot['prior'], 'optimizer_g': snapshot['optimizer_g'],
        'optimizer_d': snapshot['optimizer_d'],
        'noise': {'output_sigma': snapshot['noise_policy']['output_sigma'],
                  'input_sigma': snapshot['noise_policy']['input_sigma']},
        'rng': {'data': snapshot['data_stream'], 'torch': snapshot['torch_rng']},
    }


def banks(snapshot, generator, prior):
    """Sequential native-shaped draws, all from the saved post-evaluation streams."""
    stream = torch.Generator().set_state(snapshot['data_stream'])
    sigma = float(snapshot['noise_policy']['output_sigma'])
    d_rows, g_rows = [], []
    before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.set_rng_state(snapshot['torch_rng'])
        def d_draw():
            real = mode_hold.sample_ring(mode_hold.ring_means(), mode_hold.BATCH,
                                         mode_hold.SIGMA, stream)
            latent, indices = prior.sample(mode_hold.BATCH, generator=stream)
            clean = generator(latent)
            noise = torch.randn_like(clean)
            return {'real': real, 'fake': clean + sigma*noise,
                    'indices': indices, 'noise': noise}
        def g_draw():
            latent, indices = prior.sample(mode_hold.BATCH, generator=stream)
            clean = generator(latent)
            noise = torch.randn_like(clean)
            real = mode_hold.sample_ring(mode_hold.ring_means(), mode_hold.BATCH,
                                         mode_hold.SIGMA, stream)
            return {'real': real, 'indices': indices, 'noise': noise, 'sigma': sigma}
        d_rows = [d_draw() for _ in range(16)]
        g_rows = [g_draw() for _ in range(9)]
    if not torch.equal(torch.get_rng_state(), before):
        raise AssertionError('bank construction modified global RNG')
    def join(rows):
        return {'real': torch.cat([r['real'] for r in rows]),
                'fake': torch.cat([r['fake'] for r in rows])}
    return join(d_rows[:8]), join(d_rows[8:]), g_rows


def cloned_g_proposal(saved, critic, bank, gan):
    generator, _, prior = prior_diagnostic.modules(saved)
    optimizer = prior_diagnostic.g_optimizer(generator, prior, saved['optimizer_g'])
    params = list(generator.parameters()) + list(prior.parameters())
    base = [p.detach().clone() for p in params]
    clean0 = generator(prior.z).detach().clone()
    observation_before = float(common.common_g_loss(generator, prior, critic, bank, gan).detach())
    sharp_before = float(common.native_sharp_g_loss(generator, prior, critic, bank, gan).detach())
    optimizer.zero_grad()
    common.common_g_loss(generator, prior, critic, bank, gan).backward()
    first = [p.grad.detach().clone() for p in params]
    if not all(torch.isfinite(g).all() for g in first):
        raise FloatingPointError('nonfinite common-channel G gradient')
    optimizer.step()
    proposed = [p.detach().clone() for p in params]
    clean_unbounded = generator(prior.z).detach().clone()
    metric = _metric(optimizer)
    second = torch.autograd.grad(common.common_g_loss(generator, prior, critic, bank, gan), params)
    rho = _rho(base, proposed, first, second, metric)
    factor = min(1.0, .25/rho) if rho else 1.0
    with torch.no_grad():
        for p, old, new in zip(params, base, proposed):
            p.copy_(torch.lerp(old, new, factor) if factor < 1 else new)
        clean1 = generator(prior.z).detach().clone()
    observation_after = float(common.common_g_loss(generator, prior, critic, bank, gan).detach())
    sharp_after = float(common.native_sharp_g_loss(generator, prior, critic, bank, gan).detach())
    expected_steps = [float(saved['optimizer_g']['state'][index]['step']) + 1
                      for group in saved['optimizer_g']['param_groups']
                      for index in group['params']]
    actual_steps = [float(optimizer.state[p]['step']) for p in params]
    if actual_steps != expected_steps:
        raise AssertionError('cloned G and prior Adam did not advance exactly once')
    means = mode_hold.ring_means()
    distances = torch.cdist(clean0, means)
    owners = distances.argmin(1)
    counts = torch.bincount(owners, minlength=len(means))
    missing = sorted(set(range(len(means))) - set(owners.tolist()))
    if len(missing) != 1:
        raise AssertionError(f'cold acquisition requires exactly one genuinely empty mode: {missing}')
    missing_mode = missing[0]
    target = means[missing_mode]
    particle_geometry = []
    for row, owner in enumerate(owners.tolist()):
        direction = target - means[owner]
        direction = direction / direction.norm()
        delta = clean1[row] - clean0[row]
        raw_delta = clean_unbounded[row] - clean0[row]
        before = float((clean0[row] - means[owner]) @ direction)
        after = float((clean1[row] - means[owner]) @ direction)
        particle_geometry.append({
            'particle': row, 'owner': owner, 'owner_count': int(counts[owner]),
            'surplus_donor': bool(counts[owner] > 1),
            'clean_before': clean0[row].tolist(), 'clean_after': clean1[row].tolist(),
            'raw_displacement': raw_delta.tolist(), 'accepted_displacement': delta.tolist(),
            'raw_missing_chord_progress': float(raw_delta @ direction),
            'accepted_missing_chord_progress': float(delta @ direction),
            'beyond_owner_center_before': before, 'beyond_owner_center_after': after,
            'owner_center_distance_before': float((clean0[row]-means[owner]).norm()),
            'owner_center_distance_after': float((clean1[row]-means[owner]).norm()),
            'missing_center_distance_before': float((clean0[row]-target).norm()),
            'missing_center_distance_after': float((clean1[row]-target).norm()),
        })
    indices, noise = fixed_draw(STEP, clean1)
    return {
        'observation_game_g_loss_before_after': [observation_before, observation_after],
        'original_sharp_game_g_loss_before_after_diagnostic': [sharp_before, sharp_after],
        'rho': rho, 'accepted_factor': factor, 'adam_moment_step_after': actual_steps[0],
        'clean_output_rms_move': float((clean1-clean0).square().sum(-1).mean().sqrt()),
        'nearest_clean_counts_before': counts.tolist(), 'missing_mode': missing_mode,
        'particle_geometry_posthoc_only': particle_geometry,
        'grade_before': score_support(clean0, indices, noise, means),
        'grade_after': score_support(clean1, indices, noise, means),
    }


def run(output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    compressed = ARCHIVE.read_bytes()
    raw = gzip.decompress(compressed)
    if sha(raw) != ARCHIVE_RAW_SHA:
        raise AssertionError('archived PR84 cold snapshot raw hash mismatch')
    with torch.random.fork_rng(devices=[]):
        snapshot = torch.load(io.BytesIO(raw), weights_only=True, map_location='cpu')
    saved = saved_view(snapshot)
    original_hash = prior_diagnostic.state_hash(snapshot)
    outer_rng = torch.get_rng_state().clone()
    config = json.loads((ROOT / SOURCE[-1]).read_text())
    recipe, _, _ = declared_recipe(config)
    gan, regularizer = recipe.make_loss(), recipe.make_gradient_penalty()
    if gan.mode != 'rp' or gan.loss_type != 'logistic' or regularizer.arm != 'b_cap':
        raise AssertionError('wrong original objective')
    with torch.random.fork_rng(devices=[]):
        generator, critic, prior = prior_diagnostic.modules(saved)
    with torch.no_grad():
        clean = generator(prior.z)
        means = mode_hold.ring_means()
        counts = torch.bincount(torch.cdist(clean, means).argmin(1), minlength=8)
    if counts.tolist() != [2,1,2,2,1,2,0,2]:
        raise AssertionError('not the genuine missing-mode cold endpoint')
    train, heldout, g_banks = banks(snapshot, generator, prior)
    observation_rng = torch.Generator().manual_seed(OBSERVATION_SEED)
    noisy_train = common.bank_with_instance_noise(train, WIDTH, observation_rng)
    noisy_heldout = common.bank_with_instance_noise(heldout, WIDTH, observation_rng)
    g_augmented = [common.g_bank_with_instance_noise(bank, WIDTH, observation_rng)
                   for bank in g_banks]
    declaration = {
        'status': 'DECLARED_BEFORE_FIT', 'scope': 'offline copied critic/G assay from exact PR84 cold1200 post-final-evaluation state; not native update replay',
        'candidate_game': 'paired Rp logistic D and nonsaturating G losses on common Gaussian-noised real/fake observations; original b_cap on noisy D inputs',
        'not_logit_convolution': True, 'native_noise_clock_unchanged': True,
        'width': WIDTH, 'width_source': 'frozen 1325 first real native bank MST minimum separation /2, shared with archived warm assay',
        'observation_seed': OBSERVATION_SEED, 'critic_fit': 'one copied-D 40-iteration/80-closure fixed-bank L-BFGS; best finite D training loss',
        'bank_protocol': 'from saved post-final-eval data and torch RNG; 16 sequential D banks of 128 (8 fit, 8 heldout), then 9 separate G banks of 128; no overlap; additional instance noise from frozen independent stream',
        'generator_assay': 'nine cloned one-step G/prior Adam proposals from the same saved moments, one actual-next-shaped and eight reserved banks',
        'geometry': 'missing true nearest-mode cell 6; all-particle raw and accepted chord projections posthoc only; no center in training',
        'archived_cold_raw_sha256': sha(raw), 'archived_cold_gzip_sha256': sha(compressed),
        'source_sha256': {name: sha((ROOT/name).read_bytes()) for name in SOURCE},
        'original_state_hash': original_hash,
        'input_bank_sha256': prior_diagnostic.state_hash({'train': train, 'heldout': heldout, 'g': g_banks}),
        'observed_counts': counts.tolist(),
    }
    output.mkdir(parents=True)
    for name in SOURCE:
        destination = output / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
    (output/'declaration.json').write_text(json.dumps(declaration, indent=2)+'\n')
    print(json.dumps({'event':'DECLARED','counts':counts.tolist(),'width':WIDTH}),flush=True)
    started = time.perf_counter()
    with torch.random.fork_rng(devices=[]):
        metric = prior_diagnostic.saved_metric(saved['optimizer_d'])
        def evaluate(bank):
            total, logistic, cap = prior_diagnostic.d_loss(critic, bank, gan, regularizer, STEP)
            return {'total':float(total.detach()),'logistic':float(logistic.detach()),'b_cap':float(cap.detach())}
        d_before = {'train':evaluate(noisy_train),'heldout':evaluate(noisy_heldout)}
        fit = prior_diagnostic.relax(critic, noisy_train, gan, regularizer, STEP, metric)
        d_after = {'train':evaluate(noisy_train),'heldout':evaluate(noisy_heldout)}
        rows = [cloned_g_proposal(saved, critic, bank, gan) for bank in g_augmented]
    if prior_diagnostic.state_hash(snapshot) != original_hash or not torch.equal(torch.get_rng_state(),outer_rng):
        raise AssertionError('saved state or global RNG mutated')
    geometry = [entry for row in rows for entry in row['particle_geometry_posthoc_only']]
    donors = [entry for entry in geometry if entry['surplus_donor']]
    result = {
        'status':'COMPLETE_OFFLINE_ASSAY','declaration':declaration,
        'd_before':d_before,'d_after':d_after,'fit':fit,
        'fitted_critic_sha256':prior_diagnostic.state_hash(critic.state_dict()),
        'g_rows':rows,
        'summary':{
            'heldout_d_loss_improved':d_after['heldout']['total']<d_before['heldout']['total'],
            'g_bank_count':len(rows), 'g_after_missed_mode_count':[r['grade_after']['modes'] for r in rows],
            'g_after_hq':[r['grade_after']['hq'] for r in rows],
            'all_particle_accepted_missing_chord_progress': [e['accepted_missing_chord_progress'] for e in geometry],
            'surplus_accepted_missing_chord_progress': [e['accepted_missing_chord_progress'] for e in donors],
            'largest_surplus_accepted_missing_chord_progress':max(e['accepted_missing_chord_progress'] for e in donors),
            'surplus_over_point07_chord_progress':sum(e['accepted_missing_chord_progress']>.07 and e['beyond_owner_center_after']>.07 for e in donors),
            'input_state_and_global_rng_unchanged':True,'seconds':time.perf_counter()-started,
        },
    }
    (output/'result.json').write_text(json.dumps(result,allow_nan=False)+'\n')
    (output/'summary.json').write_text(json.dumps(result['summary'],indent=2)+'\n')
    print(json.dumps({'event':'DONE','summary':result['summary']}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        run(args.output)
    except BaseException as error:
        if args.output.is_dir():
            (args.output/'error.json').write_text(json.dumps({'status':'ERROR',
                'error':repr(error)},indent=2)+'\n')
        raise


if __name__=='__main__':
    main()
