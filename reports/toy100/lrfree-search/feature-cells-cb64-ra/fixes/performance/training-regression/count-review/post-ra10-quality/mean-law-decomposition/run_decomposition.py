"""One final RA10 saved-cloud decomposition; no actions, draws or quality score."""
import __future__
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from bindings import (HERE, ROOT, RUN, PACKAGE, PLAN, HEAD, AFFINE, STATE_HASH,
                      NUMERIC_INPUTS, sha, read, write_new, verify, function_node)


def log(message):
    print(datetime.now(timezone.utc).isoformat(), message, flush=True)


def extract(path, name, namespace, affine_only=False):
    node = function_node(path, name, affine_only=affine_only)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


def numpy_rng_hash(np):
    name, keys, position, has_gaussian, cached_gaussian = np.random.get_state()
    digest = hashlib.sha256()
    digest.update(repr((name, position, has_gaussian, cached_gaussian)).encode())
    digest.update(keys.tobytes())
    return digest.hexdigest()


def analyze(saved, clean, noisy, target, torch, F, np, state_hash):
    from particlegan.feature_cells import FeatureCellSnapshot
    from particlegan.mean_transport import freeze_moment, observe_view, group_means

    weights = saved['models']
    bd = saved['birth_death']
    assert saved['completed_steps'] == 7000 and bd['backend_schema'] == 9
    assert saved['schema'] == 5
    settings = bd['settings']
    assert settings['cells'] == 128 and settings['rank'] == 8 and settings['chunk'] == 256
    zf, ze, fifo = weights['prior']['z'], weights['ema_prior']['z'], bd['reservoir']
    assert zf.shape == ze.shape == (20000, 2) and fifo.shape == (20000, 2)
    assert bd['fill'] == len(fifo) and bd['last']['step'] == 7000
    assert len(clean) == len(noisy) == len(target) == 100000
    assert clean.shape == noisy.shape == target.shape == (100000, 2)
    assert clean.dtype == noisy.dtype == target.dtype == zf.dtype == fifo.dtype
    assert all(bool(torch.isfinite(v).all()) for v in (zf, ze, fifo, clean, noisy, target))
    assert all(weights[role]['weight'].shape == (2, 2) and
               weights[role]['bias'].shape == (2,) for role in ('G', 'ema_G'))
    raw = extract(AFFINE, 'raw', dict(F=F, weights=weights), affine_only=True)
    head = extract(HEAD, 'head_features', dict(torch=torch, F=F))
    log('evaluating immutable current-D head and raw affine maps; no constructors')
    fast_raw, anchor_raw = raw(zf, 'G'), raw(ze, 'ema_G')
    real_features = head(fifo, weights['D']).double()
    fast_features = head(fast_raw, weights['D']).double()
    anchor_features = head(anchor_raw, weights['D']).double()
    assert real_features.shape == fast_features.shape == anchor_features.shape == (20000, 128)
    rng = saved['cpu_rng']
    assert rng.dtype == torch.uint8 and rng.ndim == 1 and rng.device.type == 'cpu'
    stream = torch.Generator(device='cpu')
    stream.set_state(rng.clone())
    private_before = state_hash(stream.get_state())
    log('fitting exactly one current128/rank8 chart from saved FIFO and cloned CPU RNG')
    snapshot = FeatureCellSnapshot.fit(real_features, generator=stream, cells=128, rank=8, chunk=256)
    private_after = state_hash(stream.get_state())
    topology = snapshot._mass_topology()
    chart = dict(requested_cells=128, requested_rank=8, actual_cells=snapshot.cells,
        effective_rank=snapshot.rank, groups=snapshot.mass_groups,
        topology=snapshot.mass_topology, duplicate_fraction=snapshot.duplicate_fraction,
        valid_metric=snapshot.valid_metric, fits=1,
        saved_CPU_rng_sha256=state_hash(rng), private_rng_before_sha256=private_before,
        private_rng_after_fit_sha256=private_after,
        count_partition=snapshot.count_partition,
        geometry_sha256=state_hash(dict(mean=snapshot.mean, scale=snapshot.scale,
            basis=snapshot.basis, centers=snapshot.centers, cell_scale=snapshot.cell_scale,
            reference_counts=snapshot.reference_counts, null_scores=snapshot.null_scores,
            count_boundary=snapshot.count_boundary, cell_group_ids=topology)),
        historical_reaction_chart_reconstructed=False,
        historical_saved_topology=bd['last']['mass_topology'])
    fixed, invalid = freeze_moment(snapshot, real_features[0::2], anchor_features)
    if fixed is None:
        return dict(status='COMPLETE_DESCRIPTIVE_CHART_VETO', chart=chart,
                    invalid_moment_reason=invalid, population_measurements=None)
    gcount = snapshot.mass_groups
    weights_even = fixed.weights
    vf = observe_view(snapshot, fast_features, coordinates=zf)
    va = observe_view(snapshot, anchor_features, coordinates=ze)
    vr = observe_view(snapshot, real_features, coordinates=fifo)
    legal = vf.eligible & va.eligible & (vf.groups == va.groups)
    unwanted = ~legal
    real_psi, anchor_psi = fixed.psi(vr.metric, vr.groups), fixed.psi(va.metric, va.groups)
    populations = {}

    def population(name, points, metric, groups, psi, mask=None):
        if mask is not None:
            points, metric, groups, psi = points[mask], metric[mask], groups[mask], psi[mask]
        points = points.double()
        means, counts = group_means(psi, groups, gcount)
        raw_means, raw_counts = group_means(points, groups, gcount)
        assert torch.equal(counts, raw_counts)
        covariance_sum = torch.zeros(gcount, dtype=torch.float64)
        covariance_sum.index_add_(0, groups, (points - raw_means[groups]).square().sum(1))
        covariance_trace = covariance_sum / counts.clamp_min(1)
        standardized_radius = ((metric - fixed.centers[groups]) / fixed.scales[groups, None]).norm(dim=1)
        clipped_counts = torch.bincount(groups[standardized_radius > fixed.radius], minlength=gcount)
        out = dict(rows=len(points), counts=counts, feature_mean=means,
                   raw_mean=raw_means, raw_covariance_trace=covariance_trace,
                   clipped_counts=clipped_counts)
        populations[name] = out
        return out

    even = torch.arange(len(fifo)).remainder(2) == 0
    odd = ~even
    population('real_even_all', fifo, vr.metric, vr.groups, real_psi, even)
    population('real_odd_all', fifo, vr.metric, vr.groups, real_psi, odd)
    population('real_even_inside', fifo, vr.metric, vr.groups, real_psi, even & vr.eligible)
    population('real_odd_inside', fifo, vr.metric, vr.groups, real_psi, odd & vr.eligible)
    population('anchors_A', anchor_raw, va.metric, va.groups, anchor_psi)
    population('anchors_L', anchor_raw, va.metric, va.groups, anchor_psi, legal)
    population('anchors_U', anchor_raw, va.metric, va.groups, anchor_psi, unwanted)
    assert torch.equal(populations['anchors_A']['counts'], fixed.ema_counts)
    assert torch.allclose(populations['anchors_A']['feature_mean'], fixed.ema_means, atol=1e-12, rtol=1e-12)
    log('evaluating saved clean/noisy100k pair and original saved target on the single fixed chart')
    fc, fy, ft = (head(v, weights['D']).double() for v in (clean, noisy, target))
    vc = observe_view(snapshot, fc, coordinates=clean)
    vy = observe_view(snapshot, fy, coordinates=noisy)
    vt = observe_view(snapshot, ft, coordinates=target)
    psi_c = fixed.psi(vc.metric, vc.groups)
    psi_y_natural = fixed.psi(vy.metric, vy.groups)
    # The original clean group controls BOTH bin and clipping frame for the paired increment.
    psi_y_clean_frame = fixed.psi(vy.metric, vc.groups)
    population('saved_clean_C', clean, vc.metric, vc.groups, psi_c)
    population('saved_noisy_Y_natural', noisy, vy.metric, vy.groups, psi_y_natural)
    population('saved_noisy_Y_clean_frame', noisy, vy.metric, vc.groups, psi_y_clean_frame)
    population('saved_real_target', target, vt.metric, vt.groups, fixed.psi(vt.metric, vt.groups))
    noise_feature, noise_counts = group_means(psi_y_clean_frame - psi_c, vc.groups, gcount)
    noise_raw, noise_raw_counts = group_means(noisy.double() - clean.double(), vc.groups, gcount)
    assert torch.equal(noise_counts, noise_raw_counts)
    changed = vc.groups != vy.groups
    transitions = torch.bincount(vc.groups * gcount + vy.groups, minlength=gcount*gcount).reshape(gcount, gcount)
    assert torch.equal(transitions.sum(1), noise_counts)
    A, L, U = (populations['anchors_' + name] for name in ('A', 'L', 'U'))
    C = populations['saved_clean_C']
    Yc, Yn = populations['saved_noisy_Y_clean_frame'], populations['saved_noisy_Y_natural']
    real_even = populations['real_even_all']
    real_inside = populations['real_even_inside']
    fL, fU = L['counts'].double() / A['counts'], U['counts'].double() / A['counts']
    assert torch.equal(L['counts'] + U['counts'], A['counts'])
    mixture_feature = fL[:, None] * L['feature_mean'] + fU[:, None] * U['feature_mean']
    mixture_raw = fL[:, None] * L['raw_mean'] + fU[:, None] * U['raw_mean']
    feature_identity_error = float((A['feature_mean'] - mixture_feature).abs().max())
    raw_identity_error = float((A['raw_mean'] - mixture_raw).abs().max())
    assert feature_identity_error <= 1e-10 and raw_identity_error <= 1e-10
    rA = fixed.even_means - A['feature_mean']
    rL = fixed.even_means - L['feature_mean']
    rU = fixed.even_means - U['feature_mean']
    rL_inside = real_inside['feature_mean'] - L['feature_mean']
    contribution_L, contribution_U = fL[:, None] * rL, fU[:, None] * rU
    assert float((rA - contribution_L - contribution_U).abs().max()) <= 1e-10
    delta_anchor_clean = C['feature_mean'] - A['feature_mean']
    rC = fixed.even_means - C['feature_mean']
    paired_identity = Yc['feature_mean'] - C['feature_mean'] - noise_feature
    raw_paired_identity = Yc['raw_mean'] - C['raw_mean'] - noise_raw
    assert float(paired_identity.abs().max()) <= 1e-10
    assert float(raw_paired_identity.abs().max()) <= 1e-10
    rYc = fixed.even_means - Yc['feature_mean']
    assert float((rYc - (rC-noise_feature)).abs().max()) <= 1e-10

    def vector_comparison(left, right, valid):
        w = weights_even * valid.double()
        dot = float((w * (left*right).sum(1)).sum())
        left2 = float((w * left.square().sum(1)).sum())
        right2 = float((w * right.square().sum(1)).sum())
        return dict(covered_even_mass=float(w.sum()), weighted_dot=dot,
            left_weighted_norm=math.sqrt(left2), right_weighted_norm=math.sqrt(right2),
            weighted_cosine=dot / math.sqrt(left2 * right2) if left2*right2 > 0 else None,
            weights_renormalized=False)

    aggregate_populations = {}
    for name, pop in populations.items():
        valid = pop['counts'] > 0
        w = weights_even * valid.double()
        feature_energy = float((w * (fixed.even_means-pop['feature_mean']).square().sum(1)).sum())
        raw_gap2 = float((w * (real_even['raw_mean']-pop['raw_mean']).square().sum(1)).sum())
        aggregate_populations[name] = dict(rows=pop['rows'], covered_even_mass=float(w.sum()),
            missing_groups=(~valid).nonzero().flatten().tolist(),
            feature_residual_energy_on_covered_groups=feature_energy,
            feature_residual_energy=feature_energy if bool(valid.all()) else None,
            raw_mean_gap_even_weighted_norm=math.sqrt(raw_gap2),
            raw_covariance_trace_even_weighted=float((w*pop['raw_covariance_trace']).sum()),
            clipped_rows=int(pop['clipped_counts'].sum()), weights_renormalized=False)
    full = (A['counts'] > 0) & (C['counts'] > 0) & (Yc['counts'] > 0)
    inside_valid = (L['counts'] > 0) & (real_inside['counts'] > 0)
    all_groups = torch.ones(gcount, dtype=torch.bool)
    l2 = float((weights_even * contribution_L.square().sum(1)).sum())
    u2 = float((weights_even * contribution_U.square().sum(1)).sum())
    cross = 2 * float((weights_even * (contribution_L*contribution_U).sum(1)).sum())
    aggregate = dict(populations=aggregate_populations,
        joint_current_cohort=dict(L_rows=int(legal.sum()), U_rows=int(unwanted.sum()),
            inside_p_gt_Q_both_views=True, same_real_topology_group=True,
            excludes_no_rows_from_the_all_anchor_objective=True,
            actual_legal_action_pairs_proposed=0),
        exact_mixture=dict(feature_max_abs_error=feature_identity_error,
            raw_max_abs_error=raw_identity_error, squared_L_contribution=l2,
            squared_U_contribution=u2, cross_contribution=cross,
            total=l2+u2+cross, exact_empty_cohort_zero_mass=True),
        paired_noise=dict(rows=len(clean), group_transition_fraction=float(changed.double().mean()),
            feature_mean_identity_max_abs_error=float(paired_identity.abs().max()),
            raw_mean_identity_max_abs_error=float(raw_paired_identity.abs().max()),
            noise_vs_anchor_target_residual=vector_comparison(noise_feature, rA, full),
            noise_vs_clean_target_residual=vector_comparison(noise_feature, rC, full),
            paired_raw_mean_increment=vector_comparison(noise_raw, noise_raw, full),
            paired_feature_mean_increment=vector_comparison(noise_feature, noise_feature, full),
            same_frame_residual_identity='rY_gC = rC - mean[psi(Y,gC)-psi(C,gC)]'),
        unpaired_anchor_to_clean=dict(
            distribution_delta_vs_anchor_target_residual=vector_comparison(delta_anchor_clean, rA, full),
            contains_latent_perturbation_and_row_sampling=True, is_a_paired_effect=False),
        conditioning=dict(
            all_anchor_residual_vs_legal_inside_target=vector_comparison(rA, rL_inside, inside_valid),
            all_legal_residual_vs_legal_inside_target=vector_comparison(rL, rL_inside, inside_valid),
            all_vs_inside_real_targets=vector_comparison(fixed.even_means-real_inside['feature_mean'],
                rA, real_inside['counts']>0),
            L_vs_U_weighted_contributions=vector_comparison(contribution_L, contribution_U, all_groups)),
        noisy_natural_vs_clean_frame=dict(
            feature_mean_gap=vector_comparison(Yn['feature_mean']-Yc['feature_mean'],
                Yn['feature_mean']-Yc['feature_mean'], (Yn['counts']>0)&(Yc['counts']>0)),
            includes_frame_and_membership_reassignment=True))
    group_rows = []
    for group in range(gcount):
        pop_rows = {}
        for name, pop in populations.items():
            n = int(pop['counts'][group])
            pop_rows[name] = dict(rows=n,
                feature_mean=pop['feature_mean'][group].tolist() if n else None,
                raw_mean=pop['raw_mean'][group].tolist() if n else None,
                raw_covariance_trace=float(pop['raw_covariance_trace'][group]) if n else None,
                clipped_fraction=int(pop['clipped_counts'][group])/n if n else None)
        valid_noise = bool(noise_counts[group]>0)
        group_rows.append(dict(group=group, even_mass=float(weights_even[group]),
            populations=pop_rows, L_fraction=float(fL[group]), U_fraction=float(fU[group]),
            mixture_feature_max_abs_error=float((A['feature_mean'][group]-mixture_feature[group]).abs().max()),
            anchor_target_residual=rA[group].tolist(),
            L_weighted_residual_contribution=contribution_L[group].tolist(),
            U_weighted_residual_contribution=contribution_U[group].tolist(),
            inside_even_target_minus_L=rL_inside[group].tolist() if bool(inside_valid[group]) else None,
            unpaired_anchor_to_clean_feature_mean_delta=delta_anchor_clean[group].tolist()
                if bool(C['counts'][group]>0) else None,
            paired_feature_mean_increment=noise_feature[group].tolist() if valid_noise else None,
            paired_raw_mean_increment=noise_raw[group].tolist() if valid_noise else None,
            noisy_natural_group_destination_counts={str(destination): int(value)
                for destination, value in enumerate(transitions[group].tolist()) if value}))
    return dict(status='COMPLETE_FIXED_DESCRIPTIVE_DIAGNOSTIC', chart=chart,
                clipping_radius=fixed.radius, aggregate=aggregate, groups=group_rows,
                oracle_annotations_used=False, new_statistical_hypotheses=0)


def report(result):
    lines = ['# Final RA10 saved mean-law decomposition', '',
        'One descriptive CPU chart; original grid validity VALID and quality FAIL remain unchanged.', '']
    if result['analysis']['status'] != 'COMPLETE_FIXED_DESCRIPTIVE_DIAGNOSTIC':
        lines += ['The single chart did not support the fixed moment frame.',
                  str(result['analysis'].get('invalid_moment_reason')), '']
    else:
        analysis = result['analysis']
        chart, agg = analysis['chart'], analysis['aggregate']
        lines += [f"Chart: K={chart['actual_cells']}, rank={chart['effective_rank']}, groups={chart['groups']}.",
            f"Current joint cohort: L={agg['joint_current_cohort']['L_rows']}, U={agg['joint_current_cohort']['U_rows']}.", '',
            '| Population | Feature residual energy | Raw group-mean gap | Raw covariance trace |',
            '|---|---:|---:|---:|']
        for name in ('anchors_A','anchors_L','anchors_U','saved_clean_C','saved_noisy_Y_clean_frame',
                     'saved_noisy_Y_natural','saved_real_target'):
            pop = agg['populations'][name]
            lines.append(f"| {name} | {pop['feature_residual_energy']} | {pop['raw_mean_gap_even_weighted_norm']:.8g} | {pop['raw_covariance_trace_even_weighted']:.8g} |")
        lines += ['', '## Fixed comparisons', '']
        for name, value in (
            ('Paired feature-noise increment versus anchor target residual', agg['paired_noise']['noise_vs_anchor_target_residual']),
            ('Paired feature-noise increment versus clean target residual', agg['paired_noise']['noise_vs_clean_target_residual']),
            ('Unpaired A-to-C distribution delta versus anchor target residual', agg['unpaired_anchor_to_clean']['distribution_delta_vs_anchor_target_residual']),
            ('All-anchor residual versus L inside-even target residual', agg['conditioning']['all_anchor_residual_vs_legal_inside_target'])):
            lines += [f"- {name}: dot={value['weighted_dot']:.8g}, cosine={value['weighted_cosine']}, covered even mass={value['covered_even_mass']:.8g}."]
        lines += ['', f"Natural clean-to-noisy group transitions: {agg['paired_noise']['group_transition_fraction']:.8g}.",
                  f"Paired raw mean increment norm: {agg['paired_noise']['paired_raw_mean_increment']['left_weighted_norm']:.8g}.", '']
    lines += ['## Limits', '',
        'C/Y use g(C) for both binning and clipping frame in the paired increment. Natural g(Y) energy is separate.',
        'A-to-C is unpaired. These are final-state empirical comparisons, with trained D/shared FIFO, assignment, clipping and finite-sample limits.',
        'No historical causal attribution, equality certificate, scorer result or production repair is established.',
        'No actions, new samples, data/latent/output-noise draws, training updates, constructors, CUDA contexts or scoring calls occurred.',
        'The one fitted chart uses the declared private random projection from a clone of saved CPU RNG.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args()
    if args.output.resolve().parent != HERE or args.output.exists():
        raise SystemExit('each retained attempt must be a new direct child of this private area')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit('CPU-only execution requires CUDA_VISIBLE_DEVICES to be empty')
    if sha(HERE/'SOURCE-FROZEN.json') != args.source_sha256:
        raise SystemExit('exact approved pre-execution source seal required')
    preseal = read(HERE/'SOURCE-FROZEN.json')
    verify(preseal['source_and_input_sha256'])  # Raw guard BEFORE Torch or numerical interpretation.
    args.output.mkdir()
    begun = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        assert not torch.cuda.is_initialized()
        global_rng_before = torch.get_rng_state().clone()
        numpy_before = numpy_rng_hash(np)
        python_before = random.getstate()
        sys.path.insert(0, str(PACKAGE))
        state_hash = extract(STATE_HASH, 'tensor_state_hash', dict(torch=torch, hashlib=hashlib))
        log('loading one sealed final step7000 state; no module constructors')
        packet = torch.load(NUMERIC_INPUTS[0], map_location='cpu', weights_only=False)
        saved = packet['trainer']
        saved_before = state_hash(saved)
        # Original save path preserves the clean/noisy rows from a single draw.
        with np.load(NUMERIC_INPUTS[1], allow_pickle=False) as cfile, np.load(NUMERIC_INPUTS[2], allow_pickle=False) as yfile:
            assert set(cfile.files) == set(yfile.files) == {'live', 'ema', 'target'}
            assert np.array_equal(cfile['target'], yfile['target'])
            clean = torch.from_numpy(cfile['ema'].copy())
            noisy = torch.from_numpy(yfile['ema'].copy())
            target = torch.from_numpy(cfile['target'].copy())
        with torch.no_grad():
            analysis = analyze(saved, clean, noisy, target, torch, F, np, state_hash)
        assert state_hash(saved) == saved_before
        assert torch.equal(torch.get_rng_state(), global_rng_before)
        assert numpy_rng_hash(np) == numpy_before and random.getstate() == python_before
        assert not torch.cuda.is_initialized()
        verify(preseal['source_and_input_sha256'])
        result = dict(status='COMPLETE', utc=datetime.now(timezone.utc).isoformat(),
            started_utc=begun, elapsed_seconds=time.perf_counter()-started,
            scope='one final RA10 same-chart saved EMA clean/noisy100k decomposition',
            source_seal_sha256=args.source_sha256, selected_design_sha256=sha(PLAN/'DESIGN.md'),
            original_grid_fixture_status='VALID', original_grid_quality='FAIL', analysis=analysis,
            saved_semantic_state_sha256=saved_before,
            checks=dict(raw_guards_before_and_after=True, saved_state_unchanged=True,
                CPU_global_Torch_RNG_unchanged=True, NumPy_RNG_unchanged=True, Python_RNG_unchanged=True,
                constructors=0, PT_objects_loaded=1, chart_fits=1, global_RNG_draws=0,
                new_data_latent_or_output_noise_draws=0,
                private_chart_random_projection_draws=int(analysis['chart']['effective_rank']>0),
                private_chart_random_projection_only=True, new_samples=0, actions=0, optimizer_updates=0,
                scoring_calls=0, CUDA_contexts=0, oracle_decisions=0, quality_gate_changes=0),
            limits=['new final CPU chart is not historical pre-action GPU chart',
                    'learned D/shared FIFO and finite samples preclude an independent null certificate',
                    'C/Y are paired; A/C are unpaired distributions',
                    'inside-conditioned real target is not an identical double-view sampling cohort',
                    'no causal or production qualification from these descriptive values'])
        write_new(args.output/'result.json', result)
        with (args.output/'REPORT.md').open('x') as handle:
            handle.write(report(result))
        log('COMPLETE; source/state/RNG unchanged, exactly one chart, no draws/actions/scoring')
    except BaseException:
        write_new(args.output/'failure.json', dict(status='PRIVATE_DIAGNOSTIC_ERROR',
            utc=datetime.now(timezone.utc).isoformat(), traceback=traceback.format_exc(),
            source_seal_sha256=args.source_sha256))
        raise


if __name__ == '__main__':
    main()
