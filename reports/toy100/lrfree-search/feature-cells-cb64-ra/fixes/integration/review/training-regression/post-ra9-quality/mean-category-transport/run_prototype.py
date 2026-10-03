"""Single frozen grid/toy scratch prototype; no quality score, training or CUDA."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
PACKAGE = ROOT / 'pkg-CB64-RA9'
GRID_HEAD = ROOT / 'performance/training-regression/count-review/post-ra8-quality/grid-covariance/diagnose.py'
TOY_HOST = ROOT / 'integration/review/training-regression/post-ra5-saved-diagnosis/analyze_saved.py'
HASH_HOST = ROOT / 'integration/review/training-regression/post-ra4-quality/measure_saved_utils.py'
CASES = (('grid', ROOT / 'validation-cb64-ra9/screens/runs/grid100/final-state.pt'),
         ('toy', ROOT / 'validation-cb64-ra9/learned/training/toy/CB64-RA9/checkpoint-2000.pt'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify():
    preseal = json.loads((HERE / 'SOURCE-FROZEN.json').read_text())
    for path, expected in preseal['source_and_input_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'source/input guard mismatch: {path}')
    return preseal


def extract(path, name, namespace):
    nodes = [n for n in ast.parse(Path(path).read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def log(message):
    print(datetime.now(timezone.utc).isoformat(), message, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('output exists; retain each attempt separately')
    preseal = verify()  # Raw byte guards precede Torch or numerical PT interpretation.
    args.output.mkdir(parents=True)
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(PACKAGE))
    from particlegan.feature_cells import FeatureCellSnapshot, LatentLineage, BoundedLatentGeometry
    from witness import freeze_moment, odd_witness, common_count_family, group_means, Q
    from transport import observe_view, propose_pairs, preview_pairs, commit_packet, tensor_digest
    namespace = dict(torch=torch, F=F, hashlib=hashlib)
    head_grid = extract(GRID_HEAD, 'head_features', namespace.copy())
    host_toy = extract(TOY_HOST, 'forward', namespace.copy())
    state_hash = extract(HASH_HOST, 'tensor_state_hash', namespace.copy())
    global_before = torch.get_rng_state().clone()
    outputs = []
    try:
        with torch.no_grad():
            for label, path in CASES:
                log(f'{label}: loading one frozen saved final state')
                started = time.perf_counter()
                packet = torch.load(path, map_location='cpu', weights_only=False)
                saved = packet['trainer']
                saved_hash = state_hash(saved)
                weights = saved['models']
                zf, ze = weights['prior']['z'], weights['ema_prior']['z']
                assert zf.shape == ze.shape and zf.dtype == ze.dtype and zf.device == ze.device
                assert bool(torch.isfinite(zf).all() & torch.isfinite(ze).all())
                bd = saved['birth_death']
                assert bd['fill'] == len(zf) and bd['reservoir'].shape[0] == len(zf)
                assert bd['last']['step'] == saved['completed_steps']
                assert bd['settings']['cells'] == 128 and bd['settings']['rank'] == 8
                # Fixture-only host decoding; the mechanism receives generic callbacks.
                if label == 'grid':
                    def raw(points, role):
                        return F.linear(points, weights[role]['weight'], weights[role]['bias'])
                    def head(points):
                        return head_grid(points, weights['D'])
                else:
                    def raw(points, role):
                        return host_toy(points, weights[role])
                    def head(points):
                        return host_toy(points, weights['D'], head=True)
                features_fast = lambda points: head(raw(points, 'G'))
                features_ema = lambda points: head(raw(points, 'ema_G'))
                real_features = head(bd['reservoir'])
                fast_features, ema_features = features_fast(zf), features_ema(ze)
                assert bool(torch.isfinite(real_features).all() & torch.isfinite(fast_features).all()
                            & torch.isfinite(ema_features).all())
                stream = torch.Generator(device='cpu')
                stream.set_state(saved['cpu_rng'].cpu().clone())
                stream_initial = tensor_digest(stream.get_state())
                log(f'{label}: one current real chart; geometry and boundary even-only')
                snapshot = FeatureCellSnapshot.fit(real_features, generator=stream,
                    cells=bd['settings']['cells'], rank=bd['settings']['rank'], chunk=bd['settings']['chunk'])
                snapshot.cache_queries(fast_features)
                topology = snapshot._mass_topology()
                # Exact declared order: even-only geometry + clean EMA -> fixed directions -> all odd.
                fixed, invalid = freeze_moment(snapshot, real_features[0::2], ema_features)
                witness = odd_witness(snapshot, fixed, real_features[1::2])
                witness['frozen_moment_invalid_reason'] = invalid
                log(f"{label}: witness valid={witness['valid']} fires={witness['fires']} lower={witness.get('lower_bound')}")
                counts = common_count_family(snapshot, fast_features)
                vf = observe_view(snapshot, fast_features, coordinates=zf)
                ve = observe_view(snapshot, ema_features, coordinates=ze)
                n = len(zf)
                earlier = bd['last']['ordinary_moves']
                total_budget = math.floor(Q * n)
                assert type(earlier) is int and 0 <= earlier <= total_budget
                # Row IDs of historical actions are absent. These selected cases
                # allow an honest prefix: grid empty, toy all slots consumed.
                assert earlier == 0 or earlier == total_budget
                if earlier == 0:
                    assert bd['last']['moves'] == 0  # No hidden isolation/source prefix.
                reservations = torch.empty(0, dtype=torch.long)
                pairs = propose_pairs(snapshot, fixed, vf, ve,
                    earlier_ordinary=earlier, reserved_rows=reservations)
                pair_supply = dict(jointly_eligible_rows=pairs.eligible_rows,
                    positive_pairs_before_budget=pairs.positive_pairs_before_budget,
                    candidate_pairs_after_budget=len(pairs.children), total_budget=total_budget,
                    earlier_ordinary_actions=earlier, residual_budget=pairs.budget,
                    protected_rows=len(pairs.protected_rows),
                    prefix_policy='saved last ordinary count; empty rows only when no prior actions, otherwise exhausted budget')
                log(f'{label}: legal positive supply={pairs.positive_pairs_before_budget}, residual budget={pairs.budget}')
                action_detail = dict(attempts=0, accepted=0, reason='witness_veto', paired_noise_draws=0)
                application = None
                stream_after_chart = tensor_digest(stream.get_state())
                if witness['fires'] and len(pairs.children):
                    row_sources = []
                    for parameter_id, values in saved['optimizers'][0]['state'].items():
                        if any(isinstance(v, torch.Tensor) and v.shape == zf.shape for v in values.values()):
                            row_sources.append(values)
                    assert len(row_sources) == 1
                    row_state = deepcopy(row_sources[0])
                    regularizer = saved['optimizers'][0]['regularizer']['latent']
                    history = None if regularizer is None else regularizer['history'].clone()
                    lineage = LatentLineage(n, bd['lineage_neighbors'].shape[1], torch.device('cpu'))
                    lineage.neighbors = bd['lineage_neighbors'].clone()
                    lineage.validate(lineage.neighbors)
                    geometry = BoundedLatentGeometry(rank=8, neighbors=64, chunk=256, lineage=lineage)
                    model_tensors = [weights[role][key] for role in ('G', 'ema_G', 'D')
                                     for key in saved['requires_grad'][role]]
                    def buffer_epoch():
                        return tuple((role, key, id(value), tensor_digest(value))
                                     for role in ('G', 'ema_G', 'D')
                                     for key, value in weights[role].items()
                                     if key not in saved['requires_grad'][role])
                    scratch = SimpleNamespace(prior=SimpleNamespace(z=zf.clone()),
                        ema_prior=SimpleNamespace(z=ze.clone()), lineage=lineage,
                        row_state=row_state, history=history, consumed=set(),
                        bandwidth=saved['controller']['latent_bandwidth'].clone(),
                        model_tensors=model_tensors, buffer_epoch=buffer_epoch, stream=stream)
                    row_state_before = deepcopy(row_state)
                    history_before = None if history is None else history.clone()
                    prepared, action_detail = preview_pairs(snapshot, fixed, vf, ve, pairs, scratch,
                        stream=stream, features_fast=features_fast, features_ema=features_ema,
                        geometry=geometry)
                    assert tensor_digest(scratch.prior.z) == tensor_digest(zf)
                    assert tensor_digest(scratch.ema_prior.z) == tensor_digest(ze)
                    if prepared is not None:
                        stream_before_commit = tensor_digest(stream.get_state())
                        application = commit_packet(prepared, scratch, snapshot)
                        assert tensor_digest(stream.get_state()) == stream_before_commit
                        new_fast = features_fast(scratch.prior.z[prepared.children])
                        new_ema = features_ema(scratch.ema_prior.z[prepared.children])
                        assert tensor_digest(new_fast) == tensor_digest(prepared.fast_features)
                        assert tensor_digest(new_ema) == tensor_digest(prepared.ema_features)
                        final_fast_features = fast_features.clone(); final_fast_features[prepared.children] = new_fast
                        final_ema_features = ema_features.clone(); final_ema_features[prepared.children] = new_ema
                        af = observe_view(snapshot, final_fast_features, coordinates=scratch.prior.z)
                        ae = observe_view(snapshot, final_ema_features, coordinates=scratch.ema_prior.z)
                        for old, new in ((vf, af), (ve, ae)):
                            assert torch.equal(old.categories, new.categories)
                            assert torch.equal(old.cells, new.cells) and torch.equal(old.groups, new.groups)
                            assert torch.equal(old.eligible, new.eligible)
                        means, group_counts = group_means(fixed.psi(ae.metric, ae.groups), ae.groups, snapshot.mass_groups)
                        energy_after = float(fixed.energy(means))
                        assert torch.equal(group_counts, fixed.ema_counts)
                        assert energy_after < float(fixed.energy())
                        assert math.isclose(energy_after, action_detail['objective_after_virtual'], rel_tol=1e-10, abs_tol=1e-12)
                        child, parent = prepared.children, prepared.parents
                        untouched = torch.ones(n, dtype=torch.bool); untouched[child] = False
                        for key, old in row_state_before.items():
                            new = scratch.row_state[key]
                            if isinstance(old, torch.Tensor) and old.shape == zf.shape:
                                assert tensor_digest(new[child]) == tensor_digest(old[parent])
                                assert tensor_digest(new[untouched]) == tensor_digest(old[untouched])
                            else:
                                assert state_hash(new) == state_hash(old)
                        if history_before is not None:
                            assert tensor_digest(scratch.history[child]) == tensor_digest(history_before[parent])
                            assert tensor_digest(scratch.history[untouched]) == tensor_digest(history_before[untouched])
                        assert tensor_digest(scratch.prior.z[untouched]) == tensor_digest(zf[untouched])
                        assert tensor_digest(scratch.ema_prior.z[untouched]) == tensor_digest(ze[untouched])
                        assert earlier + len(child) <= total_budget
                        try:
                            commit_packet(prepared, scratch, snapshot)
                        except ValueError:
                            pass
                        else:
                            raise AssertionError('consumed packet allowed a second commit')
                        application.update(objective_after_actual=energy_after,
                            category_group_supported_ledgers_exact=True, original_sources_untouched=True,
                            exact_row_optimizer_history_inheritance=True, complete_moved_rows=True,
                            no_commit_noise_draw=True, packet_consumption_rejects_second_commit=True,
                            production_rebase_serving_orchestration_implemented=False)
                assert state_hash(saved) == saved_hash
                assert torch.equal(torch.get_rng_state(), global_before)
                assert sha(path) == preseal['source_and_input_sha256'][str(path)]
                record = dict(case=label, completed_steps=saved['completed_steps'], rows=n,
                    actual_cells=snapshot.cells, effective_rank=snapshot.rank,
                    real_topology=dict(snapshot.mass_topology), historical_saved_topology=bd['last']['mass_topology'],
                    witness=witness, common_count_family=counts, pair_supply=pair_supply,
                    preview=action_detail, scratch_application=application,
                    saved_state_typed_sha256=saved_hash, saved_state_unchanged=True,
                    saved_file_unchanged=True, global_CPU_RNG_unchanged=True,
                    source_streams_unchanged=True, raw_FAST_and_EMA_views=True,
                    planner_stream_initial_sha256=stream_initial,
                    planner_stream_after_chart_sha256=stream_after_chart,
                    planner_stream_final_sha256=tensor_digest(stream.get_state()),
                    work=dict(snapshot.work), elapsed_seconds=time.perf_counter()-started,
                    no_CUDA_context=True, emitted_clouds=0, training_updates=0,
                    limitations=['current CPU chart, not historical CUDA chart',
                        'clean nonlinear feature objective, not noisy emitted mean or quality certificate',
                        'trained D/shared FIFO empirical negative evidence, not prospective iid certification',
                        'scratch row packet only; production phase ordering/rebase/serving integration absent'])
                (args.output / f'{label}.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
                outputs.append(record)
                log(f"{label}: done accepted={action_detail['accepted']} state and RNG unchanged")
                del packet, saved
        verify()
        result = dict(status='PASS_FIXED_PROTOTYPE', protocol_sha256=sha(HERE/'DESIGN.md'),
            source_preseal_sha256=sha(HERE/'SOURCE-FROZEN.json'), cases=outputs,
            no_production_patch=True, no_quality_acceptance=True, single_fixed_run=True,
            finished_UTC=datetime.now(timezone.utc).isoformat())
        (args.output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    except BaseException as error:
        failure = dict(status='FAILED_TECHNICAL_ATTEMPT', error=repr(error), traceback=traceback.format_exc())
        (args.output / 'failure.json').write_text(json.dumps(failure, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
