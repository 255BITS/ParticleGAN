"""Independent stdlib-only immutable source/receipt audit; never imports Torch."""
import ast
import datetime
import gzip
import io
import json
import math
import statistics
import zipfile
from pathlib import Path

import inspection_helpers as h

HERE = Path(__file__).resolve().parent
ATTEMPT = HERE.parent
RUNS = ATTEMPT / 'repo/experiments/nonlinear_macrostep'
OLD = Path('/ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T232429Z-1607588/constant_rate_stability/20260926T232429Z-1607596/repo/experiments/constant_conditioning')
RESEARCH = Path('/ml2/hypergan/ParticleGAN-k3p-continuous-search/reports/toy100/continuous-api-search')
h.RUNS = RUNS
now = datetime.datetime.now(datetime.timezone.utc).isoformat()


def checkpoint_details(path):
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith('/data.pkl'))
        prefix = name[:-len('data.pkl')]
        state = h.StorageOnlyUnpickler(io.BytesIO(z.read(name))).load()
        state = state.get('trainer', state)

        def raw(t):
            width = {'FloatStorage': 4, 'ByteStorage': 1}[t['storage']['dtype']]
            count, expected = math.prod(t['size']), 1
            for length, stride in reversed(list(zip(t['size'], t['stride']))):
                assert length <= 1 or stride == expected
                expected *= length
            data = z.read(prefix + 'data/' + t['storage']['key'])
            offset = t['offset'] * width
            return data[offset:offset + count * width]

        return dict(completed_steps=state['completed_steps'],
                    native_state_counts=[len(o['state']) for o in state['optimizers']],
                    model_tensor_hashes={m: {k: h.sha(raw(t)) for k, t in values.items()}
                                         for m, values in state['models'].items()},
                    model_nonparameter_keys={m: sorted(set(values) - set(state['requires_grad'][m]))
                                             for m, values in state['models'].items()},
                    stream_hashes={k: h.sha(raw(v)) for k, v in state['streams'].items()},
                    cpu_rng=h.sha(raw(state['cpu_rng'])),
                    cuda_rng=h.sha(raw(state['cuda_rng'])),
                    game_stats_serialized='game_stats' in state,
                    serial_backward=state['serial_backward'])


def costs_and_rates(folder):
    rows = h.read_rows(folder / 'learning-rates.jsonl')
    assert [r['step'] for r in rows] == list(range(1, len(rows) + 1))
    fields = [r['game']['field_evaluations'] for r in rows]
    assert all(2 <= n <= 16 for n in fields)
    assert all(r['step'] == r['controller']['calls'] == r['controller']['observed_steps']
               == r['game']['accepted_updates'] for r in rows)
    assert all(r['controller']['ema_updates'] == max(0, r['step'] - 799)
               and r['controller']['ema_skips'] == r['controller']['ema_reseeds'] == 0 for r in rows)
    for r in rows:
        game = r['game']
        trials = game['trials']
        assert [t['field_evaluations'] for t in trials] == list(range(2, game['field_evaluations'] + 1))
        selected = next(t for t in trials if t['field_evaluations'] == game['selected_field'])
        assert selected['residual_squared'] == game['selected_residual_squared']
        assert selected['relative_implicit_residual'] == game['relative_implicit_residual']
        assert game['fraction'] == 1.0
        if game['converged']:
            assert all(v <= .5 for v in game['relative_implicit_residual'].values())
            assert game['selected_field'] == game['field_evaluations']
        else:
            assert game['nonconvergence'] == 'accept_minimum_actual_residual_iterate'
            assert game['selected_residual_squared'] == min(t['residual_squared'] for t in trials)
    return dict(updates=len(rows), nominal_rate_ranges={k: [min(r[k] for r in rows), max(r[k] for r in rows)]
                    for k in ('generator_0', 'critic_0', 'prior_1')},
                controller_and_accepted_counts_exact=True, reference_counts_exact=True,
                total_joint_fields=sum(fields), mean_joint_fields=statistics.mean(fields),
                joint_fields_range=[min(fields), max(fields)],
                solver_converged=sum(r['game']['converged'] for r in rows),
                solver_not_converged=sum(not r['game']['converged'] for r in rows),
                selected_earlier_field=sum(r['game']['selected_field'] < r['game']['field_evaluations'] for r in rows),
                selected_residual_ratio_ranges={role: [min(r['game']['relative_implicit_residual'][role] for r in rows),
                                                       max(r['game']['relative_implicit_residual'][role] for r in rows)]
                                               for role in ('D', 'G', 'prior')},
                actual_update_l2_ranges={role: [min(r['updates'][role]['l2'] for r in rows),
                                                max(r['updates'][role]['l2'] for r in rows)]
                                        for role in ('D', 'G', 'prior')},
                zero_l2_updates={role: sum(r['updates'][role]['l2'] == 0 for r in rows) for role in ('D', 'G', 'prior')},
                last_game=rows[-1]['game'])


def archive_image(run, candidate, identifier, audit_name):
    folder = RUNS / run
    result = h.read_json(folder / 'result.json')
    declaration = h.read_json(folder / 'declaration.json')
    rows = h.read_rows(folder / 'metrics.jsonl')
    source = h.source_summary(run)
    assert not source['declaration_hash_mismatches'] and not source['missing_declared_files']
    assert [r['step'] for r in rows] == list(range(25, 601, 25))
    bounds = declaration['spec']['thresholds']
    passing = [r['modes'] >= bounds['modes'] and r['hq'] >= bounds['hq_min'] for r in rows]
    suffix = 0
    for ok in reversed(passing):
        if not ok:
            break
        suffix += 1
    assert result['status'] == ('PASS' if suffix >= 5 else 'FAIL')
    assert result['metrics']['convergence']['passing_observations'] == sum(passing)
    assert result['metrics']['convergence']['passing_suffix'] == suffix
    assert all(r['modes'] == sum(f >= bounds['min_mode_fraction'] for f in r['quality_mode_fractions']) for r in rows)
    assert all(abs(r['hq'] - sum(r['quality_mode_fractions'])) < 1e-12 for r in rows)
    assert all(abs(r['distribution_tv'] - sum(abs(f - .5) for f in r['mode_fractions']) / 2) < 1e-12 for r in rows)
    with zipfile.ZipFile(folder / 'source.zip') as z, zipfile.ZipFile(OLD / 'C12-img_intensity2/source.zip') as previous:
        frozen = ['benchmarks/transfer_suite/image_tasks.py', 'benchmarks/locked_shared/observation.py',
                  'benchmarks/transfer_suite/plans/default_comparison.json', 'benchmarks/toy100/models.py',
                  'benchmarks/toy100/device.py', 'benchmarks/locked_shared/mode_hold.py', 'benchmarks/locked_shared/mlp.py']
        helper_differences = [n for n in frozen if z.read(n) != previous.read(n)]
        plan = json.loads(z.read(frozen[2]))
        assert next(x['spec'] for x in plan if x['spec']['name'] == 'img_intensity2') == declaration['spec']
        host_equal = z.read('experiments/nonlinear_macrostep/image_gate.py').replace(candidate.encode(), b'API-C12') == previous.read('experiments/constant_conditioning/image_gate.py')
    assert not helper_differences and host_equal
    initial = checkpoint_details(folder / 'initial-state.pt')
    old_initial = checkpoint_details(OLD / 'C12-img_intensity2/initial-state.pt')
    initial_match = {k: initial[k] == old_initial[k] for k in ('model_tensor_hashes', 'stream_hashes', 'cpu_rng', 'cuda_rng', 'native_state_counts')}
    assert all(initial_match.values()) and initial['native_state_counts'] == [0, 0]
    assert not any(initial['model_nonparameter_keys'].values())
    native = h.checkpoint_summary(folder / 'final-state.pt')
    assert native['completed_steps'] == 600 and native['adam_step_values'] == [[600.0], [600.0]]
    assert native['adam_step_devices'] == [['cpu'], ['cpu']] and native['serial_backward']
    costs = costs_and_rates(folder)
    assert costs['updates'] == 600
    assert costs['nominal_rate_ranges'] == {'generator_0': [.00425, .00425], 'critic_0': [.00425, .00425], 'prior_1': [.0085, .0085]}
    original_manifest = h.read_json(folder / 'artifact-sha256.json')
    assert all(h.sha((folder / n).read_bytes()) == digest for n, digest in original_manifest.items())
    destination = RESEARCH / 'evidence' / identifier
    destination.mkdir(parents=True, exist_ok=True)
    files, external = {}, {}
    for path in sorted(folder.iterdir()):
        raw = path.read_bytes()
        if path.suffix == '.pt':
            external[path.name] = dict(original=str(path), sha256=h.sha(raw), bytes=len(raw),
                                       verified_against_original_manifest=path.name in original_manifest)
            continue
        if path.suffix not in ('.json', '.jsonl', '.zip'):
            continue
        data = gzip.compress(raw, mtime=0) if path.suffix == '.jsonl' else raw
        target = destination / (path.name + '.gz' if path.suffix == '.jsonl' else path.name)
        if target.exists():
            assert target.read_bytes() == data
        else:
            target.write_bytes(data)
        files[str(target.relative_to(RESEARCH))] = dict(sha256=h.sha(data), original_sha256=h.sha(raw),
            original=str(path), bytes=len(data), original_bytes=len(raw), listed_in_original_manifest=path.name in original_manifest)
    manifest_path = destination / 'archive-manifest.json'
    archive_manifest = dict(schema=1, id=identifier, created_utc=now, original_directory=str(folder),
        files=files, external_checkpoints=external,
        note='Source/declarations/results copied exactly; JSONL gzip mtime0 is lossless; checkpoints remain external with verified hashes.')
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(archive_manifest, indent=2) + '\n')
    entry = dict(id=identifier, source_directory=str(folder), source_repo=str(ATTEMPT / 'repo'), recorded_utc=now,
        candidate=candidate, protocol='frozen22_img_intensity2', scope='PUBLIC_API', archived_source_verified=True,
        independent_evaluator_audit='VERIFIED_MEASURED_' + result['status'], result=result,
        derived=dict(convergence=dict(complete=True, observations=24, passing_observations=sum(passing), passing_suffix=suffix,
            first_pass_step=next((r['step'] for r, ok in zip(rows, passing) if ok), None),
            stable_from_step=rows[-suffix]['step'] if suffix >= 5 else None, minimum_stable_checks=5),
            passing_steps=[r['step'] for r, ok in zip(rows, passing) if ok], failing_steps=[r['step'] for r, ok in zip(rows, passing) if not ok],
            final_live={k: v for k, v in rows[-1].items() if k != 'ema'}),
        verification=dict(source, artifact_hash_mismatches=[], frozen_helper_differences=helper_differences,
            image_host_equal_to_C12_except_candidate_label=host_equal, raw_initial_state_equal_to_C12=initial_match,
            initial_nonparameter_keys=initial['model_nonparameter_keys'], runtime=declaration['runtime']),
        numerical_cost=costs, checkpoint=native, artifacts=files, external_checkpoints=external,
        scope_note='Own measured stateless fixed-draw image host only. No borrowed result for another source version; no mature critic-memory observation before first blend800; no exact residual guarantee on nonconverged accepted updates.',
        archive_manifest=dict(path=str(manifest_path.relative_to(RESEARCH)), sha256=h.sha(manifest_path.read_bytes())))
    audit = dict(created_utc=now, scope='Source, receipts, raw checkpoint storage and stdlib arithmetic only; no Torch/model/training/GPU execution.',
        run_count=1, passing_runs=int(result['status'] == 'PASS'), failed_runs=int(result['status'] == 'FAIL'), broader_entries=[entry],
        limits=['Stored metrics/source audited; samples/RMSE not regenerated.', 'No shared manifest edits; ready entry for parent incorporation.'])
    # Ready entries may already have been incorporated by the supervisor.
    # Preserve their timestamps and bytes on a repeated independent audit.
    if not (RESEARCH / audit_name).exists():
        (RESEARCH / audit_name).write_text(json.dumps(audit, indent=2) + '\n')
    return entry


if __name__ == '__main__':
    entry = archive_image('C13-img_intensity2', 'API-C13', 'api-c13-img_intensity2', 'c13-image-audit.json')
    out = dict(created_utc=now, scope='Independent source/receipt audit, no Torch execution, no worker edits.', original_image=entry)
    revision = 'C13-R1-img_intensity2'
    if (RUNS / revision / 'source.zip').exists():
        source = h.source_summary(revision)
        assert not source['declaration_hash_mismatches'] and not source['missing_declared_files']
        source['changed_package_files_from_C13'] = [n for n, digest in source['package_sha256'].items()
            if entry['verification']['package_sha256'].get(n) != digest]
        assert source['changed_package_files_from_C13'] == ['particlegan/nonlinear_game.py']
        out['R1_source'] = source
        out['R1_declaration_sha256'] = h.sha((RUNS / 'C13-R1-declaration.json').read_bytes())
        out['R1_quality_status_at_audit'] = h.read_json(RUNS / revision / 'result.json')['status'] if (RUNS / revision / 'result.json').exists() else 'RUNNING'
        if (RUNS / revision / 'result.json').exists():
            revised = archive_image(revision, 'API-C13-R1', 'api-c13-r1-img_intensity2', 'c13-r1-image-audit.json')
            out['R1_image'] = revised
            out['R1_image_equivalence_to_original'] = dict(
                final_checkpoint_byte_equal=(RUNS / revision / 'final-state.pt').read_bytes() == (RUNS / 'C13-img_intensity2/final-state.pt').read_bytes(),
                observations_byte_equal=(RUNS / revision / 'metrics.jsonl').read_bytes() == (RUNS / 'C13-img_intensity2/metrics.jsonl').read_bytes(),
                independent_execution=True)
    if (RUNS / 'C13-R1-single/source.zip').exists():
        source = h.source_summary('C13-R1-single')
        assert not source['declaration_hash_mismatches'] and not source['missing_declared_files']
        assert source['package_sha256'] == out['R1_source']['package_sha256']
        initial = checkpoint_details(RUNS / 'C13-R1-single/initial-state.pt')
        old_initial = checkpoint_details(OLD / 'C12-single/initial-state.pt')
        initial_match = {k: initial[k] == old_initial[k] for k in ('model_tensor_hashes', 'stream_hashes', 'cpu_rng', 'cuda_rng', 'native_state_counts')}
        assert all(initial_match.values())
        out['R1_single_source'] = dict(source, package_matches_R1_image=True,
            raw_initial_state_equal_to_C12=initial_match, quality_status='RUNNING_AT_SOURCE_REVIEW_NO_QUALITY_CLAIM')
    with zipfile.ZipFile(RUNS / 'final-regression-source.zip') as z:
        manifest = h.read_json(RUNS / 'final-regression-declaration.json')['source_sha256']
        hashes = {n: h.sha(z.read(n)) for n in z.namelist()}
        assert hashes == manifest
        assert all(hashes[n] == digest for n, digest in out['R1_source']['package_sha256'].items())
    out['existing_CPU_regression_receipt'] = dict(log=(RUNS / 'regression-final.log').read_text().strip(),
        source_zip_sha256=h.sha((RUNS / 'final-regression-source.zip').read_bytes()),
        source_entries=len(hashes), all_entries_match_declaration=True, package_matches_R1=True,
        independently_rerun=False)
    out['findings'] = [
        dict(id='C13_REJECTED_TRIAL_STATE', severity='generalized_public_API_blocker', source='particlegan/nonlinear_game.py:98-139',
             issue='Original C13 commits chosen parameters/base native moments but retains buffers and managed/global RNG from the final trial, including when an earlier trial won.',
             current_image_scope='No buffers and fixed draw counts: does not invalidate this own measured quality result.',
             R1='Resolved in immutable R1 by explicitly committing one base field of buffers/RNG together with base-gradient native history; accepted reference/EMA see accepted parameters/base buffers.'),
        dict(id='C13_DIAGNOSTIC_SERIALIZATION_WORDING', severity='declaration_only',
             issue='Original declaration calls game_stats checkpointed, but it is diagnostic only and omitted; no solver memory survives a step.',
             R1='Metadata clarification and R1 declaration explicitly document diagnostic-only game_stats; no missing causal solver state.'),
        dict(id='FINITE_NOT_NONZERO_PREDICATE', severity='declared_contract_edge',
             issue='Both versions test finite displacement/residual entries but do not explicitly require nonzero displacement or finite scalar sum-of-squares; declaration says finite nonzero iterate.',
             observed='No zero role displacement in completed original image receipts; no numerical failure inferred.'),
    ]
    (HERE / 'c13-source-audit.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(dict(audit=str(HERE / 'c13-source-audit.json'), ready_entry=str(RESEARCH / 'c13-image-audit.json'),
        score=entry['derived']['convergence'], costs={k:v for k,v in entry['numerical_cost'].items() if k!='last_game'},
        checkpoint=entry['checkpoint'], R1_source=out.get('R1_source'), regressions=out['existing_CPU_regression_receipt']), indent=2))
