"""Audit one sealed research mode_hold result using only the standard library.

Requires that candidate's prepared-index row and matching independent CPU proof.
This verifies retained source/state/score evidence; it never runs learner code.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import io
import json
import math
import struct
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('raw_checkpoint', ROOT / 'audit_public3_runtime.py')
raw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raw)
read = lambda path: json.loads(path.read_bytes())
sha = lambda data: hashlib.sha256(data).hexdigest()
hashfile = lambda path: sha(path.read_bytes())


def checkpoint(path):
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith('/data.pkl')]
        assert len(names) == 1
        name = names[0]
        prefix = name[:-8]
        assert archive.read(prefix + 'byteorder') == b'little'
        payload = raw.Reader(io.BytesIO(archive.read(name))).load()

        def binary(tensor):
            kind, key, device, count = tensor.storage
            width = {'FloatStorage': 4, 'ByteStorage': 1}[kind]
            data = archive.read(prefix + 'data/' + key)
            assert len(data) == count * width
            stride = 1
            for length, actual in reversed(list(zip(tensor.shape, tensor.stride))):
                assert length <= 1 or actual == stride
                stride *= length
            return data[tensor.offset * width:(tensor.offset + math.prod(tensor.shape)) * width]

        def material(value):
            if isinstance(value, raw.Tensor):
                return dict(shape=list(value.shape), dtype={'FloatStorage': 'torch.float32', 'ByteStorage': 'torch.uint8'}[value.storage[0]], sha256=sha(binary(value)))
            if isinstance(value, dict):
                return {str(k): material(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [material(v) for v in value]
            assert value is None or isinstance(value, (int, float, str, bool)), type(value)
            return value

        clocks = []
        if isinstance(payload, dict):
            for optimizer in payload.get('optimizers', []):
                ids = [p for group in optimizer['param_groups'] for p in group['params']]
                assert len(ids) == len(set(ids)) and set(ids) == set(optimizer['state'])
                entries = []
                for parameter in ids:
                    state = optimizer['state'][parameter]
                    counter = state['step']
                    assert counter.shape == () and counter.storage[0] == 'FloatStorage'
                    assert state['exp_avg'].storage[2] == state['exp_avg_sq'].storage[2]
                    entries.append(dict(parameter=parameter, step=struct.unpack('<f', binary(counter))[0], device=counter.storage[2], moment_device=state['exp_avg'].storage[2], state_keys=list(state)))
                clocks.append(entries)
        return material(payload), clocks


def summary(rows):
    good = lambda row: row['modes'] == 8 and .9 <= row['hq'] <= 1
    first = next((r['step'] for r in rows if good(r)), None)
    after = [r for r in rows if first is not None and r['step'] >= first]
    suffix = []
    for row in reversed(rows):
        if not good(row): break
        suffix.append(row)
    return dict(observations=len(rows), passing_observations=sum(map(good, rows)), first_arrival=first,
                passing_since_arrival=sum(map(good, after)), observations_since_arrival=len(after),
                departures=[r['step'] for r in after if not good(r)], passing_suffix=len(suffix),
                final_suffix_start=suffix[-1]['step'] if suffix else None,
                minimum_hq_since_arrival=min((r['hq'] for r in after), default=None),
                minimum_modes_since_arrival=min((r['modes'] for r in after), default=None))


def audit(source, row, index_path):
    prepared = Path(row['directory'])
    proof_path = Path(row['required_review'])
    plan_path = prepared / 'source-plan.json'
    plan = read(plan_path)
    proof = read(proof_path)
    seal = read(prepared / 'manifest.json')
    assert row['candidate'] == plan['candidate'] and row['queue_row'] == plan['source_queue_row']
    for key, file in [('manifest_sha256', 'manifest.json'), ('source_plan_sha256', 'source-plan.json'), ('bridge_sha256', 'initialization_bridge.py'), ('worker_sha256', 'run_research_mode_hold.py')]:
        assert row[key] == hashfile(prepared / file)
    assert proof['status'] == 'PASS' and proof['cuda_initialized'] is False and proof['learner_steps'] == 0
    for field, file in [('source_plan_sha256', 'source-plan.json'), ('bridge_sha256', 'initialization_bridge.py'), ('runner_sha256', 'run_research_mode_hold.py'), ('manifest_sha256', 'manifest.json')]:
        assert proof[field] == hashfile(prepared / file)
    required = ('all_initial_tensors_match_public_host', 'repeat_without_rng_reset', 'constructor_rng_cursor_preserved', 'initializer_rng_neutral', 'historical_prior_registration_preserved', 'all_bindings_restore_on_exception', 'batch_distance_scope_explicit')
    assert all(proof['checks'][k] is True for k in required)
    assert read(source / 'source-plan.json') == plan and read(source / 'prepared-source-manifest.json') == seal
    seal_receipt = read(source / 'prepared-source-sha256.json')
    assert seal_receipt == dict(source_zip_sha256=hashfile(source / 'prepared-source.zip'), prepared_manifest_sha256=hashfile(prepared / 'manifest.json'), reviewed_cpu_proof_sha256=hashfile(proof_path))
    with zipfile.ZipFile(source / 'prepared-source.zip') as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert set(archive.namelist()) == set(seal['files']) | {'manifest.json', 'reviewed-cpu-proof.json'}
        for name, expected in seal['files'].items():
            assert sha(archive.read(name)) == expected == hashfile(prepared / name), name
        assert sha(archive.read('manifest.json')) == seal_receipt['prepared_manifest_sha256']
        assert sha(archive.read('reviewed-cpu-proof.json')) == seal_receipt['reviewed_cpu_proof_sha256']
        for name, expected in plan['candidate_files'].items():
            assert sha(archive.read('candidate-source/' + name)) == expected
        with zipfile.ZipFile(io.BytesIO(archive.read('historical-runtime-source.zip'))) as historical:
            assert {n: sha(historical.read(n)) for n in historical.namelist()} == plan['historical_runtime_files']
    # Validate original immutable candidate source authority as well as the port seal.
    definition = plan['source_definition']
    assert definition['id'] == row['queue_row']
    authority = definition['candidate_archive']
    original_binding = definition['source_binding']
    if original_binding['receipt'] is not None:
        receipt = original_binding['receipt']
        if isinstance(receipt, list):
            import gzip
            assert receipt
            for item in receipt:
                snapshot = item['retained_snapshot']
                content = Path(snapshot['path']).read_bytes()
                assert sha(content) == snapshot['sha256'] and len(content) == snapshot['bytes']
                original = gzip.decompress(content)
                assert sha(original) == item['sha256'] and len(original) == item['bytes']
                assert definition['candidate_files'][Path(item['source']).name] == item['sha256']
        else:
            assert hashfile(Path(receipt['path'])) == receipt['sha256']
    for name, expected in original_binding['recorded_files_verified'].items():
        assert definition['candidate_files'][name] == expected
    for name, expected in plan['candidate_files'].items():
        assert definition['candidate_files'][name] == expected
    authority_path = Path(authority['path'])
    assert hashfile(authority_path) == authority['sha256']
    with zipfile.ZipFile(authority_path) as original:
        for name, expected in plan['candidate_files'].items():
            assert sha(original.read(name)) == expected, name

    result = read(source / 'result.json')
    init = read(source / 'initialization-receipt.json')
    runtime = read(source / 'runtime.json')
    assert result['status'] in ('PASS', 'FAIL') and result['task'] == 'mode_hold'
    assert result['backend'] == 'cuda' and not result['cpu_random'] and not result['init_only']
    assert result['initialization_fixture_sha256'] is None
    assert result['worker_sha256'] == plan['candidate_files']['probe.py']
    assert result['config'] == dict(plan['policy']['original_candidate_config'], device='cuda:0')
    assert init['candidate'] == row['candidate'] and init['source_plan_sha256'] == hashfile(plan_path)
    assert init['initializer_commit'] == plan['initializer_commit']
    assert not init['historical_tensor_fixture_loaded'] and not init['old22_scores_inherited']
    assert init['transform'] == proof['transform'] and init['serial_context_restored']
    expected_runtime = plan['runtime_expected']
    for key in ('torch', 'cuda', 'gpu', 'threads', 'interop_threads', 'deterministic', 'tf32', 'learner_default_device'):
        assert runtime[key] == expected_runtime[key], key
    assert runtime['serial_backward'] and runtime['matmul_precision'] == expected_runtime['float32_matmul_precision']
    assert runtime['environment']['CUBLAS_WORKSPACE_CONFIG'] == expected_runtime['CUBLAS_WORKSPACE_CONFIG']
    for name, path in runtime['historical_imports'].items():
        path = Path(path)
        relative = str(path.relative_to(plan['historical_runtime']))
        assert hashfile(path) == plan['historical_runtime_files'][relative], name

    initial, _ = checkpoint(source / 'new-initial-state.pt')
    initial_optimizers, _ = checkpoint(source / 'initial-values.pt')
    final, clocks = checkpoint(source / 'final-state.pt')
    fixture = read(ROOT / 'mode-hold-source/fixture-receipt.json')
    roles = ('generator', 'critic', 'prior')
    assert {role: initial[role] for role in roles} == proof['all_initial_material']
    assert {role: init['initial_material'][role] for role in roles} == proof['all_initial_material']
    for key in ('cpu_rng', 'data_rng'):
        assert initial[key] == fixture['expected_initial'][key], key
    assert initial['cuda_rng'] == [fixture['expected_initial']['cuda_rng']]
    assert init['initial_material']['data_rng_sha256'] == initial['data_rng']['sha256']
    assert list(final['streams'].values()) == [fixture['expected_final_data_rng']]
    parameters = proof['all_named_parameters_and_buffers']
    expected_optimizer_initial = [[*parameters['generator']['parameters'].values(), *parameters['prior']['parameters'].values()], list(parameters['critic']['parameters'].values())]
    assert initial_optimizers == expected_optimizer_initial
    assert result['proof']['initial_optimizers'] == [[{k:v for k,v in tensor.items() if k != 'dtype'} for tensor in group] for group in initial_optimizers]
    assert [len(c) for c in clocks] == [len(group) for group in initial_optimizers]
    steps = plan['policy']['evaluation_steps']
    assert steps == 1200
    assert all(c['step'] == steps and c['device'] == c['moment_device'] == 'cuda:0' for opt in clocks for c in opt)
    counter_proof = result['proof']['optimizers']
    assert len(counter_proof) == len(clocks)
    assert result['proof']['adam_calls'] == sum(v['calls'] for v in counter_proof.values()) == steps * len(clocks)
    for counter in counter_proof.values():
        assert counter['calls'] == steps and counter['device'] == 'cuda:0'
        assert counter['state_devices'] == ['cuda:0'] and counter['parameter_dtypes'] == ['torch.float32']
    assert result['checkpoint']['sha256'] == hashfile(source / 'final-state.pt')
    for key in ('modules', 'trainers', 'streams', 'optimizers', 'response_history'):
        assert result['checkpoint'][key] == len(final[key])
    assert not final['trainers']
    assert [v for k,v in final['scalars'].items() if k.endswith(':train_mode_hold:step')] == [steps - 1]
    # Own sparse update receipts are checked, not promoted to full per-update evidence.
    expected_receipt_steps = list(range(1,81)) + list(range(100, steps + 1, 50))
    for optimizer in range(len(clocks)):
        sampled = [r for r in result['mobility'] if r['optimizer'] == optimizer]
        assert [r['step'] for r in sampled] == expected_receipt_steps
    randomness = result['randomness']
    assert len(randomness['sha256']) == 64 and len(randomness['prefix']) == 64
    assert randomness['calls'] >= 64 and randomness['elements'] > 0

    observations = result['result']['observations']
    assert [r['step'] for r in observations] == list(range(50, steps + 1, 50))
    metrics = summary(observations)
    convergence = result['result']['convergence']
    for key in ('observations', 'passing_observations', 'passing_suffix'):
        assert convergence[key] == metrics[key]
    assert convergence['first_pass_step'] == metrics['first_arrival']
    assert convergence['complete'] is True and convergence['minimum_stable_checks'] == 5
    assert result['verdict']['convergence'] == convergence
    assert result['status'] == result['verdict']['status'] == ('PASS' if metrics['passing_suffix'] >= 5 else 'FAIL')
    assert result['result']['live'] == dict(modes=observations[-1]['modes'], hq=observations[-1]['hq'])
    assert result['spec']['steps'] == steps and result['spec']['thresholds'] == [['modes', '>=', 8], ['hq', '>=', .9]]
    mechanism_path = source / 'mechanism-receipt.json'
    mechanism = read(mechanism_path) if mechanism_path.exists() else None
    # Mechanism-specific call/switch counts are preserved without a KA2 template.
    # atexit may add final fields; both pre-exit and terminal views remain retained.
    limits = [
        'Historical custom research loop with its original eager CUDA Adam metadata; not public GANTrainer qualification.',
        'Inner host diagnostic labels do not override the strict eight-mode/HQ 0.9 final-five gate.',
        'No per-update batch receipt table or full per-update rates were emitted: exact sealed source, initial/final shared cursors, 206 sparse update records and original random-operation digest are retained.',
        'Random-operation digest is retained evidence, not an independent replay of the random stream.',
        'Final checkpoint is inspected for saved state and clocks; this does not prove complete fresh-process continuation, including mechanism module-global caches.',
        'Historical continuous eligibility is retained unchanged; no old quality scores are inherited.'
    ]
    return dict(status='PASS', scope='Independent stdlib-only retained source/runtime/raw-state/score audit; no Torch import or learner execution',
        source=str(source), candidate=row['candidate'], queue_row=row['queue_row'], prepared_index=str(index_path), prepared_index_sha256=hashfile(index_path),
        prepared_row=row, historical_eligibility=definition['historical_eligibility'],
        artifacts={f.name: hashfile(f) for f in source.iterdir() if f.is_file()},
        prepared_source_sha256=seal_receipt, original_candidate_authority=authority,
        quality_status=result['status'], **metrics, final=result['result']['live'], seconds=result['seconds'],
        initial_cuda_material_matches_own_cpu_proof=True, original_initial_optimizer_parameters_equal_own_CPU_proof=True,
        all_initial_global_and_shared_rng_equal_frozen=True, final_shared_cursor_equal_frozen=True,
        initialization_fixture_loaded=False, optimizer_clocks=clocks, optimizer_call_proof=counter_proof,
        sparse_update_receipts=len(result['mobility']), randomness=randomness,
        mechanism_pre_exit=result['regularizer_receipt'], mechanism_terminal=mechanism,
        no_inherited_quality=True, limits=limits,
        audit_source_sha256=hashfile(Path(__file__)), raw_reader_sha256=hashfile(ROOT / 'audit_public3_runtime.py'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prepared-index', type=Path, default=ROOT / 'research-screen-queue/prepared-index.json')
    parser.add_argument('--queue-row', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [row for row in read(args.prepared_index)['rows'] if row['queue_row'] == args.queue_row]
    assert len(rows) == 1
    out = audit(args.source.resolve(), rows[0], args.prepared_index.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    args.output.with_suffix('.md').write_text(f"# {out['candidate']} runtime audit\n\nIndependent source and retained-state audit PASS. Strict screen {out['quality_status']}: {out['passing_observations']}/24 observations, first arrival {out['first_arrival']}, final suffix {out['passing_suffix']}; final {out['final']['modes']}/8 modes, HQ {out['final']['hq']:.6f}. Cost {out['seconds']:.3f}s is retained without a comparative speed claim.\n\nThe exact original learner/configuration, sealed preparation, own CPU proof, captured initialized CUDA tensors and optimizer parameter copies agree. Frozen initial RNG and final shared sampling cursor agree. Own final optimizer clocks, 24 observations and sparse update records are present.\n\n" + '\n\n'.join(out['limits']) + '\n')
    print(json.dumps({k: out[k] for k in ('status', 'candidate', 'quality_status', 'passing_observations', 'passing_suffix', 'first_arrival', 'final')}))


if __name__ == '__main__':
    main()
