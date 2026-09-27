"""Verify and retain DV12's fixed unequal-mass follow-up without importing Torch."""
import argparse
import gzip
import json
from pathlib import Path
import zipfile

from audit_public3_runtime import checkpoint, read, sha

ROOT = Path(__file__).resolve().parent


def without_devices(value):
    if isinstance(value, dict):
        tensor = {'shape', 'dtype', 'sha256'} <= value.keys()
        return {k: without_devices(v) for k, v in value.items() if not (tensor and k == 'device')}
    if isinstance(value, list):
        return [without_devices(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    bundle = ROOT / 'port-source/new-init-dv12-unequal-mass'
    review = read(ROOT / 'dv12-vector-review/source-audit.json')
    assert review['status'] == 'PASS_SOURCE_AND_CPU_INITIALIZATION'
    assert sha((bundle / 'bundle-sha256.json').read_bytes()) == review['bundle_sha256']
    cpu_path = ROOT / 'dv12-vector-review/cpu/cpu-preflight.json'
    assert sha(cpu_path.read_bytes()) == review['cpu_receipt_sha256']
    cpu = read(cpu_path)
    artifacts = read(source / 'artifact-sha256.json')
    assert artifacts == {p.name: sha(p.read_bytes()) for p in source.iterdir()
                         if p.is_file() and p.name != 'artifact-sha256.json'}
    declaration = read(source / 'declaration.json')
    candidate = read(ROOT / 'port-source/api-dv12/candidate-declaration.json')
    assert declaration['candidate'] == candidate
    assert not declaration['old_initial_weights_loaded']
    assert declaration['task'] == read(bundle / 'task.json')
    with zipfile.ZipFile(source / 'source.zip') as archive:
        hashes = {name: sha(archive.read(name)) for name in archive.namelist()}
        assert len(hashes) == len(archive.namelist())
        assert hashes == declaration['source_sha256'] == cpu['source_sha256']
        for name, digest in read(bundle / 'bundle-sha256.json').items():
            assert hashes[name] == digest
        assert {n: v for n, v in hashes.items() if n.startswith('particlegan/')} == candidate['package_sha256']
    result = read(source / 'result.json')
    assert result['cpu_receipt_sha256'] == review['cpu_receipt_sha256']
    assert result['completed_steps'] == 1200
    assert result['candidate'] == candidate['candidate'] and result['task'] == 'vector_unequal_mass'
    for item in result['imports'].values():
        path = Path(item['path'])
        assert sha(path.read_bytes()) == item['sha256'] == candidate['package_sha256']['particlegan/' + path.name]
    initial, clocks0 = checkpoint(source / 'initial-state.pt')
    final, clocksN = checkpoint(source / 'final-state.pt')
    assert initial == without_devices(read(source / 'initial.json'))
    assert initial['trainer']['models'] == without_devices(cpu['initial_material']['models'])
    model_proof = read(source / 'initial-model-cpu-cuda-proof.json')
    assert model_proof['status'] == 'EXACT_ALL_MODEL_BYTES'
    assert initial['trainer']['models'] == without_devices(model_proof['models'])
    assert initial['trainer']['completed_steps'] == 0 and final['trainer']['completed_steps'] == 1200
    assert initial['trainer']['recipe'] == final['trainer']['recipe'] == result['recipe']
    assert result['recipe']['initialization'] == 'batch_feature_zero'
    assert result['recipe']['total_steps'] is None and result['recipe']['continuous_policy'] == 'dv12'
    assert clocks0 == [[], []]
    assert all(role and all(clock == dict(step=1200., device='cpu') for clock in role) for role in clocksN)
    batches = read(source / 'expected-batches.json')
    assert [row['step'] for row in batches] == list(range(1, 1201))
    assert final['data_rng'] == without_devices(batches[-1]['data_cursor'])
    assert final['trainer']['streams']['latent_generator'] == without_devices(batches[-1]['latent_cursor'])
    rates = [json.loads(line) for line in (source / 'learning-rates.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rates] == list(range(1, 1201))
    rows = [json.loads(line) for line in (source / 'metrics.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == list(range(50, 1201, 50))
    thresholds = declaration['task']['spec']['thresholds']
    def good(row):
        return all((row[k] <= bound if op == '<=' else row[k] >= bound) for k, op, bound in thresholds)
    assert all(op in ('<=', '>=') for _, op, _ in thresholds)
    passing = [row for row in rows if good(row)]
    suffix = 0
    for row in reversed(rows):
        if not good(row):
            break
        suffix += 1
    assert result['final'] == rows[-1]
    convergence = result['verdict']['convergence']
    assert convergence['passing_observations'] == len(passing) and convergence['passing_suffix'] == suffix
    assert result['status'] == ('PASS' if suffix >= 5 else 'FAIL')
    entry = dict(status='PASS', scope='Independent stdlib source, raw-state and scoring audit; no training',
        quality_status=result['status'], candidate=candidate['candidate'], task=result['task'], source=str(source),
        observations=24, passing_observations=len(passing), passing_suffix=suffix,
        first_arrival=passing[0]['step'] if passing else None,
        final={k: rows[-1][k] for k, _, _ in thresholds}, seconds=result['seconds'],
        old_result=dict(status='FAIL', passing_observations=2, passing_suffix=0,
                        component_covariance_error=.8843883238732815),
        final_failed_bounds=[dict(metric=k, value=rows[-1][k], op=op, threshold=bound)
                             for k, op, bound in thresholds
                             if not (rows[-1][k] <= bound if op == '<=' else rows[-1][k] >= bound)],
        source_zip_sha256=artifacts['source.zip'], cpu_receipt_sha256=result['cpu_receipt_sha256'],
        optimizer_clocks=clocksN, artifacts=artifacts,
        limits=['The sealed worker asserts actual data and latent draws against all1200 dry receipts; this audit verifies those retained expected receipts and the raw final cursors, without rerunning sampling.',
                'Candidate-specific controller/rates are retained, not independently replayed here.',
                'No first passing observation within1200 updates; this finite failure is not proof that later arrival is impossible.'])
    (ROOT / 'dv12-vector-runtime-audit.json').write_text(json.dumps(entry, indent=2) + '\n')
    destination = ROOT / 'followup-evidence/api-dv12-unequal-mass-new-init'
    destination.mkdir(parents=True, exist_ok=False)
    retained, external = {}, {}
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        content = path.read_bytes()
        if path.suffix == '.pt':
            external[path.name] = dict(path=str(path), sha256=sha(content), bytes=len(content))
            continue
        compressed = path.suffix in ('.json', '.jsonl') and len(content) > 100000
        target = destination / (path.name + '.gz' if compressed else path.name)
        target.write_bytes(gzip.compress(content, mtime=0) if compressed else content)
        retained[target.name] = dict(sha256=sha(target.read_bytes()), original_sha256=sha(content))
    (destination / 'archive-manifest.json').write_text(json.dumps(dict(
        audit=entry, retained=retained, external_checkpoints=external), indent=2) + '\n')
    table_path = ROOT / 'followup-results.json'
    table = read(table_path) if table_path.exists() else dict(results=[])
    assert not any(row['candidate'] == entry['candidate'] and row['task'] == entry['task'] for row in table['results'])
    table['results'].append(entry)
    table_path.write_text(json.dumps(table, indent=2) + '\n')
    print(json.dumps({k: entry[k] for k in ('quality_status', 'passing_observations', 'passing_suffix', 'final_failed_bounds')}))


if __name__ == '__main__':
    main()
