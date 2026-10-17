"""Select one prospective output-moment package from closed fixed evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / 'performance/sampler-regression/cpu-plan-review/post-ra10-quality/linear-output-mean-prototype'
OUTPUT = ROOT / 'quality/results/RA11-selection.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_map(manifest, key):
    seen = {}
    for raw, digest in read(manifest).get(key, {}).items():
        path = Path(raw)
        if not path.is_absolute():
            matches = {p.resolve() for p in (manifest.parent / path,
                       manifest.parent.parent / path, ROOT / path) if p.is_file()}
            assert len(matches) == 1, (manifest, raw)
            path = matches.pop()
        assert sha(path) == digest, path
        seen[str(path)] = digest
    return seen


def main():
    assert not OUTPUT.exists()
    source = PROBE / 'SOURCE-FROZEN.json'
    final = PROBE / 'attempt1/FINAL-FROZEN.json'
    assert sha(source) == '7497913b6b0fdec2da849d46386b9a2c64dabab754c01a9c9fa7b8abe0f36d05'
    assert sha(final) == '8c6b11bacf00940c8c3164a6f7a06aa1b420b62fc1701fa05ed2a39238e965a4'
    guards = verify_map(source, 'source_and_input_sha256')
    guards.update(verify_map(final, 'files'))
    closed = read(final)
    result_path = PROBE / 'attempt1/result.json'
    result = read(result_path)
    assert closed['status'] == result['status'] == 'PASS'
    assert sha(result_path) == closed['result_sha256']
    exit_receipt = read(PROBE / 'EXIT-attempt1.json')
    assert exit_receipt['returncode'] == 0 and exit_receipt['numerical_invocations_started'] == 1
    assert result['device'] == 'cpu' and not result['cuda_initialized']
    assert result['production_package_unchanged'] and result['no_resumable_newlaw_checkpoint']
    assert not result['prototype_source_changed']
    for key in ('new_training_steps', 'new_optimizer_steps', 'new_quality_emissions', 'new_seed_experiments'):
        assert result[key] == 0, key
    grid, toy = result['records']
    assert grid['case'] == 'grid' and toy['case'] == 'toy'
    assert grid['mean'] > 0 and grid['witness']['lower_bound'] > 0
    assert toy['mean'] == 0 and toy['witness']['lower_bound'] <= 0
    for record in (grid, toy):
        assert record['status'] == 'PASS'
        assert all(value for value in record['checks'].values() if isinstance(value, bool))
    moments = read(PROBE / 'attempt1/grid/raw-moments.json')
    for view in ('FAST', 'EMA'):
        delta = moments['change'][view]
        assert delta['group_transitions'] == 0
        assert delta['raw_mean_objective_after'] < delta['raw_mean_objective_before']
        assert delta['centered_covariance_trace_after'] < delta['centered_covariance_trace_before']
    paths = [Path(__file__), ROOT / 'quality/RA11-PLAN.md', source, final,
             result_path, PROBE / 'ROOT-GO.json', PROBE / 'EXIT-attempt1.json',
             ROOT / 'quality/results/CB64-RA10.json', ROOT / 'quality/ra10/READY.json',
             ROOT / 'configs/overrides-CB64-RA10.json']
    for raw, digest in read(PROBE / 'ROOT-GO.json')['independent_review_sha256'].items():
        path = Path(raw)
        assert sha(path) == digest
        paths.append(path)
    value = dict(status='PROSPECTIVE_OUTPUT_MOMENT_CANDIDATE_SELECTED',
        variant='CB64-RA11', selected_utc=datetime.now(timezone.utc).isoformat(),
        source_and_input_sha256={str(p): sha(p) for p in paths},
        verified_unique_raw_files=len(guards), base_package=str(ROOT / 'pkg-CB64-RA10'),
        config_change=None, backend_schema=10, trainer_schema=5,
        evidence=dict(grid_mean_copies=grid['mean'], grid_lower_bound=grid['witness']['lower_bound'],
                      raw_moment_changes=moments['change'], toy_mean_copies=toy['mean'],
                      toy_lower_bound=toy['witness']['lower_bound']),
        production_source_tested=False, quality_verdict=None,
        limits='One fixed CPU scratch reaction per case; no emitted quality or scaling certificate.',
        next_gate='Frozen production contracts and short CUDA mechanics, then unchanged final CUDA toy and full canonical grid; replay/portability after both pass.')
    with OUTPUT.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(status=value['status'], selection_sha256=sha(OUTPUT),
                          verified_unique_raw_files=len(guards))))


if __name__ == '__main__':
    main()
