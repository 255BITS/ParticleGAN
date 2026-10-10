"""Bind the completed single prototype to a prospective production selection."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / 'integration/review/training-regression/post-ra9-quality/mean-category-transport'
OUTPUT = ROOT / 'quality/results/RA10-selection.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_maps(value):
    count = 0
    if isinstance(value, dict):
        for name, item in value.items():
            if isinstance(name, str) and name.startswith('/') and isinstance(item, str) and len(item) == 64:
                assert sha(name) == item, name
                count += 1
            count += verify_maps(item)
    elif isinstance(value, list):
        count += sum(verify_maps(item) for item in value)
    return count


def main():
    assert not OUTPUT.exists()
    receipt = read(PROBE / 'receipt.json')
    frozen = read(PROBE / 'FROZEN.json')
    assert receipt['status'] == frozen['status'] == 'PASS'
    assert receipt['post_exit'] and frozen['post_exit'] and receipt['exit_code'] == 0
    assert receipt['numerical_attempts'] == 1 and receipt['failed_numerical_attempts'] == 0
    maps = verify_maps(receipt) + verify_maps(frozen)
    result = read(PROBE / 'attempt1/result.json')
    assert result['status'] == 'PASS_FIXED_PROTOTYPE' and result['single_fixed_run']
    assert result['no_production_patch'] and result['no_quality_acceptance']
    assert [row['case'] for row in result['cases']] == ['grid', 'toy']
    grid, toy = result['cases']
    assert grid['witness']['valid'] and grid['witness']['fires'] and grid['witness']['lower_bound'] > 0
    assert grid['preview']['accepted'] == 936 and grid['pair_supply']['total_budget'] == 1000
    assert grid['scratch_application']['objective_after_actual'] < grid['preview']['objective_before']
    for key in ('category_group_supported_ledgers_exact', 'original_sources_untouched',
                'exact_row_optimizer_history_inheritance', 'complete_moved_rows',
                'no_commit_noise_draw', 'packet_consumption_rejects_second_commit'):
        assert grid['scratch_application'][key], key
    assert toy['witness']['valid'] and not toy['witness']['fires']
    assert toy['pair_supply']['residual_budget'] == 0 and toy['preview']['accepted'] == 0
    for row in (grid, toy):
        for key in ('saved_state_unchanged', 'saved_file_unchanged', 'global_CPU_RNG_unchanged',
                    'source_streams_unchanged', 'raw_FAST_and_EMA_views', 'no_CUDA_context'):
            assert row[key], key
        assert row['emitted_clouds'] == row['training_updates'] == 0
        assert not row['witness']['authoritative']
    paths = [Path(__file__), ROOT / 'quality/RA10-PLAN.md', PROBE / 'SOURCE-FROZEN.json',
             PROBE / 'receipt.json', PROBE / 'FROZEN.json', PROBE / 'attempt1/result.json',
             ROOT / 'quality/ra9/READY.json', ROOT / 'configs/overrides-CB64-RA9.json',
             ROOT / 'quality/results/CB64-RA9.json']
    value = dict(status='PROSPECTIVE_MEAN_COPY_CANDIDATE_SELECTED', variant='CB64-RA10',
        selected_utc=datetime.now(timezone.utc).isoformat(), verified_map_entries=maps,
        numerical_source_sha256={str(path): sha(path) for path in paths},
        base_package=str(ROOT / 'pkg-CB64-RA9'), config_change=None,
        backend_schema=9, trainer_schema=5,
        evidence=dict(grid_lower_bound=grid['witness']['lower_bound'],
            grid_accepted_copies=grid['preview']['accepted'],
            objective_before=grid['preview']['objective_before'],
            objective_after=grid['scratch_application']['objective_after_actual'],
            toy_lower_bound=toy['witness']['lower_bound'], toy_actions=0),
        production_source_tested=False, quality_verdict=None,
        limits='One saved-state CPU feature-objective prototype; no emitted quality, independent prospective certificate or CUDA training replication.',
        next_gate='Independently reviewed and frozen production contracts, then fresh unchanged full CUDA toy and canonical grid; required replay and portability regressions follow both passing.')
    OUTPUT.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(status=value['status'], selection_sha256=sha(OUTPUT), verified_map_entries=maps)))


if __name__ == '__main__':
    main()
