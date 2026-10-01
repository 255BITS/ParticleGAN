"""Close only actual complete latest-source original40update CUDA replay."""
import json
from pathlib import Path

from launch_replay import ROOT, sha, verify, write


def main():
    assert not (ROOT / 'CLOSED.json').exists()
    integrity = verify()
    completion = json.loads((ROOT / 'COMPLETION-replay.json').read_text())
    assert completion['status'] == 'COMPLETE' and completion['returncode'] == 0
    assert completion['fresh_training_updates'] == 0 and completion['fresh_replay_updates_total'] == 40
    path = ROOT / 'replay-RA15-partial-recovery.json'
    results = json.loads(path.read_text())
    assert set(results) == {'toy', 'mnist'}
    files = [p for p in ROOT.iterdir() if p.is_file()]
    files += list((ROOT / 'logs').glob('*.log'))
    for problem, result in results.items():
        assert result['status'] == 'PASS' and result['start_step'] == 1000 and result['steps_replayed'] == 10
        assert result['native_continuation_control']['status'] == 'PASS'
        assert result['restoration_semantic_bit_identical'] and result['semantic_state_bit_identical']
        assert result['losses_bit_identical'] and result['primary_sample_bytes_bit_identical']
        assert result['primary_sampling_preserves_training_state']
        assert len(result['branches']) == 2
        assert [r['step'] for r in result['per_update_comparison']] == list(range(1001, 1011))
        for branch in result['branches']:
            assert all(branch['restored_semantic_sections_identical'].values())
            assert [r['step'] for r in branch['update_fingerprints']] == list(range(1001, 1011))
            endpoint = Path(branch['endpoint'])
            assert sha(endpoint) == branch['endpoint_sha256']
            files.append(endpoint)
        files.append(ROOT / 'replay' / problem / 'RA15-partial-recovery/result.json')
    receipt = dict(status='PASS_LATEST_RA15_ORIGINAL_CUDA40_REPLAY', package_sha256=integrity['package_sha256'],
        source_integrity=integrity, fresh_training_updates=0, fresh_replay_updates_total=40,
        original_learned_fresh_training_source='RA13-settled', original_learned_updates_per_fixture=2000,
        replay_status={task: result['status'] for task, result in results.items()},
        native_and_CPU_map_loss_state_sample_bits_identical=True,
        native_matches_pinned_original_control=True, checkpoint_alias_protocol_preserved=True,
        file_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))})
    write(ROOT / 'CLOSED.json', receipt)
    print(json.dumps(dict(status=receipt['status'], closed_sha256=sha(ROOT / 'CLOSED.json'))), flush=True)


if __name__ == '__main__':
    main()
