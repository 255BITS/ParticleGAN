"""Read-only raw provenance/JSON audit of the closed root CUDA mechanics phase."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / 'integration/review/ra11-mechanics-gpu'
OWNER = ROOT / 'integration/review/training-regression/post-ra10-quality/linear-output-production'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def verify(mapping):
    for p, h in mapping.items():
        assert sha(p) == h, p


def main():
    prep = read(HERE / 'PREPARATION-FROZEN.json')
    verify(prep['source_and_output_sha256'])
    phase_path = ROOT / 'quality/ra11-mechanics-phase/READY.json'
    result_path = OUTPUT / 'result.json'
    assert sha(phase_path) == 'afcb27d8bf375c1f259c76544653663fe893c5ac279402f364f49bc785193d5e'
    assert sha(result_path) == 'bba1f2ddc2bb697b3eaf9066b5c9851997b5836c7716f6454b87383250f66a98'
    phase, result = read(phase_path), read(result_path)
    launch, closed = read(OUTPUT / 'LAUNCH.json'), read(OUTPUT / 'PHASE-RESULT.json')
    root_ready, owner_ready = read(ROOT / 'quality/ra11/READY.json'), read(OWNER / 'READY.json')
    verify(phase['source_sha256'])
    verify(result['source_and_input_sha256'])
    assert phase['package_sha256'] == root_ready['package_sha256'] == owner_ready['package_sha256']
    assert phase['config_sha256'] == root_ready['config_sha256'] == owner_ready['config_sha256']
    assert root_ready['package_source_sha256'] == owner_ready['package_source_sha256']
    assert launch['command'] == phase['command'] == owner_ready['gpu_command']
    assert launch['command'][-1] == str(result_path)
    assert launch['pid'] == 1232431 and launch['startticks'] == '168642194'
    assert launch['phase_ready_sha256'] == closed['phase_ready_sha256'] == sha(phase_path)
    assert closed['result_sha256'] == sha(result_path)
    assert launch['gpu_uuid'] == closed['gpu_uuid'] == phase['gpu_uuid'] == 'GPU-72c1b506-891d-b8bc-b353-e020585e1c47'
    assert launch['numerical_parallelism'] == closed['numerical_parallelism'] == phase['numerical_parallelism'] == 1
    assert launch['source_integrity'] == closed['source_integrity'] == 'VALID'
    assert closed['status'] == result['status'] == 'PASS' and closed['returncode'] == 0
    assert closed['original_supervisor_still_parked'] is True
    assert closed['quality_verdict'] is result['quality_verdict'] is None
    assert datetime.fromisoformat(closed['finished_utc']) >= datetime.fromisoformat(launch['started_utc'])
    assert result['device'] == 'cuda:0' and result['cuda_initialized'] is True
    assert result['fixture_version'] == 2
    assert result['source_preseal_sha256'] == sha(OWNER / 'SOURCE-FROZEN.json')
    assert all(result[k] == 0 for k in ('new_training_steps', 'new_optimizer_steps', 'new_quality_emissions', 'new_seed_experiments'))
    wrapper = (ROOT / 'quality/run_ra10_mechanics.py').read_text()
    assert 'pass_fds=(lock.fileno(),)' in wrapper
    assert wrapper.index('verify(frozen)') < wrapper.index('subprocess.Popen')
    assert wrapper.rindex('verify(frozen)') > wrapper.index('child.wait()')
    helper = (OWNER / 'run_mechanics.py').read_text()
    assert helper.index('seal = verify()') < helper.index('import torch')
    assert 'torch.cuda.set_per_process_memory_fraction(.2, device)' in helper
    assert 'torch.backends.cuda.matmul.allow_tf32 = False' in helper
    assert 'torch.backends.cudnn.allow_tf32 = False' in helper
    assert 'torch.use_deterministic_algorithms(True)' in helper
    summaries = []
    assert [r['case'] for r in result['records']] == ['grid', 'toy']
    for rec in result['records']:
        label = rec['case']
        assert rec['status'] == 'PASS' and all(v is True for v in rec['checks'].values())
        verify(rec['output_sha256'])
        trace = read(OUTPUT / label / 'trace.json')
        provenance = read(OUTPUT / label / 'provenance.json')
        verify(provenance['input_binding'])
        verify({str(OUTPUT / label / p): h for p, h in provenance['output_sha256'].items()})
        assert provenance['case'] == label and provenance['device'] == result['device']
        assert provenance['seed'] == 314159
        assert provenance['package_root'] == str(ROOT / 'pkg-CB64-RA11')
        assert provenance['source_model_table_FIFO_raw_tensors'] is True
        assert 'fresh native backend10 constructor' in provenance['construction']
        assert provenance['no_optimizer_or_GAN_gradient_step'] is True
        assert (provenance['completed_steps_before'], provenance['completed_steps_after']) == (0, 1)
        assert sha(provenance['original_checkpoint']) == provenance['original_checkpoint_sha256']
        assert trace['mean_transport'] == rec['witness'] and trace['checks'] == rec['checks']
        assert len(trace['mean_children']) == len(trace['mean_parents']) == rec['mean']
        assert len(trace['ordinary_children']) == rec['ordinary'] - rec['mean']
        assert rec['ordinary'] <= int(.05 * rec['rows'])
        old_calls = trace['old_copy_calls']
        children = [row for call in old_calls for row in call['children']] + trace['novel_children'] + trace['mean_children']
        sources = [row for call in old_calls for row in call['parents']] + trace['novel_source_seeds'] + trace['mean_parents']
        assert len(set(children)) == len(children) and len(set(sources)) == len(sources)
        assert not set(children) & set(sources)
        assert sorted(children) == sorted(trace['moved_rows'])
        assert {r['kind'] for r in trace['caller_hooks']} == {'rebase', 'row_evidence_reset'}
        assert all(sorted(r['rows']) == sorted(children) for r in trace['caller_hooks'])
        assert not set(trace['mean_children'] + trace['mean_parents']) & set(trace['prefix_reserved'])
        if label == 'grid':
            assert rec['rows'] == 20000 and rec['cells'] == 128 and rec['mean'] == rec['ordinary'] == 908
            detail = trace['preview_detail']
            assert detail['attempts'] == rec['witness']['attempts'] == 1000
            assert detail['accepted'] == rec['witness']['moves'] == 908
            assert detail['paired_noise_draws'] == 1 and detail['no_redraw'] is True
            assert rec['witness']['lower_bound'] > 0
            assert rec['witness']['objective_after_mean'] < rec['witness']['objective_before_mean']
        else:
            assert rec['rows'] == 1024 and rec['cells'] == 64 and rec['ordinary'] == 51 and rec['mean'] == 0
            assert rec['witness']['status'] == 'veto' and rec['witness']['attempts'] == 0
            assert trace['preview_detail'] is None and trace['packet_fingerprint'] is None
        summaries.append(dict(case=label,ordinary=rec['ordinary'],mean=rec['mean'],rows=rec['rows'],cells=rec['cells'],checks=len(rec['checks'])))
    verify(prep['source_and_output_sha256'])
    receipt = dict(status='PASS', utc=datetime.now(timezone.utc).isoformat(), scope='READ_ONLY_CLOSED_CUDA_MECHANICS_PROVENANCE',
        phase_ready_sha256=sha(phase_path), result_sha256=sha(result_path), package_sha256=root_ready['package_sha256'],
        config_sha256=root_ready['config_sha256'], guarded_sources=len(phase['source_sha256']),
        records=summaries, raw_maps_exact=True, unchanged_owner_command=True, source_integrity='VALID',
        original_seeds_and_inputs=True, serial_physical_gpu0=True, fixture_fresh_backend10=True,
        observed_complete_phase_hook_unions=True, declared_packet_checks_all_PASS=True,
        no_postexecution_PT_deserialization=True, numerical_reruns=0, quality_verdict=None,
        limitations=['Checks numerical assertions through the closed authoritative mechanics report; does not independently rerun them.',
                     'One reaction on fresh backend10 fixtures; device-native inputs/seeds, no historical or cross-device replay.',
                     'No quality, long-run stationarity or serving-population certificate.'])
    assert not (HERE / 'receipt.json').exists()
    (HERE / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', guarded_sources=receipt['guarded_sources'], records=summaries, numerical_reruns=0)))


if __name__ == '__main__':
    main()
