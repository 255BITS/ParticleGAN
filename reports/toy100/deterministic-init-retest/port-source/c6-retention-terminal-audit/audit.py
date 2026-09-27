"""CPU checkpoint deserialization/audit only; never constructs a model or trains."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import shutil
import zipfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

P = Path(__file__).resolve().parents[2]
BASE = Path('/ml2/hypergan/gan-attempts/deterministic-init-retest-20260927/20260927T025403Z/c6-retention/20260927T025403Z-1946078/repo/reports/reviewed-probe-output/20260927T025536Z')
OUT = P / 'supplementary-evidence'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_with_placement(path):
    placements = {}
    def mapper(storage, location):
        placements[storage.data_ptr()] = location
        return storage  # Audit on CPU; original serialized device retained above.
    state = torch.load(path, map_location=mapper, weights_only=False)
    assert not torch.cuda.is_initialized()
    return state, placements


def exact(value, placements):
    if isinstance(value, torch.Tensor):
        assert value.device.type == 'cpu'
        cpu = value.detach().contiguous()
        return dict(shape=list(value.shape), dtype=str(value.dtype),
                    device=placements[value.untyped_storage().data_ptr()],
                    sha256=sha(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()))
    if isinstance(value, dict):
        return {str(k): exact(v, placements) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [exact(v, placements) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(type(value))


def digest(tensor):
    value = tensor.detach().contiguous()
    return sha(json.dumps(['tensor', str(value.dtype), list(value.shape)]).encode()
               + value.reshape(-1).view(torch.uint8).numpy().tobytes())


def summarize(points):
    passing = lambda p: p['modes'] >= 8 and p['hq'] >= .9
    first = next(p['step'] for p in points if passing(p))
    since = [p for p in points if p['step'] >= first]
    suffix = 0
    for p in reversed(points):
        if not passing(p): break
        suffix += 1
    return dict(first_arrival=first, passing_since_arrival=sum(passing(p) for p in since),
                observations_since_arrival=len(since), departures=[p['step'] for p in since if not passing(p)],
                min_hq_since_arrival=min(p['hq'] for p in since), min_modes_since_arrival=min(p['modes'] for p in since),
                final_passing_suffix=suffix, final_suffix_start=points[-suffix]['step'] if suffix else None)


def archive(source, target):
    target.mkdir(parents=True, exist_ok=False)
    receipt = {}
    for f in sorted(source.iterdir()):
        if not f.is_file(): continue
        data = f.read_bytes()
        compressed = f.suffix in ('.pt', '.jsonl') or len(data) > 200_000 and f.suffix == '.json'
        name = f.name + '.gz' if compressed else f.name
        payload = gzip.compress(data, mtime=0) if compressed else data
        (target / name).write_bytes(payload)
        assert (gzip.decompress(payload) if compressed else payload) == data
        receipt[f.name] = dict(archived_file=name, original_sha256=sha(data), archive_sha256=sha(payload), bytes=len(data))
    return receipt


def main():
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    seal = read(P / 'port-source/c6-retention-continuation/bundle-sha256.json')
    rows = []
    for n in [6]:
        name = 'api-c6-retention'
        source = BASE / name
        result = read(source / 'result.json')
        assert result['status'] == 'COMPLETE_SUPPLEMENT' and result['updates'] == 2400
        artifacts = read(source / 'artifact-sha256.json')
        assert all(sha((source / file).read_bytes()) == h for file, h in artifacts.items())
        plan = read(source / 'continuation-protocol.json')
        original = plan['original_artifacts']
        for value in original.values():
            assert sha(Path(value['path']).read_bytes()) == value['sha256']
        declaration = read(Path(plan['candidate_declaration']))
        with zipfile.ZipFile(source / 'source.zip') as z:
            assert len(z.namelist()) == len(set(z.namelist()))
            for file, h in seal.items(): assert sha(z.read(file)) == h
            for file, h in declaration['package_sha256'].items(): assert sha(z.read(file)) == h
            assert json.loads(z.read('candidate-declaration.json')) == declaration
        old, old_locations = load_with_placement(original['final-state.pt']['path'])
        restore = read(source / 'restore-proof.json')
        assert restore['status'] == 'EXACT_SERIALIZED_STATE_AND_DEVICES'
        assert restore['state'] == exact(old, old_locations)
        assert restore['original_checkpoint_sha256'] == original['final-state.pt']['sha256']
        initial, initial_locations = load_with_placement(original['initial-state.pt']['path'])
        final, final_locations = load_with_placement(source / 'final-state.pt')
        assert final['trainer']['completed_steps'] == 2400
        assert final['identity'] == old['identity'] == initial['identity']
        assert final['trainer']['recipe'] == old['trainer']['recipe'] == initial['trainer']['recipe']
        assert json.loads(json.dumps(final['trainer']['recipe'])) == result['recipe'] == declaration['resolved_recipe']
        for state, locations, steps in [(old, old_locations, 1200), (final, final_locations, 2400)]:
            for opt in state['trainer']['optimizers']:
                for entry in opt['state'].values():
                    if 'step' not in entry: continue
                    assert locations[entry['step'].untyped_storage().data_ptr()] == 'cpu'
                    assert float(entry['step']) == steps
                    for key in ('exp_avg', 'exp_avg_sq'):
                        assert locations[entry[key].untyped_storage().data_ptr()] == 'cuda:0'
            assert torch.equal(state['data_rng'], state['trainer']['streams']['latent_generator'])
        before = lines(Path(original['metrics.jsonl']['path']))
        after = lines(source / 'metrics.jsonl')
        assert [p['step'] for p in before] == list(range(50,1201,50))
        assert [p['step'] for p in after] == list(range(1250,2401,50))
        restored = read(source / 'restored-observation.json')
        assert all(restored[k] == before[-1][k] for k in restored)
        expected = read(source / 'expected-extension-batches.json')
        actual = lines(source / 'batch-receipts.jsonl')
        assert expected == actual and [p['step'] for p in actual] == list(range(1201,2401))
        assert actual[-1]['accepted_cursor'] == digest(final['data_rng'])
        sampling = read(source / 'sampling-preflight.json')
        assert sampling['status'] == 'PASS' and sampling['matched_original_batches'] == 1200
        assert sampling['declared_extension_batches'] == 1200 and sampling['learner_updates'] == 0
        assert sampling['final_cursor'] == exact(final['data_rng'], final_locations)
        rates = lines(source / 'learning-rates.jsonl')
        assert [r['step'] for r in rates] == list(range(1201,2401))
        for row in rates:
            groups = row['applied_group_rates']
            assert [[g['lr'] for g in role] for role in groups] == [[.00425, .0085], [.00425]]
            assert row['input_noise'] == 0 and row['output_noise'] == .029
        summary = summarize(before + after)
        assert all(result[k] == v for k,v in summary.items() if k != 'final_suffix_start')
        assert result['original_gate_status'] == 'FAIL' and result['original_gate_unchanged'] is True
        target = OUT / name
        archive_manifest = archive(source, target)
        audit = dict(status='PASS_INDEPENDENT_ARTIFACT_AND_CPU_STATE_AUDIT', candidate=result['candidate'],
            source_directory=str(source), original_gate_status='FAIL', supplement_status='COMPLETE_SUPPLEMENT',
            summary=summary, final_hq=after[-1]['hq'], final_modes=after[-1]['modes'],
            own_serialized_checkpoint_restore_bytes_and_devices_equal=True,
            original_live_and_ema_endpoint_reproduced=True, extension_batches_exact=1200,
            original_batch_replay_runtime_receipt=1200, new_observations=24,
            full_applied_rates_and_noise_retained=True,
            rate_policy='Original C6 constant applied rates; total_steps7000/continuousTrue/LRfloors1 remain unchanged',
            applied_network_lr_range=[min(r['applied_group_rates'][0][0]['lr'] for r in rates), max(r['applied_group_rates'][0][0]['lr'] for r in rates)],
            applied_prior_lr_range=[min(r['applied_group_rates'][0][1]['lr'] for r in rates), max(r['applied_group_rates'][0][1]['lr'] for r in rates)], final_native_CPU_clocks_and_CUDA_moments_verified=True,
            cuda_initialized=False, no_model_construction_or_training=True,
            independent_freshprocess_update_replay='NOT_CLAIMED: no uninterrupted future control was run',
            checkpoint_mode_scope=plan['checkpoint_coverage_caveat'], archive=archive_manifest)
        dump = lambda path, value: path.write_text(json.dumps(value,indent=2)+'\n')
        dump(target / 'independent-audit.json', audit)
        rows.append({k:v for k,v in audit.items() if k != 'archive'} | {'archive_directory':str(target),
                    'independent_audit_sha256':sha((target/'independent-audit.json').read_bytes())})
    assert not torch.cuda.is_initialized()
    report = dict(status='PASS_C6_SUPPLEMENT_ARTIFACT_AUDIT', scope='Separate supplementary retention; original1200scores unchanged', rows=rows)
    (P/'port-source/c6-retention-terminal-audit/audit.json').write_text(json.dumps(report,indent=2)+'\n')
    table=['Original 1,200-step scores remain FAIL. These are unchanged checkpoint continuations, not replacement screen scores.','',
           '| Candidate | First arrival | Retained since arrival | Departures | Minimum HQ | Final suffix |',
           '|---|---:|---:|---|---:|---:|']
    for row in rows:
        s=row['summary'];table.append(f"| {row['candidate']} | {s['first_arrival']} | {s['passing_since_arrival']}/{s['observations_since_arrival']} | {', '.join(map(str,s['departures'])) or 'None'} | {s['min_hq_since_arrival']:.6f} | {s['final_passing_suffix']} from {s['final_suffix_start']} |")
    table += ['', 'C6 first reaches eight modes/HQ≥0.9 at1,200, then departs at1,800,1,850,1,900 and2,250. Worst retained observation has three modes/HQ.183349609375. It later returns to eight modes with final suffix3; this is a measured retention failure, not just an early arrival deadline.', '',
              'Independent CPU-only deserialization verified saved tensor bytes and original placements against restore receipts, including native CPU Adam clocks and CUDA moments. Complete old endpoint live/EMA values, 1,200 extension batch receipts, 24 observations, unchanged recipe, constant applied rates and noise and source hashes match. No model construction/training or GPU initialization occurred. Original module-mode omission and lack of an uninterrupted future update-control comparison remain explicit limits.']
    (P/'port-source/c6-retention-terminal-audit/table.md').write_text('\n'.join(table)+'\n')
    print(json.dumps(dict(status=report['status'], candidates=len(rows), cuda_initialized=False)))


if __name__ == '__main__': main()
