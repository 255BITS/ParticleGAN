"""Read-only artifact audit. Standard library only; does not import Torch."""
from pathlib import Path
from collections import OrderedDict
import gzip, hashlib, io, json, math, pickle, struct, zipfile

ROOT = Path(__file__).resolve().parent
BATCH = Path('/ml2/hypergan/gan-attempts/deterministic-init-retest-20260927/20260927T020007Z')
sha = lambda b: hashlib.sha256(b).hexdigest()
read = lambda p: json.loads(p.read_bytes())

class Tensor:
    def __init__(self, storage, offset, shape, stride):
        self.storage, self.offset, self.shape, self.stride = storage, offset, shape, stride

def rebuild(storage, offset, shape, stride, *ignored):
    return Tensor(storage, offset, shape, stride)

class Reader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'): return OrderedDict
        if (module, name) == ('torch._utils', '_rebuild_tensor_v2'): return rebuild
        if module == 'torch' and name in ('FloatStorage', 'ByteStorage'): return name
        raise AssertionError((module, name))
    def persistent_load(self, value):
        assert len(value) == 5 and value[0] == 'storage'
        return value[1:]

def checkpoint(path):
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith('/data.pkl')]
        assert len(names) == 1
        name = names[0]; prefix = name[:-8]
        assert archive.read(prefix + 'byteorder') == b'little'
        envelope = Reader(io.BytesIO(archive.read(name))).load()
        def raw(tensor):
            kind, key, device, count = tensor.storage
            width, dtype = {'FloatStorage': (4, 'torch.float32'), 'ByteStorage': (1, 'torch.uint8')}[kind]
            value = archive.read(prefix + 'data/' + key)
            assert len(value) == count * width
            expected = 1
            for length, stride in reversed(list(zip(tensor.shape, tensor.stride))):
                assert length <= 1 or stride == expected
                expected *= length
            value = value[tensor.offset * width:(tensor.offset + math.prod(tensor.shape)) * width]
            assert len(value) == math.prod(tensor.shape) * width
            return value, dtype
        def material(value):
            if isinstance(value, Tensor):
                binary, dtype = raw(value)
                return dict(shape=list(value.shape), dtype=dtype, sha256=sha(binary))
            if isinstance(value, dict): return {str(k): material(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)): return [material(v) for v in value]
            assert value is None or isinstance(value, (str, int, float, bool))
            return value
        clock = []
        for opt in envelope['trainer']['optimizers']:
            entries = []
            for state in opt['state'].values():
                t = state['step']; binary, dtype = raw(t)
                assert t.shape == () and dtype == 'torch.float32'
                assert state['exp_avg'].storage[2] == state['exp_avg_sq'].storage[2] == 'cuda:0'
                entries.append(dict(step=struct.unpack('<f', binary)[0], device=t.storage[2]))
            clock.append(entries)
        return material(envelope), clock

def main():
    seal = read(ROOT/'harness-sha256.json')['files']
    fixture = read(ROOT/'mode-hold-source/fixture-receipt.json')
    expected_batches = [json.loads(x) for x in gzip.decompress((ROOT/'mode-hold-source/batch-receipts.jsonl.gz').read_bytes()).splitlines()]
    results = []
    for p in sorted(BATCH.glob('*/*/repo/reports/fixed-init-mode-hold')):
        alias = p.parents[3].name
        decl = read(ROOT/(alias+'-declaration.json'))
        assert read(p/'declaration.json') == decl
        manifest = read(p/'artifact-sha256.json')
        assert {str(f.relative_to(p)): sha(f.read_bytes()) for f in p.rglob('*') if f.is_file() and f.name != 'artifact-sha256.json'} == manifest
        with zipfile.ZipFile(p/'source.zip') as z:
            assert {n:sha(z.read(n)) for n in z.namelist() if n.startswith('particlegan/')} == decl['package_sha256']
            for n, digest in seal.items():
                if n in z.namelist(): assert sha(z.read(n)) == digest
            assert sha(z.read('candidate-declaration.json')) == seal[alias+'-declaration.json']
        source = read(p/'source-preflight.json'); runtime = read(p/'runtime.json'); init = read(p/'initial.json')
        assert source['status'] == 'PASS' and source['declaration_sha256'] == seal[alias+'-declaration.json']
        assert source['protocol_sha256'] == seal['protocol.json']
        assert source['harness_sha256'] == {n:seal[n] for n in source['harness_sha256']}
        assert runtime['learner_default_device'] == 'cpu' and runtime['serial_backward'] is True
        assert (runtime['torch'],runtime['cuda'],runtime['gpu']) == ('2.13.0+cu126','12.6','NVIDIA RTX A6000')
        assert runtime['deterministic'] and not runtime['tf32'] and runtime['threads'] == runtime['interop_threads'] == 1
        for item in runtime['imports'].values():
            path = Path(item['path']); key = 'particlegan/' + path.name
            assert item['sha256'] == decl['package_sha256'][key] == sha(path.read_bytes())
        assert read(p/'sampling-preflight.json') == dict(status='PASS',matched_batches=1200,actual_cuda_draws=True,optimizer_updates=0,training_stream_unchanged=True)
        batches = [json.loads(x) for x in (p/'batch-receipts.jsonl').read_text().splitlines()]
        assert batches == expected_batches
        observations = [json.loads(x) for x in (p/'metrics.jsonl').read_text().splitlines()]
        assert [x['step'] for x in observations] == list(range(50,1201,50))
        passing = [x for x in observations if x['modes'] >= 8 and x['hq'] >= .9]
        suffix = 0
        for row in reversed(observations):
            if row['modes'] < 8 or row['hq'] < .9: break
            suffix += 1
        result = read(p/'result.json')
        assert result['metrics']['final'] == observations[-1]
        convergence = result['metrics']['verdict']['convergence']
        assert convergence['passing_observations'] == len(passing) and convergence['passing_suffix'] == suffix
        assert result['metrics']['updates'] == result['metrics']['matched_batch_receipts'] == 1200
        assert result['status'] == ('PASS' if suffix >= 5 else 'FAIL')
        rates = [json.loads(x) for x in (p/'learning-rates.jsonl').read_text().splitlines()]
        assert [x['step'] for x in rates] == list(range(1,1201))
        recipe = runtime['recipe']; horizon = recipe['total_steps']; base = recipe['lr']
        def scale(step, n, floor):
            start = recipe['lr_anneal_start'] * n
            progress = min(1., max(0., (step-start)/(n-start)))
            return floor + (1-floor)*.5*(1+math.cos(math.pi*progress))
        for row in rates:
            step = row['step'] - 1
            net = scale(step, min(horizon,recipe['network_lr_horizon_cap']),recipe['network_lr_floor'])
            prior = scale(step,horizon,recipe['lr_floor'])
            expected = [[base*net,base*recipe['prior_lr_mult']*prior],[base*recipe['d_lr_mult']*net]]
            assert [[g['lr'] for g in groups] for groups in row['applied_group_rates']] == expected
            assert row['input_noise'] == recipe['input_noise_std']*max(0.,1-step/(recipe['input_noise_anneal_end']*horizon))
            assert row['output_noise'] == recipe['output_noise_std']*min(1.,step/(recipe['output_noise_warmup']*horizon))
        raw_init, clocks0 = checkpoint(p/'initial-state.pt'); raw_final, clocksN = checkpoint(p/'final-state.pt')
        assert raw_init['trainer']['completed_steps'] == 0 and raw_final['trainer']['completed_steps'] == 1200
        assert clocks0 == [[],[]]
        assert len(clocksN[0]) == 9 and len(clocksN[1]) == 8
        assert all(x == dict(step=1200.,device='cpu') for role in clocksN for x in role)
        assert raw_init['trainer']['recipe'] == raw_final['trainer']['recipe'] == recipe
        assert raw_init['execution'] == raw_final['execution'] == dict(serial_backward=True,scope='full public GANTrainer.step')
        identity = dict(candidate_declaration=seal[alias+'-declaration.json'],source_zip=manifest['source.zip'],initializer_commit='c720645ecae6b648e9fc6034e9d6b48ccff06ed3')
        assert raw_init['identity'] == raw_final['identity'] == identity
        state = raw_init['trainer']; subset = {k:v for k,v in state.items() if k not in ('streams','cpu_rng','cuda_rng')}
        assert subset == init['material']['state']
        cpu = read(ROOT/(alias+'-cpu-preflight.json'))['cpu_initialization']['all_initial_material']
        assert state['models'] == cpu['state']['models']
        assert init['material']['all_parameters_and_buffers'] == cpu['all_parameters_and_buffers']
        assert init['old_sampling_rng_equal'] and not init['old_model_values_loaded']
        for key in ('cpu_rng','cuda_rng'): assert state[key] == fixture['expected_initial'][key]
        for key,value in fixture['expected_initial']['streams'].items(): assert state['streams'][key] == value
        assert raw_init['data_rng'] == fixture['expected_initial']['data_rng']
        assert raw_final['data_rng'] == fixture['expected_final_data_rng']
        for step in (0,1,1200):
            proof = read(p/f'optimizer-device-proof-{step:04d}.json')
            for entries in proof.values():
                for entry in entries:
                    if step == 0: assert entry['state_missing']
                    else:
                        assert entry['step'] == step and entry['step_device'] == 'cpu'
                        assert entry['parameter_device'] == entry['exp_avg_device'] == entry['exp_avg_sq_device'] == 'cuda:0'
        results.append(dict(candidate=decl['candidate'],output=str(p),status='AUDIT_PASS_SCORE_FAIL',result=result,source_zip_sha256=manifest['source.zip'],initial_state_sha256=manifest['initial-state.pt'],final_state_sha256=manifest['final-state.pt'],all_artifacts_verified=len(manifest),initial_models_equal_cpu_preflight=True,raw_checkpoint_native_cpu_counters=17,frozen_sampling_batches_verified=1200,applied_rate_and_noise_rows_verified=1200,observations_verified=24,passing_observations=len(passing),passing_suffix=suffix))
    assert len(results) == 3
    out=dict(status='PASS',scope='Independent stdlib-only receipt/source/raw checkpoint audit; no Torch import or model execution',harness_seal_sha256=sha((ROOT/'harness-sha256.json').read_bytes()),results=results,limits=['No new training or replay executed','Constant KA2 retains horizon4600/noise360-720; scheduled controls use horizon1200/noise120-240; comparison is declared-policy, not LR-only'])
    (ROOT/'public3-runtime-audit.json').write_text(json.dumps(out,indent=2)+'\n')
    rows='\n'.join(f"| {r['candidate']} | 0/24 | {r['result']['metrics']['final']['modes']} | {r['result']['metrics']['final']['hq']:.9f} | FAIL |" for r in results)
    (ROOT/'public3-runtime-audit.md').write_text('''# Fixed initialization: public control runtime audit\n\nAll three completed results are valid failures of this frozen 1,200-update mode-hold screen. No coverage observation passed. The high-quality fraction alone does not establish coverage of all eight modes.\n\n| Configuration | Passing observations | Final modes / 8 | Final HQ | Gate |\n|---|---:|---:|---:|---|\n'''+rows+'''\n\nIndependent standard-library inspection verified every archived artifact hash, the complete package source and reviewed harness seal, all 1,200 frozen data/index/cursor receipt rows, all 24 observations and the final-five rule, and all 1,200 applied LR/noise rows. Runtime imports match declared isolated packages; FP32/deterministic/serial/default-CPU metadata matches the declared execution. The separate dry CUDA sampling preflight completed before any update and retained the training stream.\n\nBoth raw checkpoints for each control were read through a restricted storage-only parser without Torch. Initial complete non-RNG state matches the initial receipt, and all initial model tensors plus every independently listed parameter/buffer match the CPU repeatability preflight. Global, private and caller initial RNG states match the frozen host, while prior/network tensors use the new initializer rather than archived random weights. Final data RNG matches the frozen final cursor. Native Adam state is absent at construction; all 17 final raw counters are CPU scalars at 1,200 with CUDA moments, matching separate step1/1200 device receipts. Source/execution identities agree across checkpoints.\n\nScheduled K3P and KA2 retain declared horizon1,200 and noise milestones120/240. Historical constant-rate KA2 retains horizon4,600 and noise milestones360/720; its nominal G/D/prior rates remain .00425/.00425/.0085 through the screen. This is a comparison of the declared configurations, not an LR-only ablation. These failures do not transfer any old quality evidence or prove failure under every horizon.\n\n[Full audit and source/checkpoint hashes](public3-runtime-audit.json). No new training, Torch import, GPU work, or replay was performed by this audit.\n''')
    print(json.dumps({r['candidate']:{'audit':r['status'],'final_modes':r['result']['metrics']['final']['modes'],'hq':r['result']['metrics']['final']['hq']} for r in results},indent=2))

if __name__ == '__main__': main()
