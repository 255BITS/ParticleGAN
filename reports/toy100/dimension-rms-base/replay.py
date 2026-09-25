"""Verify and replay the selected dimension-RMS GAN base in a fresh work directory."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

bundle = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--task', required=True, choices=['two_pole', 'trajectory', 'mode_hold',
    'vector_unequal_mass', 'img_intensity2', 'img_bars4', 'img_blobs4'])
p.add_argument('--workdir', type=Path, required=True)
p.add_argument('--gpu', default='1')
a = p.parse_args()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
declaration = json.loads((bundle / 'original-declaration.json').read_text())
for name, expected in declaration['file_hashes'].items():
    assert sha(bundle / name) == expected, name
sys.path.insert(0, str(bundle.parent / 'cpu-recipe-gpu-port'))
from prepare import prepare
work = prepare(a.workdir.resolve())
assert json.loads((work / 'prepared-sources.json').read_text()) == json.loads(
    (bundle / 'prepared-sources.json').read_text())
fixture = bundle / 'initialization-fixtures' / a.task / 'initial-values.pt'
if not fixture.exists():
    fixture = bundle.parent / 'cpu-recipe-gpu-port/initialization-fixtures' / a.task / 'initial-values.pt'
reference = json.loads(gzip.decompress((bundle / 'results' / (a.task + '.json.gz')).read_bytes()))
assert sha(fixture) == reference['initialization_fixture_sha256']
env = os.environ.copy()
for key in ('LD_PRELOAD', 'PYTHONPATH', 'CODEX_THREAD_ID'):
    env.pop(key, None)
env.update(CUDA_VISIBLE_DEVICES=a.gpu, CUBLAS_WORKSPACE_CONFIG=':4096:8',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           PYTHONHASHSEED='0', ATEN_CPU_CAPABILITY='avx2', MKL_ENABLE_INSTRUCTIONS='AVX2',
           ONEDNN_MAX_CPU_ISA='AVX2', DNNL_MAX_CPU_ISA='AVX2')
command = [sys.executable, str(bundle / 'probe.py'), '--repo', str(work / 'repos/cuda'),
           '--config', str(bundle / 'config.json'), '--task', a.task, '--backend', 'cuda',
           '--initial-state', str(fixture), '--output', str(work / 'result')]
(work / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
subprocess.run(command, env=env, check=True)
actual = json.loads((work / 'result/result.json').read_text())


def without_timing(value):
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items() if 'seconds' not in k}
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


assert actual['status'] == reference['status']
assert actual['spec'] == reference['spec'] and actual['config'] == reference['config']
assert without_timing(actual['result']) == without_timing(reference['result'])
assert without_timing(actual['verdict']) == without_timing(reference['verdict'])
assert actual['randomness'] == reference['randomness']
assert actual['proof']['initial_optimizers'] == reference['proof']['initial_optimizers']
assert actual['proof']['adam_calls'] == reference['proof']['adam_calls']
assert all(r['device'] == 'cuda:0' and r['state_devices'] == ['cuda:0']
           for r in actual['proof']['optimizers'].values())
receipt = dict(status='PASS', replay_verdict=actual['status'], task=a.task,
               all_non_timing_metrics_equal=True, initial_parameters_equal=True,
               random_draws_equal=True, cuda_updates_verified=True,
               probe_sha256=sha(bundle / 'probe.py'), mechanism_sha256=sha(bundle / 'mechanism.py'))
(work / 'replay-check.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
