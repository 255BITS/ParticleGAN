"""Prepare or explicitly launch the sealed original1000→1500 CUDA pair.

--prepare performs only stdlib source work. --run is root-authorized execution;
the shared GPU lock stays held across two separate package-isolated workers.
"""
import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
LANE = STUDY / 'validation-ra14-r2'
ORIGINAL = LANE / 'moving/rotated100'
CONFIG = STUDY / 'configs/RA14-replay.json'
CONTROL = STUDY / 'pkg-RA14-replay'
CANDIDATE = STUDY / 'pkg-RA15-partial-recovery'
LOCK = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/quality/.serial-phase.lock')
GPU_UUID = 'GPU-72c1b506-891d-b8bc-b353-e020585e1c47'
ENV = dict(CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID', CUBLAS_WORKSPACE_CONFIG=':4096:8',
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
    PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def digest(package):
    h = hashlib.sha256()
    for path in sorted((package / 'particlegan').rglob('*.py')):
        h.update(str(path.relative_to(package / 'particlegan')).encode() + b'\0' + path.read_bytes() + b'\0')
    return h.hexdigest()


def parked_owner():
    for pid, ticks, stopped in ((384331, '163720702', True), (383348, '163716372', False)):
        stat = Path(f'/proc/{pid}/stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        assert fields[19] == ticks, 'original slot process identity changed'
        if stopped:
            assert fields[0] == 'T', 'original numerical supervisor is no longer parked'


def adapted(package, variant):
    source = (ORIGINAL / 'adapted_runner.py').read_text()
    replacements = [
        (f'REPO = {str(CONTROL)!r}', f'REPO = {str(package)!r}'),
        ('snap(0)\nfor step in range(1, args.steps + 1):',
         "assert args.task == 'rotated100' and args.steps == 1500 and args.every == 500\n"
         "assert args.points == 4096 and args.gate and args.rotate_every == 500 and args.rotate_deg == 30\n"
         "assert round(math.degrees(angle(1001))) == 60 and round(math.degrees(angle(1500))) == 60\n"
         "support.resume(trainer, real_batch, stream, gate_rows, owned_output)\n"
         "for step in range(1001, args.steps + 1):"),
        ('    if step % args.every == 0:\n        snap(step)',
         '    support.note_update(trainer, step, stream)\n    if step % args.every == 0:\n        snap(step)')]
    for old, new in replacements:
        assert source.count(old) == 1, old
        source = source.replace(old, new, 1)
    source += f'\nsupport.finish(trainer, stream, {variant!r})\n'
    ast.parse(source)
    compile(source, f'<{variant}-resume>', 'exec')
    return source


def prepare(closure):
    assert not (HERE / 'SOURCE-FREEZE.json').exists(), 'preserve each preparation'
    frozen = json.loads((LANE / 'SOURCE-FREEZE.json').read_text())
    assert sha(LANE / 'SOURCE-FREEZE.json') == '871322377b4f7330087760298dac5b1869f5bec01ae01d93ee8b3c6956760a5f'
    assert digest(CONTROL) == frozen['package_sha256'] == '68f5706590683a44348cb04b3798917411bfaa54e5d45b17d57c345a4da33c15'
    assert sha(CONFIG) == 'a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4'
    assert sha(ORIGINAL / 'checkpoint-001000.pt') == 'e850172959bf7607ed69c890632563768e352936a7a916bb4063fc74f45a3966'
    assert sha(ORIGINAL / 'adapted_runner.py') == '4d94459b25892cd37427ee940e3ad863f3676b8e1b5d6cd5748525bfaf3a33db'
    for path, expected in frozen['hashes'].items():
        assert sha(path) == expected, path
    assert closure.is_file()
    adapter_check = json.loads((HERE / 'SOURCE-ADAPTER-CHECK-V2.json').read_text())
    assert adapter_check['status'] == 'CPU_PASS' and not adapter_check['cuda_initialized']
    for path, expected in adapter_check['input_sha256'].items():
        assert sha(path) == expected, path
    bridge = json.loads(closure.read_text())
    assert bridge['status'] == 'CPU_PASS'
    assert bridge['package'] == str(CANDIDATE)
    assert bridge['package_sha256'] == digest(CANDIDATE) == '741fc3933654b985a74314b730b93be814d408ad46f6f7f6547b561f64619228'
    assert bridge['base_package_sha256'] == digest(CONTROL)
    assert bridge['config_byte_identical'] and bridge['config_sha256'] == sha(CONFIG)
    assert bridge['static_no_fire_fixture_exact_checkpoint_and_RNG_parity']
    assert not bridge['CUDA_initialized']
    candidate_freeze = closure.parent / 'SOURCE-FREEZE.json'
    assert sha(candidate_freeze) == 'ec1e68eb823e813577d56fa4d86753db75d8ac86f941d97fc9fa860495ec3777'
    candidate_frozen = json.loads(candidate_freeze.read_text())
    assert candidate_frozen['package_sha256'] == digest(CANDIDATE)
    for path, expected in candidate_frozen['hashes'].items():
        assert sha(path) == expected, path
    for variant, package in [('control', CONTROL), ('candidate', CANDIDATE)]:
        with (HERE / f'{variant}_runner.py').open('x') as handle:
            handle.write(adapted(package, variant))
    files = set(map(Path, frozen['hashes']))
    files.update(map(Path, candidate_frozen['hashes']))
    files.update((LANE / 'SOURCE-FREEZE.json', CONFIG, closure, candidate_freeze))
    files.update(CANDIDATE.rglob('*.py'))
    files.update(ORIGINAL / name for name in ('checkpoint-001000.pt', 'checkpoint-001500.pt',
        'COMPLETION.json', 'LAUNCH.json', 'adapted_runner.py', 'frames.npz.verdict.json'))
    files.update(HERE.glob('*.py'))
    files.update(HERE.glob('*.md'))
    files.update(HERE.glob('*.json'))
    receipt = dict(status='FROZEN_NOT_LAUNCHED', task='rotated100',
        hashes={str(path): sha(path) for path in sorted(files)},
        packages={variant: dict(root=str(package), sha256=digest(package))
                  for variant, package in [('control', CONTROL), ('candidate', CANDIDATE)]},
        original_source_freeze_sha256=sha(LANE / 'SOURCE-FREEZE.json'),
        config_sha256=sha(CONFIG), candidate_closure=str(closure), candidate_closure_sha256=sha(closure),
        original_checkpoint_sha256=sha(ORIGINAL / 'checkpoint-001000.pt'),
        protocol=dict(first_update=1001, last_update=1500, updates_per_variant=500,
            paired_variants=2, external_seed=1234, batch_size=2048,
            external_batches_before_restore=2000, real_batches_per_update=2,
            target_degrees=60, turn_every=500, rotation_degrees=30,
            snapshot_every=500, frame_points=4096, gate_points=20000,
            gate_latent_seed=1637, gate_noise_seed=1636, original_quality_bar=.86481,
            original_min_modes=95, serial_backward=True, full_raw_axes_max=8,
            shared_action_budget_fraction=.05, serving_coherence_fraction=.95),
        source_adaptations=['package root only', 'restore original1000checkpoint and external cursor',
            'inherit original500/1000gate rows', 'resume original loop at1001',
            'read existing chart locals and scalar state', 'compare exact control semantic state'],
        seeds_changed=False, scorer_changed=False, thresholds_changed=False,
        training_updates=0, gpu_operations=0, tensor_loads=0, model_calls=0)
    write(HERE / 'SOURCE-FREEZE.json', receipt)
    print(json.dumps(verify(), sort_keys=True), flush=True)


def verify():
    path = HERE / 'SOURCE-FREEZE.json'
    frozen = json.loads(path.read_text())
    for name, expected in frozen['hashes'].items():
        assert sha(name) == expected, f'frozen input changed: {name}'
    for variant, package in frozen['packages'].items():
        assert digest(Path(package['root'])) == package['sha256'], variant
    return dict(status='VALID', source_freeze_sha256=sha(path), files=len(frozen['hashes']),
        package_sha256={key: value['sha256'] for key, value in frozen['packages'].items()})


def worker(variant, output, lock_fd):
    os.environ.update(ENV)
    assert lock_fd is not None and lock_fd > 2, 'worker requires the inherited shared GPU lock'
    actual_lock = os.fstat(lock_fd)
    expected_lock = LOCK.stat()
    assert (actual_lock.st_dev, actual_lock.st_ino) == (expected_lock.st_dev, expected_lock.st_ino)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    before = verify()
    parked_owner()
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.2, 0)
    properties = torch.cuda.get_device_properties(0)
    actual_uuid = 'GPU-' + str(properties.uuid).removeprefix('GPU-').lower()
    assert actual_uuid == GPU_UUID
    import window_support as support
    source = HERE / f'{variant}_runner.py'
    write(output / 'LAUNCH.json', dict(status='RUNNING', variant=variant,
        numerical_started=time.time(), source_integrity_before=before,
        resources=dict(physical_gpu=0, gpu_uuid=actual_uuid, gpu_name=properties.name,
            memory_fraction=.2, total_gpu_bytes=properties.total_memory),
        script_sha256=sha(source), command=[sys.executable, *sys.argv]))
    sys.argv = [str(source), 'rotated100', str(output / 'frames.npz'), '--every', '500',
        '--points', '4096', '--steps', '1500', '--gate', '--rotate-every', '500', '--rotate-deg', '30']
    exec(compile(source.read_text(), str(source), 'exec'),
        {'__name__': '__main__', '__file__': str(source), 'owned_output': output, 'support': support})
    write(output / 'SOURCE-CLOSE.json', dict(status='VALID', source_integrity_after=verify()))


def run(output):
    for name in ('ABSENT', 'ABSENT_START', 'ABSENT_END', 'LRFREE_NATIVE_TEST_STEPS'):
        os.environ.pop(name, None)
    os.environ.update(ENV)
    before = verify()
    assert not output.exists(), 'retain every numerical attempt'
    output.mkdir(parents=True)
    write(output / 'LAUNCH.json', dict(status='WAITING_FOR_GPU', start=time.time(),
        source_integrity_before=before, command=[sys.executable, *sys.argv]))
    print(json.dumps(dict(event='waiting_for_gpu', output=str(output))), flush=True)
    with LOCK.open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        parked_owner()
        verify()
        completions = {}
        for variant in ('control', 'candidate'):
            target = output / variant
            target.mkdir()
            command = [sys.executable, '-u', '-B', str(HERE / 'run_pair.py'), '--worker', variant,
                       '--output', str(target), '--lock-fd', str(lock.fileno())]
            print(json.dumps(dict(event='worker_start', variant=variant, output=str(target))), flush=True)
            with (target / 'run.log').open('x') as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                    env=os.environ.copy(), close_fds=True, pass_fds=(lock.fileno(),), check=False)
            if process.returncode:
                write(output / 'ERROR.json', dict(status='ERROR', variant=variant,
                    exit_code=process.returncode, source_integrity_after=verify(),
                    reason='worker failed; candidate not interpreted without successful control'))
                raise SystemExit(process.returncode)
            completions[variant] = json.loads((target / 'WINDOW-COMPLETION.json').read_text())
            assert json.loads((target / 'SOURCE-CLOSE.json').read_text())['status'] == 'VALID'
            if variant == 'control':
                assert completions[variant]['control_reproduction']['status'] == 'PASS'
            print(json.dumps(dict(event='worker_complete', variant=variant,
                quality_status=completions[variant]['quality_status'])), flush=True)
        assert completions['control']['external_cursor_sha256'] == completions['candidate']['external_cursor_sha256']
        write(output / 'COMPLETION.json', dict(status='COMPLETE', evidence_validity='VALID',
            control_reproduction='PASS', quality_status=completions['candidate']['quality_status'],
            variants=completions, source_integrity_after=verify(), completed=time.time(),
            original_quality_criteria_unchanged=True, full_new_quality_validation=False,
            scope='Paired diagnostic continuation from the original1000checkpoint only'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--check-only', action='store_true')
    actions.add_argument('--run', action='store_true')
    actions.add_argument('--worker', choices=['control', 'candidate'])
    parser.add_argument('--candidate-closure', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.prepare:
        assert args.candidate_closure is not None
        prepare(args.candidate_closure.resolve())
    elif args.check_only:
        print(json.dumps(verify(), sort_keys=True), flush=True)
    elif args.worker:
        assert args.output is not None and args.output.is_dir()
        worker(args.worker, args.output.resolve(), args.lock_fd)
    else:
        assert args.output is not None
        run(args.output.resolve())


if __name__ == '__main__':
    main()
