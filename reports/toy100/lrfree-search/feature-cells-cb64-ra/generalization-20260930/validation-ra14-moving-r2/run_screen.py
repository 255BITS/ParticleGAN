"""Run one complete original quality gate on GPU0 under the shared lock."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

from freeze import CONFIG, PACKAGE, ROOT, verify

OLD = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
GPU_UUID = 'GPU-72c1b506-891d-b8bc-b353-e020585e1c47'
TASKS = ('mode_hold', 'img_intensity2', 'img_blobs4', 'img_stripes2', 'img_bars4',
         'vector_two_broad', 'vector_unequal_mass', 'vector_unequal_width', 'vector_anisotropic',
         'vector_overlap', 'vector_spiral', 'ring_shift', 'stationary',
         'grid100', 'rotated100', 'staggered100')
OPTIONS = dict(eval_output_noise=True,save_final_state=True,strict_streams=True,diagnostics=True)
ENV = dict(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8',
           OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
           PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')


def write(path, value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True) + '\n')


def parked_owner():
    for pid,ticks,stopped in ((384331,'163720702',True),(383348,'163716372',False)):
        text = Path(f'/proc/{pid}/stat').read_text()
        fields = text[text.rfind(')')+2:].split()
        assert fields[19] == ticks, 'original slot process identity changed'
        if stopped:
            assert fields[0] == 'T', 'original numerical supervisor is no longer parked'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,choices=TASKS)
    parser.add_argument('--check-only',action='store_true')
    args = parser.parse_args()
    before = verify()
    if args.check_only:
        print(json.dumps(dict(task=args.task,integrity=before)),flush=True)
        return
    for name in ('ABSENT','ABSENT_START','ABSENT_END','LRFREE_NATIVE_TEST_STEPS'):
        os.environ.pop(name,None)
    os.environ.update(ENV)
    output = ROOT / 'runs' / args.task
    assert not output.exists(), 'retain completed and interrupted attempts'
    output.mkdir(parents=True)
    start = time.time()
    receipt = dict(status='WAITING_FOR_GPU',task=args.task,command=[sys.executable,*sys.argv],
        source_integrity_before=before,start=start,options=OPTIONS,
        source_adapters=json.loads((ROOT / 'SCREEN-ADAPTER.json').read_text()),
        resources=dict(physical_gpu=0,device='cuda:0',expected_gpu_uuid=GPU_UUID,memory_fraction=.2))
    write(output / 'execution-receipt.json',receipt)
    print(json.dumps(dict(event='waiting_for_gpu',task=args.task)),flush=True)
    code = 0
    try:
        with (OLD / 'quality/.serial-phase.lock').open('r') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            parked_owner()
            verify()
            import torch
            assert torch.cuda.is_available() and torch.cuda.device_count() == 1
            torch.cuda.set_device(0)
            torch.cuda.set_per_process_memory_fraction(.2,0)
            props = torch.cuda.get_device_properties(0)
            actual_uuid = 'GPU-' + str(props.uuid).removeprefix('GPU-').lower()
            assert actual_uuid == GPU_UUID
            receipt['resources'].update(gpu_uuid=actual_uuid,gpu_name=props.name,total_gpu_bytes=props.total_memory)
            receipt.update(status='RUNNING',numerical_started=time.time(),torch=str(torch.__version__),cuda=torch.version.cuda)
            write(output / 'execution-receipt.json',receipt)
            sys.argv = [str(ROOT / 'screen_current.py'),'--package-root',str(PACKAGE),
                '--overrides',str(CONFIG),'--task',args.task,'--output',str(output),'--device','cuda:0',
                '--candidate-options',json.dumps(OPTIONS),'--cand','RA14-replay']
            runpy.run_path(str(ROOT / 'screen_current.py'),run_name='__main__')
            receipt['peak_allocated_gpu_mib'] = torch.cuda.max_memory_allocated(0) / 2**20
            receipt['peak_reserved_gpu_mib'] = torch.cuda.max_memory_reserved(0) / 2**20
            result = json.loads((output / 'result.json').read_text())
            receipt['primary_status'] = result['status']
            if result['status'] == 'ERROR':
                code = 1
    except SystemExit as error:
        code = int(error.code or 0) if isinstance(error.code,(int,type(None))) else 1
        receipt['error'] = repr(error)
        result_path = output / 'result.json'
        if result_path.exists():
            receipt['primary_status'] = json.loads(result_path.read_text()).get('status')
    except Exception as error:
        code = 1
        receipt['error'] = repr(error)
        receipt['traceback'] = traceback.format_exc()
        print(receipt['traceback'],file=sys.stderr,flush=True)
    try:
        receipt['source_integrity_after'] = verify()
    except Exception as error:
        code = 1
        receipt['source_integrity_after'] = dict(status='INVALID',error=repr(error))
    receipt.update(status='COMPLETE' if code == 0 else 'ERROR',process_exit_code=code,
                   completed=time.time(),wall_seconds=time.time()-start)
    write(output / 'execution-receipt.json',receipt)
    print(json.dumps(dict(event='screen_end',task=args.task,status=receipt['status'],
                         primary_status=receipt.get('primary_status'),wall_seconds=receipt['wall_seconds'])),flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
