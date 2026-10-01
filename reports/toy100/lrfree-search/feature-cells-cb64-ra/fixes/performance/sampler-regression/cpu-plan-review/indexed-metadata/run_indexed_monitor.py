"""CPU-only original RA4 canonical monitor with a declared API metadata hook."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
import argparse
import json
from pathlib import Path
import runpy
import indexed_adapter as adapter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    if args.watch:
        pilot=adapter.read(adapter.HERE/'PILOT.json')
        frozen=adapter.read(adapter.HERE/'PILOT-FROZEN.json')
        assert pilot['status']=='VALID' and adapter.sha(adapter.HERE/'PILOT.json')==frozen['pilot_sha256']
        assert adapter.sha(adapter.READY)==frozen['ready_sha256']
    api=adapter.install()
    print(json.dumps(dict(event='declared_indexed_api_monitor_start',pid=os.getpid(),
        output=str(adapter.OUTPUT),ready_sha256=adapter.sha(adapter.READY),watch=args.watch,api=api)),flush=True)
    sys.argv=[str(adapter.MONITOR),'--validation',str(adapter.VALIDATION),'--output',str(adapter.OUTPUT)]
    if args.watch:sys.argv.append('--watch')
    runpy.run_path(str(adapter.MONITOR),run_name='__main__')


if __name__=='__main__':main()
