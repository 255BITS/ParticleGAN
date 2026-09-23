"""Run probe.py and kernels.py on every declared numerical path; write paths.jsonl.

  python3 -m reports.transfer_suite.host_replication.portability.paths OUTDIR
Emulated paths need qemu-user (x86-64 TCG supports AVX2, not AVX512).
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
NATIVE = [('native default', {}), ('native ATen AVX2', dict(ATEN_CPU_CAPABILITY='avx2')),
          ('native ATen scalar', dict(ATEN_CPU_CAPABILITY='default')),
          ('native MKL instructions AVX2', dict(MKL_ENABLE_INSTRUCTIONS='AVX2')),
          ('native MKL CNR AVX2', dict(MKL_CBWR='AVX2')), ('native MKL CNR AVX', dict(MKL_CBWR='AVX')),
          ('native MKL CNR COMPATIBLE', dict(MKL_CBWR='COMPATIBLE')),
          ('native all AVX2 pins', dict(ATEN_CPU_CAPABILITY='avx2', MKL_CBWR='AVX2', MKL_ENABLE_INSTRUCTIONS='AVX2', ONEDNN_MAX_CPU_ISA='AVX2'))]
EMULATED = [(cpu, cnr) for cpu in ('Haswell-v4', 'Skylake-Client-v4', 'EPYC-Rome-v4', 'EPYC-Milan-v2') for cnr in (None,)] + \
           [(cpu, cnr) for cpu in ('Haswell-v4', 'EPYC-Rome-v4') for cnr in ('AVX2', 'COMPATIBLE')]


def run(label, env, cpu=None):
    rows = []
    for module in ('probe', 'kernels'):
        command = ([] if cpu is None else ['qemu-x86_64', '-cpu', cpu]) + \
                  ['/usr/bin/python3', '-m', f'reports.transfer_suite.host_replication.portability.{module}', label]
        out = subprocess.run(command, cwd=ROOT, env=os.environ | env, capture_output=True, text=True, check=True).stdout
        rows.append(json.loads(out.strip().splitlines()[-1]))
    probe, kernels = rows
    return dict(label=label, emulated_cpu=cpu, environment=env, **{k: probe[k] for k in probe if k not in ('label', 'environment')},
                sqrt_not_correctly_rounded=kernels['sqrt_not_correctly_rounded'], sqrt_digest=kernels['sqrt_digest'],
                adam_step_digest=kernels['adam_step_digest'])


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(label, env, None) for label, env in NATIVE]
    jobs += [(f"emulated {cpu}" + (f" MKL CNR {cnr}" if cnr else ''), dict(MKL_CBWR=cnr) if cnr else {}, cpu) for cpu, cnr in EMULATED]
    with ThreadPoolExecutor(int(os.environ.get('JOBS', '3'))) as pool, open(output / 'paths.jsonl', 'w') as log:
        for row in pool.map(lambda job: run(*job), jobs):
            log.write(json.dumps(row) + '\n')
            log.flush()
            print(f"{row['label']:40} fp={row['fingerprint']} archive={row['equals_archive']} "
                  f"sqrt_off={row['sqrt_not_correctly_rounded']} adam={row['adam_step_digest']}", flush=True)


if __name__ == '__main__':
    main()
