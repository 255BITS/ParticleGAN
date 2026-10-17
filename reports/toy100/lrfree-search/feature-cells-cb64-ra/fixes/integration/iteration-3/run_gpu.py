"""Run focused CUDA contracts, then the frozen three-job learned phase."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON = '/tmp/pr38-default-env/bin/python'
PACKAGE = ROOT / 'pkg-CB64-RA3'
VALIDATION = ROOT / 'validation-ra3'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    deadline = time.monotonic() + 900
    while not (VALIDATION / 'source-freeze.json').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('Candidate was not frozen within the reserved review period.')
        print(json.dumps(dict(event='waiting_for_frozen_candidate',validation=str(VALIDATION))),flush=True)
        time.sleep(15)
    spec = importlib.util.spec_from_file_location('ra3_frozen_launcher',VALIDATION/'launch.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    freeze = json.loads((VALIDATION/'source-freeze.json').read_text())
    ready = json.loads((HERE/'READY.json').read_text())
    freeze_sha = sha(VALIDATION/'source-freeze.json')
    ready_sha = sha(HERE/'READY.json')

    def guard():
        assert sha(VALIDATION/'source-freeze.json') == freeze_sha
        assert sha(HERE/'READY.json') == ready_sha
        launcher.check_sources(freeze)
        for path, expected in ready['numerical_source_sha256'].items():
            assert sha(path) == expected,path

    guard()
    jobs = [
        ('gpu-lineage',[PYTHON,'-u','-B',str(ROOT/'geometry/training-regression/gpu_lineage_check.py'),
                        '--package-root',str(PACKAGE),'--output',str(HERE/'gpu-lineage')]),
        ('gpu-count',[PYTHON,'-u','-B',str(ROOT/'integration/review/training-regression/recovery_gpu_check.py'),
                      '--package-root',str(PACKAGE),'--output',str(HERE/'gpu-count')]),
        ('gpu-sampler-profile',[PYTHON,'-u','-B',str(ROOT/'performance/sampler-regression/profile_gpu.py'),
                     '--package-root',str(PACKAGE),'--output',str(HERE/'gpu-sampler-profile')]),
        ('learned-phase',[PYTHON,'-u','-B',str(VALIDATION/'launch.py'),'--through','3'])]
    completed=[]
    for name, command in jobs:
        guard()
        print(json.dumps(dict(event='focused_gpu_job_start',name=name,command=command)),flush=True)
        start=time.monotonic()
        with (HERE/(name+'.log')).open('x') as log:
            process=subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),
                                   stdout=log,stderr=subprocess.STDOUT)
        guard()
        completed.append(dict(name=name,returncode=process.returncode,wall_seconds=time.monotonic()-start))
        (HERE/'gpu-execution.json').write_text(json.dumps(completed,indent=2)+'\n')
        print(json.dumps(dict(event='focused_gpu_job_complete',**completed[-1])),flush=True)
        if process.returncode:
            raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
