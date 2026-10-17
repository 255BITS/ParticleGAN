"""Validate merged CUDA contracts, then all original frozen acceptance jobs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PACKAGE=ROOT/'pkg-CB64-RA4'
VALIDATION=ROOT/'validation-ra4'
PYTHON='/tmp/pr38-default-env/bin/python'
COUNT=ROOT/'integration/review/training-regression/global-count'
PERFORMANCE=ROOT/'performance/sampler-regression/cpu-plan-review/plan-batching'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ready_path=HERE/'READY.json'
    ready=json.loads(ready_path.read_text())
    ready_sha=sha(ready_path)
    freeze_path=VALIDATION/'source-freeze.json'
    freeze=json.loads(freeze_path.read_text())
    freeze_sha=sha(freeze_path)
    spec=importlib.util.spec_from_file_location('ra4_frozen_launcher',VALIDATION/'launch.py')
    launcher=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    def guard():
        assert sha(ready_path)==ready_sha
        assert sha(freeze_path)==freeze_sha
        launcher.check_sources(freeze)
        for path,expected in ready['numerical_source_sha256'].items():
            assert sha(Path(path))==expected,path

    jobs=[
        ('gpu-count',[PYTHON,'-u','-B',str(COUNT/'contract_check.py'),
            '--package-root',str(PACKAGE),'--output',str(HERE/'gpu-count')]),
        ('gpu-plan',[PYTHON,'-u','-B',str(PERFORMANCE/'profile_plan_pair_gpu.py'),
            '--reference-package-root',str(COUNT/'pkg-global-count'),
            '--package-root',str(PACKAGE),'--input',str(COUNT/'inputs.pt'),
            '--contract-root',str(COUNT),'--output',str(HERE/'gpu-plan.json')]),
        ('acceptance',[PYTHON,'-u','-B',str(VALIDATION/'launch.py'),'--through','19'])]
    completed=[]
    for name,command in jobs:
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


if __name__=='__main__':
    main()
