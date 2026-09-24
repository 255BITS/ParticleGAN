"""Replay an archived GPU candidate without the original worktrees."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, required=True, help='new replay directory')
    parser.add_argument('--gpu', required=True, help='physical GPU index or UUID')
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--task', required=True, help='toy name or convergence')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    rows = json.loads((root / 'candidates.json').read_text())
    row = next((r for r in rows if r['name'] == args.candidate), None)
    if row is None:
        parser.error('unknown candidate')
    tasks = json.loads((root / 'protocol.json').read_text())['toys'] + ['convergence']
    if args.task not in tasks:
        parser.error('unknown task')
    supported = (row['supported'] == 'all'
                 or row['supported'] == 'two' and args.task in ('mode_hold', 'trajectory', 'convergence')
                 or row['supported'] == 'three' and args.task in ('two_pole', 'mode_hold', 'unipolar', 'convergence'))
    if not supported:
        parser.error('published candidate has no adapter for this task')
    archive_info = json.loads((root / 'source-archives.json').read_text())[args.candidate]
    archive = root / 'archives' / archive_info['archive']
    if hashlib.sha256(archive.read_bytes()).hexdigest() != archive_info['sha256']:
        raise ValueError('archive checksum mismatch')
    if os.environ.get('LD_PRELOAD'):
        raise RuntimeError('start replay with LD_PRELOAD unset')
    # Import CUDA only after choosing visibility; no CPU fallback or vendor shim.
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, CUBLAS_WORKSPACE_CONFIG=':4096:8',
                      OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      PYTHONHASHSEED='0')
    os.environ.pop('LD_PRELOAD', None)
    os.environ.pop('PYTHONPATH', None)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    if (str(torch.__version__), torch.version.cuda, torch.backends.cudnn.version()) != ('2.13.0+cu126', '12.6', 91002):
        raise RuntimeError('cuda_fp32_v1 requires torch 2.13.0+cu126 / CUDA 12.6 / cuDNN 91002')
    if torch.cuda.get_device_name(0) != 'NVIDIA RTX A6000':
        raise RuntimeError('this leaderboard profile requires an RTX A6000; declare a separate profile for other GPUs')
    target = args.workdir.resolve()
    target.mkdir(parents=True, exist_ok=False)
    repo = target / 'repo'
    repo.mkdir()
    hashes = archive_info['source_hashes']
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        if set(m.name for m in members) != set(hashes):
            raise ValueError('archive file list differs from manifest')
        for member in members:
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts:
                raise ValueError('unsafe archive entry')
            data = stream.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != hashes[member.name]:
                raise ValueError('source checksum mismatch: ' + member.name)
            dest = repo / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
    row['repo'] = str(repo)
    (target / 'candidates.json').write_text(json.dumps([row], indent=2) + '\n')
    for filename in ('worker.py', f'sources-{args.candidate}.json'):
        shutil.copy2(root / filename, target / filename)
    command = [sys.executable, '-u', str(target / 'worker.py'), '--candidate', args.candidate,
               '--task', args.task, '--output', str(target / 'result')]
    (target / 'replay-command.json').write_text(json.dumps(command, indent=2) + '\n')
    raise SystemExit(subprocess.call(command, cwd=target))


if __name__ == '__main__':
    main()
