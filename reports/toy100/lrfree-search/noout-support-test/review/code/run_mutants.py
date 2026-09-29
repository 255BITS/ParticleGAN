"""Build mutants of pkg-E17 (mutants.py) in review_E17_code/mut/pkg-<id>/particlegan and run a test script against each.
usage: python run_mutants.py --test <script> --tag <name> [--ids M01,M02] [--workers 4] [--keep]
The test script is called as `python <script> <mutant pkg root>` (CPU only, 2 threads each); results -> results_<tag>.json, logs -> mut/logs/."""
import argparse, difflib, json, os, re, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mutants import MUTANTS
BASE = '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E17/particlegan'
PY = '/tmp/pr38-default-env/bin/python'

def build(mut):
    i, cls, desc, edits = mut
    root = f'{HERE}/mut/pkg-{i}'
    if os.path.exists(root): shutil.rmtree(root)
    shutil.copytree(BASE, f'{root}/particlegan', ignore=shutil.ignore_patterns('__pycache__'))
    diff = []
    for f, old, new in edits:
        path = f'{root}/particlegan/{f}'
        src = open(path).read()
        n = src.count(old)
        assert n == 1, f'{i}: `{old[:70]}` occurs {n} times in {f}'
        out = src.replace(old, new)
        open(path, 'w').write(out)
        diff += list(difflib.unified_diff(src.splitlines(), out.splitlines(), f'E17/{f}', f'{i}/{f}', lineterm='', n=0))
    os.makedirs(f'{HERE}/mut/diffs', exist_ok=True)
    open(f'{HERE}/mut/diffs/{i}.diff', 'w').write('\n'.join(diff) + '\n')
    return root

def run(mut, test, tag, timeout):
    i, cls, desc, edits = mut
    root = build(mut)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    t0 = time.time()
    try:
        p = subprocess.run([PY, test, root], capture_output=True, text=True, timeout=timeout, env=env, cwd=os.path.dirname(test))
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out, err = -9, (e.stdout or b'').decode() if isinstance(e.stdout, bytes) else (e.stdout or ''), 'TIMEOUT'
    os.makedirs(f'{HERE}/mut/logs', exist_ok=True)
    open(f'{HERE}/mut/logs/{i}.{tag}.out', 'w').write(out + '\n--- stderr ---\n' + err[-3000:])
    fails = [l[7:110] for l in out.splitlines() if l.startswith('[FAIL]')]
    ok = 'ALLOK' in out
    crash = (not ok) and not any(l.startswith('FAILED') for l in out.splitlines())
    status = 'SURVIVES' if ok and rc == 0 else ('CAUGHT' if (fails or not crash) else 'CRASH')      # a crash after FAIL lines is a detection; a crash without any FAIL line is not
    return dict(id=i, cls=cls, desc=desc, status=status, fails=fails, rc=rc, secs=round(time.time() - t0, 1),
                err=(err.strip().splitlines()[-1][:160] if crash and err.strip() else ''))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--ids', default=''); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--timeout', type=int, default=900); ap.add_argument('--keep', action='store_true')
    a = ap.parse_args()
    sel = [m for m in MUTANTS if not a.ids or m[0] in a.ids.split(',')]
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda m: run(m, a.test, a.tag, a.timeout), sel))
    json.dump(res, open(f'{HERE}/results_{a.tag}.json', 'w'), indent=1)
    for r in res:
        print(f"{r['id']} {r['cls']:5s} {r['status']:8s} {r['secs']:6.1f}s  {r['desc'][:78]}")
        for f in r['fails'][:3]: print('        FAIL:', f)
        if r['err']: print('        ERR :', r['err'])
    print('caught %d, crash %d, survive %d of %d' % tuple([sum(r['status'] == s for r in res) for s in ('CAUGHT', 'CRASH', 'SURVIVES')] + [len(res)]))
    if not a.keep:
        for r in res: shutil.rmtree(f"{HERE}/mut/pkg-{r['id']}", ignore_errors=True)
