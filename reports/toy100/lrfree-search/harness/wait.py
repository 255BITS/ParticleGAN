#!/usr/bin/env python
"""Block until all queued/running jobs of the given candidates finish; print a compact matrix.

  wait.py CAND [CAND ...] [--timeout SECONDS] [--poll 10]
Exit code 0 when finished, 2 on timeout (matrix printed either way).
"""
from pathlib import Path
import argparse
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lrlib  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cands', nargs='+')
    p.add_argument('--timeout', type=float, default=None)
    p.add_argument('--poll', type=float, default=10)
    args = p.parse_args()
    start = time.time()
    wanted = set(args.cands)
    code = 0
    while True:
        left = [j for _, j in lrlib.pending_jobs() if j['cand'] in wanted]
        if not left:
            break
        if args.timeout is not None and time.time() - start > args.timeout:
            code = 2
            break
        time.sleep(args.poll)
    print(lrlib.compact_matrix(args.cands))
    if code:
        print(f'TIMEOUT: {len(left)} jobs still pending')
    sys.exit(code)


if __name__ == '__main__':
    main()
