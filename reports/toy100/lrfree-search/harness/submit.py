#!/usr/bin/env python
"""Queue screening jobs for one candidate.

  submit.py --cand NAME --package-root DIR [--overrides JSON|FILE] [--tasks gates]
            [--candidate-options JSON|FILE] [--priority 0] [--note TEXT]

Tasks: comma list of task names and/or presets quick|images|vectors|gates|ring|all.
Refuses (per task) an exact duplicate config (package content + overrides + behaviour options)
that already ran or is queued, and refuses reusing a candidate name for a different config.
"""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lrlib  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cand', required=True)
    p.add_argument('--package-root', required=True, type=Path)
    p.add_argument('--overrides', default='{}')
    p.add_argument('--tasks', default='gates')
    p.add_argument('--candidate-options', default='{}')
    p.add_argument('--priority', type=int, default=0, help='higher runs first')
    p.add_argument('--note', default='')
    p.add_argument('--rerun-errors', action='store_true', help='allow re-queueing tasks whose last result was ERROR')
    args = p.parse_args()
    if not all(c.isalnum() or c in '-_.+' for c in args.cand):
        sys.exit('candidate name: use letters, digits, - _ . +')
    root = args.package_root.resolve()
    package_sha = lrlib.package_digest(root)
    overrides = lrlib.load_json_arg(args.overrides)
    options = lrlib.load_json_arg(args.candidate_options)
    chash = lrlib.config_hash(package_sha, overrides, options)
    loaded, _, behavior = lrlib.normalize(overrides, options)
    registry_path = lrlib.RUNS / args.cand / 'candidate.json'
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
        if registry.get('config_hash') != chash:
            legacy = 'eval_output_noise' not in (registry.get('behavior_options') or {})
            sys.exit(f'candidate {args.cand!r} already registered with a different config '
                     f'({registry.get("config_hash")} != {chash}); choose a new --cand name'
                     + (' (it was registered before the 2026-09-27 clean-eval fix: its rows are noisy-eval; '
                        'e.g. submit as NAME-ce)' if legacy else ''))
    else:
        registry = dict(cand=args.cand, package_root=str(root), package_sha256=package_sha, overrides=overrides,
                        resolved_overrides=loaded, candidate_options=options, behavior_options=behavior,
                        config_hash=chash, note=args.note, submitted=time.strftime('%Y-%m-%dT%H:%M:%S'))
    done = {}
    for row in lrlib.read_ledger():
        if row.get('config_hash') == chash:
            done[row['task']] = row
    pending = {(j.get('config_hash'), j['task']): j['cand'] for _, j in lrlib.pending_jobs()}
    queued = []
    for task in lrlib.expand_tasks(args.tasks):
        if (chash, task) in pending:
            print(f'skip {task}: already queued/running as {pending[(chash, task)]}')
            continue
        row = done.get(task)
        if row and not (row.get('status') == 'ERROR' and args.rerun_errors):
            print(f'skip {task}: identical config already ran as {row["cand"]}: {lrlib.cell(row)}')
            continue
        if not queued:
            if args.note:
                registry['note'] = args.note
            registry_path.parent.mkdir(parents=True, exist_ok=True)
            registry_path.write_text(json.dumps(registry, indent=1) + '\n')
        stamp = time.strftime('%Y%m%dT%H%M%S') + f'{time.time() % 1:.6f}'[1:]
        job = dict(cand=args.cand, task=task, package_root=str(root), package_sha256=package_sha,
                   overrides=overrides, candidate_options=options, config_hash=chash, priority=args.priority,
                   submitted=stamp)
        path = lrlib.QUEUE / f'{stamp}-{args.cand}-{task}.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(job, indent=1) + '\n')
        tmp.rename(path)
        queued.append(task)
    print(f'{args.cand} [{chash}] queued {len(queued)}: {",".join(queued) if queued else "-"}')
    try:
        lrlib.rebuild_leaderboard()
    except Exception as error:
        print('leaderboard rebuild failed:', error)


if __name__ == '__main__':
    main()
