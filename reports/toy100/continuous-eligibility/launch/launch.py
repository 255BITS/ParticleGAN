#!/usr/bin/env python3
"""Preview or, after compaction, fill three reviewed external Codex lanes."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def live_drivers(driver):
    """Count actual shared-driver processes, including setup reservations."""
    live = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            argv = path.joinpath('cmdline').read_bytes().decode().split('\0')
            state = path.joinpath('stat').read_text().split(') ', 1)[1].split()[0]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        # Match the executable/script position, never text embedded in a prompt.
        if state == 'Z' or str(driver) not in argv[:2]:
            continue

        def option(name, default):
            return argv[argv.index(name) + 1] if name in argv else default

        live.append(dict(pid=int(path.name),
                         directory=option('--runs-dir', ''),
                         workers=int(option('--workers', '3'))))
    return live


def completed_attempts(record):
    directory = Path(record['directory'])
    return sorted(p for p in directory.iterdir()
                  if p.is_dir() and (p / 'run.txt').exists()) if directory.exists() else []


def make_prompt(config, lane, previous, review):
    evidence = Path(config['evidence_repo'])
    prompt = (HERE / 'COMMON.md').read_text() + '\n\n' + (
        HERE / config['lanes'][lane]['brief']).read_text()
    prompt += '\n\nRead-only research evidence checkout: ' + str(evidence) + '\n'
    prompt += ('Current eligibility audit: ' + str(HERE.parent) + '\n'
               'Shared evaluator declaration (not a learner input): ' +
               str(HERE / 'evaluation-protocols.json') + '\n'
               'API source base: ' + config['api_commit'] + '\n'
               'Historical research outcomes must not override this brief.\n')
    if previous:
        prompt += '\nPrevious attempts in this lane, inspect before proposing a successor:\n'
        for record in previous:
            for attempt in completed_attempts(record):
                prompt += '- ' + str(attempt) + '\n'
    if review:
        prompt += '\nSupervisor review and next instructions:\n' + review + '\n'
    return prompt


def command(config, lane, directory, brief):
    return [config['driver'], '--engine', config['engine'], '--model', config['model'],
            '--repo', config['api_repo'], '--base', config['api_commit'],
            '--gpu', config['lanes'][lane]['gpu'], '--minutes', str(config['minutes']),
            '--candidates', str(config['proposals_per_attempt']), '--workers', '1',
            '--runs-dir', str(directory), '--prompt-file', str(brief), '--focus', lane]


def validate_config(config):
    if (config['engine'], config['model'], config['reasoning_effort']) != (
            'codex', 'gpt-6-astra', 'max'):
        raise ValueError('This handoff is pinned to external Codex gpt-6-astra/max.')
    if (config['minutes'], config['max_agents'], config['workers_per_agent']) != (0, 3, 1):
        raise ValueError('Expected no hard timeout and three one-worker lanes.')
    driver = Path(config['driver'])
    if sha256(driver) != config['driver_sha256']:
        raise ValueError('Shared driver changed; inspect and update the pinned hash before launch.')
    resolved = subprocess.check_output([
        'git', '-C', config['api_repo'], 'rev-parse',
        config['api_commit'] + '^{commit}'], text=True).strip()
    if resolved != config['api_commit']:
        raise ValueError('Expected the exact pinned API commit.')
    subprocess.run(['git', '-C', config['api_repo'], 'cat-file', '-e',
                    resolved + ':particlegan/ka2.py'], check=True,
                   stdout=subprocess.DEVNULL)


def build_plan(config, selected, records, review):
    live = live_drivers(Path(config['driver']))
    live_attempts = {(row['pid'], row['directory']) for row in live}
    active_lanes = {row['lane'] for row in records
                    if (row['pid'], row['directory']) in live_attempts}
    available = max(0, min(config['max_agents'] - len(live),
                           3 - sum(row['workers'] for row in live)))
    pending = [lane for lane in selected if lane not in active_lanes][:available]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + str(os.getpid())
    wave = Path(config['runs_dir']) / stamp
    plans = []
    for lane in pending:
        previous = [row for row in records if row['lane'] == lane]
        directory = wave / lane
        brief = directory / 'brief.md'
        plans.append(dict(lane=lane, directory=str(directory),
                          command=command(config, lane, directory, brief),
                          prompt=make_prompt(config, lane, previous, review),
                          requires_review=bool(previous and not review),
                          gpu=config['lanes'][lane]['gpu']))
    return live, plans


def register(config):
    registry = Path(config['workspace']) / 'gan-attempts/active-batches.txt'
    rows = registry.read_text().splitlines() if registry.exists() else []
    directory = config['runs_dir']
    if directory not in rows:
        registry.write_text('\n'.join(rows + [directory]) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-after-compaction', action='store_true',
                        help='actually start idle lanes; omit for a read-only preview')
    parser.add_argument('--lane', action='append', help='refill only this named lane')
    parser.add_argument('--review-note', type=Path,
                        help='required evidence review/next instructions before replenishing a lane')
    args = parser.parse_args(argv)
    config = read_json(HERE / 'config.json', None)
    validate_config(config)
    selected = args.lane or list(config['lanes'])
    if len(set(selected)) != len(selected) or any(x not in config['lanes'] for x in selected):
        parser.error('Choose distinct lane names from config.json.')
    review = args.review_note.read_text().strip() if args.review_note else ''
    if args.review_note and not review:
        parser.error('Review note is empty.')
    runs = Path(config['runs_dir'])
    record_file = runs / 'batch.json'
    stop_paths = [Path(config['workspace']) / 'gan-attempts/STOP', runs / 'STOP']
    records = read_json(record_file, [])
    live, plans = build_plan(config, selected, records, review)
    if not args.launch_after_compaction:
        print(json.dumps(dict(mode='PREPARED_ONLY_NO_LAUNCH', api_commit=config['api_commit'],
                              model=config['model'], reasoning_effort='max', minutes=0,
                              live_agents=live, stop_markers=[str(p) for p in stop_paths if p.exists()],
                              plans=plans), indent=2))
        return
    runs.parent.mkdir(parents=True, exist_ok=True)
    # Shared with the existing rolling launcher: reserve all lanes atomically.
    with (runs.parent / 'continuous-launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if any(p.exists() for p in stop_paths):
            parser.error('STOP is present; preserve it until the supervisor resumes this search after compaction.')
        records = read_json(record_file, [])
        live, plans = build_plan(config, selected, records, review)
        if any(plan['requires_review'] for plan in plans):
            parser.error('Inspect completed evidence and provide --review-note before replenishing a lane.')
        if not plans:
            print(json.dumps(dict(started=[], live_agents=live, reason='no idle capacity')))
            return
        runs.mkdir(exist_ok=True)
        for plan in plans:
            # Recheck STOP between spawns; never remove a stop marker automatically.
            if any(p.exists() for p in stop_paths):
                break
            directory = Path(plan['directory'])
            directory.mkdir(parents=True)
            (directory / 'brief.md').write_text(plan['prompt'])
            (directory / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
            (directory / 'launcher.py').write_bytes(Path(__file__).read_bytes())
            (directory / 'evaluation-protocols.json').write_bytes(
                (HERE / 'evaluation-protocols.json').read_bytes())
            with (directory / 'launcher.log').open('wb') as log:
                process = subprocess.Popen(plan['command'], cwd=config['workspace'],
                                           stdin=subprocess.DEVNULL, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            record = {key: plan[key] for key in ('lane', 'directory', 'command', 'gpu')}
            record.update(pid=process.pid, base=config['api_commit'], engine='codex',
                          model='gpt-6-astra', workers=1, reasoning_effort='max', minutes=0,
                          source_sha256=sha256(Path(__file__)),
                          brief_sha256=sha256(directory / 'brief.md'),
                          evaluation_protocols_sha256=sha256(directory / 'evaluation-protocols.json'),
                          review_note_sha256=sha256(args.review_note) if args.review_note else None)
            records.append(record)
            atomic_json(record_file, records)
            register(config)
            print(json.dumps(dict(started=record)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
