#!/usr/bin/env python3
"""Keep a fixed number of independent agent attempts running on one search round.

Each slot runs its own try-gan.sh attempt in its own branch/worktree; when one
finishes the slot immediately starts a fresh attempt, so N searches are always
in flight. Lane briefs come from an existing launcher (default: the live PR139
round), so a pool explores exactly what a one-shot batch would.

Stop it by creating STOP in the batch directory: running attempts finish and
keep their artifacts, and no new attempt starts. Resize it by writing a slot
count to SLOTS in the same directory: new slots start at once, removed ones
retire after their current attempt, and no running attempt is ever interrupted.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
GPU0 = 'GPU-72c1b506-891d-b8bc-b353-e020585e1c47'
FAST_SECONDS = 300      # An attempt shorter than this failed to get going.
MAX_BACKOFF = 600


def load_round(entrypoint):
    """Import a launcher without running it and return its lane configuration."""
    spec = importlib.util.spec_from_file_location('gan_round_' + entrypoint.stem.replace('-', '_'),
                                                  entrypoint)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, 'launcher', module)


def slot_names(lanes, slots):
    """Round-robin the lanes; a repeated lane gets its own numbered slot."""
    names, used = [], {}
    for index in range(slots):
        lane = lanes[index % len(lanes)]
        used[lane] = used.get(lane, 0) + 1
        names.append((lane if used[lane] == 1 else f'{lane}-{used[lane]}', lane))
    return names


MAX_SLOTS = 16


class Pool:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.repo = Path(args.repo or config.REPO).resolve()
        self.base = subprocess.check_output(
            ['git', '-C', str(self.repo), 'rev-parse', args.base], text=True).strip()
        research = json.loads(subprocess.check_output(
            ['git', '-C', str(self.repo), 'show',
             self.base + ':reports/toy100/current-research-base.json'], text=True))
        self.gate_order = (list(config.GATE_ORDER) if getattr(config, 'GATE_ORDER', None)
                           else research['first_gates'] + research['remaining_regression_gates'])
        self.candidate = research['candidate']
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.batch = ROOT / 'gan-attempts' / f'{args.engine}-pool-{stamp}'
        self.batch.mkdir(parents=True)
        self.stop_file = self.batch / 'STOP'
        self.log_file = (self.batch / 'pool.log').open('a', buffering=1)
        self.slots = []
        self.stopping = False

    def log(self, message):
        line = f'{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} {message}'
        self.log_file.write(line + '\n')
        print(line, flush=True)

    def prepare(self):
        self.lanes = list(self.config.LANES)
        self.used = {}
        (self.batch / 'launcher.sha256').write_text(
            hashlib.sha256((ROOT / 'try-gan.sh').read_bytes()).hexdigest() + '\n')
        (self.batch / 'launch-source.py').write_bytes(self.args.lanes_from.read_bytes())
        (self.batch / 'pool-source.py').write_bytes(Path(__file__).read_bytes())
        self.note = (f'\nThis is one of several independent attempts running concurrently on'
                f' this round,\neach in its own branch and worktree on GPU {self.args.gpu}.'
                ' Other attempts are invisible\nto you: never read, wait for or edit another'
                " attempt's checkout, and never assume\nanother agent produced a result you did"
                ' not measure yourself in this checkout.\n')
        for _ in range(self.args.slots):
            self.make_slot()
        (self.batch / 'SLOTS').write_text(f'{self.args.slots}\n')
        self.register()
        self.write_batch()

    def make_slot(self):
        """Add one slot, continuing the lane round-robin from the existing ones."""
        lane = self.lanes[len(self.slots) % len(self.lanes)]
        self.used[lane] = self.used.get(lane, 0) + 1
        name = lane if self.used[lane] == 1 else f"{lane}-{self.used[lane]}"
        directory = self.batch / name
        directory.mkdir(exist_ok=True)
        (directory / 'brief.md').write_text(
            self.config.COMMON + '\nAssigned lane:\n' + self.config.LANES[lane]
            + '\n\n' + self.config.REFERENCES + self.note)
        slot = dict(lane=name, focus=lane, directory=directory, process=None, started=None,
                    attempts=0, fast=0, resume_at=0.0, history=[], retired=False)
        self.slots.append(slot)
        return slot

    def resize(self):
        """Follow the SLOTS file; growing is immediate, shrinking waits for exits."""
        try:
            want = int((self.batch / 'SLOTS').read_text().split()[0])
        except (OSError, ValueError, IndexError):
            return False
        want = max(1, min(MAX_SLOTS, want))
        live = [s for s in self.slots if not s['retired']]
        if want == len(live):
            return False
        for slot in self.slots:
            if len(live) >= want:
                break
            if slot['retired']:
                slot['retired'] = False
                live.append(slot)
        while len(live) < want:
            live.append(self.make_slot())
        for slot in reversed(live):
            if len(live) <= want:
                break
            slot['retired'] = True
            live.remove(slot)
        self.log(f'slots -> {want} (running attempts are never interrupted)')
        return True

    def register(self):
        """List this batch for monitor-gan.py alongside any other live batch."""
        registry = ROOT / 'gan-attempts/active-batches.txt'
        entries = [line.strip() for line in
                   (registry.read_text().splitlines() if registry.exists() else []) if line.strip()]
        entries = [e for e in entries if e != str(self.batch) and Path(e, 'batch.json').exists()]
        temporary = registry.with_suffix('.tmp')
        temporary.write_text('\n'.join(entries + [str(self.batch)]) + '\n')
        temporary.replace(registry)

    def write_batch(self):
        records = [dict(lane=s['lane'], focus=s['focus'], retired=s['retired'],
                        stopped_by_supervisor=s['retired'] and not s['process'],
                        pid=(s['process'].pid if s['process'] else 0),
                        directory=str(s['directory']), engine=self.args.engine, model=self.args.model,
                        base=self.base, starting_candidate=self.candidate, gpu=self.args.gpu,
                        workers=1, gate_order=self.gate_order, attempts=s['attempts'],
                        minutes=self.args.minutes, history=s['history'][-5:])
                   for s in self.slots]
        temporary = self.batch / 'batch.json.tmp'
        temporary.write_text(json.dumps(records, indent=2) + '\n')
        temporary.replace(self.batch / 'batch.json')

    def command(self, slot):
        command = [str(ROOT / 'try-gan.sh'), '--engine', self.args.engine,
                   '--repo', str(self.repo), '--base', self.base, '--gpu', self.args.gpu,
                   '--minutes', str(self.args.minutes), '--candidates', str(self.args.proposals),
                   '--workers', '1', '--runs-dir', str(slot['directory']),
                   '--prompt-file', str(slot['directory'] / 'brief.md'), '--focus', slot['focus']]
        if self.args.model:
            command += ['--model', self.args.model]
        if self.args.budget_usd:
            command += ['--budget-usd', str(self.args.budget_usd)]
        return command

    def launch(self, slot):
        with (slot['directory'] / 'launcher.log').open('ab') as log:
            slot['process'] = subprocess.Popen(self.command(slot), cwd=ROOT,
                                               stdin=subprocess.DEVNULL, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
        slot['started'] = time.time()
        slot['attempts'] += 1
        self.log(f"start {slot['lane']} attempt {slot['attempts']} pid {slot['process'].pid}")

    def reap(self, slot):
        code = slot['process'].poll()
        if code is None:
            return False
        seconds = int(time.time() - slot['started'])
        slot['process'] = None
        slot['history'].append(dict(attempt=slot['attempts'], exit=code, seconds=seconds))
        # A real attempt never finishes this fast, whatever it exits with:
        # back off so a broken brief or a rate limit cannot hot-loop the slot.
        if seconds < FAST_SECONDS:
            slot['fast'] += 1
            backoff = min(MAX_BACKOFF, 30 * 2 ** (slot['fast'] - 1))
            slot['resume_at'] = time.time() + backoff
            self.log(f"exit {slot['lane']} code {code} after {seconds}s "
                     f"(short run {slot['fast']}; waiting {backoff}s)")
        else:
            slot['fast'] = 0
            self.log(f"exit {slot['lane']} code {code} after {seconds}s")
        return True

    def total_attempts(self):
        return sum(s['attempts'] for s in self.slots)

    def run(self):
        self.log(f'pool {self.batch} engine={self.args.engine} model={self.args.model or "default"} '
                 f'slots={self.args.slots} minutes={self.args.minutes} gpu={self.args.gpu} '
                 f'base={self.base}')
        self.log(f'stop with: touch {self.stop_file}')
        for received in (signal.SIGTERM, signal.SIGINT):
            signal.signal(received, self.on_signal)
        while True:
            changed = self.resize()
            for slot in self.slots:
                if slot['process'] is not None:
                    changed |= self.reap(slot)
                    continue
                if self.stopping or self.stop_file.exists():
                    if not self.stopping:
                        self.stopping = True
                        self.log('STOP seen; letting running attempts finish')
                    continue
                if slot['retired']:
                    continue
                if self.args.max_attempts and self.total_attempts() >= self.args.max_attempts:
                    continue
                if time.time() < slot['resume_at']:
                    continue
                self.launch(slot)
                changed = True
            if changed:
                self.write_batch()
            if not any(s['process'] for s in self.slots):
                if self.stopping or self.stop_file.exists():
                    self.log('all attempts finished; pool stopped')
                    return 0
                if self.args.max_attempts and self.total_attempts() >= self.args.max_attempts:
                    self.log(f'reached --max-attempts {self.args.max_attempts}; pool stopped')
                    return 0
            time.sleep(5)

    def on_signal(self, *_):
        if not self.stopping:
            self.stopping = True
            self.log('signal received; letting running attempts finish')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--engine', default='claude', choices=('claude', 'codex', 'grok', 'opencode'))
    parser.add_argument('--model', default='', help='engine default when empty')
    parser.add_argument('--slots', type=int, default=5)
    parser.add_argument('--minutes', type=int, default=45)
    parser.add_argument('--proposals', type=int, default=3)
    parser.add_argument('--gpu', default=GPU0, help='physical GPU index or UUID for every slot')
    parser.add_argument('--repo', help='default: the launcher round\'s own repo')
    parser.add_argument('--base', default='HEAD')
    parser.add_argument('--budget-usd', type=float, default=30.0,
                        help='per-attempt spend cap, 0 for none (Claude only)')
    parser.add_argument('--lanes-from', type=Path, default=ROOT / 'launch-gan-sparse.py',
                        help='launcher whose COMMON/LANES/REFERENCES define the round')
    parser.add_argument('--max-attempts', type=int, default=0, help='0 runs until STOP')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.slots <= MAX_SLOTS:
        parser.error('--slots must be in 1..16')
    if not 1 <= args.minutes <= 180:
        parser.error('--minutes must be in 1..180')
    if not 1 <= args.proposals <= 16:
        parser.error('--proposals must be in 1..16')
    if args.engine != 'claude':
        args.budget_usd = 0
    config = load_round(args.lanes_from)
    if args.dry_run:
        lanes = list(config.LANES)
        print(json.dumps(dict(engine=args.engine, model=args.model, slots=args.slots,
                              lanes=lanes, slot_names=[n for n, _ in slot_names(lanes, args.slots)],
                              repo=str(args.repo or config.REPO), gpu=args.gpu,
                              minutes=args.minutes, proposals=args.proposals,
                              budget_usd=args.budget_usd), indent=2))
        return 0
    pool = Pool(args, config)
    pool.prepare()
    return pool.run()


if __name__ == '__main__':
    sys.exit(main())
