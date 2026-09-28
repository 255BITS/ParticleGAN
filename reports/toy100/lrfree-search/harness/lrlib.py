"""Shared paths, job hashing, ledger and leaderboard for the lrfree screening pool."""
from pathlib import Path
import hashlib
import json
import os
import time

BASE = Path('/ml2/hypergan/lrfree-20260926')
HARNESS = BASE / 'harness'
QUEUE = BASE / 'queue'
RUNNING = QUEUE / 'running'
RUNS = BASE / 'runs'
LEDGER = BASE / 'ledger.jsonl'
LEADERBOARD = BASE / 'LEADERBOARD.md'
CONFIG = BASE / 'pool-config.json'
PYTHON = '/tmp/pr38-default-env/bin/python'

IMAGE_TASKS = ['img_intensity2', 'img_blobs4', 'img_stripes2', 'img_bars4']
VECTOR_TASKS = list(json.loads((HARNESS / 'tasks' / 'vector_task_specs.json').read_text()))
RING_TASKS = ['ring_shift', 'stationary']
GATE_TASKS = ['mode_hold'] + IMAGE_TASKS + VECTOR_TASKS
# Frozen native 100-Gaussian problems (2026-09-27): opt-in only (preset 'native'); not part of 'gates'/'all'.
NATIVE_TASKS = ['grid100', 'rotated100', 'staggered100']
ALL_TASKS = GATE_TASKS + RING_TASKS + NATIVE_TASKS
PRESETS = {'quick': ['mode_hold'], 'images': IMAGE_TASKS, 'vectors': VECTOR_TASKS, 'ring': RING_TASKS,
           'gates': GATE_TASKS, 'all': GATE_TASKS + RING_TASKS, 'native': NATIVE_TASKS}
SHORT = {'mode_hold': 'mode_hold', 'img_intensity2': 'intens2', 'img_blobs4': 'blobs4', 'img_stripes2': 'stripes2',
         'img_bars4': 'bars4', 'vector_two_broad': 'v.broad', 'vector_unequal_mass': 'v.mass',
         'vector_unequal_width': 'v.width', 'vector_anisotropic': 'v.aniso', 'vector_overlap': 'v.overlap',
         'vector_spiral': 'v.spiral', 'ring_shift': 'ring_shift', 'stationary': 'stationary',
         'grid100': 'grid100', 'rotated100': 'rot100', 'staggered100': 'stag100'}
# Frozen custom22 hosts (2026-09-27): the 8 custom behavioral hosts of the 22-check suite (CPU jobs; learner =
# candidate GANTrainer policy via harness/components.py). Opt-in presets 'custom' (the 8) and 'all22'
# (gates + custom + native = 22 checks); 'gates'/'all' are unchanged.
CUSTOM_TASKS = ['two_pole', 'trajectory', 'residual_student', 'unipolar', 'ae_gan_hold', 'cover_leftover',
                'unused_token_hold', 'mid_scale_identity']
ALL_TASKS = ALL_TASKS + CUSTOM_TASKS
PRESETS.update(custom=CUSTOM_TASKS, all22=GATE_TASKS + CUSTOM_TASKS + NATIVE_TASKS)
SHORT.update(two_pole='two_pole', trajectory='traj', residual_student='resid', unipolar='unipolar',
             ae_gan_hold='ae_hold', cover_leftover='cover', unused_token_hold='unused', mid_scale_identity='mid_id')
# Options that change the measured behaviour (part of the dedupe hash).
BEHAVIOR_OPTIONS = ('evaluation_generate', 'serial_backward_argument', 'strict_streams', 'initialization',
                    'eval_output_noise', 'image_prior_perturb')
DEFAULT_CONFIG = dict(slots={'0': 7, '1': 7}, min_free_mib=1500, poll_seconds=2.0,
                      timeouts={'mode_hold': 1800, 'image': 1200, 'vector': 1800, 'ring_shift': 5400,
                                'stationary': 9000, 'grid100': 10800, 'rotated100': 10800, 'staggered100': 10800},
                      heavy_tasks={})


def expand_tasks(spec):
    out = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        for task in PRESETS.get(item, [item]):
            if task not in ALL_TASKS:
                raise ValueError(f'unknown task {task}; choose from {ALL_TASKS} or presets {list(PRESETS)}')
            if task not in out:
                out.append(task)
    return out


def load_json_arg(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    text = value.strip()
    if text.startswith('{'):
        return json.loads(text)
    return json.loads(Path(value).read_text())


def package_digest(package_root):
    root = Path(package_root).resolve() / 'particlegan'
    if not root.is_dir():
        raise FileNotFoundError(f'{root} is not a directory')
    h = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        h.update(str(path.relative_to(root)).encode() + b'\0' + path.read_bytes() + b'\0')
    return h.hexdigest()


def normalize(overrides, options):
    """Recipe overrides as screen.py will apply them + behaviour options (for hashing)."""
    loaded = dict(overrides)
    declared = {}
    if 'recipe_overrides' in loaded:
        declared = {k: loaded[k] for k in ('evaluation_generate', 'serial_backward_argument') if k in loaded}
        loaded = dict(loaded['recipe_overrides'])
    opts = dict(declared)
    opts.update(options)
    init = opts.get('initialization', 'batch_feature_zero')
    if init is not None:
        loaded.setdefault('initialization', init)
    behavior = {k: opts[k] for k in BEHAVIOR_OPTIONS if k in opts and k != 'initialization'}
    # Always hashed (2026-09-27 clean-eval fix): every new config hash differs from every pre-fix hash, whose
    # rows were scored with the training output noise added to the evaluated samples.
    behavior['eval_output_noise'] = bool(opts.get('eval_output_noise', False))
    # image_steps (2026-09-27): longer image runs, same every-25 cadence. Hashed only when set, so every config
    # without it keeps its existing hash.
    if opts.get('image_steps') is not None:
        behavior['image_steps'] = int(opts['image_steps'])
    # native_steps (2026-09-27): longer native100 runs (same schedule extended, gates scored at that budget).
    # Hashed only when set, so every config without it keeps its existing hash.
    if opts.get('native_steps') is not None:
        behavior['native_steps'] = int(opts['native_steps'])
    return loaded, opts, behavior


def eval_mode(row):
    """'clean' / 'noisy' scoring of a ledger row. Rows without the option predate the fix -> noisy."""
    options = row.get('options')
    if not isinstance(options, dict) or 'eval_output_noise' not in options:
        return 'noisy'
    return 'noisy' if options['eval_output_noise'] else 'clean'


def config_hash(package_sha, overrides, options):
    loaded, _, behavior = normalize(overrides, options)
    blob = json.dumps(dict(package=package_sha, overrides=loaded, options=behavior), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def read_ledger():
    if not LEDGER.exists():
        return []
    rows = []
    for line in LEDGER.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def latest_results(rows=None):
    """Last ledger row per (cand, task)."""
    out = {}
    for row in rows if rows is not None else read_ledger():
        out[(row['cand'], row['task'])] = row
    return out


def pending_jobs():
    """[(state, job)] for queued and running job files."""
    jobs = []
    for state, folder in (('queued', QUEUE), ('running', RUNNING)):
        if not folder.exists():
            continue
        for path in sorted(folder.glob('*.json')):
            try:
                job = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            job['_path'] = str(path)
            jobs.append((state, job))
    return jobs


def fmt(value, digits=3):
    return f'{value:.{digits}f}'.rstrip('0').rstrip('.') if isinstance(value, float) else str(value)


def failing_metrics(row):
    final = row.get('final') or {}
    bad = []
    for name, op, bound in row.get('thresholds') or []:
        value = final.get(name)
        if not isinstance(value, (int, float)):
            bad.append(name)
        elif not (value >= bound if op == '>=' else value <= bound):
            bad.append(name)
    return bad


def margin_text(row):
    """Final-value vs threshold margins for FAIL rows: 'mse+.0043' means the
    final misses by that much (smaller is closer); '' when nothing to show.
    Read-only: never changes PASS/FAIL, only surfaces near-miss distance."""
    if row.get('status') == 'PASS':
        return ''
    final = row.get('final') or {}
    parts = []
    for name, op, bound in row.get('thresholds') or []:
        value = final.get(name)
        if not isinstance(value, (int, float)) or not isinstance(bound, (int, float)):
            continue
        miss = (bound - value) if op == '>=' else (value - bound)
        if miss <= 0:
            continue
        scale = abs(bound) or 1.0
        parts.append(f'{name}{miss:+.{3}g}(x{miss / scale:.2g})')
    return ' '.join(parts)


def cell(row):
    status = row.get('status', '?')
    if status == 'ERROR':
        msg = str(row.get('error') or '')
        for key, short in (('budget exhausted', 'budget'), ('timeout', 'timeout'), ('stream transaction', 'stream'),
                           ('out of memory', 'OOM'), ('engine parity', 'parity'), ('EngineRefusal', 'refused')):
            if key in msg:
                return f'ERROR {short}'
        return 'ERROR ' + msg.split('(')[0][:20]
    if row['task'] in RING_TASKS:
        parts = []
        for s in row.get('segments') or []:
            if s.get('arrival') is None:
                parts.append('no arrival')
            else:
                lead = f"@{s['arrival']}" if s['start'] == 0 else f"+{s['delay']}"
                parts.append(f"{lead} {s['retained']}/{s['checks_since_arrival']}")
        return f"{status} " + ' | '.join(parts)
    text = f"{status} {row.get('passing_checks')}/{row.get('observations')}"
    if row.get('first_arrival') is not None:
        text += f" @{row['first_arrival']}"
    final = row.get('final') or {}
    if status != 'PASS':
        if row['task'] == 'mode_hold' or row['task'] in IMAGE_TASKS:
            if 'modes' in final:
                text += f" ({final['modes']}m hq{fmt(final.get('hq', 0), 2)})"
        else:
            bad = failing_metrics(row)
            if bad:
                text += ' [' + ','.join(b.replace('component_', 'c.').replace('_normalized', '') for b in bad) + ']'
    else:
        # PASS rows: confirm step = late arrival is itself a signal (v2
        # trajectory confirmed at the last step; a mid-run confirm is stronger).
        if row.get('first_arrival') is not None and row.get('observations'):
            text += f" (conf{row['first_arrival']}/{row['observations']})"
    if status != 'PASS' and row.get('final_streak'):
        text += f" sfx{row['final_streak']}"
    if status != 'PASS':
        margin = margin_text(row)
        if margin:
            text += f' <{margin}>'
    if row.get('stream_deviations'):
        text += ' *dev'
    return text


def rebuild_leaderboard():
    rows = read_ledger()
    latest = latest_results(rows)
    pending = pending_jobs()
    cands = {}
    for (cand, task), row in latest.items():
        cands.setdefault(cand, {})[task] = row
    registry = {}
    for cand in list(cands) + [job['cand'] for _, job in pending]:
        path = RUNS / cand / 'candidate.json'
        if path.exists() and cand not in registry:
            try:
                registry[cand] = json.loads(path.read_text())
            except json.JSONDecodeError:
                registry[cand] = {}
        cands.setdefault(cand, {})
    status_of = {}
    for state, job in pending:
        status_of[(job['cand'], job['task'])] = 'running' if state == 'running' else 'queued'
    present = [t for t in ALL_TASKS if any(t in tasks or (c, t) in status_of for c, tasks in cands.items())]

    def score(cand):
        tasks = cands[cand]
        gates = sum(1 for r in tasks.values() if r.get('status') == 'PASS')
        mode = tasks.get('mode_hold', {}).get('passing_checks') or 0
        total = sum(r.get('passing_checks') or 0 for t, r in tasks.items() if t not in RING_TASKS)
        run = sum(1 for r in tasks.values() if r.get('status') in ('PASS', 'FAIL'))
        return gates, mode, total, run
    order = sorted(cands, key=lambda c: tuple(-x for x in score(c)))
    lines = ['# lrfree screening leaderboard', '',
             f'Updated {time.strftime("%Y-%m-%d %H:%M:%S")}. Rows rank by gates passed, then mode_hold passing '
             'checks, then total passing checks over the 24-observation tasks. Cells: `PASS 19/24 @300` = status, '
             'passing observations, first passing observation; failing cells add final modes/HQ (images, mode_hold) '
             'or the threshold metrics failing at the final observation (vectors). Ring cells: '
             '`@arrival retained/checks | +changed-target-delay retained/checks`. `*dev` = stream deviation '
             '(non-strict run). Harness: harness/screen.py (frozen PR155 new-init hosts).', '',
             '`eval` column: `clean` = scored on clean generator samples (default since 2026-09-27); `noisy` = the '
             "training output noise was added to the scored samples (every row run before the fix, or option "
             '`eval_output_noise=true`). Noisy rows are not comparable to clean rows on HQ-limited tasks '
             '(img_intensity2 HQ needs rmse<=.06; eval noise .029 eats much of it).', '',
             '**PR #215/#217 source audit:** use [the fixed 13-gate and native comparison]'
             '(reports/pr215-vs-pr217.md). This raw table does not join the 1,200-step image and '
             '7,500-step stationary continuations. In particular, the `pr217-dv12-qr` image cells '
             'omit the source prior\'s latent jitter during evaluation and are provisional; the corrected '
             '`pr217-dv12-qr-imgjitter` and `-i1200` runs supply those image verdicts. The raw rank '
             'and `gates` count are not the fixed comparison.', '',
             '| # | cand | eval | gates | ' + ' | '.join(SHORT[t] for t in present) + ' | note |',
             '|---|---|---|---|' + '---|' * len(present) + '---|']
    for rank, cand in enumerate(order, 1):
        tasks = cands[cand]
        gates, _, _, run = score(cand)
        cells = []
        for task in present:
            if (cand, task) in status_of:
                prefix = status_of[(cand, task)]
                cells.append(prefix if task not in tasks else f'{cell(tasks[task])} ({prefix})')
            elif task in tasks:
                cells.append(cell(tasks[task]))
            else:
                cells.append('')
        note = (registry.get(cand) or {}).get('note', '') or ''
        modes = sorted({eval_mode(r) for r in tasks.values()})
        mode = '/'.join(modes) if modes else ''
        lines.append(f'| {rank} | {cand} | {mode} | {gates}/{run} | ' + ' | '.join(cells) + f' | {note} |')
    queued = sum(1 for s, _ in pending if s == 'queued')
    running = sum(1 for s, _ in pending if s == 'running')
    lines += ['', f'Queue: {queued} queued, {running} running; ledger rows: {len(rows)}.', '']
    tmp = LEADERBOARD.with_suffix('.md.tmp')
    tmp.write_text('\n'.join(lines))
    os.replace(tmp, LEADERBOARD)


def compact_matrix(cands):
    latest = latest_results()
    pending = {(j['cand'], j['task']): s for s, j in pending_jobs()}
    out = []
    for cand in cands:
        tasks = [t for t in ALL_TASKS if (cand, t) in latest or (cand, t) in pending]
        out.append(f'== {cand}')
        for task in tasks:
            if (cand, task) in pending:
                out.append(f'  {SHORT[task]:<11} {pending[(cand, task)]}')
            else:
                row = latest[(cand, task)]
                out.append(f'  {SHORT[task]:<11} {cell(row)}  ({row.get("seconds")}s)')
    return '\n'.join(out)
