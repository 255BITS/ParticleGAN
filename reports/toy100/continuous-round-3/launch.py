#!/usr/bin/env python3
"""Launch selected rolling search lanes, enforcing 1 Codex + 7 Grok globally."""
import argparse
from collections import Counter
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(os.environ.get('GAN_WORKSPACE', '/ml2/hypergan')).resolve()
GPU0 = 'GPU-72c1b506-891d-b8bc-b353-e020585e1c47'
GPU1 = 'GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69'
REPO = ROOT / 'ParticleGAN-k3p-continuous-search'
LIMITS = {'codex': 1, 'grok': 7}
LANES = {
    'verify_p3_gates': ('grok', GPU0, '''Verification ONLY, no new mechanisms.
Use UNCHANGED reports/toy100/continuous-round-2/sources/p3_floor_reopen/;
verify hashes in sources/source-hashes.json and policy installation. This explicitly
overrides the normal search hold/shift-first order: P3 already has hold1200+300,
stationary5/5, prehold120/120, shiftFAIL77/81. Do not repeat those protocols or
baseline runs. Its noise remains scheduled; preserve that limitation. Test own
mode_hold, vector_unequal_mass, vector_unequal_width, img_stripes2 in that order,
original fixtures/runtime/budgets. Stop on first failed sensitive gate, reporting
the rest NOT_RUN. Only if all4pass, complete own remaining18 within the cap,
including native7000 coverage AND accuracy; no coefficient or seed experiments.
Record actual applied rates and source hashes. This is prospective-lead screening,
not promotion; its existing failed shift is binding. Leave complete evidence.'''),
    'k3p_extragradient': ('codex', GPU0, '''Own a clean shared-sample game
predictor/corrector from selected K3P. Read research-notes.md. Preserve and restore
temporary optimizer/EMA/particle/RNG state, one committed update per role, declare
added work. Compare to failed predictive raw-delta GD1, not an unchanged rerun.
Do not spend the attempt building a new benchmark. If this cannot be made sound,
report the exact implementation limitation and preserve measured evidence.'''),
    'k3p_local_curvature': ('grok', GPU0, '''Own reversible local-curvature step
control from selected K3P. Read research-notes.md and its secant/growth rule.
Sampling noise and opponent motion require an explicit treatment; avoid an
ever-growing accumulator that silently stops learning. Retain acquisition and
declare how critic mixing and noise become horizon independent. No LR grid.'''),
    'k3p_negative_momentum': ('grok', GPU1, '''Own negative momentum using
previous ACTUAL displacement in alternating game updates. Read research-notes.md.
GD1's raw_delta + .5*(raw_delta-previous_raw_delta) already failed: recursive
negative actual-displacement memory is different. Declare network/prior/sparse
row handling, preserve K3P lineage and acquisition, no beta grid. Horizon-based
components are allowed only as explicitly labeled intermediate ablations.'''),
    'k3p_responsive_precision': ('grok', GPU1, '''Own evidence-driven precision
recovery from selected K3P. Read round-2 adversarial-progress report and sources.
AP3 held 1200+300 and pre-hold120/120, recovered72/81, but forced moderate mobility
for800updates and later lost precision. AP1 reopened early R1 and trapped4modes;
AP2 full-rate anchor-on reopen lost precision. Propose a training-signal-based
reversible closing rule without that forced dwell, and remove horizon-dependent
noise. AP3 is diagnostic, not a verified new shared base; earn every own score.'''),
    'k3p_reference_response': ('grok', GPU0, '''Own coupling between critic
reference tracking and ordinary gradient innovation. First read completed round-2
anchor-tracking results to avoid repeating them. Faster EMA alone already failed
recovery; from-start anchoring trapped acquisition. Propose only a distinct,
evidence-driven mechanism retaining K3P acquisition and positive adaptive motion.'''),
    'k3p_joint_trust': ('grok', GPU1, '''Own a coupled trust bound on actual
game displacement from selected K3P. Read round-2 balanced-updates and game-damping
results first. Distinguish actual displacement control from independent cosine
learning-rate controllers that ruined acquisition. Preserve sparse prior hooks;
do not add a per-toy rule or repeat an unchanged failed mechanism.'''),
    'k3p_signal_noise': ('grok', GPU0, '''Own horizon-independent noise and
acquisition from selected K3P. Read round-2 noise-and-signal failures first. A
stationary noise/rate substitution alone failed and constant LR disables the
old anchor trigger. Use measured failures to declare a distinct reversible rule,
with no total budget, known shift times, or evaluation signal in the learner.'''),
    'k3p_particle_mobility': ('grok', GPU1, '''Own relative learned-prior and
network mobility from selected K3P. Inspect round-2 precision and joint-update
failures first. Preserve bounded sparse-latent and direct-response mechanisms
unless a measured failure motivates a precise change. No per-task exceptions,
coefficient grids or copied A3 scores. Acquire before damping, adapt without a
final horizon, and measure all required protocols for each proposal.'''),
}


def live_attempts():
    """Count reservations, including agents preparing their benchmark worker."""
    registry = ROOT / 'gan-attempts/active-batches.txt'
    live = []
    seen = set()
    for entry in registry.read_text().splitlines() if registry.exists() else []:
        for row in json.loads((Path(entry) / 'batch.json').read_text()):
            pid = row['pid']
            if pid in seen:
                continue
            seen.add(pid)
            try:
                stat = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0]
                cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
            except FileNotFoundError:
                continue
            if stat != 'Z' and b'try-gan.sh' in cmd and row['directory'].encode() in cmd:
                live.append(row)
    return live


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane', action='append', choices=LANES, required=True)
    args, remaining = parser.parse_known_args()
    if any(x.split('=')[0] in ('--engine', '--model', '--gpu') for x in remaining):
        parser.error('Engine/model/GPU overrides would bypass the capacity policy')
    if len(args.lane) != len(set(args.lane)):
        parser.error('Duplicate lane')
    (ROOT / 'gan-attempts').mkdir(exist_ok=True)
    with (ROOT / 'gan-attempts/continuous-launch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        live = live_attempts()
        engines = Counter(r['engine'] for r in live)
        gpus = Counter(r['gpu'] for r in live)
        for name in args.lane:
            engine, gpu, _ = LANES[name]
            if any(r['lane'] == name for r in live):
                parser.error(f'{name} is already active')
            engines[engine] += 1
            gpus[gpu] += 1
        if any(engines[k] > v for k, v in LIMITS.items()) or engines.get('claude', 0):
            parser.error(f'Engine capacity exceeded: {dict(engines)}; limits {LIMITS}')
        if any(n > 4 for n in gpus.values()):
            parser.error(f'Four-worker GPU capacity exceeded: {dict(gpus)}')
        spec = importlib.util.spec_from_file_location('rolling_launcher', ROOT / 'launch-gan-formulations.py')
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        launcher.REPO = REPO
        base = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
        launcher.COMMON = '\n\n'.join(subprocess.check_output([
            'git', '-C', str(REPO), 'show', base + ':' + p], text=True) for p in (
                'reports/toy100/k3p-base/continuous-search.md',
                'reports/toy100/continuous-round-3/SEARCH.md'))
        launcher.LANES = {name: LANES[name][2] for name in args.lane}
        launcher.LANE_SETTINGS = {name: dict(engine=LANES[name][0], gpu=LANES[name][1],
            model='gpt-6-astra' if LANES[name][0] == 'codex' else 'grok-4.7') for name in args.lane}
        launcher.REFERENCES = '''Read-only evidence: reports/toy100/continuous-round-1/README.md;
reports/toy100/continuous-round-2/README.md and attempts/LANE/result.md;
reports/toy100/continuous-search-tools/research-notes.md;
/ml2/hypergan/gan-attempts/formulations-20260925T165310Z/.
Pinned source/runtime/fixtures: reports/toy100/gap-fill-20260925/manifest.json.
Do not modify previous attempts. Keep canonical results in this attempt/tests.jsonl.
'''
        sys.argv = [sys.argv[0], *remaining]
        for flag, value in [('--minutes', '45'), ('--proposals', '3')]:
            if not any(a == flag or a.startswith(flag + '=') for a in remaining):
                sys.argv.extend([flag, value])
        launcher.main()


if __name__ == '__main__':
    main()
