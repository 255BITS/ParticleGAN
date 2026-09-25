#!/usr/bin/env python3
"""Start bounded formulation attempts and point the live monitor at them."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT / 'ParticleGAN-selected-h-stability-base'
GPU = 'GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69'
PREVIOUS = ROOT / 'gan-attempts/cpu-recipe-gpu-port-20260924T225604Z'

COMMON = '''Find a better GAN formulation that passes the frozen GPU toy suite and
stays good after convergence. This is formulation research, not another numerical
portability audit. Start from configs/toy100/constraints_simple_regularization.json
with CPU initialization and native CUDA training/random draws. Read AGENTS.md,
reports/toy100/current-research-base.json and reports/toy100/formulation-search.md.

The user now explicitly authorizes changes to adversarial objectives, critic
regularization, and optimizer/update formulations. Preserve the original schedules,
noise schedules and auxiliary AE/token host terms unless a specifically declared
mechanism requires a change. This supersedes older brief restrictions that would
freeze the exact loss or optimizer. Keep architecture, data, seeds, evaluation,
thresholds, and frozen training-step budgets fixed. No target fitting, mode labels,
centers, target statistics, or metric feedback in training. It must remain a GAN.
Do not spend extra optimizer steps silently: declare added forward/backward work
and update counts, and distinguish equal-step results from equal-compute results.

Spend time on measured proposals. Use the existing probe.py and prepare.py in
reports/toy100/cpu-recipe-gpu-port; prepare sources once, and reuse retained CPU
initialization fixtures. A copied probe may install your mechanism; snapshot/hash
the exact executed code and record the declaration before launching each candidate.
Do not build a replacement benchmark or repeat completed baseline/version controls.
No recursive scans outside your checkout. Prior artifacts below are read-only.

Run BOTH mode_hold and vector_unequal_mass for every meaningful proposal, even when
the first fails: learning the tradeoff is cheap. Only candidates that pass both
advance to trajectory, img_intensity2, img_bars4, img_blobs4, then their OWN full 22
GPU gates. Do not combine passes from different variants, or infer 20/22 from 4/6.
The frozen verdict requires sustained success, not a favorable final observation.
Once a candidate clears all 22, run relevant regressions and a continuation from
its own converged checkpoint/RNG with its declared schedule. Passing acquisition
alone is not stable-convergence qualification. Keep metrics visible in tests.jsonl.

At most six distinct proposals total, in short adaptive waves of up to three.
These are different mechanisms or evidence-driven refinements, not a coefficient
grid. Keep existing fixed seeds: no seed sweeps. Start real GPU training within
five minutes. One benchmark worker; no nested agents. Keep logs concise: inspect
targeted source slices and summarized JSON fields, never dump full manifests or
observation arrays. Reuse existing audits, add only necessary mechanism checks.
After failures, adapt the next proposal within your assigned lane. Do not stop at
the first partial pass. Stop early only for qualification, exhausted useful ideas,
the six-proposal cap, or the time limit. Reserve time to preserve failed evidence,
report exact commands and rank the measured candidates with NOT_RUN explicit.
'''

LANES = {
    'critic_formulation': '''Own critic/adversarial-objective formulation. Inspect the
existing relativistic logistic loss and b_cap regularizer. Test a few generally
applicable ways to provide stable, non-saturating gradients without starving rare
components (for example a smooth or zero-centered critic constraint supported by
existing code). Keep ordinary Adam and native RNG. Prefer available implementations
to speculative infrastructure. Other lanes own particle preconditioning and game
update dynamics. Do not merely scan regularization coefficients.''',
    'particle_geometry': '''Own particle-update formulation. The prior round found
multiplicity-averaged particle gradients pass unequal mass with eigen ratio .487
but fail ring (7 modes); post-Adam displacement capping passes ring (8 modes/HQ1)
but loses the rare component entirely. Row clipping failed. Gaussian stream
isolation plus multiplicity averaging also failed ring, so do not repeat it.
Inspect sampling-frequency-aware, shared-moment or geometry-aware preconditioning
that preserves rare-particle mobility rather than blindly clipping displacement.
Use the existing particle hooks as implementation references. Keep critic loss,
network optimizer and RNG unchanged unless explicitly motivated in a declaration.
Do not select a different rule per toy or use target-mode information.''',
    'game_dynamics': '''Own the adversarial game update rule. Test coherent corrections
to alternating G/D dynamics, such as a memory-based optimistic/corrective gradient
update, using existing training hooks where possible. Preserve the original decay,
noise, data and loss as the starting formulation. Target convergence and sustained
coverage, not merely a quiet transient. Keep the same outer step budgets and
declare any extra model/gradient evaluations. Do not silently add critic steps,
freeze training based on evaluation, or implement a non-GAN target fitter. Other
lanes own loss regularization and particle-only preconditioning.''',
}

DEFAULT_COMMON = COMMON
GATE_ORDER = None
LANE_SETTINGS = {}  # Optional per-lane engine/model/GPU defaults; CLI overrides them.
REFERENCES = ('Read-only prior results: ' + str(PREVIOUS / 'review.json') + '\n'
              + 'Particle implementation reference: ' + str(PREVIOUS / 'particle_updates/20260924T225604Z-1994076/repo/reports/toy100/particle-port-attempt') + '\n')


def register(batch):
    """List this batch for monitor-gan.py, which follows every live batch at once."""
    registry = ROOT / 'gan-attempts/active-batches.txt'
    entries = [line.strip() for line in
               (registry.read_text().splitlines() if registry.exists() else []) if line.strip()]
    entries = [e for e in entries if e != str(batch) and Path(e, 'batch.json').exists()]
    temporary = registry.with_suffix('.tmp')
    temporary.write_text('\n'.join(entries + [str(batch)]) + '\n')
    temporary.replace(registry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minutes', type=int, default=60)
    parser.add_argument('--proposals', type=int, default=6)
    parser.add_argument('--engine', choices=('codex', 'claude', 'grok'))
    parser.add_argument('--model', default='', help='engine default when empty')
    parser.add_argument('--gpu', help='physical GPU index or UUID; overrides lane defaults')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.minutes <= 180:
        parser.error('--minutes must be in 1..180')
    if not 1 <= args.proposals <= 6:
        parser.error('--proposals must be in 1..6')
    base = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    research = json.loads(subprocess.check_output(
        ['git', '-C', str(REPO), 'show', base + ':reports/toy100/current-research-base.json'], text=True))
    default_order = research['first_gates'] + research['remaining_regression_gates']
    gate_order = default_order if GATE_ORDER is None else list(GATE_ORDER)
    if len(gate_order) != len(set(gate_order)) or set(gate_order) != set(default_order):
        raise ValueError('Gate order must contain each selected-base gate exactly once')
    common = COMMON
    if common == DEFAULT_COMMON:
        common = subprocess.check_output(
            ['git', '-C', str(REPO), 'show', base + ':reports/toy100/h_stability/SEARCH.md'], text=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    batch = ROOT / 'gan-attempts' / ('formulations-' + stamp)
    plans = {}
    for lane in LANES:
        settings = LANE_SETTINGS.get(lane, {})
        engine = args.engine or settings.get('engine', 'codex')
        if engine not in ('codex', 'claude', 'grok'):
            raise ValueError('Unsupported engine for ' + lane)
        plans[lane] = dict(engine=engine,
                           model=args.model or ('' if args.engine else settings.get('model', '')),
                           gpu=args.gpu or settings.get('gpu', GPU), workers=1)
    if args.dry_run:
        print(json.dumps(dict(base=base, batch=str(batch), minutes=args.minutes,
                              execution=plans, lanes=list(LANES), proposals_per_lane=args.proposals,
                              starting_candidate=research['candidate'], gate_order=gate_order), indent=2))
        return
    batch.mkdir()
    (batch / 'launcher.sha256').write_text(hashlib.sha256((ROOT / 'try-gan.sh').read_bytes()).hexdigest() + '\n')
    (batch / 'launch-source.py').write_bytes(Path(__file__).read_bytes())
    (batch / 'launch-entrypoint.py').write_bytes(Path(sys.argv[0]).resolve().read_bytes())
    records = []
    for lane, focus in LANES.items():
        plan = plans[lane]
        directory = batch / lane
        directory.mkdir()
        brief = directory / 'brief.md'
        brief.write_text(common + '\nAssigned lane:\n' + focus + '\n\n' + REFERENCES)
        command = [str(ROOT / 'try-gan.sh'), '--engine', plan['engine'], '--repo', str(REPO),
                   '--base', base, '--gpu', plan['gpu'], '--minutes', str(args.minutes),
                   '--candidates', str(args.proposals), '--workers', '1',
                   '--runs-dir', str(directory), '--prompt-file', str(brief), '--focus', lane]
        if plan['model']:
            command += ['--model', plan['model']]
        with (directory / 'launcher.log').open('wb') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        records.append(dict(lane=lane, pid=process.pid, directory=str(directory), command=command,
                            base=base, starting_candidate=research['candidate'],
                            gate_order=gate_order, **plan))
        temp = batch / 'batch.json.tmp'
        temp.write_text(json.dumps(records, indent=2) + '\n')
        temp.replace(batch / 'batch.json')
    pointer = ROOT / 'gan-attempts/current-batch.txt'
    temporary = pointer.with_suffix('.tmp')
    temporary.write_text(str(batch) + '\n')
    temporary.replace(pointer)
    register(batch)
    print(json.dumps(dict(batch=str(batch), pids=[r['pid'] for r in records], base=base), indent=2))


if __name__ == '__main__':
    main()
