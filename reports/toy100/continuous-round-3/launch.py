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
    'k3p_penalty_balance': ('grok', GPU0, '''Own the distinction between adversarial
learning force and regularization force in the critic. Start from selected K3P;
read RP1's image/native rejection and earlier anchor-tracking failures. Existing
controllers mostly observe the TOTAL critic gradient, which may confound target
fit, penalty balance, stochastic noise and anchor stiffness. Measure a meaningful
decomposition using ordinary training batches, then declare a general reversible
controller if the evidence supports it. This is distinct from a raw cosine or
RMS threshold sweep: explain the observed failure your proposal addresses.
Preserve K3P acquisition, sparse prior and response; remove final-horizon inputs.
No task IDs, target centers, quality feedback, known changes or hidden updates.
Account for every extra gradient/forward; one committed optimizer update per role.
Up to3 coherent proposals; BOTH canonical protocols per meaningful proposal.
Ring survivors must clear img_intensity2 and native grid100 full7000 accuracy AND
coverage before deeper qualification. Failed image/native candidates cannot be
new shared bases. If decomposition cannot be installed faithfully, report the
limitation rather than silently substituting an unrelated controller.'''),
    'k3p_data_innovation': ('grok', GPU1, '''Own a distinct training-data innovation
detector for reversible mobility from selected K3P. Previous gradient-only
signals confuse stationary optimizer noise with a changed target. Inspect actual
old failures, then consider online estimates from ordinary REAL minibatches
(with uncertainty/batch-size normalization) to modulate learning rates and
precision. The detector must discover changes itself; no shift times, evaluation
metrics, target centers, task IDs or final budget. Do not directly translate
outputs, align samples to ground-truth centers, or update hidden output transforms
outside the optimizer. Keep learning through ordinary model/particle updates;
the frozen comparator must remain a true no-adaptation control. Preserve K3P
acquisition and explain a horizon-independent cold transition too. No seed or
threshold grids. BOTH own canonical hold+extension and shift per proposal.
Only ring survivors open img_intensity2 and full7000 grid100 accuracy/coverage,
then remaining gates. Do not inherit RP1 scores or adopt it as a shared base.
State limits: real-data innovation alone cannot detect generator-side forgetting;
do not conceal that or infer long-term stability without the declared tests.'''),
    'prepare_continuous_qualification': ('grok', GPU1, '''Harness preparation ONLY;
no new candidates or expensive qualification of rejected RP1. Prepare reusable
evaluation support for the declared stress-protocol.json and 30000-update
long-term-stability-protocol.json, preserving their exact events/windows. Use the
frozen ring host with scoped evaluator instrumentation, never changing its
gradient/update logic or allowing learner access to times/offsets/quality.
One uninterrupted learner state, no checkpoint reload. Preserve original noise,
RNG isolation, optimizer accounting and 9000-prefix scoring. Provide a clean
candidate entry point and fail closed if installation/provenance is incomplete.
Also document a complete pre-shift witness with normalized parameter identities:
model/Adam/EMA, controller, latent.stats, response history/prior membership, RNG.
RP1 captured only part of this; do not falsely label partial hashes complete.
Read completed-codex and completed-curvature-rp1 reports for prior harness errors.
Use focused synthetic or short smoke checks for evaluator event ordering,
absolute versus incremental shifts, exact window counts, RNG preservation and
active update accounting. No full RP1, native or30000 training. Any smoke output
is harness validation, never candidate qualification. Keep code compact, save
exact commands/tests/source hashes, finish once reviewable. Do not rebuild the
whole benchmark or change either declared protocol. One worker, no nested agents.'''),
    'k3p_progress_noise': ('grok', GPU0, '''Own the coupling of acquisition noise
and achieved optimizer motion. K3P remains selected. Read round-2 noise-and-signal,
round-3 signal/noise measured failures, and rp1-rejection.md before proposing.
RP1 replaced horizon noise by absolute120/240 warmups; it passes ring adaptation
but loses short-image stability and grid center accuracy. Investigate a general
noise/regularization transition driven by cumulative normalized training motion
or another declared acquisition statistic, rather than the final duration or a
task-specific clock. Keep reactivation possible after new data; distinguish a
noise warmup from the reversible mobility controller. Existing gap-only/noise-only
rules failed acquisition; do not repeat them or merely vary warmup constants.
Declare extra state/work, preserve sparse-row hooks, no quality feedback. BOTH
own canonical protocols per proposal. A survivor must clear img_intensity2 and
full7000 grid100 coverage AND accuracy before deeper qualification. No borrowed
RP1/P3 gates or new shared base, no coefficient or exploratory seed grid.'''),
    'k3p_transfer_acquisition': ('codex', GPU1, '''Own acquisition/precision transfer
after RP1's measured rejection. Read rp1-rejection.md and completed-transfer's
result.md first. K3P remains selected; RP1 is a diagnostic branch, not a new base.
RP1 passed the ring conjunction but img_intensity2 never closes (full-rate early
penalty for all600 updates) and native grid100 closes early, then misses center
accuracy. Design a general training-signal controller that addresses BOTH rather
than changing fixed quiet counts or tuning a task duration. Inspect the original
K3P acquisition/anchor/noise interaction and actual traces before deciding.
Up to3 coherent proposals, no coefficient grid. Early img_intensity2 screening is
authorized, then BOTH own canonical hold+extension and shift for each meaningful
proposal; record failed screens too. Only a survivor gets original native grid100
screen before deeper qualification. Preserve source and use the proven observer
adapter if needed, checking its assumptions for each new controller. Scarce Codex:
focused reads, no rebuilding harness, no deep audits on rejected candidates.
Do not duplicate RP1 unchanged, and do not inherit any of its scores.'''),
    'k3p_prox_release': ('grok', GPU0, '''Own the distinct reference-gap release
hypothesis. Read completed-reference/attempts/k3p_reference_response/result.md and
rp1-rejection.md. RR2's gap signal opened mobility and reacquired by2500, but slow
reference movement pinned the gap/high rate and failed52/81; RR3 baseline leakage
closed before reacquisition and failed0/81. From selected K3P, declare a general
signal-based precision recovery with no fixed dwell, oracle or budget. Investigate
gap contraction/innovation rather than another reference-decay coefficient grid.
Remove inherited horizon noise for a final candidate. BOTH canonical protocols
for every meaningful proposal. A ring survivor must first clear RP1's known
img_intensity2 and full7000 grid100 coverage AND accuracy failures before opening
expensive remaining qualification. All scores are own, no promotion here.'''),
    'k3p_critic_confidence': ('grok', GPU0, '''Own scale-aware acquisition evidence.
Read rp1-rejection.md: raw critic RMS/old peak never quiets for the image task yet
closes grid100 before accurate centers. Starting from selected K3P, investigate
whether minibatch uncertainty or gradient consistency provides a general closing
signal that separates coherent learning from stochastic oscillation. Declare
normalization across batch/dimension and all extra evaluations; no evaluation
scores, task IDs, target centers, final horizon or known shifts in the learner.
This is distinct from the existing local-curvature and raw signal/noise lanes;
read their measured failures before committing a mechanism. No coefficient grid.
Run BOTH canonical protocols per proposal. Screen img_intensity2 for ring
survivors, then full7000 grid100 coverage AND accuracy before deep qualification.
Use own evidence, not RP1's scores, and preserve learned sparse prior behavior.'''),
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
    'k3p_extragradient': ('codex', GPU1, '''Own a clean shared-sample game
predictor/corrector from selected K3P. Read research-notes.md. Preserve and restore
temporary optimizer/EMA/particle/RNG state, one committed update per role, declare
added work. Compare to failed predictive raw-delta GD1, not an unchanged rerun.
Do not spend the attempt building a new benchmark. If this cannot be made sound,
report the exact implementation limitation and preserve measured evidence.
Codex usage is scarce: concise targeted reads, no repeated full source dumps;
save expensive horizon/adapter/full-toy audits for a candidate that passes the
ring conjunction. Minimal correctness checks for a new update are still needed.
Do not spend the remaining cap proving invariants of already rejected variants.'''),
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
