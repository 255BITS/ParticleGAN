#!/usr/bin/env python3
"""Second 3-Codex/5-Grok wave: solve, verify, then promote; K3P stays the base."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(os.environ.get('GAN_WORKSPACE', '/ml2/hypergan')).resolve()
spec = importlib.util.spec_from_file_location('continuous_round_launcher', ROOT / 'launch-gan-formulations.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
launcher.REPO = ROOT / 'ParticleGAN-k3p-continuous-search'
base = subprocess.check_output(['git', '-C', str(launcher.REPO), 'rev-parse', 'HEAD'], text=True).strip()
launcher.COMMON = '\n\n'.join(subprocess.check_output([
    'git', '-C', str(launcher.REPO), 'show', base + ':' + path], text=True)
    for path in ('reports/toy100/k3p-base/continuous-search.md',
                 'reports/toy100/continuous-round-2/SEARCH.md'))
launcher.REFERENCES = """Read-only sources/evidence:
  reports/toy100/current-research-base.json (K3P, still selected)
  reports/toy100/gap-fill-20260925/sources/k3p/
  reports/toy100/gap-fill-20260925/manifest.json
  reports/toy100/continuous-round-1/README.md and attempts/LANE/result.md
  reports/toy100/continuous-round-1/sources/a3_reversal_strength/ (UNVERIFIED candidate)
  /ml2/hypergan/gan-attempts/formulations-20260925T154444Z/
Do not modify old attempts or runtime. Candidate outputs belong to this attempt.
"""
launcher.LANES = {
    'verify_a3_gates': """Own verification ONLY, no new formulation proposals.
The user forbids making unverified A3 the shared next base. Test the UNCHANGED
A3 bundle in reports/toy100/continuous-round-1/sources/a3_reversal_strength/ against
its own 22 frozen toy gates, sensitive four first. This explicitly overrides the
search-lane hold/shift-first order: A3's two protocols already ran; do not repeat
them. Preserve its failed shift verdict. Verify stationary_policy.py actually
installs in transfer AND native drivers. Record applied-rate/noise evidence and
exact hashes. Finish the full22 matrix within the cap even after failures so we
know exactly what it loses. No new mechanisms, seed repeats or baseline runs.
No promotion or further search from A3. Return regression diagnoses and replay
commands for any missing gates. Prior adaptive_anchor report has provenance.""",
    'k3p_reversible_schedule': """Own an adaptive schedule from K3P's verified
acquisition and stable anchored regime. Replace its irreversible LR-clock handover
with ordinary training-state signals. Immediate damping/anchor trapped modes in
round 1; sticky full-rate reopen lost precision. Test a coherent reversible
controller, not unverified A3 as a baseline or an LR grid. Remove horizon noise
for a final continuous candidate; label any scheduled intermediate components.""",
    'k3p_game_damping': """Own game-update damping from selected K3P. Investigate
optimistic/predictive corrections using ordinary gradient history, preserving
acquisition and stabilizing continued adaptive motion. Independent cosine-rate
controllers damped acquisition too early in round 1. Declare schedule/anchor
decoupling and extra compute, no hidden optimizer steps. Do not import A3 wholesale.""",
    'k3p_adversarial_progress': """Own reversible phase detection from ordinary
adversarial progress, not total steps. Preserve K3P acquisition before damping,
while allowing future movement at arbitrary ages. Critic cosine was negative
during healthy mode acquisition and failed as a trigger. Choose a distinct signal
from that evidence, no quality metrics or target oracle. K3P remains your base.""",
    'k3p_anchor_tracking': """Own adaptive critic reference tracking from K3P.
Faster EMA memory alone in INNOV3 held but recovered 0/81; from-step-zero anchoring
trapped modes. Investigate reference/restoring-force behavior that preserves
acquisition and permits adaptation, with an explicit horizon-independent schedule.
No decay grid or timed resets. A3 is diagnostic evidence, not your new base.""",
    'k3p_balanced_updates': """Own coupled generator/prior/critic mobility from
verified K3P. Avoid independent role controllers and sticky full-rate boosts
that failed round 1. Test a bounded or normalized joint update from training
signals. Preserve particle hooks; declare exceptions. Separate critic mixing from
an LR clock, audit actual rates. Do not replace the baseline wholesale with A3.""",
    'k3p_precision_recovery': """Own reversible precision control from K3P.
SN3 recovered 80/81 but failed hold; A3 held but recovered 71/81 and failed pre-hold.
Neither is verified. Use these failure modes to propose a distinct step/damping
rule retaining K3P acquisition and precision while adapting without a final
horizon. No coefficient grid or later-window selection. Whole shift must pass.""",
    'k3p_noise_and_signal': """Own removal of horizon-dependent noise from K3P
in a reversible training-state policy. Full-rate stationary noise lost hold;
constant LR alone disables K3P's anchor. Explicitly preserve useful acquisition
and anchor behavior while decoupling the clock. Ordinary signals and measured
failures, no amplitude grid, task identity, centers or A3 as the shared base.""",
}
gpus = ['GPU-72c1b506-891d-b8bc-b353-e020585e1c47',
        'GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69']
launcher.LANE_SETTINGS = {
    lane: dict(engine='codex' if i < 3 else 'grok',
               model='gpt-6-astra' if i < 3 else 'grok-4.7', gpu=gpus[i % 2])
    for i, lane in enumerate(launcher.LANES)
}
if __name__ == '__main__':
    for flag, value in [('--minutes', '45'), ('--proposals', '3')]:
        if not any(arg == flag or arg.startswith(flag + '=') for arg in sys.argv[1:]):
            sys.argv.extend([flag, value])
    launcher.main()
