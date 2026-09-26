#!/usr/bin/env python3
"""Bounded continuous-learning search from the selected K3P; --dry-run launches nothing."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(os.environ.get('GAN_WORKSPACE', '/ml2/hypergan')).resolve()
spec = importlib.util.spec_from_file_location('k3p_search_launcher', ROOT / 'launch-gan-formulations.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
launcher.REPO = ROOT / 'ParticleGAN-selected-h-stability-base'
base = subprocess.check_output(['git', '-C', str(launcher.REPO), 'rev-parse', 'HEAD'], text=True).strip()
research = json.loads(subprocess.check_output([
    'git', '-C', str(launcher.REPO), 'show', base + ':reports/toy100/current-research-base.json'], text=True))
if research['candidate'] != 'k3p':
    raise SystemExit('This search requires the selected K3P source bundle.')
launcher.COMMON = subprocess.check_output([
    'git', '-C', str(launcher.REPO), 'show', base + ':reports/toy100/k3p-base/continuous-search.md'], text=True)
launcher.GATE_ORDER = None  # The complete 22-task order comes from the selected declaration.
launcher.REFERENCES = '''Read the committed selected-formulation report and exact sources:
  reports/toy100/k3p-base/README.md
  reports/toy100/gap-fill-20260925/sources/k3p/
  reports/toy100/gap-fill-20260925/manifest.json (frozen runtime, fixtures, commands)
  reports/toy100/continuous-practical-leaderboard.md
  reports/toy100/overnight-20260925/evidence.json (parent hold and recovery)
Old launch-gan-directions/plasticity briefs describe earlier formulations and are
not this round's instructions. Do not inherit their seed sweeps or old conclusions.
'''
launcher.LANES = {
    'continuous_critic': '''Own a critic constraint that works during both acquisition
and continued adaptation without the LR-triggered early/late switch. Explicitly
keep the EMA anchor active under stationary positive rates. Begin with a simple
declared time-independent anchored penalty, then adapt from measured cold/hold/
recovery failures. Preserve anchor decay, optimizer/history rules and guard for
attribution. No blend-coefficient grid; test distinct evidence-driven mechanisms.
Any retained horizon-dependent noise is an intermediate diagnostic, not the final
continuous-policy claim. Report actual mixing and anchor activity.''',
    'reversible_plasticity': '''Own reversible step-size/update control from ordinary
gradient or optimizer signals. The controller must be able to regain mobility
after changes at arbitrary training ages, without an end-of-run schedule, fixed
restart cycle or access to shift/convergence times. Decouple K3P's critic anchor
from the LR ratio before interpreting rate changes, and hold that declared critic
choice fixed within a comparison. Preserve the particle mechanisms. Rank retained
precision and timely recovery together; explicitly report applied-rate traces.''',
    'adaptive_anchor': '''Own whether the EMA critic's memory resists adaptation or
can supply continuous damping without a one-way phase. At declared stationary
positive rates with an active anchor, inspect anchor disagreement and ordinary
gradient coherence, then test a small generally applicable anchor-memory or
restoring-strength rule. Preserve optimizer/particle rules for attribution. No
test-time resets, data-shift oracle, time-scheduled release or EMA-decay grid.
Require cold acquisition, prolonged hold and recovery with the same rule.''',
    'relative_update_control': '''Own bounded, reversible update-size control.
Use parameter displacement, optimizer moments or ordinary gradient innovation
to maintain useful positive updates without a horizon or one-way decay. Keep
the declared stationary critic anchor active. Compare against the parent's
measured stability/recovery tradeoff; do not scan scalar learning rates. Preserve
the existing particle rule unless a precise update-control mechanism requires
a declared change. Report actual relative update magnitudes and rate traces.''',
    'reversible_critic_mixing': '''Own replacing the LR-clock critic handover with
a reversible training-signal controller. Investigate gradient agreement or
normalized critic/anchor disagreement, with bounded hysteresis or smooth memory
if needed. It must reacquire damping and responsiveness at arbitrary ages.
Keep anchor memory and particle rules fixed for attribution. No scheduled phase,
known target change or evaluation feedback. Log mixing and active anchor work.''',
    'anchor_innovation': '''Own an innovation-based critic memory mechanism.
Investigate whether a slowly moving reference can remain precise yet track new
data using ordinary critic-gradient innovation. Distinguish changing the EMA
reference from changing its penalty strength, and declare one coherent mechanism
per proposal. No decay grid or scripted resets. Preserve optimizer/particle
mechanisms and explicitly activate the anchor independently of LR decay.''',
    'particle_network_balance': '''Own reversible balance between particle,
generator and critic update timescales. Use existing optimizer/gradient signals
to prevent one subsystem becoming immobile while others move. Preserve A2 sparse
latent damping and direct response as the starting point; declare any change.
Keep an active clock-independent critic anchor. No extra hidden optimizer steps,
task-specific rules, target statistics or horizon-dependent control.''',
    'stationary_noise': '''Own the horizon-dependent noise component and its
interaction with continuous precision and responsiveness. Declare a stationary
or training-signal-adaptive noise rule together with an explicitly active,
clock-independent critic anchor and positive horizon-free optimizer policy.
Do not quietly retain annealing tied to total_steps. Use measured hold/recovery
failures to refine mechanisms, not a noise-amplitude grid. Preserve host losses
and the frozen data/score protocol; report which noise sources remain active.''',
}
GPU0 = 'GPU-72c1b506-891d-b8bc-b353-e020585e1c47'
GPU1 = 'GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69'
launcher.LANE_SETTINGS = {
    lane: dict(engine='codex' if index < 3 else 'grok',
               model='gpt-6-astra' if index < 3 else 'grok-4.7',
               gpu=GPU0 if index % 2 == 0 else GPU1)
    for index, lane in enumerate(launcher.LANES)
}

if __name__ == '__main__':
    # Explicit CLI budgets override these bounded first-round defaults.
    for flag, value in [('--minutes', '45'), ('--proposals', '3')]:
        if not any(arg == flag or arg.startswith(flag + '=') for arg in sys.argv[1:]):
            sys.argv.extend([flag, value])
    launcher.main()
