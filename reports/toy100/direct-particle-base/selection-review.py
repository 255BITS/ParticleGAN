from pathlib import Path
import hashlib, json

root = Path('/ml2/hypergan/ParticleGAN-epsilon-gan-followup')
bundle = root / 'reports/toy100/direct-particle-base'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
audit = json.loads((bundle / 'audit.json').read_text())
report = '''# Selected research base: direct_particle_response

The user selected this formulation as the default research/launcher base.
It remains an Rp logistic GAN with the parent dimension-RMS critic penalty.
Public training-package defaults are unchanged; release qualification is pending.

**15 PASS / 1 FAIL / 6 NOT_RUN across the 22 GPU toys.** All seven initial
gates pass. Unequal-width covariance is the remaining measured failure:
component covariance error .981321 exceeds .85, with zero terminal passing
checks. Its first component contributes error 2.72220; mass, HQ, SW1 and the
minimum-eigenvalue bounds pass. There were 11 passing observations out of 24,
so an earlier passing check would have hidden the failure.

The critic retains the parent's penalty:

`lambda/2 * (mean(||grad D(real)||² / d) + mean(relu(||grad D(fake)|| / sqrt(d) - kappa)²))`

Here d counts input elements per example; lambda=kappa=1. This is normalized
real R1 and fake RMS b-cap, with the original schedules and auxiliary host terms.
The new response applies only to direct sample-particle optimizer groups,
excluding registered ParticlePrior latent parameters and network parameters.
For their ordinary Adam update it uses betas (0,.9) and multiplies the scheduled
learning rate by `1 + clamp(cos(center(g_t), center(g_previous)), 0, 1)`.
The first gain is 1; all gains are between 1 and 2. Centering measures alignment;
the raw gradient is unchanged. The scheduled LR and betas are restored afterward.
No target statistics, mode identities, evaluation feedback, extra model calls
or extra optimizer steps enter this mechanism. The two optimizer changes have
not been causally isolated.

This response is active on two_pole among the sixteen measured hosts. Movement
improves from .104395 to .644163, with ten terminal passing checks. The six
previous regression hosts reproduce the parent's non-timing results in fresh
executions. Unequal width uses the unchanged latent path; boosting the direct
response further cannot fix that host.

**Use config.json + mechanism.py + response.py through the exact probe.py.**
The legacy config field `reg_arm: a_r1r2` selects an installed patch; config alone
runs a different formulation. All four hashes are pinned in the declaration.
Model training, gradients, response history and all Adam state tensors are CUDA
FP32. Retained CPU initialization performs zero optimizer updates. Keep the
PyTorch2.13.0+cu126 deterministic, TF32-off, one-thread environment and original
noncapturable Adam arithmetic.

| Measured toy | Verdict | Terminal passing checks |
|---|---|---:|
'''
for row in audit['rows']:
    report += f"| {row['gate']} | {row['status']} | {row['passing_suffix']} |\n"
report += '''
NOT_RUN: vector_anisotropic, vector_overlap, vector_spiral, grid100, rotated100,
staggered100. No full22 qualification or own-state post-convergence result exists.
The module-global response history must be saved/restored with model, optimizer
and RNG state before a checkpoint-resume stability claim can be made.

[Independent audit and metrics](audit.json) · [All sixteen raw records](results/)
· [Exact declaration](original-declaration.json) · [Previous round](round-summary.json)
· [Original detailed report](original-attempt-report.md)

The completed round tested nine proposals and 34 GPU training gates: 25 PASS,
9 FAIL. Real-gradient dead-zone and real-penalty warmup candidates failed either
movement or ring. Applying coherent response to latent priors too lost ring.
Do not rerun those unchanged proposals. The unipolar receipt auditor initially
assumed one regularizer call per step; its frozen host uses two scales. This
was corrected without training, and the original audit error is retained.

The bundle includes all sixteen zero-update initialization fixtures, exact code,
raw curves, CUDA receipts, immutable source manifest and original command logs.
Original attempt helpers are retained for provenance; replay.py is the portable
entry point and reconstructs the archived baseline sources without changing them:

```bash
/tmp/pr38-default-env/bin/python reports/toy100/direct-particle-base/replay.py \\
  --task two_pole --gpu 1 --workdir /tmp/direct-particle-replay-new
```

The helper rejects existing work directories and verifies code, source and
fixture hashes. It compares every non-timing metric, verdict, RNG digest,
initial parameter, response/mobility receipt and CUDA update count against the
recorded gate. Promotion checks are in replay-checks.json. Reproducing the
unequal-width FAIL validates replay; it does not qualify the candidate.

Next: unequal width first, then protect all fifteen passes. Only a candidate
clearing those sixteen continues to its own six remaining GPU toys, then
own-state retention under its declared schedule. No inherited passes or seed
sweeps. Historical native GPU16/22 and CPU22/22 results belong to other recipes.
'''
(bundle / 'README.md').write_text(report)

search = '''# Current task: improve the selected direct-particle-response GAN

Read AGENTS.md, reports/toy100/current-research-base.json and
reports/toy100/direct-particle-base/README.md. The user selected
direct_particle_response as the default research base. Start from its exact
config.json + mechanism.py + response.py + probe.py. Config alone is NOT this
formulation. Use the checksum-verified archived source preparer and retained
CPU initialization fixtures. Do not revert to an older recipe or harness.

The parent has 15 PASS / 1 FAIL / 6 NOT_RUN on GPU. Two_pole now passes movement
.64416 with ten terminal checks, and all six original GPU regressions still
pass. The first measured failure is vector_unequal_width: component covariance
error .98132 > .85, zero terminal passing suffix; the first component's error
2.72220 dominates. HQ, mass, SW1 and minimum eigen ratio pass. Direct response
is inactive on this host, so strengthening that boost alone cannot fix it.

Keep the successful direct-particle response intact unless its change is an
explicit, evidence-driven proposal. The baseline is Rp logistic GAN, normalized
real R1 + fake RMS b-cap, with original schedules and auxiliary host losses.
No target fitting, task-name switches, mode labels/centers/statistics, or metric
feedback in training. Architecture, data, fixed seeds, evaluation, thresholds
and training-step budgets stay frozen. Original LR/noise schedules are starting
defaults; the assigned lane may explicitly declare a formulation change.
All training, gradients and optimizer/history state stay CUDA. Preserve original
noncapturable Adam arithmetic; CPU initialization is pinned and has zero updates.

Gate order: vector_unequal_width FIRST, then two_pole, trajectory, mode_hold,
vector_unequal_mass, vector_two_broad, img_intensity2, img_bars4, img_blobs4,
residual_student, unipolar, ae_gan_hold, cover_leftover, unused_token_hold,
mid_scale_identity, img_stripes2. Stop a failing proposal and adapt within the
lane; do not spend on the rest after a measured failure. Every candidate must
earn its own passes. If all sixteen pass, keep testing the six unmeasured toys:
vector_anisotropic, vector_overlap, vector_spiral, grid100, rotated100,
staggered100. Use canonical native coverage AND accuracy for the last three.
Do not stop a promising candidate just because it passes the early blockers.

After all22 pass, run own-state post-convergence continuation under the declared
schedule. Preserve model, optimizer, RNG AND response history across checkpoint
resume: response.py's module-global previous-gradient map is not automatically
part of Adam state_dict. No continuation has been validated yet. Prioritize
quality after convergence; acquisition and retention are separate claims.

Three fresh Astra/max attempts, one GPU worker each, maximum three proposals and
45minutes per attempt. Caps, not quotas. Start real training within five minutes.
No nested agents, detached training, seed sweeps, coefficient grids, extra
undeclared updates, pushes or comments. A generally applicable measured change
beats theory or new benchmark infrastructure. Do not repeat unchanged failures,
version upgrades or completed controls. In particular, applying recent-memory
coherent response to all latent priors already loses ring. The original small
real dead zones, sextic release and real-penalty warmups also failed this round.
R1-containing variants remain eligible when measured tests pass.

Prepare source once. Snapshot/hash candidate code and declare the formula before
each execution. Use the exact probe as a template; preserve frozen hosts, native
random draws, CUDA-state proof and all metric observations. Rebuild expected
specs from the candidate recipe; image penalty/kappa fields are aliases. Audit
counts from actual frozen hosts: unipolar uses two regularizer calls per step,
mid_scale_identity four. Receipt repairs are not new training. New CPU fixtures
may only capture initial constructors with zero updates and no CPU autograd.
Use focused mechanism checks, append tests.jsonl, preserve every FAIL/ERROR and
explicit NOT_RUN, keep tail-able logs, and finish with a measured leaderboard
and exact replay commands. Avoid dumping full source manifests or raw curves.
'''
(root / 'reports/toy100/h_stability/SEARCH.md').write_text(search)
(root / 'reports/toy100/h_stability/START.md').write_text('''# Active research base: direct_particle_response

Use [the complete selected formulation](../direct-particle-base/README.md): config,
critic mechanism, response and probe. It has 15 PASS, 1 FAIL, 6 NOT_RUN on GPU.
Gate on unequal-width covariance first. Full22 and own-state stability remain
unqualified; public package defaults are unchanged.

[Declaration](../current-research-base.json) · [Search brief](SEARCH.md)

Older dimension-RMS, CPU, native GPU16/22 and H/epsilon bases are historical.
Their scores cannot be added to this candidate's measurements.
''')
p = root / 'reports/toy100/current-research-base.json'
old = json.loads(p.read_text())
new = dict(candidate='direct_particle_response', parent='dimension_rms_hybrid',
    role='user-selected GPU GAN research and launcher default', requires_mechanism=True,
    requires_response=True, report='reports/toy100/direct-particle-base/README.md',
    measured_gpu_passes=15, measured_gpu_failures=1, measured_gpu_total=16, toy_total=22,
    unmeasured_toys=6, full_toy_status='INCOMPLETE', own_state_stability='NOT_RUN',
    gpu_release_qualified=False, initialization='retained CPU initial parameters; native CUDA random draws and training',
    first_gates=['vector_unequal_width', 'two_pole', 'trajectory', 'mode_hold', 'vector_unequal_mass'],
    remaining_regression_gates=['vector_two_broad', 'img_intensity2', 'img_bars4', 'img_blobs4',
        'residual_student', 'unipolar', 'ae_gan_hold', 'cover_leftover', 'unused_token_hold',
        'mid_scale_identity', 'img_stripes2'],
    remaining_unmeasured_gates=['vector_anisotropic', 'vector_overlap', 'vector_spiral', 'grid100', 'rotated100', 'staggered100'],
    known_failures=['vector_unequal_width'], schedules='retain original decay/noise as starting defaults',
    auxiliary_host_terms='retain original AE and unused-token terms', seed_sweeps=False,
    stability_priority='quality after convergence', historical_reference=old['historical_reference'])
for key, filename in [('config', 'config.json'), ('mechanism', 'mechanism.py'), ('response', 'response.py'), ('probe', 'probe.py')]:
    new[key] = 'reports/toy100/direct-particle-base/' + filename
    new[key + '_sha256'] = sha(bundle / filename)
p.write_text(json.dumps(new, indent=2) + '\n')
p = root / 'reports/toy100/formulation-search.md'
(root / 'reports/toy100/formulation-search-before-direct-particle.md').write_text(p.read_text())
p.write_text(search + '\n[Previous history](formulation-search-before-direct-particle.md).\n')
p = root / 'reports/toy100/continuous-practical-leaderboard.md'
s = p.read_text()
start = s.index('**Selected research base:')
end = s.index('**Yes, there is a recorded 22/22 PASS.**')
s = s[:start] + '''**Selected research base: [direct_particle_response](direct-particle-base/README.md).**
**15 PASS / 1 FAIL / 6 NOT_RUN** on its own GPU suite. Unequal-width covariance
fails (.981321 > .85, zero terminal passing checks). All seven initial gates
and eight further toys pass. Full22 and own-state retention are unqualified.

| Current research candidate | GPU measured | GPU unmeasured | Blocker | Own-state hold |
|---|---|---:|---|---|
| direct_particle_response (selected) | **15 PASS / 1 FAIL** | 6 | unequal width | NOT_RUN |
| dimension_rms_hybrid (parent) | 6 PASS / 1 FAIL | 15 | two_pole | NOT_RUN |

[Sixteen raw results and independent audit](direct-particle-base/audit.json) ·
[Latest completed round: 9 proposals, 34 gates, 25 PASS / 9 FAIL](direct-particle-base/round-summary.json).
These are partial candidate results, not directly ranked full22 scores. Do not
combine them with the historical native GPU16/22 control below.

''' + s[end:]
s = s.replace('The selected dimension-RMS formulation above now replaces it as the research base.',
              'The selected direct-particle formulation above now replaces it as the research base.')
p.write_text(s)
Path('/ml2/hypergan/selected-h-stability-brief.md').write_text(search)
print('Selected direct_particle_response; metadata, report and search briefs updated.')
