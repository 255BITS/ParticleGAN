# Current task: improve the selected direct-particle-response GAN

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

[Previous history](formulation-search-before-direct-particle.md).
