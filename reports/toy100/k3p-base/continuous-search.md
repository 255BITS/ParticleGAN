# Continuous experiment: current package default

Current phase: prepare the fresh baseline, then run it after the requested
compaction. Do not launch search agents, candidate training or seed sweeps.
The historical launch STOP markers stay in force.

Start from `particlegan.get_recipe()` and `GANTrainer` on master `0ff9a7af`
(package 0.8.0), merged into this branch. The formula is documented in
[docs/k3p.md](../../../docs/k3p.md). The exact baseline parameters and commands
are in [the handoff](../public-default-baseline/README.md).

The frozen research bundles remain immutable evidence. Their config fields
`reg_arm`, `reg_method`, `gan_mode` and `loss_type` are not public recipe options.
Do not monkey-patch Adam or install the historical mechanism/latent/response
hooks around package optimizers: the recipe optimizers already perform that work.
Use `Recipe(**saved_fields)` when restoring a resolved recipe.

The next baseline is a new measurement, not an inherited 22/22 or hold PASS.
Use current default batch size, particle count, latent dimension, rates, penalties,
anchor, guard and noise settings. Preserve the ring target, host MLP architecture
and evaluation thresholds. The declared run budget is passed as `total_steps`;
noise uses that budget through the trainer, not the old hardcoded 1200 horizon.
Fresh initialization is required because old host fixtures have different shapes.

After the baseline, investigate removing independent tuning choices while
protecting acquisition, hold, precision and recovery. Horizon independence and
hyperparameter reduction are separate claims. Count hardcoded thresholds,
windows, reset timers and smoothing constants as choices too. A release signal
that does not read the clock does not make scheduled rates/noise horizon-free.
Never infer convergence from a quiet optimizer alone.

Use metrics, not images: first-confirmation hold + extension, all stationary and
pre-shift checks, all 81 recovery deadline checks and the matched frozen control.
Log actual role rates, blend, noise, live and EMA quality, and wall time.
Keep archived and new-API scores separate. Full 22-toy and native qualification,
long continuation and repeated shifts remain NOT_RUN until explicitly measured.
Do not run seed-only repeats; mechanism comparisons use the declared seed.

The [previous brief](continuous-search-before-public-default.md) is historical.
Its automatic search/allocation and additional-seed instructions do not apply
to this phase. R2 is an unpromoted historical lead, not the new starting formula.
