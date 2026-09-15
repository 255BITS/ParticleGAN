# Next session: real motion completion

The hybrid toy round is complete; its outcome is in READOUT.md beside this
file. Hybrid/continuous achieved SW1 .0320 vs MLP .0385 and extrapolation
validity91.9% vs76.7%, but increased collisions2.0% vs0.2% and did not recover
missing within-route variation (.36 vs .34; target1). Carry both as candidates. The user intends to compact, then plan real motion completion. No real
motion job, dataset download, or new architecture is queued. Stay on the feature
branch until the user chooses whether to merge/push; this turn requests a commit.

## Proposed next task

Observe a short sequence of a person's 3D joint positions and generate several
possible complete futures. Start with a short horizon and a small consistent
skeleton, with side-by-side animated samples sharing the same observed prefix.
The task is conditional generation from recorded motion, without environment
interaction or reward optimization.

Use the same basic sampler, with x shaped [joint coordinates, future frames]:

```
context = observed motion (+ meaningful action label, if available)
x4 = Gaussian noise shaped like the whole future
clean = G(xt, latent_particle, t, context)
xt-1 = A[t]*clean + B[t]*xt + sqrt(posterior_var[t])*Gaussian_noise
# Four G calls jointly refine the whole future, then play the result.
```

Keep learned latent particles, Gaussian reverse noise, DDGAN posterior,
Rp logistic, existing bcap/VICReg, constant LR and EMA. Temporal U-Net skips
remain a sensible starting G design. Particle IDs are latent codes, not joints.
Physical frames and diffusion timesteps are different axes.

## Decisions for the next planning discussion

1. Choose a dataset after checking accessible files, usage terms, existing splits,
   skeleton representation and meaningful conditioning labels. No dataset has
   been selected, downloaded or inspected in this round.
2. Choose an observed-prefix duration, future duration and frame rate to keep
   pilots fast and human-readable. Preserve global movement as well as pose;
   normalize using training data only.
3. Joint time/class UCD needs meaningful discrete labels. Use supplied action
   labels if the chosen dataset has them; otherwise discuss the UCD conditioning
   choice explicitly before adapting the trainer. Our toy preference classes
   do not carry over automatically.
4. Split source sequences/subjects before making sliding windows, using the
   dataset's official protocol where available. Neighboring overlapping windows
   must not leak across train/test. Report the split and sample exposure.

## Evaluation must change with the domain

The toy had exact route probabilities, a known three-coefficient path family,
and collision geometry. Those diagnostics cannot be transferred literally.
Real motion usually supplies only one recorded future per prefix, so matching
that future alone cannot establish distribution quality or mode coverage.

Start with a held-last-pose and constant-velocity reference, prediction error
against the recorded future, prefix/future boundary continuity, bone-length
consistency, velocity/acceleration statistics, and diversity across generated
futures for the same prefix. Show several samples together with the reference
animation. Evaluate plausibility alongside spread; increasing variance alone
can reward bad motion. Any best-of-K score must report K and accompany single
sample error and diversity. Compare real-vs-real motion statistics where useful.

After a functional baseline, use matched-budget ablations for learned particles
versus Gaussian latent draws, then one-shot versus DDGAN if useful. Do not
interpret sample diversity or visual quality alone as a proven particle gain.

## Working preferences and current artifacts

- Both GPUs available after this round; one training worker per GPU.
- Config-controlled, tail-friendly runs; inspect each batch only on completion.
- 1k viability and 10k comparisons; equal samples seen when changing batch.
- No seed-only experiments. No LR decay tuning. Preserve core formulation.
- Ten trajectory tests pass, including hybrid D gradients and double backward.
- Configs: configs/trajectory/hybrid/{scout_1k,confirm_10k}/*.yaml.
- Live log: results/trajectory/hybrid/live.log.
- Report/gallery: reports/trajectory/hybrid/READOUT.md and confirm_10k/index.html.
- All raw samples, source archives and EMA inference checkpoints remain under
  ignored results/trajectory/hybrid/. Reports preserve configs, metrics,
  provenance, certificates, figures and viewers.
- No-argument trajectory defaults still select discrete geometry / MLP; use an
  explicit continuous config for the newer comparison. CIFAR defaults unchanged.
- Earlier geometry round: reports/trajectory/diversity/READOUT.md. Continuous
  geometry improved generalization, but all models had limitations. Baseline
  extrapolation varied between runs without deterministic GPU enforcement;
  treat single-run results cautiously, without launching seed sweeps.
