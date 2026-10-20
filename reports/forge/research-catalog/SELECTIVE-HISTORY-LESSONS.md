# Selective gradient history: lessons from the completed screen

The custom transport/history experiment produced a negative qualification result and useful evidence about optimizer composition. **None of the seven modified recipes passed both diagnostic tasks.** Across eight recipes, all 16 pole/Ring attempts completed their original schedules under the prescribed deterministic seed-0 initializer. The unchanged baseline B alone passed both gates. These results concern the measured tasks and recipes; they establish no reliability claim across seeds, architectures or longer horizons.

The [complete report](https://github.com/255BITS/ParticleGAN/blob/e84c2105de1ac0427cd43127fa1259a9de74916c/reports/forge/selective-history/TRAINING.md), [initial curves](https://github.com/255BITS/ParticleGAN/blob/e84c2105de1ac0427cd43127fa1259a9de74916c/reports/forge/selective-history/training-media/training-curves.png) and [repair curves](https://github.com/255BITS/ParticleGAN/blob/e84c2105de1ac0427cd43127fa1259a9de74916c/reports/forge/selective-history/training-media/repair-training-curves.png) retain the original gates and independently audited outcomes.

## History placement changes the intervention

B uses BCAP, zero-momentum DualNorm with smoothing .001, and direction blend. T adds global/local transport. W adds whole-generator gradient history to T; S instead applies history only to the adversarial generator channel. The recovered estimator uses decay 7/8 and estimates temporal signal relative to variance. It supplies a scalar rho between zero and one; it does not replace the current gradient with an averaged gradient.

The recovered historical caller scaled learning rates **after normalization**. Our W adaptation multiplies the gradient **before normalization**. These are different algorithms. Ideally, a homogeneous normalizer removes positive scalar scaling: `normalize(rho*g) = normalize(g)`. On pole, W performed all 80 history observations and ended at rho .711722, yet its recorded trajectory differed from T by at most 1.32e-6. The intervention ran but barely changed that motion.

The implemented smoothed polar update sharpens this explanation. For whole-gradient W, a singular value `s` receives weight

```text
rho*s / sqrt((rho*s)^2 + lambda^2)
  = s / sqrt(s^2 + (lambda/rho)^2),  rho > 0.
```

With `lambda=.001`, pre-normalization scaling raises the effective smoothing scale rather than uniformly shrinking every direction. Strong singular directions can remain almost unchanged while weaker ones are suppressed. At rho zero the gradient is removed. This is a **code-derived interpretation**, subject to numerical cutoffs; saved observations do not identify its per-step spectral contribution to Ring failure. Post-normalization history was not tested here.

## Selective history changes the mixture of forces

Writing total gradient as `g = a + t`, where `a` is adversarial and `t` contains the other channels, S supplies `g + (rho-1)*a = rho*a + t`. Transport therefore remains fresh when the adversarial channel is zeroed. This can change direction through normalization, but also changes the balance between complementary objectives.

On Ring, W recorded 1,122 zero-rho steps out of 1,600 (70.125%). The network generator's total gradient was zero on those steps while the learned prior retained its ordinary update clock. This is a gradient/clock observation, not a measurement of all realized parameter or output displacements. S recorded 116 zero-rho steps and eleven direction blends. Suppression of useful adversarial shape correction, or a changed balance with prior motion, is plausible; no per-update force/displacement trace establishes the causal sequence.

## Allocating mass does not ensure correct shape

Both B and T finished with sixteen modes, but transport traded improved allocation for weaker local geometry and sustained success:

| Final Ring measure | B | T |
|---|---:|---:|
| Mass total variation, lower is better | .065186 | .035156 |
| Full component covariance error, lower is better | .496567 | .673336 |
| High-quality fraction | .960938 | .912598 |
| Final consecutive jointly passing checks | 26 | 6 |

All eight recipes finished with sixteen modes and low mass TV. Those measures did not prevent covariance or quality failure. W's core covariance error .618462 looked acceptable while its full covariance error was 1.941778. Trimming to the core would hide material failure.

## Coherence and scalar repairs were insufficient

S ended pole at rho .997980 with almost the same trajectory as T, and both retained eleven critic-slope violations. Endpoint coherence can coexist with failure of the declared constraint. It does not tell us rho at each earlier violation.

Two bounded repair rounds doubled the soft penalty, halved generator/prior rates, or halved critic rates in selected parents. Doubling the penalty reduced pole violations from eleven to three but missed the required terminal suffix and lost Ring. Rate reductions also failed the joint gates. This weakens a simple finite-step overshoot explanation without proving that every step size, history placement or transport design must fail. Selective history without transport was not tested.

The unresolved diagnostic question is whether lost adversarial influence precedes local-shape deterioration or mainly responds to it. Answering that requires aligned rho, channel-gradient, realized-displacement and quality traces; these results alone do not answer it or authorize another training campaign.

The separate fresh ordinary baseline campaign passed five of six required tasks plus the diagnostic clock audit; word timed out before its full contract, so qualification remains incomplete. The [machine-readable results](https://github.com/255BITS/ParticleGAN/blob/e84c2105de1ac0427cd43127fa1259a9de74916c/reports/forge/selective-history/training-summary.json) preserve that scope. This effort records 23 attempts across four campaigns, outside the frozen 285-entry register. This documentation note changes no code, task gate or frozen register entry.

Original source/runtime bindings, recipe identities, attempt IDs and receipt hashes remain in the linked report and summary. Absolute `/ml2` paths identify local artifacts; the original tensor files are not public downloads.
