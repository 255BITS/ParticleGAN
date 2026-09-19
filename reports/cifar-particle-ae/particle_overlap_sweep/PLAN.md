# Particle overlap investigation

User authorized exploration using GPU 1. Work stays on the feature branch; GPU 0 is not used by this investigation.

Hypothesis: particle centers, particularly cloned siblings, reconverge during training. Increased component overlap might reduce independent control of generated samples. It might instead be a symptom of generator/discriminator coadaptation. A 2.67% nearest-center error is not evidence that the whole mixture has collapsed.

## Frozen sampling diagnostic

80k best checkpoint (FID 15.7527) and deteriorated 160k checkpoint (19.0584), each evaluated with noise multipliers 1, 0.75, 0.5. Same particle/noise draws across scales, 50k historical FID protocol. Baseline FID must reproduce within 0.01. Independent 10k real/fake k5 density/coverage exactly matches historical information-probe generation precision and RNG. Real feature SHA fe089b56afca1897c928bcb682e5e0cc380ab23c4fce0492521122d87a62da46. Equal-prior posterior and nearest-center classification over all 16,384 components, 32,768 draws. Split confusion into same-parent and different-parent errors.

Sampling-only improvements must not be described as training gains. Lower FID with falling coverage may be a diversity tradeoff. Failure to rescue an already damaged checkpoint does not rule out harmful overlap during training.

## Training intervention

Full-state forks from 80k to 100k, FID50k and geometry every 5k. Existing unchanged continuation is the control: 90k 15.7901, 100k 16.4609. Rates, seed, optimizer state, architecture, and E-only reconstruction stay unchanged.

1. Freeze live and EMA particle centers separately at their restored values. Skip prior optimizer and EMA-center updates; continue G/D/E. This avoids an initial evaluation distribution jump but preserves the small original live/EMA difference. Freezing tests center movement broadly, not overlap alone.
2. Reduce fixed sigma, chosen after reviewing the sweep, while continuing center and G/D/E training. This changes both the training and evaluation prior; report its initial sampling-only effect separately.

Before production, run actual 16-update checkpoint smokes verifying live and EMA center freezing, unchanged prior Adam moments in the freeze arm, sigma factors, and G/D/E updates. New standalone trainer preserves historical source certificates. Endpoint FID reproduction, density/coverage, and geometry are queued through the experiment pipeline. No automatic continuation beyond 100k.

## Logs

`tail -F runs/cifar_particle_ae/particle_overlap_sweep/PIPELINE.log`

Training after setup: `tail -F runs/cifar_particle_ae/particle_overlap_training/PIPELINE.log`

## Geometry refinement

Center variance within original clone families actually rises from 6.71% to 8.78% of total center variance between 80k and 160k. Within-family pair distance quantiles (10/50/90%) change from 1.911/2.573/4.281 to 1.030/1.344/6.527. This is local clumping with a broader upper tail, not uniform contraction of every family. 4.65% of centers at 160k have a sibling within four sigmas, versus zero at 80k. All 875 observed nearest-center classification errors at 160k are within families; nearest neighbors are siblings for 98.70% of particles. The global variance/covariance regularizer does not explicitly penalize nearby pairs. See center_variance.json.
