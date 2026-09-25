# Rare-component failure forensics

**The near miss is real particle-cloud flattening under the learned discriminator field.** The six rare output atoms are sufficient to represent a passing covariance. Resampling, local generator rank, and shared-generator compression do not explain the final failure in this run. This is diagnostic evidence, not a new GAN success.

## Fixed experiment and verification

One declared Softplus5 D96×2/F2 run is replayed twice: first for geometry/derivatives at all24 scheduled observations, then for the actual G/prior updates1001–1200. Original Rp logistic, b_cap3, κ1.25, prior regularization.05, no particle L2, Adam(0,.99), original LRs, G64×2,256particles, batch128,1200steps and seed0 remain unchanged. No heldout task, extra seed, fitted diagnostic, evaluation-informed update or new GAN candidate is introduced.

Both replays exactly match the full archived result after removing timing fields, including all24 live/EMA observations and actions. First replay also asserts unchanged parameter gradients and RNG state after each diagnostic. Runtime: 8.01s + 8.09s. Logs and both parity records are included.

## Which component fails?

Component covariance shape is tested across every component, not only the2% one. At step1050 the failing narrow axis belongs to the13%-mass component (38distinct atoms). At1200 it belongs to the2% component (6distinct atoms,104 evaluation draws). No thresholds are changed.

| Step | Actual rare atoms | Rare exact minimum eigenratio | Rare4096-draw ratio | Global official minimum | Mean component covariance error | All official metrics pass |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 800 | 9 | 0.5248 | 0.4530 | 0.4530 | 1.1030 | no |
| 850 | 9 | 0.8203 | 0.7121 | 0.5140 | 0.9601 | no |
| 900 | 9 | 0.9766 | 0.8418 | 0.5478 | 1.0235 | no |
| 950 | 8 | 0.0526 | 0.0475 | 0.0475 | 0.5077 | no |
| 1000 | 5 | 0.5647 | 0.5006 | 0.1466 | 1.4068 | no |
| 1050 | 6 | 1.2270 | 1.0998 | 0.1223 | 0.7301 | no |
| 1100 | 6 | 0.3859 | 0.3907 | 0.2806 | 0.5251 | yes |
| 1150 | 6 | 0.1678 | 0.1743 | 0.1743 | 0.5743 | yes |
| 1200 | 6 | 0.1334 | 0.1410 | 0.1410 | 0.5651 | no |

The fixed final support gives minimum ratio.13344; the official4096 resampling gives.14102. Both miss≥.15, and resampling slightly improves this snapshot. Counts are stable from1050 onward. At1100 and1150 all official metrics pass; final1200 fails, so the required five-check final suffix never occurs.

| Final component target mass | Actual atoms | HQ atoms | Exact covariance eigenratios | Max squared Mahalanobis distance |
| ---: | ---: | ---: | --- | ---: |
| 55% | 129 | 128 | 0.9296, 1.5198 | 110.12 |
| 30% | 83 | 82 | 0.8229, 1.0057 | 11.38 |
| 13% | 38 | 38 | 0.3076, 1.7023 | 7.88 |
| 2% | 6 | 5 | 0.1334, 2.3350 | 14.32 |

Earlier800–900 rare covariance is inflated by tail points, followed by migration out of the rare component and narrow-axis collapse. The original LeakyReLU baseline had6rare atoms too, but two distant tail points (max squared Mahalanobis126.0) and covariance error16.7; the smoother critic substantially changes that failure. Final Softplus near-miss has mild tail sensitivity: removing one atom can improve or worsen the minor axis, so deleting a tail is not a reliable repair. No point removal is used for scoring.

## Actual update attribution

For each actual Adam step, measure old/new G against old/new latent support. A symmetric two-factor attribution splits the output-variance change exactly; it does not rerun training with a frozen block. Along the final rare minor axis, using the same final six atom identities:

| Updates | Before variance / target variance | After | G-weight contribution | Prior-position contribution |
| --- | ---: | ---: | ---: | ---: |
| 1101–1150 | 0.43319 | 0.19245 | +0.00153 | -0.24227 |
| 1151–1200 | 0.19245 | 0.13344 | +0.01037 | -0.06939 |
| 1101–1200 | 0.43319 | 0.13344 | +0.01191 | -0.31165 |

The prior updates cause the late contraction; shared G updates oppose it slightly. Final local G Jacobians have both singular values nonzero; their smaller singular values range.807–1.106. This rules out a local rank defect, not every possible long-term generator-capacity limitation.

## Discriminator feedback and feature balance

At1100,1150 and1200, the expected Rp-logistic output-ascent field contracts the rare minor axis. At1200 every rare atom has negative discriminator Hessian curvature along this axis (−2.85 to−1.41). Splitting the frozen discriminator input gradient by feature chain rule gives:

| Frozen final feature contribution | Instantaneous rare minor-variance rate |
| --- | ---: |
| Raw coordinates | +.000696 |
| π harmonic | −.006359 |
|2π harmonic | −.001348 |
| Total | −.007010 |

The first periodic harmonic dominates contraction. Damping only the highest harmonic is less directly motivated than balancing the raw path against both periodic harmonics. The corresponding raw latent-gradient variance rate from GAN loss is−4.23e−5; prior regularization contributes+3.43e−7. This snapshot implicates the adversarial field, not the spread regularizer. These force decompositions omit Adam second-moment scaling and sampled minibatch variation; the actual-update attribution above separately includes them.

## Finite-support representability control

A predeclared offline, label-aware witness affinely centers/whitens/recolors only the same six rare OUTPUT atoms to the target covariance. Other250 outputs, weights and4096 evaluation indices remain unchanged. All final metrics then meet their original gates (minimum eigenratio.29463, mean covariance error.36224,HQ.99194). This proves six points can satisfy this final metric set. It is neither a trained generator nor a sustained GAN pass and must never enter the candidate leaderboard. Exact moments are computed manually because the official scorer uses a minimum resampled-count rule; no scoring implementation is altered.

## Ranked architectural hypotheses

1. Balance raw/global discriminator features against periodic features, reducing overall Fourier dominance or adding a smooth raw residual path. This is the most direct response to the measured gradient decomposition; gains must remain generic and the cap must differentiate through the complete encoding in original input units.
2. Give the discriminator an explicit smooth nonperiodic curvature path (for example generic quadratic features or a raw-input residual branch). This may preserve density-shape feedback through periodic ambiguities. It remains a hypothesis, and target-derived centers/scales are unnecessary.
3. More local features or more particles are weaker standalone hypotheses: prior resource experiments and the parallel RBF controls did not give a sustained shared solution. More capacity alone does not guarantee an expanding missing axis. Generator widening has little direct support from this run’s full-rank Jacobians and update attribution.

These are ranked explanations and experiment directions from one deterministic developmental trajectory, not causal proof that an architecture will retrain successfully. Final min-eigen alone is insufficient: retain every component’s covariance, mass, quality and all24 checks.

## Artifacts and reproduction

[Full captured atoms/latents/derivatives](snapshots.json.gz) · [Component geometry](geometry.json.gz) · [Actual updates](updates.json.gz) · [Attribution](update_attribution.json.gz) · [Frozen feature decomposition](spectral.json.gz) · [Geometry witness](geometry_witness.json.gz) · [First parity](parity.json.gz) · [Second parity](update_parity.json.gz) · [Original numerical sources](source.tar.gz) · [All instrumentation](instrumentation_source.tar.gz).

Run replay.py and update_decomposition.py from the matching ParticleGAN source tree with that tree on PYTHONPATH; then analyze.py, controls.py, spectral.py and seal.py. All scripts use /tmp/pr36-rare-forensics as output. Python/torch/runtime fingerprints and numerical source hashes are in plan.json.gz. No repository files were edited.
