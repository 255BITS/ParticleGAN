# Full-cloud prior regularization: matched rotated100 probe

The [scratch probe](accuracy_full_prior_probe.py) repeated the archived
[affine-identity, square-prior rotated100 run](accuracy-failure/rotated100)
with its exact saved 7,000-step configuration, seed 1234, square prior scale 5,
20,000 particles, batch 2,048, Fourier-3 discriminator, optimizer, losses, and
noise schedules. The **only training change** was evaluating the same
`ParticleRegularizer` on all 20,000 `prior.z` rows on every generator update.
The existing `GANTrainer` evaluates it on only the unique sampled rows when
the prior exceeds 1,024 particles; all native transfer hosts have 32 or 256
particles and already use the full cloud. This is an isolated proposed rule,
not evidence for the unchanged common-22 recipe.

The local run directory `artifacts/toy100-accuracy/affine-full-prior` holds
the copied probe source, byte-identical archived control config, resolved
config, provenance hashes, 29 read-only prior diagnostics, complete events,
the five final scored sample archives, and a separate 100,000-draw holdout.
The probe's initial 4,096-sample live cloud matches the archived control
**bit-for-bit**. The focused [parity tests](../../tests/test_toy100_full_prior_probe.py)
show bitwise-identical one-update weights, losses, and RNG state for a small
prior, and a nonzero gradient to unsampled rows with no RNG consumption for a
large prior. Two tests passed.

| Update | Archived live modes | Full-cloud live modes | Archived HQ | Full-cloud HQ | Full-cloud mass TV |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 750 | 100 | 100 | .8792 | .8681 | .04345 |
| 1,000 | 100 | 100 | .9180 | .9069 | .04215 |
| 1,250 | 100 | 0 | .8672 | .0187 | .16005 |
| 1,750 | 100 | 0 | .8770 | .0227 | .17095 |
| 2,000 | 0 | 0 | .0193 | .0282 | .17190 |
| 6,000 | 3 | 4 | .0718 | .0598 | .18960 |
| 7,000 | 5 | 6 | .0874 | .0814 | .18845 |

The full-cloud probe **fails both frozen gates**. All five final live checks
have only 4–6 modes, and the independent holdout has precision .0815 and
mass TV .18745. Conditional center and shape fidelity are undefined because
too few modes qualify. The archived control also fails both gates, but its
100-mode phase lasts through step 1,750; the full-cloud rule loses it by
step 1,250. A lower mass TV on a cloud with about 8% precision is not a
distributional pass. Separate-process regrading reproduced both FAIL verdicts.

The diagnostics confirm the intended mechanism changed. Across the 28
recorded updates, the mean absolute off-diagonal covariance of the selected
~1,950 rows was .1430, while that of all 20,000 rows was .00543. The same
unweighted regularizer evaluated on those selected rows averaged .03284;
the actual full-cloud value averaged .0000561. At step 1,000 the affine
generator's singular values were 1.013 and 1.004, and the fixed 1,024
particle identities had moved .458 RMS from initialization. At the collapse
check, step 1,250, the singular values were .901 and .886 and those particles
had moved .528 RMS. These readouts show that sampled-covariance noise was
present and removed; they do **not** attribute the ensuing generator change
to the prior rather than the discriminator or adversarial feedback.

Full-cloud regularization alone is rejected as a stability repair for this
matched case. It leaves the known 16/19 transfer bottlenecks unchanged and
cannot support a 22/22 claim.
