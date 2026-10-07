# Ring16 tiny weak-subspace noise: CUDA results

**The fixed tiny-noise rule applied every update passes confirmed smoke and ends
with 45 consecutive full passes. Noise applied only at update 401 ends with six
full passes, but its single first-pass confirmation fails covariance.** Both
fresh-live 1,600-update arms complete on an RTX A6000 without a checkpoint reload,
learning-rate annealing, prior changes or seed variation.

The [frozen protocol](protocol.json), [compact results](results.json),
[execution verification](execution-verification.json) and [archive receipt](archive.json)
bind the outcomes to actual sources, streams and sampling law. The earlier
CUDA-blocked [preparation verification](verification.json) remains unchanged
under its original identity; CUDA became accessible on 2026-10-07.

## Results and declared gates

Every 4,096-output draw must satisfy all bounds: 16 modes, mass TV ≤ .15,
HQ ≥ .85, mean component covariance error ≤ .85 and minimum component eigen
ratio ≥ .15. Confirmed smoke requires a full scheduled pass plus the single
independent confirmation at those same weights. The five-terminal-pass verdict
is separate; completing these diagnostics supplies no tier 2 hold.

| Schedule | First full pass | Confirmation | Terminal suffix | Terminal verdict | Final covariance | Final HQ | Final eigen ratio |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: |
| `boundary_only` (401 only) | 1400 | FAIL: covariance .958379 | 6 | PASS | .481363 | .926270 | .315379 |
| `every_step` (1–1600) | 817 | PASS: covariance .754245 | 45 | PASS | .504405 | .965576 | .332743 |

The every-step confirmation also has 16 modes, mass TV .087891, HQ .922119 and
minimum eigen ratio .501905. It is drawn at unchanged update-817 weights from
the isolated named confirmation stream. There is one confirmation opportunity
per arm; the boundary failure is retained without another draw. Training
continues through the complete declared budget after either confirmation.

Every-step noise has one later marginal covariance failure at update 850
(`.851437 > .85`), followed by 45 full passes from 867 through 1600. It passes
47 of 96 scheduled checks overall. Boundary-only noise passes 10 of 96 and
has covariance failures at 1417, 1450 and 1500 after its first passing check.
Neither result establishes that every observation after acquisition passes.

The boundary arm trains an unchanged live prefix through 400 and verifies
its entire state against PR331's saved prefix. After removing only the two
newly registered, unused diagnostic streams, the full state digest is exactly
`208e2d7b241ffeac11915972dbb90979dd686ad5285768282e2182d7da5ad2cd`.
It continues the same objects, never loading the archive into the learner.
The perturbation changes six matrix directions at 401. Every-step noise changes
9,600 matrix directions and intentionally changes the prefix from initialization.
Both arms consume the same target batch sequence, digest
`94734673d1a3559a725f58ac2a33456d4063995bb312591613c381e828121750`.

## What the noise does

[PR331's reproduction](https://github.com/255BITS/ParticleGAN/blob/49f041708931d06319213069be060f91f8ba9fb2/reports/forge/ring16-failure/REPRODUCTION.md)
found that a `1.03e-7` relative hidden-critic gradient difference at 401 becomes
a `.252` difference after ordinary full polar normalization. This intervention
perturbs weak singular subspaces before applying the ordinary polar rule:

```text
W = U S Vh
tau = max(rows, columns) * float32_epsilon * sigma_max
weak = singular_values <= tau; k = count(weak)
N[k, k] ~ iid Normal(0, float32_epsilon * sigma_max / sqrt(k))
W_prime = W + U[:, weak] @ N @ Vh[weak, :]
direction = ordinary_polar_factor(W_prime)
```

One amplitude and threshold formula applies to every generator and critic
matrix, with no tuning. Computation remains CUDA float32. The named CUDA noise
stream is checkpointed independently of data, prior and evaluation RNGs.
Bias and sampled-prior update rules remain unchanged. Exactly zero gradients
and matrices without weak directions consume no noise draw. In exact arithmetic
the additive perturbation stays in weak left/right subspaces; float32
reconstruction can round other entries and strong projections too.

The one saved-gradient GPU probe passes its implementation gate. Both inputs
have six weak directions; actual relative perturbations are about `2.94e-7`
and the noise standard deviation is `1.35e-8`. Captured ordinary factors match
exactly, identical-input/identical-noise repeats and consumed RNG after-states
are exact, and ambient RNG plus the unconsumed canonical stream are unchanged.
It costs `.412287` seconds, four perturbation tensor draws and zero training
updates or target/prior/evaluation draws.

The probe does **not** reduce paired factor discrepancy: it increases from
`.251990` to `.322775`. Within-input factor change is about `.28`, despite
the small raw perturbation. This follows the full polar rule's sensitivity
and distinguishes noise from spectral damping. Better numerical insensitivity
is not a prerequisite for this observed training success.

The uninterrupted every-step result shows the fixed public-API recipe can
acquire the full Ring16 law and finish with a long passing terminal sequence.
It does not identify the exact backward instruction order causing the original
reload benefit, establish a general noise schedule, or prove longer-term hold.
The boundary success also shows that a single update can select a better path.

## Conditions, cost and evidence identity

Both arms use protocol seed 0 and public deterministic named initialization,
G 4→64→64→2, Fourier D 10→64→64→1, LeakyReLU .2, batch 128, 256 learned uniform
MoG locations with sigma .1, and radius-3 Ring16 with component sigma .1.
Constant rates remain G .012, D .018 and prior .03; momentum and prior regularizer
are zero. Recipe horizon 400, external cap 1600, clean live serving,
`serial_backward=False` and all 96 checks remain fixed. Every consumed named
stream is retained in full checkpoints; noise never consumes the ambient stream.

Both arms complete 3,200 new updates in 53.253886 training seconds with 194
scored draws, inside the frozen 600-second/194-draw allowance. Including child
imports and exit, the controller measures 58.014546 seconds. Charging the probe's
full 30-second allowance conservatively gives an 88.014546-second total debit,
within the 630-second ceiling. There are zero retries, continuations, seed changes,
threshold searches or additional confirmation attempts. Physical GPU 1 is an
RTX A6000, capability 8.6, CUDA 13.0; receipts record Python 3.12.13/Torch 2.14.0.

Training source is commit `1ca28014bf7102975376efadc691f3e103535f41`, digest
`4165cdf4e92d38e58607699d814ccaf4b41896cf509aa63b230b12d7ff776811`, based on
develop `2859975707eb3ea4d31958ad1b03e9e69e148a53`. Report publication does not
change the execution identity. PR331's original uninterrupted baseline retains
its INCOMPLETE metadata-error receipt and printed covariance `2.220268`; its
restored baseline retains covariance `.514315` and six terminal passes under
the original sources listed in the protocol. No unchanged baseline is rerun.

## Actual training media and reproduction

[Boundary-only training GIF](boundary_only-actual-training.gif) and
[every-step training GIF](every_step-actual-training.gif) show actual retained
scheduled samples beside target contours with fixed axes, update numbers and
numeric gates. Saved-output CPU rendering creates no model forward or new
sample; [frame/source hashes](results.json) identify its exact evidence.

From a fresh checkout with the PR331 artifacts hydrated, use new exclusive
output roots. The probe uses a separate directory to preserve exclusive
training-campaign admission.

```sh
mkdir -p runs/reports/ring16-noise
timeout 30s env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 python -u \
  -m benchmarks.toy_audit.ring16_tiny_noise \
  --protocol reports/forge/ring16-noise/protocol.json \
  --raw /home/martyn/dev/ParticleGAN/runs/api/ring16-restart-diagnostic-v1 \
  --output runs/api/ring16-tiny-noise-probe-v1 --device cuda:0 \
  > runs/reports/ring16-noise/saved-gradient-probe.log 2>&1
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 python -u \
  -m benchmarks.toy_audit.ring16_interventions run \
  --protocol reports/forge/ring16-noise/protocol.json \
  --output runs/api/ring16-tiny-noise-v1 --device cuda:0 \
  > runs/reports/ring16-noise/training.log 2>&1
tail -F runs/reports/ring16-noise/training.log
```

The runner's `summarize` and `render` actions consume saved evidence only. Raw
states, tensors, noise-stream dumps and logs remain in the verified
[local archive](archive.json), outside Git. Compact metrics, provenance,
reproduction sources and actual-training GIFs are committed.

Recommendation: retain every-step noise as the successful fixed diagnostic
candidate for a separately declared, eligible tier 2 hold and later matched
trainer comparison. Keep the one-boundary success as trajectory-sensitivity
evidence while the separate runtime controls isolate the reload mechanism.
No additional paid work or default change follows automatically. The
[technique inventory](../technique-inventory.md) remains the sole generated
leaderboard; these diagnostics grant no Forge qualification.
