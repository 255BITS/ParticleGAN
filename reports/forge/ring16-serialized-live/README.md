# Ring16: uninterrupted serialized autograd

**Both fresh, uninterrupted CUDA runs reached passing terminal quality.**
Always-serialized training passed its final 11 observations but failed the one
independent confirmation at its first scheduled pass. Serializing only update
401 matched the previously isolated third trajectory exactly, confirmed
acquisition at 867 and passed all 45 observations from 867 through 1600. These
results do not establish indefinite retention or a global trainer default.

This diagnostic is based on develop `2859975707eb3ea4d31958ad1b03e9e69e148a53`.
The [frozen protocol](protocol.json) separates always-serialized fresh live
training from the boundary-only trajectory isolated in
[PR334](https://github.com/255BITS/ParticleGAN/pull/334). [Compact results](results.json),
[saved-evidence verification](verification.json) and the [archive receipt](archive.json)
retain both outcomes and their actual execution identities.

## Results and comparison

These are source-bound diagnostics, not a second generated leaderboard.
Truncation/noise numbers reuse the completed original evidence without training
again. Saved initial G/D/prior tensors, recipe, prior, all 1600 target batches,
cadence and bounds match across these four arms. Every public numerical package
file is identical; serialization adds only the typed API extension and declared
execution hooks. Each comparison retains its own source/receipt identities.

| Trainer delta | First scheduled pass | Independent smoke confirmation | Final passing streak | Covariance at 1600, bound .85 |
| --- | ---: | --- | ---: | ---: |
| Serialized every update | 1400 | **FAIL**, confirmation covariance .894789 | 11, from 1434 | .385800 |
| Serialized only 401 | 867 | **PASS**, confirmation covariance .782482 | 45, from 867 | .424426 |
| [Truncation every update, PR332](https://github.com/255BITS/ParticleGAN/pull/332) | 684 | **PASS** | 47, from 834 | .589678 |
| [Weak gradient noise every update, PR335](https://github.com/255BITS/ParticleGAN/pull/335) | 817 | **PASS** | 45, from 867 | .504405 |

Always-serialized training has 12 full passes of 96; after the first pass at
1400, it fails covariance at 1417 (.972460), then passes every check 1434–1600.
Its final HQ is .948486 and mass TV .073486. The failed first confirmation is
preserved; the later terminal streak does not turn it into a confirmed smoke pass.

The boundary-only arm has 45/96 full passes and no scheduled failures after
acquisition. Its final HQ is .968262 and mass TV .051270. All 24 prefix sample
tensors through 400 match the original live evidence exactly. Its projected 401
context matches both earlier serialized causal arms, digest
`40a84b3cafddc8d2a402a03f315da56bc97c74b91853c21e29f2f9ad6fd034f8`.
The only projections remove two unused stream registrations and change the
external allowance 1600 back to the earlier 401 limit; the recipe horizon stays 400.

Truncation's post-acquisition failures remain 717/734/750/817; noise's remains
850. All four satisfy five terminal full passes, while the continuous serial arm
misses independent acquisition confirmation. A late streak and a successful
one-time intervention answer different questions from full-budget retention.

## Interpretation and recommendation

Restoring a checkpoint does not enable serialized autograd. The earlier runtime
control showed that constraining all autograd work at 401 removes the measured
live/reload gradient difference and selects a third trajectory. This new run
establishes that the same trajectory can acquire the target **without any reload**.
Always-serialized training also reaches terminal quality, but later and without
passing its one prescribed first-state confirmation. Execution determinism and
training quality remain separate properties.

Truncation remains the first candidate for a separately frozen global Tier 1
comparison: it acquired earlier, directly reduces the measured numerical
amplification, and adds no random stream or special step. The operation removes
weak **singular directions of a gradient matrix**, rather than small individual
gradient entries. Ordinary full polar normalization replaces every singular
value by 1; truncation uses `1{s > max(matrix.shape)*float32_epsilon*smax}`.
This retains meaningful directions without giving near-null directions full
update strength. It is not evidence for discarding all small raw gradients.

Retain noise as the second supported continuous candidate. Serialization is a
useful opt-in execution/reproducibility capability and a demonstrated late-quality
diagnostic, but these results do not justify replacing the selected global
recipe. Do not turn the 401 result into scheduled restarts or periodic switches.
Complete ordinary Tier 1 under one configuration before eligible Tier 2 work.

## Protocol, cost and reproduction

The shared [intervention runner](../../../benchmarks/toy_audit/ring16_interventions.py)
comes from [PR332](https://github.com/255BITS/ParticleGAN/pull/332), extended with
execution-context hooks rather than a separate training loop. The continuous
arm uses public `GANTrainer.serial_backward=True` through a declared Forge
extension. Its mode is checkpointed and incompatible modes cannot be restored.
The boundary arm keeps the baseline trainer flag False, serializes all autograd
work only at401, and continues normally afterward. Neither learner loads a
checkpoint. The boundary400 and401 states must match archived controls after
only the declared unused stream registrations/external allowance are projected.

Both arms retain seed0, public deterministic initialization, z4, G4→64→64→2,
Fourier D10→64→64→1, learned256-location MoG sigma.1, batch128, constant
G/D/prior rates .012/.018/.03, full polar normalization, recipe horizon400,
clean live serving, and96 scheduled4096-sample evaluations through1600.
All original bounds remain:16 modes, massTV≤.15, HQ≥.85, covariance error≤.85,
minimum component eigenvalue ratio≥.15. Smoke needs a full scheduled pass and
one independent unchanged-state confirmation; failed confirmation is never
retried. Both arms finish1600 regardless of acquisition. Subsequent failures,
terminal streak and five-terminal-pass verdict are separate diagnostics.

Freeze limits:2 attempts/3200 training updates/600 subprocess seconds/194
scored draws, plus one separate CUDA API software check of at most3 tiny-model
updates/30 seconds. No scientific retries, seed experiments, annealing or
automatic retention work. [Continuous GIF](every_step.gif) and
[boundary-only GIF](boundary_only.gif) each show nine actual training observations
with fixed axes and target rings, rendered without new model calls. Frozen metric math on
saved outputs uses CPU, matching the prior scorer cohort; models and optimizers
execute on CUDA. Raw evidence stays outside Git.

Both arms completed 3200 new updates and 194 scored draws in 47.597 seconds of
whole training subprocess time. The two GPU API checks passed in 1.51 pytest
seconds (3.319 seconds including process startup; 3 tiny-model updates).
Charging the full 30-second software allowance gives 77.597 seconds within the
630-second combined ceiling. No scientific retries or reused-comparison updates
were spent. Both learners stayed live from initialization through 1600.

The byte-verified raw archive contains 35 original files and one member manifest,
7,290,351 bytes, SHA256
`a14332d30079516082d062cc22a9c60a3ada6e23233a869c7184c3e71792f9d7`.
It includes stdout, source manifests, all 96 observations per arm, initial/400/401/
acquisition/final states, confirmation samples, named/ambient streams and GPU
software receipts. Preserve the original archive; the verifier launches no
training and can check these saved inputs without rewriting evidence.

Run from this checkout with the project Python environment. One worker uses
physicalGPU0, exposed as logicalcuda:0. Easy-tail log:
`runs/reports/ring16-serialized-live/execution.log`.

```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m benchmarks.toy_audit.ring16_interventions plan \
  --protocol reports/forge/ring16-serialized-live/protocol.json \
  --output runs/api/ring16-serialized-live-v1
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m benchmarks.toy_audit.ring16_interventions run \
  --protocol reports/forge/ring16-serialized-live/protocol.json \
  --output runs/api/ring16-serialized-live-v1
```

Plan is read-only. Run creates an exclusive campaign directory and refuses
reuse. This Ring-only diagnostic supplies no ordinary qualification or default
change. The current technique inventory remains the sole generated leaderboard.

```sh
python reports/forge/ring16-serialized-live/verify_saved.py
```

The verifier requires the exact local earlier-runtime, truncation and noise
artifacts (defaults shown in its arguments); archive identities and source PRs
are retained above. It performs no new neural work or gate resampling.
