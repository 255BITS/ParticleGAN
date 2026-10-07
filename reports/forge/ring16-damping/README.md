# Ring16 spectral damping: CUDA results

**Damping matrix updates once at update 401 produces 31 consecutive full terminal
passes. Applying damping at every update produces no full pass.** The boundary
arm's single independent confirmation at its first passing observation narrowly
fails covariance, so it fails the separately declared confirmed-smoke rule
despite its later terminal success. Both fresh-live 1,600-update arms complete
on an RTX A6000 with constant learning rates and no checkpoint reload.

The [frozen protocol](protocol.json), [compact results](results.json),
[execution verification](execution-verification.json) and [archive receipt](archive.json)
bind outcomes to their actual source and sampling law. The earlier CUDA-blocked
[preparation verification](verification.json) retains its original identity;
its unmeasured status describes preparation before CUDA returned on 2026-10-07.

## Results and declared gates

Each 4,096-output draw must satisfy all bounds: 16 modes, mass TV ≤ .15,
HQ ≥ .85, mean component covariance error ≤ .85 and minimum component eigen
ratio ≥ .15. Confirmed smoke requires a scheduled full pass and the one
independent confirmation at those same weights. The five-terminal-pass
verdict is reported independently. These diagnostics supply no tier 2 hold.

| Schedule | First full pass | Confirmation | Terminal suffix | Terminal verdict | Final covariance | Final HQ | Final eigen ratio |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: |
| `boundary_only` (401 only) | 1067 | FAIL: covariance .859541 | 31 | PASS | .458855 | .969238 | .367013 |
| `every_step` (1–1600) | None | Not reached | 0 | FAIL | 1.057656 | .969482 | .346198 |

The boundary arm trains unchanged through 400, verifies its complete state
against PR331's archive, and continues those same live objects. Its projected
state digest is exactly
`208e2d7b241ffeac11915972dbb90979dd686ad5285768282e2182d7da5ad2cd`.
The projection removes only two newly registered, unused diagnostic streams.
It never restores the archived prefix. The intervention makes six matrix
normalization calls at 401; biases and prior updates retain their usual rules.

Every-step damping makes 9,600 matrix calls and intentionally changes the
trajectory from initialization. Both arms consume the same target batch sequence,
digest `94734673d1a3559a725f58ac2a33456d4063995bb312591613c381e828121750`.
Confirmation consumes a separate checkpointed evaluation stream. The first
confirmation failure is retained; there is no additional confirmation draw.

## Mechanism and saved-gradient probe

[PR331's reproduction](https://github.com/255BITS/ParticleGAN/blob/49f041708931d06319213069be060f91f8ba9fb2/reports/forge/ring16-failure/REPRODUCTION.md)
found a `1.03e-7` relative difference in the hidden critic gradient at 401 that
becomes `.252` after ordinary full polar normalization. Almost-null singular
directions receive unit strength. The experiment replaces G/D matrix directions
with one fixed smooth rule:

```text
tau = max(rows, columns) * float32_epsilon * sigma_max
direction = U @ diag(s / hypot(s, tau)) @ Vh
```

The one CUDA probe passes its preregistered tenfold-reduction gate: paired
direction discrepancy falls from `.2519904673` to `.0001956535`, about
1,288-fold. Both gradients have six weak directions and tau approximately
`2.12e-6`. Captured ordinary factors match exactly, identical-input repeats
are exact, outputs are finite and ambient RNG is unchanged. The probe costs
`.365589` seconds with zero training updates or sampling draws.

Damping therefore suppresses the archived numerical amplification. Its ongoing
version still fails quality, so suppression alone does not establish a
successful continuous trainer. The single-boundary success shows that changing
one matrix update can select a better trajectory; it does not isolate the
instruction order that produced the reload's original gradient difference.

## Conditions, cost and provenance

Both arms retain seed 0 and public deterministic named initialization:
G 4→64→64→2, Fourier D 10→64→64→1, LeakyReLU .2, batch 128, learned uniform
MoG with 256 locations and sigma .1, radius-3 Ring16 with component sigma .1.
Constant rates are G .012, D .018 and prior .03; momentum and prior regularizer
are zero. Recipe horizon 400, external cap 1600, clean live public sampling,
`serial_backward=False` and all 96 scheduled checks remain fixed. Data,
training noise and evaluation streams are isolated and checkpointed.

The arms consume 3,200 new updates, 45.351709 training seconds and 193 scored
draws, within 600 reserved seconds and 194 draws. Including child imports and
exit, the controller measures 50.003936 seconds. Conservatively charging the
probe's full 30-second allowance gives a total debit of 80.003936 seconds,
within the 630-second ceiling. There are zero retries, continuations, seed changes or
threshold searches. Physical GPU 1 is an RTX A6000, capability 8.6, CUDA 13.0;
Python 3.12.13 and Torch 2.14.0 are recorded in the receipts.

Training executes commit `996fcaa581ef145650acff96824a77b704e71dc3`, source digest
`74de41f325a59c40df404a9107b0d40d6adbaf5e394f48f3ce13fbbf4be56ccb`, based on
develop `2859975707eb3ea4d31958ad1b03e9e69e148a53`. Publication does not change
that execution identity. The original live baseline retains its post-training
INCOMPLETE receipt and printed covariance `2.220268`; the restored baseline
retains covariance `.514315` and six terminal passes under its original sources,
listed in the frozen protocol. Neither was rerun.

## Actual training media and reproduction

[Boundary-only training GIF](boundary_only-actual-training.gif) and
[every-step training GIF](every_step-actual-training.gif) render retained actual
scheduled samples with fixed axes, target contours, updates and numerical
verdicts. Saved-output CPU rendering creates no model forwards or new samples.
[Media hashes and frame provenance](results.json) identify the observation files.

Run once from a fresh checkout with hydrated PR331 artifacts and new exclusive
output directories. The probe uses a separate root so it cannot pre-create the
training campaign directory.

```sh
mkdir -p runs/reports/ring16-damping
timeout 30s env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 python -u \
  -m benchmarks.toy_audit.ring16_spectral_damping \
  --protocol reports/forge/ring16-damping/protocol.json \
  --raw /home/martyn/dev/ParticleGAN/runs/api/ring16-restart-diagnostic-v1 \
  --output runs/api/ring16-spectral-damping-probe-v1 --device cuda:0 \
  > runs/reports/ring16-damping/saved-gradient-probe.log 2>&1
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 python -u \
  -m benchmarks.toy_audit.ring16_interventions run \
  --protocol reports/forge/ring16-damping/protocol.json \
  --output runs/api/ring16-spectral-damping-v1 --device cuda:0 \
  > runs/reports/ring16-damping/training.log 2>&1
tail -F runs/reports/ring16-damping/training.log
```

The runner's `summarize` and `render` actions read saved evidence without
training. Full states, samples, logs and streams remain in the verified
[local archive](archive.json), outside Git. Compact metrics, source, hashes and
actual-training GIFs are committed.

Recommendation: retain the boundary success as evidence that one update can
repair acquisition, and prioritize the separate backward-order controls to
isolate the reload mechanism. Every-step damping is a completed negative under
this fixed formula. Any later hold or wider trainer comparison needs a new
eligible, frozen budget. The [technique inventory](../technique-inventory.md)
remains the sole generated leaderboard; this report grants no qualification.
