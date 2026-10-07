# Ring16 smooth spectral damping: prospective experiment

**The implementation and finite protocol are prepared; no candidate has been
trained.** This host exposes no usable CUDA device. The driver refuses CPU
execution before creating an attempt. Remote publication is also pending because
GitHub is unreachable from this session.

The question is whether reducing the optimizer's sensitivity to weak singular
directions repairs the fresh, uninterrupted Ring16 failure. This is a diagnostic
comparison with constant learning rates, not a change to production defaults.
The [protocol](protocol.json) freezes both schedules, bounds and budgets.

## Evidence and hypothesis

[PR331's completed reproduction](https://github.com/255BITS/ParticleGAN/blob/49f041708931d06319213069be060f91f8ba9fb2/reports/forge/ring16-failure/REPRODUCTION.md)
locates the first divergence at update 401. Identical serialized state, inputs and
forward values produce slightly different critic backward weight gradients.
For the hidden 64×64 gradient, a relative Frobenius difference of `1.03e-7`
becomes `.252` after ordinary `U @ Vh` normalization. Almost-null singular
directions receive unit weight under that rule. The exact runtime cause of the
initial rounding difference remains unresolved; neither a dtype conversion nor
same-input SVD nondeterminism has been demonstrated.

The damping helper computes

```text
tau = max(rows, columns) * float32_epsilon * sigma_max
weight(s) = s / hypot(s, tau)
direction = U @ diag(weight(s)) @ Vh
```

This smooth rule gives a zero singular direction zero weight. At `s = tau`,
the weight is approximately `.707`; at `s = 16*tau`, it is approximately `.998`.
For the archived hidden gradient, the numerical-rank threshold is approximately
`2.12e-6`. These are formula implications, not measured candidate results.
The single threshold formula is shared by every generator and critic matrix.
Bias updates, sampled prior-row updates, momentum and the learning rates retain
their baseline rules. There is no threshold grid, dtype conversion or random draw.

## Schedules and numerical decisions

| Arm | Intervention | Status |
| --- | --- | --- |
| `boundary_only` | Ordinary training through 400; damp all G/D matrix directions only at 401; ordinary updates 402–1600 | Unmeasured, CUDA blocked |
| `every_step` | Damp all G/D matrix directions at every update 1–1600 | Unmeasured, CUDA blocked |

Update 401 is the first update after the user's proposed step 400 boundary. The
boundary arm starts from a freshly trained live prefix, checks it against the
archived full 400 state, and continues those same objects. It never loads the
archived state into the candidate. The continuous arm asks a different question:
whether damping can serve as an ongoing trainer rule from initialization.

Both arms use protocol seed 0, the public deterministic named initializer, the
same Ring16 target, G 4→64→64→2, Fourier D 10→64→64→1, batch 128 and learned
256-location uniform MoG prior with sigma `.1`. Rates remain G `.012`, D `.018`,
prior `.03`; prior regularization and momentum remain zero. Recipe horizon 400,
clean live serving, 1,600 total updates and all 96 scheduled checks remain fixed.
Autograd's existing `serial_backward=False` setting is unchanged.

Every scored draw contains 4,096 outputs. The unchanged full bounds are all 16
modes, mass TV≤`.15`, HQ≥`.85`, component covariance error≤`.85`, and minimum
component eigen ratio≥`.15`, with the original sample-count bound. A smoke
diagnostic requires a scheduled full PASS and one independent 4,096-sample
confirmation at those same weights. Confirmation uses a separately checkpointed
evaluation stream and cannot change subsequent training draws. There is one
confirmation opportunity per arm, rather than repeated attempts to obtain one.
The five-terminal-check diagnostic is reported independently. Neither diagnostic
grants Forge qualification; tier 2 retention requires a separate gated budget.

Each schedule has one 1,600-update attempt and a 300-second training allowance:
3,200 updates, 600 reserved seconds, at most 194 scored draws across both arms,
zero retries. First acquisition does not stop training. Exceptions, nonfinite
values and timeouts remain visible. The saved-gradient probe has a separate
one-execution 30-second allowance and consumes zero updates or sampling draws.

## Baseline reuse and limits

The existing source-bound live result has final covariance error `2.220268`,
HQ `.936035` and zero full passing observations. Its original receipt remains
INCOMPLETE because metadata checking failed after training; the retained printed
numeric failure is not relabeled a complete qualification receipt. Restoring
the same 400 state reproduces the historical PASS exactly: covariance `.514315`
and six terminal passes. That restored runtime path is separate evidence.

The protocol preserves the original live/restored training commits and source
digests separately from PR331's report publication commit.

No unchanged baseline is repeated here. The two schedules are prospective
comparisons against those archived measurements, with exact prefix checking
for the boundary arm. Matching state 400 and batches cannot prove that every
uncheckpointed live runtime ordering at 401 matches the earlier run. An
improvement therefore remains a diagnostic result while the rounding cause is
unresolved; it does not establish a new default or a general stability claim.

The [saved-gradient probe](../../../benchmarks/toy_audit/ring16_spectral_damping.py)
uses the exact PR331 live/restored 401 hidden gradients on CUDA. Its falsifiable
mechanistic gate requires finite factors, exact same-input repeats, agreement
with the captured ordinary factors, no RNG consumption, completion within 30
seconds, and at least a tenfold reduction in the relative Frobenius discrepancy between the two factors.
Passing this check would demonstrate reduced sensitivity for this saved pair,
not distributional convergence. Failure is retained without adjusting tau.

## Prepared execution and artifacts

From this checkout, on a CUDA host, use a new output directory:

```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -u \
  -m benchmarks.toy_audit.ring16_interventions run \
  --protocol reports/forge/ring16-damping/protocol.json \
  --output runs/api/ring16-spectral-damping-v1 --device cuda:0 \
  > runs/reports/ring16-damping/training.log 2>&1
```

Create the log parent directory first; `tail -F` can follow the flushed JSON
progress. The baseline 400 checkpoint must be hydrated at the driver's `--baseline`
path with the exact declared hash. It is read only for comparison. Run the
saved-gradient probe once, enforcing its separate allowance:

```sh
timeout 30s env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -u \
  -m benchmarks.toy_audit.ring16_spectral_damping \
  --protocol reports/forge/ring16-damping/protocol.json \
  --raw /home/martyn/dev/ParticleGAN/runs/api/ring16-restart-diagnostic-v1 \
  --output runs/api/ring16-spectral-damping-v1/saved-gradient-probe \
  --device cuda:0 > runs/reports/ring16-damping/saved-gradient-probe.log 2>&1
```

After actual training, the runner's `summarize` and `render` actions retain the
numeric result and generate actual-training GIFs from saved API observations.
No candidate GIF exists yet, and the archive's original GIF is not substituted.
Bulk logs, checkpoints and scored tensors stay under ignored `runs/`; publication
will include compact metrics, exact provenance and genuine candidate GIFs.

[Static verification](verification.json) records the current zero-update checks.
[PR body](PR.md) is ready for publication. The
[technique inventory](../technique-inventory.md) remains the sole generated
leaderboard; this report adds no ranked result, gate change or tier eligibility.

Recommendation: run the saved-gradient check and these two frozen CUDA arms,
then compare acquisition and endpoint fidelity before declaring a separate
retention study. Do not infer that either schedule repairs Ring16 yet.
