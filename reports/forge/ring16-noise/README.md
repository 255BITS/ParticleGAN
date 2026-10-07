# Ring16 tiny weak-subspace noise: prospective experiment

**The finite experiment is prepared; both candidates remain unmeasured.** This
host has no usable CUDA device, and the driver refuses CPU training before
creating an attempt. GitHub publication is pending network access. There are no
new neural updates, candidate quality results or candidate GIFs in this report.

The [protocol](protocol.json) tests whether a tiny gradient perturbation at the
restart boundary changes acquisition, and whether the same intervention is useful
as an ongoing rule. It keeps the original learning rates constant and introduces
no production optimizer change.

## Mechanism and relation to the restart evidence

[PR331's reproduction report](https://github.com/255BITS/ParticleGAN/blob/49f041708931d06319213069be060f91f8ba9fb2/reports/forge/ring16-failure/REPRODUCTION.md)
finds a relative hidden-critic gradient difference of `1.03e-7` at update 401,
amplified to `.252` by full polar normalization. The weak singular directions,
rather than individual small entries, carry the measured sensitivity. Identical
inputs reproduce identical SVD factors in that probe. The initial backward
rounding cause is unresolved; a dtype conversion has not been established.

This experiment defines tiny noise precisely. For gradient matrix `W = U S Vh`:

```text
tau = max(rows, columns) * float32_epsilon * sigma_max
weak = singular_values <= tau
k = count(weak)
N[k, k] ~ iid Normal(0, float32_epsilon * sigma_max / sqrt(k))
W_prime = W + U[:, weak] @ N @ Vh[weak, :]
direction = ordinary_polar_factor(W_prime)
```

In exact arithmetic, the additive perturbation lies within the weak left/right
singular subspaces. Float32 reconstruction can round other entries and strong
projections, so this is not a promise of exact coordinate isolation. The rule
uses the original float32 dtype throughout. It draws no noise for a matrix with
no weak directions and returns a zero direction without a draw for an exactly
zero gradient. The helper captures the original polar function before the runner
patches it, preserving the ordinary update after perturbation.

One threshold and amplitude formula applies to all generator and critic matrix
weights. Biases, sampled prior-row normalization and the selected recipe remain
unchanged. There is no amplitude grid, schedule search or seed variation. Noise
uses the isolated named CUDA stream `noise/ring16_intervention/weak_directions`;
the full public context checkpoints its actual state. It does not consume the
data, prior, evaluation or ambient RNG streams.

This hypothesis differs from damping: full polar normalization still gives the
weak directions unit strength. Noise can choose another numerical path without
making the update insensitive to perturbations. A better outcome would need to
be demonstrated by the unchanged distribution gate; saved-matrix sensitivity
alone cannot establish a convergence benefit.

## Frozen schedules and gates

| Arm | Intervention | Current result |
| --- | --- | --- |
| `boundary_only` | Train the unchanged fresh live prefix through 400; perturb G/D matrix gradients at 401 only; continue ordinary updates through 1600 | Unmeasured, CUDA blocked |
| `every_step` | Apply the same perturbation at updates 1–1600 | Unmeasured, CUDA blocked |

Update 401 is the first update after the proposed step 400 boundary. The boundary
arm verifies the full fresh 400 state against PR331's archived prefix and continues
the same live objects. Neither candidate loads that reference checkpoint. The
continuous arm deliberately changes the trajectory from update 1; its 400 state
is recorded without requiring equality to the unchanged baseline.

Both arms use protocol seed 0, public deterministic named initialization, the
uniform radius-3 Ring16 target with component sigma `.1`, G 4→64→64→2,
Fourier D 10→64→64→1, batch 128 and 256 learned uniform MoG locations with
prior sigma `.1`. Rates remain G `.012`, D `.018`, prior `.03`; momentum and
prior regularization are zero. The original recipe horizon is 400, execution
cap 1,600, clean live public sampling law and 96-check cadence are fixed.
`serial_backward=False` is unchanged. Hardware/runtime are recorded; execution
on different hardware remains a separate cohort from the archived A6000 path.

Every scheduled evaluation draws 4,096 outputs and applies all original full
bounds: 16 modes, mass TV≤`.15`, HQ≥`.85`, component covariance error≤`.85`,
minimum component eigen ratio≥`.15`, and the original sample-count bound.
The smoke diagnostic requires a scheduled full PASS and one independent
confirmation at the same weights. That one confirmation uses a separate
checkpointed evaluation stream; a failed confirmation is not retried. The
five-terminal-pass diagnostic remains separate, and all 1,600 updates are
completed after acquisition. No Forge qualification or tier2 hold follows.

Each arm reserves one 1,600-update attempt and 300 seconds: 3,200 updates,
600 training seconds, at most 194 scoring draws, zero retries. A separate
one-execution CUDA algebra probe reserves 30 seconds and zero training updates
or target/prior/evaluation sampling draws. It makes at most four perturbation
tensor draws on explicit cloned streams. Exceptions, nonfinite values and
timeouts remain incomplete/error outcomes without changing the noise scale.

## Saved-gradient check and comparison limits

The [probe helper](../../../benchmarks/toy_audit/ring16_tiny_noise.py) uses the
exact saved live/restored update-401 hidden gradients. Each input and its repeat
receive a clone of the same initial named noise state. Only those diagnostic
clones are reset; no model RNG is reset. Their complete consumed stream states
are retained in an ignored artifact with a hash in the compact receipt.

Its implementation gate requires finite results, an actual relative gradient
perturbation≤`1e-5`, exact identical-input/identical-noise repeats, identical
repeat RNG after-states, agreement with captured ordinary factors, unchanged
ambient RNG and canonical template state, and completion within 30 seconds.
It reports factor sensitivity without an improvement threshold. Weak SVD bases
can differ between inputs even with matched noise coordinates; neither lower
nor higher factor discrepancy alone answers the training question.

PR331's existing uninterrupted live result has covariance error `2.220268`,
HQ `.936035` and zero full passes. Its original receipt remains INCOMPLETE
after a post-training metadata error; the printed numerical failure is retained
separately. Restoring the same 400 state exactly reproduces the historical PASS
with covariance `.514315` and six terminal passes. These retain their original
execution commits and source digests, separate from report commit `49f0417`.

No unchanged baseline is repeated. The archived comparison and exact prefix
check do not prove identical uncheckpointed runtime ordering at 401. An apparent
repair is therefore a source-bound diagnostic result while that mechanism is
unresolved, rather than a causal default or general stability claim. Later
retention requires its own frozen, eligible tier2 budget.

## Execution and publication

On a CUDA host, hydrate the declared baseline and trace artifacts, create
`runs/reports/ring16-noise/`, and use new exclusive output directories:

```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -u \
  -m benchmarks.toy_audit.ring16_interventions run \
  --protocol reports/forge/ring16-noise/protocol.json \
  --output runs/api/ring16-tiny-noise-v1 --device cuda:0 \
  > runs/reports/ring16-noise/training.log 2>&1

timeout 30s env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -u \
  -m benchmarks.toy_audit.ring16_tiny_noise \
  --protocol reports/forge/ring16-noise/protocol.json \
  --raw /home/martyn/dev/ParticleGAN/runs/api/ring16-restart-diagnostic-v1 \
  --output runs/api/ring16-tiny-noise-v1/saved-gradient-probe \
  --device cuda:0 > runs/reports/ring16-noise/saved-gradient-probe.log 2>&1
```

Flushed JSON logs support `tail -F`. The runner's `summarize` and `render` actions
use retained scored samples and produce genuine candidate training GIFs after
execution. Bulk logs, checkpoints, tensors and stream dumps stay under ignored
`runs/`; compact results, provenance and actual-training GIFs can then be
published. No original GIF is presented as a noise candidate result.

[Static verification](verification.json) covers preparation with zero neural
execution. [PR body](PR.md) is ready for publication. The
[technique inventory](../technique-inventory.md) remains the sole generated
leaderboard; no rank, gate or tier eligibility changes here.

Recommendation: compare these two frozen CUDA schedules against the saved
baseline, reporting acquisition and endpoint quality separately. Tiny noise is
an untested way to change the numerical trajectory, not yet a Ring16 repair.
