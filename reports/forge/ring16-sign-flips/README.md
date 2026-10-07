# Ring16 weak spectral sign flips: prospective CUDA experiment

**Prepared; training and CUDA algebra remain unmeasured.** This host has no
available CUDA device. The experiment rejects CPU neural execution before
creating an attempt. Package defaults, task gates and qualification are unchanged.

[PR331's reproduction](https://github.com/255BITS/ParticleGAN/blob/49f041708931d06319213069be060f91f8ba9fb2/reports/forge/ring16-failure/REPRODUCTION.md)
found that live and restored objects first produce slightly different critic
weight gradients at update 401. A `1.03e-7` relative hidden-gradient difference
becomes a `.252` relative normalized-direction difference. The original SVD rule
assigns unit strength to nearly null directions. That motivates this explicitly
randomized path-selection experiment, without establishing that randomness is
the missing technique or the cause of the restart discrepancy.

## Mechanism and schedule

The [helper](../../../benchmarks/toy_audit/ring16_small_sign_flips.py) computes:

```text
A = U diag(s) Vh
tau = max(rows, columns) * float32_epsilon * max(s)
sign[i] = +1                    when s[i] > tau
sign[i] = independent fair ±1  when s[i] <= tau
direction = U diag(sign) Vh
```

This flips weak **singular components**, rather than small individual model
weights or gradient entries. Strong components keep coefficient `+1`. A nonzero
weak gradient component changes sign before its full unit normalization. If a
singular value is exactly zero, the original polar includes an arbitrary unit
completion there; flipping that direction perturbs the normalized completion
rather than a nonzero gradient. Consequently the update perturbation can have
unit strength even when the original gradient component is tiny. This experiment
does not claim to reproduce a specific floating-point conversion or tiny-noise law.

The matrix must be nonempty, finite CUDA float32, with no matrix dtype conversion.
Only the passed isolated CUDA generator is used. Each weak component consumes
one independent uniform draw, choosing `-1` below `.5` and `+1` otherwise. An
entirely zero matrix returns zero without draws; a matrix with no weak directions
returns the ordinary factor without draws. G/D matrix weights use the same rule.
Bias normalization, aspect-ratio factors and sampled prior row updates are fixed.

| Arm | Intervention | What a success would show |
| --- | --- | --- |
| `boundary_only` | Original live updates 1–400; draw weak signs during update 401 only; ordinary full polar afterward | One weak-direction perturbation can change this trajectory at the known boundary |
| `every_step` | Draw independent weak signs on eligible G/D matrices throughout updates 1–1,600 | Continuous exploration merits a separately budgeted retention test |

The boundary change happens **after completing update 400**, during update 401.
Both arms initialize through the public API and train live objects. Neither
restores a candidate checkpoint, because restoring already changes the path.
PR331's exact fresh 400-state is only a saved-value identity check for the
boundary arm. Even identical serialized values do not prove identical subsequent
autograd ordering, so any result remains bound to its actual source/runtime.

## Protocol, gates and interpretation

The [protocol](protocol.json) freezes seed 0, named public initialization, the
selected BCAP recipe and one global trainer delta. Target: uniform 16-component
2D Gaussian ring, radius 3, sigma `.1`. G is `4→64→64→2`; Fourier-2 D is
`10→64→64→1`, with LeakyReLU `.2`. Prior: learned uniform 256-row MoG, sigma
`.1`, no standardization. Batch 128, zero momentum/prior regularizer and constant
G/D/prior rates `.012/.018/.03` are unchanged. There is no probability or threshold
grid. `serial_backward` remains at the baseline setting.

Each arm finishes 1,600 updates with recipe horizon 400, clean live public
sampling, and 96 scheduled 4,096-sample checks (24 per 400 updates). Full quality
requires all six original bounds: sample count ≥ 4,096, modes ≥ 16, mass TV ≤
`.15`, HQ ≥ `.85`, mean component covariance error ≤ `.85`, minimum component
eigenvalue ratio ≥ `.15`. Smoke requires any scheduled full passing state and
one independent confirmation draw at the same state. Confirmation uses its own
checkpointed stream and has no retry. The historical five-terminal-check verdict
and longest passing streak remain separate diagnostics. No covariance tails are
excluded and no task reducer is rewritten.

The two arms reserve **3,200 updates, 600 seconds and at most 194 scoring draws**.
One additional saved-gradient CUDA probe reserves 30 seconds, zero updates and
zero target/evaluation draws, for a combined **630-second ceiling**. There are
no scientific retries or unchanged baseline reruns. Passing this diagnostic does
not qualify a family or launch tier 2. A continuation/retention question needs
its own frozen budget and passing smoke checkpoint dependency.

The probe uses the exact saved live/restored hidden critic gradients at 401,
verified by original trace hashes. A separate named perturbation generator
matches the training stream binding. Only that stream is reset before each input
and identical-input repeat; all consumed states are saved, including on Python
errors. Matched coins need not match spectral directions: the two gradients can
have different weak bases or counts. Report both raw/full-polar sensitivity and
the perturbed pair mismatch, without requiring it to shrink. The perturbation
gate requires finite outputs, exact same-input/same-RNG repeats, strong signs
`+1`, weak signs `±1`, unchanged ambient RNG states and checkpointed consumed
streams. Separate fixed zero/identity algebra fixtures verify no-draw guards;
they never substitute for learned initialization or the archived pair.

Original full factors are also compared with captured factors as a source/runtime
compatibility diagnostic. The report publication is commit `49f0417`; actual
trace execution used live `74e5e1e` and restored `1de465e2`, whose distinct source
digests remain in the protocol. The original live arm printed covariance error
`2.220268` and zero full passing states, while preserving its postexecution
metadata-error INCOMPLETE receipt. Restored covariance `.514315` with six terminal
passes belongs to that original restart path, rather than a fresh continuous
control. These historical controls are reused only under their original identity.

## Execute and retain results

The shared [public-API runner](../../../benchmarks/toy_audit/ring16_interventions.py)
owns fresh construction, training, scoring and full context checkpoints. It keeps
the noise, prior, data and evaluation streams separate. Actual-training GIFs
come from saved sampled outputs after execution; none exists for this unmeasured
candidate. Run from this branch on a supported CUDA host, with PR331's exact
fresh `live/prefix-state.pt` hydrated locally. A relocated artifact can be supplied
with `--baseline`.

```sh
python -m benchmarks.toy_audit.ring16_interventions plan \
  --protocol reports/forge/ring16-sign-flips/protocol.json \
  --output runs/api/ring16-small-sign-flips-v1

mkdir -p runs/api/ring16-sign-flips-logs
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -u \
  -m benchmarks.toy_audit.ring16_interventions run \
  --protocol reports/forge/ring16-sign-flips/protocol.json \
  --output runs/api/ring16-small-sign-flips-v1 --device cuda:0 \
  > runs/api/ring16-sign-flips-logs/training.log 2>&1
tail -F runs/api/ring16-sign-flips-logs/training.log

timeout 30s env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -u \
  -m benchmarks.toy_audit.ring16_small_sign_flips probe \
  --protocol reports/forge/ring16-sign-flips/protocol.json \
  --live-trace /path/to/PR331/live/trace.pt \
  --restart-trace /path/to/PR331/restart/trace.pt --device cuda:0 \
  --output runs/api/ring16-small-sign-flips-v1-algebra \
  > runs/api/ring16-sign-flips-logs/algebra.log 2>&1

python -m benchmarks.toy_audit.ring16_interventions render \
  --protocol reports/forge/ring16-sign-flips/protocol.json \
  --output runs/api/ring16-small-sign-flips-v1
```

Run/probe directories are exclusive. Keep raw stdout, per-update metrics, tensors,
checkpoints and RNG states ignored under `runs/api`; archive original bytes and
publish compact metrics, provenance and actual-training GIFs afterward. The
[verification receipt](verification.json) records static checks without claiming
GPU measurements. The [current technique inventory](../technique-inventory.md)
remains the sole generated goal leaderboard; this diagnostic is unranked.

Compare confirmed acquisition first, then the full terminal diagnostic. Preserve
a negative result. A boundary-only success would justify studying path selection;
an every-step success would justify separately declaring retention. Neither is
evidence for a production default or stable convergence on other tasks.
