# Ring16: serialized autograd with spectral truncation

**The continuous combination passes confirmed acquisition at update1050 and
every scheduled check from1050 through1600:34 final full passes.** No restart,
annealing or optimizer reset was used. Final covariance error is .470678 within
the unchanged .85 bound. The positive result demonstrates compatibility in this
fixed Ring16 run, without establishing indefinite retention or global adoption.

This diagnostic is based on develop `2859975707eb3ea4d31958ad1b03e9e69e148a53`.
The [frozen protocol](protocol.json) asks whether continuous serialization and
the measured truncation rule work together under the selected BCAP recipe.
[Compact results](results.json), [saved-evidence checks](verification.json),
[actual-training GIF](actual-training.gif) and [archive receipt](archive.json)
preserve the execution source and complete outcome.

## Results and comparison

This is an unranked diagnostic comparison. The original continuous truncation
and serialization arms were not rerun. Initial models, initializer, recipe,
prior, all1600 target batches, cadence and gates match exactly. Public numerical
package files are identical across all three. The combined run uses the same
typed serialization API as PR337 and the exact original truncation helper from
PR332; its only substantive delta is composing those mechanisms every update.

| Continuous trainer | First full scheduled pass | Independent confirmation | Final full passing streak | Final covariance error, bound .85 |
| --- | ---: | --- | ---: | ---: |
| Truncation, PR332 |684|PASS|47, from834|.589678|
| Serialization, PR337 |1400|FAIL, covariance .894789|11, from1434|.385800|
| **Both, this study** |**1050**|**PASS, covariance .817187**|**34, from1050**|**.470678**|

The combination has34/96 full scheduled passes and zero failures after its
confirmed acquisition. Final metrics:16 modes, HQ .956055, massTV .078613,
minimum component eigenvalue ratio .387340,4096 samples. The one independent
first-pass confirmation is retained; no later confirmation retries were taken.

Truncation alone confirms earlier, but fails covariance at717/734/750/817 after
acquisition and then passes all47 checks834–1600. Serialization alone fails its
first confirmation and covariance at1417, then passes its final11 observations.
The combination begins its uninterrupted passing streak later than truncation
alone while ending with lower covariance error. These finite trajectories do
not establish universal superiority or strict Tier2 retention.

## Interpretation and next action

Serialization and truncation work together in this measured cohort. Keep both
the combination and truncation alone as candidates for a separately frozen
whole-configuration Tier1 comparison. Truncation alone retains its earlier
acquisition advantage; the combination supplies confirmed acquisition with
serialized execution throughout. Neither result justifies replacing a global
default before completing the required ordinary gates across tasks.

No combined restart-parity experiment was run. The earlier serialized trainer
checkpoint software evidence is reused under its actual source, and this run
checks serialized scope on all9600 G/D matrix-direction calls. This proves the
two mechanisms were active together; it does not prove their full future
trajectories or all hardware/restart cases are identical.

The extra math for truncation on this SVD path is a singular-value comparison
and column masking; SVD and the matrix product already exist. Serialization
changes the CPU backward scheduling context while retaining GPU tensor work.
No empirical speed comparison is made or inferred from the campaign timings.

## Fixed protocol and cost

One fresh live CUDA arm runs1600 updates with public `GANTrainer.serial_backward=True`
and the unchanged [PR332](https://github.com/255BITS/ParticleGAN/pull/332)
spectral rule `U diag(s > max(matrix.shape)*float32_epsilon*smax) Vh` on G/D matrix
directions. The [shared runner](../../../benchmarks/toy_audit/ring16_interventions.py)
comes from [PR337](https://github.com/255BITS/ParticleGAN/pull/337). A thin
[composition hook](../../../benchmarks/toy_audit/ring16_serialized_truncation.py)
checks that every truncated polar call executes inside the serialized scope.
The public typed extension, trainer and original truncation helper are unchanged.
No checkpoint is loaded into the learner. No empirical speed study is requested.

Keep seed0, public deterministic named initialization, z4, G4→64→64→2,
Fourier D10→64→64→1, learned256-location MoG sigma.1, batch128, constant
G/D/prior rates .012/.018/.03, recipe horizon400, clean live sampling and96
scheduled4096-sample checks. Full bounds remain16 modes, mass TV≤.15, HQ≥.85,
component covariance error≤.85 and minimum component eigenvalue ratio≥.15.
Smoke requires the first full scheduled pass plus its one independent same-state
confirmation; failed confirmation cannot be retried. Finish all1600 updates.
Report subsequent failures, terminal streak and five-terminal-check verdict
separately. No automatic retention work or ordinary qualification follows.

Budget: one attempt/1600 updates/300 whole subprocess seconds/97 scored draws,
zero retries. Reuse source-bound truncation and serialization results without
training them again; verify matched initialization, prior, recipe, target batch
sequence, cadence, gates and unchanged public numerical code. All neural work
uses physicalGPU0 as logicalcuda:0. Metric math on saved CUDA samples retains
the original CPU scorer cohort; actual-training GIF rendering adds no model calls.
Raw evidence stays outside Git. Easy-tail log:
`runs/reports/ring16-serialized-truncation/execution.log`.

The single arm completed1600 updates and97 scored draws in25.551 whole
subprocess seconds within300 reserved seconds, with zero retries. The budget is
closed; no new software-training or reused-comparison updates were spent.
The byte-verified archive contains19 original files plus its member manifest,
3,645,960 bytes, SHA256
`04bf1bdf7d4c883e0934535e11da6835f1bea48fdea4897ac6d18332d7b61653`.
It retains stdout, frozen protocol, source, initialization/prefix401/acquisition/
final checkpoints, all96 observations, confirmation samples, named/ambient RNG
states and actual-training media. Source commit:
`5e1d33f37b312a1b4a760e8417d76542570f0768`, scientific source digest
`51d30259ceee202a24a39a6a5fea3756bf46f56fc1b7a7e85fc735c6fb4734f8`.

## Reproduction

```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m benchmarks.toy_audit.ring16_interventions plan \
  --protocol reports/forge/ring16-serialized-truncation/protocol.json \
  --output runs/api/ring16-serialized-truncation-v1
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m benchmarks.toy_audit.ring16_interventions run \
  --protocol reports/forge/ring16-serialized-truncation/protocol.json \
  --output runs/api/ring16-serialized-truncation-v1
```

Output directories are exclusive. Original comparisons keep their original
source and qualification identities. The current inventory remains the sole
generated leaderboard; this explicitly scoped Ring16 combination supplies no
family default or tier eligibility.

```sh
python reports/forge/ring16-serialized-truncation/verify_saved.py
```

Saved-evidence verification launches no models or new draws. It requires the
exact original local comparison evidence and the byte-bound PR337 saved-output
reducer; locations can be supplied through its explicit arguments. The GIF shows
nine actual training snapshots with fixed axes and target rings.
