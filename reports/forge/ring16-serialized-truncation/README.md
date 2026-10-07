# Ring16: serialized autograd with spectral truncation

Prospective diagnostic based on develop `2859975707eb3ea4d31958ad1b03e9e69e148a53`.
The [frozen protocol](protocol.json) asks whether continuous serialization and
the measured truncation rule work together under the selected BCAP recipe.

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
