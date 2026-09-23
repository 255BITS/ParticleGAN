# One experimental configuration passes all three 100-mode accuracy gates

**3/3 strict PASS**, with five terminal 20,000-draw checks and a separate
100,000-draw holdout per case. This is experimental 100-mode evidence;
it does not establish a passing shared recipe across all 22 toys.

All three runs use byte-identical [configuration](declared_config.json),
[model and schedule options](model_options.json), [probe source](probe_source.py),
and [actual optimizer-rate traces](optimizer_actions.jsonl). The generic
uniform square [-5, 5]² initialization and identity affine generator use no
target samples, labels, or centers. The learned prior retains its normal
7,000-update cosine schedule. G and D use the same cosine with its horizon
capped at 1,600 updates, then continue training at its floor. No seed sweep,
per-problem configuration, or EMA substitution is used.

| Case | Strict gate | Final live modes | Holdout HQ | Holdout mass TV | Center RMS / σ | Covariance bias | Radial KS |
|---|---|---:|---:|---:|---:|---:|---:|
| grid100 | PASS | 100/100 | .98636 | .03565 | .13912 | −.03706 | .01214 |
| rotated100 | PASS | 100/100 | .98311 | .03784 | .12087 | −.03670 | .01360 |
| staggered100 | PASS | 100/100 | .98607 | .04024 | .10432 | −.03874 | .01519 |

The [combined GIF](toy100-progress.gif) shows the full convergence trajectory from
saved live samples, including early instability. Accuracy is sustained at
the final five checkpoints (6,000 through 7,000); this should not be read as
rapid sustained convergence at the first moment of 100-mode coverage.
Each case also has an individual GIF, events, snapshots, saved terminal
clouds, holdout, and independently recomputable metrics.

The older hosts retain their own frozen architectures, data, and budgets.
This H1600 rule alone leaves their shorter schedules unchanged. The matching
optimizer/noise core previously passed 15/19 older cases, so the next task is
finding one configuration that retains this 3/3 accuracy and repairs those
remaining failures. The production gate must include this model declaration,
schedule policy, and executable provenance before these probes can qualify.

```bash
python -m benchmarks.toy100 accuracy --output reports/toy100/accuracy-network-horizon
# Expected: PASS, 3/3. To reproduce one case:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u reports/toy100/accuracy_network_horizon_probe.py \
  --config reports/toy100/accuracy-failure/rotated100/config.json \
  --prior square --scale 5 --problem grid100 --horizon 1600 \
  --output /tmp/new-affine-network-horizon-run
```
