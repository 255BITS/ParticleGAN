# Cheap nonlocal signal filter before another GAN run

The local PR84 critic barrier motivates checking a nonlocal acquisition
signal before implementing another trainer controller. One fixed
Sinkhorn-drift configuration was tested in output space. **It does not pass
the distinct-particle quality proxy**, so no GAN training was launched with
this rule. This rejects this configuration as the next experiment, not the
whole family of transport methods.

The [2026 Sinkhorn-drifting paper](https://arxiv.org/html/2603.12366v1)
constructs a cross-minus-self barycentric field from balanced transport
couplings. We use squared Euclidean cost divided by2, temperature1 and
Euler step .1, values within its studied settings, with200 updates. Both
couplings converge to maximum marginal error <=1e-10 in float64. No
temperature, step, seed or iteration-budget sweep is performed.

This is a free-particle geometric diagnostic:12 equally weighted points
cover eight radius3 centers with alternating duplicate/single assignments;
the target has uniform eight-point mass. A missing-support case moves the
mode6 points to mode5. Centers are the synthetic data of this diagnostic,
not inputs to a proposed GAN update. Clean HQ is the fraction within .21 of
a center, without the host's sampling noise. Thus it is a proxy and cannot
establish production failure or success.

| Diagnostic | Initial | After200 | First loss of passing quality |
| --- | --- | --- | --- |
| Exactly coincident duplicates, eight-mode support | 8 modes/HQ1 | 8/HQ1 | None |
| Exactly coincident duplicates, seven-mode support | 7/HQ1 | 0/HQ0 | Already incomplete |
| Distinct offsets, eight-mode support | 8/HQ1 | 4/HQ .3333 | Update33 |
| Distinct offsets, seven-mode support | 7/HQ1 | 0/HQ0 | Already incomplete |

The initial coincident case cannot assess mode splitting: identical points
remain identical under this deterministic field. Its failure is retained
but is not used as acquisition evidence. A second, declared symmetry check
places the12 particles at distinct, evenly directed offsets of radius .029
(the existing output-noise scale). It keeps the same field and all numerical
settings. These offsets are fixed geometric construction, not sampled seeds
or selected successful restarts. That distinct passing cloud loses quality.
The two checks took16.15 and5.57 seconds, respectively. Both exactly matched
empirical clouds produce zero drift, with verified transport marginals.

The paper's full-domain positive-density result does not supply a finite-GAN
guarantee. Its section3.4 explicitly leaves regularized empirical
identifiability for general n>=3 open. In particular, matching the law and
passing the ring's coverage/HQ thresholds are different properties when12
equal components approximate eight equal modes. Exact rest when the laws
match is a useful property but does not settle this host's stability issue.

Next, test a sampled-data coverage signal which allows surplus particles to
share a covered region, rather than enforcing balanced mass transport. Such
a rule adds an objective or constraint and must be labeled accordingly.
It must pass the original warm and cold host gates before any longer hold.

Source, full200-update traces, both original declarations in the source
headers, and stored/raw SHA256 values are in the
[evidence manifest](continuous-evidence/sinkhorn-fixed-cloud/manifest.json).

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/pr38-default-env/bin/python reports/toy100/sinkhorn_fixed_cloud_filter.py \
  --distinct --output NEW.json
```
