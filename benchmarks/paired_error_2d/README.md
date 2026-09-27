# Paired 2D transport without output MSE

Learn a deterministic map from a square to its affine transform or radial swirl.
This asks whether a paired-error adversarial objective preserves **which target
belongs to which input**, not merely the target point cloud.

For a source point `x`, its known target `y=f(x)`, and a routed particle student
`G(x)`, let `T` standardize by the training targets. The critic sees:

```text
real = n
fake = (n + T(G(x))) - T(y)
n ~ Normal(0, sigma(step)^2 I)
```

The same noise is used within a pair. There is no output MSE, feature matching,
coverage or reconstruction loss in training. MSE only measures paired accuracy.
A shuffled target cloud has the same marginal samples but scores poorly.

The student is `x + MLP(concat(x, softmax(q(x) P^T / sqrt(4)) P))` with one
128×4 particle table `P`, the recipe's particle prior. Inference is an ordinary
forward pass. The fixed-cloud control uses the same initial table and routing;
only table updates are disabled.

The two maps are:

- `affine2`: `x @ [[.8,-.6],[.6,.8]] + [.2,-.3]`.
- `swirl2`: rotate by `1.7 * ||x/sqrt(3)||²`, preserving radius.

`task.py` defines only the problem: data, networks, the paired-error sample,
metrics and verdict. Training (optimizers, schedule, loss, critic penalty, noise,
EMA, logging) is the shipped recipe on `benchmarks.toy_runner`. A problem passes
when validation NMSE ≤ .01 and p95 paired distance ≤ .2; the runner's `hold`
summary reports from which observation it stays passing.

```bash
CUDA_VISIBLE_DEVICES= python -u -m benchmarks.paired_error_2d
tail -f runs/toy-refactor/paired_error_2d_swirl2_movable.log
```

This runs the four (map, cloud) problems in turn, one JSON log line per observation.

See [source provenance](SOURCE.md). The [historical results](../../reports/paired_error_2d/README.md)
come from the pre-runner harness (hand-built Adam, cap/cosine arms), so they are
not reproduced exactly by this version.
