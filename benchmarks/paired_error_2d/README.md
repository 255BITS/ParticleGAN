# Paired 2D transport without output MSE

Learn a deterministic map from a square to its affine transform or radial swirl.
This asks whether a paired-error adversarial objective preserves **which target
belongs to which input**, not merely the target point cloud.

For a source point `x`, its known target `y=f(x)`, and a routed particle student
`G(x)`, let `T` use the training targets' per-coordinate mean and sample standard
deviation (floor `1e-4`). The critic sees:

```text
real = n
fake = (n + T(G(x))) - T(y)
n ~ Normal(0, sigma(step)^2 I)
D loss = mean softplus(D(fake) - D(real)) + b_cap
G loss = mean softplus(D(real) - D(fake)) + particle VIC
```

The same noise is used within a pair; D and G draw separate batches. There is no
output MSE, feature matching, coverage or reconstruction loss in training. MSE is
used to measure paired accuracy and select checkpoints. A shuffled target cloud
has the same marginal samples but scores poorly on paired accuracy.

The generator is `x + MLP(concat(x, softmax(q(x) P^T / sqrt(4)) P))` with one
128×4 cloud. Inference is an ordinary forward pass. The fixed-cloud control uses
the same initial particle values and routing; only particle updates are disabled.

The two maps are:

- `affine2`: `x @ [[.8,-.6],[.6,.8]] + [.2,-.3]`.
- `swirl2`: rotate by `1.7 * ||x/sqrt(3)||²`, preserving radius.

All runs use seed 0, 6,000 updates, batch 64, 1,024 training and 1,024 validation
points, Adam `(0,.999)` and EMA `.995`. The baseline cap is threshold/coefficient
1/1. Candidates use 1.25/3, learning rates ×.85, and cosine from 60% to a 5% floor;
the third arm separately reduces particle VIC from 1 to .05. The cap remains lazy
every four steps with ×4 compensation. Noise follows the source's 8,000-step
anneal/hold rule, not a rescaled 6,000-step schedule. Full settings are in `task.py`.

Run from a checkout of this PR with a PyTorch environment:

```bash
mkdir -p artifacts
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u -m benchmarks.paired_error_2d \
  --output artifacts/paired_error_2d --workers 4 > artifacts/paired_error_2d.log 2>&1
tail -f artifacts/paired_error_2d.log
```

To compare with the original local model-glue run, additionally pass
`--reference-artifacts /path/to/model-glue/artifacts/particle-cap-comparison-20260922`.
The experiment itself needs no application checkout, external weights or data.
`README.md`, `results.json` and `reference-audit.json` in the output directory
contain the leaderboard and comparison. Checkpoints and validation point samples
are retained under `runs/`; logs include every observed checkpoint. Resume rejects
source, runtime or protocol changes.

Every checkpoint choice is frozen before test evaluation. The historical test
set is intentionally reused for reproduction, with no new-holdout claim. Live
weights are reported separately at the same selected step. A stable pass requires
NMSE ≤.01 and p95 distance ≤.2 at the final three validation observations. No
single toy or fixed seed establishes a native image-model or LM improvement.

See [source provenance](SOURCE.md) and [measured results](../../reports/paired_error_2d/README.md).
