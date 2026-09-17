# Prespecified local routing comparison

Chosen before launching either new arm, after committing the seven-arm scout
as `40a55a7`. No seed or hyperparameter sweep.

Both `route_local` and `route_local_balanced` retain bounded offsets and exact
global nearest-particle forward selection. Replace only the routing surrogate:

```text
E(X) -> (query, u)
k = nearest particle to query
z = p[k] + sigma * 3*tanh(u/3)
G(z) -> X_hat

Backward routing:
neighbors = eight nearest particles to query
d_i = squared distance to neighbor i, sorted ascending
t = stop_gradient(max(d_8 - d_1, 0.000001))
w = softmax(-(d_i - d_1) / t) over those eight neighbors
proxy = sum(w_i * stop_gradient(p[i]))
center = p[k] + proxy - stop_gradient(proxy)
```

Membership is discrete and bandwidth detached. The floor is in squared latent
units and protects coincident neighbors. The eighth neighbor has at least
exp(-1) times the first neighbor's unnormalized weight, so this is a spatially
local surrogate, not a nearly one-hot softmax. The old global temperature 0.25
remains in shared configurations but is unused by these two local arms. This
changes both support and bandwidth; it does not isolate either change alone.

`route_local_balanced` additionally uses the existing weight-0.01 hard batch
usage loss, differentiated through these same local weights. No other loss or
optimizer changes. Seed 24002, same initialization and RNG streams, 6,000 updates,
batch 256, fixed sigma, 100k final unconditional samples, 8,192 reconstructions.

Compare local against the original bounded arm, and local-balanced against
local (primary balancing comparison). Audit hard/soft frequencies on the same
100k real examples. Require lower hard-usage TV and higher effective hard usage
before describing balancing as successful. Report generation, reconstruction,
width, offset ablation, and runtime regardless of the outcome. Final checkpoints
only; no selection using intermediate results.

The estimator is still biased and ignores neighbor-boundary derivatives.
Locality may reduce the ability to move examples into distant unused particles;
the detached adaptive bandwidth can also change gradient magnitudes.
