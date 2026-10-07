# Combined magnitude response at constant rates

Predeclared study `gaussian-combined-magnitude-v1` composes the fixed G/D spectral
scale `.1` from PR320 with the learned-prior row scale `.001` from PR319. The
primary arm uses extrapolation from the past; alternating and simultaneous are
explicit timing controls. This tests the interaction recommended by the
[completed independent round](../gaussian-response-round/README.md).

The [protocol](protocol.json) freezes all gates, inputs, source hashes, predictions
and a finite nine-trial reservation before training: six stationary trials at
4,000 updates and three Gaussian continuations at 2,000 additional updates;
30,000 new updates and 4,500 reserved loop seconds, zero retries. Scales remain
fixed across tasks, roles and time. No tuning or extension belongs to this study.

Use public `GANTrainer`, seed0, the public deterministic initializer, batch128
and the identical initial model tensors and named random streams. Both tasks
use256 learned uniform MoG locations, sigma.1, without standardization.
Gaussian remains N(2,.5²), z2, width32/depth2, Fourier critic2. Ring retains its
z4, width64/depth2 and original16-cluster law. The ordinary sigma.025 Gaussian
task is a separate cohort; this diagnostic supplies no ordinary qualification.

Rates remain G.012, D.012×1.5, prior.012×2.5. Acquisition requires five terminal
full passes at1,000 Gaussian/1,600 ring updates. Every scheduled later check
through4,000 must pass:72 Gaussian/144 ring. At4,000 the Gaussian mean changes
from2 to3 with sigma.5 unchanged. Continue to6,000 without resetting optimizer
or extrapolation history; reacquire by5,000 and retain every24 later check.
Each shifted arm also evaluates a matched no-update copy of its4,000 checkpoint.
Late windows and good endpoints cannot replace acquisition or strict retention.

CUDA performs model training, sampling, mechanism fixtures and checkpoint
restores. Original CPU target generation, stored-output numerical scoring and
media rendering preserve the previous explicit protocol exceptions. Match the
archived real-batch digests and checkpoint every consumed named stream.

[controls.json](controls.json) retains the original normalized, prior-only and
network-only compact evidence identities. They cost no new training and receive
no requalification. The existing single Forge qualification leaderboard remains
the publication for that goal.

Reproduction after the frozen scientific commit is recorded in provenance:

```sh
mkdir -p runs/api
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 /usr/bin/python -u \
  -m benchmarks.toy_audit.gaussian_combined_magnitude run \
  --output runs/api/gaussian-combined-magnitude-v1 --device cuda:0 \
  > runs/api/gaussian-combined-magnitude.run.log 2>&1
tail -f runs/api/gaussian-combined-magnitude.run.log
```

Per-trial `.log` files inside the output directory emit scheduled numerical
checks and are also directly tail-able. Bulk logs, curves, samples and state
dumps stay outside Git; the final report publishes compact evidence and actual
training GIFs. This branch targets develop and depends on PR317; it integrates
the cap capabilities from PR319/320 without merging their independent branches
or revising their archived protocols.
