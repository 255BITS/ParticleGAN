# Full DDGAN public-API toys

The frozen proposal adds an actual adversarial diffusion transition game and
four-step ancestral sampling. It does not reuse the direct clean-posterior toy
as a substitute for a DDGAN chain. Full training results and actual-training
GIFs will be recorded after the common source freeze; no trained-quality PASS
is claimed by the implementation or its oracle controls.

The original [100-Gaussian denoising study](../../denoising-toy/README.md) retains
its own source, initialization, budgets, seeds and reported quality. These new
small targets preserve its useful transition-calibration and independent-noise
questions. They neither rerun nor requalify those historical runs. The
[public-API toy contract](../../toy_audit/api_contract/README.md) requires a
numerical verdict and actual observed training media for each completed toy.

## Proposal and frozen protocol

The root agent approved both bounded tasks before training. One global trainer
is selected: `bispf-optimized-quintic-v1`, including its declared DualNorm
optimizer, Born–Infeld critic penalty, learned prior and SBNP regularizer. The
trainer delta is **none**. Task adaptation sets the DDGAN model, scalar
time/observation-conditioned critic, no encoder, diffusion schedule and host
resources; it does not change optimizer, prior update or prior regularizer.
Unsupported adaptive row/serving/noise/transport and strict-progress policies
fail closed rather than silently losing their mechanisms.

| Task | Target | Budget | Intended tier |
| --- | --- | --- | --- |
| `model_ddgan_gaussian_chain` | Gaussian mean `(0.6,-0.35)`, diagonal standard deviation `(0.45,0.30)` | 2,500 updates | Tier 1, basic transition and chain functionality |
| `model_ddgan_bimodal_chain` | Equal mixture at `(-0.75,0)` and `(0.75,0)`, isotropic standard deviation `0.18` | 5,000 updates | Tier 2, multimodal conditional uncertainty and local spread |

The multimodal target is a separately declared harder claim, with no checkpoint
dependency. Failures remain visible in the full suite; final required/held
placement follows Forge's existing disposition rule and the shared recipe's
measured evidence. A failed gate does not authorize tuning its bound or recipe.

Both tasks use protocol seed `0`, batch `64`, `128` initial MoG locations in
four latent dimensions, absolute fixed kernel standard deviation `0.25`, and
no read standardization. G and D are two-hidden-layer width-32 LeakyReLU MLPs.
The public deterministic initializer uses component offsets G `0`, D `1`, prior
`2`; its prior table follows the public R2 initializer. No analytic target
weights or privileged posterior samples initialize or train G.

The public `DDGAN` schedule is `[1,0.9,0.5,0.05,0.0001]`. Training draws a fresh
clean target and independent uniform timestep for each example and each D/G
batch, then obtains real `(x_prev,xt)` from `DDGAN.forward_pair`. G predicts
clean `x0`; `DDGAN.reverse` supplies its actual transition sample. The critic
observes `(x_prev,xt,t/4)`; its gradient penalty differentiates the candidate
`x_prev` while holding context fixed. Recipe-owned prior regularization is
included once, and the public protected-objective backward binds G's actual
adversarial objective. The full public `UpdatePolicy` lifecycle owns schedule
boundaries, observations, update hooks, averages and completed update count;
its complete checkpoint is retained. Every consumed constructor/data/time/latent/kernel/
forward/reverse/penalty stream is named, isolated and checkpointed. Checkpoints
also retain models, optimizers, gradients, modes and a digest of both seen
clean-data/timestep sequences. Observation uses independent local streams.

Final sampling starts from `N(0,I)` and performs all four reverse steps. Every
step independently redraws latent indices, MoG jitter and Gaussian reverse
noise. An exact-target-terminal start is not supplied to the learned model.
Evaluation separately holds out eight fixed transition panels: both
`xt=(-0.5,0.25)` and `(0.5,-0.25)` at each of steps 1, 2, 3 and 4.

## Predeclared gates

Evaluation uses `2,048` draws per panel and for the final distribution, 24 fixed
post-update checks, and five passing terminal checks. Each of the eight
conditional reverse panels **and** the final chain must satisfy:

- Maximum analytic CDF error over eight fixed projections: `<=0.12`.
- Mean error after exact target-covariance whitening: `<=0.25`.
- Minimum and maximum whitened covariance eigenvalues: `>=0.65` and `<=1.40`.

The bimodal final distribution additionally requires three-sigma quality
`>=0.95`, mode-mass total variation `<=0.10`, worst per-mode mean error
`<=0.50` within-mode standard deviations, and local covariance eigenvalues
within `[0.50,1.50]`. Nonfinite output fails explicitly. The exact complete
metric/operator/bound lists are exported as `case.forge_thresholds` and stored
unchanged in the corresponding Forge task declarations.

## Software and scorer validation

`tests/test_toy_api_model_ddgan.py`: **16 passed**. Exact target and exact
conditional reverse oracle samples pass at the actual `2,048`-sample gate
count. Independent Gaussian precision algebra checks the posterior formula.
The exact conditional reverse chain also recovers each final target starting
from ordinary terminal Gaussian noise, bounding this task's terminal-start
approximation separately from learned quality.

Destructive controls reject posterior-mean collapse, collapsed mixture centers,
one-mode collapse, unchanged standard Gaussian output, nonfinite output, correct
unconditional samples at a fixed conditional query, and an oracle answering the
wrong observation. Full-chain instrumentation verifies all four G/reverse
calls and new latent IDs/jitter/reverse noise at every step. Public-component
updates change the learned prior, leave a frozen-prior recipe unchanged,
preserve caller RNG state, exercise the ordered public policy lifecycle and
critic diagnostics, survive exact checkpoint/restart replay, and preserve
all training state during observation. An undersized observation explicitly
fails the default evaluation-count requirement.

Reproduction from the frozen shared implementation uses the normal toy runner:

```sh
python -m benchmarks.toy_audit.api_run --case api-model-ddgan-gaussian-chain \
  --device cpu --output /nvme/dev/particlegan/artifacts/api-model-toys/ddgan/gaussian
python -m benchmarks.toy_audit.api_run --case api-model-ddgan-bimodal-chain \
  --device cpu --output /nvme/dev/particlegan/artifacts/api-model-toys/ddgan/bimodal
```

Use a fresh exclusive output directory for each run, redirect stdout to a
local `.log`, and `tail -f` that log. Raw logs, observation arrays and checkpoints
stay in the artifact directory. Publish compact final metrics and real training
GIFs with the single shared current leaderboard for this goal; this report does
not create a second leaderboard.
