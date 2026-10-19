# Full DDGAN public-API toys

The Gaussian task completed its frozen 2,500-update budget and **FAILS trained
distribution quality**. Its public-API transition game, four-step sampling,
checkpoints and numerical observations execute successfully. The bimodal full
run is **BLOCKED** by the declared basic-task gates; its separately budgeted
three-update software prefix is **INCOMPLETE** against the 5,000-update task.
Neither result establishes a DDGAN quality pass.

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

The multimodal target is a separately declared harder claim, with a Gaussian
gate dependency and no checkpoint dependency. The campaign also requires all
six basic tasks to pass before either full Tier 2 run. All six completed:
three passed and three failed. Initial Tier 1/Tier 2 placements and numerical
gates are unchanged; failure-driven re-tiering, tuning and reruns were not used.

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

## Frozen research results

All receipts bind source
[`bee9f06ed21695e518c1cb702e12a385ab9389e8`](https://github.com/255BITS/ParticleGAN/tree/bee9f06ed21695e518c1cb702e12a385ab9389e8),
global recipe `bispf-optimized-quintic-v1`, protocol seed 0, public deterministic
initialization, clean live serving and CPU execution with one Torch thread.
The [shared protocol](protocol.json) records the exact task and recipe binding;
the [compact publication](../../toy_audit/api_contract/model_modes/publication.json)
retains the observed metrics and original artifact identities. Source remained
unchanged throughout both executions. A later report-only export repair read
the saved receipts and GIFs; it changed no training, sampling or observations.

**Gaussian: FAIL**, 2,500/2,500 updates, 24 post-update observations, 2,048 draws
per panel, 18.97200583400263 seconds. All updates and final metrics were finite.
No post-update observation passed all bounds (0/24); all final five failed.
The endpoint values below are copied from the saved receipt, with conditional
extrema taken across the eight endpoint panels.

| Endpoint metric | Observed | Gate |
| --- | ---: | ---: |
| Final-chain projection CDF error | 0.2022633460624323 | <=0.12 |
| Final-chain normalized mean error | 0.6056332609240245 | <=0.25 |
| Final-chain minimum covariance eigenvalue | 0.5499189424785275 | >=0.65 |
| Final-chain maximum covariance eigenvalue | 1.398722490869161 | <=1.40 |
| Worst transition projection CDF, `t1_q0` | 0.42795975312463663 | <=0.12 |
| Worst transition normalized mean, `t1_q0` | 1.1555381742511794 | <=0.25 |
| Minimum transition covariance eigenvalue, `t2_q1` | 0.6072135984759013 | >=0.65 |
| Maximum transition covariance eigenvalue, `t1_q1` | 2.4374239160286817 | <=1.40 |

The exact endpoint failed metric bounds are
`final_covariance_eigen_min`, `final_normalized_mean`, `final_projection_cdf`,
`t1_q0_covariance_eigen_max`, `t1_q0_normalized_mean`, `t1_q0_projection_cdf`,
`t1_q1_covariance_eigen_max`, `t1_q1_normalized_mean`, `t1_q1_projection_cdf`,
`t2_q0_normalized_mean`, `t2_q0_projection_cdf`, `t2_q1_covariance_eigen_min`,
`t2_q1_normalized_mean`, and `t2_q1_projection_cdf`. The full verdict also fails
the requirement that all last five post-update observations pass.

[Actual Gaussian training GIF](../../toy_audit/api_contract/model_modes/media/api-model-ddgan-gaussian-chain.gif)
shows nine real observed states. The actual four reverse steps are executed at
every observation; media does not interpolate a successful chain.

**Bimodal: full 5,000-update run BLOCKED; prefix INCOMPLETE**, 3/5,000 updates,
three post-update checks, 2,048 draws per panel, 0.5842560800083447 seconds.
The prefix verifies finite execution and media only. Its endpoint final CDF
error is `0.49965300603990986`, normalized mean `0.3052264056419862`, covariance
eigenvalues `[0.02102167678704736,0.08743028294723189]`, HQ
`0.1552734375`, mode-mass TV `0.259765625`, local mean error
`3.914991759583327` sigma and local covariance eigenvalues
`[0.023697648270426128,0.31880969013331273]`. Worst conditional CDF is
`0.995692693706883`, worst normalized mean `4.55162606927346`, minimum
covariance eigenvalue `0.09741582561445582`, and maximum `1.0580691114691039`.

Prefix failed bounds are `final_covariance_eigen_min`, `final_hq`,
`final_local_covariance_min`, `final_local_mean_sigma`, `final_mode_mass_tv`,
`final_normalized_mean`, `final_projection_cdf`; `covariance_eigen_min`,
`normalized_mean` and `projection_cdf` for every panel in
`{t1_q0,t1_q1,t2_q0,t2_q1}`; and `covariance_eigen_min` for both
`{t3_q0,t3_q1}`. Its raw API verdict is FAIL because the full budget/cadence
and five terminal passing observations were not completed. The publication
labels this **INCOMPLETE**, not a measured 5,000-update quality failure.
[Actual three-update bimodal prefix GIF](../../toy_audit/api_contract/model_modes/media/api-model-ddgan-bimodal-chain.gif)
contains four saved states including initialization.

The completed Gaussian endpoint passes all four high-noise panels (`t3` and
`t4`), while the lower-noise panels and final marginal fail. This localizes the
observed weakness to the clean end of the chain; it does not identify its cause.
Before proposing another bounded trainer hypothesis, inspect saved low-noise
transition predictions, conditional residual means/covariances and critic
gradients against the exact posterior. No new training, continuation, tuning or
promotion is authorized by this readout. Results remain in the single existing
[Forge leaderboard](../technique-inventory.md).

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

The completed [saved-state investigation](../ddgan-failure-investigation/README.md)
finds low-noise conditional errors beyond the frozen queries. Step-4 reverse
noise supplies about 99% of the exact transition variance, allowing a poor clean
predictor to pass that later transition panel. The saved critic's local output
gradient opposes needed mean correction on 7/8 vanilla fixed panels; this does
not establish the actual optimizer-step direction or a temporal cause. The
original generation failure and blocked deeper run remain unchanged.
