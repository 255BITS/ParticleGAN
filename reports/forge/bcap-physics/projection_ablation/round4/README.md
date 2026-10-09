# Projection repair: direction versus finite acceptance

This ready bounded diagnostic isolates three predeclared arms on one common source: corrected schema2 `nonascent`, the identical strict common-descent `direction_blend` at full scale without finite probes, and the measured `strict_progress` package with finite Armijo acceptance. It admits no fourth arm, sweep, seed repeat or gate changes. Full results will be published after all admitted jobs finish.

The primary question is which smallest mechanism retains BOTH trajectory and residual full sustained identity passes while preserving the mid-scale passing guardrail. A residual endpoint MSE forecast <= .02 is registered, but every original bound and required suffix decides scientific success. The finite-only boundary arm is degenerate because its protected derivative is zero and cannot meet strict acceptance.

The direction is unchanged from [PR361](https://github.com/255BITS/ParticleGAN/pull/361): for an actual conflicting base displacement d and normalized existing protected gradients n_j, q = -||d|| mean(n_j)/||mean(n_j)|| and p = (project_nonascent(d,a)+q)/2. The direction-only arm always applies p at scale1; opposed gradients restore the pre-step tensors. The finite arm tries scale1 through 1/256, accepting rounded strict derivatives and same-batch Armijo decrease with coefficient1e-4. The base optimizer clocks advance once. Inactive steps preserve updated tensors bitwise. Protected losses, unchanged host loss coefficients and target information are unchanged; no identity supervision is introduced.

For at most two nonzero unit normals, their mean is the minimum-norm convex combination. Its negative is common descent unless the normals oppose. This is a local directional property, consistent with [MGDA](https://mgda.inria.fr/mgda) and [Sener and Koltun](https://arxiv.org/abs/1810.04650), not a stochastic GAN convergence guarantee. Strict derivatives do not guarantee finite decrease; changing-batch finite decrease does not guarantee distribution fidelity.

Read-only [saved endpoint attribution](saved-attribution.json) restores six certified states: original winner and both PR361 arms for trajectory/residual. It includes adversarial, set coverage, latent L2/spread and the existing paired residual on the actual both-land mask. It transforms the summed gradients with the saved full-DualNorm rates/smoothing before attribution, and also reports normalized components and leave-one-out proposals. These CPU FP64 derivatives/public FP32 polar probes are a separate diagnostic cohort; they consume zero training steps or samples and do not prove an actual rounded step or unique causal culprit. [Reproduce](probe_saved.py).

All arms use the exact saved winning BCAP settings: nonsaturating, full DualNorm smoothing .001/momentum0/per_offset, G/E .012, D .018, prior .030, constant floors1, cap/coefficient1 each update, zero additive training output noise and clean/live scoring. `get_recipe('bcap')` alone is not this winner. The only trainer delta is `constraint_geometry_mode`.

The six unchanged tasks are trajectory400, residual400, mid-scale800, two-pole80, Gaussian smoke1000 and own-checkpoint Gaussian stability+5000. Each arm reserves6420 seconds; the campaign reserves19260 within the21600-second ceiling. Failed own smoke blocks stability with zero spend. Gaussian/vector adapters retain their one-real-tensor D/G reuse; the fixed two-pole identity/zero/stored-weight fixture remains separate. Protocol seed0, public deterministic initialization, task architectures/priors/laws/seen batches/update budgets/scoring cadence and named checkpointed RNG streams stay fixed. Full numerical scorer controls and saved actual-training GIFs will accompany the report. The focused software/protocol suite passes231 checks, including CPU/CUDA float32/64 parity, public factory, checkpoint resume and nonlinear overshoot without evaluator calls. This is a research diagnostic; profiles remain provisional and no default adoption or ordinary qualification follows.

Tail execution:

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/projection_ablation/logs/driver.log
```

Source and ready declarations are pushed before admission. Public Queue/drain uses one shared GPU0 worker and no full-compile callback. Bulk logs, JSONL, checkpoints and tensors stay outside Git. The parent owns the single current goal leaderboard; this report will contain a scoped arm-comparison table only.
