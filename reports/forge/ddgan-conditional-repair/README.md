# Conditional DDGAN repair with the selected BISPF recipe

This bounded research tests an auxiliary conditional energy score on the plain
DDGAN Gaussian and bimodal chains. The baseline is exactly
`bispf-optimized-quintic-v1`, selected by the technique inventory. Its historical
CPU Gaussian FAIL and blocked bimodal full run remain unchanged. A new matched
GPU0 baseline supplies the comparison on this execution cohort. GPU0 is shared
and under load; elapsed cost is recorded without any throughput comparison.

The [saved-state diagnosis](../ddgan-failure-investigation/README.md) found
low-noise conditional mismatches at central and fresh contexts, while high-noise
reverse kernels masked poor clean predictions. Endpoint critic gradients opposed
mean correction on seven of eight fixed panels; these local gradients do not
establish optimizer-step direction or critic lag as a cause.

The [DDGAN paper](https://arxiv.org/abs/2112.07804) and
[official implementation](https://github.com/NVlabs/denoising-diffusion-gan/blob/main/train_ddgan.py)
model each reverse transition using a stochastic conditional generator and the
Gaussian posterior kernel. That core is retained. The auxiliary is the negative
beta1 [energy score, equation 22](https://doi.org/10.1198/016214506000001437):
`mean ||X-y|| - sum(i != j)||Xi-Xj|| / (2*K*(K-1))`, with eight independent
samples per existing clean/context pair and weight1. Its repulsion preserves
conditional spread and its off-diagonal U-statistic avoids the finite-ensemble
underdispersion bias. Training uses only the existing forward-paired clean x0;
no analytic posterior, target weights or evaluation query enters the objective.

The public `conditional_energy_score` implements that objective. The existing
public DDGAN fixture, Recipe factories, learned MoG, SBNP, Born–Infeld critic
penalty, DualNorm and ordered UpdatePolicy still own actual updates. The GAN
objective retains coefficient1; the existing backward API receives both
objectives as protected losses where supported, without a nonascent guarantee. New energy latent/kernel streams are independent and checkpointed;
existing data, timestep and adversarial streams are unchanged. Unsupported hosts
refuse an active energy recipe before reservation or construction.

Both recipes share protocol seed0, the public deterministic initializer, the
2x32 G/D MLPs, target laws, initial128-location four-dimensional MoG with sigma.25,
batch64, independent D/G clean/timestep sequences, all four reverse sampling
steps, evaluation2048 per panel, 24 scheduled checks and five passing terminal
checks. Original numerical bounds remain exact. Gaussian trains2500 updates and
bimodal5000. Separate diagnostic task identities remove ordinary tier dependency
only for this explicitly scoped research; their results grant zero ordinary
qualification and cannot fill the original blocked cells.

Round1 is frozen by two ready Forge studies, `ddgan-repair-baseline-r1` and
`ddgan-repair-energy-r1`, in one2400-second reservation campaign (600 seconds per
full task). If the energy recipe fails, at most two further substantively
justified global recipe revisions may run both unchanged tasks. Maximum scope:
8 full runs,30000 generator updates and4800 reserved seconds. A stopped round is
never continued, seed-repeated or retried for scientific failure. All failures
remain visible. A changed task or capacity requires an explicit new cohort.

Scorer controls compare oracle draws with collapsed means, wrong means and
wrong spread; unchanged DDGAN conditional scorer controls reject wrong-context
and unconditional samples. Replay tests check the exact energy streams,
initial-model equality, target-batch equality, observation purity and restored
checkpoint equality. Actual training GIFs and compact final metrics will be
published after execution. This goal has one current generated leaderboard.

Raw logs/checkpoints/arrays stay outside Git. Tail progress with:

```sh
tail -F /nvme/dev/particlegan/artifacts/ddgan-bispf-20261019/progress.log
```
