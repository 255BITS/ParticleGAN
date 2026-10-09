# Native finite movement, round four

This ready bounded diagnostic tests one substantive global trainer change:
`Recipe.finite_step_mode='armijo'` versus the exact BCAP DualNorm winner.
All actual G/prior proposals, including locally downhill ones, undergo
same-batch sufficient decrease with c=.1 at scales 1, 1/2, ..., 1/1024.
A rejected proposal takes zero parameter motion; optimizer history still
advances once. D updates and optimizer directions stay the winner's.

The [reversible saved-state probe](saved-probe.json) restores original source
753f28a5 and the complete seed0 grid100 checkpoint and named streams. It finds
first-order loss change -.129034, but actual full-step change only -.003762.
Thus the full step decreases batch loss yet fails the -.012903 Armijo requirement.
Its median finite travel is 7.607 target sigmas and diagnostic precision
.23970 → .23435; the half step gives .61550. These fixed evaluation draws,
fractions and geometry are diagnostic oracles, not trained results or a
target-informed acceptance rule. All parameters, optimizer state, buffers and
named/global RNG return to their original identities. No saved state is continued.

Armijo's [original sufficient-decrease analysis](https://msp.org/pjm/1966/16-1/pjm-v16-n1-p01-p.pdf)
motivates checking finite realization of a descent proposal. The
[stochastic line-search paper](https://arxiv.org/abs/1905.09997) studies
interpolation assumptions and same-batch line searches. Its convergence rates
do not apply automatically to this changing GAN game or normalized direction.
The finite inequality is the implementation property; distribution repair is
the empirical hypothesis, falsified by terminal precision below .97 or a failed
full sustained gate/guardrail.

Both arms use seed0, the public deterministic initializer and one whole global
winner recipe: nonsaturating BCAP cap1/coeff1/every-update, full smoothed
DualNorm .001/momentum0/per-offset, G .012/D .018/prior .030, constant floors1,
prior_reg0, clean live serving, no output noise or EMA. Task architectures,
priors, data/batch laws, schedules and gates stay fixed. Scalar adapters reuse
one real batch for D/G as in their original contracts. Constructor/data/prior/
kernel/model/evaluation RNGs are isolated and checkpointed. Acceptance replays
the existing kernel offsets, buffers and stochastic model draws without extra
sampling. Conditional/fixed two-pole components do not bind this scalar replay
and are explicitly BLOCKED for the candidate with zero spend.

Five unchanged task declarations reserve at most 6,420 seconds per arm,
12,840 total, inside the 14,400-second track ceiling. Full grid1007k, Gaussian
smoke/own-state stability and vector_two_broad guardrail complete their frozen
protocols; a failed own smoke blocks dependent stability. No sweep, second
candidate, continuation or qualification follows. Results, scorer controls,
actual-training GIFs and final recommendation will be published after completion.

```sh
PYTHONPATH=$PWD /home/martyn/dev/ParticleGAN/.venv/bin/python -u \
  reports/forge/bcap-physics/native_overshoot/round4/run.py \
  > /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/native_overshoot/logs/driver.log 2>&1
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/native_overshoot/logs/driver.log
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/native_overshoot/queue/events.jsonl
```

Scientific screening/calibration remain provisional. These diagnostic results
cannot fill ordinary Tier2 or pool with other recipes into a new winner.
