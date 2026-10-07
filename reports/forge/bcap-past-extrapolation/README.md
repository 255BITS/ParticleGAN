# BCAP extrapolation from the past

Prospective, bounded GPU diagnostic based on [Gidel et al., §3.3](https://arxiv.org/pdf/1802.10551).
The [protocol](protocol.json) freezes acquisition, stationary retention and a
separate scalar target shift. This study uses the selected BCAP configuration
from the [existing leaderboard](../technique-inventory.md); it supplies no ordinary
qualification or default promotion.

`Recipe.game_update="extrapolation_from_past"` uses a joint G/D/prior lookahead
with the previous normalized update field, evaluates one fresh gradient per
role there, then restores the base before correction. The first lookahead is
zero. Previous directions, sampled-row masks embodied in those directions,
optimizer histories and named RNG streams travel with the checkpoint.
`game_update="simultaneous"` is the joint-gradient control; the default remains
`"alternating"`. Only stateless dualnorm or SGD on scalar GANTrainer hosts is
supported. BatchNorm, optimizer momentum, averaging and controller combinations
are rejected. This is a normalized-field adaptation of equations20–21, not
ExtraAdam; the paper's smooth/monotone convergence hypotheses are not assumed
for normalized neural BCAP.

All arms keep the same task architecture, learned256-location uniform MoG
sigma.1, batch128, deterministic public initialization, seed0, exact real batch
sequence, clean live4096-sample evaluation law and cadence. Constant rates are
G.012/D.018/prior.03. Gaussian retains z2, width32/depth2 and Fouriercritic2;
ring retains z4, width64/depth2. Original recipe horizons stay1000/400, independent
of the external4000 cap. The diagnostic sigma.1 Gaussian remains separate from
the ordinary sigma.025 task.

There are four new4000-update stationary trials and three2000-update scalar
shift continuations, with3300 reserved seconds, no retries, tuning or seed changes.
The two completed alternating stationary baselines retain original source and
zero new cost. After4000, Gaussian mean shifts2→3 with sigma.5 unchanged; active
learning and an independent frozen checkpoint copy receive matched GPU evaluation
draws. No cache or optimizer history is reset. Passing this finite diagnostic
would not establish unbounded continuous learning.

Reproduction sources: [public mechanism](../../../particlegan/extrapolation.py)
and [driver](../../../benchmarks/toy_audit/bcap_past_extrapolation.py). The driver
checks frozen inputs before spend and refuses CPU training. Neural training,
sampling and software fixtures use CUDA; the original CPU target law, scorer,
metadata checks and saved-sample rendering retain their existing numerical law.
Raw logs, curves, samples, checkpoints and complete source maps stay outside Git.

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /usr/bin/python -u -m benchmarks.toy_audit.bcap_past_extrapolation run \
  --device cuda:0 --output runs/api/bcap-past-extrapolation-v1 \
  > runs/api/bcap-past-extrapolation.run.log 2>&1
tail -F runs/api/bcap-past-extrapolation.run.log
```

Preflight:14 CUDA/metadata software checks pass. Independent frozen scorer
controls reject collapse, shift, wrong width and mode atoms. Numerical results
and actual-training GIFs will be published after the single declared round.
