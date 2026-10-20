# Incomplete-data MisGAN and a learned multimodal neural imputer

This is an explicitly scoped four-mode research diagnostic using the selected
`bispf-optimized-quintic-v1` recipe. It grants no ordinary qualification and no
PASS on the unchanged historical 100-mode recovery/imputer tasks.

The legacy public fixtures used plain Adam and a scalar recipe loss, omitted
the Born-Infeld critic penalty, never optimized the prior or applied SBNP, and
omitted consumed RNG streams from their checkpoints. Their published failures
retain their original source and execution identity; they are not selected
BISPF measurements. The new host uses the public recipe optimizers, all three
Born-Infeld games, learned prior/SBNP, direction-blend backward and UpdatePolicy.

The [frozen proposal](proposal.json) declares the target, architecture, bounds,
three initial arms and finite budget before full training. Four noisy corners
are lifted to eight sensors. The historical block-mask patterns and their
40/30/20/10% probabilities remain. A single sensor sees only the sum of the
two coordinates: its central query has two equiprobable complete-data branches.
This retains an identifiable incomplete-data recovery problem and genuine
multimodal imputation while explicitly changing target size and architecture.

Training receives only observed entries and masks. The complete training
matrix is discarded when constructing the observed pool. Auxiliary round one
adds uniform masked-mixture likelihood and observed-mask categorical likelihood.
Round two additionally trains neural conditional component logits using labels
from its own complete generator. At serving, the neural imputer receives only
the observed entries and mask, samples its predicted categorical probabilities,
and decodes independent prior noise. It never computes component distances,
Bayesian particle resampling or analytic posterior probabilities. The
[MisGAN paper](https://arxiv.org/abs/1902.09599) and
[official imputation implementation](https://github.com/steveli/misgan/blob/master/src/misgan_impute.py)
motivate the independent data, mask and imputer games; these auxiliaries and
architectures are declared extensions, not an exact paper reproduction.

All arms keep seed 0, public deterministic initialization, the same architecture,
MoG64 R2 initial distribution and fixed width .025, observed-data batches,
sampling, 1200 updates and 24 scoring checks. Five terminal checks must pass all
bounds, including mode/mass/manifold recovery, mask law, observed identity,
held-out imputation accuracy and ambiguous branch quality/mass/coverage/spread.
Destructive controls reject conditional averaging, one draw reused across
latents, wrong branches, incorrect mass, generator collapse and mask collapse.

Run through the frozen research view and ready studies:

```sh
python -m experiments.forge enqueue bispf-optimized-quintic-v1 --study misgan-bispf-baseline-r1-reviewed
python -m experiments.forge enqueue bispf-misgan-observed-v1 --study misgan-bispf-observed-r1-reviewed
python -m experiments.forge enqueue bispf-misgan-amortized-v1 --study misgan-bispf-amortized-r1-reviewed
python -m experiments.forge drain --gpus 0 --workers-per-gpu 3 --allow-sharing --campaign misgan-bispf-four-mode-r1
```

Local raw logs and state archives are under
`/nvme/dev/particlegan/artifacts/misgan-bispf-20261019`; tail `progress.log`.
GPU 0 is shared and elapsed costs are not speed benchmarks.

Three initial unexecuted requests were cancelled before spend for a review repair: retain the fixture RNG state, modes and gradients, honor prior ownership, and match the imputer training/serving MoG kernel. Replacement `-reviewed` studies preserve those administrative identities.
