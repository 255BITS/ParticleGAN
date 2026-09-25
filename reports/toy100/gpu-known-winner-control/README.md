# Previously passing 22-toy recipe: CPU and GPU control

**The recorded CPU winner remains 22/22 PASS. Its preserved recipe scores 16/22
on the CUDA profile, including PASS on all three native 100-mode problems.**

The first GPU report evaluated eight continuous-learning variants and omitted
this baseline. That cohort did not establish that the earlier 22/22 result had
vanished. This control restores the missing comparison.

| Recipe / evaluation | Older toys | Native 100-mode toys | Total |
|---|---:|---:|---:|
| Original recorded CPU recipe, independently regraded | 19/19 | 3/3 | **22/22 PASS** |
| Same recipe, CUDA training | 13/19 | 3/3 | **16/22 FAIL** |

GPU failures: `mode_hold`, `trajectory`, `img_bars4`, `img_blobs4`,
`img_intensity2`, and `vector_unequal_mass`. The GPU cold ring ends with seven
modes at HQ .997802734375. Every supported run completed its full fixed budget;
all 22 saved GPU verdicts were independently recomputed. No CPU training was
rerun: the CPU check regraded the already retained evidence.

## What differs from the continuous-learning variants

This is `constraints_simple_regularization.json`: ordinary Adam with G/D rates
.00425, particle rate .0085, betas (0,.999), discriminator b_cap with unit
threshold/coefficient, and no global particle regularizer. It retains cosine
learning-rate decay, annealed input noise, and output noise .029. The frozen
autoencoder and unused-token hosts retain their auxiliary reconstruction,
coverage, hold, and particle terms. These are GAN hosts with auxiliary losses,
not a claim of exclusively adversarial training in every host.

The recent variants changed the optimizer/noise mechanisms and kept learning
rates active. H and its optimizer descendants also disabled those auxiliary
host terms. Their lower toy totals therefore are not a same-recipe CPU/GPU
comparison. Neither the original CPU 22/22 nor this CUDA finite-budget control
establishes continual stability with non-decaying rates.

## Reproduction and evidence

The original config SHA-256 is
`4af9863a319378b362bfb925b161d9ae8b8b07c9ecf1a452bb645570e04b99b7`.
Only its device changes from `cpu` to `cuda:0`. Training uses the same
`cuda_fp32_v1` profile as the continuous cohort: RTX A6000, torch 2.13.0+cu126,
CUDA 12.6, cuDNN 91002, deterministic algorithms, TF32 disabled, and one CPU
thread. Optimizer parameters, gradients, and moments are checked on CUDA.

```bash
python replay.py --gpu 0 --candidate simpler22_reference \
  --task mode_hold --workdir /tmp/simpler22-gpu-ring-new
python audit.py
```

Any of the 22 names in [protocol.json](protocol.json) can be replayed. The
archives contain the executed GPU source and the earlier device/RNG adaptations.
The [source comparison](original-source-comparison.json) binds the original
23-file native and 119-file transfer scopes: all public training package hashes
match before device adaptation. The two other differences add an unused
initialization option and change CLI defaults; this explicit-config, direct
runner takes neither changed path. The full diff is retained in
[inactive-source-differences.patch](inactive-source-differences.patch).

[GPU audit](audit.json) · [CPU regrade](cpu-regrade.json) ·
[Per-toy GPU results](LEADERBOARD.md) · [Raw GPU ledger](ledger.jsonl) ·
[Original CPU evidence](../simpler22/README.md)

The original common-22 initialization regrader explicitly supports CPU only.
The CUDA control uses the same frozen per-toy sustained gates and native
coverage/accuracy sample audits, plus device/config/update-count verification;
it does not claim that the old CPU-only common-22 regrader accepts CUDA.
