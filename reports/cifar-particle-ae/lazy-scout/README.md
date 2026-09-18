# Lazy bcap scout: N=4, 8, 16

Branch: `feat/cifar-ae-gan-pretrained-encoder`.

This follows the pretrained-encoder round, whose scratch encoder won with FID5k
26.523. All three new runs use that architecture and optimizer configuration,
the same seed 24002, and 5,000 training steps. Only `reg_every` changes.
The default remains N=4; an applied penalty receives N times the base coefficient.
The gradient norm uses exact double backpropagation. Skipped steps do no penalty
work. Learning rates and Adam betas stay fixed across the sweep.

The runs execute sequentially on GPU 0 (RTX A6000) to compare speed without
concurrent training. N=4 is rerun under this source, hardware schedule and new
evaluation protocol; this is a matched control, not a seed experiment.
Each run measures FID5k at step 2,500 and **FID50k at step 5,000**, using the
existing CIFAR train50k reference and torch-fidelity TF-compatible Inception.
Test reconstruction uses 1,000 unaugmented images. Numbered checkpoints remain.

```sh
bash experiments/cifar_ae_lazy_pipeline.sh
tail -F runs/cifar_particle_ae/lazy_scout/PIPELINE.log
```

The pipeline generates `LEADERBOARD.md`, `leaderboard.json`, and a proposed
`winner_long.yaml` after all certified results are available. The leaderboard
ranks final FID50k, reports speedup versus N=4, and estimates 30k/50k/100k-update
runtime including measured evaluation overhead. The longer run is not launched.
Evaluation timing is recorded separately from training; projected runtime assumes
unchanged throughput and scales reconstruction work from 1k to 10k images.
It cannot predict the training budget needed to achieve a particular quality.

The original [BigGAN paper, appendix C.2](https://arxiv.org/html/1809.11096#A3.SS2)
reports CIFAR-10 FID 14.73 and IS 9.22 without truncation. Use that as historical
context, not a matched baseline: BigGAN is conditional, while this AE-GAN is
unconditional and uses an ImageNet-pretrained discriminator. Its encoder starts
from scratch. Exact metric compatibility with the paper has not been established;
the [author implementation](https://github.com/ajbrock/BigGAN-PyTorch#an-important-note-on-inception-metrics)
distinguishes PyTorch monitoring metrics from official TF scores. A strict
comparison would evaluate a specified CIFAR BigGAN checkpoint through our evaluator.

Do not compare intermediate FID5k to final FID50k as evidence of improvement:
sample count affects the estimate. The final 50k sample measurement is included
now to remove that ambiguity between scout arms and measure its actual cost.

Validation covers N=4/8/16 skipped steps, penalty and discriminator-gradient
scaling, config validation and runtime arithmetic. Separate 32-step GPU smoke
runs exercise every schedule, checkpoint saving and reconstruction evaluation.
