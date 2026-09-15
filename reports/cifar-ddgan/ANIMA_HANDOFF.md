# Compaction handoff: back on master after Anima experiments

The user requested committing the feature branch and returning to the main
branch. This repository calls that branch `master`. Experimental code, configs,
tests, and full reports remain committed on `experiment/anima-transplant`:

- `6e53731`: frozen Anima block transplants and completed experiments.
- `5102d8c`: fully trainable transplants and completed experiments.

No experimental code was merged or cherry-picked into master. Nothing was
pushed. Both GPUs are free; no training is active or queued. Preserve unrelated
untracked `.claude/` and `sparse-ucd.log`.

## Results to carry forward

All rows below trained for 10k updates at batch64 and used final FID50k.
Historical baselines have different source provenance; each simultaneous
pretrained/random transplant pair is a matched comparison.

| Generator | FID50k ↓ | Training min |
|---|---:|---:|
| Attention U-Net, from scratch | **29.327** | 10.73 |
| Frozen pretrained Anima transplant | 30.263 | 19.98 |
| Plain U-Net, from scratch | 31.555 | **9.22** |
| Frozen random transplant | 32.550 | 20.82 |
| Trainable pretrained Anima transplant | 36.558 | 24.89 |
| Trainable random transplant | 496.351 | 25.62 |

Two width-2048 Anima blocks attach to the U-Net's 8×8 encoder features through
learned image/particle/class adapters. Frozen pretrained weights helped versus
the frozen random control, but did not beat the cheaper attention U-Net.
Unfreezing at the ordinary constant G rate 0.0006 regressed. The random
trainable model collapsed: its final EMA probe produced one quantized yellow
image for 100 inputs with heavy output saturation. The pretrained trainable
model recovered from early instability but scored worse than the frozen model.

Each round completed two 128-update profiles, two 1k scouts and two 10k runs.
The trainable implementation updated every donated parameter tensor, including
time/AdaLN, with FP32 parameters/Adam/EMA and BF16 donor matrix operations.
Tests verified live timestep conditioning, gradients and EMA restoration;
checkpoint audits verified actual changes in trainable donor weights and
unchanged frozen donor weights. Full validation is in the feature-branch reports.

The 1k ranking did not predict the 10k outcome. Separate launches did not
produce bitwise-identical prefixes despite the same seed/training settings;
TF32/cuDNN benchmark execution was enabled. No seed-only repeats were run.
Do not equate 5k-sample diagnostic FID with final FID50k.

## Next-session constraints

- Keep constant learning rates. The user explicitly rejected the earlier
  learning-rate decay proposal because of its tuning burden.
- Preserve learned latent particles, four-step DDGAN, Gaussian step noise,
  joint time/class UCD, Rp logistic, VICReg and exact lazy-4 bcap.
- Use both GPUs for useful independent comparisons, config files and easy
  tail logs. Explicitly pass `--workers_per_gpu 1` for CIFAR runs.
- No seed-only experiments. Preserve sample exposure when changing batch.
- Report results after completion; do not repeatedly inspect active runs.

No next experiment is selected. Attention remains the better 10k architecture
and cost tradeoff. Plain U-Net remains the default, with the best certified
longer result: FID 25.397 at 50k in 45.73 training minutes. Its config is
`configs/cifar_ddgan/duration_50k/baseline.yaml`. The old attention 30k checkpoint
was overwritten; checkpoint retention remains unimplemented.

A one-block donor or a smaller constant donor rate are untested ideas, not
queued tasks. Do not reintroduce LR decay or launch Anima training automatically.

Inspect full reports without switching branches:

```sh
git show experiment/anima-transplant:reports/cifar-ddgan/anima/READOUT.md
git show experiment/anima-transplant:reports/cifar-ddgan/anima_trainable/READOUT.md
```

Raw results, checkpoints, source archives and donor weights remain locally
under ignored `results/cifar_ddgan/anima*` and `data/anima`. Saved source
archives are authoritative; master does not contain the transplant trainer.

Historical logs:

```sh
tail -F results/cifar_ddgan/anima.live.log results/cifar_ddgan/anima_trainable.live.log
```
