# Completed attention and duration review

Both GPUs are idle. Both training summaries pass current config/source certificates, and all reported parent/best/final checkpoint SHA256 values were rechecked. Frozen pretrained D features and fixed sigma remain unchanged. No new experiment launched in this review.

| Model | Best sampled FID50k | Step | Endpoint FID50k | Endpoint step |
|---|---:|---:|---:|---:|
| Historical residual CNN,16k | 15.7527 | 80000 | 19.0584 | 160000 |
| Wider GroupNorm deconv,16k | 17.4961 | 45000 | 17.6565 | 80000 |
| Wider GroupNorm deconv + G/D attention,16k | 19.6425 | 40000 | 19.6425 | 40000 |
| Small unnormalized deconv,16k | 22.0022 | 80000 | 22.0022 | 80000 saved before user stop |

Historical residual CNN used1k->16k checkpoint expansion at10k; all deconv variants used independent16k centers from scratch. The historical CNN score is context, not a matched architecture ablation. No run has reached targetFID<13.

## Attention experiment

SAGAN-style G/D attention completed40k with attention active at unit strength from initialization: no gate or phase-in. All logged losses/penalties are finite; every reported G/D attention gradient norm is finite and positive. Final attention gradient normsG0.33188,D0.41898. The sample grid contains varied recognizable objects and remaining shape/detail artifacts; no obvious wholesale collapse is visible in the100-image panel. This supports compatibility of active attention with this training recipe; it does not establish that all GAN instability mechanisms are eliminated or exclude partial mode loss. No density/coverage probe for these deconv runs has been performed.

Matched40k: attention19.6425 versus no-attention18.2285, difference+1.4140. Attention was better at5k/25k/35k and worse at10k/15k/20k/30k/40k. No consistent quality advantage is established. Last10k improved22.2370->19.6425, so40k is not an established attention ceiling. Reconstructed-imageMSE0.12712 is lower than baseline0.13045 despite worseFID. Keep generation and reconstruction evaluations separate.

Attention required28.42 train minutes versus24.84 for the matched wide no-attention40k run, approximately14.4% extra training time. Total elapsed37.86 versus34.20 minutes. G has930,883 parameters versus925,763; D adds5,120 attention parameters. Existing base G/D/E tensors and prior initialization/recipe are matched, but both G and D changed together, so any effect cannot be assigned to one side.

## Wider deconv continuation

FID40k18.2285 ->45k17.4961 ->50k19.3232 ->55k18.7246 ->60k18.5205 ->65k18.4515 ->70k18.6665 ->75k17.8848 ->80k17.6565. Endpoint improved0.5720 over40k but never surpassed45k. This is diminishing progress with a rebound and recovery, not a monotonic curve. Last20k improved0.8640, so a strict capacity ceiling is not established. Best checkpoint45k and endpoint80k are separately retained.

Inspected45k and80k grids retain similar varied subjects/layouts with differences in rendering; the visual panels do not establish memorization or coverage. Reconstruction improves0.13045->0.12488 while FID improvement is much smaller. Continued low reconstruction loss does not establish improved sampling quality.

## Recommendation

1. Continue SAGAN40k->80k unchanged to compare at the same duration; it is still improving and40k cannot resolve whether attention mostly changes learning speed or the eventual score.
2. On the other GPU, fork the wide deconv45k best checkpoint and halve all learning rates while preserving their ratios, then train to80k. Existing45k->80k constant-rate continuation is the control, so no baseline retraining is needed. This tests whether update size contributes to the rebound/slow progress; it does not assume that is the cause.

These are recommendations only, not queued work. Keep16k particle count and no seed-repeat experiments. If attention later helps, separateG-only andD-only effects before adding further architectural changes. Curve artifact: `curves.png`; plotted values: `curves.json`.
