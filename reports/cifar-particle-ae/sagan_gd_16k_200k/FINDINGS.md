# Completed SAGAN duration review

Final and best sampled FID50k **12.53446 at200k**, versus19.64250 at40k. This reaches the user's numeric targets14 and13 under our existing protocol. Source/config certificate and parent/final/best checkpoint hashes verified; REVIEW_VERIFICATION.json records verification. Full constant-rate continuation completed with frozen pretrained features and sigma unchanged. No new jobs launched.

## Trajectory and comparison

| Model | Step | FID50k |
|---|---:|---:|
| SAGAN G+D | 200k | **12.5345** |
| Residual CNN, frozen-center fork | 90k | 15.7493 |
| Residual CNN, original | 80k | 15.7527 |
| Wide GN deconv without attention | 45k best /80k final | 17.4961 /17.6565 |
| Plain small deconv | 80k | 22.0022 |

These are observed checkpoints at unequal budgets, not isolated architecture effects. CNN also has different prior initialization history. At matched80k, SAGAN15.7728 versus wideGN17.6565; at40k SAGAN19.6425 was worse than wideGN18.2285. The short scout ranking reversed with duration.

SAGAN first crossed14 at120k (13.7778), rebounded to14.8527 at140k, then resumed improving. Last190/195/200k:13.3829/12.9593/12.5345. The final20k improved1.1995 FID; no final ceiling established. Total improvement from40k is7.1080; final is3.2182 below the previous original CNN best. The120k-to140k rebound shows why a temporary rise is not enough to establish an irreversible plateau.

This weakens an architecture-independent ceiling from the shared fixed sigma, particle count, E-only L2 or pretrained feature extractor: this configuration learned beyond it. It does not isolate whether G attention, D attention, the deconv backbone, prior history or interactions explain the difference. A benefit from more duration in this model is directly observed; an attention-specific mechanism is not established.

ReconstructionMSE40k0.127118 ->200k0.123156 while FID drops7.1; reconstruction alone understated the generative improvement. Feature variance ratio1.08128 ->1.06570 is not a coverage measurement. The final100-image grid contains varied recognizable vehicles and animals, alongside blurred/distorted objects; visual inspection is qualitative. No endpoint density/coverage measurement yet. BigGAN protocol parity remains unverified, so describe this as crossing the numeric target rather than an established benchmark victory.

Continuation took110.18 active-training minutes,146.73 wall minutes including32 FID50k evaluations. All5k checkpoints retained; best equals final. FinalSHA b45e411a8d3d0d0d3b7d45b877e719a7d3ed416edd223e513e07f6e05b3300f5.

Recommendation (not launched): preserve this checkpoint, measure endpoint density/coverage, then continue this recipe to300k to test the still-falling curve. A matched long no-attention wideGN continuation would help determine whether attention changes the long-run outcome. Avoid changing learning rates and architecture simultaneously with the duration test. No seed repeats.
