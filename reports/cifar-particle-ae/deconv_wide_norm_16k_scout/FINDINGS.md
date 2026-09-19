# Completed wider GroupNorm scout

Finished40k withFID50k18.2285, best sampled checkpoint, versus small unnormalized deconv26.2232 at40k (difference-7.9946). Training24.84 minutes, total34.20 minutes; original small model training23.94 minutes. Current-source/config certificate verified after completion.

FID at5/10/15/20/25/30/35/40k:50.4709,29.9273,22.7725,22.3330,22.0368,20.9390,20.9987,18.2285. The final interval improved2.7701 points after an almost flat30k->35k interval; a ceiling is not established. Wider generator plus normalization helps this matched scratch recipe, but this experiment cannot separate the two contributions.

Final reconstructionMSE0.13045 versus small model0.12820 despite much betterFID, again showing pixel reconstruction does not rank sampling quality. The inspected40k sample grid shows varied recognizable subjects with remaining malformed object details; a100-image panel cannot establish coverage.

Both use the same16k independent prior initialization, sigma, D/E initialization and training recipe. G increased298,595->925,763 parameters and gained hidden GroupNorm. Historical residualCNN16k40k17.0982 remains a contextual reference with different prior training/expansion history; overall historical best remains15.7527 at80k.

User chose next to test active-from-start SAGAN-style attention jointly inG andD using this wider GroupNorm architecture. This completed40k curve is the no-attention baseline. No wider-deconv continuation is queued.
