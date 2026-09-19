# Completed continuation review

Finished80kFID50k17.6565. Best sampled checkpoint45k17.4961. Original40k18.2285; continuation endpoint gains0.5720, with a50k rebound to19.3232 followed by recovery. No later checkpoint beats45k, although60k->80k improves0.8640. Diminishing progress is evident; a strict capacity ceiling is not established.

ReconstructionMSE improves0.13045->0.12488 but does not track FID closely. Inspected45k/80k grids retain varied recognizable subjects and rendering artifacts; no deconv density/coverage probe yet. Frozen features/sigma unchanged; final source/config certificate and all checkpoint hashes verified. Continuation training24.93min, elapsed34.31min.

Recommend forking45k with half learning rates (preserving ratios) and comparing to the existing constant-rate continuation through80k. This tests update-size effects without repeating a baseline. No new run queued. Full review and curves: `../deconv_attention_review/`.
