The normalization variant changes only the frozen game scale in the generated
caption task from PR240. It uses untrained FIT residual coordinate standard
deviation instead of the previous `.04` floor. Every coordinate must have finite
standard deviation greater than the fixed `1e-8` numerical protection; a
degenerate coordinate stops execution before training. There is no scale scan.

```sh
CUDA_VISIBLE_DEVICES=0 python -m examples.e22_routed_caption_normalization \
  --run --out runs/caption-normalization-v1
python -m examples.render_e22_routed_caption_normalization \
  --run-directory runs/caption-normalization-v1
```

Run from the repository root with Torch and ParticleGAN available. NumPy and
Pillow>=10.1 are optional dependencies for the saved-state renderer. Outputs
must be fresh. The science command uses GPU0 and has a fixed 300-second budget
from startup through final writes; rendering has a separate 60-second CPU cap.
Exit codes are 0 for completed PASS, 1 for completed FAIL, and 2 for incomplete
execution. JSON progress every 64 updates is suitable for tailing.

The immutable imported PR240 helper executes three fresh public-API native
game arms for exactly 512 updates each: ordinary BF16 LoRA, shared-Up BF16
particles, and untied-Up BF16 particles. Captions, frozen teacher/backbone,
latents, times, named initialization, sampled C with H/b/Up zero, native D/G
DV12, row controls, optimizers and observation schedule remain unchanged.
The wrapper records its new task identity and the inherited helper identity
separately. Actual imported package hashes are recorded and checked within each
run; future compatible API versions remain runnable.

The unchanged numeric gate requires the untied arm to improve raw TEST48 RMSE
by at least .1% against both controls, harm no source by more than `1e-6`, and
benefit from codes by at least .1% overall and strictly for all six sources.
Bank/router gradients must be live for at least 90% of updates 2–512, with
positive C and particle-Up norms at all sites. Raw physical accuracy is measured
offline; training and structural decisions retain the native learned game.

Before training, the run saves the exact FIT residuals, raw standard deviations,
new and previous scales, floor fractions, and raw/normalized token-mean and
centered powers. Its GIF uses actual public clean observations at updates
0/64/128/256/384/512, a zero residual target, fixed source queries, and one color
range determined only from the initial frames. It does not affect the gate.

A PASS rejects normalization alone as sufficient to remove the win in this one
fixed generated fixture. A FAIL reports the failed numeric bounds; code/live-only
failure does not reproduce an accuracy gap. Neither result uniquely explains
the actual pretrained-caption failure or establishes a full-Supra win. The
actual task also had an absolute `.0005` accuracy requirement and different
backbone, token directions and trajectory sampling. All earlier evidence stays
unchanged.
