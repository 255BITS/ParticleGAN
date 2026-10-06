# Constructed terminal-cancellation API test

This is an analytic stress case for a small paired edit behind a cancelling
frozen terminal projection. It does not reproduce measured Supra activation
directions or establish a unique convergence cause. No models, software tests,
API updates or scientific training have executed for this variant yet.

The frozen head has one publicly initialized weight owner `W`, with effective
projection `[W, -W]` on `[a + u, a - u]`. The separate carrier `a` stays frozen.
Both sequential adapted/routed sites affect the edit `u`, whose fixed amplitude
is 1/64. Six source one-hot conditions map to six distinct positive conditions;
the independent target teacher always uses BF16. There is no caption-shrink law,
external model/data asset, empirical renormalization, or actual-weight copy.

Four fresh arms train for exactly 512 updates: ordinary and untied particles,
each with BF16 or FP32 student terminal arithmetic. All use the same BF16
Down/Up products, native D/G DV12, paired RpGAN/KA2 updates, source/time/CFG3
conditioning, private streams, and one shared initial FIT-std normalization.
Fresh owners receive public `initialize_` before optimizers and EMA; the declared
subsequent neutralization sets both Up heads and H/b to zero while preserving
sampled C. The correlated terminal weight remains frozen.
Its Parameter storage is FP32 in every arm; BF16 versus FP32 changes weight
rounding along with input, multiplication and output arithmetic. This matches
the earlier depth-one control's storage scope. Widening an already quantized
tensor cannot recover lost bits; no BF16-storage claim about the actual full
model is made here. Its storage and compute laws require independent inspection.

The main numerical gate requires particle FP32 to improve ordinary FP32 RMSE by
at least 0.1%, with no source harmed by more than 1e-6. Zeroing particle codes
must worsen aggregate RMSE by at least 0.1% and every source strictly. Bank and
router gradients must be live on at least 90% of the 511 post-initial updates,
and C/code-Up norms must be positive at both sites. The score is unscaled F64
physical paired error on all 48 fixed TEST queries. Accuracy never controls
optimization, structural guards, stopping, or snapshot selection.

The last BF16 particle update has a separate saved-output/VJP diagnostic. It
reports whether the native displacement has a helpful clean parameter slope but
worsens finite error, and whether FP32 reduces squared output motion and the
finite-change/VJP discrepancy to at most 75%. A negative or undefined precision
witness remains visible and cannot invalidate a future API's convergence PASS.
Changing final arithmetic changes student baseline predictions too; this is not
an output-cast-only or particle-specific explanation.

From a clean checkout with Torch and ParticleGAN:

```sh
python -m examples.e22_routed_terminal_cancellation --run \
  --out runs/terminal-cancellation-v1 --device cuda:0
```

The science budget is 300 seconds from startup through final writes. Exit 0
means completed numerical PASS, exit 1 completed FAIL, and exit 2 incomplete.
The completion companion records `complete=true` and the matching
`scientific_status` for either completed verdict; incomplete attempts use null
status and exit 2.
The actually imported package path/hash is recorded and checked within the run;
there is no permanent native-version allowlist. Output directories are exclusive.

Render the saved actual observations afterward, without models or native calls:

```sh
python -m examples.render_e22_routed_terminal_cancellation \
  --run-directory runs/terminal-cancellation-v1
```

The optional renderer needs NumPy and Pillow >= 10.1 and has its own 60-second
CPU budget. Its GIF shows the true zero target and all four observed arms at
steps 0, 64, 128, 256, 384 and 512, using six fixed source cameras and one color
scale derived from the initial observations. It does not substitute thumbnails
for the complete TEST score.

Preparation is source-only pending root review. The four new CPU qualification
cases reserve exactly four tiny native updates for a 2+2 public replay; all other
checks use zero updates. Old tests, actual-caption results and frozen protocols
remain unchanged. Even a future PASS here would not establish sustained Supra
improvement; the original full-model goal remains unmet.
