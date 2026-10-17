The generated caption toy still beats its controls after replacing the residual-scale floor with untrained FIT standard deviations. The actual pretrained-caption task retains uneven source convergence and a failed accuracy gate. This separate variant changes only cross-time latent correlation; it does not adjust the target’s token contrast, trainable architecture or optimizer.

Run the complete asset-free public-API test from the repository root:

```sh
CUDA_VISIBLE_DEVICES=0 python -m examples.e22_routed_caption_flow --run --out runs/caption-flow-v1
```

The command trains fresh ordinary BF16 LoRA, shared-Up BF16 particles and untied-Up BF16 particles for 512 updates each. It uses public `initialize_`, recipes, optimizers, policies and routed generation through immutable bundled helpers. Exit 0 means completed scientific PASS, 1 means completed FAIL, and 2 means incomplete/error. The imported package path/hash is recorded and checked for within-run immutability; future API versions are not rejected by a permanent native-version allowlist.

For each pool, source and latent occurrence, two independent Gaussian anchors define

```text
z(t) = cos(ω(t − .1)) a + sin(ω(t − .1)) b
ω = acos(.992) / .1
```

The coefficient squares sum to one, preserving each time’s Gaussian marginal. At time `.1`, the input is exactly `a`. The same fixed private CPU102 draw addressing is retained; the first two time draws become the anchors and later draws are consumed but unused. There are 12 independent FIT trajectories, six GUARD trajectories and 12 TEST trajectories, containing the original 48/12/48 rows in the original order. Train and test anchors are disjoint. Retained anchors, coefficients and row metadata allow exact CPU reconstruction without a model call.

The rounded `.992` correlation comes from the retained actual adjacent-time description at a `.1` gap. The existing toy grid has `.25` gaps, whose expected correlation is about `.950`. Its marginal RMS stays one, whereas real marginal amplitudes vary with time. The generated backbone, synthetic token directions and novel TEST times remain different from the actual task. The actual receipt used nearest cross-time pairing and did not recover trajectory IDs.

The same untrained FIT-std normalization **rule** is applied to the new contexts; its numerical scales are recomputed consequences. The teacher/caption pairing, six adapted projections, rank16, 128×4 shared particle bank, initial H/b/Up zeros and sampled C, BF16 matmuls, native D/G DV12 and RpGAN/KA2 updates, feature-only row controls, sampling and horizon remain fixed. Untied heads retain their previously declared 82,944 extra parameters and separate BF16 product rounding. The explicit sigma-zero particle-cloud bank is a task exception, not learned-MoG/default Forge qualification.

Only terminal TEST48 physical RMSE, reduced in float64 offline, determines the accuracy gate. Untied particles must improve aggregate RMSE by at least 0.1% against **both** controls, harm no source by more than `1e-6`, and benefit from codes by at least 0.1% overall and strictly on every source. Bank/router gradients must be live on at least 90% of the 511 post-first updates; all six C and particle-Up norms must be positive. Output errors never enter training, structural guards, stopping or checkpoint selection. No historical failure is required for future APIs to pass.

The sole science budget is 300 seconds on physical GPU0, including startup, data construction, prerequisites, updates, observations, restoration and final writes. Fixed saved clean observations at updates 0/64/128/256/384/512 illustrate the goal without selecting a checkpoint. After the complete run, render them separately on CPU within 60 seconds:

```sh
python -m examples.render_e22_routed_caption_flow --run-directory runs/caption-flow-v1
```

The GIF uses the actual zero target and three observed model residual maps on the same six source cameras, patch axes and initial-only physical-error color scale. Rendering needs optional NumPy and Pillow>=10.1. Run and media outputs are exclusive; existing evidence is never overwritten. Raw arrays, checkpoints and logs remain local and ignored.

Before the first science run, the NEW reduced CPU suite checks Gaussian covariance algebra, original anchor addressing, exact reconstruction, public initialization/ownership, recomputed scales, fresh restore and a two-update checkpoint replay using four tiny native updates total. Scorer/source/code destructive controls remain numerical. Software PASS is separate from the terminal convergence verdict.

The [preparation receipt](e22_routed_caption_flow_preparation.json) records 12 new CPU cases passing in 1.93 seconds, with four tiny public native updates spent once. Total software plus metadata-only process time was 4.871 seconds of the 60-second preparation allowance; no full576 host or CUDA execution occurred during preparation. The subsequent sole fixed GPU campaign passed in 162.183 seconds; its [results and actual goal GIF](e22_routed_caption_flow_results.md) and [independent CPU reduction](e22_routed_caption_flow_independent_review.json) preserve the measured outcome. This PASS rejects the correlation-only change as sufficient to reproduce the actual failure in this generated fixture. The actual-caption failure and original full-Supra goal remain unchanged.
