# Saved learned-state regression diagnosis

This separate CPU diagnosis reads the frozen CUDA E22, CB64-RA and CB64-RA2 learned checkpoints at 1000 and 2000 updates, for toy and MNIST. It does not change their packages, configurations, streams, checkpoints or acceptance outcomes.

CUDA is hidden before importing Torch. Two CPU threads, one interop thread and deterministic algorithms are used. No training update, checkpoint continuation, GPU operation or additional seed is performed. CUDA generator byte states are retained as read-only evidence; they are never installed in a CPU generator.

The checkpoint stores the fast training iterate. The served pair is derived from the saved table tester's `last_decisive == -1` condition and `serve_average > 0`. Fast, EMA and crossed generator/prior pairs are inspected separately. All toy output perturbations use the existing `geometry/gpu-inputs.pt` fold2 noise. Latent gradient probes use its fold128 noise and the same first 128 prior rows across variants. MNIST probes omit output noise because this immutable input artifact has no image-shaped output noise. These conditional comparisons are not iid quality retests or exact CPU replays of CUDA randomness.

For each saved step the gradient probe uses that update's original generator real batch from the frozen toy or MNIST stream. It evaluates the fast G/D state with and without that variant's latent displacement, runs backward without an optimizer update, and checks that prior and generator gradients remain finite and connected. Saved Adam steps/moments, A2 damping history, critic guard/anchor, LR/tester state, row evidence and controller state are summarized.

Source, checkpoint, data and noise hashes are checked before and after the diagnosis. All outputs are confined to this directory. Ordinary reaction feasibility belongs to the stability diagnosis; exact versus bounded latent neighborhoods belongs to the geometry diagnosis.

`diagnose_support_counts.py` reads stability's saved CPU reconstruction of the same learned toy states. It preserves that partition, score, Q and heldout null, then decomposes each existing cell count by the existing pointwise p>Q parent eligibility and by the existing family BH flags. Real calibration self-scores are reported as an in-sample calibration reference, not as an independent false-positive test. This reconstruction differs from the archived CUDA random projection; no checkpoint snapshot or new score is substituted into training. Augmented count TV is descriptive and introduces no controller test or acceptance gate.

Run:

```
CUDA_VISIBLE_DEVICES='' /tmp/pr38-default-env/bin/python -B /ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/performance/training-regression/diagnose_state.py
CUDA_VISIBLE_DEVICES='' /tmp/pr38-default-env/bin/python -B /ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/performance/training-regression/diagnose_support_counts.py
```
