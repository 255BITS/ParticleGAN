This harness prepares the frozen 1,200-update mode-hold screen for the new deterministic initialization epoch. It does not inherit any earlier score or the initializer study's 22/22 result. Preparation executed CPU initialization contracts only: no forward, backward, optimizer step, training or GPU operation.

`protocol.json` keeps 12 particles, z4, batch128, std .5, the original 96×3 G/D architecture with Fourier3, all 24 observations, 4,096 evaluation samples, all-eight-mode/HQ≥.90 thresholds and the five-pass terminal suffix. `mode-hold-source/` retains the independently reviewed exact host definitions and all 1,200 old real/index/cursor receipts. Those receipts constrain the random sample transaction. Old model values are never loaded.

The initializer is independently pinned to develop `c720645ecae6b648e9fc6034e9d6b48ccff06ed3`. A candidate declaration separately seals its algorithm origin, complete new package hashes, recipe configuration, port changes, execution signature and Adam counter policy. Current public K3P and KA2 declarations are included; the previously defined constant-nominal-rate KA2 has a separate declaration. Its original recipe horizon 4,600 and input/output noise milestones 360/720 remain unchanged for this 1,200-step screen. Finite and continuous policies retain their own declared horizon/noise; screening performance and indefinite-use eligibility are separate.

Construction uses the public `recipe.make_prior(init_std=.5, generator=shared_stream)` in the original prior-first CUDA scope. Its ordinary Gaussian constructor draw still occurs, then deterministic R2 values replace z without consuming RNG. G/D are constructed next, and public GANTrainer→recipe.make_optimizers initializes fresh network parameters with keys0/1 and synchronizes the EMA critic. The host CUDA scope closes before trainer/optimizer construction or updates, preserving native CPU defaults. No process-wide initialization registry hook is installed. Caller-supplied random priors and old fixture copies would bypass the intended new prior and are forbidden here. A port must compute width/bandwidth/other derived state after deterministic z replacement.

`init_contract.py` builds twice on CPU without resetting the seed, with additional global/private random draws between builds. It compares every checkpointed value except actual RNG-state entries and every named parameter/buffer of G, D, prior and their EMAs. It also checks the exact R2 table and DV-style initial bandwidth against the final initialized z. Temporary observation wrappers around the public initializer functions record pre/post CPU and shared sampling RNG; they only read state, restore the functions immediately, and never run in the training harness. Constructor RNG consumption is retained, while each initializer replacement is required to consume none. No derived random tensor is excluded from repeatability checks.

The runtime performs a complete dry reconstruction of all 1,200 CUDA real/index/cursor receipts before the first optimizer update. During training, the original shared order remains D-real → D-latent → G-latent → G-real. The cloned-cursor adapter supplies the public `GANTrainer.step` with G-real, requires exactly two accepted latent draws, and then commits the original caller cursor. The whole public step is inside a restoring serial-backward scope. A constructor serial-mode argument is passed only when that candidate actually supports it; current public K3P/KA2 use the external scope alone.

Evaluation retains latent seed9 and separate global output noise at402+completed under fork_rng. Candidates with their own indexed latent support retain the previously audited private support stream2303+completed, using their own package `_generate` path with output sigma0 before the separate output noise. Plain candidates use their ordinary package generation path. The private generation helper is used only to preserve the candidate's sampling law; all learner updates go through public GANTrainer.step. Live and EMA observations must leave full trainer/caller RNG state unchanged.

Counter placement is candidate-owned. Ordinary noncapturable lazy Adam uses CPU scalar steps; a declared eager candidate may retain parameter-device steps. The manifest explicitly declares G/D placement, and the runtime validates initial/accepted clocks and moment placement without injection or repair. The harness does not change optimizer, EMA, controller or numerical mechanism arithmetic.

The public CPU preflights use the complete merged packages: K3P files are identical between research merge `c714f59d` and the recorded research HEAD `f3d551d8`; KA2 is `25751c0864dd8259b00c5804f600cd41cce6e4cf`. Existing preparation error receipts retain the corrected KA2 manifest mistake that initially claimed a constructor serial argument; the public package was unchanged.

CPU initialization-only verification, in a fresh process:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/pr38-default-env/bin/python preflight.py \
  --package-root /ABSOLUTE/SOURCE-PORTED-CANDIDATE \
  --declaration /ABSOLUTE/CANDIDATE-DECLARATION.json \
  --cpu-init --output /ABSOLUTE/NEW-CPU-RECEIPT.json
```

The external worker, after root's source review, uses its assigned GPU and a new output directory:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/pr38-default-env/bin/python mode_hold_harness.py \
  --package-root /ABSOLUTE/SOURCE-PORTED-CANDIDATE \
  --declaration /ABSOLUTE/CANDIDATE-DECLARATION.json \
  --output /ABSOLUTE/NEW-RUN
```

Adding `--preflight-only` performs CUDA construction and sampling verification with zero optimizer updates; it does not produce a quality score. No CUDA preflight was run during this preparation. Every run snapshots learner+harness+host+fixture sources and emits raw initial/final/error state, complete batches, rates/noise/counters/controller diagnostics, all observations and failed bounds. An error is retained, never silently repaired or treated as a quality pass. This is one quick screen; broader and long qualification still require each surviving unchanged candidate's own evidence.
