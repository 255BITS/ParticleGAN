Prepared source only: exact public ParticleGAN **v0.8.0 K3P**, release `0ff9a7afe5dcb828239369446cfe71971bce687b`. No training, Torch import, model construction, GPU probe or Torch tests were run during preparation.

This is a supporting public reference, not another candidate proposal. The only recipe override is `total_steps=4600`. It preserves the release's schedule rules: input noise reaches zero at completed step460; output noise reaches .029 at920; network cosine decay starts960 and reaches its .01 floor at1600; prior decay starts2760 and approaches its .05 floor at4600. Update4600 uses schedule index4599. This is explicitly different from literal default7000 and from a candidate's fixed360/720 initialization or indefinite recipe.

All12 package files are copied unchanged from the retained release ZIP. The original floating-buffer EMA uses mul/add, and ordinary native Adam states start empty. With nonfused/noncapturable Adam, scalar steps must stayCPU while parameters, gradients and moments stayCUDA. The worker records proofs after actual update1 and4600 and after the frozen control is restored. It never populates, relocates or repairs optimizer state. The public trainer has no serial_backward argument: an external restoring context encloses each entire public step, including graph creation and both backwards. Released checkpoint schema3 remains unchanged inside a separate host envelope that binds execution/source identity and caller data/target state.

The fixed public recovery ring uses20,000 particles,z2,batch2048,96-wide three-layer G/D and Fourier3 critic. Seed0 constructs G onCPU then moves it toCUDA, then D likewise; the released trainer constructs the prior onCPU and moves it toCUDA. Full model/EMA/private/global/data RNG and target hashes must match the retained public-ring fixture before updates. Trainer/optimizer hashes are deliberately not compared to the RP5 candidate. The data stream is separate CUDA0; D and G reuse the identical real batch. Evaluation uses4096 samples every10 updates with isolated CUDA seed9 and the baseline's actual public output-noise amplitude. Live weights determine the measurements; EMA is diagnostic.

At update2400, the worker saves complete state, constructs a separate frozen released trainer and loads that state into the frozen trainer only. Loading restores global RNG altered by construction; a full main-state/caller hash must remain identical before changing the target to +[1,0]. The main learner is never reloaded or restarted. The frozen trainer never updates. All220 shifted frozen observations are retained alongside the live curve. Summary reports first arrival, every later miss/departure, retention counts/minima and final suffix, including the3600 comparison endpoint. There is no81/81 deadline or automatic winner gate; terminal `COMPLETE` means the4600-update evidence was collected. Runtime/assertion failure produces `ERROR`, a traceback and an error checkpoint where possible.

The bundle avoids candidate import contamination. `source/frozen_host.py` contains only exact selected model, geometry, scorer and receipt definitions from the retained recovery-host ZIP. `extraction.json` pins every source span and AST to complete originals in `original-host/`; that directory is provenance and is never placed on the import path. No candidate worker or benchmark package is imported. Imported release paths/hashes and the pinned Torch version/Adam source hashes are checked during authorized execution.

Source preparation check (standard library only):

```sh
python preflight.py
```

Execution is pending an independent second source review and the external owner's launch. Runtime assertions require the existing Torch2.13.0+cu126, CUDA12.6, A6000 environment, deterministic FP32/TF32off, one thread, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Select the assigned physical GPU through CUDA_VISIBLE_DEVICES. No seed/task/budget/policy options exist on the worker CLI.

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /tmp/pr38-default-env/bin/python worker.py --output /absolute/new/reference-output
```

The worker accepts only a new output directory outside the bundle. It retains the sealed source ZIP/declaration, import/runtime receipts, initial/change/final states and caller RNG/means, every applied rate/noise/penalty row, live/EMA/frozen curves, complete state receipts every100 updates, native Adam device proofs and artifact hashes. It does not append a shared ledger or edit the bundle. No resumed training protocol is declared here. Unexpected initialization or runtime behavior must remain an error; do not repair the reference after seeing scores.

Remaining blockers: independent second review; actual pinned-runtime/import/initialization validation; first/final native Adam device proofs. Passing the source preflight does not claim any of these runtime results or a quality score.
