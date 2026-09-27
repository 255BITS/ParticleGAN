# Released K3P mode_hold reference — prepared, not run

This bundle prepares one finite comparator using the exact 12-file public K3P v0.8.0 package at commit `0ff9a7afe5dcb828239369446cfe71971bce687b`. No package changes, Torch import, model execution, training or GPU use occurred during preparation. Independent source review and supervisor release are required before the external precision worker executes it. This is not a fourth mechanism or a qualification claim.

The only recipe overrides are `total_steps=1200, num_particles=12, z_dim=4, batch_size=128`. Every other released setting remains intact, including ordinary sequential updates, native lazy Adam, original critic EMA buffer arithmetic, losses and prior regularization. The explicit benchmark horizon gives input-noise cutoff 120, output-noise saturation 240 and LR decay beginning 720; update 1200 uses schedule index 1199. It is not the literal released default horizon 7000 or the candidate's absolute 360/720 schedule.

The frozen tiny host uses prior-first CUDA initialization at std .5 on shared seed 0, then CUDA G 4→96×3→2 and D 2→96×3/Fourier 3 on global seed 0. CUDA factory scopes close before constructing and stepping the released optimizers; those retain the CPU default device. The complete public step runs in an external restoring serial-autograd context because the released trainer has no `serial_backward` argument. Nothing eagerly allocates or moves Adam counters. First/final receipts require CPU scalar steps, CUDA parameters/moments and exact accepted counts; a mismatch is recorded as ERROR, without repair.

`source/frozen_host.py` contains 21 exact extracted definitions/constants. `extraction.json` binds every source span. `original-host` and `frozen-mode-hold-host.zip` retain immutable RP9 host and canonical-source provenance; they are never imported. `source/particlegan` is the only learner import path, verified against the released ZIP. No candidate controller, game transaction, eager state or legacy learner is imported.

`fixtures` retains RP5/RP7 initial/final checkpoints, initial/source/runtime receipts and all 1,200 batch/cursor hashes. `fixture-receipt.json` records raw tensor/storage hashes read without Torch. Both candidates' saved initial models, global/shared RNG and final caller cursor agree; all 1,200 batch receipts are identical. Runtime asserts model and EMA digests, individual initial tensors and all released RNG streams, then verifies every real-D, latent-D, latent-G, real-G and accepted cursor. It keeps the frozen cloned-cursor adapter and checks the public step consumes exactly two latent draws. No old candidate checkpoint is loaded into K3P. These are recorded source-defined CUDA reconstructions; the missing historical full initial tensor fixture is not claimed to have been recovered.

Evaluation preserves all 24 checks at 50..1200, 4096 prior samples from private CUDA seed 9, and separate global output noise at seed 402+completed inside `fork_rng`, using the released actual noise amplitude. Live and EMA measurements are paired and checked not to mutate training state. Public `sample()` is deliberately not used because it joins latent/output draws. Exact original scorer definitions require all 8 modes and HQ ≥ .90 with a final suffix of at least 5; every failed observation is retained.

Initial/final/error checkpoints wrap the untouched released schema 3 state with source/recipe/protocol identity, full-step execution mode and caller RNG. The validator rejects an incompatible envelope before restoration. Output includes sealed source ZIP, runtime/import proof, complete initial raw state, all batch/rate/noise/loss/observation receipts, native counter placement, result and artifact hashes. The runner has no mechanism, seed, horizon or architecture switches.

Source-only verification (Python 3.9+; no Torch import):

```sh
python /ml2/hypergan/gan-attempts/continuous-api-20260926/supervisor-support/k3p-mode-hold-reference/preflight.py
```

After independent review and supervisor release, the assigned external precision worker may use its assigned GPU visibility and a new output directory:

```sh
env CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /tmp/pr38-default-env/bin/python \
  /ml2/hypergan/gan-attempts/continuous-api-20260926/supervisor-support/k3p-mode-hold-reference/worker.py \
  --output /ABSOLUTE/NEW/EXTERNAL-WORKER/OUTPUT
```

Do not run this during preparation or alter the sealed files after review. Keep reviewer receipts outside the bundle. `preflight-result.json` is an unsealed output recording the seal hash; all actual source, declaration, fixture and documentation inputs are sealed. Root owns shared evidence manifests and launch decisions.
