Prepared only: one public K3P v0.8.0 reference for `vector_unequal_mass`. No training, Torch import, model construction, or GPU work was performed during preparation.

The package is exactly release `0ff9a7afe5dcb828239369446cfe71971bce687b`: all 12 Python files are copied unchanged and pinned against the archived release receipt. The only recipe overrides are `total_steps=1200`, `num_particles=256`, `z_dim=4`, and `batch_size=128`. This is the released schedule rule with a declared benchmark horizon, not literal `get_recipe()`'s 7000-update configuration. Input noise ends at completed step120; output noise reaches .029 at240. Both LR schedules begin declining at720, toward the released .01 network/.05 prior floors at1200. The final optimizer update uses schedule index1199.

The host uses the same frozen unequal-mass task, promoted BatchDistance critic, canonical full parameter fixture, data CUDA0, latent CUDA1, penalty CUDA2 and fresh generator-real callback as current RP5/DV7 vector runs. All15 fixture tensors are pinned. Initial complete G/D/prior model hashes must match RP5's retained receipt before updates. Evaluation uses4096 samples at all24 checks50..1200, latent990, target991, projection992, and the explicitly declared paired output-noise seed2303. That is the current shared observation protocol; it is not historical global402. The original numerical scorer and final-five sustained gate are retained without edits, as exact selected source definitions in `source/frozen_host.py`; `extraction.json` binds each definition to complete originals under `original-host/`.

CUDA tensor placement and implicit-generator routing are scoped only to host sampling/scoring, with restoration on exceptions. The public trainer is constructed and updated with defaultCPU factory placement and its own explicit CUDA streams. Released Adam states start empty and initialize lazily; native scalar step counters must remain CPU while parameters/moments stayCUDA. No counter relocation, eager state, optimizer monkeypatch, extra candidate policy, or current EMA arithmetic is injected. Unexpected Adam placement after the first actual update produces retained ERROR evidence and never triggers a repair. Training calls the public `GANTrainer.step` inside an exception-safe full-step serial-autograd context, covering nested higher-order graph creation and backwards. The release package and checkpoint schema3 remain unchanged; the external checkpoint envelope separately binds this execution mode and exact source/fixture/protocol identity.

`preflight.py` uses only the standard library. It checks package/source/fixture hashes, exact host source spans and ASTs, declared recipe/task/card/schedules, complete-step serial scope, restoring contexts using a stub, and checkpoint-mode/source rejection. It does not certify runtime execution or training success.

Preparation check:

```sh
python preflight.py
```

For the authorized owner to launch later, in the declared Torch2.13.0+cu126 / CUDA12.6 / A6000 environment and selected lane GPU:

```sh
/tmp/pr38-default-env/bin/python worker.py --output /absolute/new/reference-output
```

The worker accepts only the output path; its task, seed,1200-update horizon and policy are fixed. It runs no training when imported. Output retains immutable source/declaration, runtime/import receipts, actual initialization and RNG state, all observations and per-update LR/noise/penalty diagnostics, optimizer-device proof, final or error checkpoint, result, and artifact hashes. The worker does not append to a lane's shared ledger: the owner can record the retained result after inspecting it. Checkpoint restoration helpers validate execution/source identity before touching trainer state; no resumed quality run is declared here.

A PASS would be evidence for this finite scheduled reference. It would not make K3P an eligible indefinite-use learner or replace the later recovery-ring comparator. Every failure remains part of the comparison.
