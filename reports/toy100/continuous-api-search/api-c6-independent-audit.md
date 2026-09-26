# API-C6 delta audit

The reviewed C6 snapshot fixes C5's accepted-reference bug and scopes serial backward over the complete public update. No new source blocker was found in this delta. This is a read-only implementation audit, not a quality or CUDA continuation pass. C6's `result.json` appeared between the initial inspection and the final audit write; its quality results were not assessed here.

All 24 source files in `C6-single/source.zip` match their manifest hashes, and the reviewed working files match the same hashes. Archive SHA-256: `4c9776c8547db6e4029899bf2e8bab98b8c5b2e7c8674c8ea90e4718e65cb690`. Unchanged same-oracle replay, constant-rate, sparse-prior and horizon-isolation analysis is in `API-C5.md`; C5's scores do not transfer.

`JointStage.after("g")` now only restores pending critic parameters. `GANTrainer._ordinary_step` restores original critic trainability flags in its `finally` block and then calls `stage.finalize()`. The secant resolution and accepted critic reference update therefore happen with the original flags restored, before generator/prior EMAs are updated. Predictor moments/controller state are retained; the reference is reset to its value before the provisional critic step, then averaged once toward final accepted critic parameters. The moving controller sets alpha .1 with minimum decay .9, yielding the intended .99 parameter EMA. Permanently frozen parameters retain the existing copy behavior.

Existing checkpoints support the correction:

| Accepted step | Parameter tensors differing from reference | Maximum absolute D/reference difference | Recorded EMA updates |
|---:|---:|---:|---:|
| 1600 | 7/8 | .01935410 | 801 |
| 1740 | 7/8 | .01504860 | 941 |
| 1750 | 7/8 | .02040728 | 951 |
| 1800 | 7/8 | .01233032 | 1001 |
| 2400 | 7/8 | .01604504 | 1601 |

All these checkpoints have controller calls, observed steps and both Adam step counts equal to their accepted trainer step, alpha .1, and `serial_backward=True`. The initial checkpoint has D equal to the reference, as expected. Nonzero later differences refute C5's hard-copy behavior; they alone do not reconstruct a one-step EMA equation because adjacent accepted parameter snapshots were not saved. The exact averaging equation follows from the frozen source ordering and arithmetic. The added conformance test checks that equation, but this audit did not execute tests.

The public `GANTrainer.step` enters `torch.autograd.set_multithreading_enabled(False)` before `_dispatch_step`. That scope contains both predictor/corrector `_ordinary_step` calls, critic penalty construction and all nested `torch.autograd.grad` work, outer D/G backward, accepted-state commit and error handling. It is not limited to the outer `.backward()` calls. The context manager restores the caller's mode on success or exception.

Execution mode is an explicit constructor boolean and is checkpointed when enabled. `load_state_dict` checks the stored boolean against the trainer's mode before mutating models or optimizers, treating absent mode as historical false. Consequently an unchanged C5 checkpoint cannot load into a serial C6 trainer. All inspected C6 checkpoints contain the true marker. Actual fresh-process CUDA continuation still needs the separately declared C6 subprocess audit; static coverage and correctly marked checkpoint state are not sufficient to pass that gate.

No training, GPU allocation, model invocation, unit execution or source changes were performed. Only source/artifact reads, existing checkpoint loads onto CPU, and these supervisor audit files were written.
