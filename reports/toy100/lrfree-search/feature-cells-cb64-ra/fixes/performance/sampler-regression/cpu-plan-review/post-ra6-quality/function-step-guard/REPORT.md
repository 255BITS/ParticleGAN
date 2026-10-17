# Reserved function-motion guard: negative feasibility result

The fixed design is **not a production repair**. It leaves the late saved proposal unchanged. No settings or thresholds were tuned after observing this result; RA8 contains no G guard.

| Fixed saved state | Represented real mass | Active | Full-step motion q95 | Budget | Accepted G fraction |
| --- | ---: | --- | ---: | ---: | ---: |
| RA7 step 100 | .239583 | No | .452236 | .223607 | 1 |
| RA7 step 2000 | .916667 | No | .207192 | .223607 | 1 |

The early neutral fallback preserves exploration. The late activation also stays off; its full motion would satisfy the displacement budget even if activation were forced. Five of 128 late probes change fitted cell and two initially inside probes become outside. The eight-dimensional late covariance chart has a median of two even real points per cell, so shrinkage supplies most directional information. A small fresh subset cannot establish fine support width for every thin or rare component, especially grid100. The allowance for initially distant probes and the 95% motion quantile also permit motion that may accumulate across many steps.

## Mechanical evidence

Two saved checkpoint cases repeat the recorded beta1=0 gradient in private joint Adam/A2 optimizers. This is not the actual next gradient or a reconstruction of historical actions. Full moments, A2 history, step counts, prior and sigma results match the unguarded private proposal exactly. G interpolation cannot modify them. Applied/base LR accounting is explicit; rejected motion would have zero G intrinsic increment. EMA parameters are computed once after acceptance with the original table-derived rate (.0625 early, .00390625 late), without serving or noise changes.

Mixed module modes, BatchNorm buffers, Dropout behavior, hooks and CPU RNG are preserved. A controlled stochastic/stateful evaluation callback is rejected and restored. Serialization of the saved inputs followed by deterministic chart reconstruction reproduces the exact decision and trial records. Three small directional/halving/endpoint controls exercise a successful quarter fraction, four-trial rejection, direction-dependent covariance and bit-exact zero/full parameter endpoints. These are private mechanical controls, not additional quality trajectories.

All 29 frozen RA7 modules, recipe/READY bytes, input checkpoint bytes and loaded checkpoint tensors remain unchanged. No oracle is imported. No CUDA context, new seed, whole-table G forward, training step or emitted quality evaluation is used.

## Integration and scaling limits

The existing backend snapshot is ephemeral and omitted on load. Its callback uses current D; mixing it with a stale head epoch would change the geometry. This design instead recomputes a matched bounded chart from the last 128 real rows plus 256 deterministic older FIFO rows and uses only 128 detached probes. Its private generator copies saved CPU RNG state and never advances a production stream. A production integration would need a declared policy/schema and same-law continuation tests; this proof does not establish cross-device parity.

Each decision adds one D forward on at most 384 real rows, one clean G+D baseline on 128 probes, and at most four trial G+D forwards. Parameter backup/delta cost is O(P_G); covariance solves are bounded by 64 cells and rank eight. Reference fitting and the original cold MST still contain scalar decisions that could be expensive on CUDA. Their cost has not been profiled. Images may incur material forward and memory costs even though no work scales with the full particle population inside a trial. Unsupported backends or invalid real geometry need a declared neutral original-update fallback.

The proof passes its mechanical contracts; the proposed quality repair fails this fixed feasibility criterion and remains reserved. [receipt.json](receipt.json), [controls.json](controls.json), and the retained logs contain the complete evidence.
