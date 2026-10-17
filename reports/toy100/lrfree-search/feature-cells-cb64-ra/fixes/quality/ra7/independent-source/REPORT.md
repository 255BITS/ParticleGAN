# Independent RA7 source and configuration proof

PASS for prospective source/configuration readiness. The full29-module RA7 package is byte-identical to the frozen GROUP-COUNT proposal and its existing CPU/CUDA mechanical proofs. Relative to RA6,28 modules are byte-identical; replacing only FeatureCellSnapshot._group_counts with its original method restores the whole feature_cells.py bytes and AST exactly. Owner sources, inputs, retained proofs and base numerical maps are unchanged.

Exactly three override values change: lr .00425→.0010625, prior_lr_mult2→8, d_lr_mult1→4. Generator and learned log-sigma base learning rates both become one quarter. Prior base LR stays .0085; critic base LR stays .00425. The unchanged per-group stationarity scales, applied/base intrinsic clocks and critic .75 prior-scale floor remain active. Learned sigma is coupled to this generator-base change; it is not a G-only ablation.

No source law, population survival threshold, count family, birth/copy budget, sampling/serving rule, noise formula/floor, checkpoint schema or hidden state changes. This audit executes no Torch/model code, numerical test, CUDA context, new seed or optimizer update. Existing mechanical parity is separate from trajectory quality. The candidate has no quality result; both the strict learned toy and original full Grid100 still must pass.
