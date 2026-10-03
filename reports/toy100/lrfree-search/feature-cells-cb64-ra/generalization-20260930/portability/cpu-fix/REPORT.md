# CPU optimizer context repair

PASS: all 22 original RA12 contracts plus 13 focused optimizer checks with CUDA_VISIBLE_DEVICES=0. Both blocked-lazy-init and normal Torch runs passed 35 checks; CUDA remained uninitialized. Four saved stream buffers remain CPU uint8. Each final run took 9.16 seconds with concurrent CPU proofs.

The first attempted initialization comes from the inherited Torch 2.13 Adam accelerator graph check on the first CPU critic step, before the feature reaction. The caller-created table Adam has the same upstream behavior.

The separate candidate changes only k3p.py and three lines at the end of UpdatePolicy initialization. K3P optimizers skip accelerator checks for CPU parameters. Exact caller Adam/AdamW instances registered on CPU receive the same instance-local guard, preserving any caller instance overrides. Any non-CPU parameter delegates to the original check. CUDA dispatch was verified with device marker objects; no CUDA tensor was constructed.

CPU parameters and Adam moments match the reference exactly across three fixed updates for all four optimizer types. The original feature reaction, reference fallback, checkpoint/replay, serving, table ownership, routed and atomic rejection contracts pass unchanged. The original pkg-RA12-auto and its original contract file hashes match the source closure.

Apply cpu-adam-context.patch to the integrated package; add test_cpu_optimizer_scope.py to repository tests. Frozen source was untouched. This receipt does not replace parent-owned CUDA numerical validation.
