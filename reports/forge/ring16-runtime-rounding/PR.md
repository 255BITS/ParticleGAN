Ring16 restart sensitivity first appears in backward gradients despite identical
float32 model state and initial forwards. This report audits saved tensor bytes
and checkpoint loading, finds no evidence of a lossy conversion before backward,
and identifies the existing public serial-backward control as the next bounded
CUDA diagnostic.

Adds a frozen four-arm 804-update protocol and public-API runner comparing live
and restored update401 with ordinary versus serialized autograd execution.
The intervention applies immediately after400; every-step execution and any
newly identified conversion remain distinct future experiments. Reports ULP
counts, source/input hashes and causal limitations without changing optimizer
behavior, qualification results or the technique inventory.

Hard subprocess timeouts cover setup and persistence. Full state is retained
before metadata checks; interrupted campaigns retain errors, conservative costs
and unexecuted peer statuses without retries.

Validation: saved-tensor analysis, Python syntax, frozen bindings and mocked
controller timeout/failure/success accounting checks with zero model calls.
CUDA neural execution is blocked by unavailable GPU exposure; no CPU fallback, new
training PASS or actual-training GIF is claimed. Publication is pending network
access.

The requested CUDA campaign was actually invoked and exited at its CUDA
preflight, before any arm or output directory was created. Its command, source
bindings and local stderr hash are retained in `execution-blocker.json`.
Kernel-driver metadata is present, but this process has no NVIDIA device nodes
and `cuInit(0)` returns CUDA_ERROR_NO_DEVICE. The blocker is GPU exposure in
the execution namespace; no driver-reinstall conclusion is justified. Zero
scientific attempts or updates were consumed, and the boundary hypothesis
remains UNTESTED.
