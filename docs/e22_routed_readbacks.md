# Many-site E22 synchronization

DV12 diagnostic collection is deferred for both native `e22` and
`e22_routed`. The training law, reduction arithmetic, noise draws, and
checkpoint schema are unchanged. There are no new recipe options.

Previously, every recorded routing site converted five statistics to host
floats, even though only the last two applications were retained. Two
71-site training forwards therefore performed 710 diagnostic readbacks.
Each site also read finite-logit and finite-softmax predicates, adding
284 readbacks. Those 994 readbacks scaled with the number of routing sites.

The controller now retains detached radius, clipped displacement, and clip
fraction tensors for only the last two applications. It computes their five
statistics when `controller.latent_applications`, `controller.diagnostics()`
or `controller.state_dict()` is requested, and transfers pending statistics
in one batch. Earlier applications are discarded without reducing them.
The public result remains the same list of float dictionaries. Repeated
reads reuse the materialized list; checkpoint recovery also accepts the
previous controller dictionary. Assignment to `latent_applications`,
including `[]`, remains supported.

Pending storage retains at most two applications, with
`2 * batch * tokens * (latent_dim + 2)` floating elements for equally sized
sites. It retains no autograd graph. For example, batch 48, 128 tokens,
latent dimension 4 and FP32 use about 288 KiB. Reading diagnostics releases
these tensors after materialization. The original autocast context is
preserved for the deferred arithmetic.

On accelerators, `RoutedExecution.mix()` keeps detached scalar predicates
for logits and softmax weights. `finish()` checks all sites with one host
readback before the complete model result returns. CPU finite checks and
shape/dtype/device/site-order checks remain immediate. A NaN/Inf site or
invalid mass still raises `ValueError`; accelerator value errors are reported
at completion of the full callback. Execution cleanup releases all pending
predicates, candidates and usage references on success or callback failure.
The single-site callback contract also combines finite, nonnegative and
normalized-weight checks into one readback.

## Reproducing the overhead measurement

[`examples/e22_routed_readbacks.py`](../examples/e22_routed_readbacks.py)
uses 71 sequential sites with dependent inputs and one 128×4 bank. It has
frozen BF16 hosts, FP32 trainable owners, source/time conditioning and full
E22 row controls. Its explicit `probe_interval=1000` isolates updates between
structural evaluations. This is a small overhead fixture; it does not predict
Supra throughput or establish output quality.

```bash
PYTHONPATH=. python -u examples/e22_routed_readbacks.py \
    --device cuda --sites 71 --steps 20 \
    --output /tmp/readbacks.json --state-output /tmp/readbacks.pt
```

Profiling counts host `aten::_local_scalar_dense` operations separately from
timing. The forward-pair count covers one no-grad critic-side generator pass
and one differentiable generator-side pass. The complete-update count also
includes controller/evidence/optimizer work and the example's caller logging.
Explicit diagnostics, checkpoint reads, table geometry validation, and
algorithm decisions can still synchronize. The removed cost is the repeated
per-site diagnostic and validation readbacks.

Measured on an RTX A6000 with PyTorch 2.13.0+cu126, comparing revision
`f459cb6d` with this change, using 4 contexts and 8 tokens per site:

| Measurement | Before | After |
| --- | ---: | ---: |
| Host scalar reads, two generator forwards | 1,014 | 22 |
| Host scalar reads, complete update with caller logging | 1,088 | 96 |
| Synchronized milliseconds per update, 20 updates | 165.34 | 150.93 |
| Structural evaluations | 0 | 0 |

The former 994 site-dependent reads become two complete-forward validation
reads. The remaining 20 forward reads come from representation geometry.
The observed time reduction is 8.7%; these sequential runs shared the GPU
with other work, so the timing is provisional. The readback count is the
direct regression target. The [machine-readable receipt](e22_routed_readbacks_results.json)
records configuration, runtime hashes, recovery comparison and timing scope.

The benchmark fixes public initializer keys, data and noise streams, and
ambient RNG states. Kernel warmup is restored before measurement. Before/after
recovery receipts compare full policy/controller/optimizer/evidence states,
all RNG streams and update traces exactly, including serialized diagnostics.
The focused CUDA regression also checks that one-site and 71-site forward
pairs have equal scalar-readback counts. Existing conformance tests continue
to cover accepted moves, BF16 serving, checkpoint replay and native E22 parity.

Validation passed: 1,367 full-suite CPU cases plus 10 separately collected
DV12 cases; 52 CUDA cases, including resume across accepted structural moves,
learned-noise initialization, frozen BF16 and activation replay. The CPU run
skipped CUDA cases and the existing opt-in real-data/research checks; the GPU
run skipped two tests that require CUDA to be absent. Wheel/source builds,
strict metadata checks, and installed native/replay/readback examples passed.
Recipe/config defaults and R1 remain unchanged.

## Other integration costs

`RoutedRows(probe_interval=K)` already sets the separate, checkpointed
structural-probe clock. Gradient evidence continues between probes. Each
candidate still reruns the complete caller-owned model so dependent sites
and final-output guards remain correct. Tune cadence using the integration's
quality measurements, retaining the fixed and movable-bank controls.

Caller-owned teacher/student passes, frozen-model checkpoint storage, and
shared-GPU contention remain separate costs. This change preserves exact
frozen-weight copies in averages and the complete recovery/snapshot contract.
