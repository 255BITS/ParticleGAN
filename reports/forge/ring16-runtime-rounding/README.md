# Ring16 runtime rounding: saved-state audit and next CUDA diagnostic

**The saved evidence does not show a floating-point conversion causing the
restart difference.** All 25 model-state tensors are float32. The complete
400-update checkpoint, critic weights before update 401 and first six forwards
are byte-identical between live and restored execution. The first difference
is still the newly computed critic backward gradient.

An existing public execution option provides a concrete next control:
[`GANTrainer(serial_backward=True)`](../../../particlegan/training.py) disables
autograd multithreading for the whole update. Its documentation describes exact
CUDA continuation with higher-order penalties and explicitly warns that this
changes gradient summation order. That is the API's documented intent and
existing implementation, **not a new verification for this Ring16 recipe**.

No neural experiment ran in this investigation: CUDA is unavailable in the
current execution environment. CPU work only inspects saved tensors and source.
The original Ring16 PASS/FAIL outcomes and current inventory remain unchanged.

On the user's request to execute, the exact CUDA-only campaign command was
actually invoked and refused before any arm or output directory was created.
The [execution-blocker receipt](execution-blocker.json) preserves its command,
source/protocol hashes and local stdout/stderr hash: **zero scientific attempts,
updates or scored draws**. The four-arm campaign remains unconsumed.

## Saved numerical findings

The [saved-only analyzer](analyze_saved.py) reproduces the following results from
the original PR331 tensors. [Its compact receipt](saved-results.json) retains
every input file hash and the original full-context digest
`208e2d7b241ffeac11915972dbb90979dd686ad5285768282e2182d7da5ad2cd`.

| Critic gradient at 401 | Changed float32 elements | Maximum absolute difference | Maximum ULP distance |
| --- | ---: | ---: | ---: |
| First weight, 64 × 10 | 346 / 640 | 5.59 × 10⁻⁹ | 96 |
| Hidden weight, 64 × 64 | 2,647 / 4,096 | 3.73 × 10⁻⁹ | 2,560 |
| Output weight, 1 × 64 | 41 / 64 | 1.30 × 10⁻⁸ | 32 |
| All three bias gradients | 0 / 129 | 0 | 0 |

ULP distance counts representable float32 values between two outputs. Large ULP
counts near zero can accompany tiny absolute differences; they do not establish
a dtype conversion. Relative hidden-gradient difference is about `1.03e-7`.
The earlier CUDA SVD probe measured its normalized direction difference at about
`.252` relative Frobenius norm. This audit performs no new SVD or neural call.

## What checkpoint loading actually does

[`GANTrainer._load_state_dict`](../../../particlegan/training.py) checks saved
model dtype against the constructed model dtype, then invokes each model's
`load_state_dict`. Both models and checkpoint tensors are float32 in this cohort.
The original diagnostic saves detached CPU copies and maps the file to CPU
before loading it back into CUDA models. This is a device transfer; the audited
path contains no fp16/bfloat16 conversion or model-precision change.

The saved tensors also survive a float32 → float64 → float32 round trip exactly
in CPU metadata arithmetic. That additional check is a property of those finite
saved values; it is not a new CUDA transfer experiment. The stronger observation
is the original measured equality of all serialized state and initial forwards
after the actual CUDA reload.

Therefore, a lossy conversion of the **stored weights before backward** is not
supported as the explanation. A kernel's internal arithmetic precision or an
unrecorded intermediate remains a separate hypothesis; neither is established
by dtype metadata. If a future graph/kernel trace identifies an actual changed
conversion, declare that conversion as its own bounded intervention.

## Why accumulation order is worth isolating

The baseline checkpoint has `serial_backward=False`, which inherits the
caller's autograd multithreading setting. Legacy checkpoints do not preserve
that ambient setting. The BCAP update combines the logistic game gradient and
two penalty branches computed with `torch.autograd.grad(create_graph=True)`.
The final `loss_d.backward()` merges contributions from this higher-order graph.

The concrete hypothesis is that rebuilding runtime objects changes the order
in which those contributions are added to the same critic weight gradient.
Floating-point addition rounds after each operation, so changing the order can
change the final few bits even when every input, weight and forward value is
identical. The measured SVD normalization can then magnify those few bits in
almost-zero singular directions. This explains how the observed pattern could
arise; it does not identify the responsible engine operation.

Locally installed PyTorch 2.14 engine headers order ready nodes using sequence
numbers; their input buffer explicitly accumulates multiple contributions.
Those implementation facts and the public serial-backward option motivate an
ordering control. **No saved graph or execution-order trace exists for the
original first differing backward, so this is a hypothesis, not an identified
root cause.** The old runtime-buffer/statistics/double-load controls ruled out
their specific deltas without recording engine execution order.

Parameter version counters and object/storage identity change when rebuilding.
Recorded tensor shape, stride, contiguity, dtype and device agree. Counter or
address differences alone cannot explain the arithmetic; do not treat them as
proof of a cause.

## Boundary first; every-step behavior is a separate question

The [prospective protocol](protocol.json) investigates **update 401**, immediately
after the user's 400-update boundary. It leaves the first 400 updates unchanged.
Disabling autograd multithreading only at that backward distinguishes a local
summation-order effect from retraining an entirely different trajectory.

| Planned diagnostic arm | New updates | Change |
| --- | ---: | --- |
| Fresh prefix and live graph | 401 | Read actual backward graph at 401 |
| Fresh prefix and serialized 401 | 401 | Same graph reads; disable multithreading only at 401 |
| Restored graph | 1 | Original full400 checkpoint; graph reads at 401 |
| Restored serialized 401 | 1 | Same restore; disable multithreading only at 401 |

These are diagnostic controls, not a technique leaderboard. The two fresh
prefixes are necessary to retain live runtime objects in independent processes;
loading the saved prefix would already substitute the restored numerical path.
The graph observation is a new explicit instrumentation cohort, checked against
the archived gradients. It reads topology/sequence numbers, adds no node
execution hooks, and invokes original backward/optimizer operations once.

Budget: **4 attempts, 804 new updates, 360 reserved seconds, zero retries**.
Controller hard timeouts include process startup, context construction, training,
persistence and exit: 150 seconds per fresh arm and 30 per restored arm.
Completed arms charge measured process wall time; interrupted arms retain their
original errors and consume their full reservations. The campaign summary marks
unexecuted peers explicitly. Full401 state and observations are saved before
final metadata validation, so a receipt error cannot discard completed results.
Fresh runs retain their 24 original-cadence observations of 4,096 clean samples;
restored arms add no scored sampling draws. Every fresh400 context must match
the archived digest exactly. Any mismatch stops that arm without substituting
the archived fixture. Compare actual inputs, raw gradients, polar factors,
graph topology/relative sequence order, named streams and full401 contexts.

If the serialized pair agrees while the ordinary pair differs, that shows this
execution constraint removes the measured boundary discrepancy. It does not
prove exactly which engine ordering detail changed or establish 1,600-update
acquisition/4,000-update continuous stability. If ordinary diagnostic gradients
do not reproduce the archived ones, report a new instrumentation/runtime cohort
and withhold historical causal attribution.

The causal test is therefore **UNTESTED**. Exact equality of the serial pair
would isolate the update-401 discrepancy to the changed execution constraint;
following the resulting states to 1,600 would still be necessary to establish
whether that one boundary intervention explains the full-quality PASS.

For a continuous learner, setting the public `serial_backward=True` **from the
start and on every step** is a separate global trainer candidate worth evaluating
after this control. Its fresh trajectory differs, and its checkpoint mode must
remain true across restore. The API rejects loading a false-mode checkpoint
into a true-mode trainer; do not edit that field to bypass the contract.
No all-step candidate or conversion arm is declared or paid in this campaign.

## Execution and limitations

The current process has no NVIDIA device nodes. The parent host inspection found
an installed NVIDIA kernel module and a loadable `libcuda.so.1`, but `cuInit(0)`
returned `100` (`CUDA_ERROR_NO_DEVICE`). Its mount inspection found `/dev`
overlaid by a `nodev` tmpfs. PyTorch reports zero CUDA devices and
`torch.cuda.init()` reports “No CUDA GPUs are available”; `nvidia-smi` cannot
communicate with a GPU. This identifies missing GPU exposure in this execution
namespace, without establishing a driver installation failure. The exact parent
inspection identity is retained in the blocker receipt. Device/mount/driver
changes were not attempted.

Use the unchanged, unconsumed protocol from this branch on a CUDA host. Its output
root must be absent; it is still absent after the failed preflight:

```sh
mkdir -p runs/reports/ring16-runtime-rounding
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 /home/martyn/dev/ParticleGAN/.venv/bin/python -u \
  -m benchmarks.toy_audit.ring16_runtime_rounding run \
  --prior-root /home/martyn/dev/ParticleGAN/runs/api/ring16-restart-diagnostic-v1 \
  --output runs/api/ring16-runtime-rounding-v1 \
  --device cuda:0 \
  > runs/reports/ring16-runtime-rounding/cuda.log 2>&1
tail -F runs/reports/ring16-runtime-rounding/cuda.log
```

After all four CUDA receipts complete, the saved-only
[boundary comparison](compare_boundary.py) creates the compact causal readout.
Render the retained actual training snapshots into a fixed-axis GIF before
publishing any completed neural-test result. No new training or GIF is claimed
here; the original media remain bound to
[PR331](https://github.com/255BITS/ParticleGAN/pull/331).

Source and bindings are frozen in the protocol. Diagnostics operate through
the public ParticleGAN API, use seed 0 and named deterministic initialization,
retain every consumed RNG stream, preserve constant rates, and grant no
qualification. [The current inventory](../technique-inventory.md) remains the
single generated leaderboard. GitHub publication also remains pending because
network access is unavailable in this environment.
