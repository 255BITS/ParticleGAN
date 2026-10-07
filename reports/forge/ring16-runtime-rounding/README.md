# Ring16 runtime rounding: CUDA accumulation-order diagnostic

**Serializing autograd only at update 401 eliminates the measured live/reload
discrepancy.** The two ordinary CUDA arms reproduce the historical live and
reloaded critic gradients exactly. The two serialized arms instead produce
identical critic gradients and **identical entire 401-update contexts**, including
G, D, prior, optimizers and every consumed RNG stream. The serialized result
matches neither ordinary trajectory: it is a third numerical path, with full
1,600-update quality still unmeasured.

The experiment isolates the execution constraint exposed by
[`GANTrainer(serial_backward=True)`](../../../particlegan/training.py) disables
autograd multithreading for the whole update. Its documentation describes exact
CUDA continuation with higher-order penalties and explicitly warns that this
changes gradient summation order. That is the API's documented intent and
existing implementation. Here it is applied externally to the whole update 401,
including the `create_graph=True` derivative calls and final backward; all four
checkpoint declarations retain their original false-mode contract.

All four declared arms ran on an NVIDIA RTX A6000, through the public API,
using seed 0, the public deterministic initializer, the original learned
256-component MoG, batch 128 and constant learning rates. The original full
Ring16 PASS/FAIL outcomes and current inventory remain unchanged.

Before GPU exposure was restored, the exact CUDA-only campaign command was
actually invoked and refused before any arm or output directory was created.
The [execution-blocker receipt](execution-blocker.json) preserves its command,
source/protocol hashes and local stdout/stderr hash: **zero scientific attempts,
updates or scored draws** in that refused invocation. It remains preserved under
its original source; the completed CUDA execution uses source
`fa60ca4af1fc872e650a6c4b13ca35d0a755a0ca` and the unchanged ready protocol.

## Completed comparison

[Compact CUDA results](cuda-results.json) retain original receipt identities,
gradient/state equality, graph metadata and source hashes. These are causal
diagnostic comparisons, with no technique ranking or quality qualification.

| Compared update-401 pair | Critic gradients exact | Whole context exact | Hidden critic polar relative difference | Changed graph sequence relations |
| --- | --- | --- | ---: | ---: |
| Ordinary live / reloaded | No | No | .25199047 | 1,054 / 7,503 |
| Serialized live / reloaded | **Yes** | **Yes** | **0** | **0 / 7,503** |
| Ordinary live / serialized live | No | No | .28366329 | Recorded in receipt |
| Ordinary reloaded / serialized reloaded | No | No | .31745353 | Recorded in receipt |

The complete 400-update contexts match the archived digest exactly. All 24
4,096-sample prefix observations match between the two fresh arms and the
original retained prefix. At 401, real batches and the first six forward inputs
and outputs match in every comparison. Every named RNG state matches after the
update. The ordinary gradients reproduce all six historical critic parameter
gradients bit for bit, so the graph inspection preserves the original observed
boundary discrepancy.

The [actual-training GIF](media/shared-prefix.gif) shows the common ordinary
prefix through update 400; its [index](media/index.json) binds both fresh CUDA
observation files. It does not imply that update-401 differences are visually
resolved, or that the serialized trajectory later passes quality.

**Causal gate: PASS for removing the measured 401 discrepancy with serialized
execution. Quality gate: INCOMPLETE by design.** This is one fixed-state,
same-seed intervention, not a seed study or an independent convergence result.

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

## What changed in the runtime

The baseline checkpoint has `serial_backward=False`, which inherits the
caller's autograd multithreading setting. Legacy checkpoints do not preserve
that ambient setting. The BCAP update combines the logistic game gradient and
two penalty branches computed with `torch.autograd.grad(create_graph=True)`.
The final `loss_d.backward()` merges contributions from this higher-order graph.

The tested hypothesis is that rebuilding runtime objects changes the order
in which those contributions are added to the same critic weight gradient.
Floating-point addition rounds after each operation, so changing the order can
change the final few bits even when every input, weight and forward value is
identical. The measured SVD normalization can then magnify those few bits in
almost-zero singular directions. This explains how the observed pattern could
arise. The paired CUDA result now supports the ordering mechanism, while the
exact executed node sequence remains unrecorded.

The installed PyTorch `2.14.0+cu130` primary headers provide a specific explanation:

- `torch/include/ATen/SequenceNumber.h:7` describes the node enumeration as thread local.
- `torch/include/torch/csrc/autograd/node.h:117` explains that sequence numbers from different threads have no guaranteed relative order; line 151 obtains the incrementing value when constructing a node.
- `torch/include/torch/csrc/autograd/node.h:349` describes how larger numbers take priority, and that parameter accumulators receive `UINT64_MAX`.
- `torch/include/torch/csrc/autograd/engine.h:102` compares ready-node sequence numbers; `input_buffer.h:52` exposes accumulation of multiple contributions.

The [verification receipt](cuda-verification.json) records each installed header's
exact path, hash and cited lines. These citations describe implementation of the
executed version, rather than an assumption about another PyTorch release.

All four actual critic graphs have the same 123-node topology. With ordinary
execution, live nodes occupy two sequence-number ranges: 46 below 20,000 and
69 above 50,000, consistent with different histories of main/worker counters.
After reload, all 115 non-accumulator nodes have small numbers. This changes
1,054 pairwise priority relations, including newly equal ties. Parameter
accumulators themselves retain their maximum sequence number.

With serialization, the two graphs' relative sequence relations agree exactly,
despite the fresh graph's large absolute offset. This supports the inference
that runtime-local thread counter history changes ready-node priority and
floating-point accumulation. **Graph metadata records priority, not actual node
execution order.** Disabling multithreading also changes thread execution, so
the specific internal addition/kernel is not independently isolated here.
The earlier buffer/mode/statistics/double-load controls retain their original
scope and do not explain this measured difference.

Parameter version counters and object/storage identity change when rebuilding.
Recorded tensor shape, stride, contiguity, dtype and device agree. Counter or
address differences alone cannot explain the arithmetic; do not treat them as
proof of a cause.

## Boundary first; every-step behavior is a separate question

The frozen [protocol](protocol.json) investigates **update 401**, immediately
after the user's 400-update boundary. It leaves the first 400 updates unchanged.
Disabling autograd multithreading for that whole update distinguishes a local
execution effect from retraining an entirely different trajectory. It does not
separate thread choice during higher-order graph construction from final
backward scheduling.

| Completed diagnostic arm | New updates | Change |
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
All four complete; measured whole-subprocess debit is **24.795115167 seconds**.
This is campaign cost, not a speed comparison. Fresh arms add 48 scored draws;
restored arms add none. The campaign is now fully consumed, with no automatic
continuation or repeat authorized within its declaration.
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

The preregistered equality prediction passes, with historical instrumentation
parity also passing. The serialized result selects a **third trajectory**; it
does not reproduce the ordinary restored gradient known to lead to PASS.
Following this state to 1,600 is a separate budgeted question, needed to establish
whether a one-boundary intervention repairs quality. No such outcome is claimed.

For a continuous learner, setting the public `serial_backward=True` **from the
start and on every step** is a separate global trainer candidate worth evaluating
after this control. Its fresh trajectory differs, and its checkpoint mode must
remain true across restore. The API rejects loading a false-mode checkpoint
into a true-mode trainer; do not edit that field to bypass the contract.
No all-step candidate or conversion arm is declared or paid in this campaign.

## Execution and limitations

In the earlier blocked execution, the process had no NVIDIA device nodes.
The parent host inspection found
an installed NVIDIA kernel module and a loadable `libcuda.so.1`, but `cuInit(0)`
returned `100` (`CUDA_ERROR_NO_DEVICE`). Its mount inspection found `/dev`
overlaid by a `nodev` tmpfs. PyTorch reports zero CUDA devices and
`torch.cuda.init()` reports “No CUDA GPUs are available”; `nvidia-smi` cannot
communicate with a GPU. This identifies missing GPU exposure in this execution
namespace, without establishing a driver installation failure. The exact parent
inspection identity is retained in the blocker receipt. Device/mount/driver
changes were not attempted.

GPU exposure subsequently became available after the execution environment
changed; both physical A6000s and CUDA were visible. The completed campaign
selected physical GPU 0 with `CUDA_VISIBLE_DEVICES=0`, logical `cuda:0`.
Its original command is retained for reproduction. **The output root is now
populated and must not be reused; no unchanged campaign repeat is recommended.**

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

The saved-only [boundary comparison](compare_boundary.py) and
[renderer](render_prefix.py) use retained tensors, with zero model calls or new
random draws. Direct helper invocations require `PYTHONPATH=.` from the checkout
root. An initial comparison invocation lacking that import path failed without
training; its error is archived, followed by successful corrected analysis.
[Archive receipt](archive.json) preserves the exact raw training states, traces,
observations, stdout, source manifests and analysis logs outside Git.
Original historical evidence remains bound to
[PR331](https://github.com/255BITS/ParticleGAN/pull/331).

Source and bindings are frozen in the protocol. Diagnostics operate through
the public ParticleGAN API, use seed 0 and named deterministic initialization,
retain every consumed RNG stream, preserve constant rates, and grant no
qualification. [The current inventory](../technique-inventory.md) remains the
single generated leaderboard. This report changes no production optimizer,
default, task, qualification result or inventory row. The next scientific
question is whether a globally declared every-step serial trainer continuously
acquires and retains Ring16, or whether a single-boundary intervention suffices;
both need new frozen budgets and their own actual quality gates.
