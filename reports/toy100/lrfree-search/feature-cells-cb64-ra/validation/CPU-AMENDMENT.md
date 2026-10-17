# Environment amendment, declared before candidate outcomes

On 2026-09-29 the managed execution workspace lost access to CUDA. The local
minimal check returns `torch.cuda.is_available() == False`, device count 0,
and no `/dev/nvidia0`, `/dev/nvidiactl` or `/dev/nvidia-uvm`. Approval policy
is never. This amendment supersedes only GPU/device and canonical fixture
identity clauses in PROTOCOL.md; quality thresholds, schedules, budgets,
task constants, package/config freeze and one-seed requirements remain.

## Matched learned-model comparison

All four E22/CB64-RA × toy/MNIST runs use CPU, two intra-op threads and one
interop thread, deterministic PyTorch algorithms, the same seed 314159,
2000 updates, previous architectures and read-only saved real data streams.
The deterministic CPU-QR recipe initialization is unchanged and all initial
G/D/prior hashes must match the previous GPU study. If a hash mismatches,
record an error, without silently relabeling the fixture. CPU latent, penalty,
controller and output-noise RNG draws are matched between the two packages;
their sequences are different from historical CUDA draws. The trained evaluator
and metric formulas are unchanged. Evaluation draws likewise use matched CPU
RNGs and are not claimed to reproduce historical GPU evaluation clouds.
Update timers measure CPU trainer calls; report contemporary CPU throughput,
whole-run time and process peak RSS. GPU peak fields are null. Matched-wall-time
comparison uses only the two contemporary CPU runs within each task.

## Original portability and full native scorer diagnostics

Canonical GPU screen acceptance is **UNAVAILABLE** in this environment. It is
separate from quality: no CPU result is labeled canonical GPU acceptance.
An owned mechanical adapter of the original frozen screen runs all original
13 portability tasks and three full 7000-update native tasks on CPU, one thread,
with original hosts, scorers, thresholds, budgets and seeds read only. A source
diff and replacement receipt are frozen before execution. No candidate code or
scorer is patched and strict candidate data/latent transaction checks remain.

CUDA devices/generators/fork scopes become CPU equivalents. Canonical GPU
fixture hashes and RNG cursors are saved as identity comparisons, without
forcing CPU tensors or cursors to match them. CPU mode_hold batch receipts are
saved separately; two candidate latent draws are still checked on each update.
Image hosts reset their shared CPU global data/latent stream to seed 0 after
CPU constructor draws and before the trainer, matching the original host's
untouched CUDA stream transaction. Native prior normal→uniform draw and G/D
constructor order remain; actual CPU initial hashes/range are recorded against
the canonical GPU expected hashes. Canonical tensor assets for those full
initial fixtures were not found in the targeted archive inventory; they are
not reconstructed from partial snapshots or relabeled as canonical.

Each native run still has all 34 observations, the five final 20000-sample
quality checks and 100000-sample holdout. The unchanged official coverage and
accuracy scorer processes judge this CPU evidence. Report their PASS/FAIL/ERROR
as **CPU diagnostic scorer verdicts**, alongside host/RNG identity differences.
Archived E22 GPU verdicts/timings are noncontemporary reference evidence only.

## Checkpoint semantics

Replay 10 next updates from each saved 1000-update checkpoint twice. Require
bit equality of losses and every semantic checkpoint section: models,
optimizers, streams, CPU RNG, recipe, controller, lr_settle, birth_death,
row_evidence and all other state. Exclude only
`birth_death.last.eval_seconds`, an observational wall-time diagnostic.
Keep raw section/whole hashes, semantic hashes and evaluation/move counter
deltas for each branch. Wall-time differences cannot mask any other state
inequality. These replay updates do not count toward the quality budget.

Candidate source/config execution still waits for integration READY. No seed
sweep, outcome-driven tuning, shortened native run or quality-gate change is
authorized by this environment amendment.
