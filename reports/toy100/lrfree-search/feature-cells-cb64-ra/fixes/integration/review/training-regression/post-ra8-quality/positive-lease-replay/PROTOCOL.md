# Actual RA8 positive serving lease: sample and reload contract

This is a separate mechanical CUDA contract, pending root execution. The
original ten-update replay from update1000 remains unchanged and required.
This contract makes no quality decision and performs no training update.

## Fixed inputs and budget

Use only frozen CB64-RA8 sources/config and its sealed toy checkpoint2000.
The real saved paired stamp must have coherent_rows977, required973,
eligible=true, a matching snapshot serial and a live FIFO turnover lease.
No stamp, expiry, model, sigma, controller or optimizer field may be changed.

Two independently constructed trainers restore the same complete CUDA state.
Each executes exactly the first two256-row chunks of the original toy scorer
generation sequence. The scorer uses its original SEED+100=314259 on cuda:0;
this reuses an existing evaluation seed and does not select a new seed.
The original scorer has8192 draws in256-row chunks. This contract repeats
only its first512 draws per branch, with no oracle/scorer invocation.

Both calls use ordinary sample(256, generator=original_scorer_stream), with
the default ema=false. A positive saved lease must activate the existing
live G/prior swap to their paired averages. The original _generate and
perturb_latent methods are called once, unchanged, per sample batch. Trace
wrappers only clone their inputs/outputs and RNG states; they add no random
draw, generator forward or learned-feature evaluation. The original
learnable output-noise rule and bounded local latent perturbation remain in
effect, and effective output sigma must be finite and positive.

After the first chunk, both branches save the complete FAST training state
returned by state_dict. Branch0 continues normally. Branch1 writes and
reloads that actual state plus the already advanced scorer generator state
before the second chunk. There are two independent initial restores and one
same-trainer intermediate reload, with no new state evidence fabricated.

## Required comparisons

- Both initial restores reproduce every serialized trainer section and its
  complete typed bit fingerprint, including the actual positive stamp.
- The serving swap is live: G/prior parameters equal their own paired EMA
  parameters, the retained FAST parameters equal the saved FAST checkpoint,
  and FAST differs from the paired averages in at least one parameter.
- For both chunks, sampled row IDs, corresponding unperturbed codes,
  actual perturbed codes, effective sigma and noisy _generate outputs are
  bit identical across branches. Incoming codes equal saved EMA prior rows.
- Scorer stream state advances and matches at every corresponding cursor,
  including after the intermediate save/load. All saved training streams,
  backend stream and global CPU/CUDA RNG states remain bit identical to the
  original checkpoint after each chunk, establishing exact serialized RNG
  continuation. No extra test draw from these streams is added.
- Every serialized trainer section remains exactly unchanged during sample
  calls. No timing or other state leaf is excluded. Module modes, existing
  gradients and registered buffers are also unchanged by each sample.
- Loading discards the ephemeral chart, learned heads and latent axis cache,
  while keeping the positive typed lease and current serving swap. Sampling
  leaves the chart absent and deterministically rebuilds the existing
  versioned latent axis cache. Two subsequent cache accesses are warm and
  draw no randomness. The rebuilt order fingerprint agrees across branches,
  even after the intermediate reload. Sample work is bounded by256 query
  rows and64+8 local/lineage candidates per row, with no N-squared pass.

The post-chunk state_dict call can change parameter versions while releasing
and reapplying serving parameters. The ordinary branch therefore also
rebuilds a stale axis cache on its next sample. The reload branch additionally
clears the cache. Parameter versions/cache identities/work counters are
derived diagnostics, not serialized semantic state or bit-comparison inputs.

## Freeze and execution

prepare_contract.py hashes the helper/protocol, original CUDA harness/scorer,
all29 package modules, config/READY/lane receipts, root serial wrapper and
its COMPOSITION, and the final checkpoint/run receipt before any numerical
read by this contract. SOURCE-FROZEN.json is the pre-execution seal.
CPU preflight may only load this checkpoint onto CPU to inspect its typed
metadata and CPU uint8 RNG buffers, compile source and verify the fixed
interfaces. It does not construct a model/trainer, restore or consume an RNG,
call sample or a forward, create CUDA contexts, or inspect quality metrics.

Only root launches the CUDA command, serially after the active grid job
finishes. The helper uses the original common runtime/physicalGPU0 identity
and learned lock. Root's wrapper supplies the inherited serial phase lock.
The original input files are checked before and after execution. The output
result and branch evidence are exclusive: an earlier attempt is retained.

The CUDA workload is four256-row toy G forwards, two independent checkpoint
restores, one intermediate reload, four bounded latent queries and exact
state/trace fingerprints. Expected GPU numerical time is a few seconds;
Python imports, source hashing, checkpoint validation and host transfers may
make total elapsed time tens of seconds. These are estimates, not a measured
runtime. No D forward, reaction, birth solver, optimizer step or real batch
is used. Passing this contract would certify this saved lease's sample/reload
mechanics on the owned CUDA runtime; it would not qualify quality, arbitrary
models, future leases or training continuation from a positive lease.
