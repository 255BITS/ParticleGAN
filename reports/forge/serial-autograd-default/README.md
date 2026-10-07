# Fixed autograd scheduling policy

The user requested that project training, experiments and example benchmarks
disable Torch's multithreaded autograd scheduling after the Ring16 reload
investigation exposed sensitivity to gradient accumulation rounding. This is a
fixed execution policy, combined with the independently merged DualNorm
truncation default. It is not a new distribution-quality qualification.

Importing `particlegan`, `benchmarks` or `experiments.forge` disables autograd
multithreading on the importing thread. This covers current example and
standalone experiment component loops as well as module-launched benchmarks.
`GANTrainer.step` also enforces False around the complete update, including
forward graph construction, higher-order penalties and backward. The common
Forge adapter and toy comparison runner enforce the same scope in worker
threads and when called inside an explicitly enabled external context. Those
scopes restore caller state on exit, including failure.

The policy uses Torch's existing context manager. It does not replace Torch
functions or change CPU operation thread pools, CUDA parallelism, model
architecture, batches, priors, learning rates or budgets. Current runtime
receipts and Forge compute profiles record
`autograd_multithreading_enabled: false`; prospective toy comparisons use the
new `toy-comparison-v3-serial-autograd` identity.

`GANTrainer(..., serial_backward=True)` remains accepted as a compatibility
assertion, and the property always reports True. False is rejected. Checkpoints
record True. Existing serialized-mode checkpoints can load when their remaining
recipe, optimizer and execution bindings match. Unmarked or False historical
checkpoints must resume from their pinned original source: automatically loading
them under a different accumulation order would change their continuation
contract. Archived research sources, qualification receipts and reports retain
their original identities.

The original common26 canonical image and MoG owners are strictly pinned
historical reproduction paths, outside the ordinary Tier 1 roster. Their False
contract markers and original source pins remain unchanged. Current direct
factories now explicitly report `BLOCKED` before constructing models if their
pinned GANTrainer source differs, instead of reaching an incompatible supplied
trainer assignment or silently changing the old cohort. Recover those owners
from their archived original source; the current ordinary Forge and toy adapters
use the fixed policy directly.

Torch's setting is thread-local. The public owned scopes enforce it in new
threads. Caller-owned component loops run in a new thread, or inside a context
that deliberately re-enables Torch scheduling, should wrap the complete update
in `with particlegan.serial_autograd():`. The project does not prevent unrelated
external code from changing Torch settings globally.

Six CUDA software checks passed on physical GPU1 (RTX A6000), including default
BCAP with the merged truncation implementation, forward/backward enforcement,
exact checkpoint continuation, atomic legacy rejection, failure restoration,
worker scopes and existing KA2 CUDA continuation. Five integration checks took
3.05 seconds; the additional legacy-owner preflight check took 0.28 seconds with
model and optimizer construction fenced.
The bounded checks consumed nine successful tiny public updates across pre/post
integration; no quality campaign ran in this PR. Metadata-only Forge validation
and inventory coverage also passed (11,776/11,776 tracked sources).

The [compact receipt](verification.json) binds checked code and retained local
logs. Logs remain under
`runs/software/serial-autograd-policy/pytest-final.log`. The next quality work is
a fresh GPU Tier 1 campaign on the merged source, followed by Tier 2 only for
families that pass its required gates; previous source cohorts retain their
original grades.
