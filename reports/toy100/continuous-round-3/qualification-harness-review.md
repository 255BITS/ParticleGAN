# Long-run evaluator: integration not yet ready

The first preparation attempt reports ten passing harness checks, including a
three-update bare-host CUDA smoke. Those checks do not establish that its
candidate entry point runs the same frozen formulation and initialization.
Root review therefore marks this implementation **NOT_READY** for qualification.
No 9000/30000 candidate run was started and no candidate gate is passed by it.

Concrete gaps in the inspected implementation:

- `run_frozen_host` delegates Adam directly. It does not call the canonical
  `latent.begin/end` and `response.begin/end` wrappers. Importing their modules
  does not install those hooks.
- It lacks the canonical fixture-loading and explicit CUDA FP32 Adam bootstrap
  performed by the original candidate hold/shift drivers.
- It imports host modules from its worktree rather than selecting and verifying
  the pinned frozen runtime explicitly.
- Tensor hashing uses a dtype view directly; scalar optimizer-state tensors
  require a scalar-safe flattening step.
- The optimizer witness hashes only Adam's standard moments and step. Canonical
  sparse-prior updates also store `anchor_prev` in optimizer state; that history
  is omitted. All optimizer state and group settings need identity-preserving
  comparison, including tensor step representation and model buffers.
- A `mechanism._state` snapshot and four required source filenames do not cover
  arbitrary candidate helper modules. Each supported candidate needs an audited
  complete state contract and hashes for every imported local policy file.

The declared protocol compiler, absolute offsets, window scoring and RNG
instrumentation may be reusable. Their passing synthetic tests cannot compensate
for missing learner installation. A repair lane must demonstrate a short actual
host identity check, including hook calls and initialization, before readiness.
The stress and long-term protocol documents stay unchanged.

[Original report](completed-harness-preparation/attempts/prepare_continuous_qualification/result.md)
and [original source archive](completed-harness-preparation/source/manifest.json)
are preserved without rewriting the worker's claims. This review supersedes its
claim that the candidate entry point is ready.
