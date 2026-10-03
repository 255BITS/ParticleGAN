# Critic-balance capacity binding

This helper prepares **necessary zero-update CPU capacity evidence** for one
complete Atlas config and one complete E22 config. Both use exactly
`lr=0.0053125`, `prior_lr_mult=1.5`, `d_lr_mult=2.25` on the eight declared hosts.
It performs no fitting, optimizer update, CUDA job or resource reservation.
Capacity supplies no training, persistence, robustness or default-adoption credit.

The construction inputs are the hash-pinned records in
[`policy-representation.json`](../family-winner-round1/policy-representation.json).
The six native inputs are its actual CPU captures, rather than transplanted CUDA
random states. The helper imports only fast G/D/prior parameters and the learned
output-noise scalar. It initializes a fresh public `GANTrainer`, optimizers,
averages, controllers and named streams under the candidate Recipe. One original
first real batch enters public `begin_step` followed by `abort_step`; that
controller/reservoir observation and data-stream advance are retained. No old
policy history, optimizer state, average weights or qualification is imported.

Every record binds the full scientific source union to develop
`4749b2780add539df4bd8d2dd1d3cc9f002f77ad`, the helper bytes, exact case metadata,
public preset, requested overrides, full resolved Recipe, original full horizon,
CPU runtime, original evaluation count and seed 34002. Capture retains all view
arrays and the first real batch outside Git. Verification rebuilds the fresh
public construction, compares the complete state, validates its public checkpoint
loader, then requires bitwise sampler/target-array equality and identical original
primary metrics. Native output-noise-off diagnostics remain separate from the
primary noisy native gate. Images and ordinary vectors keep output noise off and
retain their actual policy latent perturbation and selected serving.

`verify_record(record, case, family)` returns valid evidence unchanged:

- `SUPPORTED`: the original primary gate passes at this snapshot.
- `UNRESOLVED`: a complete, exactly verified snapshot fails original bounds.
- `BLOCKED`: a bound preparation error, with no sampler certificate or capacity
  credit. Available raw files and the exact error remain archived.

Tampered or stale records raise. `verify_packet(packet)` requires all 16 distinct
outcomes. Each family needs its own eight `SUPPORTED` cells for scientific
admission; a blocked sibling provides no credit. An unattested source/card
prerequisite stops export with a separate `BLOCKED_PREREQUISITE` receipt, preserving
the full requested denominator and error. It cannot become a per-cell certificate.

Root must freeze the helper/tests before the actual new export. Run from this
isolated checkout into a fresh external directory:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
/ml2/hypergan/.venvs/particlegan-develop-integration/bin/python \
reports/forge/critic-balance-20261003/bind_capacity.py \
--output /ml2/hypergan/critic-balance-capacity-20261003
```

CPU verification in a fresh process:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
/ml2/hypergan/.venvs/particlegan-develop-integration/bin/python \
reports/forge/critic-balance-20261003/bind_capacity.py \
--verify /ml2/hypergan/critic-balance-capacity-20261003/capacity.json
```

Exit 0 means all 16 capacity cells pass. Exit 1 preserves legitimate negative
outcomes. Exit 2 denotes an unattested export prerequisite or invalid CLI
verification. `verify_record` and `verify_packet` raise for invalid evidence. The
verification CLI emits a packet-hash-bound `verified` result and separate family
admission flags, distinguishing a negative sibling from a software error.
A GPU coordinator must use a fresh CPU-only verifier subprocess; importing
the verifier into an already CUDA-initialized parent is rejected. The verifier
requires one CPU thread and preserves ambient RNG and backend flags during each
construction/replay. Software controls use synthetic input/source oracles with the
actual public construction, checkpoint, sampler and scorer; they do not depend on
historical Git objects or local research tensors.
