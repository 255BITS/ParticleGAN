# Word audit health: retained diagnosis

The [pinned diagnosis](DIAGNOSIS.md) explains the original INVALID result. The
public birth/death code deliberately records an undefined reference dimension
and skips row moves. The generic audit guard rejected that diagnostic NaN,
although the saved learned tensors were finite. The narrow repair recognizes
only this exact clocked, no-move public skip record and preserves its bytes.

The old result, source, cost and numerical UNAVAILABLE status remain unchanged.
The [original retained GIF and clock context](../atlas-word-retained-context-20261004/README.md)
show 20,001 updates and 24 pure reads. The outputs missed the original goals.
`reconstruction_exact=0` means the all-five paired reconstruction test failed;
it does not mean every individual word was reconstructed incorrectly.

[Machine diagnosis](diagnosis.json), [read-only reproducer](inspect_word_health.py),
[patch check](patch-check.json) and [author's immutable handoff](handoff.json)
retain exact source/input identities. The original inspection and control logs
remain local and are referenced by hash. The author's report preserves its
actual test environment; use the installed project Python to run the relocatable
inspection command:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python inspect_word_health.py \
  --run RAW_FAMILY/attempts/five_word_joint_acquisition_word_joint_policy_min11_v1 \
  --source-root FROZEN_SNAPSHOT --output NEW_DIAGNOSIS.json
```

This reads tensor dictionaries, constructs no training model and draws no
samples. It grants no new numerical verdict, default, speed or causal optimizer
claim. Only newly bound future attempts may use the repaired guard.
