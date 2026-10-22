# Reproduce the frozen first-principles comparison

Scientific workers execute source
`a03373da3421303b89c99a39898b091e22910134`, manifest
`668914920afda7bb6fc53d9c918aac92a9c9cd5ae582884fb9f068ff1b8249e8`
(1,292 files). Later reporting commits do not replace that execution identity.
The four frozen requests retain runtime, hardware, complete task definitions,
resolved recipe ownership, original initialization and named-stream contracts.

The original campaign is running or concluded when this document is read.
Inspect its evidence before considering paid reproduction. These commands
describe a separate checkout of the exact scientific commit, matching runtime
and two-A6000 hardware, and an empty archive. They do not authorize rerunning
unchanged work for a merge.

```sh
PG_PRINCIPLES_ARCHIVE=/path/to/new-empty-first-principles-archive
mkdir -p "$PG_PRINCIPLES_ARCHIVE"
for candidate in bispf-optimized-quintic-v1 bispf-output-energy-v1 bispf-secant-extragradient-v1 bispf-bregman-game-v1
do
  python -m experiments.forge --queue-root "$PG_PRINCIPLES_ARCHIVE/queue" enqueue "$candidate" \
    --study "${candidate}-first-principles-study-v1" --through-tier 1 \
    --device cuda --cuda-model 'NVIDIA RTX A6000'
done
python -u -m experiments.forge --queue-root "$PG_PRINCIPLES_ARCHIVE/queue" drain \
  --gpus 0,1 --workers-per-gpu 1 --campaign bispf-first-principles-round2-v1 \
  > "$PG_PRINCIPLES_ARCHIVE/drain.log" 2>&1
```

Each arm has 28,020 reserved seconds for fifteen tasks; the common campaign
ceiling is 112,080 seconds and sixty paid attempts. A failed own smoke cannot
provide its continuation checkpoint: that declared continuation remains
unexecuted BLOCKED, with zero attempt/cost and no invented GIF. The rejected
thermostat has no paid request. No scientific retry, coefficient or seed sweep
is part of this protocol.

Passing Gaussian smoke supplies each arm's own continuation checkpoint.
Those trained states differ by the intended recipe delta; no arm receives
another arm's trained checkpoint. The shared initializer and prior identity
apply to the original initialized training process.

Follow the original campaign without launching work:

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/first-principles-round2-20261022/progress.log
tail -F /mnt/ml7tb/ParticleGAN-forge/first-principles-round2-20261022/queue/events.jsonl
```

The scalar `public-updates.jsonl` files reside inside each original attempt's
artifact tree. Their receipts bind hashes and observation scope. They retain
earned thermal scales, raw secant consistency and final movement statistics,
and the game's routing/prices/saturation/canonical forces. These are observed
diagnostics, not additional gates or evidence of parameter-force attribution.

After all runnable jobs finish, verify certified originals and render GIFs
from the publication checkout, using its final report sources:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
  reports/forge/first-principles-round2-20261022/publish.py \
  --queue-root /mnt/ml7tb/ParticleGAN-forge/first-principles-round2-20261022/queue \
  --raw-root /mnt/ml7tb/ParticleGAN-forge/first-principles-round2-20261022 --media
```

Publication adds zero training, sampling or rescoring. Missing original
artifacts block verification. Original source cohorts and historical
qualification remain separate. The mechanism reports contain their fixed
analytic/software controls; measured reports provide saved-data reduction
commands and compact provenance.
