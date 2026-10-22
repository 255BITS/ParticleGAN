# Reproduce the frozen comparison

The scientific executions use committed source
`2a2b662bdfa6249076b27dbea4c136e23d3e7a80`, source manifest
`8bfcc01cd7482b1b348edb2162567adfdcdaa7d308f75fc834774302c6ac4acb`.
Later report and software compatibility commits are not substituted for that
execution identity. The common request runtime and hardware profile are retained
in [comparison.json](comparison.json). Historical selected results retain their
own source, initialization and qualification identity.

The original campaign is already complete when its final readout is published.
Inspect its retained evidence before considering any new paid execution. These
commands describe reproduction in a separate checkout of the exact scientific
commit, with an empty artifact destination and the matching Python/PyTorch and
two-A6000 runtime. They are not a request to rerun unchanged work for a merge.

```sh
# Run from the repository root of the pinned scientific checkout.
PG_REPAIR_ARCHIVE=/path/to/new-empty-three-theme-archive
mkdir -p "$PG_REPAIR_ARCHIVE"
for candidate in bispf-optimized-quintic-v1 bispf-output-energy-v1 bispf-settling-displacement-v1 bispf-fidelity-quartic-v1
do
  python -m experiments.forge --queue-root "$PG_REPAIR_ARCHIVE/queue" enqueue "$candidate" \
    --study "${candidate}-three-theme-study-v1" --through-tier 1 \
    --device cuda --cuda-model 'NVIDIA RTX A6000'
done
python -u -m experiments.forge --queue-root "$PG_REPAIR_ARCHIVE/queue" drain \
  --gpus 0,1 --workers-per-gpu 1 --campaign bispf-three-theme-repair-round1-v1 \
  > "$PG_REPAIR_ARCHIVE/drain.log" 2>&1
```

Each study fixes one round, all 15 task declarations, seed 0, public deterministic
initialization, full update allowances and observation schedules. Its own
passing smoke checkpoint supplies Gaussian continuation. The reservation is
28,020 seconds per arm and 112,080 seconds for the shared campaign. The same
global recipe runs every task; no task-specific winner selection is performed.

For the original campaign, follow its append-only logs without launching work:

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/three-theme-repairs-20261022/progress.log
tail -F /mnt/ml7tb/ParticleGAN-forge/three-theme-repairs-20261022/queue/events.jsonl
```

To verify original results and render GIFs, use the report sources from this PR
in the existing comparison checkout. These commands add zero training or sampling:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
  reports/forge/three-theme-repairs-20261022/publish.py \
  --queue-root /mnt/ml7tb/ParticleGAN-forge/three-theme-repairs-20261022/queue \
  --raw-root /mnt/ml7tb/ParticleGAN-forge/three-theme-repairs-20261022 --media
```

The publisher verifies the complete common source manifest, runtime, task and
recipe declarations, certified results, initial models/priors, consumed streams,
and paid-cost ledger. It writes compact numerical results to this report and
full readouts to the artifact archive. Saved-observation GIF receipts verify
their source tensor and output hashes. Missing original artifacts block this
verification; historical metrics are never silently replaced by new runs.

Mechanism analyses have their own reproduction commands in
[allocation](allocation/MEASURED.md), [settling](settling.md) and
[fidelity](fidelity/README.md). Their supplementary saved-data diagnostics remain
separate from the original task gates.
