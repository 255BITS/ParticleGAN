# BISPF latency investigation

This investigation starts from `b5d2372f2739d7579ab66d6dba6e220d2eb776d9`
and the selected global `bispf-v2` recipe. It measures complete public
`GANTrainer.step` updates before changing the implementation. Historical
performance receipts remain bound to their executed source; they are motivation,
not a matched timing control for a changed implementation.

The current technique inventory is the single result board. A new optimized
recipe may replace its BISPF selection only after equal or better measured
quality under the task-owned protocols. A performance-only pass gives no quality
or default-adoption credit.

## Protocol and bounded decisions

1. Profile incumbent BISPF and R1/R2 on the existing 64- and 256-pixel DCGAN
   mock-data tasks. Reserve at most 960 seconds across four arms. Measure ordinary
   complete updates separately from synchronized component instrumentation and
   a Torch profiler replay. Freeze the exact source and manifest before execution.
2. Investigate the measured dominant components. Preserve exact optimizer
   semantics when practical. Explicitly declare a new numerical backend or
   optimizer recipe when arithmetic or update laws change; preserve native
   defaults and checkpoint compatibility for the original recipe.
3. Freeze each substantive candidate before its bounded performance/quality
   study. Reject clear numerical or quality regressions before further work.
   Revisions require a new identity and report their original negative evidence.
   Do not repeat unchanged experiments or vary only the seed.
4. Measure the strongest surviving global recipe on all six existing DCGAN
   performance workloads with current-source controls. The initial performance
   target is at least 8x lower update latency than incumbent BISPF at 64 and 256
   pixels, and no more than 2x R1/R2 latency at 256 pixels. These are investigation
   targets, not image-quality gates. Report every workload even when it misses.
5. Check preservation of the incumbent's counted quality pass set, including
   acquisition, own-state holds and endurance, before changing the selected
   recipe. Keep held failures and archived cohorts visible. An alternative
   optimizer is authorized, but requires its complete trainer delta and quality
   evidence rather than an implementation-parity claim.

All comparisons use seed 0, the public deterministic initializer, independent
checkpointed RNG streams, and the task's unchanged architecture, target law,
seen batches, initial prior distribution/capacity, sampling, update allowance,
and evaluation cadence. One complete trainer recipe is used across tasks.

Raw traces, profiler output, checkpoints and execution logs stay under
`/home/martyn/dev/ParticleGAN/runs/forge/bispf-optimized/` or another declared
local Forge artifact directory. Publish compact receipts, provenance, reports,
reproduction sources and actual-training GIFs. The host's GPU contention
telemetry was unavailable in the preceding campaign; verify it before asserting
a certified paired speed ranking, and label unknown contention explicitly.

## Alternative optimizer round

The guarded polar solvers are retained as negative evidence. Source
`d9d72d53761c898bce20102e1995fc3d899c1447` freezes two further single-field
recipe changes from native BISPF, with no learning-rate, momentum or budget
retuning:

- `optimizer_convolution_rule="frobenius"` uses
  `M / sqrt(||M||_F^2 + lambda^2)`. This gives a unit operator bound but can
  substantially reduce displacement relative to polar normalization.
- `optimizer_convolution_rule="quintic"` applies five fixed FP32 polynomial
  steps to the augmented matrix `[M; lambda I]`, using coefficients
  `(3.4445, -4.775, 2.0315)`, and divides its top block by `1.21`. It is a
  bounded spectral filter inspired by Muon, not an equivalent polar solver
  or the PyTorch Muon optimizer.

Both rules dispatch only to CUDA per-offset convolution matrices whose largest
channel count per group exceeds 32. CPU, tiny convolution, dense, bias and prior
updates retain the native law. The existing channel-ratio/kernel-area factor
and absolute smoothing parameter remain in effect. Defaults and archived
native checkpoint metadata retain their original meaning. Changed rules and
checkpoint versions are explicit.

The four-arm profile reserves 960 seconds in total. A storage failure interrupted
the first Frobenius 256-pixel arm; retain its error and partial artifacts. Charge
that failed allocation conservatively 60 seconds and give its separately
identified infrastructure recovery at most 180 seconds, preserving the original
240-second slot and total cap. The completed Frobenius 64-pixel arm is not
repeated. The two unexecuted quintic arms retain their 240-second slots.

`bispf-fastpath-alternatives-v1` then reserves 1,800 seconds for six direct
training arms: native BISPF and both alternatives on the same two width-64 image
tasks, with 300 seconds per arm. Require preservation of native passes and a
positive optimized stripes pass before broader execution. The tasks retain
600 updates, 24 evaluations, public initialization and seed 0. Their numerical
results and actual-training GIFs are diagnostics; they do not replace ordinary
quality gates or establish natural-image quality.

The final matrix campaign `bispf-matrix-performance-v1` reserves at most 4,800
seconds: four recipes times the six unchanged DCGAN performance workloads.
Measure surviving alternatives and both same-source controls serially on the
same physical GPU. Leave unnecessary losing-candidate arms unexecuted. Only a
surviving global recipe proceeds through its separately declared ordinary
quality study, capped at 46,620 seconds. The selection proof checks all 19
incumbent numerical passes, the counted current-policy gates, both direct wide
controls and the declared observed timing targets before proposing a whole-row
BISPF replacement. It never applies the selection itself.

## Reconstructing the executed source closure

The final candidate ran from a sparse H2 checkout. Its captured 1,273-file source
map has digest
`c82ed2dc712e4c9ebda80678e453635bd8a16dc74fae1450234ed368aa956213`.
The optional `reports/transfer_suite/unadjusted/leading_profile.json` was absent.
A full checkout can introduce that catalog support file and change the source
identity, even at the same Git commit. Reconstruct the same file set explicitly:

```sh
git worktree add --no-checkout --detach /tmp/bispf-h2-repro \
  d9d72d53761c898bce20102e1995fc3d899c1447
git -C /tmp/bispf-h2-repro sparse-checkout set \
  particlegan experiments benchmarks lib configs reports/forge \
  reports/toy100/gap-fill-20260925/sources/k3p
git -C /tmp/bispf-h2-repro checkout d9d72d53761c898bce20102e1995fc3d899c1447
```

Use the project environment from that checkout. Planning launches no training:

```sh
cd /tmp/bispf-h2-repro
python -m experiments.forge --queue-root /tmp/bispf-h2-plan-queue \
  plan bispf-optimized-quintic-v1 --study bispf-optimized-quintic-v1-quality
```

The saved requests bind the precise recipe, seed, task contracts, initialization,
RNGs, runtime and hardware separately. Matching the source map alone does not
reproduce timing on different hardware. Frozen publication asks H2's own catalog
helpers which optional files matter, preserves their captured presence/absence,
and verifies original study support, source bytes and admission signatures. Its
closure receipt records the omitted profile explicitly. A completed measured
study disappearing into unresolved declarations is an error, never qualification.
