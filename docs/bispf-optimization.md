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
