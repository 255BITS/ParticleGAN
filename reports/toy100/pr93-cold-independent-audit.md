# PR93 exact-head cold acquisition

PR93's unchanged fence-restore rule finishes the cold ring with **four modes
and HQ`.985107`**, compared with seven modes /HQ1.0 for PR84 in the same
process. Neither has a passing ring checkpoint. This practical acquisition
test was run despite the submitted warm hold's119/120 result; the cold
failure is independent evidence, not an inference from that brief warm dip.

The inspected head is
`e958a81c49b4dc3c2b3eaad2a17000d70a091831` of
[PR93](https://github.com/255BITS/ParticleGAN/pull/93). Its submitted files,
thresholds and eight-iteration latent correction are unchanged. Execution
uses the pinned torch`2.13.0+cu126` environment on CPU, one thread and AVX2.
The author's warm/hold report used torch`2.14.0+cpu`; those numbers are not
presented as reproduced here.

| Cold run | Final result | Passing checkpoints | Wall time |
| --- | --- | ---: | ---: |
|PR84 trajectory400|MSE`.000942662`, PASS|18/24; suffix18|4.13s|
|Exact PR93 trajectory factory|Setup error before update1|unrun|.015s|
|PR84 ring1200|7 modes /HQ1.0|0/24|30.79s|
|PR93 ring, clip disabled|Bitwise identical to PR84|0/24|30.62s|
|PR93 ring, clip active|4 modes /HQ`.985107`|0/24|49.09s|

The active ring costs about1.59× the same-process PR84 run. This is one
paired timing observation, not a benchmark confidence interval. No further
hold or coefficient/radius/seed comparison follows this failed acquisition.

## Controls and unchanged source

The driver uses a detached checkout at the exact submitted head, verifies
no tracked differences before and after execution, and archives every loaded
repository Python module. It declares constant base rates D/G`.00425`,
prior`.0085`, original noise clocks, and the original400/1200 budgets.

With clipping disabled, the ring matches PR84 in the **entire saved final
state**: model/prior parameters, both Adam states, EMA, global/data/input/
output RNG and complete fixed-noise policy state. All host metrics excluding
runtime, every update record, applied rates, noise receipts and generated
host source are also exact. All successful hosts advance Adam moments once
per update and replay the same noise/data in the three PR84 gradient blocks.
Ring1200 has1200 moment steps and3600 native blocks per player. The extra
latent output Jacobians are additional work, not included in that inherited
three-field count. The seven submitted focused tests pass.

The exact PR93 context unconditionally reads `module.sample_ring`, but the
conditional trajectory module has no such function. Both enabled and
disabled PR93 contexts raise `AttributeError` before yielding a recorder.
Those exact setup traces are retained. PR84's trajectory result is a control,
**not** a claimed PR93 trajectory pass. No compatibility shim was introduced.

## What the cold result does and does not isolate

Clipping executes on1195/1200 updates and touches11155 particle rows. Cold
quality is uneven: the best observed coverage is six modes; none of24
observations reaches eight. Final HQ is high for the represented subset.

The mechanism takes the current phase2 **G real minibatch**, computes a
nearest-neighbor median/MAD fence, and modifies selected latent rows after
the native bounded GAN proposal. It does not read target centers or HQ.
This is a data-derived output constraint added to the GAN update, not an
unchanged Rp-only optimizer.

Nonlinear landing is a material limitation in this cold run. The submitted
receipt reports a residual above`1e-4` on234 updates and a maximum`118.96`.
That residual is measured **before** the final latent correction, so it is
not a verified final-point error. The submitted rule has no subsequent
actual-objective or residual acceptance/rest check. The acquired subset
therefore cannot be attributed solely to the fence's geometry; imperfect
latent realization may also contribute. No repair or retuning was tested.

## Practical comparison with other current lanes

PR84 remains the closest tested GAN-first baseline: D uses the original
sharp Rp/cap objective; G uses a spatially smoothed Rp critic and own-step
bounds. PR93 retains that native proposal then adds a local data constraint.
Its stronger submitted warm hold does not compensate for worse observed
cold acquisition in this pinned environment.

The anchor family changes the accepted output map to a sampled support
objective. It demonstrated cold support acquisition in earlier evidence,
but support centers alone do not preserve empirical mass or spread, and
fixed identities can reject a later missing-component signal. That is a
different objective family, not proof that the native GAN game is fixed.

The likelihood lane uses an explicit emitted-density objective with a
finite quadrature rule and global allocation. Its free-output acquisition
evidence warrants a bounded neural/runtime check. Its current neural V2
discards the native G displacement when the likelihood fit cannot improve,
so success would be likelihood-driven particle learning with a neural
realization, not an Rp optimizer-only repair. All-history cost and finite
quadrature scope remain explicit; no full cold result is claimed here.

The [certified readout-value diagnostic](pr84-convex-profiled-cold-continuation.md)
has no demonstrated practical advantage for the acquisition leaderboard:
six accepted copied-state moves still cover three modes, using4783 cached
readout closures plus derivative/cache work. It changes G to a profiled
sharp-D value, freezes nonlinear critic features and the Adam metric, and
is not yet a native update. Its contribution is bounded causal evidence;
further expensive expansion was stopped.

The [manifest](continuous-evidence/pr93-cold-independent-audit/manifest.json)
binds raw results, setup failures, source/declaration, complete final states,
the exact submitted source/test/report, PR metadata and transcript. The
same-process controls and all historical failures remain visible.

```bash
git worktree add --detach PR93_CHECKOUT e958a81c49b4dc3c2b3eaad2a17000d70a091831
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 \
  ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES='' \
  /tmp/pr38-default-env/bin/python -u reports/toy100/pr93_cold_audit.py \
  --checkout PR93_CHECKOUT --output NEW_OUTPUT
```
