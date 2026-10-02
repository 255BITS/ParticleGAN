# Generator minibatch conditioning: fixed public E22 comparison

The predeclared endpoint test passed: **G64 reduced population excess error by
18.67% versus G16 after 500 GAN-only updates**. Both arms completed in 27.07
seconds combined. A separate matched real100 continuation also improved donor
LPIPS, but its 0.000822 gain missed the frozen 0.001 minimum. The real gate failed;
there was no continuation or default promotion.

| Fixed update | G16 excess error | G64 excess error | G64 relative to G16 |
|---:|---:|---:|---:|
| 0 | 0.0005141192814 | 0.0005141192814 | same |
| 50 | 0.0001463613589 | 0.0001420391782 | 2.95% lower |
| 100 | 0.0000448358624 | 0.0000461059426 | **2.83% higher** |
| 300 | 0.0000137248007 | 0.0000114928480 | 16.26% lower |
| 500 | 0.0000102156973 | 0.0000083087707 | **18.67% lower** |

This is a fixed500 endpoint comparison. G64 does not improve every intermediate
point: its first update and update100 are slightly worse. At the endpoint,
G16 retained 1.987% of initial excess; G64 retained 1.616%. Stock serving chose
the fast state in both arms, giving the same measured values as clean live
evaluation.

## Reproduce and obtain an explicit pass/fail

From the ParticleGAN checkout, with its normal dependencies installed:

```sh
timeout 60s env PYTHONPATH=. python -u examples/routed_generator_batch.py \
  --output /tmp/routed-generator-batch
```

Use a fresh directory. All optimizer construction, initialization, loss creation,
row configuration and update hooks use ParticleGAN's public API. The command
returns zero only when the frozen quality and native health gates all pass:

- Both fixed500 endpoint live excesses must be at most 90% of their initial excess.
- G64 fixed500 endpoint live excess must be at most 90% of G16 endpoint excess.
- All 500 native lifecycles, finite active gradients and parameter states,
  exactly one optimizer owner per live parameter, independent KA2 EMA,
  128 dense table gradients, frozen host bytes, exact common caller streams,
  source hashes and organic row-control observations/probes must pass.

Nonfinite values, missing gradients, ownership or lifecycle failures stop the
campaign. Incomplete execution or a quality-gate failure returns nonzero.
The external60-second timeout bounds a stalled update. There was one scientific
attempt; no fixture, loss, momentum, seed or budget search followed it.

The six focused public CPU contracts also passed in 1.36 seconds:

```sh
python -m pytest -q tests/test_routed_generator_batch_public.py
```

They check symmetric nuisance/conditional means, deterministic initialization
and optimizer ownership, independent EMA, native one-step behavior, exact
public checkpoint replay and read-only evaluation. A separate contract requires
the first16 G identities to be exactly the D16 identities and checks that the
G-data stream consumes exactly48 extension draws.

## One changed factor and a known evaluation optimum

Both arms initialize the same width4 spatial FiLM generator, source/time
encoder, cosine router, 128×4 particle bank and learned critic. The generator
contains frozen BF16 prefix/head layers shared by both arms. This is the
**late-common profile initialized fresh**, not the mature real-task checkpoint:
global score gain0, local gain1/16, free-sign channel-energy head initialized
to zero with public `init.KEEP`, antithetic G loss, beta1 zero and applied G
rate factor1/4. KA2, routed DV12, settling, settled reopening, row evidence and
birth/death controls remain enabled.

Each update draws D16 examples. G16 reuses those exact examples after D's step;
G64 appends48 examples from its separately named G-data stream. Both arms consume
all caller64 row identities and64 G Gaussian fields; G16's unused48 are declared
shadow draws. D16 and its independent16 Gaussian fields stay matched. Native
private DV12 consumption may differ with batch size, so private perturbation
identity is not claimed. G64 processes four times as many G examples per update;
this is not an equal-compute comparison.

Sixteen source/time fit contexts have sixteen hidden nuisance modes. The modes
are eight fixed spatial fields of RMS0.15 and their exact negatives. G/E/router
see source/time only. Four disjoint guard contexts include all16 modes;64 untouched
population contexts provide the endpoint metric. Their conditional mean `f(x,t)`
comes from a fixed seed4 teacher in the same generator family, with code columns
zero, so it is representable. The teacher is never optimized or used for a
training backward. For the finite symmetric population, the evaluation identity is

```text
Q(theta) = mean_(context,mode,channel,pixel)
           [G_theta(context) − f(context) − nuisance(mode)]²
         = excess(theta) + 0.15²

excess(theta) = mean_(context,channel,pixel)
               [G_theta(context) − f(context)]².
```

The analytic capacity minimum of excess is zero. Only excess is compared, so
the nuisance floor does not conceal differences. Clean means and Euclidean
error are evaluation references; neither is a reconstruction training loss or
row-control target. Native row controls use nuisance-bearing fit/guard targets.
This population is deliberately specified for the toy; it does not establish
that the actual bridge's labels contain such nuisance.

## GAN objective and native controls

The clean code uses the actual mass-aware route:

```text
query = E(context)
weights = softmax(2*cosine(query, table) + log_mass)
code = weights @ table.
```

Native DV12 perturbs the training code. For residual `r = G(context,code) − target`
and a caller Gaussian `epsilon`, the D loss is

```text
L_D = mean softplus(D(sigma*epsilon + detach(r)) − D(sigma*epsilon))
      + native_KA2.
```

After the D update, the G prediction is recomputed. One true scalar antithetic
loss, one backward and one native G optimizer step use

```text
L_G = 0.5 * sum_(s in {-1,+1}) mean softplus(
        detach(D(s*sigma*epsilon)) − D(s*sigma*epsilon + r)).
```

There is no MSE/reconstruction optimizer backward. The learned noise retains
the native physical floor1.3. The score is

```text
D(u) = [ (1/16)*D_local(u) + sum_c w_c*sqrt(HW/C)*mean_HW(u_c²) ] / sqrt(3).
```

The channel weights learn through D only. Global features remain available to
routed DV12 despite global score gain0; global-score parameters are inactive
by design and are excluded from required score-active gradient checks.

Both arms recorded500 KA2 applications and500 row evidence updates, with128
dense gradient rows on every update. Each recorded31 organic row evaluations,
248 deletion probes and248 proposals. G16 recorded21 guard rejections and87
hold steps; G64 recorded6 guard rejections and97 hold steps. Neither accepted
a move or split. These are qualification counters, not evidence that population
restructuring improved quality.

## Evidence and limits

The source-author offline review passed18 receipt, hash, trace, control-count
and gate-arithmetic checks without a new forward, gradient, optimizer or RNG draw.
The protocol binds33 source files at commit
`c4b53495d04a2052e266739a82053317b7f49698`.

- Driver SHA256: `85e23d03a3c8a583c2b928b024fa48b3c16a425364c64f0f92c03bb194d7adba`.
- Protocol SHA256: `f302530f82d9f0ff6ce28d7e6b38b3e547212c5313b7631813acbd7ee8f1bd3d`.
- Raw artifacts: `/ml2/hypergan/routed-generator-batch-v2-artifacts-20261002`.
- Offline receipt: `/ml2/hypergan/routed-generator-batch-v2-offline-review-20261002/receipt.json`.

The earlier V1 is preserved as an unrun draft. Static review found that its G16
examples were independent of D16, unlike the actual caller. V2 corrected that
coupling before the sole scientific execution, preserving all seeds, fixture,
losses, optimizer settings, budgets and gates.

This finite toy supports a generator minibatch-conditioning mechanism and its
declared endpoint benefit. It does not prove an E22 software bug, a universal
population optimum for the learned GAN game, or improved Nova→Qwen LPIPS.
The [compact JSON](routed_generator_batch_20261002_results.json) records the exact
metrics, original hashes, separate reviews and transfer result. Raw logs,
checkpoints, individual images and full per-case records remain outside Git.

## Actual Nova→Qwen transfer: lower LPIPS, rejected continuation

The separately frozen real trial restored the actual winning GAN-only checkpoint
at step4600 into two independent public full routed-E22 runs. It committed
100 updates per arm to step4700. D remained16; the intervention changed the
post-D generator batch from16 to64. The original full outer checkpoint, optimizer
moments, routing/control/noise/averaged owners and external caller state were
restored exactly. Two copied diagnostic lifecycle executions checked unchanged
G16 against the historical caller; they committed zero training updates.

| Native fast serving | Step | Donor LPIPS ↓ | LPIPS vs decoded Qwen target ↓ | Pixel MSE ↓ | Latent NMSE ↓ |
|---|---:|---:|---:|---:|---:|
| Starting GAN checkpoint | 4600 | 0.142711910123 | 0.141585341951 | 0.004165692648 | 0.184661813123 |
| Live matched D16/G16 | 4700 | 0.142734463263 | 0.141488942402 | 0.004163847233 | 0.185162796046 |
| Live D16/G64 | 4700 | **0.141889484515** | **0.140766583595** | 0.004172924830 | **0.183547462421** |

Both endpoints selected native **fast** serving and were decoded through ordinary
raw source/time inference on the same176 validation cases. G64 improved146 cases
and worsened30 versus G16. The paired-target codec reference donor LPIPS was
0.010601304895 for both arms. Donor LPIPS is the arithmetic mean of LPIPS-Alex
against the original donor pixels; `lpips_vs_teacher` uses the decoded paired
Qwen target. These are separate metrics. Pixel MSE became slightly worse even
though donor LPIPS and normalized latent error improved.

The frozen acceptance rule required both a matched-control win and at least
0.001 donor-LPIPS improvement from the starting4600 checkpoint:

```text
required candidate LPIPS <= 0.14171191012283618
observed candidate LPIPS  = 0.14188948451456698
gain from starting best   = 0.0008224256082692005
gain versus live control  = 0.0008449787485667326
missing required gain     = 0.00017757439173080036
```

The matched-control condition passed, but the minimum-gain condition failed.
The final scientific result is **rejected**, with nonzero child exit2, no
extension and no default promotion. This is the best measured GAN donor LPIPS
in this search; the qualified saved default remains the4600 checkpoint. A
better measured value does not relax the predeclared gate or establish a
general convergence fix.

G64 appended48 auxiliary fit rows to the exact D16 IDs. Both arms consumed those
48 IDs and output Gaussian fields, with G16 declaring them shadow draws. All100
D16 ID draws, original caller data/paired RNG states, D Gaussian fields, first16
G Gaussian fields and auxiliary hashes matched. Native whole-batch G64 DV12 ran
through the public generator call with ordinary `perturb=True` and native
recording. Its batch-dependent private RNG consumption may change first16 G
perturbations and later D perturbations; private DV12 identity is **not claimed**.
There was one post-D G prediction, one scalar antithetic loss, one backward,
one G optimizer step and one complete public lifecycle per update. Routed fit
observations/FIFO stayed16. Every update had128 dense bank-gradient rows.

The mature real state retained beta1 zero, applied G factor1/4, local critic
gain1/16, owned global-score buffer0.01 with executed global score0, and unchanged
global routed features. Native KA2, DV12, settling, settled reopening and routed
birth/death controls remained enabled. Every optimizer loss was GAN-only; clean
MSE was a reporting/health metric. Historical recipe metadata still contains
`reconstruction_weight=1`, but the native caller never invokes reconstruction;
its effective coefficient is zero. The toy instead starts the late-common
profile fresh, so it does not reproduce the mature real state or its history.

The expanded snapshot has10806 training rows,10486 fit rows and64 held guard
rows. Its original176 validation cases were preserved byte-for-byte:96 native
clean estimates,16 native finished,48 switched clean estimates and16 switched
finished. The test split stayed unopened. This fixed cohort was already used
for search, so the result is validation evidence rather than a new generalization
or statistical-significance result.

The full supervisor-observed process wall time was418.08 seconds. Active training was
28.08 seconds for G16 and96.11 for G64; bounded data preload was74.53 and61.13
seconds; each176-case decode took45.91 and46.13 seconds. There were200 committed
scientific updates plus two copied diagnostic lifecycles. G64 uses four times
as many G examples per update, so this is not an equal-compute speedup result.
The new1 GiB row cache and128 MiB shard cache allowed the fixed trial to complete
without scientific-phase disk misses; no runtime/resource stop occurred.

## Publication identity and preserved evidence

The original scientific authority remains V2 at `c4b53495`, its driver SHA256
`85e23d03a3c8a583c2b928b024fa48b3c16a425364c64f0f92c03bb194d7adba`,
and protocol SHA256
`f302530f82d9f0ff6ce28d7e6b38b3e547212c5313b7631813acbd7ee8f1bd3d`.
The original33 source hashes remain verified. A second-agent independent toy
review and a separate source-author offline review both checked the result;
neither reran the campaign.

The publication copy is based on develop `45e621f11de05816204151a271e87144ebded90a`.
All30 frozen ParticleGAN package files and `pyproject.toml` are byte-identical
to the executed c4 cohort. Only publication identity plumbing changes: the
command verifies bound source hashes and records actual checkout HEAD instead
of requiring HEAD to equal the old commit. Otherwise committing the example
would make its own command reject. The exact original driver/protocol bytes are
preserved in [the archive](routed_generator_batch_20261002_archive/README.md).

No new scientific run was performed for packaging. A structural AST audit binds
identical scientific functions/classes and constants; the original six public
CPU contracts plus one new commit-portability/source-drift contract pass:

```sh
python -m pytest -q tests/test_routed_generator_batch_public.py \
  tests/test_routed_generator_batch_identity.py
```

The public toy is a diagnostic with an explicit pass/fail endpoint. Its passing
500-update result and weaker real100 gain are both retained. It changes no
ParticleGAN core formulation, learned-prior default or model-glue default.
