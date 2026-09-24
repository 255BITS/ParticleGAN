# GAN-only nonlocal proposals fail two short native branches

The new GAN-only rule fails9 of16 live checks. All16 nonlocal proposals
converge and beat both the pre-G and ordinary bounded-G candidates on the
actual original adversarial loss, yet successive proposals delete occupied
modes. The failure is not caused by numerical landing error or a new
geometric objective. No full warm or cold run follows.

The [copied-state output and landing assays](pr84-adversarial-landing.md)
justified this native test but did not predict repeated-game behavior.
Each branch resumes one exact archived warm state and follows its own G,
prior, D and Adam states for eight consecutive updates. The original PR84
branch runs from the same state and reproduces every recorded clean output
and update receipt, plus each available full accepted-state hash.

| Branch | Original passing checks | New passing checks | New mode sequence | New minimum HQ | New final |
| --- | ---: | ---: | --- | ---: | --- |
|1324–1331|5/8|3/8|8,8,8,7,6,6,7,7|.911133|7 / .911133|
|1530–1537|1/8|4/8|8,8,8,8,7,7,7,7|.941162|7 / .966553|

The original branch endings are respectively8 /`.999268` and7 /`.799072`.
The new rule improves HQ in the later difficult branch and avoids the
1325 off-mode dip, but loses coverage. These are consecutive trajectories,
not independent one-step repairs of already bad states.

The declared update keeps ordinary alternating PR84 D/G Adam steps and
their own-curvature bounds. After that native proposal, it compares three
actual G losses against the materialized D: pre-G, native bounded G, and
one fitted nonlocal proposal. The latter chooses one donor and one actual
D-real sample by exhaustive original paired Rp G loss, then starts the
unchanged joint GN20/12 solver from pre-G parameters. A fitted candidate
must converge and strictly beat **both** other losses; otherwise the lower
native/rest candidate is used. No C+Q, target-center, quality, coverage,
distance-acceptance, memory or new discriminator-fit objective enters.
The imported parent recorder supplies phase, batch and ownership mechanics;
its geometric correction is overridden completely.

All comparisons use the phase1 G stencil width frozen across proposals.
It remains`.15` on every executed update. The new generic score function
uses the native patched critic exactly once, avoiding a second stencil.
Actual latent indices, raw output-noise draws and both real batches are
captured and replay-checked. Restoring pre-G parameters reproduces the
native emitted fake batch and phase1 paired G-loss scalar **bitwise** on
all16 updates. This active implementation is explicitly limited to zero
discriminator input noise; it is not declared cold-ready.

Every accepted move is a converged joint fit. At the first mode-loss steps,
offline target-center attribution gives:

| Step | Donor's previous mode / occupancy | Destination mode | Pre-G / native / selected G loss | Covered modes after |
| --- | --- | --- | --- | ---: |
|1327|6 / one particle|3, already occupied|.712256 / .711464 / .684325|7|
|1328|2 / one particle|3, already occupied|.690807 / .690040 / .675461|6|
|1534|7 / one particle|3, already occupied|.686642 / .685722 / .669516|7|

All other output rows remain within the joint solver's target tolerance.
Maximum landing row error across the run is`2.13e-5`, inside the declared
relative threshold; it does not explain a jump between separated modes.
The discriminator does receive one actual Adam update before each move.
That response does not prevent the next current-critic G-loss winner from
removing sole representatives. This is a concrete limitation of this
nonlocal response rule, not a general impossibility for adversarial learning.

Both branches retain exact original RNG consumption, complete observed
noise receipts, horizon1200, rates D/G`.00425` and prior`.0085`, and one
native Adam moment update per player per outer step. D/Adam/gradient buffers
are unchanged by each additional correction. Native gradient work remains
three evaluations/player/update; extra work is66 joint output Jacobians,
67 nonlinear fitting trials, and284,672 critic-score input points, each
using the five-point stencil. The native G bound does not limit these
additional global jumps. All four original/candidate branch runs and
state/source capture take4.69 seconds on the pinned one-thread CPU.

The two new tests pass in2.34 seconds: generic scoring agrees with the
corrected once-smoothed output assay, and an actual1530 host update verifies
disabled full-state parity plus active data/noise/Adam/RNG ownership and
loss acceptance. Together with the five output tests and four existing
joint-solver tests, all11 focused tests pass. Raw observations, every loss,
fit records, sources, declarations and full terminal/last-pre-step states
are retained in the [archive](continuous-evidence/gan-nonlocal-native16/manifest.json).
On an exception the driver would label a partial terminal state
NONRESTARTABLE and preserve the last safe pre-step; no exception occurred.

The next useful mechanism would have to test the discriminator's response
to a proposed redistribution, not merely improve the same frozen-current-D
loss more strongly. Increasing jump accuracy or adding a geometric
keep-one-particle rule does not address the GAN-only question. A concrete
next cheap falsifier can compare the lost-mode proposal with the untouched
baseline after a matched, explicitly bounded D response; no such extra
fit or continuation is included here.

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 \
  ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES='' \
  /tmp/pr38-default-env/bin/python -u reports/toy100/pr84_adversarial_reallocation_filter.py \
  --capture EXACT_CAPTURE_V2_DIRECTORY --output NEW_NATIVE_SCREEN_OUTPUT
```
