# Nonlocal proposals scored only by the original generator loss

One free-output move adds a fourth mode to the copied cold three-mode state
and preserves all eight modes in two warm controls. All three moves decrease
the unchanged Rp generator loss on the native selection batch and each of
eight subsequent saved-stream G batches. This passes the declared one-move
falsifier. It is not a neural update, acquisition result or stability claim.

The proposal enumerates each of12 clean output rows as donor and each of128
actual D-real minibatch samples as destination. It selects the minimum
`mean softplus(smooth_D(real) - smooth_D(fake))`, retaining the exact native
particle indices, output-noise draws and fixed five-point stencil width`.15`.
Only occurrences of the chosen donor change. No centroid, distance,
coverage objective, mode count or quality statistic enters this selection.
The native full loss is reevaluated for the unique winner; no second search
occurs. The eight other G batches only evaluate that frozen proposal.

| Copied state and critic | Native G loss | Heldout mean G loss | Modes | Paired HQ |
| --- | ---: | ---: | ---: | ---: |
| Warm1530, native accepted D | .742164→.706754 | .772543→.739360 | 8→8 | .968994→.968994 |
| Same warm G, archived locally refined D | .770716→.719505 | .813098→.764367 | 8→8 | .968994→.961914 |
| Cold472, archived best finite D | 5.570960→4.721950 | 5.510776→5.125791 | 3→4 | .752441→.776123 |

Each heldout loss decreases individually,24/24 total. The warm native-D
move reallocates a duplicate from mode6 to mode0; the refined-D control
reallocates a duplicate from mode4 to mode0. The cold move reallocates one
of four particles in mode1 to previously absent mode4. Those descriptions
use target centers only after selection. Requested output displacements
are respectively4.116,6.206 and5.837; whether the actual G/prior can realize
them must be tested separately. In particular, a local gradient direction
and a bounded scalar rate do not implement these jumps.

The warm native critic is the exact captured PR84 accepted D at1530. The
second warm critic is the previously archived1024-pair40-iteration/80-closure
local fit against that unchanged generator; it remains **nonconverged**.
The cold critic is the best finite iterate used by the repaired472 update,
not the nonfinite original fit endpoint. No new critic fit or optimality
claim is made. The raw input archives and their compressed-byte hashes are
bound in the declaration.

All three native G-loss values reproduce bitwise. Original native warm and
guarded cold G/prior/Adam endpoints reproduce their archived full hashes;
the refined warm control reproduces its complete archived G receipt. The
separable enumeration agrees with full native loss reevaluation within
`3.27e-7`, much smaller than any selected decrease. Its use of float64
summation and different forward batch sizes explains the small rounding
difference. A constant critic rests without a proposal. The three analytic
tests also verify full-loss enumeration, Rp common-bias invariance, absent
donor handling, unchanged data and RNG. All3 pass in1.78 seconds.

The three-state execution takes1.36 seconds on pinned single-thread CPU,
PyTorch2.13.0+cu126/AVX2. Model copies, source payloads and caller RNG remain
unchanged; no live moments or schedules advance. Sharp D logistic-plus-cap
losses are recorded before and after the selected move on two copied D
banks; they rise in each case, as expected when a proposal becomes harder
for that fixed discriminator. They do not select or certify the proposal.

The frozen-critic limitation remains decisive: the original G objective is
separable over fake samples and can reward moving many particles to the
same high-score region. One warm-safe move does not make repeated frozen-D
global search safe. The next bounded test is actual joint G/prior numerical
landing of these already selected clouds, accepting only a decrease of
the original G loss and reporting heldout losses afterward. Any subsequent
alternating-D trajectory needs its own declaration; this assay grants no
native training eligibility.

[Frozen evidence](continuous-evidence/gan-nonlocal-output/manifest.json)
contains the result, all1536 scores per case, target clouds, declarations,
source snapshots and log. No PR88 continuation was launched after the
user restricted candidate work to the GAN formulation.

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 \
  ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES='' \
  /tmp/pr38-default-env/bin/python -u reports/toy100/pr84_adversarial_reallocation_assay.py \
  --output NEW_OUTPUT_DIRECTORY
```
