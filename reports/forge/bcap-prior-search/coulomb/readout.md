# Coulomb prior: completed ordinary comparison

**No declared strength converted a matched Tier 2 FAIL to PASS.** Keep the BCAP
leader. The weakest strength failed Tier 1 admission; both admitted strengths
lost one different existing Tier 2 pass. The finite round is complete, with no
seed study, refinement, Tier 3 run or default adoption.

| Global `prior_reg` | Tier 1 | Tier 2 | Repairs | Regression |
| ---: | --- | --- | --- | --- |
| Matched leader, 0 | 6/6 | 9/21 | Control | — |
| .001 | 5/6 | Blocked; 0/21 measured | None observed; Tier 2 unmeasured | Ring16 acquisition, Tier 1 |
| .01 | 6/6 | 8/21 | 0 | Five-word joint hold |
| .1 | 6/6 | 8/21 | 0 | Two broad vector modes |

The optional smaller clock probe passed in all four arms. It supplies a timing
measurement, with no feature invariance credit. [All task measurements](README.md)
list each score, target and verdict for all **112 arm/task cells**; diagnostic
arrays are retained element by element and absent values remain unavailable.
[results.json](results.json) binds the applied recipe, initialization, prior,
sampling and original certificates. [Saved training GIFs](media/README.md) and
their [receipt index](media/index.json) illustrate all 91 measured PASS/FAIL
attempts, including failures; the 21 blocked cells have no GIF or invented score.
The goal's [current leaderboard](../../technique-inventory.md) remains its sole
generated leaderboard.

## Failures that decide this bracket

At .001, Ring16 retains all 16 modes, good mass allocation and high quality but
fails full assigned-component shape. Its endpoint has covariance error
**1.454606354 / ≤.85**, against core-only error .474356234 (diagnostic). The
aggregate covariance error .073898688 is also diagnostic. One full component
error is 13.350334; the .85 bound applies to the declared aggregate component
metric, rather than separately to every array element. All 96 scheduled checks
fail the full gate, leaving a **0 / ≥5** terminal suffix. Endpoint modes
**16 / ≥16**, quality **.949707031 / ≥.85**, mass TV **.056396484 / ≤.15** and
minimum eigen ratio **.358407885 / ≥.15** pass. Tier 2 remains unmeasured.

At .01, the word hold passes its endpoint but fails one mandatory primary check
and its independent confirmation, leaving **24 / ==25** in each cohort. Both
fail at total optimizer step **5001**:

| Measurement | Primary score | Confirmation score | Target |
| --- | ---: | ---: | --- |
| Quality fraction | .7822265625 | .796875 | ≥.95 |
| Modes | 4 | 4 | ==5 |
| Mass TV | .2357421875 | .209375 | ≤.1 |
| Exact reconstruction | 0 | 0 | ==1 |
| Minimum reconstruction token probability | .345343709 | .345343709 | ≥.9 |

The complete endpoint at step 8167 passes: quality1, five modes, TV .02265625,
exact reconstruction1 and minimum probability .999906898. These endpoint values
do not replace the mandatory hold checks. The table's original packet hashes
and cohort hashes are in [README.md](README.md) and [results.json](results.json).

At .1, `vector_two_broad` also passes every endpoint numerical bound, but its
terminal passing suffix is **3 / ≥5**. Step1050 interrupts the required final
window [1000, 1050, 1100, 1150, 1200]:

| Measurement | Step1050 score | Endpoint score | Target |
| --- | ---: | ---: | --- |
| Normalized SW1 | **.244543330** | .175025456 | ≤.18 |
| Mass TV | .072509766 | .060546875 | ≤.15 |
| Quality | .929931641 | .991210938 | ≥.85 |
| Full component covariance error | .697775543 | .478606895 | ≤.85 |
| Minimum component eigen ratio | .213923752 | .374263018 | ≥.15 |

The [audit receipt](audit.json) retains bound witnesses with their original
attempt and observation identities. This is an explanation from saved packets;
the original complete-curve FAIL certificate remains authoritative.

## 100-Gaussian density remains far from its gate

These are the original independent live holdout metrics on 100,000 saved draws.
Each measured row fails. The .001 arm never reaches these tasks.

| Task | Strength | Precision / ≥.97 | Radial KS / ≤.04 | Mass TV / ≤.06 |
| --- | ---: | ---: | ---: | ---: |
| Grid100 | Control | .24072 | .490882957 | .14718 |
| Grid100 | .01 | .28030 | .417246162 | .14230 |
| Grid100 | .1 | .27582 | .419202821 | .16482 |
| Rotated100 | Control | .25552 | Unavailable | .14325 |
| Rotated100 | .01 | .28116 | .393920206 | .13708 |
| Rotated100 | .1 | .28306 | Unavailable | .13717 |
| Staggered100 | Control | .30168 | Unavailable | .12277 |
| Staggered100 | .01 | .32204 | .391153730 | .12469 |
| Staggered100 | .1 | .34003 | .381504582 | .11494 |

Small precision improvements do not approach the target, and mass allocation
does not consistently improve. For example .1 worsens Grid100 mass TV. The
existing oracle diagnostic passes Grid100 (.98914 precision, .001514121 KS,
.01226 TV), confirming its scoring fixture can represent the target. Oracle,
EMA, endpoint coverage and holdout coverage diagnostics remain separate cohorts
and provide no live candidate qualification credit. Missing original KS values
are preserved even when other diagnostics are available.

## What the prior actually changed

The package holds softening .1, confinement .35 and neighbor cap 8 fixed. It
uses softened inverse-distance repulsion and a quadratic well around the
public-initializer centroid and RMS radius. That reference is captured after
initialization and restored from the checkpoint. The deterministic evenly
spaced cyclic-offset rule acts on host-supplied rows; numeric offsets depend
on that row count. `GANTrainer` supplies its whole prior for ≤1024 particles
and unique G-phase sampled rows above 1024. Native tasks have 20,000 rows, so
their interaction nodes and offsets vary with each supplied sampled batch.

The [saved geometry diagnostic](prior-geometry.json), reproduced by
[prior_geometry.py](prior_geometry.py), measures each full 20,000-row table
without forwards, updates or draws. At Grid100, RMS radius relative to the
initialized radius grows from 1.091502 in the control to 1.110004 at .01 and
1.226609 at .1; unweighted whole-table energy falls from 1.475357 to 1.472999
and1.432804. Rotated100 and Staggered100 show similar expansion and lower
repulsive energy. Thus the penalty changes latent geometry while output density
still fails. These final float64 table energies are diagnostics, with no gate;
they are not historical minibatch objectives or measured optimizer forces.

For the leader, each unique sampled prior row moves by
`-.030*g/sqrt(||g||²+.001²)`, with the actual combined GAN and auxiliary gradient
`g`. Unsampled rows remain fixed even when a whole-table penalty has gradients
there. Smoothed normalization makes coefficient-to-displacement nonlinear; an
energy decrease does not imply a proportionate force or output improvement.
The [bounded public-API demo](../../prior-coulomb/README.md) records actual
combined displacement and auxiliary-only scaling, without adding a research
cohort to this search. Some task-owned priors do not sample a latent table; their
`prior_policy.applicability` records that scope explicitly.

## Identity, continuation, budget and verification

The executed commit is **`2ce01aec33c644ded0cf7f29b130bdb091e643ff`**, frozen digest
**`f56cd32fd97769f6863f8709efd3c96a4137349b46882124c713d1917d9d062c`**, on
`origin/develop` **`0337bbbc2300fd351e77e9a222b92401137fd421`**. Later commits
publish reports only. The [spec](spec.json) and [registration](registration.json)
retain the finite declared bracket and original budgets.

Every arm inherits the explicit BCAP leader: direction blend, non-saturating
loss, G/D/prior rates .012/.018/.030, native SVD, DualNorm smoothing .001 and
`per_offset`, no EMA or added training/output noise, `prior_l2=0`. The intrinsic
task-owned sigma-.025 MoG sampling remains present. Protocol seed 0, public
deterministic initialization, task architecture/data, initial tensors and prior
capacity, sampling, updates and evaluation cadence stay fixed. The only global
recipe delta is `prior_regularizer=vicreg→coulomb` and `prior_reg=0→strength`.
Resolved task ownership packets embed those two candidate values; the audit
allows precisely that metadata delta and verifies all other task bytes match.

Word holds resume each arm's own certified earliest acquisition prefix, then
execute 4000 new updates without resetting training state. Control and .1 start
at 834 and finish 4834; .01 starts 4167 and finishes 8167. The latter consumes its
own post-acquisition window, so its hold is not a common total-step window.
The audit checks consumed non-evaluation streams and word data generators at
each fixed common budget and each matched word prefix, retaining this explicit
exception. Adaptive ring hosts, where present, retain their actual host counter;
the executed Ring16 acquisition here uses its full fixed 1600 updates. No
per-stream draw-count instrumentation is claimed.

The [identity audit](audit.json), reproduced by [audit.py](audit.py), validates
all 91 original measured certificates, actual recipe/prior/init/sampling and
checkpoint stream states. The fresh disabled leader matches the archived
`bcap-pure/ordinary` record on **28/28 gates and every retained metric value**.
Full metric dictionaries match 19/28; nine older archived records omit current
diagnostic arrays. Archived evidence stays under its original source contract
and is not reused for fresh qualification.

Actual charged cost is **7928.274754355 seconds**: control 2592.575775787,
positive search 5335.698978568; remaining reservation 0. The declaration allows
45,000 seconds per arm and 135,000 for the three-strength search, with 172,080
seconds full-task worst-case reservation under a 180,000 maximum. Intentional
GPU sharing makes these costs unsuitable for a speed ranking.

The [infrastructure receipt](../infrastructure.json) retains the original five
interrupted control-word attempts and their charged costs. Coulomb's canceled
attempt `b8487644dce44fa59a0a4cf7eb38b932` contributes 396.397135245 seconds
already included in the control total. Original 900-second word allowance and
20,001 updates were retained. There were zero scientific retries. The
[MPS fixture](../mps_probe.json) checks eight exact byte digests, and the
[control-word audit](../control_word_audit.json) finds identical scientific
acquisition/endpoint state, optimizers, named streams and all 24 observation
packets across the five controls; unused caller-global RNG snapshots differ.
These software proofs add no optimizer update or scoring draw.

Feature verification passed 286 targeted checks, including CPU/CUDA formula and
gradient parity, sampled-row ownership, post-initializer reference capture,
checkpoint continuation and exact disabled parity. The common report helpers
passed 36 checks with 12 saved training GIF fixtures. The publication-memory
regression suite passes 20 checks; Forge validate passes, summaries-only
compilation is CURRENT, and all 211 local report links resolve. Final publication,
identity/geometry audits and GIF export use saved evidence only and add zero
training updates or sampling draws. Bulk stdout, JSONL and checkpoints remain
outside Git under `/mnt/ml7tb/ParticleGAN-forge/bcap-prior-search-20261010/coulomb`;
`tail -F .../logs/final-media.log` follows this publication.

Reproduce saved-only publication from this report checkout with the archive
available (publishing regenerates README.md's tables; preserve its report links):

```sh
PYTHONPATH=. python reports/forge/bcap-prior-search/workflow.py publish \
  --name coulomb --archive /mnt/ml7tb/ParticleGAN-forge/bcap-prior-search-20261010/coulomb
PYTHONPATH=. python reports/forge/bcap-prior-search/export_saved_media.py \
  --name coulomb --archive /mnt/ml7tb/ParticleGAN-forge/bcap-prior-search-20261010/coulomb
PYTHONPATH=. python reports/forge/bcap-prior-search/coulomb/audit.py
PYTHONPATH=. python reports/forge/bcap-prior-search/coulomb/prior_geometry.py
python -m experiments.forge compile --summaries-only
python -m experiments.forge compile --check
```

Stop this declared bracket. It falsifies the specific global repair hypothesis
for the fixed .1/.35/8 package at these three strengths. It does not settle every
possible Coulomb formulation. The useful next question, if separately declared,
is whether a prior mechanism can improve output density and component shape
while keeping the mandatory hold windows stable. Latent expansion alone did
not accomplish that here.
