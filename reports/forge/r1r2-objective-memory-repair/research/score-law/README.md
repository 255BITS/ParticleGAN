# Score-law diagnosis on exact AMSGrad m110

**One native non-saturating contrast from the shared initializer is defensible
at low confidence.** Its sole recipe delta is `loss: relativistic →
non_saturating` on exact `r1r2-role-ams-m110-v1`, revision
`42d6df3eadaea8c88f227dc91f6f108324b25b8978cdf0a9efa3a3935a50cab4`.
The hypothesis concerns the critic's absolute-score anchoring and subsequent
trajectory. Reject an endpoint generator-starvation explanation and a blind
loss grid; no least-squares arm is supported here. This diagnostic registers
and executes nothing.

## Saved evidence does not show generator score starvation

The [reproducer](probe.py) and [receipt](diagnosis.json) use the exact Ring
endpoint checkpoint and its 4,096 saved samples, retaining the
[closed density result](../../density-readout.md). They construct no models,
use no autograd or samplers, and add no updates or grades. All 16 declared
target centers are a separate deterministic diagnostic fixture. Historical
real/fake training scores and pairings were not saved.

| Saved endpoint diagnostic | Range | Mean |
| --- | ---: | ---: |
| Critic score on saved fakes | −.160254–.257331 | .039517 |
| Critic score on target-center fixtures | −.086946–.254464 | .071616 |
| Analytic critic input-gradient norm | .008355–.587192 | .187253 |
| Paired G weight, uniformly averaged over centers | .453838–.557525 | .507977 |
| Native non-saturating G weight | .436020–.539978 | .490166 |
| Native least-squares G weight | .742669–1.160254 | .960483 |

For fixed scores, paired G ascent has coefficient `sigmoid(real − fake)`;
non-saturating has `sigmoid(−fake)`; default least squares has `1 − fake`.
Across every saved-fake/center combination, paired weights remain
.414771–.602219. Non-saturating/center-averaged paired weight ratios are
.960739–.968526, with cosine .999997668. Least-squares weights remain
positive. These alternatives do not reverse the endpoint input-force direction.

Under the explicit proxy `force = weight * ∇xD`, all three laws contract
tangent variance in all 16 components and expand radial variance in only four.
This agrees with a remaining shape conflict, not a missing logistic multiplier.
The proxy centers samples within each component; it excludes the generator
Jacobian, optimizer preconditioning and future critic refitting, so it does not
predict actual G parameter updates. G/D saved first-moment norms are .033328
and .071616 with beta1=.5; they are historical averages, not instantaneous
gradients. The beta1=0 prior buffer has terminal-gradient norm .016154.

## A different, bounded critic hypothesis

The paired critic objective `mean softplus(fake − real)` and both raw-score
R1/R2 penalties are invariant to adding a common scalar to all critic scores.
Their common-shift gradient and curvature are zero. The actual final critic
output-bias first, second and AMSGrad maximum-second moments are all zero,
consistent with that invariance.

The native non-saturating critic instead minimizes
`mean softplus(−real) + mean softplus(fake)` plus the same penalties.
It anchors the two score streams separately. On this fixture its common-shift
gradient is .027667 and curvature .498001. Thus it changes critic curvature
and real/fake coupling even where current G score weights are almost identical.
This is a distinct from-initialization mechanism contrast. The modest score
offset does **not** demonstrate harmful offset drift, and a flat constant-score
direction can be a harmless gauge. No endpoint reweighting is claimed to fix
the observed covariance defect.

[Mescheder et al., Theorem 4.1](https://proceedings.mlr.press/v80/mescheder18a/mescheder18a.pdf)
establish local convergence with zero-centered penalties under explicit
assumptions and sufficiently small gradient-descent steps; they do not guarantee
this finite Adam/AMSGrad run. [R3GAN, §§2.3–2.4](https://arxiv.org/html/2501.05441v1)
also establishes local convergence for regularized paired loss. Its
StackedMNIST experiment favors pairing: both R1/R2 variants train stably, but
paired/classic losses recover 1,000/693 modes. That is evidence against
expecting an unpaired diversity improvement, not proof about this Ring host.
The current Ring failure concerns component shape with all 16 modes present.

If admitted separately, retain native loss reductions without extra rescaling,
both R1/R2 at gamma .1 every update, native Adam/AMSGrad, D/G+E/prior beta1
.5/.5/0, beta2=.99, constant LR .0085 and the learned unregularized prior.
Retain every original task, sampling law, named stream, initializer, budget,
cadence and gate. Start from the shared initial tensors, not the saved endpoint.
The weak prediction is that changed critic anchoring/coupling permits component
shape to settle while preserving acquisition. **Repair falsifier:** Ring still
lacks five terminal full passes under covariance <=.85, minimum spread >=.15
and all other original bounds, or any other original task/clock fails. Lower
covariance alone is insufficient; a failure closes this contrast without an
automatic loss grid, continuation or rate adjustment.

## Public host support and batch consumption

| Original host | Existing loss consumption and ownership |
| --- | --- |
| Gaussian, Ring, own clock | `GANTrainer` calls the selected `d_loss`/`g_loss`. Both D and G phases retain their original batches and prior/noise draws; the real G scores are still computed even when the unpaired scalar G objective ignores them. |
| Direct pole | Public component binding returns `recipe.make_loss()`. Scalar G loss reaches the original direct coordinates; pairing and deterministic real batches remain task-owned. |
| Unused token | Component binding replaces the legacy loss with the selected public loss. Original scalar adversarial and unused-slot hold objectives remain active. |
| AE hold | Component binding selects the public scalar loss. Its real G scores remain detached; reconstruction, coverage, E/decoder/prior ownership and original target batches remain intact. |
| Five-word joint | The fixture calls `joint_g_loss` on `(G(z), z)` and `(x, E(x))`. Native non-saturating includes `softplus(−fake)` **and** `softplus(real)`, preserving reversed-label E gradients; default least squares similarly includes its reversed real stream. Using scalar `g_loss` alone would silently remove encoder ownership, but the actual host does not do that. |

Loss factories consume no RNG. The frozen source leaves data/prior sampling,
penalty calls and evaluation cadence outside the loss branch, so changing only
the declared loss needs no additional draws. This is source-level readiness;
an eventual admitted recipe still needs its own complete receipts and clock
measurement. Existing BCAP loss-search summaries in compiled memory retain
their different penalty, rates, tasks and source identities and confer no R1/R2
qualification.

The receipt verifies the execution commit, full frozen source manifest,
candidate revision, checkpoint file/complete-state hashes, saved-panel hash and
every stored named RNG hash against the original publication and density audit.
Functional float64 equations use the saved float32 tensors; deterministic
finite differences on 32 saved inputs agree with the analytic derivative within
1.98e-9. These offsets are separate numerical probes, not target draws or added
metric panels. Python/Torch CPU RNG states remain unchanged.

```sh
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-objective-memory-repair/research/score-law/probe.py > reports/forge/r1r2-objective-memory-repair/research/score-law/diagnosis.json
```
