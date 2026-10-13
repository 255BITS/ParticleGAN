# Saved-gradient normalization readout

Recommend exactly one prospective global delta: **`optimizer_family: adam → ada_nsgda`**
from the constant parent
`r1r2--169e9539b2dcfaab6b51e17f02907f28ea5b894f77155f7e688ab07a789de2b9`
(revision `35ebed78fe69b8923ead8335ea38bda209eb53ecc910b5e57ce5bac1c4b2f775`).
Keep paired `a_r1r2`, coefficient .1 on every update, LR .0085, betas (0,.99),
role multipliers 1, learned prior, zero extra prior regularization and constant
schedules. Apply the same recipe to all six unchanged ordinary tasks and the
clock audit. This readout creates no candidate declaration or execution.

## Conditional direction evidence

The [compact receipt](normalization.json) verifies three original parent
certificates and complete saved states using the existing publisher. Beta1=0
makes each stored first moment the actual last parameter gradient. We compare
supported laws on these **same endpoint gradients and native second moments**;
this does not predict their trained endpoint states.

| Saved task / role | Native cosine to raw gradient | Ada cosine | DualNorm cosine | Native step L2 | Ada step L2 | DualNorm step L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ring16 G | .41993 | .56748 | .40538 | .12325 | .12325 | .09036 |
| Ring16 D | .28273 | .43726 | .32760 | .31681 | .31681 | .08064 |
| Ring16 prior | .93164 | 1.00000 | .83358 | .18234 | .18234 | .08710 |
| Gaussian G | .41676 | .55701 | .36893 | .02871 | .02871 | .05765 |
| Gaussian D | .29387 | .38680 | .29375 | .09479 | .09479 | .05240 |
| Gaussian prior | .80118 | 1.00000 | .81628 | .05625 | .05625 | .08701 |
| Direct pole G | .99830 | 1.00000 | 1.00000 | .02424 | .02424 | .02944 |

Ada preserves each nonzero **parameter tensor's** gradient direction exactly
(cosine within floating-point precision of 1) and retains its native Adam
Frobenius step norm up to the norm epsilon. Role-wide cosine remains below 1
because different parameter tensors retain different Adam gains. The Ring16
64×64 tensors have native cosines .61343 for G and .54063 for D; their DualNorm
polar directions have cosines .27924 and .37463. DualNorm therefore changes
both directions and role magnitudes more broadly, while its row-normalized
prior also removes the relative gradient magnitudes between active rows.

**Hypothesis:** preserving within-tensor gradient geometry reduces component
shape distortion while keeping the constant parent's tensor step magnitudes.
The saved [density diagnosis](../../r1r2-five-track-repair/density-readout.md)
shows excessive tangent variance alongside occasional radial compression; the
observed optimizer rotations provide a concrete mechanism to test, without
establishing that they caused these density errors. The passing Gaussian and
pole are necessary controls. Nothing here establishes a solution for words.

**Falsifier:** the unchanged sustained Ring16 gates still fail, or a previously
passing ordinary task/clock audit fails. A covariance improvement alone grants
no pass; all original shape, spread, coverage and terminal-suffix gates apply.

## Actual API and pole bounds

The current [Ada step law](../../../../particlegan/optim/dualnorm.py) uses
`g/(||g||+eps) * ||g/(sqrt(v_hat)+eps)||` per parameter, then applies LR once.
Its public Recipe validator requires beta1=0 for G/E, D and prior; the parent
satisfies this, with no beta2 schedule, intervention, guard, anchor, smoothing,
proximal, geometry or continuous-policy setting. The existing public factories
route G/E, prior and critic through this law. Behavioral hosts use those same
factories; direct generated coordinates explicitly receive generator ownership
in [Recipe.make_generator_optimizer](../../../../particlegan/recipes.py).
No architecture-specific fallback or host change is needed by this proposal.
Saved prior rownorm directions are computed only on nonzero gradient rows;
final sampled IDs were not retained and we synthesize no ownership receipt.

Dense DualNorm actually uses **`sqrt(max(1,out/in))`**. Its wide output layers
therefore receive factor 1, rather than a factor below 1. For the original
12×1 direct-pole tensor at LR .0085, a nonzero polar step has L2 .02944486.
From the original zero fixture, mean absolute displacement is bounded above by
.0085 per update, hence **.68 after 80 updates**. This bound permits the .3
gate; it does not guarantee the required sustained suffix or critic gradient
bound. Unscaled global normalization would instead bound total mean absolute
travel by .19630 and would rule out the gate at this LR. Ada uses the Adam
norm graft and has no such fixed unit-L2 barrier; its saved last pole step
retains native L2 .02424012 and changes direction by only 3.34 degrees.

## Prior evidence and provenance

The receipt recalls every original point in
[BCAP AdaNSGDA](../../configuration-search/bcap-optim-ada-nsgda-tier1-v1.json)
and [zero-momentum BCAP DualNorm](../../configuration-search/bcap-optim-dualnorm-zero-tier1-v1.json),
with exact file hashes, candidate revisions and task outcomes. Ada at .00025,
.001, .00425 and .016 failed Ring16 and Gaussian in all four recipes; words
passed only at .00025. Every unsmoothed DualNorm point also failed Ring16 and
Gaussian. These are negative prior evidence under source
`c5c60a8018e144c447325f3705dd398d04069a91bb294eaaa76d37084ade67cc` and a
different BCAP formulation/task cohort. The proposed fixed R1/R2 .0085 point
does not repeat one of those complete recipes or inherit their qualification.

Current parent attempts are Ring16 `27c75f3d768e4b789a488a3fdc5a2e90`, Gaussian
`a1d5d58e4f754bbf87e2996d9da714b3` and pole
`d215b1648cf5462db90757c8834a681f`, all under source
`25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`.
Exact checkpoint, artifact-manifest, publication, optimizer-source hashes and
runtime bindings are in the receipt. CPU float32 uses the actual polar helper;
accelerator rounding may differ. No optimizer/model was constructed, no draw
or update was executed, no gate was recomputed, and Python/Torch RNG stayed
unchanged. Full supported candidate preflight remains the registration owner's
responsibility before execution.

```sh
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-role-repair/optimizer/normalization_analyze.py > reports/forge/r1r2-role-repair/optimizer/normalization.json
```
