# What the selected DualNorm row shows

This is the pre-run selected BCAP/DualNorm configuration
`bcap-dualnorm--7beb7378d81dc3be2c648438661e0376fe2805298232f5c2398be835ddaad6f9`:
constant generator step 0.012, critic multiplier 1.5, prior multiplier 2.5,
zero momentum and zero prior regularizer. It remains one global trainer recipe.
Each task retains its own original architecture, target, prior and budget.
All six required Tier1 cells completed on CUDA under the V6 source identity.
Four pass; Gaussian and word fail. No Tier2 qualification can pool another
candidate's Gaussian/word passes into this row.

| Required task | Recorded gate | Explanation |
| --- | --- | --- |
| Two pole | PASS | Original curve and terminal gate pass. |
| Unused token | PASS | Original curve and terminal gate pass. |
| AE/GAN hold | PASS | Original curve and terminal gate pass. |
| Ring16 acquisition | PASS | All 16 modes, HQ 0.95605, mass TV 0.07861, component covariance error 0.47068. |
| Gaussian acquisition smoke | FAIL | One scheduled full pass; no independent same-state confirmation. |
| Five-word joint acquisition | FAIL | Final quality/reconstruction pass, but only two terminal passing checks; original gate requires five. |

Ring16's endpoint reproduces the earlier combination diagnostic's reported
covariance/HQ/TV values. Its current gate is independently bound to the actual
V6 public recipe, prior, initializer, budget and sampling law. The earlier
source does not supply qualification credit. This supports adopting the
implementation for this exact Ring16 question; it does not establish a
whole-recipe smoke pass or general calibration.

Gaussian reaches a full primary pass at update167: CDF KS 0.04797724,
mean error 0.06317 target standard deviations, and standard-deviation ratio
1.00432. At the same unchanged state, the independent confirmation has
CDF KS 0.05418494, above the frozen 0.05 bound; its moment bounds pass.
No other scheduled state earns a confirmed pass. At update1000, mean is
1.96787909 and standard deviation 0.49788861 for target N(2,0.5²), but
CDF KS is 0.06460338. Close moments do not establish the required full
distribution match. This gate is acquisition smoke: it does not require a
stationary hold or learning-rate annealing.

The word result illustrates a different existing contract. At update20001,
quality fraction is1, all five words appear, mass TV is0.01894531, exact
reconstruction is1 and minimum reconstruction token probability is1. The
complete24-observation curve contains13 passing observations, first at1667,
and ends with only two passing checks. Its original Tier1 declaration still
requires a five-check passing terminal suffix. Therefore FAIL is the correct
verdict despite a passing endpoint. Gaussian's separate acquisition/hold split
was not silently applied to the word question.

The word worker completes all20001 updates in826.670 seconds against its
unchanged900-second deadline. The prior-source selected row recorded704.703
seconds. These are measured campaign costs, not an isolated timing experiment
or a causal speed estimate. These word matrices already used SVD in the prior
source; the removed >1024-dimension Newton–Schulz path does not apply to them.

Finish and publish the full fixed roster before proposing another trainer
change. The immediate remaining questions are confirmed Gaussian acquisition
under fixed scheduling, and the word curve's loss/recovery of an already
acquired solution. If Tier1 is to mean acquisition for words as well, declare
an explicit new word smoke/hold split rather than relabeling this original
FAIL. Keep continuous-learning hold in its separately declared Tier2 question.
Do not alter bounds, anneal the learning rate, select a new seed or pool passes
across global recipes to qualify this row.
