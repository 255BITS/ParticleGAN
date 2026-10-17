# Results and the selected DualNorm row

Current benchmarks use one selected configuration per solution family. BCAP is
the winning DualNorm row below, with four of six required passes and Ring16 PASS
at 1,600 updates. The current inventory has seven family entries. Historical
optimizer/formulation alternatives and runtime cohorts remain on detail pages.
The V6 wrapper mistakenly ran a broader catalog; its complete archived counts
follow. The runner now selects the current family configurations without
repeating those completed experiments.

The complete fixed roster has 52 declarations: 23 admitted CUDA recipes,
24 preflight blockers and five declaration refusals. All 23 admitted recipes
finished their runnable Tier1 peers. The six required tasks yield 77 PASS,
59 FAIL and two capability BLOCKED cells; the separate timing diagnostic
yields seven PASS, 15 FAIL and one capability BLOCKED cell. No whole recipe
qualifies for Tier2. There are 161 paid attempts, no scientific retries and
5,910.478148 measured worker seconds. Earlier source costs remain separate.

An archived BCAP-with-K3P comparison row is
`bcap--08689a73c551728cc82434ac9601a06d1a9f3efa1a3999d5a3ec9e69746cc212`,
with five of six required passes. Its Gaussian acquisition and original word
curve gate both pass; Ring16 fails. Gaussian's endpoint CDF KS is 0.03562607,
and word quality/reconstruction are both 1. Ring16 still finds all 16 modes,
but component covariance error is 3.33788425 and HQ is 0.85595703.
Thus Gaussian is solvable under this fixed test: the failure below belongs to
the selected DualNorm recipe, whose Ring16 success cannot be pooled with
this alternative's Gaussian/word successes. This row is not the current BCAP
benchmark configuration. See the single
[technique inventory](../../technique-inventory.md) for the full roster.

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

Gaussian reaches a full primary pass at update 167: CDF KS 0.04797724,
mean error 0.06317 target standard deviations, and standard-deviation ratio
1.00432. At the same unchanged state, the independent confirmation has
CDF KS 0.05418494, above the frozen 0.05 bound; its moment bounds pass.
No other scheduled state earns a confirmed pass. At update 1000, mean is
1.96787909 and standard deviation 0.49788861 for target N(2,0.5²), but
CDF KS is 0.06460338. Close moments do not establish the required full
distribution match. This gate is acquisition smoke: it does not require a
stationary hold or learning-rate annealing.

The word result illustrates a different existing contract. At update 20001,
quality fraction is 1, all five words appear, mass TV is 0.01894531, exact
reconstruction is 1 and minimum reconstruction token probability is 1. The
complete 24-observation curve contains 13 passing observations, first at 1667,
and ends with only two passing checks. Its original Tier1 declaration still
requires a five-check passing terminal suffix. Therefore FAIL is the correct
verdict despite a passing endpoint. Gaussian's separate acquisition/hold split
was not silently applied to the word question.

The word worker completes all 20,001 updates in 826.670 seconds against its
unchanged 900-second deadline. The prior-source selected row recorded 704.703
seconds. These are measured campaign costs, not an isolated timing experiment
or a causal speed estimate. These word matrices already used SVD in the prior
source; the removed >1024-dimension Newton–Schulz path does not apply to them.

The fixed roster is complete; these failed revisions warrant no automatic
continuation or promotion. Inspect the selected DualNorm's paired Gaussian CDF
observations before declaring another
bounded comparison. DualNorm's word curve also loses and recovers an already
acquired solution. If Tier1 is to mean acquisition for words as well, declare
an explicit new word smoke/hold split rather than relabeling this original
FAIL. Keep continuous-learning hold in its separately declared Tier2 question.
Do not alter bounds, anneal the learning rate, select a new seed or pool passes
across global recipes to qualify this row.

The word-specific KA2 declaration cannot construct the scalar trainer required
by the Gaussian/Ring adapters. Its raw capability errors remain BLOCKED; the
[V4 readout](../final-v4/readout.json) already records BLOCKED raw errors in those
same two required cells and the timing diagnostic. These are distinct from
the general KA2 recipe and from numerical failures. Their costs remain visible.
