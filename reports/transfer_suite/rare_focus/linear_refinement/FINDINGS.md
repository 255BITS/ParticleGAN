# Final linear-bypass refinement

**The original 256-particle recipe now has a sustained rare-mode witness.** Axis-Fourier2, Softplus(beta5), width96×2 with a learned raw linear bypass passes the final **6/24** live observations. The same critic's complete data profile is **3/6**, so the remaining per-task architecture choices stay explicit.

| Task | Sustained live | Final suffix | Final failing metric |
| --- | --- | ---: | --- |
| Rare mass | PASS | 6/24 | — |
| Broad modes | PASS | 15/24 | — |
| Unequal width | FAIL | 0/24 | Component covariance error8.70313 > .85 |
| Anisotropic | FAIL | 0/24 | Component covariance error .89238 > .85 |
| Overlap | FAIL | 0/24 | Global covariance error .50031 > .45 |
| Spiral | PASS | 23/24 | — |

The rare run first passes at step950, confirms five consecutive final checks at1,150 and stays passing through1,200. Final live metrics:

| Metric | Value | Requirement |
| --- | ---: | ---: |
| Normalized sliced distance | .065635 | ≤ .18 |
| Mass TV | .046826 | ≤ .15 |
| HQ | .997070 | ≥ .85 |
| Component covariance error | .441314 | ≤ .85 |
| Minimum component eigenvalue ratio | .269769 | ≥ .15 |
| Minimum mass ratio | .914862 | ≥ .25 |

Final EMA also passes separately: HQ .982666, covariance error .378588 and minimum eigenvalue ratio .376946. EMA contributes nothing to the live verdict. Generated rare-component mass is2.856% against a2% target; full component counts and covariance errors remain recorded.

The critic computes `main(x) + Linear(2,1,bias=False)(x)`. The added weights initialize to zero. This is two additional learned D parameters:10,467 total, compared with10,465 for the plain width96 smooth critic. The base initializes first. Static control checks reproduce the earlier width64/beta5 implementation exactly in state, output, input gradients, parameter gradients and active cap value. The control is a static parity check, not another training episode.

The four refinements were frozen in advance: width64/beta6, width64/beta10, width96/beta5 and width96/beta6. Only width96/beta5 passed rare mass; it then received the other five data toys. The three failed refinements remain in the [matrix](MATRIX.md). No additional cards or training extensions were run.

All settings outside D architecture remain original: G64×2/z4,256 particles,batch128,Adam(0,.99),G/D/prior LRs .001/.0015/.01,Rp logistic,b_cap coefficient3/kappa1.25,prior regularization .05,no particle L2,cosine and1:1 updates. Original budgets are1,200 steps, except spiral1,600. Every curve has24 fixed live checks with unchanged metric bounds and the final-five rule. There is no target-derived feature, normalization, extra loss, seed search or Fourier-amplitude sweep.

Exactly **9 GAN episodes**, **216 live observations**, **64.604173 seconds** summed recorded wall time. This bundle contains the screen and complete six-task candidate profile. Parent independent replay and aggregate required/image validation are separate evidence, with their own run counts.

Exact module `benchmarks/transfer_suite/linear_skip_refinement_research.py` SHA256 is `399a6d6dc268bd58952366c5d11e57886faf2cdb7accf882c47fe1e846e4bd1c`. Numerical source remained unchanged during the experiment. All61 dependency files, drivers, cards, architecture declarations, original/effective specs, failures, live/EMA curves, actions, update counts and runtime are archived.
