# BISPF ablation: is BISPF's lead over BCAP just the prior tether?

Status: PREREGISTERED (written before any training, 2026-10-15). Results are appended below the rule.

## Hypothesis

BISPF (`bispf-v1`) scores one test more than the selected BCAP recipe
(`recipe-prior-refactor-bcap-pure-v1`): `mode_hold`. BISPF differs from that BCAP recipe in exactly
three recipe fields: `critic_formulation` (bcap -> bispf, which brings the Born-Infeld chord penalty,
Sinkhorn pairing, zero-mean logit gauge and harmonic thermal cooling), `reg_arm` (b_cap -> null), and the
prior regularizer (`vicreg` at weight 0, i.e. off -> `sbnp` at `prior_reg=0.01`).

Claim: the `mode_hold` repair comes from the SBNP prior term alone. SBNP pulls the learned prior table's
mean and covariance back toward their initial values (a "moment tether") plus a weak nearest-neighbour
repulsion, and it switches itself off when the table has at most `2*k_nn + d` rows (`k_nn=3`), which spares
the 5-row word table. On `mode_hold` the table has 12 rows in 4 dimensions, so 12 > 10 and the tether is on.

## Arms (one frozen source, seed 0, CUDA RTX A6000, own queue root)

| Arm | Candidate | Change vs C |
| --- | --- | --- |
| C (control) | `recipe-prior-refactor-bcap-pure-v1` | none |
| T (tether only) | `bcap-sbnp-tether-v1` (new opt-in idea card) | `prior_regularizer=sbnp`, `prior_reg=0.01`; critic stays bcap |
| B (full BISPF) | `bispf-v1` | BISPF as published |

Known non-identity: in B the SBNP loss is multiplied by BISPF's harmonic thermal factor when `d <= 4`
(straight-through cooling). In T no BISPF critic runs, so that factor stays 1. T therefore tests the
tether at its plain strength.

## Preregistered decision rule

Verdicts are the Forge sustained gates (terminal-suffix passes, e.g. `mode_hold` needs at least 5 terminal
passing checks of `modes>=8` and `hq>=0.9` out of 24), never endpoint metrics.

Stage 1: `mode_hold` on C, T and B.
Stage 2 (only if stage 1 leaves the claim open): `gaussian1d_smoke`, `two_pole`, `ring16_acquisition`,
`five_word_joint_smoke`, `five_word_joint_hold` on C and T. B's stage 2 runs only if budget is left.

1. Precondition: C FAILS `mode_hold` and B PASSES `mode_hold` on this source. If C passes, the lead does not
   exist on the current source; verdict INCONCLUSIVE (claim moot). If B fails, BISPF's lead is not reproduced;
   verdict INCONCLUSIVE for attribution (T is still reported).
2. If the precondition holds and T FAILS `mode_hold`: REFUTED (the tether alone is not sufficient; some
   critic-side part of BISPF is needed). Stop; stage 2 is not run.
3. If the precondition holds and T PASSES `mode_hold`: run stage 2 on C and T.
   - SUPPORTED if T passes every stage-2 test that C passes.
   - REFUTED (as stated: "everything else equal") if T loses any stage-2 test that C passes. The readout
     then names the lost test; the tether reproduces the repair but at a cost.
4. Any INCOMPLETE/INVALID/TIMEOUT/BLOCKED cell that the rule needs makes the verdict INCONCLUSIVE for that
   cell, reported with its error; no reruns with other seeds.

Budget: at most about 2 GPU-hours of training. One job per GPU through the workflow lease helper.

## Exploratory follow-up (added after the preregistered verdict was decided; not part of the rule)

Stage 1 decided the preregistered rule (see Results). One cheap complementary arm was then declared before
running it: B0 = `bispf-no-prior-tether-v1`, which is `bispf-v1` with `prior_reg=0.0` (SBNP off, every
critic-side BISPF part unchanged), `mode_hold` only, same frozen source if the digest matches.
Reading: if B0 PASSES `mode_hold`, BISPF's repair comes from its critic-side parts and the tether is not
needed; if B0 FAILS, the repair needs the tether together with BISPF's critic (for example the thermally
scaled tether), and neither part alone is enough. This is a single-seed exploratory read.
