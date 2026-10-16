# dig2 harmonic-link: what gives BISPF its mode_hold pass?

Status: PREREGISTERED (rule written 2026-10-16 before any training on this branch). Results are appended
below the rule.

## Background

Round 1 (#426, `bispf-ablation`) on one frozen source, seed 0: the BCAP control fails `mode_hold` (3 terminal
passes, needs 5), BCAP + SBNP prior tether (`bcap-sbnp-tether-v1`) fails (stuck at 7 of 8 modes), BISPF without
the tether fails (7 of 8), full BISPF (`bispf-v1`) passes (12 terminal passes). So the pass needs the tether
together with some critic-side part of BISPF. The suggested link was BISPF's "harmonic/residual thermal factor".

Terms used here:

- **SBNP tether**: BISPF's prior regularizer. It pulls the learned latent table's mean and covariance back to
  their initial values plus a weak nearest-neighbour repulsion (`prior_regularizer=sbnp`, `prior_reg=0.01`).
- **Capture factor** (what the code calls "Harmonic Mode Capture", `_BISPFPenaltyKernel.penalty`): from critic
  penalty call 1000 on, for data dimension <= 4 and batches >= 64, if the 95th percentile of real-to-nearest-fake
  and fake-to-nearest-real distances (divided by the median distance) both fall below 0.038 (0.048 once
  captured), it sets a scale of 1e-5, else 1. That scale multiplies the *gradient* (straight-through, the value
  is unchanged) of (a) every critic output outside the penalty, which includes the generator's adversarial
  gradient, (b) the critic penalty, and (c) the SBNP tether when latent dimension <= 4.
- **Residual thermal noise**: `BISPFEngine` / `StratifiedBlueNoisePrior` drive a `thermal_ratio` from the
  Sinkhorn residual. Code reading: the recipe/Forge path never constructs either class (the Forge recipe uses
  `BISPFCriticPenalty` + the recipe's `GANLoss('non_saturating')` + `SBNPPriorRegularizer`), so this noise is not
  active in any Forge BISPF result. It is not an arm here; there is nothing to switch off.
- **BISPF penalty kernel parts** that are active in the Forge path, with `reg_coeff=1`, `reg_kappa=1`:
  cap `0.5*(E sp(|g_r|-1)^2 + 0.98 E sp(|g_f|-1)^2 + 0.02 E sp(|g_c|-1)^2)` where `sp` is softplus with beta 30
  and `g_c` is the critic's input gradient on a Sinkhorn transport chord between a fake and a matched real
  ("chord"); a floor `0.5*0.0005*mean|g|^2/(3*dim)`; a zero-mean logit gauge `0.5*0.0005*mean_logit^2`.
  BCAP's `b_cap` is `0.5*(E relu(|g_r|-1)^2 + E relu(|g_f|-1)^2)`. So BISPF's critic differs from BCAP's in
  four small kernel parts (softplus vs hinge, chord, floor, gauge) plus the capture factor.

Code-derived prediction, stated before running: in `mode_hold` (1,200 updates, one critic penalty call per
update) the capture factor cannot act before update 1000. Round-1 BISPF already held 8 modes from update 350.
So switching capture off (or only its tether scaling) cannot change anything up to update 1000 and is predicted
NOT to remove the pass. The lead's "thermal link" is therefore expected to be refuted; the experiment checks it
and then looks for the actual link among the kernel parts.

## Mechanism (opt-in, portable)

New recipe field `bispf_ablation` (technique-owned; default `"none"` keeps BISPF bit-identical, verified by a
penalty-and-gradient equality test against the previous `bispf.py` over 1,105 calls including the capture
branch). It is a sorted, comma-separated subset of: `capture`, `tether_thermal`, `chord`, `softplus`, `floor`,
`gauge`; each named part is switched off. Allowed only with `critic_formulation="bispf"`. With all of
`capture,chord,floor,gauge,softplus,tether_thermal` off, the BISPF penalty is exactly BCAP's `b_cap` formula
(unit test), so that arm is "BCAP + SBNP tether written inside the bispf formulation". Residual
non-identities with `bcap-sbnp-tether-v1`: BISPF keeps no EMA critic copy (BCAP keeps one but its anchor weight is
0) and attaches its critic forward hook. None of the switches uses data statistics, task names or RNG.
Diagnostic only: `PARTICLEGAN_BISPF_CAPTURE_LOG=1` prints the capture test to stderr every 50 calls and on change.

## Arms (one frozen source, seed 0, CUDA RTX A6000, own queue root)

| Arm | Candidate | `bispf_ablation` (parts OFF) |
| --- | --- | --- |
| C BCAP control | `recipe-prior-refactor-bcap-pure-v1` | (bcap formulation) |
| T BCAP + tether | `bcap-sbnp-tether-v1` | (bcap formulation) |
| B full BISPF | `bispf-v1` | none |
| H0 | `bispf-hl-no-capture-v1` | capture |
| S0 | `bispf-hl-no-tether-thermal-v1` | tether_thermal |
| K0 | `bispf-hl-no-chord-v1` | chord |
| P0 | `bispf-hl-hinge-cap-v1` | softplus |
| F0 | `bispf-hl-no-floor-gauge-v1` | floor,gauge |
| Z reduced (= BCAP b_cap + tether) | `bispf-hl-reduced-v1` | capture,chord,floor,gauge,softplus,tether_thermal |
| Z+cap (= BCAP + tether + capture schedule) | `bispf-hl-reduced-plus-capture-v1` | chord,floor,gauge,softplus |
| Z+sp | `bispf-hl-reduced-plus-softplus-v1` | capture,chord,floor,gauge,tether_thermal |
| Z+chord | `bispf-hl-reduced-plus-chord-v1` | capture,floor,gauge,softplus,tether_thermal |
| Z+fg | `bispf-hl-reduced-plus-floor-gauge-v1` | capture,chord,softplus,tether_thermal |

## Preregistered decision rule

Verdicts are each task's Forge sustained gate read from `graded-result.json` (for `mode_hold`: at least 5
terminal consecutive passing checks of `modes>=8` and `hq>=0.9` among 24), never endpoint metrics.

Stage 1: `mode_hold` on all 13 arms.

1. Precondition: B PASSES and C FAILS on this source. If B fails, BISPF's lead is not reproduced and every
   attribution below is INCONCLUSIVE (results reported). If C passes, the lead is moot on this source
   (INCONCLUSIVE). T is expected to FAIL (round-1 reproduction; reported, not required).
2. Lead hypothesis (thermal/capture link): **SUPPORTED** if H0 or S0 FAILS. **REFUTED** if H0 and S0 both PASS.
   Context check: H0 and S0 should match B's per-check metrics exactly through update 1000 (code prediction).
3. A part is **necessary** (given everything else in BISPF) if its knockout arm FAILS.
4. Reduction check: Z should behave like T (FAIL). If Z PASSES, "BCAP + tether in the bispf formulation" already
   passes and the residual non-identities (no EMA critic copy, hook) are implicated; the knock-in reading is then
   void and reported as such.
5. A part is **sufficient on BCAP + tether** if Z FAILS and its knock-in arm Z+X PASSES. The **minimal set** is
   the passing knock-in with the fewest parts restored (ties in the order softplus, chord, floor+gauge, capture).
   If no single knock-in passes, the minimal set has >= 2 parts; if budget remains, pairs of the parts found
   necessary in (3) are tested (declared here, Z + both parts), otherwise reported as open.
6. Stage 2 (only for the minimal passing Z-family arm, and C; at most 2 candidate arms by the tie order):
   `gaussian1d_smoke`, `two_pole`, `ring16_acquisition`, `five_word_joint_smoke`, `five_word_joint_hold`.
   **PORTABLE** if the arm passes every stage-2 test that C passes; **PORTABLE WITH COST** naming each lost test
   otherwise. A stage-2 test C fails and the arm passes is reported as a gain.
7. INCOMPLETE/TIMEOUT/BLOCKED cells the rule needs make that cell INCONCLUSIVE; no reruns with other seeds. A
   wall-clock TIMEOUT of the 900 s word smoke on a shared GPU is reported as infrastructure, not a FAIL.

Known limitation stated up front: round 1 showed `mode_hold` reacts non-monotonically to small changes (C misses
by one late dip; T and B0 sit at 7 modes). With one seed, a knockout that FAILS shows the part matters on this
trajectory, not that it is the mechanism in general.

Budget: at most 1.5 GPU-hours of training. One job per GPU through the workflow lease helper.

---

## Results

(appended after the runs)
