# dig2 harmonic-link: what gives BISPF its mode_hold pass?

Status: DONE. The rule was committed in `60e823d80` before any training; results are appended below it.

**Short answer.** The harmonic/thermal factor is not the link (refuted: switching it off leaves the run
bit-identical through update 1050 and still passing). On BCAP + SBNP tether, adding back only BISPF's 2%-weight
Sinkhorn transport-chord cap term gives the `mode_hold` pass and keeps gaussian1d, two_pole and ring16. The word
tests timed out in both arms. Inside BISPF the only necessary part is the softplus cap shape, and on BCAP a
different small part (floor + gauge) also works. So the "link" looks like trajectory sensitivity to small
critic-penalty changes once the tether is on, not one mechanism.

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

### Stage 1: `mode_hold` (13 arms)

All 13 attempts ran on frozen source digest
`1ad9c43959c19d8b18883da65917b0acdf4cd8b89d42492efad6730fe3662c23` (enqueued from branch commit `dfbec1b7a`; the rule
was committed earlier in `60e823d80`), seed 0, CUDA RTX A6000, `--workers-per-gpu 1`, two queue roots drained through
the workflow GPU lease (GPU 1).

"Terminal suffix" = consecutive passing checks at the end of training (PASS needs >= 5). "Passing checks" =
checks anywhere in the run with `modes >= 8` and `hq >= 0.9`.

| Arm | Candidate | Gate | Passing checks | Terminal suffix | Endpoint (not the verdict) | First 8-mode check | Runner s |
| --- | --- | --- | ---: | ---: | --- | --- | ---: |
| C BCAP control | `recipe-prior-refactor-bcap-pure-v1` | FAIL | 10/24 | 3 | modes 8, hq .989 | 650 | 37.3 |
| T BCAP + tether | `bcap-sbnp-tether-v1` | FAIL | 0/24 | 0 | modes 7, hq .996 | never | 39.0 |
| B full BISPF | `bispf-v1` | **PASS** | 13/24 | 12 | modes 8, hq 1.000 | 350 | 43.3 |
| H0 no capture | `bispf-hl-no-capture-v1` | **PASS** | 13/24 | 12 | modes 8, hq .989 | 350 | 43.4 |
| S0 no tether thermal | `bispf-hl-no-tether-thermal-v1` | **PASS** | 13/24 | 12 | modes 8, hq .996 | 350 | 43.4 |
| K0 no chord | `bispf-hl-no-chord-v1` | **PASS** (thin) | 5/24 | 5 | modes 8, hq .969 | 850 | 41.8 |
| P0 hinge cap | `bispf-hl-hinge-cap-v1` | FAIL | 7/24 | 4 | modes 8, hq .937 | 600 | 44.5 |
| F0 no floor/gauge | `bispf-hl-no-floor-gauge-v1` | **PASS** | 10/24 | 10 | modes 8, hq .982 | 650 | 41.9 |
| Z reduced (= T) | `bispf-hl-reduced-v1` | FAIL | 0/24 | 0 | modes 7, hq .996 | never | 39.9 |
| Z+capture | `bispf-hl-reduced-plus-capture-v1` | FAIL | 0/24 | 0 | modes 7, hq .996 | never | 40.6 |
| Z+softplus | `bispf-hl-reduced-plus-softplus-v1` | FAIL | 0/24 | 0 | modes 7, hq .749 | never | 40.2 |
| Z+chord | `bispf-hl-reduced-plus-chord-v1` | **PASS** | 12/24 | 11 | modes 8, hq .997 | 600 | 40.9 |
| Z+floor/gauge | `bispf-hl-reduced-plus-floor-gauge-v1` | **PASS** | 10/24 | 6 | modes 8, hq .991 | 650 | 39.2 |

Attempts (first 8 hex of the attempt id, under `queue-s1a|queue-s1b/dig2-harmonic-link-v1/`): C f9cb32e1,
T 1c76d4b1, B 2ac4ccac, H0 a302a47c, S0 f8908b4d, K0 177fee7d, P0 6e94be90, F0 c2c0d17f, Z b7d829f1,
Z+cap d484e0dc, Z+sp 73f63a80, Z+chord 0162bcf7, Z+fg 0ed6afc4. Full curves:
`/ml2/hypergan/dig2-20261016/harmonic-link/stage1-summary.txt`.

Capture log (B, from `PARTICLEGAN_BISPF_CAPTURE_LOG=1`): the capture test runs from call 1000; it first fires at
call 1075 (`r2f95=.0375 < .038`), drops out at 1079, re-captures at 1147 and stays captured; 58 of 200 eligible
calls are captured. In S0 it captures only 4 calls. In K0, P0, F0 it never fires. In Z+capture it never fires
(`r2f95 = .52`: the eighth mode is missing, so real points far from any fake keep the test open).

Stage-1 GPU seconds: 535.4 s runner wall time over 13 attempts (about 0.15 GPU-hours).

Rule outcomes:

- Precondition (rule 1): B PASS, C FAIL. Holds. T FAILS as in round 1.
- Lead hypothesis (rule 2): H0 and S0 both PASS -> **REFUTED**. The context check holds exactly: H0 and S0
  produce B's per-check metrics bit for bit through update 1050, and differ only in the last three checks
  (1100-1200), after capture first fires at call 1075. BISPF holds all 8 modes from update 350, long before the
  capture factor can act. The thermal/capture factor does not cause the pass.
- Necessary parts (rule 3): only the softplus cap shape. P0 (hinge cap) FAILS, with 7 passing checks and a
  4-check terminal suffix (hq dips to .841 at update 1000). K0 PASSES only just (suffix 5), F0 PASSES.
- Reduction check (rule 4): Z FAILS and is **bit-identical** to T, the same metrics at all 24 checks. So with all
  six parts off, the bispf formulation reproduces BCAP + tether exactly; the residual non-identities
  (no EMA critic copy, hook) have no effect here. The knock-in reading is valid.
- Sufficiency on BCAP + tether (rule 5): Z+chord PASSES (suffix 11) and Z+floor/gauge PASSES (suffix 6);
  Z+softplus FAILS (7 modes, hq .749) and Z+capture FAILS (identical to Z). The minimal set is **one part: the
  chord term** (Z+chord, 1 part restored; Z+floor/gauge needs 2).

What the minimal arm is, concretely: BCAP's `b_cap` hinge cap, with the fake-side weight reduced from 1 to 0.98 and
a 0.02-weight hinge cap added on Sinkhorn transport chords between each fake and a matched real (stratified,
no RNG draws), plus the SBNP tether at 0.01. Everything else is BCAP + tether.

Reading (not part of the rule): the part that is "necessary" inside BISPF (softplus) is not sufficient on BCAP,
and the two parts that are sufficient on BCAP (chord, floor+gauge) are not necessary inside BISPF. Each of them
changes the critic penalty by a few percent or less. That pattern says `mode_hold` on this host is decided by
small trajectory perturbations of the critic penalty once the tether is on, not by one identifiable mechanism.
BCAP + tether stalls at 7 modes; a 2% chord term, or a 0.05% floor + gauge, moves that trajectory onto one that
finds the eighth mode. One seed cannot separate "the chord term helps coverage" from "any small critic-penalty
perturbation re-rolls the trajectory".

### Stage 2: priority tests (C vs Z+chord)

Per rule 6, stage 2 ran on C and the minimal passing knock-in only (Z+chord; Z+floor/gauge restores two parts, so
it is not tied). Both arms ran on frozen source digest
`485c18ecc9cbf05c3d60fb137e81df42851c86011057035d4a95e1dad527260a` (enqueued from `21c67edf1`; it differs from
the stage-1 digest only by the coordinator-only `FORGE_SKIP_COMPLETION_COMPILE` switch and new study files).
Seed 0, CUDA RTX A6000, one GPU per arm through the lease, both GPUs shared with other jobs at ~100% use.

| Test | C BCAP control | Z+chord | Notes |
| --- | --- | --- | --- |
| `gaussian1d_smoke` | PASS (KS .072, std ratio 1.063; 36.8 s) | PASS (KS .104, std ratio .969; 31.0 s) | |
| `two_pole` | PASS (17/24, suffix 17; 7.7 s) | PASS (17/24, suffix 17; 7.9 s) | |
| `ring16_acquisition` | PASS (26/96, suffix 26; cov err .497, min eig ratio .242; 88.5 s) | PASS (32/96, suffix 31; cov err .538, min eig ratio .216; 62.9 s) | |
| `five_word_joint_smoke` | INCOMPLETE: wall-time budget (900 s) exhausted at 7,501 of 20,001 steps | INCOMPLETE: same, 7,501 of 20,001 steps | infrastructure (rule 7) |
| `five_word_joint_hold` | not run (depends on the smoke) | not run (depends on the smoke) | |

Word smoke detail (not a verdict): both arms passed every confirmation check from step 2,501 to the cut-off at
7,501 (5 words, quality fraction 1.0, mass TV <= .036). Z+chord learned the words later: mass TV .42 and .23 at
steps 834 and 1,667, against C's .04 and .03.

Rule outcome (rule 6): Z+chord passes every completed stage-2 test that C passes (3 of 3). The word smoke and word
hold cells are INCONCLUSIVE in both arms because of the shared-GPU wall-clock limit, so "portable" holds only for
the three completed tests.

Stage-2 GPU seconds: C 1,033.3 s, Z+chord 1,002.2 s (each includes the 900 s word smoke cut-off).

## GPU time

Stage 1 535.4 s + stage 2 2,035.5 s = **2,570.9 s, about 0.71 GPU-hours**, under the 1.5-hour limit. Most of
the waiting time (about 75 minutes before the first lease) was queueing for the shared lease, not training.

## Conclusion

1. **Lead hypothesis REFUTED.** BISPF's harmonic/thermal capture factor does not give the `mode_hold` pass.
   Turning it off entirely (H0), or only its scaling of the tether (S0), leaves the run bit-identical to full
   BISPF through update 1050 and still PASSING (12 terminal passes). In `mode_hold` the factor first acts at
   critic call 1075; BISPF already holds all 8 modes from update 350. The "residual-driven thermal noise"
   (`BISPFEngine`, `StratifiedBlueNoisePrior`) is not used by the Forge recipe path at all.
2. **Minimal set found, and it ports to BCAP:** BCAP + SBNP tether + BISPF's 2%-weight Sinkhorn
   transport-chord cap term (`bispf-hl-reduced-plus-chord-v1`) PASSES `mode_hold` (11 terminal passes), where
   BCAP + tether alone stalls at 7 of 8 modes. It keeps `gaussian1d_smoke`, `two_pole` and
   `ring16_acquisition`. The word tests are unresolved (wall-clock INCOMPLETE in both arms).
   The reduction is exact: with all six parts off, the bispf formulation reproduced `bcap-sbnp-tether-v1`
   bit for bit.
3. **But the attribution is fragile.** Inside BISPF, only the softplus cap shape is necessary (hinge cap
   FAILS by one check: suffix 4 of 5). On BCAP + tether, softplus is not sufficient, while the chord term alone and
   the floor + gauge pair alone are each sufficient. Each of these parts changes the critic penalty by a few
   percent or less. The data fit "with the tether on, small critic-penalty perturbations decide whether this one
   seed finds the eighth mode" better than "the chord term is the mechanism". A recommendation to adopt
   BCAP + tether + chord as a `mode_hold` repair needs evidence beyond this single seed, for example the other
   priority suites or a second host with the same failure mode. Under the user's one-seed rule, this dig cannot
   provide that.

## Exact commands

```sh
PY=/ml2/codex-storage/workspaces/2026-10-07/task-14/runtime/python312/bin/python
R=/ml2/hypergan/dig2-20261016/harmonic-link
# from the worktree root, branch task/dig2-harmonic-link-20261016
OMP_NUM_THREADS=1 $PY -m pytest -q tests/test_bispf_ablation.py tests/test_bispf.py
OMP_NUM_THREADS=1 $PY -m experiments.forge validate
# stage 1 enqueue (13 arms, two queue roots): $R/enqueue-s1.sh, i.e. for each candidate c
OMP_NUM_THREADS=1 $PY -m experiments.forge --queue-root $R/queue-s1a enqueue $c --study dig2-harmonic-link-$c-mode-hold-v1
# drain each queue root under the workflow lease: $R/drain.sh s1a|s1b, i.e.
gpu-lease.sh bash -c "FORGE_SKIP_COMPLETION_COMPILE=1 PARTICLEGAN_BISPF_CAPTURE_LOG=1 OMP_NUM_THREADS=1 \
  $PY -m experiments.forge --queue-root $R/queue-s1a drain --gpus \$LEASED_GPU --workers-per-gpu 1 --campaign dig2-harmonic-link-v1"
# stage 2
OMP_NUM_THREADS=1 $PY -m experiments.forge --queue-root $R/queue-s2c enqueue recipe-prior-refactor-bcap-pure-v1 --study dig2-harmonic-link-recipe-prior-refactor-bcap-pure-v1-priority-v1
OMP_NUM_THREADS=1 $PY -m experiments.forge --queue-root $R/queue-s2z enqueue bispf-hl-reduced-plus-chord-v1 --study dig2-harmonic-link-bispf-hl-reduced-plus-chord-v1-priority-v1
$R/drain.sh s2c; $R/drain.sh s2z
# summaries (tail-able per-check metrics are in each attempt's run.log and $R/queue-*/events.jsonl)
$PY $R/summarize.py $R/queue-s1a $R/queue-s1b      # -> $R/stage1-summary.txt
```

## Caveats

- Seed 0 only, one attempt per arm, by design. `mode_hold` margins are thin (C misses by one check, P0 by one
  check, K0 passes with exactly the minimum 5).
- Stage 1 and stage 2 used different frozen digests. Within each stage, every arm shares one digest, so the
  matched comparisons hold. The difference is coordinator-only code plus study files.
- The two word cells are INCOMPLETE because the 900 s task wall-clock budget ran out at 7,501 of 20,001 steps on
  GPUs shared at ~100% use. No rerun was made (rule 7). Whether Z+chord keeps `five_word_joint_hold` is open.
  Z+chord trained differently from C on the word smoke (slower early word acquisition). I did not trace which
  difference causes this. The bispf kernel takes a separate branch when an input has more than 64 dimensions.
- The capture log is diagnostic stderr output gated by `PARTICLEGAN_BISPF_CAPTURE_LOG=1`. It was set for every
  arm, does not change training (B reproduced round-1's 12 terminal passes and 13/24 passing checks exactly) and
  is not part of the frozen source.
- `tests/test_forge_current_technique_inventory.py` fails in this checkout (107 tests, e.g. `FileNotFoundError`
  for `reports/forge/test-performance-evidence/index.json` in a temp fixture). It does not touch the code changed
  here; it was not re-run on a clean develop checkout. `tests/test_bispf.py`, the new
  `tests/test_bispf_ablation.py`, `test_forge_boundaries.py`, `test_forge_techniques.py`,
  `test_forge_technique_bindings.py` and `test_forge_technique_inventory.py` pass (80 tests).
- The new field is opt-in and research-only. With the default `"none"`, BISPF's penalty values and gradients
  were checked bit-identical to the previous code over 1,105 calls, including the capture branch.
