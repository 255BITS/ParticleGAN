# BISPF ablation: is BISPF's lead over BCAP just the prior tether?

Status: DONE. Preregistered verdict: **REFUTED**. The rule was committed (dccb6839) before any training;
results are below the rule.

**Short answer.** No. Putting only BISPF's prior tether onto BCAP (arm T) makes `mode_hold` worse, not
better: T never holds all 8 modes (0 of 24 checks pass), while the unchanged BCAP control gets 10 passing
checks and misses the gate by one late quality dip. An exploratory reverse ablation, BISPF with the tether
switched off (B0), fails the same way (7 of 8 modes throughout). Only full BISPF passes. On this test,
BISPF's lead needs the tether **and** BISPF's critic-side parts together; neither alone gives it.
The review's §1.3 attribution ("the tether plus the exemption, not the Born-Infeld/Sinkhorn/harmonic
parts") does not hold for `mode_hold` on the current source.

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

## Results

All four attempts ran on frozen source digest
`5ea30969e082e2336f02ca1a2965a70346e86b51e7d5b508842c1f07beb24182` (branch commit dccb6839 for C/T/B,
3927ae822 for B0, which adds only config files and has the same digest), seed 0, CUDA RTX A6000,
`--workers-per-gpu 1`, each through the workflow GPU lease.

`mode_hold` gate: 24 checks over 1,200 updates; a check passes when `modes >= 8` (all 8 target modes have
high-quality samples) and `hq >= 0.9` (fraction of samples near some mode); PASS needs the last 5 or more
checks to pass in a row ("terminal suffix").

| Arm | Candidate | Request / attempt | Verdict | Passing checks | Terminal suffix | Endpoint (step 1200) | Wall s |
| --- | --- | --- | --- | ---: | ---: | --- | ---: |
| C control | `recipe-prior-refactor-bcap-pure-v1` | 428622712aa6872b49893842 / 30f5d8c4 | FAIL | 10/24 | 3 (needs 5) | modes 8, hq 0.989 | 54.4 |
| T tether only | `bcap-sbnp-tether-v1` | 558f44ba737580b22a818ad3 / f0d19e50 | **FAIL** | 0/24 | 0 | modes 7, hq 0.996 | 61.6 |
| B full BISPF | `bispf-v1` | 53957fbb49f439f8c85b7471 / 513b04a6 | PASS | 13/24 | 12 | modes 8, hq 1.000 | 54.2 |
| B0 BISPF, tether off (exploratory) | `bispf-no-prior-tether-v1` | 56c366ad6e0cb88a292dd00d / 82a11e63 | FAIL | 0/24 | 0 | modes 7, hq 0.996 | 45.3 |

Curves (`step modes hq`, every 50 updates, abbreviated):

- C: reaches 8 modes at step 650; hq 0.91-0.94 from 700 to 1000, **dips to 0.869 at 1050**, then 0.939,
  0.933, 0.989. The dip resets the suffix to 3. C's endpoint would pass; its sustained gate does not.
- T: 4 -> 6 modes by 200, 7 modes from 650 onward, never 8. hq climbs to 0.996 on the 7 modes it holds.
- B: 8 modes from step 350, hq >= 0.95 from 650 onward (one 0.954 at 800); 12 terminal passes, matching the
  recorded BISPF result (12).
- B0: 7 modes from step 400 onward, never 8; hq 0.90-0.996.

Rule outcome: precondition holds (C FAIL, B PASS on this source). T FAILS `mode_hold` -> rule 2:
**REFUTED**. Stage 2 (`gaussian1d_smoke`, `two_pole`, `ring16_acquisition`, `five_word_joint_smoke`,
`five_word_joint_hold`) was not run, as the rule specified; its queued requests were cancelled.

GPU seconds: 54.4 + 61.6 + 54.2 + 45.3 = **215.5 s** of attempt wall time (about 0.06 GPU-hours of the
2-hour allowance).

## Interpretation

- The tether alone, at BISPF's strength and without BISPF's thermal scaling, stops BCAP from finding the
  8th mode. A plausible mechanism (not tested here): pinning the 12-row prior table's mean and covariance
  to their initial values limits how far rows can move to cover a missed mode, and BCAP's critic does not
  push hard enough to overcome it.
- BISPF's critic without the tether also stalls at 7 modes. So in BISPF the two parts interact: one
  candidate is the harmonic thermal factor, which scales the tether's gradient down once modes are
  captured (`particlegan/bispf.py` around L203-206, active for `d <= 4`; `mode_hold` has d=4). That coupling
  cannot exist in T. Testing it would need a new opt-in flag that applies a capture-dependent scale to
  the tether under the bcap critic.
- The BCAP control's failure is thin: one check at hq 0.869 against 0.9, 150 updates before the end.
  "BISPF beats BCAP on mode_hold" is real on this source (12 vs 3 terminal passes), but the control is
  close to the line, so a small perturbation of either arm can move the verdict.
- The earlier #390 result (BCAP + `moment_tether` regularizer at 0.01 passing `mode_hold`) used a different
  regularizer (`moment_tether`, no repulsion term, no exemption) on an older source. It is not contradicted
  or confirmed by this run.

## Exact commands

Python: `/ml2/codex-storage/workspaces/2026-10-07/task-14/runtime/python312/bin/python` (as `$PY`),
`OMP_NUM_THREADS=1`, run from the worktree root. `LEASE=.../scratchpad/dig/gpu-lease.sh`.

```sh
$PY -m experiments.forge validate
# enqueue (no training); C was enqueued into its own queue root so it could run in parallel
$PY -m experiments.forge --queue-root /ml2/hypergan/dig-20261015/bispf-ablation/queue   enqueue bcap-sbnp-tether-v1 --study dig-bispf-ablation-tether-mode-hold-v1
$PY -m experiments.forge --queue-root /ml2/hypergan/dig-20261015/bispf-ablation/queue   enqueue bispf-v1 --study dig-bispf-ablation-bispf-mode-hold-v1
$PY -m experiments.forge --queue-root /ml2/hypergan/dig-20261015/bispf-ablation/queue-C enqueue recipe-prior-refactor-bcap-pure-v1 --study dig-bispf-ablation-control-mode-hold-v1
$PY -m experiments.forge --queue-root /ml2/hypergan/dig-20261015/bispf-ablation/queue-B0 enqueue bispf-no-prior-tether-v1 --study dig-bispf-ablation-notether-mode-hold-v1
# drain one request per lease
$LEASE bash -c '$PY -m experiments.forge --queue-root <root> drain --gpus $LEASED_GPU --workers-per-gpu 1 --campaign <campaign> --request <request_id>'
```

Evidence (graded-result.json, run.log with live per-check metrics, request/result receipts):
`/ml2/hypergan/dig-20261015/bispf-ablation/{queue,queue-C,queue-B0}/<campaign>/<attempt>/`.
Drain logs: `/ml2/hypergan/dig-20261015/bispf-ablation/drain-*.log`.

## Caveats

- Seed 0 only, one attempt per arm, by design. C's margin is one check.
- Every drain exited with a traceback **after** training and grading finished, in the checkout's
  post-run report compile (`experiments/forge/knowledge.py` `compile_memory`: `NameError: name 'deepcopy'
  is not defined` in `cached_load_tasks`; after adding the import locally, a second error,
  `TypeError: _attempts() takes 1 positional argument but 2 were given` in `cached_attempts`). This is a
  develop-branch bug in the caching wrapper, not in training. Training, independent grading and receipts
  completed; the verdicts above are read from each attempt's `graded-result.json`. The local import fix was
  reverted; it is not part of this branch. The queue's request lifecycle and the compiled boards were not
  updated by these runs.
- T differs from BISPF's tether in one known way: no harmonic thermal scaling (see Interpretation).
- Only `mode_hold` was run. Whether the tether costs other tests (e.g. word hold, as #390's tether did) was
  not measured because the rule was decided at stage 1.
- New files are opt-in only: two idea cards, two research-diagnostic views (all tasks `diagnostic`,
  `evidence_scope: research_diagnostic`), two campaigns, seven studies. No default, task, gate or budget
  changed.
