# Precision wave 4: RP10–RP12

None qualifies as the release/default winner. All six started cheap windows completed, and all three proposals failed both unchanged gates. The exact released K3P reference also failed the tiny gate; that comparison is preserved without lowering the candidate gate or inheriting an earlier candidate's successes.

| Candidate | Exact change | Tiny mode_hold,1200 updates | Intensity,600 updates | Controller events | Quality runtime |
|---|---|---|---|---|---|
| RP10 | Original RP5 two-field secant/precision; stationary input noise0 and output noise0.029 from first update |0/24; maximum/final7 modes; final HQ1.0; suffix0|4/24; first450; passes450/525/550/600; misses475/500/575 after arrival; suffix1|Tiny closes1086; image always open|133.225452s|
| RP11 | RP10 plus all real/fake logit pairings, averaging the within-batch row-permutation loss for both players |0/24; maximum/final6; final HQ0.990234; suffix0|4/24; first450; same passing/missing steps as RP10; suffix1|Both always open|157.604609s|
| RP12 | RP11 plus inverse occurrence weights in both D/G adversarial losses: each sampled distinct prior row has equal mass |0/24; maximum/final7; final HQ1.0; suffix0|4/24; first525; passes525/550/575/600; no later miss; suffix4|Tiny closes1147; image always open|136.120139s|
| Released K3P v0.8.0 | Unpatched released12-file package; declared1200-update schedule, input zero120/output full240; native lazy CPU Adam clocks |0/24; maximum/final6; final HQ0.976074; suffix0|Not run in this reference|Released finite schedule|19.323610s|

RP12's four final good image checks are retained as a stable observed suffix of four. They still fail the predeclared five-check gate. No endpoint was moved and no extra run was added to turn that into a pass. Tiny coverage failures occur while open/full rate too: RP11 never reduces either task's rates, and RP12 encounters all12 distinct prior IDs in every retained tiny D/G batch. These observations do not prove a causal explanation, but they rule out treating early LR reduction or absence of latent-row sampling as a sufficient explanation for these traces.

All candidates use the same public API path: total_steps=None; stationary noise; original two-field secant arithmetic; RP5 internal gradient-reference gap/update-activity precision; native eager parameter-device Adam state; KA2/A2; full-step restoring serial backward context. There are no target labels, evaluator horizons, shift steps or quality thresholds inside the learner. The fixed internal KA2 initialization is unchanged. RP11's public GANLoss/Recipe pairing change is state-free; RP12 adds public Recipe.particle_weights and fake_weights loss arguments, and GANTrainer supplies its own sampled row IDs. RP12 leaves penalty/prior-regularization sampling unchanged and adds no stream or learned checkpoint state. Its weighted estimator affects shared G/D losses, distinct from earlier latent-gradient-only exposure/cap work. Existing failed residual/particle-noise variants remain failed evidence.

All15 public files match both run source ZIPs for each candidate. Exact package ZIP seals:

- RP10: `429604ffc7e8850824c0e4127c34376f7e8dc094b93eabcd53b89eca1c2525b2`
- RP11: `41c381829294f3019316f4fdf7edd0cfcf7e975d7c20b4daa20fceb0ed0d8a31`
- RP12: `d4fe5ac65a738cbd6e10168f4e4c970f653a5dd7dc1ff29102a5e1cad63aa63a`

Independent source/receipt auditing verified all5,400 candidate accepted updates,10,800 field evaluations and144 quality observations; all actual rates/noise/controller clocks; all3,600 tiny caller-data/index/cursor receipts; and immutable model/scorer/threshold bindings. Restricted standard-library raw-storage comparison additionally verified every initial model, optimizer, precision and private/global/caller RNG state against retained RP5 initial fixtures, plus all final native accepted clocks and final precision states. It uses no Torch import or tensor execution. This later raw-state audit strengthens earlier JSON-only RP10/RP11 receipt limits. RP12's per-player weight diagnostics satisfy count/bound/reciprocal-multiplicity consistency; actual CUDA index tensors were not regenerated. Tiny canonical prior-first CUDA construction and image CPU G/D/prior construction remain separate fixture authorities. Historical image prior_weight0.05 versus public candidate prior_reg0 is retained as an explicit learner-regularization difference.

| Original regression receipt | Tests | Failures | Errors/skips | Interpretation |
|---|---:|---:|---:|---|
| rp10-regression |93|6|0/0|New helper test incorrectly combined finite horizon with continuous precision; validator correctly rejected it. Original failure retained.|
| rp10-regression-corrected |93|0|0/0|Fixture corrected; no relaxation of continuous-precision horizon rule.|
| rp11-regression |96|0|0/0|Adds all-pairs loss/gradient permutation identity and public continuation contracts.|
| rp12-regression |100|0|0/0|Adds inverse-count mass, duplicate invariance including gradients, and public continuation/clocks.|
| final_artifact_source_initial_stream_audit |7 retained runs|0|not a pytest suite|Owner final artifact/source/initial-stream audit PASS; independently checked by supervisor raw-storage/receipt audit.|

These are the complete retained receipts from the owner's selected five-module regression command: training, recipe defaults, KA2, serial backward and precision game tests. They are not represented as a repository-wide suite. Full original logs/XML, command source, final audit JSON and all three owner-retained test-source ZIPs are copied under `supervisor-audit/precision-wave4-final-receipts/`, with original paths/bytes/hashes in provenance.json. Test-source archives were outside the learner/run seals; their timing is not retroactively certified. The supervisor reran no training or model tests.

Qualification **NOT_RUN for each RP10–RP12** after cheap-gate rejection: own public20k/z2/batch2048 shifted ring and frozen control, stationary7500, delayed/repeated9000, uninterrupted30000 with fixed6000/7800/27000 changes, remaining22-task coverage, different evaluator budgets with identical trained prefixes, and own cross-process CUDA continuation. Neither an older ring success nor CPU algebra establishes these. No observed reopening or winner is claimed. The all-pairs default-batch2048 cost remains unmeasured; its logit work/memory is quadratic. Public full-transaction binding remains limited to scalar unconditional particle-prior GANTrainer; standalone rate helpers still expect finite horizons, and conditional/auxiliary/multiple-optimizer hosts lack earned full-transaction portability.

The next direction should address missing support during acquisition in the shared generator, with a concrete causal hypothesis derived from retained states/fields and a mechanism outside the exhausted classes. Full-rate RP11 still fails coverage, so another close/reopen threshold is not justified by this wave alone. Do not repeat constant-noise amplitudes, all-pairs or multiplicity-loss variants, richer secant/residual solves, particle-disagreement noise, or learned latent-width smoothing. The separate data lane is already investigating learned bounded rank-one latent correlation from paired shape gradients; avoid duplicating it and inherit none of its reported passes. Keep the same cheap mode_hold/image gates first; a new mechanism must earn them before expensive recovery/long qualification. This review proposes no implementation or new experiment.

All original scored data are preserved in `evidence/api-rp10-*`, `api-rp11-*`, `api-rp12-*` and `public-k3p-mode_hold`. Candidate quality windows total426.950200s; supporting K3P adds19.323610s, for446.273810s of reported quality-run wall time (excluding regressions/preparation). Ready new broader entries are in rp11-completed-audit.json and rp12-completed-audit.json. No shared manifest or active learner was edited, no GPU/model/training work was launched, and nothing was merged or promoted.

## Root-reviewed next attempt

The predecessor driver1735840 has exited0; its report explicitly rejects all
three proposals. Continue the authorized search with one external gpt-6-astra/max
session, one GPU worker, and at most three distinct new mechanisms API-RP13 onward.
Finish every declared window, preserve all observations and original failures,
and report after the cap. No seed experiments, coefficient grids, extra child
models, duplicate K3P references, default promotion or merge.

Investigate a distinct shared-generator/support-acquisition mechanism grounded
in these failures. The other lane's DV16 now passes its own tiny coverage test
(11/24, final11 from700, independently verified) and reports intensity8/24/final6
and unequal_mass15/24/final15 passes. Its bounded rank-one latent correlation is
already under qualification; do not duplicate it or inherit its scores. The
constant lane C13-R1 has own changed-target arrival+130 and208/208 retention,
but expensive nonlinear correction and broader qualification remain unresolved.
Keep cheap tiny/image gates before expensive new ring/long tests.

COMMON.md and evaluation-protocols.json remain authoritative. Learners receive
no horizon, target-change information, evaluator quality/labels, task identity
or caller phase. Automatic reversible rates remain allowed; every new package
must earn complete public API, long retention/repeated recovery, checkpoint/
budget independence, broader22 and matched-reference evidence before selection.
Eventual integration targets develop first, still without merging now.
