No candidate qualifies for release. Executed 10 of the 16 allowed proposals, in batches of at most 3 new proposals. H remains unchanged. All runs used the pinned CPU/AVX2 environment, fixed seeds and one thread per worker; at most two benchmark workers ran together. First warm execution began at 17:29 UTC, within five minutes.

Ranked within the assigned update-dynamics family. A warm PASS means only 200 consecutive checks from H's borrowed checkpoint; it is not cold acquisition or own-state stability. Failure steps include the initial 1,200 H updates. Min HQ includes the failing observation.

| Rank | Candidate | Warm200 | First failure | Min HQ | Own cold ring (suffix ≥5) | Next cold gate |
|---:|---|---|---:|---:|---|---|
| 1 | [g_threequarter_rate](dynamics/batch-04/g_threequarter_rate/summary.json) | PASS 200/200 | — | 0.922363 | PASS (9) | FAIL: two_pole 0.02855 < .30 |
| 2 | [g_half_rate](dynamics/batch-02/g_half_rate/summary.json) | PASS 200/200 | — | 0.981445 | FAIL (3) | UNRUN |
| 3 | [network_half_rate](dynamics/batch-04/network_half_rate/summary.json) | PASS 200/200 | — | 0.963623 | FAIL (2) | UNRUN |
| 4 | [g_temporal025](dynamics/batch-02/g_temporal025/summary.json) | FAIL 54/200 | 1255 | 0.870605 | UNRUN | UNRUN |
| 5 | [paired_alt](dynamics/batch-01/paired_alt/summary.json) | FAIL 54/200 | 1255 | 0.788818 | UNRUN | UNRUN |
| 6 | [optimistic025](dynamics/batch-08/optimistic025/summary.json) | FAIL 48/200 | 1249 | 0.882324 | UNRUN | UNRUN |
| 7 | [paired_crn](dynamics/batch-06/paired_crn/summary.json) | FAIL 11/200 | 1212 | 0.755615 | UNRUN | UNRUN |
| 8 | [paired_sim](dynamics/batch-01/paired_sim/summary.json) | FAIL 0/200 | 1201 | 0.852051 | UNRUN | UNRUN |
| 9 | [independent_sim](dynamics/batch-01/independent_sim/summary.json) | FAIL 0/200 | 1201 | 0.808350 | UNRUN | UNRUN |
| 10 | [optimistic025_fast](dynamics/batch-08/optimistic025_fast/summary.json) | FAIL 0/200 | 1201 | 0.075439 | UNRUN | UNRUN |

g_threequarter_rate is the strongest ring-screen result, not a winner: G .001125, D .0015, prior .00225, warm minimum HQ .922363, cold terminal 8 modes/HQ .999756 with suffix9, followed by two_pole FAIL after its complete 80-update budget. g_half_rate and network_half_rate end cold at 8 modes/HQ1 but have only suffix3 and suffix2. Endpoints do not override those failures.

Every unrun gate remains unqualified. For all 10 proposals, own_state_hold_1200, grid100, rotated100 and staggered100 are SKIPPED. No native gate ran. For g_threequarter_rate the original-ten record is 1 PASS / 9 UNRUN, and the other nine older gates are two_pole FAIL / 8 UNRUN. For the two other warm survivors it is 1 FAIL / 18 UNRUN across older19. All seven warm failures have all older19 UNRUN. The unrun older names are trajectory, residual_student, img_stripes2, img_bars4, vector_overlap, img_blobs4, img_intensity2, vector_unequal_mass, vector_unequal_width, unipolar, mid_scale_identity, cover_leftover, vector_two_broad, vector_anisotropic, vector_spiral, unused_token_hold and ae_gan_hold; two_pole is also unrun except for g_threequarter_rate. Every candidate/gate status is explicit in the runtime tests.jsonl.

Pinned reference only, not new runs: H passes the original ten and has older19 = 13 PASS / 6 FAIL; native3 is unrun. Its independent own-state dense continuation fails first at1255, passes750/1200 checks, and ends at5 modes/HQ .20923. Retained initial probes critic_refresh2, average2 and extra_adam fail at1254,1228 and1203 respectively; they were not repeated.

Changes are isolated in [dynamics.py](dynamics.py), [dynamics_runner.py](dynamics_runner.py) and [run-dynamics.sh](run-dynamics.sh). Pairing shares real/prior batches; simultaneous variants preserve D-only gradients before G backward. paired_crn additionally reuses exactly matched input/output perturbations while retaining RNG state after D's draws. Temporal damping clips its correction toward the previous raw G gradient to .25 of current global gradient norm. Optimism clips current-minus-previous raw gradients to .25 of each player's norm. Histories are saved in update_state. Every proposal makes one D and one G Adam commit per update; none has predictors, extra fitting, targets/centers in updates, or evaluation feedback. H's logistic relativistic losses, R1+R2 .6, mixup .01, observation-noise policy and Adam(0,.999) remain fixed.

Only the three named rate variants and optimistic025_fast change fixed rates. network_half_rate uses G .00075 / D .0015 / prior .003; g_half_rate uses .00075 / .0015 / .0015; optimistic025_fast uses .02 / .005 / .01. All rates are constant during a run, including cold acquisition. The fast-rate optimistic probe fails on its first update, before history-based correction becomes available. Current cross-lane cold results motivated it but are not credited to this candidate.

Recommendation: preserve H and retain g_threequarter_rate only as a ring comparator. Reject simultaneous ordering, matched-noise coupling and these bounded temporal/optimistic corrections for this state. Avoid further rate interpolation: lower network rates help this warm state, but slower cold acquisition or two_pole movement rejects the tested policies. Any future proposal must first pass its relevant cheap gate and then acquire and hold its own state. Do not combine different candidates' partial successes.

The auxiliary blockers remain outside what these update changes can repair: with reconstruction disabled, the AE encoder has no GAN training path; with unused-token hold disabled, identical adversarial gradients cannot simultaneously train retention and movement. This attempt adds no architecture, conditioning, objective or supervised host loss. Restoring those supervised losses would be a separately labelled host-compatibility control, never a GAN-only qualification.

Work: 774 new warm updates + 3×1200 cold-ring updates + 80 original two_pole updates = 4454 first-run outer updates; validation reruns add 480 (including the 200-update archive-loader setup error), for 4934 outer updates and 9868 Adam commits. First-run measured gates: {'PASS': 4, 'FAIL': 10}. Warm gate compute 16.02s; cold gates 25.74s (excludes process startup and development). All backwards, commits, correction applications, noise forwards, optimizer tensors/rates and RNG states are retained.

Regression results: 63 distinct tests currently PASS, 1 CUDA skip. Four pytest invocations retained 99 passing case executions, two initial fixture failures and one skip; the fixture error was repaired once and rerun. No final viable candidate exists, so the full unit suite is explicitly SKIPPED. Logs/XML: dynamics/regression-01 through regression-04. The five new tests check unchanged H gradients/state, D-gradient isolation, actual matched noisy inputs/RNG, bounded damping and bounded optimism with raw history.

Evidence: [artifact-audit.json](dynamics/artifact-audit.json) records 14 successful checkpoint/sample regrades, constant rates and work checks. Each original warm directory contains actual-source.tar.gz, actual-source-manifest.json, effective-config.json, the actual three implementation sources, final-state.pt, metrics.jsonl and optimizer-receipt.json.gz. Cold batches03/05 contain their own source archives/manifests/configs and ring checkpoints. [capture_two_pole.py](capture_two_pole.py) adds only terminal-state capture: batch07 exactly reproduces every numeric observation and final spread/critic metric from batch05 while adding particles, critic, optimizer and RNG state.

Preserved setup errors: regression-02 reused mutable Adam tensors in a test fixture; fixed by deep-copying the fixture. Batch09 archive replay imported the current dynamics module because of inferred import paths; its numeric warm PASS is retained but it is not accepted as source-provenance validation. The actually imported source and ERROR receipt are saved there. [replay_dynamics.py](replay_dynamics.py) now explicitly loads the three archived modules. Batch10 replay matches all200 metrics/losses and the complete final model, Adam, sample, RNG and update-history state bitwise: [comparison](dynamics/batch-10/archived-replay/full-state-comparison.json). No setup error is scored as a bad GAN.

All125 pinned H training hashes, its original archive and its selected checkpoint passed the source audit. No original training file, frozen host, scoring rule, architecture, data budget or original H archive was modified. [attempt-summary.json](dynamics/attempt-summary.json) contains totals and paths. Runtime ledger: /ml2/hypergan/gan-attempts/stability-20260924T172714Z/stability_updates/20260924T172714Z-1520132/tests.jsonl.

From this checkout, exact archived warm replay (choose fresh output/ledger paths):

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES=
/tmp/pr38-default-env/bin/python -u reports/toy100/h_stability/replay_dynamics.py \
  --archive reports/toy100/h_stability/dynamics/batch-04/g_threequarter_rate \
  --output reports/toy100/h_stability/dynamics/replay-new \
  --ledger reports/toy100/h_stability/dynamics/replay-new.jsonl
```

Cold replay from initialization with complete two_pole state capture (same environment):

```bash
/tmp/pr38-default-env/bin/python -u reports/toy100/h_stability/capture_two_pole.py \
  --declaration reports/toy100/h_stability/dynamics/batch-07/cold-declaration.json \
  --output reports/toy100/h_stability/dynamics/replay-cold-new \
  --ledger reports/toy100/h_stability/dynamics/replay-cold-new.jsonl \
  --workers 1 --tasks mode_hold two_pole unipolar mid_scale_identity cover_leftover
```

Easy log tail:

```bash
tail -F /ml2/hypergan/gan-attempts/stability-20260924T172714Z/stability_updates/20260924T172714Z-1520132/tests.jsonl
# Detailed cold progress:
tail -F reports/toy100/h_stability/dynamics/batch-05/cold/g_threequarter_rate/run.log
```
