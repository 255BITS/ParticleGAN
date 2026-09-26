# Constant-rate stability: no qualified winner

Three proposals completed from PR195 `fa511ce010120b502f494d717d01b14b8551eed8`; this attempt is at its 3/3 review point. No merge or publication. The supervisor directed completion of the stationary window and then review, with longer and broader work gated off by its measured failure.

Artifact root: `/ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T214250Z-1447906/constant_rate_stability/20260926T214250Z-1447914/repo/experiments/constant_game`. Incremental history: `progress-history.md`. Executed/omitted gates: `/ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T214250Z-1447906/constant_rate_stability/20260926T214250Z-1447914/tests.jsonl`. Exact tested implementations are each run's `source.zip`; current package equals the C6 tested package. The proposed hook fix is deliberately **unapplied and untested**, not a fourth proposal or a source of quality claims.

| Candidate | Actual mechanism | Cold arrival / retention to2400 | Shift arrival delay / later retention | Decision |
|---|---|---|---|---|
| API-C6 | Secant implicit-response approximation, true .99 critic EMA, serial backward | 590 /182 of182 | 300 /191 of191 | Screen PASS; stationary7500 FAIL |
| API-C5 | Secant approximation with **unintended hard-copy reference**, historical backward | 560 /185 of185 | 270 /194 of194 | Good measured screen; declared .99 memory mismatch FAIL |
| API-C4 | Joint explicit same-oracle predictor/corrector, moving .99 reference | Not observed | Not observed | No arrival in either declared segment |

C4 non-arrival describes the finite4600 window, not impossibility or a new acquisition deadline. C5/C6 retain120/120 checks in the historical prehold window; C4 retains0/120. All frozen no-update controls pass0/220 after the change. At the prespecified3600 comparison endpoint: C5 has94/94 checks after shifted arrival, C6 has91/91; C4 has not arrived. No EMA observation supplies a live-weight pass.

**C6 stationary7500 rejects promotion:** arrival590,638/692 passing observations afterward, **54 failures** at3390–3680 and6780–7010 inclusive, every10 updates. Minimum HQ0 and minimum modes0. Final retrospective passing suffix7020–7500 is49 checks; it does not erase either collapse. Complete transitions, every departure and segment endpoints are in `C6-stationary/assessment.json`, `metrics.jsonl` and `leaderboard.json`.

| C6 observed segment | Minimum HQ since arrival | Later departures | Final stable suffix |
|---|---:|---|---|
| Original ring through2400 | .90087890625 | None | 590–2400,182 checks |
| Shifted ring through4600 | .927001953125 | None | 2700–4600,191 checks |
| Stationary through7500 | 0 |3390–3680;6780–7010|7020–7500,49 checks|

The unchanged C6 policy passes the frozen `img_intensity2` gate:6/24 observations, first pass450, final five passes500–600, final2 modes/HQ1.0. This is **1/22 broader tasks**, not full coverage or continuous image stability. The frozen residual-upsampling width16 host uses32 particles/z8/batch32/600 updates, original scoring and isolated noise seed402+update+1901. `image-host-copy.json` and `image-host.diff` pin the audited copied evaluator. Its package hashes equal the ring package; the actual runtime receipt supplement is explicitly post-run. No other lane's scores are counted.

The coupled-step failure is numerical, not an LR/noise phase change. At3380, applied G L2=.030574, D=.071790 and prior=.082554; at3390 these become1.942144,.882140,2.380322. Combined G/prior native preview L2 grows .626487→7.111960, while the secant norm factor opens .147830→.417378. The rule contracts relative to a proposal that itself becomes enormous; its single fitted plane does not ensure nonlinear game stability. Full first/second-excursion traces are in `C6-excursion-analysis.json`. No target changed during either collapse.

C5's bug is separately preserved: reference updating occurred inside the generator block while D.requires_grad was temporarily false. The inherited anchor then copied parameters. All9 D/EMA tensors match exactly in retained checkpoints. C6 moves finalization after original flags are restored and has distinct EMA weights. `reference-conformance-audit.json` verifies C4/C5/C6 checkpoints, and the focused regression checks the actual averaging equation. C5's source ZIP is `e06621731dca330decb5b40e251be382452b349ad690dee0ce931ef05fcbeff3`; its scores remain useful evidence of the actual one-step reference, never corrected retroactively.

The public opt-in implementation is `particlegan/game_update.py`, `training.py`, `recipes.py`, with copied memory/update plumbing in `ka2.py`, `k3p.py`, `update_limit.py`. It calls native PyTorch Adam twice as numerical stages, commits one accepted update, and preserves sparse prior sampling/A2 history. C6 stores moments from actual-state gradients and computes its implicit-response approximation from paired same-noise displacements. No target centers, quality feedback, resets or sample translation enter the learner.

All **21,900 quality-run public updates** verify constant G/D/prior rates .00425/.00425/.0085 and positive movement in all three roles. Per-update logs retain actual noise, controller and numerical correction state. Last500 stationary mean L2 is G0.016871, D0.072371, prior0.104356; sparse row ranges are in `update-audit.json`. The learner did not freeze.

Continuous mode ignores `total_steps` for LR/noise/stopping; the7500 run steps beyond its stored7000 default. Input noise initializes .5→0 over360 calls; output noise0→.029 over720, then both stay fixed. The799 initial penalty calls initialize the critic reference, after which .99 memory operates at arbitrary ages. These are fixed internal initialization/smoothing rules, never convergence estimates or caller/data-change phase switches. Finite tests cannot establish literal infinite stability.

Verification preserved:

- **PASS:** C6 identical2400 prefix under evaluator budgets4600/7500:24 full-state receipts and2400 complete applied update/controller/noise histories equal, including initialization, Adam/controller/EMA/RNG and real stream (`C6-horizon-prefix.json`).
- **PASS:** C6 genuine fresh-process1600→1800 continuation; all receipts equal, CUDA parameters/gradients/moments verified, ordinary CPU scalar Adam counters retained (`C6-checkpoint/continuation.json`). Shared serial-backward fix attributed and copied in `shared-serial-source.json`/`shared-serial-backward.patch`; numerical order changes were declared before C6 training. C4/C5 do not inherit this pass.
- **PASS:**68 initial regression tests;19 focused C6 reference/execution tests. **Final regression:97 PASS/1 FAIL** (`regression-final.log`). Copied KA2 optimizer plumbing calls both wrapped child and parent steps, invoking a registered post-hook twice. This is an additional unresolved public API blocker; `hook-regression.json` and `proposed-hook-repair.patch` make the proposed repair reviewable without changing tested sources.
- **PASS:**117 snapshotted source entries verified across five quality runs; unchanged shared evaluation declaration; initial ring models/EMA/RNG/data-stream hashes match retained public initialization. Source, initialization/checkpoint and artifact hashes are indexed in `source-verification.json`, per-run `artifacts.json`, `final-source.json` and `manifest.json`.

Gate ledger:14 executed/assessed entries =10 PASS,4 FAIL,0 ERROR;16 explicitly SKIPPED. This includes an externally reported declaration audit confirmed by the own saved-reference audit; counts are not qualification scores. Five quality executions across three proposals: two ring-screen passes, one ring-screen failure, one stationary failure and one image pass. No seed experiment, coefficient grid, nested agent, detached worker or borrowed pass.

**NOT_RUN:** C6 delayed/repeated9000 (changes after6000/7800), uninterrupted30000 (additional change after27000), remaining21 broader tasks, and matched ordinary public K3P. C4/C5 longer/broader gates are likewise omitted after failure/mismatch. Shared K3P ownership was transferred conditionally to this lane, then gated off by C6 retention failure. `evaluation-protocols.json` preserves all shared times/scoring; historical declarations are untouched. The canceled prospective C5/C6 declarations remain visible, with omissions recorded in the ledger.

Prior leads are preserved in `evidence-read.json` and `retained-diagnosis.json`: constant public KA2 retains61/120 prehold, arrives+120, then fails83 later checks; decayed KA2 retains120/120, arrives+1690, then48/52. Previous C1 memory improves mobility but departs; C2's late acquisition is not its rejection—the296/555 stationary result is. C3 never arrives. Historical scheduled/research-host and unrelated-runtime K3P scores remain context only.

Recommendation for reviewed refill: preserve C5's favorable **actual hard-copy** result as a lead. A successor could explicitly implement that one-step reference under the declared serial runtime, fix the public hook defect, and earn its own long retention; it cannot inherit C5 passes. If pursuing the implicit update, test local probes or a checked proximal residual that remains informative when native proposals grow, rather than another LR/coordinate-clipping grid. The measured single-plane contraction is insufficient.

Reproduce source preparation (does not launch training):
```bash
python experiments/constant_game/replay.py experiments/constant_game/C6-single experiments/constant_game/replay-C6
python experiments/constant_game/replay.py experiments/constant_game/C6-stationary experiments/constant_game/replay-C6-stationary
```
The helper rebuilds the base plus that run's exact ZIP and prints the pinned Python/GPU/thread command. Use the archived source for C4/C5. Tail completed logs with `tail -f /ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T214250Z-1447906/constant_rate_stability/20260926T214250Z-1447914/repo/experiments/constant_game/C6-stationary.log`; all raw logs, declarations, checkpoints, exact copies/diffs and failed variants remain in the artifact root.
