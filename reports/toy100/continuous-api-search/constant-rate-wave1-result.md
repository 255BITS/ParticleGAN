# Constant-rate stability: no qualified winner

Three coherent proposals completed from PR195 `fa511ce010120b502f494d717d01b14b8551eed8`. This attempt is at its 3/3 review point. No merge or publication. Experimental switches are opt-in; the released default remains unchanged. Supervisor assigned further cross-process checkpoint investigation to `data_drift_mobility` and shared K3P comparison to `reversible_precision`.

Artifact root: `/ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T211209Z-1398466/constant_rate_stability/20260926T211209Z-1398479/repo/experiments/constant_memory`. Tests: `/ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T211209Z-1398466/constant_rate_stability/20260926T211209Z-1398479/tests.jsonl`. Earlier incremental notes: `repo/experiments/constant_memory/progress-history.md`.

| Candidate | Mechanism | Cold arrival | Retention from cold arrival to 2400 | Historical prehold window | Shift arrival delay | Passing since shifted arrival | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| API-C1 | Continuous critic reference, EMA .99; always-on anchor | 580 | 172/183 | 109/120 | 570 | 164/164 | FAIL: stationary departures |
| API-C2 | C1 plus coordinate Adam displacement bound at nominal LR | 1960 | 45/45 | 45/120 | 1890 | 32/32 | Initial screen PASS; 7500 stationary FAIL |
| API-C3 | C2 plus bounded optimistic network proposals; sparse prior unchanged | Not observed | No arrival | 0/120 | Not observed | No arrival | FAIL: acquisition and adaptation not observed |

Prehold is the inherited 1210–2400 reporting window, not a new acquisition deadline. C2's 75 prehold failures precede its first acquisition. No fixed 81/81 or 400-update recovery gate was used. Non-arrival is limited to the declared finite window. At the declared3600 comparison endpoint, C1 has64/64 checks after shifted arrival (HQ .9921875); C2/C3 have not arrived (HQ .488037109375/.110595703125). The frozen controls pass **0/220** post-change checks for each candidate. Live weights are primary; EMA never supplies a pass.

| Own-state window | Minimum HQ since arrival / minimum modes | Every later failing observation (inclusive intervals, spacing 10) | Final retrospective stable suffix |
|---|---|---|---|
| C1 cold through 2400 | .001220703125 / 1 | 1220–1290, 1310–1330 | 1340–2400, 107 checks |
| C1 shifted through 4600 | .931884765625 / 8 | None | 2970–4600, 164 checks |
| C2 cold through 2400 | .986083984375 / 8 | None | 1960–2400, 45 checks |
| C2 shifted through 4600 | .987548828125 / 8 | None | 4290–4600, 32 checks |
| C2 stationary through 7500 | .034423828125 / 5 | 2940–4290, 5630–6800, 6820–6850, 6870 | 6880–7500, 63 checks |

C2's longer stationary result is **296/555** passing observations after arrival, with **259 departures**. C3 never passes; final live HQ is .12451171875, EMA HQ .710693359375. All individual observations, failures and the 3600 comparison endpoint are in each `assessment.json`/`metrics.jsonl`; complete transition summaries are also in `leaderboard.json`.

The retained public constant-KA2 lead remains 61/120 prehold, shifted arrival delay 120, then 126/209 with 83 failures. Retained decayed KA2 remains 120/120, delay 1690, then 48/52, and is ineligible as a scheduled learner. These were read, preserved and integrity-verified, never rerun or counted as descendant passes. Historical K3P on another runtime is not a matched comparison.

The retained collapse at1750 has HQ .0029296875, 2 modes, surprise ratio .84193396 and 402 skipped EMA updates; at1800 the ratio is12.4599 and anchor weight0. Only initial/2400/4600 full checkpoints are retained externally, so exact1750 Adam update tensors were unavailable. `retained-diagnosis.json` pins this evidence. New C1 telemetry finds G update L2 .064035 at1214 ->1.650854 at1218 and D L2 1.824776 at1217. C2 bounds individual coordinates but still permits collective growth: G L2 .037191 at2930 ->.344486 at2950, while HQ falls below .90 at2940. C3's previous-proposal correction substantially worsens acquisition.

The experimental API lives in `particlegan/recipes.py`, `training.py`, `ka2.py`, `k3p.py`, and new `update_limit.py`. `get_recipe(continuous=True, critic_memory="moving", bounded_updates=..., optimistic_updates=...)` selects the prototypes, and the worker calls actual `GANTrainer.step()`. PyTorch non-fused Adam, adversarial losses, critic architecture and A2 sparse prior damping remain. New policies add finite critic memory and optionally alter applied parameter displacements; no alternative Adam implementation, target fitter, sample translation, reset or quality feedback.

Every one of **21,300 quality-run updates** verifies the same G/D/prior rates: **.00425/.00425/.0085**, respectively. Each update logs noise, full controller state and applied movement. All three roles move on every update; sparse prior changed-row ranges and late-window movement means are in `update-audit.json`. C2 stationary's last500 mean L2 movements are G .024651, D .112601, prior .063062; it did not freeze after learning.

Continuous mode ignores the recipe horizon for rates, noise and stopping; the 7500 run steps beyond the stored default total_steps7000. Startup noise is explicitly absolute: input .5 to0 over360 updates, output0 to.029 over720, then fixed forever. The inherited799 pure-A calls initialize a critic reference; .99 EMA and subsequent update rules operate unchanged at arbitrary ages. These are internal cold-start/smoothing counts, not convergence estimates or caller phase switches; a target change never restarts them. This structural independence does not rescue the measured quality failures.

Public fixture preserved: seed0 only, 20,000 particles, latent2, batch2048, width96×3, critic Fourier3, CPU network initialization then CUDA FP32 training, deterministic algorithms, TF32 off, isolated4096-sample observations every10 updates, unchanged 8 modes and HQ>=.90. All candidate initial model/EMA, training/real-stream, CPU RNG and CUDA RNG hashes match retained public baseline. Applied optimizer moment/update tensors and gradients were verified CUDA; normal CPU scalar Adam step metadata was preserved. CUDA runtime is PyTorch2.13.0+cu126 on RTX A6000.

Verification:
- **PASS**: identical2400-update prefix under evaluator budgets4600 and7500; 24/24 full-state receipts, models/optimizer/controller/EMA/RNG/data-stream state equal. Two audit executions retained.
- **FAIL twice**: exact CUDA checkpoint continuation1600->1800, including replay with retained observation/movement instrumentation. Immediate loaded state and first resumed update norms/clip counts match; final RNG hashes match but model/optimizer hashes diverge. `C2-audit/` and `C2-audit-observed/` retain both results and exact audit sources verified against their pre-run declarations. Unresolved; investigation now belongs to the supervisor's designated lane.
- **PASS**: final78 regression tests (continuous schedule/beyond-budget API, both update modes' small-model checkpoint continuation, sparse bounds, training, KA2 parity and K3P). Initial regression58/59 is preserved: the one failure was this attempt's overly specific exception-message assertion, corrected to the public wrapper message.
- **PASS**: retained evidence verification (14 hashes and all archived rates/summaries).
- Executed gate totals: 11 = 5 PASS, 6 FAIL, 0 ERROR; 12 explicitly SKIPPED gates. These counts include API/regression/integrity audits, not just quality tests. Four quality executions across three proposals: one initial screen PASS, three FAIL; no qualified candidate.

Required remaining qualification is **NOT_RUN**: C1/C3 stationary7500; all candidates delayed/repeated9000 (changes after6000 and7800), uninterrupted30000 continuation (additional change after27000), and own22-task quality suite. C2 failed stationary before those gates. Local matched K3P is **NOT_RUN** under supervisor coordination. Shared declaration is `evaluation-protocols.json`, unchanged from the supplied file; historical declarations remain untouched. No seed sweep, coefficient grid, nested agent or detached worker.

Recommendation for the next reviewed attempt: retain C1's finite-memory result as a useful lead, not a pass. Coordinate-wise limiting and previous-minibatch optimistic extrapolation have measured failures here. Investigate a coupled predictor/corrector GAN update with a fresh gradient at the tentative state, using the same constant rates and public Adam semantics, to address collective game motion rather than lowering a fixed LR or sweeping bounds. Resolve the shared CUDA replay failure before promoting any policy. Preserve C2's slow but initially stable adaptation as evidence, and require its successor's own long retention measurements.

Replay preparation (does not launch an unchanged failed candidate):
```bash
python experiments/constant_memory/replay.py experiments/constant_memory/C1-single experiments/constant_memory/replay-C1
python experiments/constant_memory/replay.py experiments/constant_memory/C2-single experiments/constant_memory/replay-C2
python experiments/constant_memory/replay.py experiments/constant_memory/C2-stationary experiments/constant_memory/replay-C2-stationary
python experiments/constant_memory/replay.py experiments/constant_memory/C3-single experiments/constant_memory/replay-C3
```
Each command reconstructs the supplied base plus that run's exact `source.zip` and prints the pinned Python/GPU/thread/determinism command. Use the snapshot, not the current C3 source, to replay predecessors. Tail logs with `tail -f experiments/constant_memory/C1.log`, `C2.log`, `C2-stationary.log`, or `C3.log`. Source patches, declarations, initialization receipts, full checkpoints, metrics, per-update histories and SHA256 manifests are retained under the artifact root. Final source and all evidence are indexed by `manifest.json`.
