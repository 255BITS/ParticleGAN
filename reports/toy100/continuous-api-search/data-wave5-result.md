# Three proposals complete; no qualified winner

**Review point reached.** DV13–DV15 each completed their own public-API ring, unequal-mass and bars4 windows. Each has a measured broader-quality failure. No fourth proposal, merge, publication or default promotion. Integration target: develop. All workers exited.

Checkout: `/ml2/hypergan/gan-attempts/continuous-api-20260926/20260927T002520Z-1700486/data_drift_mobility/20260927T002520Z-1700495/repo`. Base: PR195 `fa511ce010120b502f494d717d01b14b8551eed8`. Artifact paths below are relative to `repo/reports/data-drift-api/`; ledger: `tests.jsonl` beside this report. Retained DI1–3/DV1–12, eligibility, source audits and shared protocols were inspected. No inherited score counts toward these candidates.

|Own candidate|Original arrival; retention to2400|Shift delay; retention to4600|Final shifted suffix; minHQ / modes|Ring seconds|
|---|---|---|---|---|
|DV13|610; 180/180|+350; 181/186|175 from2860; 0.711670 / 6|66.712|
|DV14|590; 182/182|+290; 191/192|188 from2730; 0.851074 / 8|78.127|
|DV15|520; 189/189|+330; 185/188|182 from2790; 0.578125 / 7|73.706|

All three preholds pass120/120 and frozen no-update controls pass0/220. No departures after original arrival. Every shifted departure: DV13=2770,2780,2790,2810,2850; DV14=2720; DV15=2740,2750,2780. Initial minimumHQ: .905518/.905273/.929199. These are early transition misses followed by finite retention; no first-touch-perfect or fixed recovery deadline gate was applied. Full travel minima, every observation and the declared3600 comparison are preserved in each `runs/dv*-single/summary.json` and `metrics.jsonl`.

|Own broader gate|Passing/24; final suffix|Final relevant metrics|Verdict|
|---|---|---|---|
|DV13 unequal|14; 10|covariance error0.411232; eigenratio0.801838; mass ratio0.683594|PASS|
|DV13 bars|10; 2|4 modes; HQ0.90625|FAIL|
|DV14 unequal|14; 14|covariance error0.275462; eigenratio0.795536; mass ratio0.329590|PASS|
|DV14 bars|8; 0|4 modes; HQ0.84375|FAIL|
|DV15 unequal|4; 2|covariance error0.682916; eigenratio0.874163; mass ratio0.439453|FAIL|
|DV15 bars|11; 8|4 modes; HQ1.00000|PASS|

Unequal-mass thresholds remain covariance error≤.85, eigenratio≥.15, mass ratio≥.25 plus all other original bounds and the required five-check suffix. DV15 passes only1000,1050,1150,1200; at1100 covariance error rises to1.474241, then ends.682916. Its final rare-component covariance error1.994825 remains the largest shape error. Adequate final HQ/spread/mass cannot erase that departure. DV13 bars fails at550 (HQ.84375) before its final two passes; DV14 bars fails at600 (HQ.84375). DV15 bars passes425–600 for its final eight checks. All six full24-check traces remain intact.

Each candidate therefore has **1PASS /1FAIL /20NOT_RUN** among22 tasks, with different passing tasks. DV14 is the strongest ring/vector research lead; DV15 is the image-repair lead. Their passes cannot be combined. The frozen released K3P unequal-mass reference from the previous attempt (21/24, final16, covariance error.181380, eigenratio.666039) remains context only; it was not rerun or tuned here.

**Mechanisms and actual behavior.** DV13 adds per-particle diagonal latent widths: bandwidth EMA.01(std(prior.z)×N**(-1/d)) multiplied by softplus(log_width)/log2. Zero-initialized new parameters consume no RNG; widths learn through the ordinary RpGAN generator objective in separate native Adam prior-role groups. G/D/z, loss, direct particle rule and canonical streams retain their contracts. O(Nd+Bd) latent work/storage avoids DV12’s exhaustive neighbor search.

DV13 has a preserved declaration/implementation mismatch: its controller was omitted from the penalty binding allowlist. Its normalized data/payoff/surprise rate policy operates, but native KA2 critic-memory mixing runs without the intended DV7 data/game feedback. The original declaration, sealed source and scores remain untouched; `dv13-declaration-deviation.json` records this correction. The mismatch itself is not an eligibility rejection; its measured bars4 failure gates off qualification.

DV14 restores the critic-memory binding and replaces each width gradient with the average of positive/negative latent-noise RpGAN gradients. Extra backward computes width gradients only; ordinary G/z/D gradients, real minibatches, global RNG and G buffers retain their single-draw behavior. A cloned pre-G noise stream supplies the opposite perturbation and identical output noise. Construction/load contracts prove the penalty points to the trainer’s live controller. DV14 changes both memory and width estimation, so their individual contributions to its ring/vector changes are not isolated.

DV15 retains DV14 but uses uniform[-1,1] latent coordinates inside learned boxes, variance1/3, in training and observation. The bounded latent law has no tuned multiplier. Output noise stays Gaussian.029. The bars4 improvement supports further study of compact support, but Gaussian tails and variance were changed together; their causal effects are not isolated. Its own rare-vector instability prevents qualification.

**Controller and lifetime.** All use `get_recipe(total_steps=None, continuous_policy=...)` and actual `GANTrainer.step()`, with input/output noise0/.029. The real-data detector compares fast.1/slow.01 means of32 normalized random linear/sin/cos features using empirical minibatch variance and covariance. It can detect more than mean changes, but is an engineering signal, not a universally calibrated test. Real statistics control scalar mobility only. Separate EMA.02 payoff drives cold/game mobility; unexplained Adam-gradient surprise brakes rates. DV14/DV15 also gate critic-memory release on current data evidence. No labels, target centers, quality, task IDs, change notifications, endpoint or caller phase enters the learner.

The799-call KA2 bootstrap initializes one critic reference. All later rates, memory, width/noise updates and learning remain available at arbitrary ages; it does not encode an ending or require a caller switch. Fixed rolling smoothers also never expire. This is a source design argument, not own long-age evidence. DV13 closes/reopens/closes at1027/2407/3257; DV15 at948/2407/3186. DV14 lowers and re-raises rates but does not cross the diagnostic mobility<.1 close threshold before the change; it first crosses at3145. Do not claim a DV14 pre-change closed-state event. Every applied G/D/z/width rate, noise, width range and policy state is logged; `dv*-*-control-events.json` summarizes them.

**Validation and limits.** Full public ring dimensions stay20,000particles/z2/batch2048, width96×3/Fourier3, seed0, CPU initialization then assignedCUDA. All frozen vector/image architectures, original ordered parameter fixtures, data/scorer streams, isolated observations and thresholds are retained. Added width parameters/EMA and optimizer groups are explicitly declared. Audit confirms original G/D/z/EMA, original lazy optimizer groups and all initial RNG/target hashes match. Whole-model/optimizer hashes differ because of the declared added widths/groups; the original summary flags are left unchanged. CUDA model tensors/moments and CPU native Adam scalar counters were verified from saved storage locations. No replacement Adam.

**138/138 final regression tests PASS; 1401/1401 source/state/artifact checks PASS.** CPU sample isolation and full-state restore cover controller, learned widths, optimizer, EMA and RNG; they do not prove fresh-process CUDA continuation. Early DV14 preflight found a syntax error, and an incorrectly queued startup also failed before initialization (zero training updates). Broken source/log and error rows remain; the startup elapsed time is unavailable and logged null. Corrected preflight passed before the completed cold run.

Own stationary7500, delayed/repeated9000 (changes6000/7800), uninterrupted30000 (also27000), differing-budget prefix, fresh-process CUDA continuation and the remaining20 tasks per candidate are **NOT_RUN**, gated by own broader failures. Shared protocol bytes are unchanged and separately hashed. Matched K3P ring comparator is NOT_RUN here; shared ownership remains with supervisor. Eight custom hosts still need faithful controller/sampler integration retaining auxiliary losses; no unsupported route receives credit. Finite windows cannot establish infinite stability.

**Recommendation for root review:** retain learned local support as a bounded-cost repair of the earlier atomic rank deficit, and preserve DV14’s vector strength plus DV15’s compact-support image evidence as distinct leads. Next investigate why rare-component assignment/shape becomes unstable under compact support using ordinary adversarial signals. DV15’s rare mass ratio jumps.256348→.439453 at1100 while covariance error spikes; this is a diagnostic lead, not causal proof or a reason to move an endpoint. Re-earn one unchanged policy’s joint gates before expensive expansion. No coefficient grid, seed sweep or fourth proposal was run.

Sources: `dv13.json`, `dv14.json`, `dv15.json`; per-run immutable `source.zip`; `inherited-source.zip` and `predecessor-source-receipt.json`; `predecessor-to-dv13.diff`, `dv13-to-dv14.diff`, `dv14-to-dv15.diff`; per-variant libraryZIPs and `final-source-sha256.json`. Final library patch SHA256: `ffed0126e1acaaa8de6b34c9ac31a5f9a14b220d626d05aa0311eb7e8239b4ac`. Exact results: `runs/dv{13,14,15}-{single,unequal,bars}/result.json`. All state/artifact hashes are retained.

Ledger totals: {"PASS": 11, "FAIL": 5, "ERROR": 1, "SKIPPED": 81}. Measurement-complete PASS and regression/audit PASS are not quality qualification.

Replay from this checkout, restoring the selected run’s exact source.zip over PR195 first:
```sh
bash reports/data-drift-api/run.sh dv15-replay reports/data-drift-api/worker.py --schedule dv15 --protocol single_shift --output reports/data-drift-api/runs/dv15-replay
bash reports/data-drift-api/run.sh vector-replay reports/data-drift-api/vector_gate.py --candidate dv15 --task vector_unequal_mass --output reports/data-drift-api/runs/vector-replay
bash reports/data-drift-api/run.sh bars-replay reports/data-drift-api/remaining_images.py --candidate dv15 --task img_bars4 --output reports/data-drift-api/runs/bars-replay
tail -F reports/data-drift-api/runs/current.log
```
`run.sh` reads supervisor.md and explicitly sets the assigned GPU/CUBLAS/thread variables; workers enforce FP32/determinism/TF32 off. Vector/image adapters also require the corresponding candidate’s original single-run declaration for source matching. Use their own ZIPs for complete historical replay.
