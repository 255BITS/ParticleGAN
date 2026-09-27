# Review before a fifth data-drift attempt

DV10–DV12 exhausted this attempt's three proposals. Each has its own completed public-API ring result and a measured broader-quality failure. The driver exited successfully; no qualified default remains. This note prepares root review and does not launch another worker.

Attempt: `/ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T232546Z-1609794/data_drift_mobility/20260926T232546Z-1609802`.
Read its [final report](/ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T232546Z-1609794/data_drift_mobility/20260926T232546Z-1609802/result.md), ledger, per-run immutable source ZIPs, and the supervisor's [DV12 final audit](/ml2/hypergan/gan-attempts/continuous-api-20260926/supervisor-audit/dv12-final-audit.md) and [source audit](/ml2/hypergan/gan-attempts/continuous-api-20260926/supervisor-audit/dv12-source-audit.md). No predecessor or sibling pass transfers to a successor.

| Candidate | Original arrival; retention through 2400 | Shift arrival; retention through 4600 | Final shifted passing suffix | Ring elapsed |
|---|---|---|---|---|
| DV10 | 620; 179/179 | 2720 (+320); 184/189 | 180 from 2810 | 61.353 s |
| DV11 | 560; 185/185 | 2690 (+290); 190/192 | 187 from 2740 | 65.723 s |
| DV12 | 590; 182/182 | 2690 (+290); 191/192 | 187 from 2740 | 1168.141 s |

All original preholds pass 120/120; frozen controls pass 0/220. Preserve every shifted miss: DV10 at 2730, 2750, 2760, 2790, 2800; DV11 at 2720, 2730; DV12 at 2730. Minimum HQ after shifted arrival is respectively 0.679932, 0.617676, 0.710449; mode count stays eight. These are early settling losses followed by retention, not a first-touch-perfect rejection rule. The broader failures independently reject qualification.

| Candidate | Unequal-mass passing checks / final suffix | Final covariance error ≤0.85 | Minimum eigenratio ≥0.15 | Other own broader evidence |
|---|---|---|---|---|
| DV10 | 7/24 / 7 | 0.745409 | 0.828991 | Bars4 FAIL: 8/24, suffix 0, four modes, HQ 0.8125 <0.90 |
| DV11 | 0/24 / 0 | 1.230177 | 0.160566 | Other 21 tasks NOT_RUN |
| DV12 | 2/24 / 0 | 0.884388 | 0.624825 | Other 21 tasks NOT_RUN |

DV12 passes only at 1000 and 1050, then fails 1100/1150/1200. Its rare-component covariance error is 3.055543; passing occupancy, HQ and minimum eigenratio do not establish the correct shape. DV10 has one broader pass, one failure and twenty unrun tasks. DV11/DV12 have no image repair evidence.

DV10 adds continuous latent neighborhoods with bandwidth EMA(prior coordinate spread × N**(-1/d)). It repairs the predecessor's rare-vector spread deficit but fails image fidelity. Large image bandwidths (0.676–0.916) suggest excessive perturbation as a research hypothesis, not an isolated causal proof. DV11 globally suppresses that width using paired critic evidence; on unequal_mass its trust ranges 0.005600–1, yet covariance fails and mobility stays above 0.7499. DV12 clips displacement to half the exact nearest-distinct-prior distance. Local latent distance still does not guarantee the required output covariance.

DV12's exhaustive neighbor calculation costs **19.04× DV10's ring elapsed time** under the same observation protocol. Its vector takes 27.276 s versus DV10's 21.864 s. These include observations/checkpoints, not separate kernel timings; concurrent GPU load was not controlled, so the elapsed ratio is descriptive, not an isolated speed benchmark. Work is O(batch × particles × latent dimension) per generation; chunking only the prior dimension does not bound memory for arbitrarily large public sample requests. A successor should preserve continuous support while learning useful output shape with bounded computation, rather than shrinking one global coefficient or repeating exhaustive geometry.

The supporting exact released K3P vector reference passes 21/24, final suffix 16; covariance error 0.181380, eigenratio 0.666039, minimum mass ratio 0.739934. It uses the canonical fixture, reviewed public v0.8.0 package, native CPU Adam counters, explicit benchmark horizon 1200 and released noise milestones 120/240. It is scheduled reference evidence, not a continuous candidate or literal default-7000 result. Its isolated observation stream 2303 matches the declared current branch. Keep it frozen; do not rerun or tune it after these scores.

The final audit independently verified all 699 source/state/artifact checks and counted 130 passing recorded regression tests. DV12's fourteen package files match its single/vector snapshots, library ZIP and final receipt. Canonical ordered parameters and frozen scorer match; native Adam scalar steps remain CPU, moments/models CUDA, and vector checkpoints now save caller data RNG. Both training and observation add candidate latent noise before output noise 0.029: input noise zero does not mean noise-free latent sampling. The same ordinary scalar unconditional GANTrainer owns these behaviors. Standalone factories still need complete controller/sampler bindings; eight custom hosts must retain their native auxiliary objectives. CPU tests and complete saved states do not prove fresh-process CUDA continuation.

For the next root-reviewed attempt:

- Use one external Astra/max session, one GPU worker, and **at most three distinct new candidates**, DV13 onward. No coefficient grids, seed sweeps, fourth proposal, or chat-agent training. Explain each mechanism using these failures before running it.
- Preserve the ordinary adversarial learning objective and causal public-library ownership. No target centers, component labels, scorer values, fitted target distributions, caller phases, hidden endpoint or change-time access. Automatic reversible adaptation at arbitrary ages is allowed.
- Declare the entire learner, including latent/output noise, RNG consumption, computation, state and live/EMA routing. Keep canonical host/model/init/stream/scorer contracts; no source changes during a candidate's qualification. Global recipe changes must be explicit and shared across tasks.
- Re-earn own ring, unequal_mass and bars4 evidence before costly expansion; finish every started declared window. Preserve arrivals, every later miss, minima and final suffixes. Use unchanged COMMON.md/evaluation-protocols.json, with no 81/81 or zero-miss shortcut.
- A surviving package still needs own stationary7500, uninterrupted30000 and declared change/replay coverage, exact budget-prefix and fresh-process state/RNG continuation, and all frozen22 tasks through faithful API routes. DV10–DV12 earned none of these later gates.

Keep every failure and unsupported route explicit. No merge, release or default promotion; integration targets develop. Root owns refill, shared manifests and the final comparison.
