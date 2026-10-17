# Retained Atlas/E22 baseline debugging: a measured shape excursion

The selected C6 two-broad runs retain **original1200 PASS / original study INCOMPLETE**. Their separate 150-update H2 continuations complete to1350 and **FAIL persistence: 3/5 later checks pass**. The failure is the unchanged analytic projection-CDF bound at1250 and1300; every other original bound passes there. This is not missing modes, zero within-mode width, a timeout, or evidence that the family cannot attain a passing cloud.

No new scientific run, model, restoration, sample or official score was produced. The reader checked original receipts, actual saved arrays, checkpoint dictionaries, named RNGs and 148 frozen scientific source files. [retained-hold-debug.json](retained-hold-debug.json) preserves all25 original observations and three appended observations per family, exact source/Recipe/runtime/artifact identities, state comparisons and descriptive CDF locations.

| Step | Recorded projection KS ≤ .06 | Mass TV ≤ .15 | HQ ≥ .85 | Minimum component eigenratio ≥ .15 | Recorded result |
|---:|---:|---:|---:|---:|---|
| 900 | .03057289 | .00292969 | .98974609 | .80654645 | PASS |
| 950 | .03821006 | .00439453 | .98706055 | .84568393 | PASS |
| 1000 | .05770876 | .00244141 | .98974609 | .83047187 | PASS |
| 1050 | .05751785 | .00341797 | .98974609 | .75236636 | PASS |
| 1100 | .05929161 | .00268555 | .99487305 | .73337990 | PASS: first five confirmed |
| 1150 | .05206046 | .00268555 | .98657227 | .95187807 | PASS: hold1 |
| 1200 | .05444427 | .00268555 | .99243164 | .82847965 | PASS: hold2 |
| 1250 | **.10582368** | .00195313 | .98510742 | .81273556 | **FAIL: hold3** |
| 1300 | **.07763310** | .00195313 | .98901367 | .79033422 | **FAIL: hold4** |
| 1350 | .05125857 | .00195313 | .98950195 | .90535063 | PASS: hold5 |

The precise failed bound in both added observations is `projection_ks <= 0.06`. Margins are .04582367846 and .01763309514. The endpoint recovery cannot erase those failures. The original first five consecutive passes are900–1100; with50-update cadence, five strictly later checks require1350. The old1200 horizon was insufficient for that study condition. H2 supplied the missing three observations and exposed genuine failures, rather than retroactively qualifying the original run.

## Where the saved CDF disagrees

The target is an equal mixture with means(−1,0)/(1,0), covariance .0625I in each mode. Descriptive arithmetic against that fixed analytic law identifies the exact worst direction and coordinate without calling the official scorer or changing its grades. Its maximum agrees with each recorded KS to within1e−12. Nearest-center partitions below are observable classifications, not unobserved mixture labels.

At1250 the worst projection is **y (90°)** at y=.03492613137. The retained empirical CDF is .10582367846 above the target CDF. The nearest-left contribution is +.05034836267 and nearest-right is +.05547531579: both sides contribute. Their y-means are **−.04765235632 / −.07245578947**, versus target0. Their x-mean offsets are +.04427840169 / +.03877272503. Centered covariance eigenratios are .812736/.957690 and .867897/1.117812, so width remains nonzero and relatively close to the target while the marginal CDF still fails.

At1300 the worst direction is **28.125°**, projected coordinate−1.00238911336. The .07763309514 CDF excess comes almost entirely from the left partition (+.07763309514; right contribution about−1.2e−14). The left mean offset is **(−.07196020594,−.05038348280)**; right offset(+.03821365121,−.01764648395). Left covariance eigenratios remain .790334/1.221572. At1350 the worst gap falls to .05125856533; means and covariances also move toward the target. These spatial observations describe the failing state. They do not establish that a mean shift alone causes all CDF error or identify which training owner caused it.

Nearest counts at all three new boundaries are2056/2040, mass TV8/4096=.001953125. The analytic CDF gate tests shape beyond mode counts, coarse covariance and a few aggregate moments. At1100 the two x-mean offsets are approximately−.05405/+ .05800 while recorded global mean error is only .002528: opposing component offsets can cancel in a global mean diagnostic. `target_scale`=1.05555856 in the receipt is RMS normalization of a fixed comparison reference, not a rescaling of this law.

## Serving, controls and equality

This is the full public selected-serving law with **additive output noise off and DV12 latent perturbation still active**, not an exact256-atom population. Prior256×4, batch128, evaluation4096, protocol seed24002 and primary evaluation seed34002 are fixed. Displayed target arrays use independent seed134005; scalar comparison draws use fixed991 and sliced-W1 directions992; projection CDF uses no RNG. All retained target arrays at1250/1300/1350 match1200 byte-for-byte.

All **150 original arrays (819,300 values)** and **18 appended arrays (98,316 values)** match in shape, dtype and bytes across Atlas/E22. The appended NPZ and12-frame GIF files also match exactly. At1200 and1350, the21 model tensors,64 optimizer tensors, DV12 controller, birth/death, row-evidence, LR-state, output-noise state and named/data RNG agree. Model and optimizer numeric tensors are finite. Paired source-owned unavailable LR diagnostic sentinels compare as missing values, not valid learned parameters.

Complete policy equality does not follow. Recipes, Atlas backend/settled guard, surprise memory and global process RNG differ. Both saved endpoints select **fast**; average outputs at1250/1300 were not retained. Atlas selects reference kNN atN256 because its finite-resolution rule is infeasible (40 minimum flags versus12 maximum guard flags), so this host does not exercise its distinct high-population feature-cell backend. E22 uses its reference/DV12 path. Identical arrays here supply no native feature-cell or whole-family equivalence claim.

Cumulative endpoint counters show no new discrete birth/death move (11→11), isolation move (0→0), row-evidence reset (11→11), surprise fire (0→0) or controller reopen (0→0) during1201–1350. Birth/death evaluations continue600→675 and row-hold steps835→943. Controller mobility changes .00244964→.00115541, game trust .53557084→.99162148, and latent bandwidth moves by at most .000262. G, D, prior, paired averages, optimizer memories and continuous row/controller statistics all evolve together. Maximum parameter changes are .07114293(G), .03705844(D), .07028669(prior), .03050286(averagedG) and .04229856(averagedprior). No retained decomposition determines their separate contributions.

Saved G/table/noise LRs remain .0053125/.00796875/.0053125 at both endpoints; critic LR .00597655784→.00597656061 also matches cross-family. Atlas guard critic scale changes .25→.125; E22 has no guard. Matching endpoint rates and learned states do not establish matching every intermediate guard/rate action. Surprise ratios differ (1350:.871102 Atlas /1.446095 E22) without a recorded fire.

## Harness check and concrete next diagnostics

Retained H2 results explicitly attest exact complete-checkpoint restore, original1200 sampler/metric/array parity, observer purity, unchanged original input hashes and original Recipe preservation. Frozen source corroborates the contract: `GANTrainer.extend_execution` changes only the external cap; `policy.begin_step` uses that cap only for exhaustion checks; schedules retain `Recipe.total_steps=None`. The continuation changes1200→1350 external trainer/caller allowances, preserves data cursor and named streams, and appends150 updates. It is not a new1350-step initialization or a schedule reset.

There is **no demonstrated observer, seed, target or horizon-handling defect** in this evidence. The zero-update parity control and pure reads address checkpoint/sampler integrity. They do not prove a complete observed/unobserved150-update GPU trajectory twin; that twin was not retained. H1 engineering errors are separately preserved as0-update namespace-guard failures, not numerical failures.

The smallest useful follow-ups preserve all gates and add diagnostic evidence rather than another rate sweep:

1. **Retain boundary state and applied-update traces.** A source-bound observer should retain exact1250/1300 G/D/prior/controller/optimizer states, selected serving, actual applied role update norms, row scales/holds, critic payoff/penalty record and cumulative event counters. Complete state/RNG purity and a short observed/unobserved prefix check are prerequisites. Existing missing states must stay missing; the current endpoint pair cannot recover the causal path.
2. **Separate coupled serving factors in a named frozen-state diagnostic.** Evaluate the actual paired G/prior fast and average snapshots with the same frozen controller and named evaluation RNG, recording both laws separately. This would test whether an available average state attenuates the observed shape excursion. Averaging is already configured (`serve_average=4`, `ema_decay=.995`) and does not force averaged selection; neither saved endpoint selects it. No alternate score can replace the primary FAIL, and no “always use EMA” fix is justified yet.

These are proposed diagnostics, not executed fixes or qualification. The strongest historical noisy Atlas19 positive control remains a distinct law/init/host/protocol; its successful replay is consistent with this stricter noise-off sampled-CDF persistence failure. A source-only baseline-selection review owns that comparison separately. No learned-MoG,19/19, public-default or speed credit is borrowed.

## Provenance and preserved cost

Scientific source **8021a1c50c4aff90ddea5010d368cffdc857b2f6**:148 files independently hash-checked. H2 helper source82c85cc32c43746074cb2809ae0f2aed271484bd, SHA `adcfbd92b39e3affa2626965bf32101bd4ade8a0ad00c9a4e4bbd4a0df4a0b58`; execution digest429fc7d9a90a710346637da72ebe611a0a73515c05aede32c00640824fdfa760. Runtime RTX A6000, Python3.12.13, Torch2.13.0+cu126/CUDA12.6, one CPU thread, physicalGPU1/logicalcuda:0. Each had180s acquisition plus60s export allowance.

| Family | Original1200 receipt SHA | H2 result SHA |
|---|---|---|
| Atlas | e2bfc5fdc66ea2c32e77b22d16a00346a83852b3192d9f4b4ee2da2619da20a6 | 1844252366b6b5dfa65a382736572ef236448e9320446653443ac994f1d42cd3 |
| E22 | bf6005d2118b3f6994ca240e706a5e8a4e5ef94818f8b54ecc695a6fb00a03dd | 4c0d798b25208bf897dc0df3582ae2e96be4331e7ef965c721a85453cb8ad2b6 |

Original checkpoint hashes95401e30… /c68b983b…; continued467455f8… /45df3f3d…; appended NPZf49fc8c1… and GIFb56db267… are bound fully in the JSON. **179 consumed files remained byte-identical before/after reading.** Atlas H2 paid15.290889646159485s, E22 15.893738066079095s, plus preserved H1 engineering6.8186783420387655s =38.003306054277346s of480s. This analysis spends0 scientific seconds and makes no reservations, model/scorer calls, source edits or Git changes.
