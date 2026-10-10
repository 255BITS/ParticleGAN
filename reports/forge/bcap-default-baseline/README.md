# Full ordinary BCAP baseline comparison

Direction blend passes **6/6 Tier 1 and 9/21 Tier 2**, versus the matched incumbent's **6/6 and 7/21**. It preserves all seven incumbent Tier 2 passes and repairs trajectory and residual student. All 21 Tier 2 questions were measured for each recipe. Twelve required Tier 2 failures remain; both required Tier 3 cells are **BLOCKED** by the ordinary prerequisite veto.

This is a best-observed research baseline comparison. The screening profile remains provisional: these results do not establish a calibrated scientific default, robustness across seeds, scale transfer or endurance. The frozen [selection rule](spec.json) also requires software acceptance and exact saved-state comparison before changing the named preset or family selection. The final API decision is recorded in [DEFAULT_SELECTION.md](DEFAULT_SELECTION.md).

## Comparison contract

Execution source is `d378734f40b09ce223a389e8f54a9783ec6a0c75`, scientific digest `6a225fcdd6922cdad37c9c947e163fb6f164f6ad4390293a3d3091b8f741ce44`. Both ordinary studies use view revision 8, protocol seed **0**, public deterministic initialization and CUDA on RTX A6000. Each task fixes architecture, data law, batch sequence, prior, sampling, update allowance and evaluation cadence. Constructor, data, training-noise and evaluation streams are isolated and checkpointed. There is one global recipe per arm.

The sole trainer delta is `constraint_geometry_mode: none → direction_blend`. Both recipes use non-saturating BCAP, zero-momentum DualNorm, native SVD, smoothing `.001`, per-offset convolution, G/E LR `.012`, D multiplier `1.5`, prior multiplier `2.5`, coefficient/cap/lazy interval `1/1/1` and constant rates. Transport, finite critic guards, prior regularization and additive training noise stay disabled. Task-owned priors override the preset's optional prior factory; the task's actual MoG/cloud identity is retained. [Registration](registration.json) and [specification](spec.json) contain the complete effective recipes and budgets.

The earlier selected-cloud and conditional diagnostics motivate this comparison; they supply no ordinary gate credit. Historical search results retain their original source, seed and initialization contract. The fresh `none` arm is the causal control here.

## Complete Tier 2 results

| Tier 2 task | Incumbent (`none`) | Direction blend |
| --- | --- | --- |
| [gaussian1d_stability](media/candidate-gaussian1d_stability.gif) | FAIL | FAIL |
| [five_word_joint_hold](media/candidate-five_word_joint_hold.gif) | PASS | PASS |
| [trajectory](media/candidate-trajectory.gif) | FAIL | PASS |
| [residual_student](media/candidate-residual_student.gif) | FAIL | PASS |
| [unipolar](media/candidate-unipolar.gif) | PASS | PASS |
| [cover_leftover](media/candidate-cover_leftover.gif) | PASS | PASS |
| [mid_scale_identity](media/candidate-mid_scale_identity.gif) | PASS | PASS |
| [mode_hold](media/candidate-mode_hold.gif) | FAIL | FAIL |
| [vector_two_broad](media/candidate-vector_two_broad.gif) | PASS | PASS |
| [vector_unequal_mass](media/candidate-vector_unequal_mass.gif) | FAIL | FAIL |
| [vector_unequal_width](media/candidate-vector_unequal_width.gif) | FAIL | FAIL |
| [vector_anisotropic](media/candidate-vector_anisotropic.gif) | FAIL | FAIL |
| [vector_overlap](media/candidate-vector_overlap.gif) | FAIL | FAIL |
| [vector_spiral](media/candidate-vector_spiral.gif) | PASS | PASS |
| [img_stripes2](media/candidate-img_stripes2.gif) | PASS | PASS |
| [img_bars4](media/candidate-img_bars4.gif) | FAIL | FAIL |
| [img_blobs4](media/candidate-img_blobs4.gif) | FAIL | FAIL |
| [img_intensity2](media/candidate-img_intensity2.gif) | FAIL | FAIL |
| [grid100](media/candidate-grid100.gif) | FAIL | FAIL |
| [rotated100](media/candidate-rotated100.gif) | FAIL | FAIL |
| [staggered100](media/candidate-staggered100.gif) | FAIL | FAIL |

Both arms additionally pass all six Tier 1 tasks: Gaussian acquisition, two pole, unused token, AE/GAN hold, ring acquisition and full word acquisition. The optional clock audit passes for both. Ring hold and extension were requested through Tier 3 but receive no execution budget after the required Tier 2 failures. Unrequested additional Tier 3 view requirements remain unknown.

The queue records the two requested Tier 3 cells as **BLOCKED** by prerequisite veto. The technique inventory's measured-cell reducer displays these unexecuted cells as **UNKNOWN**. These are distinct operational and measurement states; neither counts as a pass or supplies endurance evidence.

Every completed task has a linked saved actual-training GIF, including failures. [Media index](media/index.json) covers all **56** GIFs; no training, scorer samples or optimizer updates were added to make them. Metrics and gates are the comparison evidence.

## What the two repairs mean

Trajectory identity MSE falls from **.2398618907** to **.0002584800**, below its `.02` bound, with **19** consecutive passing observations. Residual identity MSE falls from **.0610360131** to **.0002490885**; correct assignment rises from **.5 to 1**, wrong-pad assignment falls from **.5 to 0**, and its passing suffix reaches **21**. Both controls have a zero passing suffix. This repairs complete sustained gates, rather than only the final plotted point. Word hold remains PASS from each arm's own exact producer, with matching selected prefix **834** and final local step **4834**; its complete final numerical packet is identical.

Direction blend detects harmful parameter directions after the normal optimizer proposes its actual joint generator/encoder/prior update. It removes motion that would increase an existing protected loss locally, then blends in a direction that improves the protected objectives together. The critic's optimizer remains unchanged. See the notation and pseudocode in [the PR](https://github.com/255BITS/ParticleGAN/pull/377). This is first-order protection; full finite steps can still increase a loss through curvature or rounding. It does not certify convergence.

Saved optimizer counters show **337 blends and zero stalls** across six tasks: unused token `167/200`, AE hold `16/250`, trajectory `12/400`, residual student `55/400`, cover leftover `44/800` and mid-scale identity `43/800`. The other **21** required task pairs have zero activation and byte-identical actual trained tensors for every applicable role (model, direct coordinates or prior). All twelve remaining Tier 2 failures are in that inactive group. Counter ownership and clocks are checked separately from wrapper metadata. The stored maximum derivative after correction is floored at zero; it is not a finite-loss guarantee. [Counter summary](saved-counter-summary.json) and [transparent saved-layout repair](counter-proof-repair.json) retain the original proof attempt and the exact source of standalone fixture parameters.

## Where it still fails, and likely causes

The twelve remaining failures have identical final metric packets in the two arms. Their distinction matters when choosing the next mechanism:

- **Gaussian retention:** shifted final standard deviation is only `.662330` of target, normalized mean error is `.255238`, and KS is `.320623`. The unchanged retention failure is consistent with contraction and imperfect reacquisition, not simply missing acquisition. The saved complete stationary/deadline/shifted gate remains FAIL.
- **Unstable mode retention:** the endpoint has all **8 modes** and quality **.989258**, but its terminal passing suffix is **3**, below the required **5**. A good final image or endpoint score would hide the failure to stay good.
- **Mass and width allocation:** unequal-mass core/global summaries conceal an underfilled small component (minimum mass ratio `.207520`, minimum component eigen ratio `.009073`, full component covariance error `3.691653`). Unequal width has mass TV `.291504` and full component covariance error `6.287559`, while its core error is `.444178`. These support distinct occupancy, within-component shape and spill problems; an improved aggregate covariance alone would not repair the gates.
- **Anisotropy and overlap:** anisotropic SW1 `.197952` and mass TV `.195964` exceed `.18` and `.15`, even though quality `.979980` and the terminal covariance/eigen checks pass. Overlap SW1 is `.255769` with covariance error `.438490`. The evidence points toward spatial allocation and distribution mismatch as well as density shape.
- **Image diversity and quality:** bars and blobs each reach only **2/4** quality modes. Bars have quality `.96875` but mass TV `.40625`; blobs have quality `.75`. Intensity has **2/2** modes and low TV `.03125`, but quality `.6875` fails `.9`. Improving only diversity or only sample quality would leave different failures behind.
- **100 Gaussians:** final precision is **.24072** on grid, **.25552** rotated and **.30168** staggered, and all full accuracy/holdout gates fail. Grid also reports center RMS **1.520315** target sigmas and radial KS **.490883**. Some rotated/staggered component accuracy fields are unavailable (`null`), not zero-valued successful measurements. Broad occupancy is insufficient evidence of resolved narrow components.

The observed benefit is consistent with protecting competing existing objectives in conditional hosts. It does not show that adjusting direction alone supplies missing mass-allocation or density forces. This interpretation is a hypothesis supported by unchanged failed metrics and saved activation evidence; no new gradient-field or finite-loss measurements were fabricated. The next registered study should isolate one global mass/density mechanism, measure actual force and displacement, and preserve all six Tier 1 gates plus the repaired trajectory, residual and word hold. Stop the already rejected five moonshot revisions; their source-bound evidence remains in the [three-phase report](../bcap-three-phase/README.md).

## Evidence, costs and software

[Final results](results.json) preserve compact numerical gates, actual checkpoint references and every paid attempt. [Saved audit](audit.json) checks the frozen requests, initial state, stream bindings, complete consumed states and own producers. [Independent audit](independent-audit.json) distinguishes recorded batch hashes from batch identity inferred from the fixed source/data law and matching actual data-stream states. Those are separate kinds of proof. The [saved counter supplement](saved-counters.json) separates actual activation, model/prior tensor equality and optimizer metadata.

The initial source-inventory reconstruction retained the correct numerical task outcomes but marked the studies BLOCKED because its resolver omitted their registered study IDs. The [reporting repair receipt](reporting-repair.json) preserves that failed reconstruction and binds the correction: resolve the exact original admitted study, hydrate only its Git-pinned motivation files, and require unchanged admission, source, view and full scientific cohort. Missing, tampered or ambiguous bindings fail closed. This reporting-only adapter sits outside the 1,206-file paid scientific manifest; it changes no task gate, training source, original receipt or replacement predicate. The corrected reconstruction uses normal study validation, while all original numerical outcomes remain intact.

Current-inventory publication also exposed older task fingerprints in four existing technique selections: K3P, KA2, R1/R2 and Release07. Their AE, word, pole and unused-token measurements belong to their original host sources. The [retained-pin publication receipt](retained-pin-publication.json) records the failed publication, exact original Git task/view validation and explicit stale-measurement labels. Their raw selection pins, numerical cells, source identities and scientific tiers are preserved. They receive no current-task qualification or cross-source ranking credit. The new BCAP configured-standard pin still passes the unchanged live guard against all current required task contracts. The frozen reconstruction helper and subsequent publication helper are bound separately; this correction adds no training or regrading of archived science.

The full publication path also required descriptions for 15 existing legacy BCAP cards. The [editorial documentation receipt](family-documentation-repair.json) binds their original identities, source/readout links and navigation under BCAP configuration alternatives. All 22 existing registry entries and its policy remain unchanged; the selected main BCAP configuration stays `bcap-dualnorm`. These editorial paths sit outside the paid scientific manifest. The descriptions retain historical scope and resolved recipes, and add no gate or qualification credit.

The pair used **56 paid attempts**, **4,187.683587 charged seconds**, **zero retries**, and no quality rescue. The full reservation was **93,240 seconds** within the **96,000-second** ceiling. Completed tasks reserved **86,040 seconds**; Tier 3 spent nothing. Logs, JSONL, checkpoints, tensor packets and JUnit remain under `/mnt/ml7tb/ParticleGAN-forge/bcap-default-adoption-20261010`; compact receipts and reproduction sources are committed. The initial abbreviated-SHA admission error spent no training and is retained in `logs/admission.log`.

[Software provenance](software-provenance.json) retains the original full CPU run: **5,938 passed**, **24 failed**, **299 skipped**, **1 expected failure** and **131 passed subtests**. All 24 failures were the archived comparator rejecting two added inactive metadata defaults. The immutable comparator and all its numerical assertions were preserved. A scoped AST/hash-checked adapter verifies `critic_step_mode=none` and `optimizer_svd_backend=native` in the ordinary pytest path, including its subprocess. The corrected **67-case** group passes, as do the separate **3** required renderer checks; **42** complete disabled-state pairs match actual develop. Earlier focused verification passed **450** cases. These counts overlap and are not a sum of unique tests. The full failed run is not relabeled PASS; the narrow corrected follow-up resolves its exact failures. Reported pytest time before publication checks is **1,155.67 seconds**, within the declared **1,800-second** software allowance.

The registered-study reporting repair adds **12 passing software checks in 1.71 seconds**, including missing/tampered/mixed study bindings, changed source/runtime, Git evidence hashes and unchanged numerical outcomes. All four targeted development invocations total **6.39 seconds**; including them brings reported pytest time to **1,162.06 seconds**. Their overlapping counts and two initial fixture-access errors are retained in the [repair receipt](reporting-repair.json). These are reporting checks and add no training or task samples.

The retained-measurement adapter's final **10 checks pass in 5.74 seconds**. Its development history totals **16.54 seconds**, including the preserved initial setup errors and renderer assertion failure; see the [retained-pin receipt](retained-pin-publication.json). These checks exercise original-source validation, stale labels and rejection of altered pins or contracts.

The editorial registration adds **3 passing checks in .18 seconds**, and its full saved-only publication preview passes. Including all these development invocations brings reported pytest time before the final post-selection check to **1,178.78 seconds**. Publication previews and inventory reconstruction are reported separately from pytest; none adds paid training, samples or optimizer updates.

The final bounded CPU post-selection group passes **38 checks**, with **1 expected CUDA skip**, in **5.86 seconds**. This brings the recorded pytest total, including the preserved failed development invocations, to **1,184.64 seconds**, within the **1,800-second** allowance. The [post-selection receipt](post-selection-software.json) binds the actual published main slot, preset fields, original receipts, all 27 required final pairs, three rejected saved-proof negative controls, other unchanged selection pins and every original scientific source file. Earlier focused CUDA coverage remains in the original software provenance. These checks do not rerun the paid experiments.

## Reproduce or inspect

Preparation and both study declarations are committed. Run the original frozen source for a new execution; preserve its source and task contracts instead of attributing a changed checkout's outcome to this run. Saved publication and selection add no training.

```sh
python reports/forge/bcap-default-baseline/run.py \
  --source-commit d378734f40b09ce223a389e8f54a9783ec6a0c75 --submit
python reports/forge/bcap-default-baseline/run.py \
  --source-commit d378734f40b09ce223a389e8f54a9783ec6a0c75 --drain --gpus 0,1
python reports/forge/bcap-default-baseline/publish.py \
  --source-commit d378734f40b09ce223a389e8f54a9783ec6a0c75
```

For the existing execution, use its exact original registration and artifact archive; do not rerun unchanged experiments for a reporting or merge commit. Tail `logs/driver.log` for the archived central progress stream or each receipt's original worker log.

[Exact v3 reconstruction source](reproduction/registered_study_reporter_v3.py) preserves the helper bytes used before the subsequent retained-pin publication correction. To reproduce that reporting version, restore those bytes to `reports/forge/regenerate_technique_inventory.py` in an isolated checkout and supply its explicit repository root and frozen execution commit. The archived copy's original default-path calculation assumes that location.
