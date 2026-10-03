# Named KA2/K3P pair: recovered modes, failed finite-template fidelity

Both original 600-update intensity protocols completed and failed. KA2 recovered two modes with HQ .9697265625, but finite-template TV .12890625 exceeds .10. K3P recovered two modes with HQ .896484375 below .90 and finite-template TV .2021484375 above .10. Neither has one primary passing observation among the25 retained checks, hence neither acquired a five-pass window. Seven downstream questions per family remain UNKNOWN. All16 zero-update capacity outcomes are SUPPORTED; they confer no convergence or whole-family credit.

| Frozen terminal metric | KA2 | K3P | Original condition |
|---|---:|---:|---|
| modes | 2 | 2 | 2 |
| HQ | .9697265625 | .896484375 | >=.90 |
| nearest-assignment TV | .0986328125 | .0986328125 | <=.10 |
| rejected mass | .0302734375 | .103515625 | included in finite-template TV |
| finite-template TV | .12890625 | .2021484375 | <=.10 |
| mean nearest RMSE | .03644286096 | .03356356174 | diagnostic |

The .06 RMSE cutoff is applied to each image; mean RMSE is not an independent gate. K3P's lower mean RMSE does not imply fewer rejected images. The original final-five gate and added first-five-plus-later-five persistence gate both FAIL; no original grade is changed here.

## Retained population attribution

The final held-out arrays contain32 distinct outputs in each family. The public prior has exactly32 uniformly indexed rows; the original generator is deterministic, serving uses fast weights with no DV12 perturbation and no additive output noise. Thus these retained draws cover every distinct row output without a new forward or sampler call. They determine population counts, but not the missing latent-row-ID-to-output mapping.

Both populations have19 nearest dim outputs and13 nearest bright outputs. The1024 fixed evaluation draws contain613 dim and411 bright assignments. This gives the recorded nearest TV .0986328125, just inside .10. It does not by itself explain rejection by the finite-template gate.

KA2's one rejected bright output accounts for31 draws. Its central-patch mean is .97360134 against target .85, with background mean .00008776 and whole-image nearest RMSE .06275224. Its valid counts are613 dim /380 bright. K3P has three rejected bright outputs accounting for31+38+37=106 draws; patch means are .98209369/.97210270/.97228134 and RMSEs .06659586/.06259526/.06204615, with tiny backgrounds. Its valid counts are613/305. These are bright-patch overshoots, not near-black brightness collapse.

The original finite-template statistic includes rejected mass as a separate outcome: `.5*(sum(abs(valid_mass-target_mass))+rejected_mass)`. Here all rejections reduce the already underweight bright outcome, so it equals nearest TV plus rejected mass, yielding the recorded .12890625/.2021484375. That equality is specific to this partition, not a general identity. With uniform population weights, the same32 retained outputs give descriptive nearest TV3/32=.09375, rejected1/32 or3/32, and finite-template discrepancy4/32=.125 or6/32=.1875. This demonstrates that the observed problem is not solely a bad1024-draw frequency fluctuation; these population diagnostics do not replace the original sampled gate.

Both retained populations become19/13 by step275 and retain those nearest counts through600. Their off-template counts continue changing: at475/500 both have four rejected bright outputs; KA2 improves to one by550, while K3P retains three. Only endpoint prior parameters are available. This does not show which rows moved at unsaved updates or whether generator or prior gradients caused the allocation.

## What agrees and what differs

All20 retained sample arrays at steps0,25,...,475 are bitwise equal between the families. At500–600 they differ, with maximum final pixel difference .0581009984. The two original goal GIFs contain nine faithful frames and retain default FAIL. Equal retained observations do not establish equality of every unobserved update or earlier full optimizer state.

Final G, D, prior, G-EMA and prior-EMA tensors differ. Maximum absolute differences are .02396835/.21891862/.02476075/.00698282/.00734553 respectively. Both Adam optimizer numeric states differ. The learned-table damping histories differ, although their counters agree:12283 observed rows /19200 possible row observations (.63973958), started=true, no direct-particle response. This final rate exceeds the declared damping threshold .5; the checkpoint lacks an applied-step counter, so it cannot prove that damping never applied earlier. All named latent/penalty/evaluation/noise/model streams and the data-generator state agree. Global CPU and CUDA RNG state differ; this does not erase the measured common prefix or demonstrate random-stream identity for unsaved work.

The full Recipes differ only in name and the explicit K3P critic-formulation field; omitted KA2 formulation resolves to KA2. The final policy metadata agrees: `served_source=fast`, output sigma .029, independent rows. `serve_average=0` means retained G/prior EMA parameters are not the observed primary law. There is no continuous controller, DV12/feature-cell backend, learned-output-sigma owner, birth/death, evidence gate, LR-settle or reopen state in either checkpoint. Absence of those owners is a declared family boundary, not a disabled hidden control.

Both declared role rates start at .006375. The final applied G/D rate is .00006402035005 and prior rate .0003190094268: last update used the schedule at599, not the exact floor at600. Network/prior cosines start at360; network floor is .01 and prior floor .05. Fixed training output sigma warms to .029 by120; primary image observations explicitly add no output noise. Input noise decays to zero by60. `amsgrad=False` gives the nonmax Adam second moment, but public KA2/K3P guard, penalty and conditional row-damping mechanisms remain active owners.

Both critic records report600 calls/600 observed updates, and neither guard clipped a tensor. KA2 has `anchor_started=false`, zero critic-EMA updates and zero blended-surprise history: its source uses799 pure-A penalty calls and begins blending at call800. This short image budget never enters that phase. K3P's LR-driven record reports `anchor_started=true`; its EMA critic differs from KA2's retained unstarted anchor. Under the declared schedule and previous-step LR record, source arithmetic makes update483 the first possible K3P blended call. The observed exact475 prefix and500 divergence are consistent with that boundary. No per-update phase log or intermediate owner snapshot was retained, so the actual first numerical divergence and the reason for the bright-output overshoots cannot be proven from these artifacts.

## Causal limits and next contrast

The measured failures are finite-template outcome deficits, with an additional K3P HQ failure. These are scientific failures, not API/startup failures or demonstrated family incapacity. The exact common prefix, later parameter differences and distinct anchor records establish observable/source differences; they do not identify a defective critic, row damping, optimizer memory or an optimal coefficient. Losses, gradients and phase events were not retained because the runner discards each `step()` return; there are no intermediate model/optimizer snapshots or EMA image evaluations to reconstruct those paths.

Earlier near-black Atlas/E22 results used different public family mechanisms, DV12 sampling, rates and penalty strength. Their frozen failures remain separate; improved visible photometry here imports no earlier grade and is not a matched causal attribution to one changed knob.

A single new shared prior-rate contrast `.006375 / 2 / 1` is defensible as a row-transport question. It keeps G/D and all original goals, schedules, serving laws, seeds, budgets and gates; it doubles only the nominal prior group rate. The existing19/13 allocation leaves little finite-template margin, while bright-output quality still independently matters. A faster prior could change allocation or reject rates, could worsen them, or could leave the same outcome. There is no monotonicity or repair claim. Full candidate-bound capacity/source/state/sampler admission is required; the current fixed-tuple helper cannot simply reuse its old positive card. The finite declaration and remaining shared-campaign accounting are in `NEXT_PRIOR_RATE.md/json`; nothing is executed or reserved by this proposal.

## Provenance and cost

Execution source: `26ff278c3796d775969391adc0bde52e3af11149`. Scientific family worktree `/ml2/hypergan/ParticleGAN-ka2-k3p-defaults-20261003`; immutable archive `/ml2/hypergan/forge-ka2-k3p-defaults-20261003/{ka2,k3p}`. Runtime: Python3.12.13 / Torch2.13.0+cu126 / RTX A6000 / one Torch thread, logical CUDA0 in the root-owned physicalGPU1 lane.

KA2 study SHA `58721647a201b569c1ac0a992880d410ca5375836986807afbc547397e89aabf`, receipt `b60afb0fd92a55d3cd97dc5bdeaeb53af5004061653cc1722cf7a96b77f7aeb7`; K3P study `385a48da6efb236c9b2a388e5eb8fbebf43c0add3d1a34270c36ad71be749461`, receipt `2cb069fdd6cf3560385984f3a9d35d651e0d0df3e1aca28d864ea0582679c480`. Exact paths, artifacts, full Recipe/runtime/source manifests, all recorded scalar observations, descriptive counts and typed owner comparisons are in `pair-intensity-analysis.json`. Original GIF/NPZ/checkpoint SHA256s and sizes were checked; all consumed raw files were unchanged after reading.

Paid scientific attempts:13.6175829151s KA2 and13.5760902760s K3P, total27.1936731911s. Earlier campaign debit113.9942518845s remains separate, yielding141.1879250756s cumulative under the unchanged15360s ceiling. Receipt acquisition clocks6.8023999340/6.7122315301s exclude post-acquisition export; these are not speed rankings. Capacity CPU diagnostics and this read-only analysis do not add scientific run cost.

Source references (exact file hashes in `source-questions-and-observables.json`): `api_images.py:542–585,879–927` partition/serving; `training.py:336–398,450–486` update/owner serialization; `api_run.py:485–491,523–534` discarded step returns/retained arrays and final state; `ka2.py:219–291` penalty call clock; `grad_regularizers.py:150–155,235–292` LR blend; `recipes.py:696–785` role schedules; `k3p.py:149–232` conditional row damping. Independent readable-media/source QA: `/ml2/hypergan/ka2-k3p-defaults-goal-review-20261003/MEDIA_QA.json`.
