# G observation recurrence round11

Five fresh 2k scouts build on proposal_mixed_pair25 at2k. Configs keep the
10k schedule and all winning point feedback, mixed judging, pair25 and clock
settings. Variants: G state8; state16; state16 update reads pre-write D memory;
state16 without proposal repair; state16 without access to D memory.
The no-D control still trains D with its own M. It zeros M before every G
reader/adapter call and its observation-only state updater never receives M.
No seed sweeps. No MSE or additional auxiliary training objectives, clipping,
EMA, geometry labels, moving cursor or generated full-rollout training.
Default API B-cap and prior regularization remain unchanged.

D owns M. G owns S, updated by GRU(actual observation,S) during real prefix
encoding and GRU(generated observation,S) at runtime. One variant also consumes
pre-write M. G reads z,M,S,clock through existing stateless proposal refinement.
Neither proposal nor final read changes S. Zero-prefix examples start M=S=0.
Point feedback writes the identical blended observation to both memories.
Pair branch independently starts from a real prefix and generates two points
with one intervening write to each state. No third generated point.
S learns by full BPTT over real prefixes (up to63 points), plus bounded local
feedback. This adds126 batched prefix GRU calls per D/G phase, plus two local
writes per phase. D phase builds S without gradients; G phase freezes D and
trains S with GAN gradients. Four complete G calls/eight reader calls per phase
for proposal configs, four reader calls without repair. Actual cost is reported.

Saved control: runs/memory_path/recovery_round10/runs/proposal_mixed_pair25.
Legacy equivalence checked separately, without a duplicate-seed training scout.
Training uses both GPUs via memory_dispatch; stdout only completed/failed events.
Central log: tail -F runs/memory_path/core_round1/train.log
Inspect results only after completion; compare all scouts before promotion.

## Fixed selection rules before training

Primary axes remain full1024 cold circles and warm original-orbit passes at
prefix8/32. Preserve256 results, direction diversity, stopping and phase errors.
Secondary rank: minimum warm1024 Q over both prefixes. Record longest good arc,
late Q and radial error separately; Q is not a success probability.
Up to two exact2k->5k continuations, same10k schedule, selected after all scouts
finish. A candidate qualifies if both warm prefix pass fractions improve over
baseline, OR if Q improves >=20% at BOTH prefixes while late Q is no worse,
radial error is <=5% worse, direction agreement <=2 percentage points worse,
and cold late stopping <=1 percentage point worse. Apply the stopping condition
also to the primary pass route. Rank qualifiers by minimum warm pass fraction,
then minimum Q. No obligatory extensions when nothing qualifies.
A cold-only pass improvement is recorded separately, not evidence of retaining
an expert process. No automatic changes to gates after seeing results.

Completed-model diagnostics: change radius/speed/direction in matched real
prefixes; measure late process response. Independently zero/shuffle D and G state
at runtime. Compare G16 with separately trained no-D and no-repair controls.
Intervention dependence is distinct from benefit over a trained control.
Compute/capacity differ across architectures and must be stated.
