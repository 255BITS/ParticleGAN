# Ring16: CUDA runtime controls and numerical interventions

**CUDA access returned on 2026-10-07. All five prepared studies completed on the
two RTX A6000s and have separate PRs to develop.** The runtime control reproduced
the original rounding discrepancy and removed it by disabling autograd
multithreading during update 401. Continuous truncation, continuous weak-gradient
noise, and one-time weak-direction sign flips passed independently confirmed
acquisition. None establishes indefinite retention.

[Compact results](results.json) pin every execution source, protocol, receipt,
publication commit and raw archive. [Saved-evidence verification](execution-verification.json)
checks frozen sources, artifact hashes, budgets and shared batches without new
model calls. The [earlier preparation](preparation.json),
[CUDA refusal](gpu-execution-20261007.json) and [preparation archive](archive.json)
remain historical receipts from before device access returned. They are not
current blockers. Original [PR331](https://github.com/255BITS/ParticleGAN/pull/331)
and all five follow-up PRs remain unmerged.

## Later continuous combination follow-up

[PR338](https://github.com/255BITS/ParticleGAN/pull/338) tests both mechanisms
continuously under a separately frozen one-arm GPU protocol. It confirms
acquisition at1050 (independent covariance .817187) and passes all34 scheduled
checks1050–1600. Final covariance .470678, HQ .956055, massTV .078613 and minimum
component eigenvalue ratio .387340 satisfy all original bounds. No restart,
annealing or optimizer reset is used. [Compact results](serialized-truncation-results.json)
pin execution source `5e1d33f37b312a1b4a760e8417d76542570f0768`, archive and
original individual-arm comparisons. All9600 matrix updates pass a serialized
execution-scope guard; unchanged API software evidence is reused without training.

Matched initialization, recipe, prior, all1600 target batches, cadence, gates
and public numerical code are verified. Truncation alone confirms684 and settles
834, earlier than the combination's1050; the combination finishes with lower
covariance (.470678 vs .589678). Both are supported candidates for a separately
frozen whole-configuration Tier1 comparison. Compatibility is demonstrated;
universal superiority, combined restart parity and indefinite retention remain
unestablished. No empirical speed comparison or ordinary qualification is added.

This closed campaign spent1600 new GPU updates/97 scored draws/25.551 whole
subprocess seconds within300 reserved seconds, one attempt and zero retries.
The report includes an actual-training GIF and a byte-verified19-file raw archive.
Original individual results and budgets below remain unchanged.

## Later uninterrupted serialization follow-up

[PR337](https://github.com/255BITS/ParticleGAN/pull/337) completed a separately
frozen two-arm CUDA quality protocol; [its compact results](serialized-live-results.json)
retain execution source `b060fb4623cc831cb47dd592acea49b6704641be` and original
truncation/noise identities. Always-serialized fresh live training reaches
terminal quality (11 final full passes), but fails its one first-state independent
confirmation at1400 (covariance .894789). It fails again at1417, then passes all
checks1434–1600, ending at covariance .385800. Serializing only401 matches the
previously isolated third trajectory exactly, confirms acquisition867 and passes
all45 checks867–1600, ending at covariance .424426. Neither learner reloads.

Initialization, recipe, prior, all1600 target batches, cadence and gates match
prior continuous truncation/noise; those arms were not rerun. New spend is3200
GPU updates/194 draws/47.597 whole subprocess seconds, plus3 tiny-model GPU
software updates; charging its full30s allowance yields77.597s within630s.
Two CUDA API checks and saved-source/state/batch/archive/media checks passed.
The report includes two actual-training GIFs and a35-file byte-verified archive.

Truncation remains the first candidate for a separately frozen whole-configuration
Tier1 comparison. Serialized execution reaches late quality and provides an
opt-in checkpointed execution mode, while the failed first confirmation remains
FAIL. The one-time401 success does not justify scheduled switches or restarts.
No ordinary qualification, default change or Tier2 eligibility is added. The
original twelve-arm results and closed budgets below remain unchanged.

## What the reload changes

[Runtime PR334](https://github.com/255BITS/ParticleGAN/pull/334) ran four frozen
CUDA arms: fresh live versus restored, each with ordinary or serialized autograd
at update 401. Both fresh 400 prefixes matched the original complete checkpoint
exactly. The ordinary critic gradients matched their respective historical
live/reloaded gradients exactly, reproducing the original discrepancy.

Disabling multithreading for **all autograd work during update 401**, including
the higher-order `autograd.grad(create_graph=True)` calls and final backward,
made the live/reloaded gradients, normalized directions and complete resulting
contexts bit-identical. This selects a **third trajectory**, matching neither
ordinary path. Its 1,600-update quality was not measured by that closed causal protocol;
the separate quality follow-up above now supplies this measurement.

The saved critic graphs have identical topology but differ in 1,054 of 7,503
pairwise sequence-number relations. The serialized pair has zero differences.
Installed Torch 2.14 headers declare thread-local sequence counters and use node
sequence numbers to prioritize ready nodes at equal reentrant depth. Live graph
nodes occupy two widely separated counter ranges; restored counters start near
zero. This supports a specific explanation: runtime-local counter history changes
higher-order graph priorities and floating-point accumulation order after reload.

**Actual engine execution order was not traced.** The intervention does not
independently separate graph-construction thread choice from final backward
scheduling. These details remain an inference, while removal of the measured
401 discrepancy is demonstrated. No lossy dtype conversion was found: saved
weights are float32, complete checkpoint values and the first six forward inputs
and outputs match. Reload still runs SVD; its slightly different backward
gradient is amplified by unit-strength almost-null singular directions.

## Intervention results

These are unranked, source-bound diagnostics, each with 1,600 fresh live updates.
"Confirmed smoke" requires the **one** independent draw at the first full
scheduled pass. Failed confirmations were not retried at later states. Terminal
quality independently requires five consecutive final full passes.

| Matrix intervention | Schedule | Confirmed smoke | Terminal full passes | Final covariance error, bound .85 |
| --- | --- | --- | ---: | ---: |
| [Damping, PR336](https://github.com/255BITS/ParticleGAN/pull/336) | 401 only | FAIL confirmation | 31 | .458855 |
| Damping | Every update | FAIL; no full passes | 0 | 1.057656 |
| [Truncation, PR332](https://github.com/255BITS/ParticleGAN/pull/332) | 401 only | FAIL confirmation | 0 | 1.372586 |
| Truncation | Every update | **PASS at 684** | **47** | .589678 |
| [Tiny noise, PR335](https://github.com/255BITS/ParticleGAN/pull/335) | 401 only | FAIL confirmation | 6 | .481363 |
| Tiny noise | Every update | **PASS at 817** | **45** | .504405 |
| [Sign flips, PR333](https://github.com/255BITS/ParticleGAN/pull/333) | 401 only | **PASS at 1300** | **19** | .448597 |
| Sign flips | Every update | FAIL; no full passes | 0 | 1.477256 |

Each full pass satisfies every original bound: 16 modes, mass TV ≤ .15, HQ ≥ .85,
component covariance error ≤ .85, minimum component eigenvalue ratio ≥ .15 and
4,096 samples. All boundary arms matched the original live 400-state without
loading it into their learner. All eight arms used the same seen target batches.

**The continuous candidates did leave the gate after acquisition.** Truncation
failed covariance at 717, 734, 750 and 817 after its confirmed pass at 684, then
passed every check from 834 through 1600. Noise failed marginally at 850
(covariance .851437) after its confirmed pass at 817, then passed every check
from 867 through 1600. These are useful smoke results and long late streaks,
without proving "reach the solution and stay there."

## Mechanism, fixed conditions and cost

The four numerical interventions use one fixed rank rule:
`tau = max(matrix.shape) * float32_epsilon * sigma_max`.
Damping weights each singular direction by `s / hypot(s, tau)`; truncation uses
`1{s > tau}`. Noise perturbs the weak gradient subspace before ordinary polar
normalization; sign flips randomize weak polar components. G/D matrix weights
change, while bias and sampled prior-row update rules remain unchanged. Random
interventions consume an isolated, checkpointed training-noise stream.

CUDA saved-gradient probes reduced the archived live/reloaded normalized-direction
discrepancy about **1,288× with damping** and **1,713× with truncation**. Noise and
sign flips instead increased it, while passing their implementation/RNG gates.
Thus reduced sensitivity is not sufficient for quality, and a successful
perturbation need not reproduce the reload's particular update. Tiny raw-gradient
noise produces a substantial direction change after normalization.

Every study retains seed 0, public deterministic named initialization,
G 4→64→64→2, Fourier D 10→64→64→1, batch 128, learned 256-location MoG with sigma .1,
constant G/D/prior rates .012/.018/.03, zero momentum and prior regularizer,
recipe horizon 400, clean live sampling and the original evaluation cadence.
No seed study, annealing, optimizer reset or checkpoint restart was introduced
into candidate training. Each intervention PR includes two actual-training GIFs;
runtime PR334 includes the measured common-prefix GIF. Rendering uses retained
CUDA outputs with zero new model calls.

New work totals **13,604 updates, 822 scored draws and 12 completed attempts**,
with zero retries. Training subprocesses consumed 235.313 seconds; four probes
are conservatively charged their full 30-second allowances, making the debit
355.313 seconds within the frozen 2,880-second ceiling. Original reproduction
cost is excluded. Five byte-verified raw archives reside under the main
workspace's `artifacts/`; their exact paths/hashes are in [results.json](results.json).
Bulk stdout, traces, samples and checkpoints remain outside Git.

## Recommendations

Prioritize continuous truncation for a separately declared global trainer
comparison: it directly removes the measured amplification and adds no random
stream. Keep continuous weak-gradient noise as a second supported candidate.
Evaluate one whole configuration across the complete Tier 1 task set before
eligible Tier 2 retention work. This fixed Ring16 result alone supplies no
ordinary qualification or family default.

For exact reload attribution, the next bounded control should distinguish
higher-order graph construction from final backward scheduling or capture actual
node execution order. End-to-end serialized-autograd quality was a separate
unmeasured question at that campaign completion; PR337 above records its follow-up. One-time perturbations demonstrate path sensitivity;
periodic reloads or repeated sign flips are not supported as continuous fixes.
Stop the completed negative every-step damping/sign-flip revisions.

The [technique inventory](../technique-inventory.md) remains the single generated
leaderboard and is unchanged. Diagnostics grant no additional tier eligibility.
