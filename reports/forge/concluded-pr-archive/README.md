# Concluded research PR archive

These compact recall records preserve concluded research before retiring its PRs.
Original source commits, document SHA256s and Git blob IDs remain in each record;
report, compact metric, protocol and actual-training GIF links point to immutable
original commits. Bulk logs, checkpoints, tensors and event streams stay in their
original local archives. Closing a PR does not delete its source branch.

This publication adds no training, current qualification, default change or cost.
The [current technique inventory](../technique-inventory.md) remains the goal's
single current leaderboard. Historical gate counts below keep their original
view, recipe, task and protocol; they are not a new combined ranking.

| PR | Concluded result | Retained evidence |
| --- | --- | --- |
| [#320](https://github.com/255BITS/ParticleGAN/pull/320) | G/D spectral cap .1 improves Gaussian shape but all continuous-learning gates fail; every ring arm passes 0/240 checks. Stop the exact recipe. | [Recall/provenance](../records/archive-pr320-network-spectral-cap-c4627fb14619.json), [original report](https://github.com/255BITS/ParticleGAN/blob/c4627fb14619ca0bcd1b8d904e7796d0834a4e8a/reports/forge/gaussian-network-magnitude/README.md) |
| [#360](https://github.com/255BITS/ParticleGAN/pull/360) | Latest winner follow-up: both hydraulic arms pass 4/17 against archived winner 5/17, gain zero gates, and lose ring or spiral. Stop both revisions. | [Recall/provenance](../records/archive-pr360-hydraulic-winner-bbac181e1281.json), [original follow-up](https://github.com/255BITS/ParticleGAN/blob/bbac181e128160a1e5770b60490ba99ebaffbf01/reports/forge/bcap-physics/hydraulic/winner/README.md) |
| [#395](https://github.com/255BITS/ParticleGAN/pull/395) | No global winner among annealing, half G/E motion, prior smoothing and weak critic damping. First three fail Tier 1; damping adds no matched Tier 2 pass. Stop all four revisions. | [Recall/provenance](../records/archive-pr395-four-bcap-repairs-51ee43476e8b.json), [original report](https://github.com/255BITS/ParticleGAN/blob/51ee43476e8bfa64f0eb55f630fe3b0086dddd9b/reports/forge/bcap-tier2-round6/README.md) |
| #397, already closed | Stall-reach critic read loses ring16; mode hold is unmeasured. | [Recall/provenance](../records/archive-pr397-stall-reach-critic-read-de93dbf48cab.json), [historical readout](../stall-reach/README.md) |
| #398, already closed | Generator idle 1-SE loses ring16; separate off-ladder Gaussian and mode-hold diagnostics still fail. | [Recall/provenance](../records/archive-pr398-generator-idle-se-835e93900c21.json), [historical readout](../bcap-generator-idle/README.md) |

#320 retains protocol seed 0, public deterministic initialization, batch 128,
256 uniform learned MoG locations at sigma .1, original sampled-row prior
normalization, clean live sampling and 30,000 new updates across nine CUDA trials.
The reported 362.227 seconds measure training loops. Its archived PR317 controls
retain their own source identities and grant no new control cost or qualification.

#360's added record covers the later 17-task winner comparison missing from the
round-three summary. Original rounds 1–3 remain separately pinned in that record.
The two winner arms use seed 0, public deterministic initialization and unchanged
task-owned priors, sampling, budgets and cadence. The control is reused archived
ordinary evidence, with the original source-parity/disabled-flag binding audit;
it is not relabeled a fresh matched run. Gaussian retention improves to 71/72
and 68/72 stationary checks, but complete retention still fails. All 34 new cells
complete for 5,159.622 paid seconds, with zero retries and a 64,000-second ceiling.

#395 retains its revision-9 denominator and frozen source, rather than applying
later task dispositions. Three blocked Tier 2 sets are unmeasured. Damping's
word-hold failure uses unequal acquired prefixes, control 834→4834 versus hybrid
1667→5667, and is preserved as a separate original-gate cohort. The other 20 Tier
2 tasks have matched complete budgets and consumed streams; they yield the same
eight passes. All 77 paid attempts and 6,773.287 charged seconds retain their
original archive identities. This summary adds no qualified attempt records.

Six further PRs already have sufficient source-pinned preservation on `develop`:

| PR | Existing comparison and original source | Preserved decision |
| --- | --- | --- |
| [#358](https://github.com/255BITS/ParticleGAN/pull/358) | [Round three](../bcap-physics/round3/README.md), [source](https://github.com/255BITS/ParticleGAN/blob/f9f6419ac251033e9076f7ba4bab87e715d321bd/reports/forge/bcap-physics/kinetic_transport/round3/README.md) | Transport backtracking loses local-v2's rare-density pass; stop successor. |
| [#361](https://github.com/255BITS/ParticleGAN/pull/361) | [Round three](../bcap-physics/round3/README.md), [source](https://github.com/255BITS/ParticleGAN/blob/745642f756cee1e540b823bb77ca8ebae792e3cc/reports/forge/bcap-physics/constraint_geometry/round3/README.md) | Strict-progress package repairs both conditional identities; preserve scoped result and finite certificate. |
| [#367](https://github.com/255BITS/ParticleGAN/pull/367) | [Round four](../bcap-physics/round4/README.md), [source](https://github.com/255BITS/ParticleGAN/blob/bfb06c2083cbc41c301797350cfb429776da59ee/reports/forge/bcap-physics/projection_ablation/round4/README.md) | Direction alone preserves both conditional repairs; prefer simpler measured option. |
| [#368](https://github.com/255BITS/ParticleGAN/pull/368) | [Round four](../bcap-physics/round4/README.md), [source](https://github.com/255BITS/ParticleGAN/blob/60944075e353834df313c74f1adf6324d94f2811/reports/forge/bcap-physics/projection_transport/round4/README.md) | Rare/broad passes survive; Gaussian regresses and conditional composition remains blocked. |
| [#373](https://github.com/255BITS/ParticleGAN/pull/373) | [Round five](../bcap-physics/round5/README.md), [source](https://github.com/255BITS/ParticleGAN/blob/fed122eed540a100a83d0b2614895f35d6c9e5c9/reports/forge/bcap-physics/conditional_integration/round5/README.md) | Combined recipe passes 6/7 scoped tasks; Gaussian retention still fails. |
| [#375](https://github.com/255BITS/ParticleGAN/pull/375) | [Round five](../bcap-physics/round5/README.md), [source](https://github.com/255BITS/ParticleGAN/blob/f5634547edf9c6eb3acd15d24750780027927a58/reports/forge/bcap-physics/mass_allocation/round5/README.md) | Balanced assignment repairs width but loses rare density and native quality; reject global replacement. |

Each round's comparison receipt retains source/runtime/protocol identities,
worker certificates, metrics, task/initializer/prior/named-stream proofs and
actual-training media. No additional preservation gap was found for these six
concluded comparisons. Their useful positive results remain evidence even when
the corresponding integration PR is closed.

The local copies of #397/#398 readouts now label their archived runtime scope
and link missing measurements, media and reproduction code to the exact original
commit. The original recall records' hashes still identify the unchanged original
reports; local archive notices and link repairs do not replace those identities.
