# Gaussian response magnitude: three independent investigations

This round follows the normalized-field extrapolation study in PR
[#317](https://github.com/255BITS/ParticleGAN/pull/317). The user requested three
subagents, independent worktrees and separate PRs. The common question is whether
a constant-rate learner can acquire the scalar Gaussian, retain live quality
during continued updates, and respond again when the target changes.

**None of the three independent changes solves continuous Gaussian learning.**
Network magnitude suppression with past extrapolation gives the largest gain:
41/72 stationary retention checks and8/24 shifted retention checks pass, versus
3/72 and0/24 for the original normalized-field extrapolation. It still misses
acquisition and full retention. All27 new combined gates fail. The three executed
protocols retain their own evidence identities and costs. No diagnostic result
supplies ordinary qualification.
The [single current technique leaderboard](../technique-inventory.md) remains the
qualification publication for this goal.

## Common contract

All three investigations retain the selected BCAP configuration, batch 128,
protocol seed 0, public deterministic initialization, named checkpointed streams
and CUDA training/model sampling. Nominal rates stay G .012, D .018, prior .03;
both LR floors are 1. There is no elapsed-step annealing, averaged serving,
best-checkpoint selection or history reset.

The diagnostic prior is 256 uniform MoG locations, fixed sigma .1, initialization
scale 1 and no standardization. Gaussian retains z=2, MLP width32/depth2 and
Fourier-2 critic, with target N(2,.5²). Ring retains its original z=4,
width64/depth2 and sixteen-component target. The ordinary Gaussian sigma-.025
task is a separate evidence cohort and remains unchanged.

Each study compares alternating, simultaneous and extrapolation-from-the-past
timing, using one complete configuration across both tasks. Stationary training
continues to 4,000 updates. Acquisition requires five terminal full passes at
1,000 Gaussian / 1,600 ring updates. Retention requires **all** 72 Gaussian / 144
ring checks after acquisition. The Gaussian gate retains mean error <=.2 sigma,
std ratio [.8,1.2], KS <=.05, at least4,096 samples and finite fraction1.

Each Gaussian arm then continues from its own exact4,000-update checkpoint when
the target mean changes2→3, with sigma.5 unchanged. It must reacquire by5,000
with five terminal passes and retain all24 checks through6,000. An independent
no-update copy receives matched public evaluation draws. Shift diagnostics run
even after stationary failure, which cannot become a continuous-learner pass.

Real batch sequences and initial tensors are checked against original archived
controls. Existing learned-prior controls are reused under their original
source/initialization/prior identity, with zero new training cost.

## Independent changes and reservations

| Investigation | Explicit delta | New updates | Reserved loop seconds |
| --- | --- | ---: | ---: |
| Frozen prior | Freeze the original initial locations before optimizer construction; G/D still learn. Separate prior-learnability control cohort. |30,000|4,500|
| Prior magnitude | Preserve G/D dualnorm; sampled prior row direction g/max(norm(g),.001). Prior remains learned. |30,000|4,500|
| Network magnitude | Preserve learned-prior row normalization; G/D matrix direction U diag(min(s/.1,1)) Vᵀ with existing aspect factor, vectors g/max(norm(g),.1). |30,000|4,500|

Every study freezes six stationary trials and three shift continuations, zero
scientific retries. Total reservation is90,000 new updates and13,500 loop
seconds. Each new scale is a predeclared constant, shared across tasks, rather
than a value chosen from successful trained states. Initial-gradient audits
are explicit separate cohorts with zero parameter updates.

One scientific worker runs per RTX A6000. Frozen-prior and prior-magnitude
studies use GPUs0 and1 respectively; the network study waits for a released GPU.
GPU time here is accounting, not a speed ranking. Agents do not mutate shared
selections, qualification snapshots or the current leaderboard.

## Interpretation

Prior row normalization makes most nonzero sampled rows move nearly.03 at every
update, even if their gradient is small. The prior cap retains the old maximum
motion while allowing smaller forces to make smaller moves. The initialization
probe at the fixed.001 scale finds mean implied moves approximately.00298 on
Gaussian and.00157 on ring, compared with the old nearly.03 moves.

Network polar normalization also discards singular-value magnitude. For a
rank-deficient nonzero gradient, its unit completion can move directions whose
gradient singular values are zero or tiny. The spectral cap keeps those small
directions small. Its correction and cached lookahead use the same field.

These are role-isolation experiments. Failure of freezing alone would not prove
that prior motion is irrelevant; failure of either independent cap would not
reject their interaction or every fixed scale. The whole live acquisition,
retention and adaptation protocol determines repair success. Endpoint snapshots
and moments alone cannot establish that the Gaussian law stays learned.

## Completed readouts

The independent PRs target develop and declare PR317 as their prerequisite:

- [PR318: frozen prior](https://github.com/255BITS/ParticleGAN/pull/318),
  [complete report](https://github.com/255BITS/ParticleGAN/blob/dcbcd3ed72b0b3d19bc2f4d52443574dc44d3d4e/reports/forge/gaussian-frozen-prior/README.md).
- [PR319: magnitude-sensitive learned prior](https://github.com/255BITS/ParticleGAN/pull/319),
  [complete report](https://github.com/255BITS/ParticleGAN/blob/51b8794dd06db0cd6eb25e90ea9d2f0d48c77dbc/reports/forge/gaussian-prior-magnitude/README.md).
- [PR320: magnitude-sensitive networks](https://github.com/255BITS/ParticleGAN/pull/320),
  [complete report](https://github.com/255BITS/ParticleGAN/blob/c4627fb14619ca0bcd1b8d904e7796d0834a4e8a/reports/forge/gaussian-network-magnitude/README.md).

These are diagnostic comparisons of explicitly different role/update cohorts,
not a second qualification leaderboard. Every Gaussian acquisition verdict below
is FAIL. Every shifted reacquisition verdict is also FAIL. A whole live learner
must pass acquisition and every retention check in both phases.

| Cohort | Timing | Stationary hold /72 | Longest stationary streak | Final stationary KS | Shift hold /24 | Final shifted KS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original learned prior | Alternating |1|1|.09594|0|.20058|
| Original learned prior | Simultaneous |0|0|.29329|0|.35284|
| Original learned prior | Past extrapolation |3|2|.26799|0|.13998|
| Frozen initial prior | Alternating |3|2|.11345|0|.17680|
| Frozen initial prior | Simultaneous |0|0|.14677|0|.47287|
| Frozen initial prior | Past extrapolation |7|1|.07294|1|.03256|
| Magnitude-sensitive prior | Alternating |2|1|.03028|1|.08544|
| Magnitude-sensitive prior | Simultaneous |0|0|.20625|0|.35390|
| Magnitude-sensitive prior | Past extrapolation |10|2|.19872|3|.11994|
| Magnitude-sensitive networks | Alternating |11|2|.04838|0|.07497|
| Magnitude-sensitive networks | Simultaneous |0|0|.13628|0|.23232|
| Magnitude-sensitive networks | Past extrapolation |41|5|.06083|8|.03837|

The network-cap past arm has43/96 full stationary passes. Its first five-pass
window ends at update3,875, well after the1,000-update acquisition deadline.
Its final stationary mean error is.01287 sigma and std ratio.99263, but KS.06083
fails the full distribution gate. After the shift,17/48 full checks pass, with
longest streak3; final mean error.03319 sigma, std ratio.92232 and KS.03837 pass
the endpoint. Retention remains8/24. The frozen-prior past shift also passes its
endpoint, yet retains only1/24 checks. Neither endpoint establishes a solution.

The prior-cap alternating arm passes its final stationary endpoint with KS.03028
but retains only2/72 checks, with longest streak1. This repeats the original
problem: a good snapshot does not persist. The fixed.001 threshold reduces
initial row motion approximately10–20×, but does not make it uniformly tiny:
the final past Gaussian cache implies mean row displacement.02014, with37.11%
of rows saturated. This rejects one fixed-scale revision, not every possible
learned-prior force response.

### Ring regression

| Cohort | Alternating acquisition / hold | Simultaneous acquisition / hold | Past acquisition / hold |
| --- | --- | --- | --- |
| Original learned prior |PASS /144 of144|PASS /75 of144|FAIL /115 of144|
| Frozen initial prior |FAIL /0 of144|FAIL /0 of144|FAIL /0 of144|
| Magnitude-sensitive prior |PASS /142 of144|PASS /46 of144|FAIL /0 of144|
| Magnitude-sensitive networks |FAIL /0 of144|FAIL /0 of144|FAIL /0 of144|

Keep the adopted alternating ring recipe. Freezing loses its cluster precision;
all frozen arms cover16 modes but fail quality/spread bounds. Network caps also
fail all240 full ring checks in every timing arm, with endpoint component
covariance errors approximately2.02/1.85/2.06. The prior-cap alternating arm
comes closest but fails mode coverage at3,884 and the minimum eigenvalue ratio
at4,000. Its final eigenvalue ratio.14670 misses the.15 floor.

### What to try next

Test the **interaction of network and prior magnitude suppression**, starting
with past extrapolation. The independent results give a reason: reducing network
motion makes Gaussian considerably more faithful, while its prior still receives
nearly unit row directions; reducing prior motion alone leaves the unstable
normalized G/D field in place. Their combined effect is untested, and improvement
cannot be assumed. A fixed-prior plus network-cap control could distinguish that
interaction from the learned-prior response.

At the network-cap past Gaussian endpoint, the saved G/D hidden-matrix directions
use2.84%/7.23% of their original cap norms, but97 active prior rows still imply
mean displacement.02997. At shift end those figures are2.99%/8.12% and.02999.
These are different parameter units, so their ratio does not measure causal
importance. They verify that this experiment calms network motion while leaving
the prior's nearly unit response in place.

Freeze the combined mechanism, gradient units, finite budget and matched protocol
as a new study before any spend. Initial gradient scales differ substantially
between the Gaussian and ring: the network-cap initialization audit finds mean
singular direction fractions.04383 for Gaussian G hidden weights and.00682 for
ring G hidden weights. A shared absolute cap can suppress useful ring acquisition
motion much more. One global trainer rule across tasks remains necessary; a
task-specific winning-checkpoint scale is not a fair replacement.

Do not add a position spring yet. The controlled evidence points first to response
magnitude and its interaction, while an anchor would introduce another force
before either is resolved. There is no justification here for a larger latent
dimension or network: the earlier unchanged-network affine representation control
already demonstrates capacity for this Gaussian law. More unchanged updates,
larger batches and normalized-field extrapolation alone have already failed.

No combined-cap trial, scale search, further continuation or promotion was run
in this round. The three authorized investigations are complete.

## Cost, verification and publication

| Investigation | New loop seconds | GPU | Software checks | Exact final restores | Saved sample metric recomputations |
| --- | ---: | --- | ---: | ---: | ---: |
| Frozen prior |337.193384|0|8|9|1,308|
| Prior magnitude |368.727114|1|69|9|1,308|
| Network magnitude |362.226751|0|67|9|1,308|

Total:90,000 new updates,11,520,000 real training examples and1,068.147249 measured
loop seconds, within the13,500-second reservation. There are zero scientific
retries. Loop time includes scheduled evaluations; setup, restore, serialization
and publication are separate. These are accounting totals, not elapsed wall time
or a speed comparison. Software counts include GPU fixtures and, on the cap
branches, metadata-only checks; plaintext/numerical metadata checks contain no
CPU neural execution.

All27 final CUDA contexts restore exactly, all3,924 saved metric sets reproduce,
and the27 actual-training GIFs contain9 saved GPU observation frames each. Initial
models and real-batch digests match the original controls with declared role
deltas. Every consumed stream is checkpointed. Original CPU target generation,
stored-output numerical scoring and rendering are explicit exceptions to CUDA
neural execution. The original scorer oracle/destructive controls are reused
under their frozen identity because scoring and bounds remain unchanged.

Each PR contains its own report, protocol, source-bound compact results,
reproduction code, provenance and actual-training GIFs. Bulk curves, stdout,
states and sample arrays remain in ignored local runs and immutable archives.
The existing PR317 receives this synthesis and a display-only context record;
that record launches no work and adds no duplicate cost or qualification.

Publication links and exact source/archive identities are recorded in
[results.json](results.json). The new PRs depend on the still-open PR317. Merge
the shared prerequisite before its dependent PRs; merging the independent cap
capabilities requires combining their small Recipe/optimizer/Forge additions and
regenerating derived Forge publications. A merge does not require repeating any
unchanged scientific trial.

The compact comparison is reproducible from the three committed publications,
without model execution or scoring their samples again:

```sh
/usr/bin/python reports/forge/gaussian-response-round/collect.py \
  --frozen /home/martyn/dev/ParticleGAN-gaussian-frozen-prior/reports/forge/gaussian-frozen-prior \
  --frozen-ref dcbcd3ed72b0b3d19bc2f4d52443574dc44d3d4e \
  --prior /home/martyn/dev/ParticleGAN-gaussian-prior-magnitude/reports/forge/gaussian-prior-magnitude \
  --prior-ref 51b8794dd06db0cd6eb25e90ea9d2f0d48c77dbc \
  --network /home/martyn/dev/ParticleGAN-gaussian-network-magnitude/reports/forge/gaussian-network-magnitude \
  --network-ref c4627fb14619ca0bcd1b8d904e7796d0834a4e8a \
  --output reports/forge/gaussian-response-round/results.json
```

The projection checks each declared9-cell/30,000-update completion and matching
real-batch digests across the three studies, preserving original gate outcomes.
It debits zero new training and marks referenced costs against their original
records to prevent double-counting.
