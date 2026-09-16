# Decisions from completed runs

No seed sweeps or image-based decisions. Fractions below use the first 128 table
rows, fixed particles, zero initial memory, and original circle thresholds.

## First completed pair: learned vs frozen GRU (2,000 updates)

Learned GRU is a candidate for continuation: 14.1% full-256 passes versus 6.3%
frozen, 1.6% versus 7.8% late stopping, and full passing direction counts 5 CW /
13 CCW versus 8 CW / 0 CCW. Full-1,024 passes are 15.6% versus 7.0%.

Learned GRU center spread .609 (real .587), radius std .203 (real .233), speed
std .084 (real .079); initial spread 1.135 (real 1.180). Coverage is substantially
better than the historical frozen baseline, but geometry is still poor.

Overlapping full-256 failure counts for learned GRU: 109 radial, 13 drift,
9 direction consistency, 5 speed range, 3 radius range. Prefix pass rates are
14.8% at 64 and 128 steps versus 14.1% at 256; thus simply extending the horizon
is unlikely to be a complete solution. The 32-step diagnostic fits only the
first 16 points by its existing convention, so it is not directly comparable.

Frozen GRU was not selected for continuation. Planned promotion: exact continuation
to 5,000 updates first, followed by completed-run evaluation and a further
decision about the preconfigured 10,000-update schedule.

## Completed first wave

Full-256 pass rates: trace + multiscale + differences 38.3%, trace + multiscale
33.6%, delay + multiscale 32.8%, learned GRU 14.1%, frozen GRU 6.3%, matched
trace + flattened critic 3.1%. All three multiscale models' full passing circles
are clockwise. Their late-only passes are much higher (85.2%, 92.2%, 89.1%),
showing that startup and consistency with the early fitted circle remain problems.

Promoted **trace_multi_diff** for geometry and **gru_flat** for direction coverage,
each from 2,000 to 5,000 updates. The small 38.3% vs 33.6% gap is not claimed as
statistical superiority: difference features also give better initial/radius
spread, motivating this exploratory choice. Both retain their original planned
10,000-update LR schedule and exact optimizer/RNG continuation.

Second-wave new scouts: **gru_multi** combines useful mechanisms; **trace_multi_film**
lets each fixed particle modulate G's hidden layers to make particle-dependent
motion easier to express; **gru_silu** tests smooth G activations against radial
error. FiLM begins at zero modulation and initially reproduces its baseline G.
Writer-rate and invariant-geometry critic options are implemented but not yet run.

No plots were generated or inspected for these decisions. Source snapshots differ
as disabled future options were added; compatibility checking against all six
launch snapshots confirms identical initialization, rollouts and critic outputs
for the original configs on CPU.

## Completed 5,000-update continuations

**gru_flat** improves from 14.1% to **34.4%** full-256 circles, reduces radial
RMSE from .237 to .149, stops no trajectories, and has **25 CW / 19 CCW** full
passes. Radius std .220 is close to real .233, initial spread 1.122 vs real
1.180. Select this model for **10,000 total updates**, keeping its original
schedule and optimizer/random states; queue it after the GRU-multiscale scout
on GPU1.

**trace_multi_diff** trades geometry for some direction coverage: 38.3% becomes
**28.9%**, radial RMSE .164 becomes .221, but full passes now include **33 CW /
4 CCW**, and no trajectories stop. Preserve both checkpoints; do not label
this continuation an overall geometry improvement. A targeted follow-up is to
freeze its learned writer at update 2,000 and continue G/D-head training to
5,000, testing whether changing memory representations harm cold-start consistency.
That config is prepared but not yet launched.

## All-particle deployment check for the 5k GRU

CPU evaluation over all 512 learned rows (no scoring head constructed) gives
35.2% full-256 passes and 33.6% full-1,024, with 101 CW / 79 CCW full-256 passes.
Late stopping is 1/512. Zero-memory intervention is completely stationary;
shuffling memory gives zero circle passes. Real references here use CPU RNG and
512 rows, so they are distribution references rather than the identical arrays
in the 128-row GPU studies. This is enumeration of the learned table, not an
unseen-data test split. CPU and saved CUDA trajectories differ by at most
7.2e-6 through 64 steps, 8.4e-5 through 256, and .000917 through 1,024.

## Completed FiLM scout

Particle-conditioned hidden modulation improves trace/multiscale full passes
from 33.6% to 50.8%, with 98.4% late-only passes. All 65 full passes are CCW,
so it has not solved direction coverage. Preserve it as a geometry candidate;
the learned GRU remains the preferred balanced model. GRU-multiscale and GRU-SiLU
are still running. A GRU with a learned geometry head is queued on GPU0 after
SiLU; it adds pairwise distances and centered signed cross-products to D, not
an analytic circle generator or geometry loss.

## Remaining second-wave outcomes

**gru_multi** reaches 44.5% full passes, .146 radial RMSE, no stopping, and
radius std .235 (real .233), but all 57 full passes are CCW. Do not prefer it
over the balanced 5k GRU solely on pass rate.

**gru_silu** gives zero full or late-only passes at 2,000 updates, radial RMSE
.245 and narrow radius variation (std .108). Reject this scout for extension.
This is evidence about this configuration/budget, not a general claim about SiLU.

The 10k balanced GRU continuation and the 2k GRU geometry-head scout are running.
Geometry features are squared pairwise distances and cross-products of centered
points (each scaled by 1/4). A learned MLP scores those features, averaged equally
with the original flattened-path critic score before the same GAN/B-cap losses.
The writer continues to train through the original memory-containing score.

## Geometry scout and 10k GRU

The geometry-head scout reaches **27.3%** full passes at 2k vs the matched
GRU baseline's 14.1%, with radial RMSE **.150 vs .237** and 4 CW / 31 CCW
passes. Promote it to 5k, preserving schedule and optimizer/RNG states.

The original GRU reaches **53.1%** full passes at 10k (31 CW / 37 CCW), radial
RMSE .127. Its full-table deployment evaluation confirms **52.5%** at 256 and
**50.4%** at 1,024 over all 512 particles. Passing directions: 134 CW / 135 CCW.
Both memory interventions still give zero passes. It is the best overall
completed model, but remains far below the working success target.

Launch a controlled writer-freeze branch: restore the same GRU 5k checkpoint,
freeze its learned writer, and continue G, prior and scoring head through 10k
on the same original schedule/data stream. This isolates continued writer
adaptation from holding an already useful representation fixed. It is an
intentional training-rule change, not an exact continuation of the unfrozen run.

## Freeze and private-recurrence results

Freezing the learned GRU writer at 5k gives **39.8%** full passes at 10k vs
**53.1%** with continued writer learning; radial RMSE .141 vs .127. Passing
directions remain mixed (31 CW / 20 CCW). Keep the writer trainable. This does
not support the hypothesis that continued writer adaptation is the main problem.

The generator with its own GRU reaches **55.5%** full passes at 2k and 54.7% at
1,024 steps, but all passing trajectories are CCW. It still depends on M:
zero/shuffle interventions both give zero passes; zero memory stops 90.6% late.
Promote to 5k and run the parameter-matched recurrent G with memory reading
disabled for 2k. That control keeps D's learned writer and all other settings;
only G's access to M changes. Private G recurrence can move without M, so this
is a meaningful control unlike the old feedforward static no-memory variant.

The round will conclude with these two jobs and final full-table evaluations.
Full-table diagnostics for the 10k GRU show 70.5% passing over 64 points versus
52.5% over 256. Radial error accounts for 243/512 failures; only 2 fail radius
range, 2 speed range, and 5 direction consistency (criteria overlap). This gives
a quantitative rationale for a future longer-horizon/curriculum experiment,
preserving scorer capacity and initialization when possible. Marginal coverage
is already close: radius std .226, speed std .084, center spread .618.

## Matched private-recurrence control

At 2k, G with private recurrence but no memory reading reaches **52.3%** full-256
passes and 53.1% full-1,024 (66 CW / 1 CCW). With memory reading, the same
architecture reaches 55.5% and 54.7% (all passing CCW). The modest raw pass
difference does not establish a shared-memory advantage. The with-memory model's
zero/shuffle interventions still show dependence on M, which is distinct from
showing it is better than a separately trained no-reading control. All G, D,
writer and prior initialization tensors were verified identical between controls.
Neither model at 2k resolves direction collapse. Wait for the with-memory 5k
continuation, then finish full-table validation and the round report.

## Final result and closeout

The private-recurrent G reaches **85.2%** full-256 passes at 5k (109/128),
84.4% at 1,024, and no late stopping in the scout subset. Full-table CPU
validation confirms **85.2% at 256 and 84.0% at 1,024**; all 436 full-256
passing trajectories are CCW. One of 512 trajectories stops late. Zero/shuffle
memory again give zero passes; zero memory stops 95.3% late.

Do not declare the toy solved: this model misses direction coverage, while
the balanced model remains at 52.5% full-table passes. All 18 training jobs
(12 scouts, 6 continuations; 46,000 new updates) completed successfully.
No jobs remain running or queued. Full-table results, configs, manifests,
environment and source hashes are saved in the round report.

Prepared **unrun** next combinations are private G + FiLM and private G + the
oriented geometry critic. These target direction coverage while retaining the
successful recurrent dynamics. The balanced-model longer-horizon study remains
another grounded next step. The 5k recurrent no-reading control is also still
unrun, so do not claim a 5k shared-memory advantage.
