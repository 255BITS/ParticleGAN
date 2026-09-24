# PR84 follow-up: G critic reach driven by D's slope use and G's stall

Host: neural. Seed 0, one thread, PyTorch 2.14.0+cpu. The repository's parity pin
is `ATEN_CPU_CAPABILITY=avx2`; the AVX512 cold ring is a build check, not a second
seed. Receipts: [continuous-evidence/pr84-reach](continuous-evidence/pr84-reach/).
Runner: [gan_followup_probe.py](gan_followup_probe.py). Candidate:
[pr84_reach_candidate.py](pr84_reach_candidate.py).

## Mechanism (kept: stall reach)

PR84 reads the critic for G through a five-point stencil of width
`min(.15, .5/s)`, where `s` is D's RMS input slope on the clean particles. The
.15 cap binds on every update, so the width never adapts. With the host's b_cap
slope limit `κ = 1`:

- **reach .5:** width `max(.15, .5·min(s, 1/s))`. This is exactly PR84 for `s ≤ .3` (a covered, indistinguishable cloud).
- **stall reach:** as reach .5, except the width is .5 when the game has stalled. A stall is D near its slope limit (`s ≥ .6`) while G's own-curvature trust factor averages ≤ .1 over the last 50 updates.

D, both curvature bounds and the losses are unchanged.

**Purity:** GAN dynamics only. The trigger reads D's slope and G's own trust
factor; there is no coverage, likelihood, anchor, assignment, mode-count or clip
term. Mode centers are read only by offline diagnostics, after training.

## Why this formulation

Every failure pinned here has the same signature: D at its slope limit (`s` about
1.0–1.1, advantage .12–.17), G's trust factor at .06–.08, and G's read fixed at .15.
It appears in two places: the AVX512 cold ring that sticks at 7 modes, and PR84's
continued run after its collapse. In the covered state `s ≈ .15`.

At the stuck state the missing mode has the highest critic value. The nearest
particles point toward it only when G reads the critic at width ≥ .5. That makes
this the "idle once covered" critic channel #97 asked for.

Rejected alternatives:
- R1/R2 zero-centred pulls: already screened, and ruled out by the board.
- Optimism and ExtraAdam: both failed mode-hold.
- The one-step unroll: it does not reverse the outward field.
- Functional output metrics: they fail the trajectory gate.
- Rest-damping on slope: it damps acquisition.
- Barrier probes (#97) and the idle anti-gradient field (#101).
- Occupied-basin D curvature (#100), and #100's "shape D where the cloud is absent": D already points at the missing mode, so the gap is G's read.

## Gates (AVX2 pin unless noted)

| Gate | PR84 pin | Reach .5 | **Stall reach** | Reach 1.0 | Saturating ramp |
| --- | --- | --- | --- | --- | --- |
| Warm 1001–1200 | 196/200, min HQ .866 | 200/200, min HQ .921 | **200/200**, min HQ .921 | 199/200 | 199/200 |
| Cold trajectory | PASS | PASS | **PASS** | PASS | PASS |
| Cold ring (AVX2) | 8, terminal HQ .988–.999, suffix 5 | 8, terminal HQ .991–1.0, suffix 10 | **8, terminal HQ 1.0 ×5, suffix 8** | 5 | 6–7 |
| Cold ring (AVX512) | 7 | 7 | **8, terminal HQ .92–1.0, suffix 8** | 1 | 8 |
| Continued from own ring, 1210–2400 | 53/120, **final 6 / .904** | 103/120, final 8 / 1.0, suffix 33 | **97/120, final 8 / .971, suffix 25** | 18/120 | 36/120 |

The "suffix" is the number of consecutive passing checks at the end of the run.

**Harness warning for all GAN lanes.** Warm is valid only on AVX2. On AVX512 the scheduled warm prefix itself
reaches only 6 modes at update 1000 (identity fork 0/200, min HQ .935; receipt
`pin-warm-avx512`). A warm "regression" reported against a 6-mode baseline is
therefore measured from an unsolved state. #101's reported baseline matches this
AVX512 artifact exactly, and so do #102's "held 6" and #103's .935 baseline. Rerun those warm comparisons with `ATEN_CPU_CAPABILITY=avx2`.

**Stuck-state fork.** [pr84_reach_stuck_fork.py](pr84_reach_stuck_fork.py) forks
the AVX512 reach .5 state at update 1000 (7 modes) and continues it to 1200:

| Fork | Modes at 1050 / 1100 / 1150 / 1200 | G trust factor |
| --- | --- | --- |
| Reach .5 as-is (width about .35) | 7 / 7 / 4 / 7 | .10 |
| Width held at .5, bound on | 8 / 8 / 8 / 8 | .09 |
| Width .5, bound off | 0 / 0 / 2 / 1 | 1.0 |

The blocker was the width, not the trust region; the bound is protective. The
saturating ramp (.5 whenever `s ≥ .6`) shows that D's slope alone is a bad
trigger: it is also high in healthy acquisition, and the ramp just swaps which
build acquires. Adding G's own stall fixes that. In stall reach the width is .5
on 9–12% of updates outside the early acquisition window (42% during
updates 400–800).

## Rank claim (GAN-native track only)

Stall reach ranks first. It is the only GAN-native entry here with warm 200/200,
trajectory PASS, and cold ring 8 on both builds. Its continued constant-rate run
from its own ring ends back on the full ring. PR84 ends that run degenerate at
6 modes.

**This is not a clean stay.** Every variant, PR84 included, has an episode
between about 1720 and 2150. In stall reach, 23 of 120 checks fail, including
whole-cloud dropouts to 0 modes at 1770 and 1820. Each recovers within 10–30
updates, and there are no failures after 2150.

## Keep / kill / next bet

- **Keep** stall reach as the GAN-native reference. Reach .5 is the fallback, with a slightly better stay (103 vs 97) but no AVX512 ring.
- **Kill** reach 1.0 and the saturating ramp.
- **Kill** a global G curvature bound of .125 on stall reach. Warm improves to 200/200 with min HQ .961, but the cold ring fails: 2 modes on AVX2, and 7 modes at HQ .69 on AVX512. The continued run is 30/120 because it never acquired.

## The shared continuation dropout

Every variant, PR84 included, has an unstable episode between about 1720 and 2150.

**Per-update trace** ([pr84_reach_dropout_trace.py](pr84_reach_dropout_trace.py), 1755–1800, [receipt](continuous-evidence/pr84-reach/dropout-trace-stall-1755-1800.json)):
- Almost all of the cloud's motion is common translation, growing from .02 at 1756 to .91 at 1769, when the cloud reads 0 modes.
- G's own-curvature ratio falls from 5.0 to .9, so its trust factor opens from .05 to .27.
- D's slope stays .29–.43 while the translation grows, then spikes to 1.04 at 1770, where D's curvature ratio reaches 5.9.

**Translation is not specific to the dropout** ([trace 1–1760](continuous-evidence/pr84-reach/trace-stall-1-1760.json.gz)). It carries 56–83% of G's output motion in every phase: 83% in early acquisition, with jumps up to 2.4, and 56–63% in healthy holds. A translation bound would slow acquisition and act during holds, and its only distinguishing feature, magnitude, is clip-ladder territory. That bet was killed offline.

**Timescale fork at 1755** ([pr84_reach_lag_fork.py](pr84_reach_lag_fork.py), [receipt](continuous-evidence/pr84-reach/lag-fork-1755/)). Clean-support checks from 1760 to 1900:

| Fork | Passing | Min modes |
| --- | --- | --- |
| As-is | 4/29 | 0 |
| D steps ×2 | 0/29 | 0 |
| G steps ×.5 | 28/29 | 7 |

A faster critic worsens the dropout, so it is not caused by D lagging G. It is a cross-coupled mode that grows with G's step. Slowing G globally removes it but also kills acquisition (the .125 kill above). This is #60's trade-off, reproduced inside the game.

### Game bound (tested, partial)

G's trust bound uses `max(ρ_own, ρ_game)`. Here `ρ_game` is measured after one virtual D Adam step that answers G's proposal. D's parameters and Adam state are restored afterwards, so D's actual step is unchanged. This adds a fourth replay per update. From the 1755 fork it scores 22/29, against 4/29 as-is and 28/29 with G ×.5.

| Gate | Stall reach | Stall reach + game bound |
| --- | --- | --- |
| Warm (AVX2) | 200/200, min HQ .921 | 200/200, min HQ .915 |
| Cold trajectory | PASS | PASS |
| Cold ring AVX2 / AVX512 | 8 / 8 (passing observations 10 and 8 of 24) | 8 / 8 (5 and 7 of 24) |
| Continued run 1210–2400 | 97/120, min 0 modes, 7 severe dips (≤4 modes), last 25 checks pass | 96/120, min 2 modes, 2 severe dips, last 6 checks pass (dip at 2340) |
| Cold ring time (AVX2) | 28 s | 37 s |

The game bound softens the dropouts but does not reduce how many checks fail, and it thins the acquisition margin. **Partial: not promoted.**

- **Original bet (now tested above):** bound G by the game's cross-curvature instead of its own-curvature alone. That means measuring how D's response to G's last step changes G's gradient. This mode grows with the D×G step product, and G's own-curvature ratio falls while it grows, so an own-curvature bound cannot see it. Keep the stall-reach read and test on the 1755 fork first. The PR82-era cross-curvature bounds predate the reach channel, so their ring failures do not settle this. Do not use a slope- or advantage-gated G rest damping: PR84's rest-damping and #104 already kill that family.
