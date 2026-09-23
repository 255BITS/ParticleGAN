# Fast Lunar lander: full command evidence

The complete command ran from an empty output directory in about **72 seconds**
on CPU. It collected experts and simulator branches, trained dynamics and the
policies, extracted faster trajectories, selected on validation, and exported
the real-flight demos below. Both landing and speed gates passed. The declared
environment is the bidirectional-main-thrust Box2D variant, not stock Lunar.

| Learned controller | Validation landings | Validation success steps | Regression landings | Regression success steps |
| --- | ---: | ---: | ---: | ---: |
| Slow | 20/20 | 210.35 | 30/30 | 219.83 |
| Fast, selected round 3 | 20/20 | 184.35 | 30/30 | 186.77 |

Fast won **30/30 paired regression flights**, with **1.177× paired speedup**
(15.0% fewer steps) and **32 median steps saved**. No flight in these evaluation
cohorts crashed, flew away, or ended with an incomplete landing. This is a
finite-cohort result, not a universal landing guarantee. Validation selected
`fast_round_3.pt` from three candidates; the selected checkpoint is `fast.pt`.

![Learned slow and fast flights on seed 94000 at the same playback rate](comparison.gif)

The GIF uses the first matched regression world in fixed order: **94000**,
slow **225** steps and fast **173**. Both are actual learned-policy rollouts.
Playback uses 50 simulation steps/s with one rendered frame every three steps;
the finished side is held. The indicator marks active downward thrust.

- [Slow GIF](slow.gif), [fast GIF](fast.gif), [paired GIF](comparison.gif)
- [Standalone demo page](index.html): download this folder and open locally.
- [Report](report.json), [configuration/cohorts](config.json),
  [slow validation](slow_validation.json), [slow training](slow_metrics.json),
  [validation candidates](validation.json), [world-model errors](world_metrics.json),
  [run log](run.log), [artifact hashes](manifest.json)
- [World model](world.pt), [slow policy](slow.pt), [fast policy](fast.pt)

Scoring version `box2d-active-ground-contacts-v1` requires an enabled, touching
Box2D terrain contact on each leg, a sleeping lander centered on the pad, and
no crash. Reports retain both actual contacts and Gym's cached observation
flags so the decision can be audited. Physics and policy inputs are unchanged.

The previous calibrated checkpoints were falsely scored 28/30 slow and 29/30
fast because Gym can clear a leg's cached flag while another ground contact
remains. The [frozen-checkpoint audit](audit/contact_correction/README.md)
replays those same weights and scores **30/30 each**, with 1.222× paired
speedup. That is distinct from the **newly trained result above**: correcting
expert labels admits all 96 expert flights per controller and 96 faster pairs,
giving 21,727 slow and 18,162 fast imitation transitions.

The 94000–94029 cohort was inspected during that diagnosis. This full rerun
uses it as a fixed regression cohort, not a new untouched test. It holds the
training recipe and cohorts fixed; no hyperparameter or seed search was used.

This run uses clean source revision `194fe91`, based on `develop` `3d08a3a`.
Dynamics training uses 31,671 expert transitions and 5,472 exact-replay branches
from world-training episodes only. Slow policy training uses 8,000 cloning
plus 1,200 RpGAN updates; each fast round uses 8,000 plus 400. Every checkpoint
and flight uses **live weights, with EMA disabled**. Historical long-budget
action drift remains documented in the [training rounds](../../docs/lunar-training-rounds.md).

The compact evidence folder retains selected checkpoints and media. The full
local output also contains every candidate, dataset, source snapshot, and
structured log; `report.json` hashes identify these regenerable artifacts.

```bash
uv run --extra lunar python -u examples/fast_lander.py --out results/gym/fast_lander
```

See the [guide](../../docs/fast-lander.md),
[failure analysis](../../docs/lunar-failure-analysis.md), and
[consolidated PR map](../../docs/lunar-consolidation.md). The final controller
performs real RpGAN updates through a frozen learned world model after an
explicit cloning initialization. This establishes its measured behavior, not
an isolated benefit of RpGAN over cloning.

Local validation: **725 passed, 5 skipped, 1 expected failure**, plus 27 passing
subtests. The expected failure is the legacy native toy already recorded on
`develop`; four skips need opt-in CUDA data and one needs optional local
`torch-fidelity`. The complete command and separate smoke export both passed.
