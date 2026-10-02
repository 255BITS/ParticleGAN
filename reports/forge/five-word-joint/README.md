# Five-word joint BiGAN acquisition

The [registered task](../../../configs/forge/tasks/five_word_joint_acquisition.json)
asks whether a generator, encoder and joint critic can learn five equally likely
words and the correctly paired inverse mapping. It preserves the actual question
from [`examples/five_modes.py`](../../../examples/five_modes.py): `apple_`,
`grape_`, `lemon_`, `melon_`, `berry_`, each represented by six positions over
28 characters, including underscore padding. The source calls this BiGAN-style
joint adversarial learning, also known as ALI; it is not five Gaussian clusters.
The question covers this finite vocabulary, not language generation or unseen
word/typo generalization.

## Review the evidence

- [Current experiment guide](../EXPERIMENTS_BY_TIER.md#experiment-five-word-joint):
  task/view policy, original API result, actual-training GIFs and compact evidence.
- [Original-example convergence readout](../../toy_audit/SOURCE_DEMOS.md) and
  [source-example GIF](../../toy_audit/media/source-five-modes.gif): full-budget
  clean/live and EMA PASS under their original source protocol.
- [Public-API word-host explanation](../../toy_audit/api_contract/images/README.md)
  and [full API training GIF](../../toy_audit/api_contract/media/image-five-words-joint-ae.gif):
  a separate 20,001-update KA2 CPU PASS. Its [compact published receipt](../../toy_audit/api_contract/runs.json)
  records five modes, quality fraction 1.0, TV 0.02890625, exact paired
  reconstruction and minimum correct reconstruction-token probability 0.999753.
- [Frozen short-demo protocol](demo-protocol.json): one 32-update API integration
  check, 60 seconds maximum, seed 0, 1,024 samples, 24 post-update observations.
  Its results are published separately after execution, with no full-task credit.

The historical/source and API results keep their original source, recipe, prior,
initialization, sampling, budgets and runtime. They do not qualify this new Forge
task. The [current solution leaderboard](../technique-inventory.md) remains
unchanged; this view has no qualified winner.

## Why it was missing from Forge

The [original question catalog](../../toy_audit/catalog.json) retains this example
as `source-family-15`, originally a source-reviewed question. The later source
and [API migration](../../toy_audit/api_contract/cases.json) added runnable
training, strengthened numerical gates and actual-state GIFs. That migration did
not register a Forge task or qualification-view assignment.

Forge's [image adapter](../../../experiments/forge/adapters.py) builds scalar
image GAN hosts through `GANTrainer`; it has no inverse encoder or joint
`D(x,z)` objective. The existing [ae_gan_hold](../../../configs/forge/tasks/ae_gan_hold.json)
is a different 250-update behavioral test with reconstruction-MSE/hold gates.
Neither supplies the five-word mass, confidence or paired-correctness evidence.
There was also no explicit retained-question join from a Forge host name to
`source-family-15`, so the generated tier report could not find its explanation
and GIF. These are registration, host-binding and evidence-identity gaps; the
original example was not omitted because it failed to learn the five words.

## What this registration contains

The [shared WordFixture](../../../benchmarks/toy_audit/api_images.py) retains the
source-shaped networks: G 2→64→128→168, E 168→128→64→2, and joint D
170→256→128→1. It distinguishes real `(x,E(x))` pairs from fake `(G(z),z)`
pairs. Its relativistic generator/encoder objective reverses those pair scores;
there is no reconstruction lookup or auxiliary reconstruction loss. Prior
spread regularization retains useful code separation.

The [Forge binding](../../../experiments/forge/word_adapter.py) injects the
candidate's public recipe, five learned 2D particle rows and named initialization,
data, prior, input/output-noise and evaluation streams into that existing host.
The finite vocabulary gets an explicit `particle_cloud`, sigma-zero exception;
the new Forge prior is nonstandardized. A learned-MoG variant needs its own task
identity and calibrated measurement. Observations use actual clean/live public
sampling and reconstruct each canonical input in its correctly paired row.
EMA/controller serving and independent-row controls are blocked before
reservation; incompatible resources/objectives/priors and supplied initialization
are rejected instead of silently dropping mechanisms.

The [source-shaped KA2 candidate](../../../configs/forge/ideas/five-word-joint-ka2-v1.json)
uses LR 0.0006, D multiplier 1.5, prior spread 1.0, Adam `(0,0.999)`, EMA decay
0.995 and zero training input/output noise. The original 20,000-update recipe
schedule is preserved while the actual training allowance is 20,001 updates,
matching the source's inclusive loop. Its explicit `host_adaptation` delegates
`prior_reg` to each existing behavioral prerequisite's own objective; the joint
word task keeps candidate spread 1.0. This preserves host ownership without
discarding a formulation mechanism. No production recipe or default changes.

| Numerical condition at every terminal observation | Bound |
| --- | --- |
| Generated samples | 1,024 |
| Confident canonical-word fraction | ≥ 0.95; each of all six tokens has probability ≥ 0.90 |
| Accepted modes | Exactly 5 |
| Word/rejection total variation from uniform five-word law | ≤ 0.10 |
| Correctly paired six-token reconstruction | Exact for all five inputs |
| Minimum correct reconstruction-token probability | ≥ 0.90 |

The shared [five_word_metrics / score_words](../../../benchmarks/toy_audit/definition_quality.py)
scorer is checked against oracle samples, correct-argmax low confidence, collapse,
unbalanced mass, wrong padding and swapped reconstruction. A perfect generated
marginal with wrong reconstruction pairs fails. The existing transfer grader
requires all 24 observations and five consecutive passing terminal checks.
Finite state, G/E/prior/D optimizer updates, mechanism activation and RNG
isolation also remain required; an endpoint or PASS stamp cannot qualify a run.

The new [five_word_joint view](../../../configs/forge/views/five_word_joint.json)
keeps the three existing mechanism smoke prerequisites at Tier 1 and places word
acquisition provisionally in Tier 2. A 20,001-update quality budget is too costly
to assert cheap-smoke placement without calibration. Existing stability views,
leaderboards, calibration and search contracts stay frozen. A scientific tier
study should preregister positive/negative references, false rejections and
cost-to-rejection, and change only view policy after collecting evidence.

## Reproduce and review

Use the project Python environment. The first three commands are read-only:

```sh
python -m experiments.forge validate
python -m experiments.forge plan five-word-joint-ka2-v1 --through-tier 2 --device cpu
python -m experiments.forge experiments-by-tier --output reports/forge/EXPERIMENTS_BY_TIER.md
```

The sole short integration demo is explicitly outside ordinary qualification:

```sh
python reports/forge/five-word-joint/reproduce_demo.py --output runs/forge/five-word-joint-demo-v1 > runs/forge/five-word-joint-demo-v1.log 2>&1
tail -F runs/forge/five-word-joint-demo-v1.log
```

It renders actual targets, generated strings and paired reconstructions, showing
numerical status at real updates. Reduced-budget metrics can fail at 32 updates;
the evidence is always **INCOMPLETE against the full 20,001-update task**.
Bulk stdout, observations and state remain under ignored `runs/forge`; publish
only the compact receipt and final GIF. Output directories are exclusive.

For a later explicitly requested ordinary run, the
[1,800-second campaign](../../../configs/forge/campaigns/five-word-joint-v1.json)
reserves three 300-second smoke allowances and one 900-second word allowance:

```sh
python -m experiments.forge run five-word-joint-ka2-v1 --through-tier 2 --gpus cpu --campaign configs/forge/campaigns/five-word-joint-v1.json
python -m experiments.forge logs --follow --candidate five-word-joint-ka2-v1
python -m experiments.forge board --goal five_word_joint
```

Ordinary failures stop later tiers. No full-budget current Forge word run has
been launched, no smoke prerequisite has been bypassed for qualification, and
no historical pass has been transferred into the new leaderboard. Review this
registration and its controls before deciding whether to run, revise or retier
the task. A release choice still needs full Forge qualification.
