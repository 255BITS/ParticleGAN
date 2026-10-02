# Public-API toy tests

Each executable toy variant has a scientific goal, public ParticleGAN execution,
a declared numerical PASS/FAIL gate, and a GIF comparing its desired behavior
with actual outputs over training. The historical catalog, training receipts,
ratings and GIFs retain their original identities. New variants link back to
their original questions and explain every change of scope, architecture,
resource, recipe or sampling law.

The [completed full-budget readout](RUN_REPORT.md) covers all 176 variants:
51 PASS, 120 completed FAIL and five ERROR/FAIL attempts. Start with the
[sorted results and per-failure bounds](LEADERBOARD.md) or [goal GIF gallery](GALLERY.md).
Three later standalone protocols and their strict API reproduction/export
commands are in the [PR233/234/235 readout](recent_prs/README.md).
The [PR236 batch-size diagnostic](pr236/README.md) adds its frozen endpoint
comparison and actual-metric goal GIF. The [PR239–243 caption diagnostics](caption_prs/README.md)
add five distinct questions and six actual-observation GIFs, including a separate
failed actual-caption context. The [complete question ranking](QUESTION_RANKING.md)
sorts all 119 retained questions and links their goals, API results and 187 GIFs.
The [Forge config-selection readout](../CONFIG_SELECTION_READINESS.md) applies
the current requirements from PR228 and reports eligible config options with
their exact source, recipe and runtime; these standalone diagnostics supply no
automatic Forge qualification.

## Contract

1. State the falsifiable question and exact target law. Include the conditioning
   input and held-out queries when the question is conditional, paired or temporal.
2. Train and sample with the current public ParticleGAN API. Use `GANTrainer`
   for supported scalar hosts. Conditional and routed hosts use the public
   `Recipe`, prior, loss, optimizer and policy components with their actual
   context/row contracts. Record the resolved recipe and actual sampling law.
3. Declare numerical bounds and budget before training. Validate the scorer
   with oracle samples and destructive controls relevant to the question.
   Distribution fidelity, paired correctness and controller counterexamples
   have different gates. Report which bounds fail, including nonfinite output.
4. Render actual observed states, showing the desired target or behavior beside
   API samples/predictions. Use fixed comparison axes, conditional inputs and
   held-out rollouts where relevant. Label updates and numerical gate status.
5. Introduce an explicit variant when the original setup cannot meet this
   contract. Preserve its useful hypothesis and controls; describe a narrower
   question or changed execution law in the variant's scope.

## Run

From the repository root in the project environment:

```sh
python -m benchmarks.toy_audit.api_run --list
python -m benchmarks.toy_audit.api_run --inventory /tmp/toy-api-inventory.json
```

Choose an ID from `--list`, then execute its declared default protocol:

```sh
python -m benchmarks.toy_audit.api_run --case CASE_ID --output runs/toy-api
```

`--recipe auto` uses the explicitly declared preset for each host. Set `--device
cuda:0` for a visible GPU. A named recipe override creates its own resolved
cohort; unsupported host/policy combinations are rejected explicitly.

For a bounded API/metric/media check:

```sh
python -m benchmarks.toy_audit.api_run --case CASE_ID --steps 16 \
  --eval-samples 128 --output runs/toy-api-short
```

The short run renders genuine target/output states and reports the instantaneous
numeric gate. It cannot pass the default-budget test. The full verdict requires
the declared training budget and evaluation sample count plus the declared
terminal observations passing the numerical gate (five for learned-quality
tasks; individual causal/geometry units declare their own observation count).
The frozen metric cadence is independent of `--frames`: by default, 24 evenly
spaced post-update observations (or every update for shorter declared units).
Image and word fixtures retain their declared 24 checks. GIF frame selection
adds real observations but cannot remove a scoring check or change the terminal
PASS requirement. Receipts record both exact scoring and media update schedules.
Exceptions, NaNs and
incomplete protocols fail explicitly. The command exits nonzero on FAIL.

`--all` runs every registered variant and can consume the sum of their budgets;
inspect the inventory first. `--jobs 8 --device cpu` runs independent cases in
eight processes, each with one Torch thread and its own frozen random streams.
Case output directories are exclusive so a new run
cannot overwrite an earlier receipt. Logs are JSON progress lines suitable for
tailing. Checkpoints and observation arrays remain in the run directory; publish
only the compact receipt and final goal GIF.

## Families

- [Images and actual conditional image queries](images/README.md)
- [Vector distributions, units, support, density and continuation](vectors.md)
- [Conditional, paired, temporal and causal diagnostic variants](conditional-diagnostics.md)

The complete migration ledger is generated from the executable providers. Every
original catalog question, including useful architecture and negative controls,
must map to a runnable variant; PR231's whole-FiLM question is included too.
