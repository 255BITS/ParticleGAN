# Public-API toy tests

Each executable toy variant has a scientific goal, public ParticleGAN execution,
a declared numerical PASS/FAIL gate, and a GIF comparing its desired behavior
with actual outputs over training. The historical catalog, training receipts,
ratings and GIFs retain their original identities. New variants link back to
their original questions and explain every change of scope, architecture,
resource, recipe or sampling law.

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
the declared training budget and evaluation sample count plus five terminal
post-update observations passing the numerical gate. Exceptions, NaNs and
incomplete protocols fail explicitly. The command exits nonzero on FAIL.

`--all` runs every registered variant and can consume the sum of their budgets;
inspect the inventory first. Case output directories are exclusive so a new run
cannot overwrite an earlier receipt. Logs are JSON progress lines suitable for
tailing. Checkpoints and observation arrays remain in the run directory; publish
only the compact receipt and final goal GIF.

## Families

- [Images and actual conditional image queries](images.md)
- [Vector distributions, units, support, density and continuation](vectors.md)
- [Conditional, paired, temporal and causal diagnostic variants](conditional-diagnostics.md)

The complete migration ledger is generated from the executable providers. Every
original catalog question, including useful architecture and negative controls,
must map to a runnable variant; PR231's whole-FiLM question is included too.
