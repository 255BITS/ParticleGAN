# AE, VAE, DDGAN and AE-DDGAN API research

Eight new public-API toy cases cover four API model/encoding modes. They use
one global recipe, the existing selected `bispf-optimized-quintic-v1`, with
protocol seed 0 and the public deterministic initializer. No optimizer, loss,
learning-rate, prior-update or prior-regularizer search is proposed. Each task
owns its architecture, target law, initial MoG capacity/width and model objective;
the recipe retains learned prior updates and SBNP regularization.

The research proposals and controls are frozen separately for [AE](ae.md),
[hard/categorical VAE](vae.md), [DDGAN](ddgan.md), and [AE-DDGAN](ae-ddgan.md).
The [bounded protocol](protocol.json) preserves every task contract and source
hash before execution. Six basic tasks enter required Tier 1 of the existing
`discriminator_stability` view. The two bimodal chain tasks enter Tier 2 with
their own Gaussian gate dependencies, reflecting their larger target and
conditional-generation claim. No gate is weakened or task moved after observing
training. Initial placement remains provisional; this is not calibrated default
adoption.

Each task has 24 numerical observations and requires the last five to pass.
Oracles and destructive controls check the scorers before training. The full
DDGAN tasks learn real forward-pair transitions and sample four reverse steps;
direct posterior toys cannot fill these cells. AE-DDGAN separately scores its
paired encoded-input chain and an independent ancestral chain with fresh latent
and reverse noise at every step.

The standalone bounded research cohort supplies its own CPU measurements. It
does not pool with the older CUDA family row or qualify a whole recipe against
the main suite. All six basic tasks finish before deciding eligibility for the
two deeper runs. Any basic failure blocks full Tier 2 research; an explicitly
budgeted three-update software prefix still supplies an actual-training GIF for
each blocked implementation, labelled INCOMPLETE against its full task.

The scientific setup was committed at `bee9f06ed` before execution. Its
[complete execution source manifest](execution-source.json) retains the bytes
used for training. Later `develop` changes are software integration, with no
transferred scientific qualification. To reproduce the frozen cohort, create
a detached checkout and apply only the verified report-only adapter from this PR:

```sh
git worktree add --detach /outside-git/api-model-toys-source bee9f06ed
git diff bee9f06ed HEAD -- experiments/forge/model_toy_publication.py \
  experiments/forge/publication_memory.py reports/forge/api-model-toys/run.py \
  reports/forge/api-model-toys/export-repair.json > /outside-git/model-reporting.patch
git -C /outside-git/api-model-toys-source apply /outside-git/model-reporting.patch
cd /outside-git/api-model-toys-source
PYTHONPATH=. /path/to/python reports/forge/api-model-toys/run.py \
  --output /outside-git/api-model-toys-run --jobs 3
tail -F /outside-git/api-model-toys-run/*/stdout.log
```

Outputs are exclusive: an unchanged run cannot be overwritten or retried.
`--export-only` reads saved receipts and GIFs without training or sampling.
Raw stdout, metric curves, arrays and complete model/optimizer/policy/RNG
checkpoints stay outside Git. The compact supplemental publication and saved
actual-training GIFs feed the existing [Forge leaderboard](../technique-inventory.md)
and [experiments by tier](../EXPERIMENTS_BY_TIER.md); this goal has no second
generated leaderboard. Historical AE, VAE and DDGAN evidence keeps its original
source, seed, initialization, budget and serving contract.

The MoG replay tests exposed a public `UpdatePolicy` checkpoint validator that
assumed every module state entry was a tensor. Its narrow metadata fix validates
immutable `_extra_state` before mutation. The two current word declarations
receive an explicit source-only binding layer because they pin that shared file;
the [migration receipt](policy-checkpoint-migration.json) retains their original
Git blobs, task hashes and source hashes. No word run or qualification is changed
or rerun.

The completed round has **three PASS, three full-budget FAIL and two BLOCKED**
full runs. Both AE tasks and categorical VAE pass. Hard VAE passes at update
1,200 but fails generation bounds at 1,000 and 1,100, leaving only two consecutive
passing terminal checks. Gaussian DDGAN and AE-DDGAN fail conditional-transition
and independent-generation bounds. Both bimodal full runs remain blocked and
have only their declared three-update software GIFs. The current leaderboard
retains every outcome with its full task budget; no test was re-tiered after a
failure.

AE-GAN reconstructs with MSE 0.009270 and independently covers all four modes
with precision 0.996826 and mass TV 0.117676. The reconstruction-only control
has lower paired MSE 0.004427 but independent precision only 0.632813; its PASS
does not certify prior generation. Categorical VAE retains input information
(shuffling raises reconstruction error 592 times), matches posterior latent-noise
variance, and passes both decoder-mean and likelihood coverage gates. These
bounded two-mode gates do not establish general density calibration.

AE-DDGAN's paired four-step reconstruction passes all five terminal paired
checks (worst normalized MSE 0.04409, worst axis RMSE 0.21342 and minimum shuffled
error gain 45.17). Its independent chain still fails, as does plain DDGAN on the
same Gaussian host. This separates useful encoding from the unresolved reverse
transition/generation problem; it does not identify a causal trainer defect.

Keep the passing AE and categorical VAE cases as scoped positive references.
Keep hard VAE as an explicit stability failure and both Gaussian chain cases as
generation failures. Inspect the saved code usage, decoder residuals and
per-timestep critic/transition moments before proposing one bounded mechanism
change. No seed sweep, tuning, continuation, larger run or default promotion is
authorized by this readout. The [individual reports](ae.md) and their VAE/DDGAN
links provide exact failed bounds and recommendations; the existing leaderboard
is the sole generated comparison table for this goal.

All saved states are finite, role update counts match the executed budgets,
and evaluation leaves training state/RNGs unchanged. The eight API executions
cost 78.528 summed CPU-run wall seconds, within the 3,480-second ceiling. This
includes initialization, updates and numerical evaluation; source import and
post-run media/export are excluded. Concurrent CPU runs supply no speed ranking.

All numerical executions, checkpoints and GIFs completed before an exporter
metadata check raised `KeyError: adam_variant`: public `Recipe.to_dict()` omits
inactive/default fields. The [report-only repair receipt](export-repair.json)
records the original and repaired source hashes. It rehydrates the public recipe
for comparison and admits only these exact reporting bytes while verifying the
entire unchanged scientific source manifest. Saved receipts/GIFs were exported
again with **zero optimizer updates, sampling draws or rendered frames**; no
experiment was rerun. Software checks separately cover exact checkpoint replay,
evaluation isolation, independent training streams, recipe ownership, complete
chain execution, scorer counterexamples and fail-closed publication.

The [saved-state VAE investigation](../vae-failure-investigation/README.md) reproduces
the archived final metrics and attributes prior-sample errors to row usage. The
same association occurs in the passing categorical control, so the cause of the
unsaved hard-VAE terminal failures remains unresolved. It adds zero training
updates and leaves every original result, gate and tier unchanged.

The [saved-state DDGAN investigation](../ddgan-failure-investigation/README.md)
finds low-noise conditional errors at central and fresh marginal contexts. It
measures how reverse noise masks poor clean predictions at later steps and
checks the critic's local output-gradient direction. It adds zero training
updates and leaves every original result, gate and tier unchanged.
