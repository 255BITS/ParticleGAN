# Global K3P Tier 1 search

The first bounded round rejected all 12 complete global recipes at `two_pole`. None reached the word task. Their later cells remain UNKNOWN; these results do not establish that the K3P family cannot work.

The [single current family leaderboard](../technique-inventory.md) owns family selection. This report retains the research explanation and exact receipts without creating a second leaderboard. Qualification requires all five ordinary Tier 1 tasks to pass for one complete recipe: `two_pole`, `unused_token_hold`, `ae_gan_hold`, `ring16_acquisition`, and `five_word_joint_acquisition`. The screening profile remains provisional. Passing it can support a configured family standard; it never adopts public defaults or establishes higher-tier qualification.

The [first declaration](plans.json) tested critic coefficients 10, 30 and 85, base learning rates .00425 and .006375, and critic multipliers .25 and 1.5. Every combination retained the same clean-output, full-horizon global K3P controls. The latent-prior multiplier was .0012 divided by the base rate. This fixes the nominal latent-prior base rate, not the realized learning-rate trajectory. `two_pole` directly optimizes generated coordinates using the public direct-particle optimizer: its base rate is the candidate learning rate, and latent-table `prior_lr_mult` and `prior_betas` are inapplicable there.

The [first summary](summary.json) records 12 measured FAIL and 300 UNKNOWN across the complete 26-task denominator. All 12 tasks completed 80 updates and 24 scored checks, with no passing observation. Final movement ranged from .0257011 to .141911, below the unchanged .3 minimum; critic-gradient medians satisfied their bound. The best movement in this finite grid was coefficient 85, base rate .00425 and critic multiplier .25. Increasing the base rate did not consistently improve movement. Charged execution cost was 70.279050 seconds against the declared 25,200-second ceiling. These measurements reject the declared global recipes, while leaving the later tasks unmeasured.

A [second declaration](plans-input-noise.json) activates the existing symmetric critic input-noise schedule globally at initial standard deviation .5, annealing to zero over the first tenth of each task. Generator output noise remains zero and serving remains clean/live. This activation changes the strict technique signature, so it is registered as a separate structural base within the existing K3P formulation family. Its finite numerical search keeps that signature fixed: eight whole recipes use coefficients 1, 10, 30 and 170, base rates .006375 and .0085, critic multiplier 1.5, and the same nominal latent-prior rate. The original word geometry, BiGAN update, initialization, prior exception, gates and task horizons are retained. No new optimizer, loss, stabilizer or task-specific recipe is introduced. Historical word passes supply motivation, with zero qualification reuse.

The follow-up round is pending publication. The first round's receipts and media remain unchanged. Comparisons must identify complete recipes and actual source/runtime bindings; a changed activation together with a changed coefficient or rate does not isolate the cause of an outcome.

Both phases use ordinary prerequisite progression through Tier 1, stopping a recipe at its first required failure. Each recipe reserves 2,100 seconds. Higher tiers retain their full denominator as UNKNOWN. Scientific execution uses the declared system Python 3.14.7 runtime, one CPU thread per task, GPU 0, one GPU worker and one automatic CPU worker, permitting two concurrent jobs. There is no speed comparison or default-adoption claim.

The adapters retain already-scored outputs without additional training or sampling. Behavioral GIFs show the saved numerical observations against their frozen gate bounds; ring and word GIFs, when those tasks are reached, render the saved scored tensors with the existing public-API observer renderer. Publication retains all 24 checks while displaying nine frames. [Media receipts](media/receipts.json) bind the GIFs to their original raw request/result bytes and renderer source. Bulk logs, traces, states and tensor outputs remain local and will be preserved in the final artifact archive.

Reproduce publication only after both ordinary studies finish; these commands perform zero training updates or sample draws:

```sh
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/k3p-global-tier1-v2/materialize.py
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/k3p-global-tier1-v2/materialize.py --plans reports/forge/k3p-global-tier1-v2/plans-input-noise.json --output-subdirectory input-noise
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/k3p-global-tier1-v2/archive.py
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/k3p-global-tier1-v2/archive.py --verify reports/forge/k3p-global-tier1-v2/archive.json
```

`prepare.py`, `prepare_input_noise.py` and `run.py` retain the declared execution workflow. A registered study and the final archive identity are immutable; do not rerun unchanged training for publication or overwrite an existing archive. The queue's `worker.log` remains the live execution log.
