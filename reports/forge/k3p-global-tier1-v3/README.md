# K3P global search: rates, direct moments and failure analysis

**No new 5/5 Tier 1 winner.** This continuation tested 12 complete global configurations in two separately frozen studies. Eight rate-search recipes passed movement, token hold and AE hold, then failed ring. Four clean, strongly regularized recipes failed movement. All 12 word cells remain UNKNOWN. The existing 4/5 K3P selection remains selected in the [single current family leaderboard](../technique-inventory.md); it fails words. The [historical clean word recipe](../word-root-cause/receipts/k3p-coeff170-cap1.json) still passes 24/24 checks under its own identity. These are different complete recipes, and their task passes cannot be combined.

The studies measured 36 task cells: 24 PASS, 12 FAIL and 276 UNKNOWN across the full 312-cell denominator. Charged cost was 273.364063 seconds against separate ceilings of 16,800 and 8,400 seconds. Both executed commit `72da7275034422bc171f9b9b1955c9b150cda045`, scientific digest `730b77d2b4e1e6ec34f9aed7a5d061874e70eaebb5af0ff839cff1781b849a8d`, with the declared system Python, NVIDIA RTX A6000 cohort, GPU0 and one automatic CPU worker. The two studies ran sequentially, with at most two training workers. Costs are not convergence-speed rankings.

## What failed before this search

The [earlier readout](../k3p-global-tier1-v2/README.md) measured 20 recipes. Strong critic regularization failed the 80-update movement task; the two coefficient-1 survivors acquired 16 ring modes but produced clusters that were too broad. Their word cells were never reached.

[Saved-output forensics](ring-analysis.json) reproduces all 48 earlier ring observations without training or sampling. Tangential variance averaged 4.56 and 4.81 times the target; radial variance averaged 2.50 and 2.87 times the target. Recentring each component improved quality only from .7791 to .7876 and .7761 to .7925. The problem was more than displaced centers. Task geometry, prior width, initialization, scorer and clean/live evaluation bindings match the incumbent. Training rates and output-noise controls differ, preventing a single-cause attribution across historical cohorts.

[The direct-rate analysis](direct-rate-analysis.json) establishes a separate obstruction. At the word-winning base LR .0006, the existing direct-particle beta2 .9 and the unchanged 80-update schedule bound absolute movement by .226663, below the .3 gate. With beta1 zero and zero initial moments, `v_t >= (1-beta2)*g_t^2`; the normalized Adam step is at most `sqrt((1-beta2^t)/(1-beta2))`. Summing scheduled rates with the existing gain bounded by two proves the limit independently of critic strength. Beta2 .999 raises that upper bound to .423865, which allows but does not imply success.

Penalty units also matter. In its initial phase, K3P weights the real squared-gradient norm by `coefficient/(2*input_dimension)`. Coefficient 170 therefore gives weight 85 in one-dimensional movement and .5 in the 170-dimensional joint word input. Later cap/proximity units differ. This is a formulation tradeoff to inspect, not evidence that the BiGAN task is misimplemented or that dimension scaling alone caused every failure. Higher-rate movement failures plateau before scheduled decay; behavioral receipts do not retain critic/particle states sufficient to reconstruct their exact generator forces.

## The new rate search

[Eight prepared recipes](plans.json) retain input noise .5, output noise zero and a full network horizon. They fix base LR .006375 and vary positive coefficient .5/1, critic multiplier .5/1 and latent-prior multiplier .5/1. The prior base rates are .0031875/.006375, compared with .0012 in the previous survivors. The incumbent's 1600-step network cap is inactive on the 400-update ring, so changing that cap cannot explain its ring pass.

Every recipe passed the first three tasks. All eight then failed the required full component covariance error bound .85, with errors 1.4863–14.0935 and no passing ring observation. Seven acquired all 16 modes; one acquired 15. Four passed the final high-quality fraction bound .85. All eight passed the final mass-TV and minimum eigen-ratio bounds. [Original compact receipts and exact outcomes](summary.json) retain every required task, bound, source identity and UNKNOWN cell.

The selector's **best observed**, unqualified recipe is `01eca360219e…`: coefficient1, critic multiplier1 and prior multiplier1. Its final ring quality was .899902, mass-TV .059814, 16 modes and minimum eigen-ratio .308152. Full covariance error1.486334 failed. All recipes had three required passes; the declared configuration-hash tie-break determines selection and grants no superiority or speed claim.

[Current saved-tensor analysis](current-ring-analysis.json) reproduces all 192 scored ring checks and records 12 contrasts changing one numerical field within the same source/runtime/initialization/sampling cohort. Critic multiplier .5→1 improved final quality and full covariance in all four pairs. Prior multiplier .5→1 improved quality in all four pairs and full covariance in three. Coefficient .5→1 improved full covariance in all four pairs while worsening quality in three. These observations do not establish monotonic behavior outside this finite grid.

The improved cores still have meaningful served tails. Five of eight recipes have four-sigma core covariance error below .85, but all fail the declared full covariance gate. For `01eca…`, core error is .665041; 200 of 4096 samples lie beyond four sigma and carry 34.34% of centered covariance energy. Its worst component has error 8.0644, core error .3297 and samples reaching 13.35 sigma. Excluding that component still leaves the other 15 averaging 1.0478, above .85. Removing outliers, substituting the core diagnostic or relaxing the gate would change the scientific question; none was done.

## The direct-moment search

[Four prepared recipes](plans-direct-moments.json) retain the clean word-positive global base: coefficient170, critic multiplier1.5, nominal latent-prior LR .0012, clean training and a full horizon. Base LR .0006/.0012/.002125/.00425 is coupled to the prior multiplier, while the already-public direct-particle beta2 changes from .9 to .999. This setting applies to direct generated-coordinate optimizer groups, not word networks or sampled latent locations; it is one global recipe field, with no task-specific override.

All four fail movement. Final movement is .030081/.045952/.073910/.119392, below .3; critic-gradient medians .003212–.004705 satisfy their upper bound. No observation passes and all subsequent tasks are UNKNOWN. [The direct-moment summary](direct-moments/summary.json) retains the complete evidence. Removing the analytic step exclusion did not restore the actual trajectory. A longer second-moment memory can slow responses to decaying gradients; the upper bound was never a success prediction.

Search admission now exposes `direct_particle_betas` as an existing numerical pair, validates its types/ranges and requires a consuming direct-coordinate formulation host. Enabling/removing either moment remains a structural change; inactive word-only and plain-Adam searches reject it. The optimizer, losses, controllers, architectures, priors, initializers, task budgets, numerical gates and sampling laws are unchanged. Focused search, identity, field-boundary, public-optimizer and decision checks passed 189 tests before execution.

The compact receipts report each named numerical prediction as false and its
falsifier as observed. Forge's full-scope decision evaluator remains `incomplete`
because ordinary first-fail progression intentionally leaves later authorized
tasks UNKNOWN. That administrative completeness result neither reverses the
recorded FAIL nor authorizes further spending. The [independent decision and
contrast diagnosis](diagnosis-current.json) retains both facts.

## Recommendation and reproduction

Do not replace the incumbent or infer failure of every K3P configuration. The measured next target is the served ring tail law: inspect existing latent-prior moment/rate and critic-balance controls around coefficient1/D1/prior1 while retaining the full covariance gate and one recipe across all tasks. For the strong word recipe, simply raising direct beta2 is falsified as a movement repair in this grid; another rate increase alone does not address the observed low-force plateau. Any next study needs a new finite declaration. There is no additional paid round, seed study, public-default adoption or higher-tier run in this continuation. The screen remains provisional.

All 36 attempts have actual-training GIFs: 28 behavioral metric curves and eight ring sample animations. Ring frames use all 4096 already-scored clean/live samples, with deterministic target centers. All 24 numerical checks are retained while nine frames are displayed. Media generation adds zero updates/draws. There is no new word GIF because words were not reached. [The immutable archive](archive.json) preserves exact raw logs, certificates, scored tensors, source snapshots and queue records; [its audit](archive-audit.json) verifies hashes and bindings without training. Bulk outputs stay outside Git.

The preparers create reviewable READY contracts only before registration. The runner requires the exact reviewed commit and executes ordinary Forge gates. Registered studies and archives are immutable; never rerun unchanged training for publication. Tail `runs/forge/k3p-global-tier1-v3/worker.log` or the corresponding queue's `events.jsonl`; the direct study uses `runs/forge/k3p-global-direct-moments-tier1-v1/`.

Publication reproduction after hydrating exact originals performs no training:

```sh
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/k3p-global-tier1-v3/materialize.py
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/k3p-global-tier1-v3/materialize.py --plans reports/forge/k3p-global-tier1-v3/plans-direct-moments.json --output-subdirectory direct-moments
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/k3p-global-tier1-v3/archive.py --verify reports/forge/k3p-global-tier1-v3/archive.json
```

The forensic helper sources and JSON receipts retain their input hashes, zero-spend scope and reproduction commands. Old study declarations, word passes and archive identities remain unchanged. The [search operating guide](../../../docs/forge-configuration-search.md) describes strict family search and whole-configuration selection.
