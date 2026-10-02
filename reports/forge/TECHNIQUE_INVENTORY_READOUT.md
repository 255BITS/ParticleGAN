# Technique inventory through ordinary Forge tiers

The [generated leaderboard](technique-inventory.md) covers all **12 current
technique declarations**. Matched BCap leads this exact cohort: **3/3 smoke,
5/19 quality, 0/2 endurance**, attaining tier 1 before failing mode hold.
Matched R1/R2, K3P, KA2 and four K3P ablations stop at the first smoke task.
E22, Atlas and both released GAN v3 host variants are blocked before training.

The campaign completed **16 unique attempts, 5,090 updates and 127.788 paid
seconds**, with no execution errors or remaining reservations. Across the full
288 required cells there are **8 PASS, 8 FAIL, 90 BLOCKED and 182 UNKNOWN**.
Unknown cells are unmeasured; the fixed denominator does not label them failures.
The [run receipt](technique-inventory-run.json) binds all metrics, costs, task
budgets, revisions and original result hashes.

## What passed and what stopped

| Technique | Two-pole mean absolute movement, minimum .30 | Median critic gradient, maximum 1 | Gate | Tier 1 | Tier 2 | Tier 3 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| BCap, matched | .403200 | .465270 | PASS | 3/3 | 5/19 | 0/2 |
| R1/R2, matched standard penalty | .178637 | .064047 | FAIL: movement | 0/3 | 0/19 | 0/2 |
| K3P | .106521 | .063724 | FAIL: movement | 0/3 | 0/19 | 0/2 |
| KA2 | .111375 | .049373 | FAIL: movement | 0/3 | 0/19 | 0/2 |
| K3P without critic anchor | .104825 | .076785 | FAIL: movement | 0/3 | 0/19 | 0/2 |
| K3P without critic penalty | .447748 | 1.254604 | FAIL: gradient | 0/3 | 0/19 | 0/2 |
| K3P without A2 | .106521 | .063724 | FAIL: movement | 0/3 | 0/19 | 0/2 |
| K3P without training output noise | .264117 | .130802 | FAIL: movement | 0/3 | 0/19 | 0/2 |

These are terminal metrics; each gate also requires the complete 24-observation
curve and five passing terminal checks. BCap additionally passes
`unused_token_hold` and `ae_gan_hold`, followed by quality tasks `trajectory`,
`residual_student`, `unipolar`, `cover_leftover` and `mid_scale_identity`.
Its learned-MoG `mode_hold` completes all 1,200 updates but retains **5/8 modes**
with HQ **.999756**; precision cannot compensate for missing modes. This failure
stops vectors, images, native 100-Gaussian tasks and endurance. Their outcomes
remain unknown in this cohort. BCap's total paid cost is **86.357 seconds**.

The A2-off result matches K3P because the two-pole host has no latent-table
optimizer and cannot exercise A2. It establishes no causal A2 comparison.
Removing training output noise increases movement here but still misses the
unchanged gate; this does not revise the earlier failed full native diagnostic.

The [roster audit](TECHNIQUE_ROSTER.md) explains all four preflight blockers,
historical-only techniques and the explicit K3P binding repair. R1/R2 means the
standard zero-centered real/fake squared-gradient penalty within the matched
K3P optimizer/host recipe. It is not a separately qualified stock Adam recipe.
The full released BCap recipe remains distinct from the coefficient-1 matched
BCap arm. Archived Atlas's 19/19 cloud/noisy passes, clean failures, historical
22-host results and current learned-MoG evidence retain separate identities.

## Recommendations

Use BCap as the most advanced **measured ordinary reference in this cohort**;
stop its failed revision rather than extending it or calling it a quality winner.
Inspect the saved mode-hold state before declaring a new bounded mechanism test.
Keep R1/R2, K3P and KA2 as explicit comparator rows and retain all negative and
unknown cells. Do not repeat unchanged science, tune thresholds or vary seeds.

Calibrate the provisional screen before adopting it as a scientific default.
The smoke negatives provide no ranking of their unmeasured native quality.
To evaluate E22/Atlas, freeze policy-aware cloud/serving tasks with their own
sampler, state selection and budgets; they cannot borrow clean-live qualification.
Released GAN v3 needs compatible declared hosts. Register historical techniques
through truthful public formulation/task bindings before adding executable rows.

## Regeneration and artifacts

New `configs/forge/ideas/*.json` cards enter the next inventory automatically.
`inventory plan` reports task coverage, reuse, blockers and explicit ceilings;
`inventory run` uses the ordinary queue and stops on failed prerequisites.
The current campaign's reservation ceilings are 529,200 seconds overall and
44,100 per technique; actual paid execution is reported separately. Expanding
an immutable campaign requires a new ID and explicit budgets.

```sh
python -m experiments.forge inventory plan --through-tier 3
python -m experiments.forge inventory run --through-tier 3 --gpus 0,1
python reports/forge/regenerate_technique_inventory.py --device cuda \
  --output-prefix reports/forge/technique-inventory
python -m experiments.forge logs --follow --campaign technique-inventory-v1
```

The snapshot is bound to source digest
`bb31b3f77bee0199fa51df84856dd04171fb58baca7c9328270f3e2a7c67d62d`
and implementation commit `b04b1b27`. All attempts use protocol seed 0 and their
named streams. CPU behavioral hosts and the CUDA mode-hold host retain their
declared priors and sampling laws; the requested CUDA cohort includes CPU-only
hosts. A coordinator restart reused completed work while batching report
compilation; it created no duplicate scientific attempts or retries.

Full new execution envelopes contain per-update diagnostics and stay outside
Git. Their original bytes, artifacts, source snapshot, logs and full readouts
are in the local archive `runs/forge/technique-inventory-v1/receipts-and-source.tar.gz`.
Its [manifest](technique-inventory-archive.json) records all original receipt
hashes and archive SHA-256
`0574554f0b98b46308c9fc282ff5a993748a4e0a1fc7568fad30aab6d38581e8`.
Published [summary receipts](technique-receipts/) contain final metrics, gates,
costs and provenance; they are projections and never qualification inputs.
Compact readout records retain exact attempt coverage and result hashes, with
the original full readout bytes preserved in the archive.

For independent regeneration in another checkout, obtain the exact local
archive, verify its hash against the manifest, and extract it from the repository
root. The original envelopes are restored into ignored paths; full archived
readouts remain under `runs/`. Use the recorded implementation, runtime and
hardware cohort to reproduce this table. Different cohorts remain separate.

```sh
sha256sum runs/forge/technique-inventory-v1/receipts-and-source.tar.gz
tar -xzf runs/forge/technique-inventory-v1/receipts-and-source.tar.gz
python reports/forge/regenerate_technique_inventory.py --device cuda \
  --output-prefix reports/forge/technique-inventory
```

Regeneration validates the original certificates and regrades compatible
evidence without training. It refuses to replace a measured report when its
original envelopes are missing. Raw stdout, per-update streams, checkpoints
and state dumps remain local or in the archive; no ignored logs were force-added.

Validation passed **708 Forge tests**, **53 focused inventory/knowledge tests**
and **9 publication tests**; Forge validates **47 tasks and 6 views**. The
publication tests cover trace removal, immutable originals, invalid certificate
rejection, summary links and missing-archive protection. Metrics and receipts
drive this comparison; no image inspection or seed-only experiment was used.
