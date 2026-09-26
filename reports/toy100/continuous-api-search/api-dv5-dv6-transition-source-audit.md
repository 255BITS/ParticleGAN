# DV5/DV6 source and transition audit

All declared source/artifact hashes verify for DV5-single, DV6-single and DV6-stationary. Both single runs contain all460 observations; stationary contains750. Model, optimizer and all RNG/data initialization receipts match the original public fixture. DV5/DV6 cold observations through2400 are identical except the variant label. DV6 uses identical package bytes for single and stationary runs.

| Evidence | First arrival | Passing since arrival | Departures | Final uninterrupted suffix |
|---|---:|---:|---|---|
| DV5 original target |550|186/186 through2400|None|550–2400 |
| DV5 shifted target |2750 (+350)|184/186|2760:7modes,HQ.5849609375;2780:8modes,HQ.88916015625|2790–4600,182checks,minHQ.916259765625 |
| DV6 shifted target |2790 (+390)|182/182|None|2790–4600,182checks,minHQ.90673828125 |
| DV6 stationary7500 |550|696/696|None|550–7500,minHQ.900390625 |

Both single runs also pass120/120 prehold checks; frozen controls pass0/220. DV5 first touches earlier, has two disclosed misses through30updates after that touch, and starts its uninterrupted suffix40updates after firsttouch. Both variants start that suffix at2790. DV6's cleaner first-touch record alone does not establish uniformly better quality, but its own completed stationary run adds evidence.

**DV5 longer retention remains UNVERIFIED, not permanently disqualified solely by these transition misses.** The archived worker automatically labels any post-first-arrival departure FAIL (`worker.py:297–300`), which is stricter than the shared requirement to report arrival, all departures and stability for supervisor assessment. Its original ledger is untouched. No new numerical acceptance rule or rewritten first-arrival time is introduced. Early transition settling differs from collapse after an established long stationary period.

No horizon/target oracle was found. Actual GANTrainer owns the controller and penalty binding; signals come from real minibatch temporal features, adversarial payoff imbalance, generator gradients and previous optimizer surprise. Noise stays input0/output.029; total_steps=None removes exhaustion. DV6 replaces slow data-memory permission with current data-drive permission for surprise exemption and anchor release. Applied-rate trust has a one-update causal lag. Full-update serialized backward is explicit and checkpointed; ordinary Adam state is not relocated.

Image compatibility is source-supported: real samples flatten across arbitrary feature dimensions, projection shape derives from data, gradient diagnostics flatten arbitrary parameters, and payoff losses are scalar. KA2 uses per-sample numel/flattened gradients; GANTrainer accepts rank≥2. The same recipe and serial_backward=True can be tested on the frozen image host without a new detector. Factory-only loops omit the controller binding. No image pass is inferred.

The detector remains a normalized engineering heuristic rather than a universally calibrated significance test. Trust can reduce rates below nominal floors. Exact continuation, differing-budget prefixes, repeated/long recovery, own frozen22 and ordinary public K3P remain separate qualifications; predecessor passes cannot substitute. Detailed hashes/metrics are in the paired JSON. No training, tests, GPU work, worker edits or ledger changes were performed.
