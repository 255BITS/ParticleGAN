# Independent control mode-hold audit

**Both registered controls fail mode hold. No integrity discrepancy was found. Both candidates remain open for the next bounded-calibration decision.**

Registration `host-profile-control-mode-v1`, hash `0c3688b09c50b8dbb5fc036180bbf82fc5940b84f0f45036540faa1f4eec7c51`, matches the file published at `1b2d8915`. All cells share source `5c9c929877c141ccf7352c16987d3a5aadf1aff3a0fedbfa78e7d9b8fe06fdb7`, cohort `eca8051cdbaaf1e7b56ef6096a6d29bb403e581403b48d3b4f85796190eab08b`, the same materialized task, runtime, compute profile, prior, sampling law and seed zero.

| Formulation | Verdict | Modes | HQ | Actual penalty / anchor / guard / A2 applications | Charged seconds |
|---|---|---:|---:|---|---:|
| Baseline | FAIL | 5/8 | 0.998779 | 1200 / 236 / 0 / 0 | 17.930310757 |
| No critic penalty | FAIL | 0/8 | 0.000000 | 0 / 0 / 4 / 0 | 21.671807714 |
| Anchor off | FAIL | 5/8 | 0.992920 | 1200 / 0 / 0 / 0 | 17.569934434 |

All three complete 1,200 updates for generator, discriminator and prior and have zero passing observations across the 24 required checks. The no-penalty recipe changes only `reg_coeff: 1 -> 0`; under the public mechanism this disables both the critic penalty and its anchor contribution. The anchor ablation changes only `reg_anchor_weight: 1 -> 0`, preserving all 1,200 penalty applications. Guard activations differ as an observed trajectory consequence, not an extra recipe change. A2 is requested/enabled but applies zero times on this host in all three cells; separately labeled synthetic probes are not training activation evidence.

Reproduced all three exact frozen evaluator grades, verified all 1,029 frozen source-file hashes, immutable diagnostic registration, scientific compatibility keys and durable/raw receipt hashes. Initializer methods and named parameter-seed receipts match exactly, as do all 14 named RNG bindings and all 24 recorded evaluation-isolation audits. Each records zero unintended stream deviations and finite-state guards. Learned MoG sigma .025, fixed width/uniform masses, no standardization and clean public-prior sampling are unchanged.

**Verification limit:** these transfer receipts do not save initial parameter tensor hashes/full initial tensors, model checkpoints or final RNG states. Exact tensor and final-state equality cannot be independently demonstrated from these artifacts; the verified evidence is the matching initialization declarations/bindings and recorded per-evaluation isolation audits.

Exactly two new attempts ran, with no retry or additional registered cell: no-penalty `6d10bbf90f1b4233a73034391599ecdb` (request `1ecb2cad5a5080af7160d599`) and anchor-off `eac540ccfe564a5ab396abdd545ab013` (request `3bd0097419b99a5836ca77e6`). Queue subscriptions and charged jobs name only `mode_hold`; process timestamps verify serial GPU0 execution. Both worker and last-child identities have exited. Campaign charges sum to **39.241742148 seconds**, with zero reserved seconds. These measured wall costs include setup and evaluation; no general speed or FLOP claim is justified.

Neither control provides a positive reference. Their independent downstream reference cells remain unmeasured, so these smoke failures do not establish true-reject/false-reject classification for the controls or calibration adoption. No candidate was concluded, no training was launched, and queue/reports/source were unchanged. Both submissions are completed/awaiting_readout; this audit leaves their lifecycle decisions to the coordinator. The eleven watched queue, registration and durable-receipt files remained byte-identical.
