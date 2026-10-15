# G-only gradient mean/coherence: original seed-zero matched pair

The candidate improves stationary retention from **35/72 to 69/72**, reacquisition from **FAIL to PASS**, and shifted hold from **12/24 to 20/24**. **Both acquisition gates pass and both retention gates fail.** This is a promising partial repair, not a winner or qualification. The recorded gates, sampling, horizons and live-weight scoring are unchanged.

| Fresh matched arm | Acquisition | Stationary observation checks | Reacquisition phase | Shifted-hold observation checks | Original retention gate |
| --- | --- | --- | --- | --- | --- |
| Protected NS + coefficient-one empirical scalar transport, OFF | PASS | 35 of 72 checks | FAIL | 12 of 24 checks | FAIL |
| Same source/configuration, G-only coherent mean ON | PASS | 69 of 72 checks | PASS | 20 of 24 checks | FAIL |

The control is a **fresh same-source acquisition and its own terminal-checkpoint retention**. It is distinct from the historical balanced-primary comparison even where its recorded trajectory reproduces earlier numbers. Neither arm borrows a historical prefix. The effective task prior is the original learned 256-component uniform MoG, latent dimension 2, fixed isotropic sigma 0.1; the protocol's reference-prior sigma 0.025 does not replace the task's actual prior.

The sole matched formulation delta is `generator_gradient_response: none -> coherent_G_mean_v1`: bias-corrected EMA7/8 of G gradients, with G-only coherence response and a fixed reciprocal effective-weight-count floor. Original prior gradients/critic-confidence response, critic law, protected NS+transport objective, sampler, deterministic initializer, isolated named streams, batch 128, 1,000 acquisition updates and 5,000 new continuation updates remain fixed. The empirical transport objective is an output-distance term; this G update operator is distinct from input-gradient or discriminator-output/logit regularization. No Atlas-native, call800 handover, whole-gauntlet or held-out transfer success is inferred.

[Original grades, final metrics, costs, checkpoints and receipt hashes](results.json), [task contracts](original-task-contracts.json), [protocol](protocol.json), [declarations and normal study reviews](candidate-declarations.json), and [full frozen-source manifest](source-manifest.json) bind the four original attempts. Retention receipts confirm exact restoration, unchanged histories and the own-prefix identity. Frozen shifted-reference measurements remain 0/48 for both arms; they are retained rather than substituted for live results.

The candidate's final CDF KS is 0.018775055506869753 versus the control's 0.07612553755008189. A good endpoint cannot replace the failed original sustained windows. Queue cost is 303.51600334700197 seconds; measured supervisor cost is 308.9983502320538 seconds. There are four new attempts, 12,000 new lifecycle training updates, zero paid retries and zero model sessions.

[Saved actual-training GIFs](media/index.json) render only the retained sampled panels; they add zero draws/updates and do not regrade the runs.

Each arm passed **1 of its 2 benchmark tasks run**: Tier1 Gaussian acquisition PASS and recorded Tier2 stability FAIL. The revision9 discriminator_stability context requires29 unique benchmark tasks; **27 are untested** for each arm. The phase fractions above count observation checks inside the same stability task, not additional benchmark tests. See the [visible main board](../technique-inventory.md#source-bound-benchmark-rows) and [machine row cells](../source-bound-board.json).

## Reproduction and limits

The [unapplied scientific source delta](reproduction/source-delta.patch) records the scientific implementation against pinned develop `dacf96749a4effc0648a7a0945266d508804110e`; its [manifest](reproduction/source-delta.json) lists included scientific paths and the exact local full-source archive. Operational runner changes and unselected host changes are excluded from this evidence patch. It is a reproduction archive and is not applied to the public core package by this evidence PR. Restore source files and compare every SHA256 in `source-manifest.json` before replay. [Exact original artifact locators and hashes](artifact-archive.json) retain all checkpoints, scheduled observations, original commands/provenance, raw stdout and independent grades locally. Bulk traces/tensors remain outside Git; absence of hydrated originals is an evidence-access limitation, not permission to invent or rerun a score.

The archived controls retain rejected joint scalar and joint mean responses as well as the accepted eight-panel G-only analytical direction checks. Default-OFF parity, matched consumed streams, prior-response equality and exact next-four-step active restoration are software/analytical controls, not trained Gaussian qualification.

A separately registered alternate-seed research cohort has its own evidence identity. Its results are excluded from this original seed-zero publication. The selected family configurations, scientific rows, historical denominators and original numerical gates remain unchanged. Recommendation: retain the FAIL and examine the remaining stationary/shift misses; do not promote this formulation from the partial score.
