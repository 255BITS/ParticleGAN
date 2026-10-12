# Slower learned-prior R1/R2 comparison

The proposed global recipe changes **only `prior_lr_mult: 1 → 0.25`** from the
measured constant beta2=0.99/gamma=0.1/LR=0.0085
[parent](../../../../configs/forge/configurations/r1r2--169e9539b2dcfaab6b51e17f02907f28ea5b894f77155f7e688ab07a789de2b9.json).
Learned prior locations use native Adam at LR **0.002125**; G/D retain
**0.0085**. The prior remains learned, with zero prior regularization/L2.
Paired `a_r1r2` penalties run every update at gamma 0.1; all LR, beta2 and
penalty laws remain constant. This is one finite hypothesis, with no measured
result or qualification yet.

The [candidate](../../../../configs/forge/ideas/r1r2-prior-slow-v1.json) and
[ready study](../../../../configs/forge/studies/r1r2-five-track-prior-v1.json)
reserve **2,520 seconds** for all six ordinary Tier 1 tasks plus clock. Root
registration owns aggregate paid/storage limits, queueing and collection.
The [read-only preflight receipt](preflight.json) records READY admission,
the five exercised task deltas, exact files, seven authorized jobs and unchanged
scientific source digest
`25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`.
Preparation added zero training updates and zero sampling draws.

## Why test it

The [completed comparison](../../r1r2-tier1-search/README.md) found that halving
the entire LR helped density and words but lost sustained pole transport.
The faster constant parent preserves pole, token, AE and Gaussian passes,
while Ring16 and word acquisition fail. Its Ring16 endpoint covariance error
is **2.386427** against **0.85**, with HQ **0.842285** against **0.85**.
Separating prior pace from network pace tests whether slower co-adaptation lets
G/D correct local density while preserving their original movement budget.

The [saved density diagnosis](../../r1r2-tier1-search/density-diagnosis.json)
at the half-rate beta2=.99/gamma=.1 endpoint shows every mapped prior mean within
a target 3-sigma region, yet all components have tangential variance 2.08–3.88
times target variance. Its first-order generator-noise trace is 2.827451 versus
between-row trace 0.371232, in target-variance units. This motivates examining
generator/prior interaction, but **does not establish prior motion as the
cause**. Slower prior updates can leave the Jacobian unchanged or worsen mode
acquisition; no isotropic noise contraction or sampler change is proposed.

The readout binds attempt `36f9ba8f0f1d4b69b379f0189598f210`, observed-samples
SHA256 `d938da2a44530c2c337cb6600964988f6c9e2483fc37b86289300910b7b84ccb`
and saved-state SHA256
`46eb431722fe1a1b13fbfc994ef20618efaba176b71adecb95d15c323dd52309`.
The parent revision is
`35ebed78fe69b8923ead8335ea38bda209eb53ecc910b5e57ce5bac1c4b2f775`;
its original execution commit is
`078f3897e672bfae29504d40019d99c7c5c91ba9`. All are motivation only.

## Prior negatives and scope

The earlier [coupled-rate study](../../configuration-search/r1r2-global-repair-rates-v1.json)
used prior multipliers **0.282353/0.376471** to hold prior LR 0.0024 while
raising global LR. The 0.0085 global arm passed pole/token/AE then failed ring;
the 0.006375 arm failed pole. Their revisions are respectively
`98d01c8b382226c46c312202c5a952ad5d37355192778c149090a03b8fd17bfc` and
`ecd7497dc47ac901328a0e28a32d2657cf2ce0601db5690c629b1f33a6626d45`.
They retained changing beta2/gamma and older task/source contracts. These
negatives remain visible and do not grade the proposed constant-law comparison.
Historical BCAP frozen-prior failures likewise provide no R1/R2 repair credit.

`Recipe.make_optimizers` and the behavioral public component bindings already
apply the multiplier to learned prior groups. It is active in Gaussian, AE,
Ring16, word and clock. Two-pole optimizes direct generated coordinates at the
global LR, and unused-token hold has no sampled prior; their multiplier is
explicitly not applicable. Under equal scientific source, initialization and
consumed data, their numerical trajectories should exactly reproduce the
constant parent's passing trajectories. This is existing field ownership,
not a per-task recipe override. The archived parent word failure remains part
of the whole-recipe comparison.

Protocol seed 0, public deterministic initializer, all architectures/data laws,
actual seen batches, initial prior distribution/capacity, sampling law, update
caps, cadence and gates stay fixed. Fixed identity/zero fixtures retain their
separate cohorts; the five-row word particle-cloud exception remains explicit.
Initial tensor/data hashes and every consumed named RNG checkpoint must match
the common comparison contract. There are no engine or task edits in this track.

## Predictions, falsifiers and stopping

The registered numerical prediction is Ring16 endpoint full component covariance
error **≤0.85**; **>0.85** falsifies it. Endpoint improvement alone is insufficient:
the ordinary gate also requires all 16 modes, mass TV ≤0.15, HQ ≥0.85,
minimum eigen ratio ≥0.15 and at least five passing terminal checks at update
1,600. A complete repair requires every ordinary gate and clock PASS with this
one global recipe. Gaussian/word acquisition and sustained pole/hold checks
retain their distinct meanings.

Additional predictions are improved word acquisition, preserved AE/Gaussian
passes, exact pole/token trajectory parity and clock independence. Any missing
ordinary pass falsifies the complete-repair claim even if density improves.
Center/HQ loss would support the competing explanation that reduced prior
mobility hinders acquisition. Finish runnable current-tier peers, report all
metrics and actual-training GIFs, then stop this revision. No seed variation,
scientific retry, longer budget, Tier 2 work or automatic rate refinement follows.
