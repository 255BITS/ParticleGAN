# Hard-VAE failure: saved-state investigation

Hard routing leaves reconstruction gradients concentrated on a small part of
the uniformly sampled prior. At the saved final state, all 183 decoder-mean
quality misses come from its 12 low-usage rows. The passing categorical control
has the same concentration pattern, with only 26 misses. This supports a local
coverage diagnosis but does **not** establish why hard VAE failed at updates
1,000 and 1,100: those intermediate checkpoints were not saved. The original
hard **FAIL** and categorical **PASS** remain unchanged in the
[existing leaderboard](../technique-inventory.md).

This is an exploratory investigation of the two original final checkpoints,
not a new trained candidate or independent confirmation. A preliminary read-only
seed-713 diagnostic on these same checkpoints informed the hypothesis and the
75% attribution criterion. The canonical panel uses the original evaluation
seed 10,000 so every original final metric can be checked exactly. The
[analysis and protocol](protocol.json) were frozen in commit `255403912` before
that canonical execution. No statistical significance, general robustness or
causal conclusion follows from either panel.

The [compact results](results.json) bind both original receipts, complete final
checkpoints, recipes and scientific sources. Analysis uses frozen execution
commit `bee9f06ed21695e518c1cb702e12a385ab9389e8`, rather than loading archived
states into a newer implementation. It reproduces **every final metric exactly**,
including hard HQ .955322/.915039 and categorical HQ .993652/.985596 for decoder
means/likelihood samples. Both final states individually pass; only categorical
passes the original five-observation terminal requirement. See the
[original VAE readout](../api-model-toys/vae.md) for its failed states and bounds.

Each checkpoint receives one fixed 4,096-input panel with eight posterior draws.
It follows the original observation's draw order, including its shuffled-code
control, independent prior samples and separate Gaussian likelihood draws.
Expected posterior mass is computed from the true posterior probabilities,
rather than substituting sampled categorical counts or hard routing's soft
backward weights. A row is low usage below 1% posterior mass; it still receives
1/16 of independent prior probability. A quality miss retains the task's
distance greater than .2 from both target anchors. These are diagnostic groups,
not new qualification gates.

Hard VAE has 12 low-usage rows receiving 3,113 independent prior draws; 183 miss
the decoder-mean bound (5.879%). Its four higher-usage rows receive 983 draws
and have zero misses. Eight rows have posterior mass below 1e-6; these account
for 136/183 misses. Its low-usage row centers are farther from an anchor on
average: .12180 versus .08537 for higher-usage rows. Of 348 likelihood misses,
340 arise from low-usage rows. Posterior/prior row-mass TV is .72900.

Categorical VAE has seven low-usage rows receiving 1,820 independent prior draws;
26 miss (1.429%). Its nine higher-usage rows receive 2,276 draws and have zero
misses. Three rows have posterior mass below 1e-6, accounting for 25/26 misses.
Low/high-usage center distances average .09856/.03758. All 59 likelihood misses
arise from low-usage rows. Posterior/prior row-mass TV is .46069. Thus unused
rows or attribution concentration alone cannot distinguish failure from success;
the error magnitude and original sustained generation gates matter.

The gradient panel reuses the first 64 held-out inputs, with separate declared
posterior, prior-identity and prior-noise generators. It differentiates the
actual weighted ELBO, adversarial term and recipe-owned SBNP regularizer through
public API components. Hard ELBO gradients reach four prior rows; adversarial
and SBNP gradients reach all 16. Categorical ELBO gradients reach all 16,
including extremely small values on seldom used rows. Its categorical KL and
score estimator also differentiate routing probabilities; hard routing detaches
the unselected means and uses a biased straight-through encoder gradient.

Hard weighted-ELBO gradient norms are .50035 for the encoder, .17049 for the
decoder and .03774 for the prior, versus .004875/.08365/.008826 for categorical.
Adversarial prior norms are .01204/.002738 and SBNP norms .000424/.001052.
These are gradients at one saved state, before the optimizer's normalization;
they do not establish actual update magnitudes or prove instability. Decoder
ELBO/adversarial gradient cosines are -.3985 for hard and -.8813 for categorical.
The passing control has greater directional conflict on this panel, so the
sign of that conflict alone also fails to explain the terminal difference.

All checkpoint/model/optimizer/policy/RNG state and existing gradient buffers
remain byte-exact. The execution uses .866 CPU seconds under its 60-second cap,
one Torch thread and zero optimizer updates. The raw tail-friendly log is
`/nvme/dev/particlegan/artifacts/vae-failure-investigation/saved-state-v1/stdout.log`.
No samples, checkpoints, per-update streams or new generated leaderboard are
committed. The original actual-training GIFs remain the illustrations of the
[hard](../../toy_audit/api_contract/model_modes/media/api-model-vae-hard.gif) and
[categorical](../../toy_audit/api_contract/model_modes/media/api-model-vae-categorical.gif)
tasks; no new frames or training were needed.

Keep categorical as the scoped positive reference and hard as the unresolved
stability failure. The next useful causal experiment would test one explicitly
declared reconstruction-coverage mechanism, applying one global recipe to both
unchanged VAE tasks and saving all 24 scheduled checkpoints plus per-row usage
and error summaries. Keep seed 0, initialization, data sequence, architecture,
prior capacity/width, sampling, budget, cadence and gates fixed. Reuse the
archived baseline's numerical verdicts; do not rerun it merely to recover
missing intermediate states or promote a passing endpoint. This investigation
does not select such a mechanism or launch more training.

Reproduce from the original saved artifacts with a frozen source checkout:

```sh
git worktree add --detach /outside-git/vae-scientific-source bee9f06ed
/path/to/python -u reports/forge/vae-failure-investigation/analyze.py \
  --source-root /outside-git/vae-scientific-source \
  --evidence-root /nvme/dev/particlegan/artifacts/api-model-toys/research-v1 \
  --output /outside-git/vae-analysis/results.json \
  > /outside-git/vae-analysis.log 2>&1
tail -F /outside-git/vae-analysis.log
```

Missing originals, changed scientific bytes, altered receipts, mismatched final
metrics, mutated state or an existing output path cause an explicit refusal.
No training fallback exists. Tests cover destructive attribution controls,
opposite explanations, checkpoint/source binding, exact state/RNG/gradient
preservation, exclusive outputs and the published original final-metric join.
