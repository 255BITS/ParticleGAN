# Learned regression diagnosis

## Findings

The remaining toy failure is a diffuse distribution with little mass near any of the 25 supported modes. Its single covered mode is the only mode exceeding the coverage mass threshold; outputs are not concentrated in a single mode. A structural blind spot is reproduced: ordinary count evidence assigns unsupported emitted points to the nearest existing cell and contains no support/unsupported category. Most unsupported rows therefore produce no categorical count discovery. The broad-flag isolation guard then prevents their repair.

No additional serving or optimizer implementation error was found in the 12 saved states. RA2 serves fast G/prior at toy1000/2000 because the prior tester is in drift (`last_decisive=1`). E22 serves EMA at both toy checkpoints; old CB serves EMA at2000. RA2's clean final precision is .2910 with fast weights and .3096 with EMA, both far below E22's .8203 clean EMA precision. Serving a different saved average cannot rescue this state. At1000 RA2 EMA is worse than fast (.1201 versus .1611). These comparisons contain no latent or output noise.

All model and Adam moment tensors are finite, every optimizer parameter's step counter equals the saved1000/2000 update, and prior moments remain active. The final toy prior has117 active last-gradient rows in each variant, all1024 rows remain exactly distinct, and A2's cumulative observed-row fraction is .1176 in every variant. The tested latent perturbation has an exactly identity derivative to the indexed prior rows. CUDA RNG bytes were never loaded into a CPU generator or trainer.

## LR and controller explanation

The final toy generator-network LR is6.640625e-5 in all three variants. RA2 prior LR=.0085 and critic LR=.0031874155 are2×old CB and4×E22, respectively. This follows the unchanged trainer rule `critic_scale=max(critic_tester.s,.75*prior_tester.s)`, followed by payoff damping. RA2's prior scale remains1 while old CB/E22 reach.5/.25; the critic LR floor therefore remains high even though RA2's own critic tester scale is4.6566e-10. This is consistent controller state, not a restored-LR mismatch. It is a feedback consequence of the unsettled prior, and it does not justify changing the LR recipe after seeing quality.

Learnable toy output sigma is floored at the recipe's .029 in all final states. The saved output-noise gradient is zero. Undefined stationary-window diagnostic entries are preserved explicitly as `NaN`/`Infinity` strings plus their source paths: the source uses NaN to exclude copied/unobserved rows and for undefined statistics. These placeholders are not nonfinite model or Adam moments. The initial default-Python SciPy/NumPy import error and the first strict-JSON serialization error are retained in failed logs; final scripts run with the frozen harness environment.

## Training-field sensitivity

The matched toy2000 conditional probe gives RA2 existing-jitter versus zero-jitter gradient cosine .0113 for G and .1485 for the prior, with60/128 prior rows having an opposed direction. E22's corresponding cosines are .9283/.6632 with19 opposed rows; old CB's tiny kernel gives .9998/.9986 with zero opposed rows. The larger kernel changes the training field, rather than merely broadening samples at evaluation. This is one saved batch and a fixed table enumeration, and does not establish that zero or smaller jitter is the correct learning rule. E22 and old CB both already fail the canonical toy gate.

The effect is problem-dependent. MNIST2000 G/prior cosines are .7104/.3474 for E22 and .7433/.4111 for RA2. RA2's canonical MNIST active FD=.81414 and recall=.7139 improve on old CB's1.48883/.3228, while E22 remains better at .54449/.8472. Large gradient differences alone cannot explain or rank learned quality universally.

## Within-cell support blind spot

The frozen CPU reconstruction at1000 has887 flagged clean rows and913 flagged emitted rows. Only6/64 cells have a categorical count discovery. Of those flagged rows,805 clean and788 emitted rows are in the58 cells without count discovery. The real calibration point-eligibility fraction is.9531, emitted fake is.1016 and clean table is.1250. At2000 the corresponding fractions are.9531/.1572/.2861, with710/860 clean/emitted flags and5 discoveries;688/820 flags are in cells without discovery.

Full categorical mass TV is.3096/.2734 at1000/2000, so full counts are not exactly balanced. The existing multiplicity-corrected count test detects few discrepancies, and cannot see the much larger support difference within those cells. Decomposing each cell descriptively by current eligibility increases TV to.8604/.8008. This uses no oracle mode label. Fifty-six and52 cells without count discovery still have lower eligible emitted mass than real calibration mass. Real calibration has zero BH flags against its own null; this is an in-sample calibration description, not an independent false-positive qualification.

The missing category is independent of the previously diagnosed parent/target accounting restriction. Simply recovering count-certified moves still leaves few actions and no signal for many same-cell holes. The support score's earlier rare false-positive failures also remain unresolved. No score, support law, Q, guard, partition or acceptance gate was changed in this diagnosis.

## Scope and next test

Only CPU reads, forward/backward probes and descriptive counts were run. There were no optimizer updates, extra seeds or CUDA contexts. The stability snapshots reuse its saved CPU projection and sampling; they do not reproduce CUDA projection RNG or the CUDA birth/death snapshot exactly (archived GPU flags888/711 versus reconstructed887/710). All source/checkpoint/data/noise hashes were verified unchanged.

The qualified next investigation is one real-only, even-fit inside/outside partition evaluated on untouched odd real and emitted fake counts, with the existing exact count law and multiplicity accounting. Its support boundary must not reuse the odd-calibration p threshold used for this descriptive table. Parent availability, full-reference targets, unique-parent and supported-deletion guarantees need independent checks. This report qualifies no new score or integration patch. The conservative accounting and lineage experiment proceeds separately.

See [LEADERBOARD.md](LEADERBOARD.md), [state-diagnosis.json](https://github.com/255BITS/ParticleGAN/blob/bdf05d1be0f68cfdb0c71e81e7e0d3cce477572f/reports/toy100/lrfree-search/feature-cells-cb64-ra/fixes/performance/training-regression/state-diagnosis.json), [support-count-diagnosis.json](support-count-diagnosis.json), and [PROTOCOL.md](PROTOCOL.md) for measurements, receipts and runnable commands.
