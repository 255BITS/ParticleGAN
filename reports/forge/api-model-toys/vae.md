# Public-API particle VAE toys

The soft categorical VAE **PASS**es the full two-anchor protocol; the hard
constant-KL VAE completes its budget but **FAIL**s sustained generation quality.
Both run the same `bispf-optimized-quintic-v1` trainer on protocol seed 0 at frozen
source `bee9f06ed21695e518c1cb702e12a385ab9389e8`. Both are declared Tier 1
requirements. Their gates remain provisional: these results do not calibrate
a scientific screen or qualify any public default.

Historical [12-arm scout](../../mog-vae/README.md) and
[five-arm stability study](../../mog-vae/stability/README.md) found useful
stochastic encodings but unresolved prior coverage, mode width and late
stability on 100 modes. Those experiments retain their original optimizer,
seed, initialization, budget and evidence identity. This two-mode cohort is a
narrow, reproducible public-API regression question, not a historical rerun.

The hard posterior selects one component, adds the prior's fixed local noise,
and retains `KL=log(16)` in the reported negative ELBO. Its discrete gradient is
the API's biased straight-through approximation. Soft categorical inference
uses `KL=log(16)-H(q)` and the public leave-one-out score estimator. Removing
this variable KL would create a stochastic AE; no such model is called a VAE.

The frozen setup uses balanced means `(-1.5,0)` and `(1.5,0)`, isotropic data
standard deviation .05, a 16-by-2 learned MoG with fixed local sigma .025, and
width-32 two-layer LeakyReLU encoder/decoder/critic networks. The Gaussian
decoder observation sigma is .03. The task allows 1,200 updates of 64 inputs,
24 fixed scoring observations, 4,096 held-out inputs per observation, and five
passing terminal checks. Training uses two posterior draws; evaluation uses
eight. Live decoder means and actual Gaussian likelihood samples have separate
quality metrics and gates.

The baseline is the exact declared `bispf-optimized-quintic-v1` recipe. The task
changes encoder mode, observation model and fixed resources only; learning
rates, optimizer law, prior learning and SBNP regularization remain recipe-owned.
Both modes share the same target batches, public deterministic initializer,
network sizes, initial prior, observation cadence and allowance. The Gaussian
negative ELBO is multiplied by `.03²`, making its 2D reconstruction term MSE;
the variable categorical KL remains in the objective with its required weight.
The adversarial and recipe prior objectives are also retained.

Training executes through public `Recipe.encode`, `ParticleEncoding`, loss,
optimizer, critic-penalty and prior-regularizer components, with `UpdatePolicy`
owning schedules and update lifecycle. `GANTrainer` explicitly excludes encoder
hosts. Constructor, critic data, generator data, prior identity, prior noise,
posterior noise, input/output noise, evaluation and penalty streams are isolated
and checkpointed. Evaluation uses local generators and cannot change training
state. Encoder, generator, critic, prior, policy and optimizer state are saved
for exact continuation.

The identity diagnostic promotes the same observed predictions, targets and
posterior values to float64, calls the public negative ELBO, and compares it
with an independent Gaussian log-density calculation. It adds no model forward
or sample and leaves training arithmetic unchanged. This prevents float32
cancellation before training from rejecting a valid bound. Injected recipes
must already match the task's model, encoder, resources, observation model and
1,200-update schedule horizon; incompatible injections fail explicitly.

The gates require reconstruction MSE ≤.05; shuffled-code reconstruction ratio
≥5; posterior mutual information ≥.3 nats; local-noise second-moment ratio
between .8 and 1.2; KL identity error ≤1e-5; and Gaussian negative-ELBO identity
error ≤1e-4. Both decoder means and actual likelihood samples require at least
.85 within radius .2 of an anchor, quality mass TV ≤.15, and both modes.
Nonfinite metrics fail. Oracle samples pass; shuffled reconstruction, collapsed
posterior, missing local noise, missing KL and collapsed prior/likelihood samples
are destructive controls and must fail their relevant bounds.

Both runs complete all 1,200 updates and 24 post-update observations, with
4,096 evaluation inputs at each check. Initial encoder, decoder, critic and
prior hashes match across modes, as does the target batch-sequence digest
`5854de9ddd8e4759cb31dd1a98e0b5e72ee7e12540e14a540c53837fbbe20b06`.
Each encoder, generator, prior and discriminator receives 1,200 updates;
all trained states remain finite, the critic penalty applies 1,200 times,
and the RNG guard records zero deviations. Execution takes 7.414 seconds for
hard inference and 7.983 seconds for categorical inference on the recorded CPU
runtime; these are quality tasks, not throughput comparisons.

The categorical VAE passes 19/24 post-update checks and its final **15** checks
consecutively. At update 1,200, reconstruction MSE is .00377060, shuffled-code
MSE is 2.23370 (592.40 times higher), and posterior mutual information is .69348
nats. Mean posterior effective component count is 4.2886 per input and aggregate
effective usage is 8.5801 of 16. Local latent-noise second-moment ratio is
.995591, decoded posterior pair RMS is .034606, categorical KL is 1.31662 nats,
and Gaussian negative ELBO is .330942 nats. KL identity error is 4.91e-7 and
ELBO identity error 4.44e-16. Independent decoder means have HQ .993652 and
quality mass TV .009766; actual Gaussian likelihood samples have HQ .985596
and TV .014404. Both laws cover both modes. These measurements establish useful
stochastic input-dependent inference and the declared prior-quality gates on
this two-mode law; they do not establish calibrated within-mode variance.

The hard VAE passes 14/24 post-update checks, ending with only **two** consecutive
passes. Its final update independently passes every metric: reconstruction MSE
.00558006; shuffled-code MSE 2.36365 (423.59 times higher); posterior mutual
information 1.4720 nats; local-noise ratio 1.005192; decoded pair RMS .031370;
constant KL 2.77258897 nats (`log(16)`, not zero); and negative ELBO 3.79742 nats.
Final decoder-mean HQ/TV are .955322/.070068, and likelihood HQ/TV are
.915039/.084961; both modes remain covered. The hard posterior uses one
component per input and 4.3579 effective components across held-out inputs.

The five required terminal observations are updates 1,000, 1,050, 1,100, 1,150
and 1,200. Updates 1,000 and 1,100 fail all four generation bounds below; updates
1,050, 1,150 and 1,200 pass. The limits are HQ ≥.85 and quality mass TV ≤.15.

| Hard-VAE update | Decoder-mean HQ | Decoder-mean TV | Likelihood HQ | Likelihood TV |
|---:|---:|---:|---:|---:|
| 1,000 | .753418 | .246582 | .757080 | .242920 |
| 1,100 | .726318 | .273682 | .703125 | .296875 |

This is recurring late distribution instability: generation also fails at
updates 450, 500, 700, 750, 900 and 950, with intervening recoveries. It is not
an ELBO arithmetic rejection or loss of the input code. Throughout the hard
terminal window reconstruction MSE stays in [.002699, .008892], shuffling
worsens it by at least 255.78 times, mutual information stays above 1.3339 nats,
the local-noise ratio remains 1.005192, KL identity error is 7.62e-9, and ELBO
identity error never exceeds 1.78e-15. The initial ELBO identity errors are also
0 for hard and 2.27e-13 for categorical inference. The earlier float32
diagnostic issue was repaired before the frozen run and cannot explain these
HQ/mass failures. A passing final state or earlier checkpoint does not replace
the failed five-observation requirement.

Actual-training GIFs show fixed-axis targets beside observed reconstructions,
decoder means and likelihood samples:
[hard VAE](../../toy_audit/api_contract/model_modes/media/api-model-vae-hard.gif)
and [categorical VAE](../../toy_audit/api_contract/model_modes/media/api-model-vae-categorical.gif).
Media selection uses nine genuine observations; all 24 numerical checks remain
in the verdict. The [single current goal readout](README.md) owns the comparison
table, and the [publication receipt index](../../toy_audit/api_contract/model_modes/publication.json)
owns exported identities. The publication exporter was repaired after numerical
completion to tolerate omitted inactive recipe fields; it reused saved receipts
and did not rerun training, change frozen source, or revise numerical verdicts.

Retain the categorical case as observed passing coverage and retain the hard
failure in the full comparison. Before another training hypothesis, a bounded
saved-state analysis should test whether hard posterior routing leaves more
uniform-prior rows weakly constrained. Aggregate posterior usage is smaller
for hard inference (4.36 effective rows versus 8.58), but that is an association,
not an established cause. Use only the two saved final checkpoints, one fixed
4,096-input evaluation panel with eight posterior draws, and one 64-input
gradient panel; cap analysis at 60 CPU seconds with zero optimizer updates.
Measure posterior visitation, decoded distance and mode for each prior row,
plus separate ELBO/adversarial/prior gradient norms. Reject this explanation if
rarely visited rows do not account for larger decoder errors or generation
mass imbalance. A later structural repair would require a separate proposal
and fixed global trainer comparison; no tuning or follow-up run is launched.

Bulk stdout, checkpoints, raw receipts and observations remain under
`/nvme/dev/particlegan/artifacts/api-model-toys/research-v1/model_vae_{hard,categorical}`;
each run has a tail-friendly `stdout.log`. Only compact metrics, provenance and
the actual-training GIFs are published.
