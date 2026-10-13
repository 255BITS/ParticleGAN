# Prospective AMSGrad on the near-density role policy

One **bounded, globally applied `amsgrad=true` delta on the exact m110 parent**
is supported as an unmeasured optimizer interaction. Acquisition support is
stronger than density support. Neither existing AMSGrad recipe passes Ring16,
and the evidence does not predict a qualified replacement with confidence.
Complete the current spectral comparison and its audits before any admission.

The [receipt](ams-role-note.json) and [reproducer](ams_role_analyze.py) examine
15 certified Ring16/Gaussian/word endpoints, existing observation records,
saved moments and fixed canonical word inputs. Checkpoint bytes, complete
artifact manifests/state hashes and every recorded named stream pass the
existing publisher checks. No models, optimizers or gradients are constructed;
no draws, training, new gate scoring or RNG-state changes occur.

## Exact prospective delta and applicability

Parent `r1r2-role-m110-v1`, revision
`555692af38701e21f5fce7fa5b976702761ecea51f4a525ed8ab308b20439b2a`:
change only **`amsgrad: false -> true`**. Retain native PyTorch Adam, G/E and
D beta1=.5, sampled-prior beta1=0, beta2=.99, constant LR=.0085, multipliers1,
epsilon1e-8, paired R1/R2 coefficient.1 every update, learned prior and zero
extra prior regularization/noise/EMA. D betas/epsilon and prior epsilon stay
null; prior betas stay `[0,.99]`. This is distinct from the tested all-zero
and all-.5 AMSGrad recipes; local declaration recall finds no identical recipe.

**All seven existing public preflights PASS** on in-memory copies: Gaussian,
pole, token, AE, Ring16, words and clock. Original task-contract hashes remain
identical. The public flag reaches G/E, sampled prior and D; no role-specific
AMSGrad setting exists. Word G/E follow .5, its manually supplied prior group
consumes `[0,.99]`, and its D uses .5. Direct pole follows G at .5 with AMSGrad;
sampled-prior betas are unconsumed in pole/token. Original task resources,
initialization, prior distribution/capacity, data law/batches, sampling,
update/evaluation allowances and gates remain fixed. Preflight validates
capability/binding, not a future trained pass. The sole constructor exception
is Recipe's exact parameter-free inactive Coulomb-option validator; model
construction and all module forwards remain blocked.

## Evidence for the interaction

m110 reaches best covariance **.904193 at1584**, with all other bounds passing,
but still misses <=.85. Its endpoint covariance **1.083997** and minimum
eigen ratio **.111723** both fail. A substantial saved late excursion precedes
the near-pass: covariance **1.26879 / 4.00145 / 16.59699** at1384/1400/1417,
then returns to **.904193 / 1.083997** at1584/1600. These are recorded observation
changes, not measured optimizer spikes or a recovered instability mechanism.

The matched all-.5 native→AMSGrad intervention has a much steadier final16
covariance sequence: SD **.102945**, versus **1.383724** for native m111 and
**3.935175** for m110. At its actual saved Ring endpoint, retaining the maximum
second moments reduces the same-moment G/prior/D displacement L2 to
**.18353 / .19947 / .16011** of the corresponding current-v denominator steps.
This demonstrates materially active memory rather than an inert flag.
m110's zero-prior first moment is active in **105/256 rows** at its last step;
AMSGrad preserves that zero-first-momentum policy rather than adding historical
prior drift. Its denominator history could still damp sampled-row motion.

Both measured AMSGrad policies pass Gaussian and words. m110 word G is
effectively inactive: saved first-moment L2 **1.99e-18**, last reconstructed
displacement L2 **2.28e-14**. Its fixed prior/canonical panels have maximum
decoder logits **419.897 / 747.803**, minimum prior/encoded separation
**.021834 / .019505**, and every canonical reconstruction is `grape_`.
AMS111 instead reconstructs all five words, with logits **31.712 / 30.769**,
separations **.693040 / .663759** and active G displacement **4.26e-5**.
AMS000 also passes words, supporting compatibility of denominator memory with
zero prior momentum. This motivates prevention from the original initializer;
it supplies no evidence that continuing a saturated m110 endpoint would recover.

## Negative evidence and falsifier

AMSGrad worsens Ring covariance in both measured matched contrasts:
m000 endpoint **2.386427→2.960424**, m111 **1.185580→2.009387**.
AMS111 best covariance is **1.816302**; it has **0/96 covariance passes**.
Its final16 minimum-spread bound fails seven observations. Smoothing alone
does not impose directional covariance or a positive eigen floor. Combining
AMS acquisition with m110 geometry could converge to another broad failing
distribution; these benefits cannot be added as qualification.

The narrow prediction is that maximum-second-moment memory prevents the
saturated word outcome and attenuates m110's large late density excursions
while retaining its zero-prior-momentum geometry lead. The complete-repair
hypothesis requires **every original ordinary gate and clock** under this one
recipe, including covariance<=.85, eigen ratio>=.15 and five terminal full
Ring passes. Any failed original gate falsifies that complete repair. Renewed
word saturation or unabated excursions reject the proposed mechanism;
smaller variance of a failing density is not success. No further momentum,
epsilon/rate grid or automatic refinement follows.

Raw last gradients for m110 D/G/E were not saved; beta1=.5 first moments are
EMAs. Therefore **raw-to-moment contraction is unavailable**. Native m110 also
has no retained historical second-moment maxima, so an AMSGrad counterfactual
cannot be recovered from that checkpoint. Reported memory ratios use actual
AMS states with the same saved moments and do not predict a changed feedback
trajectory. Fixed float64 word panels are diagnostics, not sampled gate inputs.

Source remains `25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`
and runtime `640b92546b7cefb9a81f0bd8d7c20139fc353d7aff632548e34eec7736155ebc`.
Exact original candidate/attempt/checkpoint identities for all15 cases reside
in the receipt; m110 Ring attempt is `20aca4f16e8a4c31900b652e2c71ee3b`,
AMS111 Ring `4f034971e59c4f62b8ce1444c4f58979`, and AMS111 word
`118b25d36cb74ce9b17cd146ae43b6dc`. Preserve their original evidence, grades
and charges. No declaration, registration, queue, source/task edit or commit
is created by this note.

```sh
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-spectral-repair/optimizer/ams_role_analyze.py > reports/forge/r1r2-spectral-repair/optimizer/ams-role-note.json
```
