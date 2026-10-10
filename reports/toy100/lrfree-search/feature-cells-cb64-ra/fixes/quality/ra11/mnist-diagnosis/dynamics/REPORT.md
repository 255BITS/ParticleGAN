# RA11 MNIST dynamics diagnosis

Source and completed JSON review **PASS**. No PT was loaded; no model, forward, draw, update, scorer or CUDA context was used. This diagnosis leaves the frozen candidate and running original queue unchanged.

## Recorded failure path

| Step | RA11 active FD | Recall | Applied sigma | G / sigma LR | D LR |
|---:|---:|---:|---:|---:|---:|
| 100 | 109.452670 | 0 | .032943636 | .0010625 | .000042230466 |
| 250 | 108.793480 | 0 | .040219925 | .0010625 | .000024820175 |
| 500 | 107.133848 | 0 | .058408342 | .0010625 | .000020074307 |
| 1000 | 101.020951 | 0 | .138781622 | .0010625 | .000021284863 |
| 2000 | 40.544410 | 0 | 1.310713768 | .0010625 | .000019760388 |

The failure is present at step 100, before the late large sigma. Final precision is .28125, confident class coverage 1 and clipping .563290. Clipping alone is not a clean generator quality measure: correctly learned near-boundary background pixels can also cross the boundary under the ordinary output noise.

All accepted mean, ordinary, isolation and novel moves remain zero through all 250 reactions. There are no copy resets or lineage edges. Every recorded paired EMA stamp is ineligible (zero eligible/coherent rows), so evaluation uses FAST. The added mean phase never fires or previews; its invalid witness cannot directly move the table. This excludes direct mean-copy and serving-swap causes, not every difference between the packages.

## Exact source and reference checks

Seventeen optimizer/controller/noise/averaging methods have identical body ASTs across RA11, RA4 and corrected original CUDA E22, ignoring docstrings. The optimizer kernel files are unchanged. RA11 and RA4 also have identical `_generate`, `sample`, backend latent perturbation and bounded geometry ASTs. Mean frame fitting draws no RNG; extra mean observations preserve registered buffers, modes, gradients and global/dedicated Torch streams. The actual image fixture definitions have no stochastic eval layers or custom buffer registration. Arbitrary external Python state remains outside that observation contract.

All three runs share initial G/D/prior hashes and the same 39-dimensional active evaluator mask, mean, standard deviation, reference and classifier hashes. The corrected CUDA E22 comparator has final active FD .544488, precision .869141 and recall .847168. RA4 has .393402, .895508 and .786133. The older scaling-portability score uses a different normalization and was excluded from this comparison.

## What changed in the training path

The RA7 rate configuration retained by RA11 changes `lr=.00425 → .0010625`, `prior_lr_mult=2 → 8`, and `d_lr_mult=1 → 4`. Prior and D base rates remain .0085 and .00425; G and learnable log-sigma share the quarter base. All RA11 G/sigma tester scales remain 1, so there is no adaptive compensation of this reduction on MNIST.

The inherited controller computes payoff error from the generator-versus-discriminator loss gap and multiplies D LR by `1/(1+payoff_error**2)`. Saved RA11 payoff error is 10.029 at 100 and 14.632 at 2000, with persistent aligned G gradients. D remains strongly damped. RA4/E22 reduce that error, recover D rates and return sigma to .029 by 500. This is a concrete feedback path consistent with stalled learning and later noise amplification; the logs do not isolate which trajectory difference initiated it.

## Next prospective work

The smallest existing config-only prospect is to restore the historical G+sigma base: `lr=.00425`, `prior_lr_mult=2`, `d_lr_mult=1`, keeping prior/D bases, learnable-noise formula, support/count, serving and evaluator laws intact. Successful original MNIST runs support testing it. **It is not a proved repair**: the packages differ in other laws, and toy/grid retention under that configuration is unmeasured. No config or source was changed here.

A raw saved-state probe is unnecessary for the present validity report. If the next investigation specifically needs clean generator saturation separated from late noise amplification, the protocol reserves one separately sealed deterministic 32-row functional G/EMA probe at 100 and 2000. It has not been implemented or run. Generic high-dimensional support, failed birth recovery and coupled optimizer dynamics remain unresolved; no broad package recommendation follows from the toy/grid passes alone.

`receipt.json` preserves all stored scalar curves, progress losses, evaluator identities and exact source comparisons. `FROZEN.json` pins this report, closed log and every source/input byte.
