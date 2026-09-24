# Separate estimator noise from the objective's restoring behavior

There is still no qualified replacement for LR decay. The latest cheap tests
reject two apparent fixes: averaging sixteen banks repairs the original bad
step but fails continued training, and local MMD can improve while deleting a
mode. Fixed-support memory passes a long hold but rejects a genuinely unseen
component after incomplete bootstrap. These failures remain on the same target
distribution; no whole-target shift is required.

| Candidate / cheapest decisive test | Result | Decision |
| --- | --- | --- |
| Pre-start sampled group anchors | Acquisition, own hold1200 and response pass; one omitted minibatch deletes a mode | Reject memoryless rule |
| Two-bank fixed-support memory | Saved44, warm200, hold1200 pass; later ordinary bank's22 missing-mode samples all rejected | Reject frozen identities |
| Fresh-data mass discovery outside fixed cells | Ring acquisition eventually supported; a new component inside an old cell never proposed | Reject outside-only discovery |
| Actual Adam with16-bank D/G means | Saved1325 repaired; own consecutive saved windows24/44 | Reject before warm |
| Local emitted-law MMD + donor search | Cold acquires; first fresh-warm update loses a mode while population MMD improves | Reject before neural training |
| Emitted-law forward KL + donor search/EM | One cold update0→8 modes/HQ .999756; own warm bank retains8/HQ .999512 | Continue short free-output tests only |
| Profiled sharp critic value over frozen features | Convex inner-fit/value-bound diagnostic in progress | No training claim |

The [mean-field screen](mean16-continuous-screen.md) preserves actual new Adam
moments, constant rates and exact original replay controls. Its failure is not
an optimizer-counter or RNG mismatch. The [finite-bank diagnosis](pr84-finite-bank-variance-geometry.md)
finds substantial critic-side noise, but also nonmonotone local mean-field
geometry. Reducing noise alone does not make that tested update stable.

The [MMD audit](local-mmd-independent-audit.md) verifies the emitted-Gaussian
formulas and replacement deltas independently. The whole losing proposal also
improves true population MMD, so simply checking the whole proposal on more
data would not cure this example. A particular mode-deleting submove is harmful
in population, but a single held-out bank incorrectly approves it. This
distinguishes a data-estimation failure from an objective/finite-particle issue.

The [new forward-KL filter](round8-forward-kl-first-bank.md) evaluates smoothed
target cross-entropy under the actual emitted Gaussian-mixture family.
Observed-data global replacements address allocation and equal-weight EM
refines locations. Both current one-bank gates pass with each state's own
native real bank; the earlier common-bank warm result remains archived and
does not substitute for that test. The [independent audit](forward-kl-free-output-independent-audit.md)
checks the formulas, exact donor selection, EM step and data-stream hashes.

Its current guarantee is descent of a finite positive5×5 quadrature objective.
A9×9 numerical audit also decreases at every accepted one-bank step, but
the cold endpoint gradient differs materially between the two rules. This is
not an exact continuous-KL optimum or a neural training result. The
[research-scope note](forward-kl-research-scope.md) connects the experiment to
current inclusive-KL and Gaussian-mixture EM research and states why their
theorems do not automatically cover this model.

Next tests, in cost/failure order: consecutive cold/warm output updates;
an omitted bank after acquisition; initially missing support followed by real
signal; a same-target model error; and only then actual neural realization and
the strict44-window gate. Any memory-bearing neural survivor must pass exact
complete-state restart, warm200, borrowed hold1200, cold acquisition, its own
hold, immediate error response, and the longer stationary run. Full mass/shape
fidelity and the shared production22 gates remain required before promotion.

The separate critic lane freezes nonlinear features, solves the original
sharp Rp+b_cap readout objective, and tests a generator derivative of that
same profiled value, including fake-cap input dependence. Certified value
bounds must decide acceptance; an inner numerical fit cannot simply be called
an exact best response. This is a bounded structural diagnostic, not a sweep.

[Round8 memory evidence](continuous-round8.md) and earlier failures remain
available. The [emitted-law/chart note](emitted-law-and-chart-principles.md)
proves only a scoped unrestricted-critic no-pure-Nash statement; it is not an
impossibility theorem for the actual finite capped critic or bounded quality.
No unavoidable obstruction has been established. Production PR60 head remains
983d037a7028afc5c1b0df4d4097eff8e6abe9b5. No CI wait or seed sweep is involved.
