# One constant weak-penalty contrast

The selected hypothesis is **constant `reg_coeff: .1 → .01` on the exact
measured non-saturating AMSGrad-m110 parent**
(`r1r2-non-saturating-ams-m110-v1`, revision
`3d02499f9ad5be09db7ba59e1ba83c5de287330a687f88a09a2468a654d83ad1`).
This is a low-confidence **floor_constant** contrast, not a new structural
mechanism or an estimated optimum. This review registers no round.

## Mechanism and coefficient

[Mescheder et al., ICML 2018, §§2.3–4](https://proceedings.mlr.press/v80/mescheder18a/mescheder18a.pdf)
identify ill-conditioning near narrow distributions and establish local
convergence with positive zero-centered penalties under their assumptions and
sufficiently small gradient-descent steps. Their Dirac example also shows that
reducing the coefficient weakens damping; it does not select a coefficient for
stochastic AMSGrad. [R3GAN, §2.3](https://arxiv.org/html/2501.05441v1)
discusses an approximate density-smoothing interpretation for classic GANs, with
additional weighting/Laplacian terms, and explicitly excludes an analogous
RpGAN derivation. Neither source guarantees this finite suite. See the
[earlier score-law review](../../r1r2-objective-memory-repair/research/score-law-literature.md)
for theorem scope and the rejected variance-to-gamma heuristic.

Weaker critic-gradient pressure could permit finer local discrimination during
training while retaining both positive R1/R2 penalties. That is a trajectory
hypothesis, not measured penalty-gradient dominance or endpoint starvation.
The [closed NS comparison](../../r1r2-score-law-repair/README.md) provides the
exact whole parent: five ordinary passes plus clock, but Ring covariance fails
all 96 checks. NS endpoint covariance is 1.556546 versus paired AMSGrad's
1.679120; NS minimum spread .094075 fails its .15 bound. Saved mean radial
variance is .297 times target and tangent variance 3.053 times target. A repair
must address both directions; uniform contraction is insufficient.

**.01 is one finite tenfold reduction and a historically declared terminal
coefficient.** It is not inferred from Ring's .01 target variance, an optimum
blur bandwidth, or a paper recommendation. The selected NS parent has measured
all-task coverage and a pole gradient endpoint .027244; it supplies a concrete
control, not evidence that weak damping will preserve those passes.

## Counterevidence and falsifier

The [historical scheduled .1→.01 pole run](../../technique-receipts/2a719ec706924f0ab120961cd58f7582.json)
at LR .0085 failed: gradient 1.020285 exceeds 1 despite movement 1.006284.
The [LR .00425 counterpart](../../technique-receipts/8ff64e32bf294e5597d8deb036527075.json)
also has an overall FAIL; its passing endpoint values do not replace its
sustained gate. Both retain their different source/schedule contracts and
unmeasured peers. Existing constant .1/.3/1 failures do not establish a monotone
strength-response law. Reduced damping could restore oscillation, damage pole
stability, or lose word acquisition. Gaussian smoke acquisition already lacks
endpoint retention on the NS parent.

Only the coefficient changes: native global AMSGrad, NS loss, D/G+E/prior beta1
.5/.5/0, beta2 .99, constant LR .0085, epsilon 1e-8, learned prior and every
other field remain fixed. Both penalties apply every update; coefficient end
is null. All original task/init/prior/data/noise/update/cadence/gate contracts
remain fixed, including words' joint reverse encoder path. The numerical
falsifier is every original ordinary gate plus own clock, with Ring requiring
five full terminal passes within 1,600 updates. Improved loss or covariance
alone is insufficient.

## Unselected smoothing option

The public `optimizer_smoothing` is a DualNorm spectral scale:
`s / hypot(s, smoothing)` for matrix singular values, with analogous vector/row
damping. It is not temporal averaging and cannot be added to this native Adam
parent. Its [measured full-DualNorm parent](../../r1r2-spectral-repair/density-readout.md)
fails Ring and words, with radial variance .635 times target and tangent
variance 2.586. Damping small gradient directions might further compress the
radial direction; saved states supply no historical raw-gradient spectrum or
globally justified scale. It remains unselected, with no extra arm.
