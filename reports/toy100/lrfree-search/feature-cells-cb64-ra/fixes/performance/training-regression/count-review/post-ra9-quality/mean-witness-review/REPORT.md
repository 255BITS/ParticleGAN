# Source review of the fixed mean witness

This review concerns conditional mathematics and source ownership before the
single proposed CPU prototype run. It does not interpret checkpoints, execute
the helper, fit a chart, evaluate a model, or establish inferential validity for
the saved training data.

The score uses the fitted even-reference chart and topology, even group
centers and positive RMS scales, and current clean EMA group means. Each group
direction is the even clipped mean minus the EMA clipped mean, normalized to
length at most one; a zero residual gives direction zero. No odd value, odd
group count, or odd score selects these definitions. Missing or insufficient
even groups, zero scales, missing EMA groups, and invalid geometry veto the
whole witness. Every odd observation, including a zero-direction observation,
remains in the mean and unbiased sample variance.

For R=sqrt(rank/Q), radial clipping gives norm(psi)<=R. An EMA group mean is a
convex mean of clipped vectors, so norm(m)<=R and X=u dot (psi-m) lies in
[-2R,2R]. The reflected variable (2R-X)/(4R) lies in [0,1]. Applying the lower
tail form of [Maurer and Pontil, Theorem 4](https://arxiv.org/pdf/0907.3740)
therefore gives

`mean(X)-sqrt(2*variance_ddof1*log(2/alpha)/n)-(7/3)*(4R)*log(2/alpha)/(n-1)`.

The source checks n>=2 first and fires only for a strictly positive lower
bound. This bounds the directional mean under a fixed-score iid law. It does
not test equality or certify a stable population.

The count family contains actual K original-cell tests, 2K refined tests, two
global support tests, and this one witness. All decisions use Q/(3K+3), with
fresh raw count pvalues and recomputed masks; overlap between tests does not
require independence for a union bound. The validity of each individual law
still requires its own sampling assumptions.

The witness uses the ordinary mean across odd real rows. Its conditional
expectation weights the true real group mixture. The separate copy objective
uses even empirical group masses. These quantities are not interchangeable,
and a positive global witness does not certify every group or bound the
empirical squared objective. Actual preview progress must be checked against
that objective independently.

Before measurement, the copy eligibility was tightened to require inside
membership in both FAST and EMA, for original child/parent rows and actual
offspring. Category retention by itself had allowed supported outside rows in
the broader draft. The ownership reviewer separately covers row reservations,
one-draw paired perturbations, exact packets, and scratch state changes.

The learned discriminator, real FIFO, generator, and EMA share training data
and history. The fixed-score iid premise is not established for these saved
states; the prototype labels its results empirical and non-authoritative.
Its EMA means also use clean anchors. Nonlinear learned features, latent
jitter, and output noise separate this clean-table objective from emitted
feature moments and the original quality gates. No unconditional, repeated
adaptive, per-group, stationarity, or quality certificate is claimed.

The previous squared standardized output residual is not the new clipped
unit-direction scalar mean. It supplies no power verdict for this witness.
The fixed measured mean, variance, range penalty, and legal action capacity
must decide the single run; no result-dependent clipping or score variant is
part of this review.

The first private receipt attempt stopped at a package digest guard because
it included an extra `particlegan/` path prefix. The corrected guard uses the
declared canonical digest relative to that directory. The first helper and
failure log are retained; no owner source or input was changed.
