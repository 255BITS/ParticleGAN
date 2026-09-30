# Prospective latent-geometry isolation preflight

Written before the [synthetic script](simulate.py) was run. The user permits
the added controller to compare learned particle positions with other learned
particles, while forbidding added reads of generated samples or real data.
This probe uses only synthetic latent tables. It is not a native trainer
candidate or an A2 repair.

## Score and candidate set

Use the standard local-outlier-factor (LOF) score from Euclidean k-nearest
neighbors, excluding the point itself. For point `i`, `kdist(j)` is neighbor
`j`'s kth-neighbor distance; its reachability distance to `j` is
`max(dist(i,j),kdist(j))`; local reachability density `lrd(i)` is the reciprocal
of the mean reachability distance; `LOF(i)` is the mean neighbor `lrd` divided
by `lrd(i)`. The proposed added mobility is
`a_i=max(0,1−1/LOF(i))`; the row's complete gain would be
`s_bulk+(1−s_bulk)a_i`. There is no hold, exclusion, p-value, or gradient
history window. Candidate neighborhood sizes are exactly `{5,10,20,40}`.

## Synthetic tables and fixed decision

For calibration, use NumPy `default_rng(20261005)` and dimensions `{2,8}`;
for separate validation, use `default_rng(20261006)` and dimensions `{3,16}`.
For each dimension, generate four tables:

1. Normal: 1,024 standard-normal legitimate rows.
2. Anisotropic: the normal rows transformed by a diagonal axis stretch of
   `10` on the first axis and a fixed orthogonal rotation.
3. Heavy-tail: 1,024 independent unit-variance t5 legitimate rows.
4. Rare group: 1,008 standard-normal rows plus 16 tightly grouped legitimate
   rows centered `6 sqrt(d)` from the origin along axis one, with independent
   normal coordinate spread `.25`.

Append eight planted isolated rows to each table. Construct each as a random
unit direction multiplied by `6` times the median norm of that table's
legitimate rows, plus independent normal coordinate jitter `.02`. For the
anisotropic table, construct isolates in the *untransformed* normal table and
apply the same stretch and rotation. Tags for the legitimate rows, the rare
group, and the planted isolates are retained only by the synthetic evaluator.
The LOF implementation sees only the unlabeled positions.

Evaluate all four `k` values on the calibration tables. A `k` qualifies only
if **every** table has legitimate-row mean gain ≤ `.07`, legitimate-row p95
gain ≤ `.30`, planted-isolate mean gain ≥ `.40`, and at least `.75` of planted
isolates have gain ≥ `.30`. The rare group's mean gain must also be ≤ `.10`.
For the normal table at each dimension, check that a uniform rescale by
`1e−3` or `100` and a fixed orthogonal rotation change every gain by at most
`1e−6`. Choose the *smallest* qualifying `k`, or reject the mechanism if none
qualifies. Then run exactly the same criteria on the separate validation
tables at dimensions `{3,16}` with the chosen `k`; a failed validation rejects
the mechanism. These thresholds are synthetic feasibility requirements, not
task gates, and they are fixed before any output is inspected.

If this passes, a native implementation still needs prospective runtime
planning for a 20,000-row table and a separate performance test. Latent
isolation can differ from generated-output error, especially with nonlinear
generators, so a synthetic pass alone cannot establish that this fixes PR
#155.
