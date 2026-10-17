# Signed-location mixture preflight: oracle fixture passed

The [prospective spec](SPEC.md) was fixed before the synthetic fixture ran;
the only amendment replaced broken SciPy bounded minimization with the same
bounded golden-section rule before any data were generated. The [script](simulate.py),
[full numeric results](results.json), and [tail-friendly log](results.log) are
archived. No benchmark data or native training were used.

| Summary-noise family | Cases passed | Largest null mean gain | Largest null p99 gain |
| --- | ---: | ---: | ---: |
| Normal | 75/75 | .0273 | .3867 |
| Standardized t5 | 75/75 | .0316 | .6250 |
| Total | **150/150** | — | — |

All fits converged. For the two specified normal-noise power checks, minimum
drift mean gains were `.9682` at `(m,|δ|)=(128,1)` and `.9786` at
`(700,.5)`. The largest fraction of null rows with gain above `.5` was
`.0152` under t5 noise. The pure-null normal and t5 cases also passed.

The fit has an oracle advantage: it knows the true gradient direction,
effective information under correlation, and summary-noise family, and it
assumes one shared signal magnitude. Passing this test only shows that a
signed-location likelihood can avoid the *scale-mixture* failure on this
fixture. It is **not** an online detector, native result, or A2 repair.
An independently specified memory and dependence estimator would be needed
before this route could be tried in the trainer.
