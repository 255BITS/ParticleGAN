# Gaussian scale-mixture preflight: rejected

The [prospective design](SPEC.md) was fixed before [execution](simulate.py).
The run used only synthetic standardized gradient summaries, with three
independent synthetic tables per grid point. [Full numeric results](results.json)
and the [tail-friendly log](results.log) are archived.

| Noise summary | Cases passed | Largest null mean gain | Largest null p99 gain | Nonconverged fits |
| --- | ---: | ---: | ---: | ---: |
| Normal | 56/75 | .0962 | .4349 | 10 |
| Standardized t5 | 63/75 | .0788 | .7204 | 0 |
| Total | **119/150** | — | — | **10** |

The preset null mean gain limit was `.05`. For example, at `π=.05`,
`|δ|=.5`, `m=128`, and `ρ=.5`, the three normal replicates gave null mean
gains `.0943`, `.0897`, and `.0851`. The two preset normal drift-power targets
were met (minimum drift mean gains `.9564` and `.9714`), but gain leaked to
null rows. The largest null fraction receiving gain above `.5` was `.0161`
under t5 summary noise. The pure-null normal table also had one nonconverged
fit; the pure-null t5 tables returned zero gain.

**Decision:** reject this Gaussian scale-mixture gain. It is not ported to the
trainer or tested on native tasks. The finding rejects this fitted model under
the fixed criterion; it does not rule out every empirical-Bayes approach.
The E4 A2 violation remains unresolved.
