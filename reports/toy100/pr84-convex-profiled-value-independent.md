# Independent fixed-feature value assays

The unchanged saved1530 rule certifies one finite-bank value decrease at
both independent warm state1325 and cold state472. Warm quality improves;
cold quality improves but coverage remains three modes. These are copied
float64 states with a frozen metric and zero new moment updates, not native
training or acquisition passes.

| State | Paired noisy HQ before / after | Modes | Accepted scale | Certified value decrease | Readout closures |
| --- | ---: | ---: | ---: | ---: | ---: |
|1325|.996582 /1.000000|8 /8|1/4|.01199426|437|
|472|.752441 /.968018|3 /3|1/4|.01195943|433|

Both reject scales1 and1/2 before accepting1/4. Each point has the same one
L-BFGS attempt, maximum100 iterations /200 closures, and strict target gap
`1e-7`. All returned inner fits remain `NOT_CERTIFIED` at that strict gap.
Nevertheless, the trial's global convex lower bound exceeds the base's
evaluated upper bound by the listed amount, which verifies the decrease
without claiming an exact critic optimum. Four final read-only readout
gradients per assay are additional to the closure counts.

The [1530 report](pr84-convex-profiled-value1530.md) gives the finite paired
objective, original-cap ellipsoid bound, fixed bias gauge, and limitations.
No objective, solver budget, coefficient, tolerance or acceptance rule is
changed here. Each assay freezes sixteen native-sized D banks and the
nonlinear features of its current accepted critic. It uses the partial
negative sharp D loss, including fake-input cap derivatives. Target centers
and quality measurements never enter the proposed direction or acceptance.

At1325, the base D-loss interval is `[.62608082,.63016685]`; the accepted
interval is `[.64216111,.64792462]`. At472 they are
`[.23301099,.23438020]` and `[.24633963,.24975213]`. The remaining inner gaps
are visible and substantial compared with the desired tolerance. These are
float64 numerical bounds, not interval-arithmetic certificates.

The two central derivative checks agree within `5.40e-9` relative error at
1325 and `5.30e-8` at472, with zero observed LeakyReLU or cap switches.
Both original first D gradients and next G banks are exact. At1325, a
one-bank control reproduces all original model and both optimizer states.
The original472 fit failed before G; its reference therefore comes from
the separately archived exact guarded-G reconstruction. That reference's
full G/prior/Adam state is reproduced exactly, and its post-Adam metric is
held fixed here. The cold critic starts from the captured best finite D.
No nonexistent original post-G capture is assumed.

Cold472 clean supports remain assigned to the same three nearest ring
centers before and after the accepted move. Post-hoc displacement has a
positive projection toward a nearest missing center for12/12 particles,
yet none crosses into a missing center's nearest region. Thus a useful
local field and better HQ do not demonstrate allocation or eventual
acquisition. The next authorized diagnostic is at most eight consecutive
moves, including this first move, on exactly this frozen feature class,
finite bank and metric. It will stop on certificate rejection/rest and
retain every own state; it is not a native training branch.

The [archive manifest](continuous-evidence/convex-profiled-value-independent/manifest.json)
binds both raw results, full tensor payloads, exact source copies, guarded
generated runners and logs. Payloads contain the selected input phases,
frozen D/G banks, features, metric, original and fitted readouts, proposal,
and accepted G/prior state. The input and caller RNG are unchanged. The
guarded wrapper verifies the original1530 algorithm source hashes before
execution and changes only declared state/provenance selectors.

Run each case with the same single-thread CPU/AVX2 environment recorded in
the1530 report:

```bash
/tmp/pr38-default-env/bin/python -u reports/toy100/pr84_convex_profiled_value_assays.py \
  --case 1325 --states FULL_CAPTURE_V2/selected-states.pt --output NEW_1325_OUTPUT
/tmp/pr38-default-env/bin/python -u reports/toy100/pr84_convex_profiled_value_assays.py \
  --case 472 --states FAILURE_CAPTURE/failed-fit.pt --output NEW_472_OUTPUT
```
