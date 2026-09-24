# Exact diagnosis of the first cold ring critic-fit exception

The original cold ring run stopped inside its local D fit at update 472;
it produced **no update-1200 ring verdict**. A passive replay with unchanged
frozen sources matched all 23 original 20-update fit-evaluation counts through
update 460, then saved the first failed fit. The [capture archive](continuous-evidence/pr84-critic-refinement-failure-capture/manifest.json)
contains its source-bound state (`failed-fit.pt` raw SHA-256
`19ceb38a92ba95ffaccd3aafbda55bbd77917765612666d923e986e5523d6965`),
declaration, log and observer. The [independent one-fit replay receipt](continuous-evidence/pr84-refinement-failure-audit/diagnosis.json)
comes from [`pr84_refinement_failure_audit.py`](pr84_refinement_failure_audit.py).
It verifies all original cold source hashes, the captured pre-step and
accepted-D state hashes, the finite bank and Adam metric, the best finite
point's loss, and bytewise identity of the repeated invalid critic parameters
and gradients. It performs no outer training update.

The failure is a **float32 strong-Wolfe cubic interpolation overflow after a
huge but finite trial**, not a nonfinite accepted-D state or a nonfinite
starting fit objective. Its first 47 fit evaluations were finite; the best
same-bank penalized D loss was `0.23826262355` versus `0.28395605087` at
the fit start. Re-evaluating the saved best point gives exactly the recorded
loss, a finite penalty `0.00086493546`, and finite D gradients (raw infinity
norm `0.0073673`, so no convergence claim). Evaluation 48 tried another
finite point with loss `8.65885286e19` and gradient infinity norm
`1.10365253e18`. Its line-search bracket was `[0,1]`; endpoint directional
derivatives were `−6.83729267` and `6.87753354e20`. PyTorch's cubic formula
formed `d1=4.27987769e20`, whose float32 square overflowed to `inf`; the
interpolated step became `NaN`. Evaluation 49 then put all eight learned D
parameter tensors and their gradients into NaN (20,161 elements). The
standalone replay reproduces that exact trial, including its NaN tensors.
The recorded Adam metric is finite and is used only for fit diagnostics, not
as the L-BFGS search metric.

At the saved pre-step after update 471, G, prior, D and both optimizer moment
states are finite. A **posthoc single-state** reproduction of the frozen
4,096-draw evaluation gives 3 modes and HQ `0.755126953125`. This indicates
incomplete ring acquisition at that intermediate checkpoint. Earlier ring
quality was not captured, so it does not establish deterioration; nor does
one checkpoint determine the uncompleted 1,200-update cold ring gate.

A bounded numerical policy can reject a nonfinite L-BFGS trial and restore
the lowest previously evaluated finite point, recording the rejection and
leaving D Adam moments, training RNG and G unchanged. If the initial fit
point is nonfinite, it must still fail. The saved update-472 best point shows
that this fallback is *available* in this incident; it does not show that a
continued ring run acquires all modes or remains stable. The next checks are
an exact resumed short segment and then the original full cold gate under
the frozen guarded source. This diagnosis supplies no quality-based fit
selection, extra fit, or new schedule.

Reproduce the audit from the two tracked archives in the source-bound root
worktree with one CPU thread:

```bash
/tmp/pr38-default-env/bin/python reports/toy100/pr84_refinement_failure_audit.py \
  --capture reports/toy100/continuous-evidence/pr84-critic-refinement-failure-capture \
  --original reports/toy100/continuous-evidence/round5-critic-refinement-cold \
  --output /tmp/pr84-refinement-failure-audit.json
```

The script refuses mismatched source or captured state hashes. The recorded
posthoc grade uses update 471's frozen evaluation draw and known ring means
only for diagnosis, never for fitting or selecting D.
