# Lower penalized critic loss repairs the saved G guidance

Keeping the generator and prior fixed, one bounded fit of the **existing
penalized critic objective** improves heldout D loss and reverses the harmful
G direction at all three inspected states. This is evidence that the saved
wrong guidance can be repaired within the current critic architecture/loss.
It is **not** a best-response certificate: all three fits retain substantial
gradient residuals, and heldout residual norms increase.

No training candidate, LR schedule, regularizer change, target-based controller,
seed sweep or full episode is introduced. The diagnostic copies the accepted
critic at updates 1325, 1530 and 1539; G/prior and original Adam state stay fixed.
It constructs 1024 training pairs (eight 128-pair batches) from cloned captured
streams, followed by a separate 1024-pair heldout bank. The first batch exactly
reproduces the actual original D gradient; with beta1 zero this matches the saved
post-D Adam `exp_avg` bit for bit. Both unbounded and bounded original G proposals,
including updated Adam tensors, also replay exactly before the fit.

The loss is sharp Rp logistic plus the same `b_cap` penalty. Each state gets one
L-BFGS strong-Wolfe attempt, maximum 40 iterations and a hard maximum of 80
closure calls, history 10, initial step one, and no restart. The finite evaluated
point with lowest total training-bank loss is retained. Selection never reads
quality, centers or heldout measurements. The original `.15` G stencil is frozen
for all before/after field comparisons. The input-noise level is zero at these
captured late states; this diagnostic is not a cold input-noise implementation.

| Update | Training penalized D loss, before → after | Heldout penalized D loss, before → after | Heldout saved-metric residual ratio | Closures / iterations |
| --- | --- | --- | ---: | ---: |
| 1325 | .679025 → .640426 | .665913 → .629188 | 2.874 | 52 / 25 |
| 1530 | .671223 → .619487 | .654239 → .617165 | 4.831 | 54 / 40 |
| 1539 | .672532 → .573142 | .651912 → .559909 | 3.362 | 80 / 23 |

The fits take .60/.59/.86 seconds respectively in the pinned one-thread CPU
environment. At 1539 the hard closure budget interrupts the optimizer and the
best already evaluated finite point is retained. The other two terminate within
that budget; 1530 reaches the iteration limit. Final training raw gradient
infinity norms are `.00773`, `.00847`, `.01931`, well above the declared `1e-7`
stationarity tolerance. All are labeled **NONCONVERGED_RESIDUAL**. The training
saved-metric residual ratios are `.845`, `.619`, and `2.463`. Lower objective
values do not imply lower residuals or local optimality in this nonsmooth game.

The penalty is exactly zero on both initial banks at all three states. After
fitting, heldout penalty contributions are `.001884`, `.002083`, `.010629`.
Heldout fractions above the slope cap are respectively `.304/.172`,
`.490/.279`, `.324/.248` for real/fake samples. The diagnostic therefore retains
and measures the actual cap rather than substituting an unrestricted critic.

## Guidance and network coupling

Known centers are used only after fitting to explain motion. Positive radial
work means outward motion from the initial nearest center. At the exact captured
G minibatch, summed raw shared-network/prior radial work changes:

| Update | Learned critic | Locally fitted critic | Original bounded G HQ | Fitted-critic bounded G HQ |
| --- | ---: | ---: | ---: | ---: |
| 1325 | +.42707 | −3.33669 | .824219 | 1.000000 |
| 1530 | +.01220 | −4.51481 | .918457 | .999756 |
| 1539 | +.68310 | −8.26135 | .795166 | .828125 |

At 1530 the two endangered particles' distances become `.0728/.0629` after
the fitted-critic bounded proposal, compared with `.2169/.1897` under the learned
critic (base `.1902/.1739`). At 1539 the mode is already missing. Fitted guidance
moves both particles inward, but a single proposal does not recover eight modes.

The raw parameter direction is measured by an exact JVP through clean G/prior,
before Adam. A separate clean-input derivative measures the critic's local
guidance; network-only and prior-only finite proposals are also retained. Thus
the diagnostic separates the learned field from shared-network coupling and
Adam's metric. At 1539 one particle still has outward clean-input guidance after
fitting, while the shared-network proposal moves both endangered particles
inward; the complete per-particle values are preserved rather than reduced to
one aggregate claim.

The captured G noise is the next global draw after the first D draw, so it
overlaps a training-bank noise draw, although particle indices differ. A separate
check therefore uses **all eight already reserved heldout batches**, with no
new fit or seed. Each G counterfactual starts from the same saved Adam/model
state. Fitted D gives inward raw and accepted G motion in **24/24** cases;
learned D gives inward motion in only **2/24 raw** and **1/24 accepted** cases.
Fitted proposals pass 8/8 at 1325 (HQ 1), 8/8 at 1530 (minimum HQ .998779),
and 0/8 at the already collapsed 1539 state, despite all eight moving inward.

This supports testing actual D refinement before G as a separate, bounded
training-dynamics candidate. It does not establish that such a candidate will
maintain quality through a continuation or learn from scratch. The changing
empirical bank, nonconvergence, persistent Adam moments at refined parameters,
and additional field cost remain practical issues to test.

## Reproducibility

[`pr84_critic_relaxation.py`](pr84_critic_relaxation.py) is frozen at SHA256
`9f5f0d630e818d5259081c333f55e667f8a619ae81b25ebdd1bb62d334d7639b`.
The [evidence manifest](continuous-evidence/pr84-critic-relaxation/manifest.json)
contains every closure receipt, banks and fitted critic tensors, source snapshot,
before/after fields and independent heldout-G results. Input full capture-v2 SHA:
`37aa612bd3b3e1867a2d1508674158849ccf5940c907af76b1f8b061f4eb3e47`.
Saved input tensors and the outside global RNG are verified unchanged.

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 \
  ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES='' \
  /tmp/pr38-default-env/bin/python -u reports/toy100/pr84_critic_relaxation.py \
  --capture /path/to/full/capture-v2 --output /tmp/new-critic-fit
python reports/toy100/pr84_critic_relaxation_heldout.py \
  --capture /path/to/full/capture-v2 --fit /tmp/new-critic-fit \
  --output /tmp/new-critic-fit-heldout
```
