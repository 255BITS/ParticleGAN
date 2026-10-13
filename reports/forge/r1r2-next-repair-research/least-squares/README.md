# Native LS: defer; the saved proxy is not a stronger shape prerequisite

**Defer a native least-squares experiment and reject the claimed stronger
from-initialization density evidence.** The existing loss is supported, but its
roughly doubled saved NS input-force proxy does not predict a repaired density.
The original full word reservation also cannot fit the remaining paid cap.
No candidate, task, closed research file or scientific source is changed here.

Native [GANLoss](../../../../particlegan/gan_loss.py#L30) uses half squared losses
with inherited labels `(fake, real, generator)=(0,1,1)`: D targets real 1/fake 0,
G targets fake 1, and the joint word E stream targets real 0. The
[primary LSGAN analysis](https://arxiv.org/html/1712.06391)
distinguishes this classification variant (Eq. 14) from the Pearson-divergence
variant, whose label conditions it does not satisfy. That paper does not supply
a density guarantee for this regularized, fixed-budget AMSGrad recipe.

Let `s` be an NS logit and compare score coordinates `q=.5+.25s`. Near equal
distributions, the **unregularized** best responses obey `q*=sigmoid(s*)`, so
this is their first-order local map. It is not the actual native LS initializer.
Using `softplus(s)=log2+s/2+s²/8−s⁴/192+O(s⁶)` gives:

| Adversarial term after the analytic map | Native LS polynomial in s | NS Taylor polynomial |
| --- | --- | --- |
| D real / G fake | `1/8−s/8+s²/32` | `log2−s/2+s²/8+O(s⁴)` |
| D fake / reversed E real | `1/8+s/8+s²/32` | `log2+s/2+s²/8+O(s⁴)` |

Thus all adversarial gradients/Hessians match **one quarter** of NS locally,
up to constants. The [native R1/R2 penalty](../../../../particlegan/grad_regularizers.py#L209)
is `gamma/2` times the summed mean squared input gradients; under the same map
it scales by **1/16**. Multiplying the mapped D objective by four yields the NS
quadratic objective with **effective gamma/4**, or `.025` from `.1`. This is
conditional loss-unit algebra, not a claim that the native gamma field changes,
that full LS training equals weaker NS, or that changing gamma is selected.
Comparing raw common-shift curvatures 2 versus .5 ignores this score coordinate;
the mapped LS curvature is .125. NS already anchors common score shifts locally.

The preserved [old probe](../../r1r2-energy-distance-repair/research/least-squares/diagnosis.json)
evaluates LS directly on saved NS scores near zero, while LS's balanced score
is .5. Its mean raw weight ratio is 1.999622. After the analytic map and chain
rule, the ratio of recorded mean weights is approximately .25, not two.
The original proxy expands radial variance in only 6/16 components under either
loss and contracts tangent variance in all 16. It omits G/E Jacobians, Adam
preconditioning, future critic refitting and historical paired training scores.
It therefore supports neither generator-starvation nor a from-init width repair.

Native LS still changes the global game: linear residuals retain tail curvature
and reverse G/E forces beyond targets. Native LS retains the original raw critic
initialization; a different offset/scale must be learned, and the analytic affine
map cannot replace that initialization.
Adam/AMSGrad's fixed epsilon and the unchanged auxiliary objectives further
prevent exact optimizer equivalence. The [weaker-penalty failure](../../r1r2-weak-penalty-repair/README.md)
is contrary evidence, not a tested `.025` LS arm or a transferable grade.
The [prior-freeze result](../../r1r2-energy-prior-freeze-repair/density-readout.md)
also supplies no LS trajectory evidence.

If reconsidered, declare exactly `loss: non_saturating → least_squares` on the
exact fresh NS control, energy 0, learned prior, gamma .1 and all other global
fields fixed. Preserve all six ordinary tasks plus own clock, seed 0/public
initializer, architecture, data/batches, initial prior law/capacity, sampling,
full budgets/cadence and original gates. One failure falsifies the whole repair.
Reuse only exact same-source controls; new science requires fresh controls.
Full allowances remain 2520 seconds per role / 5040 total. Current 238 charges
cost 7168.248061007995 seconds, leaving **831.7519389920053** under 8000:
the original **900-second word reservation cannot fit even before other work**.
No paid admission, cap increase, shortened task or automatic follow-up is implied.

[algebra.py](algebra.py) / [receipt.json](receipt.json) check exact rational
coefficients and all 1216 current/snapshot scientific file hashes against the
old archived NS request, plus original probe/artifact identities. This is
standard-library read-only algebra: no tensor runtime, models, autograd, draws,
optimizer steps, live evaluations or rescoring. Its PASS validates those
identities/equations, not a scientific repair prerequisite.

```sh
PYTHONDONTWRITEBYTECODE=1 /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-next-repair-research/least-squares/algebra.py
```

Raw stdout/stderr are archived in the closed prior-freeze artifact directory as
`least-squares-algebra.stdout.json` and `least-squares-algebra.stderr.log`.
