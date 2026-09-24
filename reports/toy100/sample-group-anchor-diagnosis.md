# Sampled group anchors expose missing support without a mode oracle

An exact output-space calculation on the saved failed cold update-100 state
finds a nonlocal coverage signal that the local density-ratio field lacked.
The generator's 12 clean particles occupy only **three of eight** groups
inferred from the next native 128-real-sample minibatch. A distinct-anchor
assignment sends five different particles toward the five missing sampled
groups. One exact quadratic output-space step decreases the declared loss
from **9.68407 to 2.12064** and raises nearest-group occupancy from three to
six. It asks for a maximum 3.389-unit output move, so this is **not** a
training result or evidence that the shared G/prior can land safely.

The [source](sample_group_anchor.py),
[fixed-bank receipt](continuous-evidence/sample-group-anchor/receipt.json),
and [tests](../../tests/test_sample_group_anchor.py) reproduce the calculation.
The state comes from the [exact cold-prefix archive](continuous-evidence/pr84-finite-cold-prefix100/manifest.json),
after update 100 of the safe critic-refinement run. Its real-data RNG is
restored to draw one native-size bank; target ring centers enter only the
unchanged host sampler, never the grouping, assignment, or field. This is one
bank and its two disjoint halves, with no seed or setting sweep.

For real points X, cut their minimum spanning tree at the largest additive
gap between sorted tree-edge lengths and take the resulting group means
`c_1,...,c_K`. For clean generated particles `y_1,...,y_N`, choose an
**injective** minimum-cost matching `a(k)` of one particle to each group.
The explicit additional support objective is

```
L(Y;C) = (1/K) sum_k ||y_{a(k)} - c_k||²
       + (1/N) sum_j min_k ||y_j - c_k||².
```

The first term gives missing sampled groups a nonlocal donor. The second
returns extra particles toward an observed group without assigning all 12
particles equal mass across eight groups. Both coefficients are one in their
native mean-square units. This is a proposed **new data objective**, not an
optimizer-only repair of the frozen relativistic GAN. Minimum-spanning-tree
clustering is a classical data-derived geometry method
([Zahn, 1971](https://www.slac.stanford.edu/pubs/slacpubs/0500/slac-pub-0672.pdf));
the distinct assignment is the classical assignment problem
([Kuhn, 1955](https://onlinelibrary.wiley.com/doi/10.1002/nav.3800020109)).
GAN papers document why a generator can fail to receive a useful missing-mode
signal ([Che et al., 2017](https://arxiv.org/abs/1612.02136),
[Sharma and Namboodiri, 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11790)).
Those papers neither propose this exact objective nor prove it solves this
host; the objective and following finite-geometry argument are our own.

| Fixed-bank check | Observation |
| --- | ---: |
| Full real bank | K=8; 8–25 samples per group |
| Largest MST gap / second-largest | 1.82840 / 0.04509 |
| First 64 / last 64 real points | K=8 / K=8 |
| Half-bank mean centroid shift versus full | 0.02114 / 0.02224 |
| Saved 12-particle support | 3 groups occupied; 5 missing |
| Exact output step | L 9.68407→2.12064; 3→6 groups; max move 3.38874 |
| Controlled good empirical 8+4 cloud | L=0 and gradient=0 |
| One extra good particle perturbed 0.07 | restoring projection −0.011667 |

The good cloud places one particle at each **sample-derived** group mean and
four extras at already occupied means. Thus the rest calculation does not
require matching 12 particle masses to eight target masses. The `.07`
perturbation is a diagnostic distance scale, not a training gain. The
half-bank shift measures this one draw's grouping variability; it does not
prove stable group recovery on future banks. Finite minibatches will move
centroids, so exact empirical rest is not time-uniform stochastic rest.

There is a useful limited geometric guarantee for *fixed distinct centers*
and freely movable outputs when `N>K`. Write `L` as the minimum over all
injective anchor assignments and nearest-center choices of their smooth
quadratic pieces. At a local minimum of `L`, every active quadratic piece
must also attain a local minimum. If anchored particle `j` is assigned to
center `a` but nearest center `b`, minimizing that piece gives
`y_j = (N c_a + K c_b)/(N+K)`. For `a != b` and `N>K`, this point is closer
to `a` than to `b`, contradicting the active nearest choice. Consequently
every anchored particle sits at its assigned center; unanchored particles
sit at a nearest center; all local minima have `L=0`. Equivalently, exact
output-space minimization of an active quadratic piece decreases positive
`L`; it can still require a large step. This statement says nothing about
the restricted image and Jacobian of the shared network, minibatch noise,
optimizer moments, or the coupled GAN game.

The preceding [bidirectional Chamfer host run](chamfer-projection-report.md)
shows why that distinction matters. Its prior-only pullback had median cold
latent movement 189.92 and nonlinear target error 0.882, and accepted a
decrease relative to the **post-GAN** point while the whole update could harm
quality. A viable next gate must first test whether a **joint G+prior** map
can reach this fixed output target with bounded nonlinear error and a
whole-map acceptance check on saved cold and warm states. Only a passing
fixed-target test would justify a fresh warm/cold training run. This report
does not promote an adapter or claim indefinite stability.

Run the 1.3-second tests and reproduce the fixed-bank receipt with:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
/tmp/pr38-default-env/bin/python -m pytest -q tests/test_sample_group_anchor.py
/tmp/pr38-default-env/bin/python -m reports.toy100.sample_group_anchor \
  --archive reports/toy100/continuous-evidence/pr84-finite-cold-prefix100 \
  --output reports/toy100/continuous-evidence/sample-group-anchor/receipt.json
```
