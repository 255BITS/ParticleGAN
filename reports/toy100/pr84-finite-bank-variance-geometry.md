# Finite-bank variance and geometry at update 1325

Variance reduction is a valid new mechanism to distinguish from the earlier
same-sample extragradient experiments. This saved-state diagnostic finds
substantial critic noise and a coherent generator field. Removing all
declared finite-bank sampling noise does **not** make the local game uniformly
contractive, but changing the critic with the mean field avoids this one
quality failure. No training candidate or continuation was run.

## Exact controls and finite-oracle scope

The input is the original PR84 update-1325 capture. The native float32
discriminator gradient matches the saved Adam first moment bit-for-bit, and
the original generator/prior parameters and Adam state after its bounded
proposal match the captured state bit-for-bit. The first generated/real G
batch is exact. All later calculations use detached copies and preserve the
input state and caller's global RNG.

There are sixteen native-sized D banks and sixteen G banks, each with 128
paired real values, particle indices and output-noise values. D's first bank
and G's first bank are the actual native draws. The rest continue cloned
native streams, without changing a seed. Input noise is already zero;
output noise remains `.029`. Both players retain the actual logistic Rp
objectives, D-only `b_cap`, and G-only `.15` five-point stencil.

Once these arrays are fixed, the arithmetic mean is the **exact declared
finite operator**. It is not the exact expectation under the host's fresh
Gaussian real law or its output-noise distribution. Freezing only real
examples would not remove the latter variance. The reference point has
metric field norm `.062701`, so it is not an identified equilibrium.

## What removing finite-bank variance changes

For a uniform finite operator `F(x)=m^-1 sum_i F_i(x)` and snapshot `w`, define

`v_i(x;w)=F_i(x)-F_i(w)+F(w)`.

Its exact mean is `F(x)`. At `x=w` every component equals `F(w)`, even when
the individual component fields disagree. At an actual root `w=x*`, this
gives exact rest without reducing the step size; a model perturbation still
produces a field. If the components are Lipschitz, the estimator variance is
bounded by the average squared component Lipschitz constant times
`||x-w||²`. Replacing `F(w)` with a noisy estimate preserves its estimation
error even at the snapshot; an online Gaussian expectation is therefore a
different problem from this finite-sum identity.

The implemented identity is exact at the saved reference. At the captured
post-update point, the controlled mean differs from the full finite mean by
at most `2.78e-17`, and metric variance falls from `.00741971` to `.00062870`
(8.47% of its original value). The nonzero mean remains unchanged.

| Role at the pre-D point | Squared mean metric norm | Component metric variance |
| --- | ---: | ---: |
| D | .00231648 | .00728322 |
| G and prior | .00161494 | .00013719 |

D supplies 98.15% of this total component variance. The earlier finding of
a coherent outward **G field at a fixed trained D** cannot rule out noise in
the preceding critic updates.

At that original accepted critic, a single real Adam G update using the
entire sixteen-bank G objective still fails: HQ `.996582 → .877686`, versus
the native `.824219`. Its original curvature factor is `.888317`, accepted
clean-output RMS `.047297`, and raw joint radial work remains positive
(`.031333`, outward under the offline nearest-center diagnostic). No center
or HQ value chooses an update.

## Mean-game geometry and the critical scope distinction

The separate local-map calculation uses float64 copies of the captured
float32 weights and draws. It fixes `P_D` and `P_G` to the original saved
post-proposal Adam metrics, including actual role rates `.00425`, `.00425`,
and `.0085` for D, G and prior. It evaluates mean D first, applies the
original own-field bound `3`, then evaluates mean G at that accepted D and
applies its bound `.25`. It does not advance or differentiate moments.

This is a faithful partial-field and ordering diagnostic with an explicitly
frozen metric. It is **not** the full Adam augmented-state transition,
the actual new-moment mean-batch method, SVRE, or a promoted training rule.

The base mean D→G map itself keeps all eight modes and changes HQ
`.996582 → .997314`, with clean-output RMS `.014294` and G factor `.544278`.
This differs from averaging only G at the original critic. It is therefore
incorrect to claim that the saved event proves joint variance reduction
cannot help.

For local sensitivity, perturb the joint parameters along three declared
directions: the captured D step alone, captured G/prior step alone, and
their concatenation. Norms use `P^-1`. The refined central secants give:

| Direction | Mean-operator symmetric Rayleigh | Mean alternating-map amplification |
| --- | ---: | ---: |
| D only | +.331280 | .693243 |
| G/prior only | **−.137772** | **1.086325** |
| Captured joint step | +.253716 | .761277 |

The G-only row disproves local monotonicity of this finite operator and
uniform contraction of this fixed-metric map at the point. A one-step
directional expansion is not a spectral-radius calculation, a proof of
asymptotic divergence, or a claim about the population game.

The first fractions `1e-4` and `5e-5` of the captured parameter displacement
crossed local pieces and gave unresolved map secants. Those results remain
archived. A separate, predeclared numerical audit used `1e-7` and `5e-8`;
all reported Rayleigh/amplification pairs agree within `3.2e-8` relative.
The achieved double-precision input displacement has at most `2.5e-7`
relative error. An independent own-G autograd quadratic form gives
`−.137772406` at pre-D and `−.270007275` at the native accepted D, with no
exact-zero LeakyReLU inputs. Thus the negative own term does not depend on
interpreting the coarse kink-crossing secants as derivatives.

## Research implication and next cheapest test

[Chavdarova et al., NeurIPS 2019](https://papers.neurips.cc/paper/8331-reducing-noise-in-gan-training-with-variance-reduced-extragradient.pdf)
explicitly combine snapshot variance reduction with extragradient and show
why stochastic game noise can defeat an otherwise convergent batch method.
Their theory requires operator assumptions absent here, and their GAN
discussion treats the infinite noise distribution separately. Reusing the
same two noisy draws, as in our earlier EG experiments, is not the snapshot
control variate tested above.

[Alacaoglu and Malitsky](https://arxiv.org/abs/2102.08352) establish
variance-reduced VI methods in monotone settings. The newer
[Dvurechensky et al. 2026 hierarchical result](https://arxiv.org/abs/2602.13510)
also assumes monotone Lipschitz finite-sum operators and analyzes its
hierarchical problem under additional structure. Neither certifies this
nonsmooth general-sum neural game. The negative measured direction is a
concrete reason not to transfer those guarantees.

The authorized follow-up **actual new-moment alternating Adam update** passed
this saved point: HQ `.996582 → .997314`, all eight modes, clean-output RMS
`.0142941`. D's factor is one; G's is `.544292`. Both G-field stencil widths
are `.15`, using the original rule recomputed at the base and proposal. D/G
Adam each advance once to moment step 1325 at their original rates. The
finite mean accumulates per-native-bank gradients in float64 and rounds once
to float32 before each actual Adam step.

Its one-bank control reproduces **all D/G/prior parameters and both optimizer
state dictionaries bit-for-bit** at the captured post-bounded-G boundary,
and reproduces the original `.824219` HQ. This is before EMA; no equivalence
of an unexecuted host's future RNG or noise counters is implied. It uses
cached tensors and leaves caller RNG/input snapshots untouched. The
standalone computation evaluates 32 useful gradients per player, omitting
the native scaffold's unused phase-0 G and phase-2 D fields. A host adapter
must account for those separately if it executes them.

The [control source](pr84_finite_bank_adam_control.py) exposes
`mean_adam_update(pre_step, d_rows, g_rows, recipe)`, returning the receipt and
copied G/D/prior/Adam state. Its
[separate archive](continuous-evidence/finite-bank-adam1325/manifest.json)
binds this additional one-point test. It supports only the declared late
zero-input-noise host. The next filter is the original three sequential
saved-state branches, not a long run justified by this point. No cold or
continuous-hold success is claimed.

The twelve-atom/eight-mode ring is also distributionally misspecified;
perfect HQ is not zero game residual. Those representation facts and the
positive residual cannot be relabeled stochastic noise. Production's
20,000-atom/100-mode allocation removes the simple integer mass mismatch,
but still needs its original mass and conditional-shape gates.

## Evidence and reproduction

The [manifest](continuous-evidence/finite-bank-vr1325/manifest.json) binds the
raw results, failed initial input-schema attempt, coarse and fine numerical
audits, exact source bytes and complete update-1325 input subset. The three
analytic tests verify zero-variance rest with fixed-gain response, unbiasedness
without curing an anti-monotone mean, and the residual from a biased
snapshot mean. They passed; the native replay checks are separate executed
controls in the result.

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 \
  ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES='' \
  /tmp/pr38-default-env/bin/python -u reports/toy100/pr84_finite_bank_vr_diagnostic.py \
  --states reports/toy100/continuous-evidence/finite-bank-vr1325/complete1325.pt.gz \
  --output NEW_FINITE_BANK_RESULT

# Use the same environment variables for the separate numerical audit.
python -u reports/toy100/pr84_finite_bank_vr_audit.py \
  --states reports/toy100/continuous-evidence/finite-bank-vr1325/complete1325.pt.gz \
  --initial NEW_FINITE_BANK_RESULT --output NEW_FINITE_BANK_AUDIT
```
