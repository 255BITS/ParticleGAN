# Why a four-weight initialization change can matter

For the full QR/bias/particle construction and architecture-specific mapping,
see [the initialization guide](../../../docs/initialization.md).

The batch-distance critic has the form

    D_i(X) = a^T h(x_i) + b^T phi_i(X) + c,

where h is the learned per-point network and phi contains normalized local
squared-distance features. The candidate keeps the deterministic QR parameters
and initializes only b to zero. It does not change the architecture, forward
pass, optimizer, or later gradients. The rule uses the architecture's feature
partition, not task names or target data.

## Exact initialization guarantee

At b=0, D_i depends only on x_i. Therefore, for every input batch and i != j,

    partial D_i / partial x_j = 0.

All input derivatives contributed by the batch-distance branch are zero. The
per-point input gradients and the gradients into its hidden layers remain
those of the original QR critic. In contrast, zeroing the entire output head
also eliminates those per-point gradients at initialization.

This does not freeze b. Its gradient is a weighted real/fake feature difference,
which need not vanish at b=0. The unchanged optimizer can learn batch-dependent
scores on its first update. Thus the zero-cross-sample-derivative property is
an initialization guarantee, not an invariant throughout training.

## An initial contracting force in the failing QR run

For scale s_k, this implementation uses

    phi_ik = sum_{j != i} K_ijk ||x_i-x_j||^2
             / (s_k^2 (sum_{j != i} K_ijk + eps)),
    K_ijk = exp(-||x_i-x_j||^2 / (2 s_k^2)).

Consider a coincident cloud perturbed by small deviations. Write

    q = sum_k b_k / s_k^2.

The quadratic term of the summed batch score is

    sum_i b^T phi_i(X)
      = 2 n q / (n-1+eps) sum_i ||x_i-mean(X)||^2 + O(||deviations||^4).

Consequently its exact Hessian at a coincident cloud is

    4 n q / (n-1+eps) (I - 11^T/n) tensor I_data.

If q<0, ascending that initial summed score contracts every nontranslation
direction to first order. No target distribution enters this preference.
For the failing original QR critic, q is about -4.35. Equal parameter RMS
does not protect against this: the four channels have very different input
derivatives because their kernel scales differ.

On all 256 actual initialized generator outputs, the batch-score input-gradient
RMS is 0.3642, versus 0.01413 for the per-point score, a ratio of 25.79. The
covariance derivative from ascending the summed batch score is negative
definite (eigenvalues approximately -0.03248 and -0.01378). The per-point term
is much smaller. This finite-cloud observation agrees with the local formula.

`analyze_batch_force.py` reproduces these numbers offline from the recorded
initial tensors. It checks the Hessian formula by autodifferentiation and
checks that setting b=0 removes cross-sample input derivatives while preserving
nonzero learning gradients into b on a fixed synthetic real/fake example.

## What this establishes about convergence

The algebra proves elimination of an arbitrary initial batch-coupling force.
The experiment isolates four changed parameters and measures the resulting
training trajectory under identical audited randomness. Together they provide
a mechanism and an intervention test.

They do not yet prove convergence of the nonlinear, sampled Adam game. The
diagnostic uses gradients of summed initial scores in input space; the actual
trainer updates D first, uses relativistic logistic weighting, and maps G's
parameter and particle updates through its network. A proof must control that
actual update map and its later stochastic dynamics. The full-suite gates
certify the measured finite training runs, not an asymptotic theorem.
