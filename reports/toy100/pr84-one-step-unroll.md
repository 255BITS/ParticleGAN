# One virtual critic step: total-gradient diagnostic

The full derivative through one virtual D step is locally correct at the
two saved states. It improves the single failing warm update's noisy HQ,
but it does not reverse that state's outward raw G field. This is a
read-only diagnostic, not a training pass or a solution to cold acquisition.

At original PR84 update 1325, noisy HQ after the cloned bounded G proposal
is 0.824219 with the original fixed critic, 0.942139 with a detached virtual
critic, and 0.970703 with the full chain. All three retain eight modes.
The total-gradient proposal is smaller because the unchanged own-secant
rule applies a smaller factor. Its raw direction remains outward on eight
of twelve particles. This distinction matters: the single-step grade gain
does not establish a corrected equilibrium field or long-term stability.

At cold update 472, starting from the saved best finite fitted critic, all
three proposals still cover only three modes. The full chain changes the
gradient substantially but does not acquire a missing mode in this one
step. The subsequently completed constant-rate fitted-critic cold run
failed acquisition; this diagnostic does not change that verdict.

## Frozen surrogate and controls

For joint G/prior parameters `g`, fixed starting critic `d0`, the native
cached D batch `xi_D`, and the actual next cached G batch `xi_G`, define

```
d_v(g) = d0 - P_D grad_d L_D(d0, g; xi_D)
Psi(g) = L_G(d_v(g), g; xi_G)

grad Psi = partial_g L_G
           - (partial_g grad_d L_D)^T P_D partial_d L_G
```

`P_D` is the exact saved post-Adam diagonal metric, held constant. No new
D moment or denominator is created or differentiated. The virtual
correction is rounded to the original parameter dtype. `L_D` is the sharp
native Rp logistic loss plus the native `b_cap` value; its fake-coordinate
graph is preserved by the separately tested functional helper. `L_G` uses
the frozen width-0.15 five-point stencil, including real logits. The two
virtual controls evaluate exactly the same virtual D parameters and loss;
one detaches those parameters, and the other differentiates through them.

The starting critic is held fixed when the surrogate is rebuilt at a
perturbed or proposed G point. At 472 it is the evaluated best finite
critic from the failed local fit; at 1325 it is the captured ordinary
accepted D*. Neither is claimed to be an exact best response. The scheme
is related to the finite unrolled objective in
[Metz et al., ICLR 2017](https://arxiv.org/html/1611.02163v4).
It is a change of G's surrogate, not an equilibrium-preserving
preconditioner: the extra term need not vanish at a general-sum stationary
point. This host's nonsaturating G objective, G-only stencil and D-only
cap prevent applying the zero-sum envelope argument automatically.

Each proposal advances only a cloned saved G/prior Adam optimizer once.
The usual PR84 own-secant factor `min(1, 0.25/rho)` then interpolates that
proposal. The second and accepted-point field evaluations rebuild the
same declared surrogate at their new G coordinates. No actual outer
training, critic fit, persistent D update, or moment mutation occurs.
Target centers are used only for the reported posthoc directions/grades.

| State and G field | Raw joint gradient norm | Own-secant rho | Accepted factor | Clean output RMS movement | Modes / noisy HQ |
|---|---:|---:|---:|---:|---:|
| 472: original fixed D | — | 3.22113 | 0.077612 | 0.121170 | 3 / 0.916992 |
| 472: detached virtual D | 2.93917 | 15.6092 | 0.016016 | 0.023092 | 3 / 0.863525 |
| 472: full virtual chain | 5.65653 | 6.05092 | 0.041316 | 0.117071 | 3 / 0.958496 |
| 1325: original fixed D | — | 0.238512 | 1.000000 | 0.057134 | 8 / 0.824219 |
| 1325: detached virtual D | 0.162004 | 0.404545 | 0.617978 | 0.036829 | 8 / 0.942139 |
| 1325: full virtual chain | 0.186046 | 0.585258 | 0.427162 | 0.029136 | 8 / 0.970703 |

At 472, full versus detached gradient cosine is 0.955626 and the chain
norm is 1.01270 times the detached gradient norm. Both raw directions point
toward a nearest missing mode for all twelve particles, and both point
inward toward current nearest centers for eleven. At 1325, cosine is
0.991336 and the chain/partial norm ratio is 0.204752. Both fields point
inward for only four particles; mean raw radial work increases from
0.033407 to 0.036738 with the full chain, meaning more outward work.

The original own-secant factor is not a nonlinear accepted-point bound.
At 472 the full-chain proposal's factor times the newly measured
accepted-point rho is 0.718157, above 0.25. This is recorded, not repaired
with an additional gain or acceptance rule.

## Derivative audit and retained failure

The initial audit used the actual float32 models. Its two predeclared
central-difference displacements were `h0=2e-4*(1+||g||)` and `h0/2`,
along the normalized negative full gradient. Both missed the declared
2% relative / 1e-5 absolute tolerance at both states. Those results remain
unchanged as `MISMATCH_OR_NONSMOOTH_CROSSING` in the initial receipt:

| State | Float32 relative error at h0 | At h0/2 |
|---|---:|---:|
| 472 | 9.62% | 18.80% |
| 1325 | 8.88% | 4.33% |

A separate audit casts the same saved weights, Fourier buffers, cached
draws and metric to float64. It evaluates **all** predeclared scales
`h0 * 2^-j` for `j = 0,2,4,6,8,10,12,14,16`, with no adaptive selection or
optimizer changes. Forward hooks count LeakyReLU sign switches and a
separate read-only calculation counts cap-active switches. The starting
critic, metric, stencil and restored G parameters are checked unchanged.

The large discrepancies persist at large double-precision displacements.
At 472's `h0`, the two endpoints change 24–38 starting-critic activation
entries, 129–136 virtual-critic entries and 7–8 cap entries. At 1325,
106–124 starting-critic and 494–541 virtual-critic entries change. Error
is nonmonotone near these crossings; for example, double 1325 reaches
45.8% error at exponent 6. These intermediate failures are retained.

At exponents 12, 14 and 16, **both endpoints have zero activation or cap
switches** at both saved states. All three scales agree with autograd:
maximum relative error is 1.08e-10 at 472 and 1.20e-9 at 1325. The native
and double gradients have cosine above 0.999999999999 and relative norm
differences below 1.8e-6. Actual perturbation norms and directional rounding
errors are recorded. This supports the local derivative implementation;
it does not make finite optimizer steps smooth or certify convergence.

The observer reproduces the initial native gradient bitwise. The initial
diagnostic also verifies native first-D gradient parity, bitwise equality
of the functional/native inner value and D gradient, identical virtual
control endpoints, unchanged saved inputs, and unchanged external RNG.
Independent read-only review found no missing graph path. Four focused
tests pass: two analytic total-chain/rest tests and two graph-preserving
cap tests. There is no full training or acquisition gate in this result.

The [diagnostic source](pr84_one_step_unroll.py),
[separate derivative audit](pr84_one_step_unroll_audit.py), and
[archive manifest](continuous-evidence/one-step-unroll/manifest.json)
retain exact declarations, source bytes, tensor inputs and both complete
results. Evidence is explicitly `shared_gate_eligible=False`.

For an independent derivative-only reproduction, decompress the archived
`initial/tensors.pt.gz` to a temporary path and run
`pr84_one_step_unroll_audit.py --input <tensors.pt> --config
reports/toy100/continuous-evidence/critic-refinement-finite-recovery/config.json
--output <new-directory>` in the pinned one-thread PyTorch 2.13 CPU/AVX2
environment. The initial diagnostic additionally uses the already archived
failed-fit payload and compact original warm snapshots, whose hashes are
in its declaration.
