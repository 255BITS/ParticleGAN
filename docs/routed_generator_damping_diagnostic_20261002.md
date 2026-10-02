# Routed FiLM initialization: conditioning sensitivity and a rejected rate bypass

**This is an initialization diagnostic, with no core-policy change or confirmed
fix.** Zeroing the whole additive FiLM branch removes a direct code-conditioning
path. The real first-step encoder gradient and the small spatial fixture show
attenuation, while a wider spatial fixture reverses the terminal quality ordering.
The matched Nova → Qwen G-rate bypass is **negative**: decoded LPIPS worsens 11.5%,
fit MSE 9.2%, and guard MSE 8.7%. A toy-only damping benefit does not qualify a
real-task fix or establish a public default, exact failure reproduction, or win
over the previous model.

The adjacent [JSON receipt](routed_generator_damping_diagnostic_20261002.json)
contains compact source/data hashes, measured results, cut records, and independent
audit findings. Bulk traces/checkpoints stay outside Git. The real-task archive
[permalink](https://github.com/255BITS/model-glue/blob/c465391f71a3bc32fb01cee904385c2c237c3852/experiments/e22_gan_damping_counterfactual_20261002/README.md) binds the frozen driver, protocol and independent verification.

## Matched task convergence

The real expanded Nova → Qwen comparison changes only fresh additive FiLM weight
rows after public deterministic orthogonal initialization. All unaffected initial
owners and all 4,000 caller batch/Gaussian draw prefixes match. The dataset has
10,806 training rows and 176 validation cases; the test split stays unopened.

| Full176 LPIPS / update | Original | Whole shift zero |
| --- | ---: | ---: |
| 100 | .484361 | .476657 |
| 300 | .389702 | .375039 |
| 2,500 | .209397 | .290473 |
| 4,000 | .193780 | .277582 |

The early advantage reverses; zeroing finishes **43.2% worse** at the fixed cap.
This complete cycle retains historical pin `bdf05d1b`. The subsequent matched
2,500→2,600 counterfactual executes `develop` core `3f02957a` and exactly reproduces
the historical native updates, all 176 endpoint cases and serving choice. The
publication base is `70de5a3e`; all 29 core Python files are byte-identical to the
executed `3f02957a` core. No complete real cycle on the publication HEAD is claimed.

[Completed task report](https://github.com/255BITS/model-glue/blob/c465391f71a3bc32fb01cee904385c2c237c3852/docs/results/nova-qwen21-e22-gan-neutral-cycle-20261002.json)
and [independent 4,000-step audit](https://github.com/255BITS/model-glue/blob/c465391f71a3bc32fb01cee904385c2c237c3852/experiments/e22_gan_neutral_cycle_20261002/verification/final-independent-review.json)
retain the full metric curves and source hashes. This isolates sensitivity to
erasing conditional additive weights; it does not establish that this pathway
explains the entire late convergence deficit.

## Conditioning pathway

For normalized code `c` and time `u`, the native form is
`h * (1 + gain(c,u)) + shift(c,u)`. At `h=0`, the gain contribution to the code
Jacobian vanishes. The surviving additive derivative is
`0.25 * diag(sech²(pre_shift)) * W_shift,code * J_LayerNorm`.
Zeroing every shift row erases this derivative, even with healthy tanh slopes and
live gain weights. Preserving additive code columns while neutralizing only their
time/bias part retains the path. This differs from neutralizing a separate
host-only `Hh+b` branch while retaining `Cz` as in PR227; no correct port of that
proposal is claimed.

A deterministic geometry contract executes the actual spatial prefix/project/input
path, controls its pre-FiLM hidden value to zero, and checks the analytic Jacobian
and the complete public bank-route/GAN gradient. It performs no optimizer updates
or settlement decisions. This controlled boundary is software geometry evidence,
not a trained experiment arm.

Measured width-4 spatial initialization shows:

| Measurement | Original | Whole shift zero |
| --- | ---: | ---: |
| Code-Jacobian Frobenius norm | 4.91086e-3 | 4.03199e-4 |
| First-step E gradient squared norm | 7.70004e-11 | 1.61101e-12 |
| First-step actual Adam E displacement squared norm | 1.08320e-5 | 4.39631e-6 |

Thus 47.8× smaller gradient energy produces 2.46× smaller displacement energy,
not the same factor. The real initialization receipt independently measures
107.039× smaller first-step E gradient energy with all unaffected owners/streams
exact. The older 31-update partial receipt is retained with its frozen input
hashes and no endpoint inference; the verified 100-step initialization receipt
confirms the same first-step observation. Neither proves every later quality gap
or native rate decision.

## Verified Nova → Qwen rate counterfactual

Both arms restore the same complete neutral checkpoint at step 2,500,
`e2effa6e0ea8601036b26e53259938d0353fb549369e35c046cd881013ba3cbe`,
and run 100 unchanged native paired-GAN updates on public package commit
`3f02957aba4517121c5283d77c8fff98895658b5`. Only applied G settlement damping
is bypassed after the public begin hook. Models, moments, E/table/D/noise,
routing and observers start matched and retain their native controls.

| Step-2,600 clean measurement | Native | Applied G bypass |
| --- | ---: | ---: |
| Fit64 normalized MSE | .218342 | .238402 |
| Guard64 normalized MSE | .239751 | .260691 |
| Decoded validation176 LPIPS | .287736 | .320730 |

Complete checkpoint readback, clean64 startup equality, decoded startup per-case
identity, finite updates and 100 caller data/paired-Gaussian identities pass.
Native matches the historical step-2,600 result. Protected inputs remain unchanged;
test is unopened. No identity claim is made for private policy stream consumption
after trajectories diverge. The inherited G scale is `1/32`: this factor-32
restoration differs from the toy's factor-two cut during acquisition. Parent
FiLM derivative bounds exceed `.865`/`.910`, excluding severe tanh saturation there.
**Reject this bypass as a fix for this checkpoint/100-update continuation.**

Independent review passes 24 checks and recomputes all 20 decoded metric means
exactly. The decoded receipt's stale `completed:false` marker is preserved;
the result/supervisor and four decoded records establish actual completion.

## Frozen toy protocol and retained results

CPU comparisons use 1,200 updates/arm, batch 16, Adam `(0,.999)`, G/E LR `.000204`,
critic/table multipliers `1.5/10`, and initial output sigma `1.3`. Named seeds are
fixed; no seed sweep was run. External caps preserve the clock-free recipe.
The coupled `128×4` particle cloud, prior sigma zero, is an explicit diagnostic
exception. Nonlinear residual FiLM and `.1` gains use public deterministic init.
Training is pure paired-error RpGAN plus KA2; MSE is only clean third-pool evaluation,
never training or row acceptance. Corrected hosts retain all routed controls,
including settled reopening. Bypasses retain the tester and actual-rate observers.
Antithetic G averages the original losses over the same prediction and `±` Gaussian;
D training and caller draw counts stay unchanged.

| Clean final MSE / host width | Original | Shift zero | G bypass | Antithetic | Antithetic + G bypass |
| --- | ---: | ---: | ---: | ---: | ---: |
| Early vector / 16, outer identity | 2.51988e-4 | 2.50915e-4 | 2.13580e-4 | — | — |
| Vector handoff / 16, large target | 4.00182e-3 | 4.01161e-3 | 4.01161e-3 | 2.49821e-3 | — |
| Vector handoff / 16, calibrated target | 6.00182e-4 | 6.13823e-4 | 6.13823e-4 | 5.81375e-4 | — |
| Vector handoff / 64, calibrated target | 1.39819e-5 | 1.63447e-5 | 1.63447e-5 | 2.24676e-5 | 1.53548e-5 |
| Spatial handoff / 4 | 1.43826e-4 | 1.66828e-4 | 1.66828e-4 | 5.11742e-5 | — |
| Spatial handoff / 16 | 9.68478e-5 | 7.45939e-5 | 7.45939e-5 | 2.77279e-5 | — |

The spatial host preserves `2×8×8` source, frozen BF16 `2→3` prefix/`3→2` head,
FP32 projected residual conv FiLM, LayerNorm code query and cosine routing, with
no outer identity. Its learned local/global critic adds a neutral free-sign
channel-energy head, `mean(error² over HW)*sqrt(HW/C)`. Named Gaussian source std
is `.2`; target is `.15` channel-swapped source plus time edit; fit/guard/test
counts are `245/64/256`.

**Every corrected ordinary shift-zero arm has zero G cuts; its G bypass is
identical.** Spatial width 16 also reverses the neutral terminal ordering.
Antithetic spatial gains occur without cuts, so they do not isolate damping.
The early outer-identity vector lacked the settled guard and is only partial
architecture evidence; its interrupted successor's incomplete arms are excluded.

The narrow vector-64 antithetic result cuts G at step 504 and damps 696 updates.
Independent audit finds identical initial owner/data hashes and all JSON rows
through step 504. First divergence at 505 is applied G LR `.000102` versus
`.000204`; retained bypass observers subsequently record two cuts. Its final MSE
is 31.7% lower than matched antithetic native. The negative real replay prevents
promoting this isolated toy benefit into a Nova → Qwen fix.

## Source and reproduction

Completed evidence uses captured source hashes in the JSON. Current common source
subsequently adds finite checks and measures critic gradients before freezing D;
completed runs were not rerun or relabeled. Spatial source stays its executed
snapshot. Independent audit verifies unchanged package bytes across 29 recorded
Python files. Later launches use external `timeout 900s`; early backfilled receipts
state their weaker provenance. The final public width-4 source was replayed for 1,200 updates per arm against
publication `develop`: all scientific trace rows, 13 clean evaluations, models,
optimizers, controllers and named streams exactly match the recorded results.
Only the corrected critic-gradient reporting field differs. The fresh process
ambient global CPU RNG snapshot differs; complete cross-process global checkpoint
identity is not claimed. Exact replay within a checkpoint is tested separately.
The reproduction commands are:

```sh
timeout 900s python -u -m benchmarks.routed_conditioning.spatial_damping \
  --steps 1200 --width 4 --output /tmp/spatial-routed-damping
timeout 900s python -u -m benchmarks.routed_conditioning.film_damping \
  --steps 1200 --width 64 \
  --profile shift_zero_antithetic --profile shift_zero_antithetic_g_bypass \
  --output /tmp/vector-routed-damping
```

Use new, empty artifact directories. No core policy/default changed; no further
toy search or production promotion follows from these diagnostics.

Publication validation: 96 CPU checks passed; three CUDA-only unit cases were
skipped. The real-task counterfactual executed on GPU independently. The fixed
two-arm publication replay took 63.39 seconds.
