# RA17 current-PR155 source closure

RA17 copies all28 current repository modules after the clean merge of PR155
cabe208 into the byte-exact RA16 integration commit. The one root-authorized
EOF newline removal is reflected byte-exactly. Config bytes remain unchanged.
RA16 packages, freezes, completed suites, replay, and qualification stay intact.

## Changes relative to RA16

- `continuous.py`: DV12 diagnostic reductions are deferred and materialized
  as ordinary float dictionaries. The public `latent_applications` checkpoint
  key and load law remain compatible; only the internal cache representation
  changes. The actual perturbation and private draw law are unchanged.
- `routing.py`: validation predicates are detached and aggregated at complete
  CUDA forward finish. Valid logits, weights, mixed codes, token/site means,
  and gradients retain the original arithmetic. CPU invalid-value checks stay
  immediate. A closed execution now also rejects finish.
- `policy.py`: learned output-noise floor settlement excludes the noise
  tester. This intentionally changes behavior once model/table testers settle,
  and is not a universal numerical-parity claim.
- `output_moments.py`: exactly one trailing newline removed; no AST change.

## Completed CPU evidence

104 contracts pass with CUDA visible and uninitialized: the94 prior focused
contracts plus relevant new upstream noise-floor and71-site CPU regressions.
The independent RA16/RA17 trace matches every loss and complete public
checkpoint state across18 feature updates/two mean reactions and2 KNN updates,
with emitted samples, valid restore in both directions, and exact next update.
Only the diagnostic elapsed clock is shared in this synthetic fixed fixture.

Direct DV12 witnesses match outputs, gradients, private stream state, last-two
float diagnostics and serialized public state for float32/float64, with and
without CPU bf16 autocast. Direct71-site witnesses cover dependent sites,
different correlated token counts, inactive mass rows, codes/usage and table,
log-mass and logit gradients; functional callback weights match too.

## Scope of the historical noise-floor bridge

The upstream floor correction can change sigma and its derivative at the
1/64 boundary, as the actual old/new method witness demonstrates. For each of
the19 original quality sources and the Toy/MNIST2000-update learned prefixes,
the retained table tester has at most2 cumulative stationary decisions.
Stationary alone halves s; its lifetime counter persists across restart, while
reopen/drift/population revocation only raise s. Hence every earlier table
scale is at least2**(-C_stationary(T))>=1/4>1/64. Both floor laws use settle=1
throughout each original finite horizon. This is a cumulative action/source
proof, not endpoint-scale interpolation. Original40 replay windows1001..1010
lie inside those proved learned prefixes. No new post-budget horizon or manual
tester mutation is covered.

The independent reviewer approved these source/math and finite-horizon claims.
Historical evidence retains its original source labels. Current-source GPU
replay and full suite are separate fresh gates and are still pending in this
closure; no agent GPU launch occurred here.
