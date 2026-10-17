# Tier1 evidence review — 2026-10-03

The exact expanded roster is complete: **47 declarations (15 ideas + 32 configurations)**,
43 scientific FAIL and 4 preflight BLOCKED. The 111 completed attempts cost
1,135.233330 paid seconds. Tier1 has 68 PASS / 43 FAIL / 20 BLOCKED / 104 UNKNOWN
cells out of 235; the full 5/19/2 view has 68 PASS / 43 FAIL / 101 BLOCKED /
1,010 UNKNOWN out of 1,222. No later-tier ordinary run is eligible after these
prerequisite stops. Unknown downstream cells do not authorize duplicate training.

This review reads the existing single [technique inventory](../technique-inventory.md)
and [Tier1 readout](../tier1-refresh/README.md); its [JSON](TIER1_EVIDENCE_REVIEW.json)
retains all 47 named IDs, original inputs and numerical bindings through those
source-bound inputs. It is a compact evidence review, not a replacement ranking
or qualification reducer.

K3P `0b37e98a` and KA2 `093c6f2b` are tied at 4/5 as **best observed** whole
configurations. Neither is a qualified family. The selection uses required PASS
count and configuration hash, never per-task recipe mixing or elapsed-time speed.
The provisional screen, historical Atlas 19 noisy-serving positives and separate
Atlas/E22 policy study cannot provide calibrated defaults or fill this clean MoG
view. The executed commit is `2899099048c0a9987eb8720214abfce56d86d92a`,
scientific digest `7306340bac0a4ea67ea7b080513116457b7d72a8cb0d0c1f0727eaae6db38185`.
All 1,137 pinned executed files still match this `153c0dd4` checkout; runtime/source
identities remain separate from the earlier `8021` policy cohort.

## Saved-trace blocker

The original 43,004,589-byte archive at
`/home/martyn/dev/ParticleGAN/artifacts/forge/tier1-existing-configs-v1.tar.gz`
is unavailable on this machine and the checked repository/artifact alternatives.
Its required SHA256 is
`1ea74b7b92cd36848a6204bd39d9ce087470f8da4210f9dc0e4bd670c11057d9`.
Compact summaries retain matching original request/evidence/result pins, but do
not contain the 24-point word/ring trajectories. Recover and verify this exact
archive before assigning temporal or optimizer causes. WordAdapter writes a
final `state.pt`; ring16’s `produces_state=false` means no original vector
checkpoint was saved. A nonexistent ring state must not be inferred from the
archive description. Neither evidence gap authorizes an unchanged rerun.

## Observed word failures and source limits

K3P attempt `51169502a0e64466b5005e7c870b5592` completes 20,001 updates with 0/24
passing checks: endpoint modes 3/5, mass TV 0.4, reconstruction exactness 0,
minimum correct token probability 0 and token NLL 7.36828. KA2 attempt
`8e0a35381d6543bd8a3ecb6056351c06` also completes 20,001 updates with 18/24
passing checks; its endpoint passes (modes 5, TV 0.0189453, exactness 1,
minimum correct token probability 0.947841), but only 1/5 required terminal
checks is in the terminal passing suffix. Exact failed-time predicates are not available
without the archive. A2 was eligible/applied zero times; these outcomes do not
establish an effect of active damping. The selected K3P/KA2 ring runs both PASS;
other saved grid configurations provide the ring covariance failures.

The word source uses a **joint word/code critic**. It has no explicit paired
G(E(word)) token-consistency term. That supports a falsifiable future structural
hypothesis, not a proved cause: first inspect retained failed checks, then, if
justified, freeze one public-API paired-consistency variant with the unchanged
20,001-update budget, 24-check cadence, terminal-five requirement and all mass,
quality, padding and inverse-confidence bounds. No such run is launched here.

## Equality reporting defect

`baseline.score_metrics` and `observation.sustained` treat every operator other
than `>=` as `<=`. Their declared `==` bounds falsely label K3P modes `3 == 5` and
exactness `0 == 1` as individual PASS. The correct endpoint flags are sample/quality
PASS and modes/mass/exactness/confidence FAIL. KA2’s endpoint flags all PASS.
The original overall FAILs remain unchanged: TV ≤ 0.1 already excludes a missing
one of five modes (TV ≥ 0.2), and minimum correct token probability ≥ 0.9 already
forces correct token argmax and exact reconstruction. A future software repair
needs explicit equality/unknown-operator controls and a new evaluator identity;
this review changes no scorer, frozen result or qualification.
