# Review before fourth constant-rate attempt

Three completed proposals (C7–C9) produce no qualified default. Continue the
existing authorized three-lane search with at most three new distinct proposals,
API-C10 onward, one external Astra/max session and one GPU worker. Start only
after the predecessor driver exits. No seed experiments or coefficient grids.

Predecessor:
`/ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T222523Z-1514963/constant_rate_stability/20260926T222523Z-1514972`

Read final result.md/tests.jsonl, immutable source/declarations, root first-results,
C9 image archive and constant-wave3 source audit. Preserve all useful measurements.

- C7 uses an explicit fresh one-step critic reference with the implicit secant
  update. Initial590,182/182; changed+270,194/194; image6/24 with final5 and exact
  fresh-process continuation pass. It then loses an unchanged target:641/692,
  failures2740–2960,6440–6690,6710–6720; minimumHQ.0009766 and1mode. Fresh memory
  alone does not cure late collective instability.
- C8 adds a same-oracle local secant probe with binary backtracking. Complete4600
  observes neither target arrival: final prechangeHQ.117432, shifted.258301.
  Cold bestHQ.423828. Own exact checkpoint replay passes. Runtime901.4seconds;
  numerical locality and constant nominal rates are conformance, not quality.
- C9 replaces the fitted-plane response with direct local extragradient
  acceptance. Neither arrival is observed by4600; cold best.187744, shifted
  best/final.597412 while improving. Runtime943seconds, mean8.56 field evaluations
  per accepted update. Own image is a separate measured FAIL:0/24, finalHQ.375,
  only1qualitymode. Own1600→1800 replay passes.

Finite nonarrival is NOT_OBSERVED, not proof that acquisition can never occur.
C9's gradual improvement remains a lead; its independent image failure and high
cost block default promotion. C7's departures occur long after settling and are
concrete retention failures. Keep constant nominal rates distinct from the
variable accepted displacement of the numerical solver.

Review how joint-field conditioning and role-wise proposal scales affect the
correction. A new mechanism should address the measured instability/acquisition
tradeoff, not repeat these variants or sweep the .5 tolerance. Inspect actual
raw traces and native optimizer state. Keep real Adam, accepted-clock and RNG
semantics; solver preview/rejected updates must not silently alter accepted
controller/EMA state. Factories alone omit the joint transaction; submitted trainer rejects
conditional/MoG/encoder hosts. Staged optimizer step/load hooks can replay
external side effects beyond checkpoint ownership. Those concrete API limits
remain part of integration work. New policies must earn their own results.

RP5 currently survives stationary694/694, all4images and3vectors; long30000 is
running. It combines reversible precision with implicit updates and has shown
close/reopen/close without caller phases. Do not duplicate that lane's mechanism.
DV7 completed30000 and4images but failed rare-component spread in unequal_mass;
DV8 loses stability and DV9 also fails that vector. Those scores remain useful.

COMMON.md and evaluation-protocols.json remain unchanged. Recovery means arrival
and sustained stability, not81/81 or an acquisition deadline. Preserve every
miss and stable suffix. Start cheap sensitive frozen gates when appropriate,
but never interrupt a started fixed window merely to reorder tests. Finish each
three-proposal attempt with a complete report; no automatic unchanged restart.

PublicK3P comparison ownership: RP5 owns eventual matched recovery-ring work.
The next data lane may run one exact publicK3P unequal_mass reference as supporting
comparison before its next mechanisms; do not duplicate either. No default
promotion, merge or publication; eventual integration targets develop.
