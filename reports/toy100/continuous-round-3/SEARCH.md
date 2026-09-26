> Current work uses the public package default after the master merge.
> Follow `reports/toy100/k3p-base/continuous-search.md` and the baseline handoff.
> Training is deferred until after compaction. The instructions below are
> historical and do not authorize autonomous launches or seed repeats.

# Rolling search with reduced Codex usage

The user reduced concurrent search to **1 Codex + 7 Grok**, replacing the previous
3+5 allocation. Keep at most eight one-worker attempts, four per physical GPU.
No Claude or nested agents. The launcher reserves capacity across all live batches
and launches explicitly selected lanes as earlier attempts finish. Use this
launcher for future launches; old round launchers describe historical allocations.

K3P is still the selected shared base. A3 verification finished **4/22 PASS**;
all three 7000-update natives fail coverage and accuracy. A3 cannot be promoted.
The loop stays: solve the failing toy, verify promising winners on all gates,
then continue from a winner only after complete verification.

Read measured reports and the research notes before proposing a mechanism. Paper
results motivate hypotheses; they do not qualify this GAN. AP3's 1200+300 hold,
120/120 pre-hold and 72/81 recovery are a partial lead, not a verified base. Its
inherited noise still uses a horizon, and its forced reopen dwell loses precision.
No known change times, quality feedback, target centers or task-specific rules.

Run BOTH canonical hold+300 extension and shift for every meaningful proposal.
Do not spend the next proposal's budget before saving both results, even when
one fails. Three proposals is a cap, not a quota. Ledger each completed protocol
immediately in the attempt's tests.jsonl; all incomplete work stays NOT_RUN/ERROR.

A prospective winner needs own hold1200/1200 and extension300/300, all stationary
and continued-hold checks, and recovery81/81. The live driver returns UNCONFIRMED
when these pass: run its matched frozen control immediately to confirm PASS.
Preserve full provenance needed by match_frozen_control rather than dropping
fields in a result wrapper. Require identical pre-shift state and frozen0/81.
Then verify sensitive gates and all22, including original native budgets,
coverage AND accuracy. The same frozen formulation must earn every pass.

An explicitly assigned verification-only lane may screen an unchanged promising
partial lead before full shift success, to avoid spending further search on a
formulation that loses transfer gates. That diagnostic exception does not weaken
qualification, authorize promotion, or inherit any missing passes.

Audit substantial identical prefixes under differing declared horizons, including
model, optimizer, controller, EMA, RNG, applied rates/noise. A successful survivor
also needs separately declared late and repeated shifts on uninterrupted state.
Do not weaken canonical gates, borrow another candidate's scores, repeat seeds,
or run coefficient grids. Leave exact frozen sources and commands for unfinished
verification. Supervisor alone can promote after auditing all evidence.

The user's latest promotion bar additionally requires **4/4 seeds for each of
all three native layouts** (grid100, rotated100, staggered100), seeds1234–1237.
All12 runs must pass full7000-update coverage AND accuracy. Three seed1234 runs
are already in the22; the remaining9 are additional winner qualification explicitly
requested by the user, not exploratory seed experiments. Long-term qualification
uses long-term-stability-protocol.json: same uninterrupted state through30000,
including the original9000 stress windows and another late target change. Stop
all remaining searches and promote in PR155 only after the complete conjunction
passes. Do not stop or promote merely on81/81 recovery or partial toy gates.
