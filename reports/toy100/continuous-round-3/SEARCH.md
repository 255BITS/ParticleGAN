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

Audit substantial identical prefixes under differing declared horizons, including
model, optimizer, controller, EMA, RNG, applied rates/noise. A successful survivor
also needs separately declared late and repeated shifts on uninterrupted state.
Do not weaken canonical gates, borrow another candidate's scores, repeat seeds,
or run coefficient grids. Leave exact frozen sources and commands for unfinished
verification. Supervisor alone can promote after auditing all evidence.
