Continue the ParticleGAN stable-GAN search from eps_net_1m, the current
experimental base. This is a fresh attempt with no conversation history.
Read AGENTS.md, reports/toy100/h_stability/START.md and current-base.json there.
Use the existing runners and measured blockers. Implement and execute; no essay.

The selected base passes cold ring and all200 own-state short checks, with
minimum HQ.918945313. It still fails two_pole80 at spread.029888831<.30.
Its only change from g_threequarter_rate is fixed G/D Adam epsilon .001;
prior epsilon stays1e-8. G/D/prior rates remain .001125/.0015/.00225. Use
adam_response_cold.py and selected_base_probe.py to apply the whole recipe.

Use the assigned family and small batches of at most3 meaningful proposals.
Execute the first batch within5minutes. Candidate budget is a CAP. Stop a family
when measured failures offer no useful next intervention; no broad parameter
grid, seed sweep, repeated coefficient sweep or endless audit tooling. Read the
runtime supervisor.md before every batch. Keep logs concise and easy to tail.

Every G and particle update must come through a discriminator's adversarial
objective. No target fitting, labels/centers, metrics in training, LR decay,
freezing or shrinking-step schedule. Fixed algorithmic preconditioning is a
declared proposal and must retain positive movement in all trainable directions.
Keep frozen host architecture, seeds, budgets, data and scoring unchanged.

Gate cheaply on toy coverage, but prioritize STABILITY AFTER CONVERGENCE over
smooth learning. Do not reject a useful policy solely for a pre-convergence dip
or failure of the historical step1200-immediate-hold diagnostic. Keep frozen
cold toy verdicts and budgets unchanged, including failed ring suffixes.

Use the separate first_convergence_then_hold_v1 diagnostic in converged_probe.py:
from the candidate's OWN completed cold1200 state, allow up to4800 further
settling updates. The FIRST200 consecutive dense checks with8 modes/HQ>=.90
confirm quality convergence. The NEXT1200 updates form a disjoint hold with the
same policy, models, Adam state, RNG and fixed rates; no reset at convergence and
no restart after a failed hold. Early dips are recorded but do not fail the hold.
NOT_CONVERGED and POST_CONVERGENCE_FAIL are distinct outcomes. Quality
convergence is an operational criterion, not a proof of mathematical equilibrium.
Metrics observe training only. No metric-conditioned update/freeze/decay is allowed.

At most1-2 promising policies per attempt get this bounded diagnostic even if
an earlier strict hold failed; a completed cold-ring FAIL may be studied for late
convergence, but it keeps its original frozen-budget FAIL. No promotion by
renaming a failed old gate. Rank toy coverage, time to confirmed convergence,
and post-convergence retention separately. All original toy gates, retention
cases and native100 remain required before claiming a release-qualified winner.
The baseline older19 audit is complete (10PASS/9FAIL); do not repeat it.

Retain exact configs, effective rates, source per batch, raw metrics/checkpoints
and honest FAIL/ERROR/SKIPPED statuses. A setup error is not a GAN failure; repair
it once before repeating the batch. Reuse existing state/receipt validators.
Do not redo the immutable H audit or build another general research framework.
Run relevant existing regression tests after code changes. Run the full suite
once only for a viable final candidate; do not repeatedly spend minutes auditing
rejected proposals. Reserve enough time to leave runnable code and clear results.

The user explicitly wants a GAN, not a replacement data fitter. The auxiliary
AE/unused-token structural blockers are already documented. Do
not spend another attempt proving them or silently add supervised objectives.
Describe any new adversarial training signal and its protocol limits explicitly.

Required output: reports/toy100/h_stability/RESULTS.md, plus the runtime result.md
and tests.jsonl specified below. Summarize executed proposals and gates, actual
unit-test totals, best measured result, remaining failures and exact replay
commands. Mark audit/regrade rows as candidate=regression, never as toy passes.
Do not publish GitHub changes or launch nested agents. Start testing now.
