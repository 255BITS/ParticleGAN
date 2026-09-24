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

Gate cheaply. Cold-mobility proposals start two_pole80 then cold ring1200;
stability proposals start the200-update diagnostic from the selected base's OWN
checkpoint and stop at the first modes!=8/HQ<.90 observation. Changed policies
must reacquire their own cold state with the same policy. All surviving proposals
need two_pole, ring, unipolar, mid_scale_identity, cover_leftover and the original
ten gates before full own-state1200 and native100 promotion. Do not credit a
partial endpoint, a borrowed state, another variant's passes or skipped tests.
One assigned baseline audit may measure remaining older gates and a1200-step
own-state diagnostic despite the known two_pole failure. Stop that hold at its
first failed check. No native100 runs until older failures are resolved.

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
