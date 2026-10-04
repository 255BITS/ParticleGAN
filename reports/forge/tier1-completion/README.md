# Current Tier 1 measurement round

`tier1-completion-v1` measures the existing selected global recipe for each of
the eleven current trainer families. MoG/cloud remains task-owned. The roster
freezes candidate declarations, task definitions, view fingerprints and task
budgets; it introduces no recipe search, seed repeat or default promotion.

Nine ordinary recipes have six required discriminator-stability tests and a
separately scoped clock audit diagnostic. Atlas and E22 have seven separately
named policy-cohort variants: four runnable tests and three explicit parameter
ownership blockers. Their results do not qualify the original clean tests.
Known scheduled clock dependencies are measured as failures by the diagnostic;
the original clock-free qualification contract remains unchanged.

The first-attempt ceiling is 27,720 charged seconds. The campaign ceiling is
55,440 seconds, allowing at most one documented infrastructure-repair retry
per authorized task. Scientific failures are final. The complete-current-tier
policy finishes eligible independent tasks after a failure and stops before
Tier 2. Workers use GPUs 0 and 1 plus Forge's CPU slot where tasks declare it.

Run from the reviewed, committed source with the repository Python environment:

```sh
python reports/forge/tier1-completion/run.py plan --queue-root "$PWD/runs/forge"
python reports/forge/tier1-completion/run.py run --queue-root "$PWD/runs/forge" --expected-commit "$(git rev-parse HEAD)" > runs/forge/tier1-completion-v1/driver.log 2>&1
tail -f runs/forge/tier1-completion-v1/driver.log runs/forge/events.jsonl
python reports/forge/tier1-completion/run.py report --queue-root "$PWD/runs/forge"
python reports/forge/tier1-completion/run.py media --queue-root "$PWD/runs/forge"
python reports/forge/tier1-completion/run.py archive --queue-root "$PWD/runs/forge"
```

Create the log directory before redirecting output. `prepare` is a one-time
source preparation step, already committed with the round. `run` verifies every
captured source byte against the supplied Git commit before submitting work.
Re-running it resumes the same queue identities and budgets.

Publish only compact metrics, original evidence certificates, reproduction
sources and actual-training GIFs. Raw logs, saved arrays, state and frozen source
snapshots stay in the ignored local queue and byte-exact artifact archive. Its
hash and members are recorded in `artifact-inventory.json` after execution.
The final inventory remains the single current leaderboard. Complete current
measurements can contain FAIL; qualification still requires the declared gates.
