# Current Tier 1 measurement round

The completed round measured **26 PASS and 45 FAIL**, with **six explicit
no-attempt policy blockers**, across eleven fixed recipes. All 71 numerical
results have [actual-training GIFs and provenance](media.json). Total charged
execution was **2,055.6923 seconds (34.3 minutes)**, with no infrastructure
errors, retries, seed repeats or later-tier work.

[Current family leaderboard](../technique-inventory.md) ·
[Final metrics and exact attempts](results.json) ·
[Validation receipt](validation.json) ·
[Bulk archive identity and member hashes](artifact-inventory.json).

Among the nine ordinary recipes, BCap, K3P and KA2 tie at four of six required
passes. None passes the whole required tier. The objective-specific choices in
this measured cohort are K3P/KA2 for ring acquisition and BCap/K3P without
training output noise for joint word acquisition. These observations do not
select a universal default or qualify later tiers; costs remain separate from
scientific ranking.

| Ordinary required test | PASS | FAIL |
| --- | ---: | ---: |
| Gaussian acquisition | 0 | 9 |
| Two-pole learning | 5 | 4 |
| Unused-token retention | 8 | 1 |
| AE/GAN retention | 9 | 0 |
| Ring acquisition | 2 | 7 |
| Joint five-word acquisition | 2 | 7 |

Passing the final observation alone is insufficient. K3P and KA2 satisfy the
terminal Gaussian bounds but retain only two or three consecutive passing
checks, below the required five. KA2 also passes the final word bounds without
the required sustained suffix. Read the saved curves and failed bounds before
attributing a failure to a formulation change.

All nine separate ordinary clock diagnostics fail their declared parity/source
checks. Atlas and E22 each have four measured scoped FAILs and three
explicit ownership blockers (unused-token, AE and joint words). Their clock
probes also detect restart differences. These policy measurements neither
replace clean/MoG parent results nor establish clock-free eligibility. Remaining
execution markers (*) identify tests with no recorded execution, including
preflight blockers. A recorded FAIL completes execution coverage. Differences
between a recorded run and today's test definition are shown separately and
do not add (*); the recorded result supplies no new qualification.

The current `clockfree_continuous` view revision 3 counts the already measured
`clockfree_audit_measurement_v1` as its required Tier 1 clock test. All nine
ordinary families therefore have complete Tier 1 coverage; BCap displays
**19/22**, including its clock **FAIL**, without an incomplete marker. The
original eligibility audit remains required in Tier 3 before the three 14k
continuations. Its task contract and the [archived revision 2 view](../../../configs/forge/view-history/clockfree_continuous-v2.json)
remain unchanged. This updates current coverage navigation only: the round's
frozen discriminator-stability qualification, source, attempts, metrics and
GIFs are preserved, with no new training. The Tier 1 reservation ceiling is
unchanged; the full clock view adds the probe's declared 300-second ceiling.

The [portable validator](validate.py) checks recipes, source/runtime/evaluator
certificates, numerical terminal rules, costs, tier limits and every selected
attempt's GIF identity without changing a grade or running training:

```sh
python reports/forge/tier1-completion/validate.py --queue-root "$PWD/runs/forge"
```

`tier1-completion-v1` measures the existing selected global recipe for each of
the eleven current trainer families. MoG/cloud remains task-owned. The roster
freezes candidate declarations, task definitions, view fingerprints and task
budgets; it introduces no recipe search, seed repeat or default promotion.
Five selected configuration cards use singleton search registrations for the
current source, task scope and campaign. Each registration retains the exact
existing card and already declared setting. Enqueue registers all five through
Forge's bounded search workflow before admitting the full eleven-family roster.

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
Use `archive --archive-path /path/to/external/artifacts.tar.gz` to keep the bulk
archive on another volume; the receipt records its exact location and hash.
The final inventory remains the single current leaderboard. Complete current
measurements can contain FAIL; qualification still requires the declared gates.
