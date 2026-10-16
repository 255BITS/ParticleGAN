# Research glossary

| Term | Meaning |
|---|---|
| **Technique / formulation** | A training method, such as BCAP. |
| **Recipe** | A concrete configuration of a technique: losses, weights, optimizer, prior learning or freezing policy, and prior regularizer. |
| **Task** | The problem to solve, including architecture, target data, initial prior distribution and capacity, sampling, and evaluation criteria. |
| **Protocol** | The fixed comparison rules: initialization, data sequence, update budget, evaluation cadence, and scoring. |
| **Experiment** | A comparison of declared recipe changes under a fixed protocol. |
| **Run** | One execution of a recipe on a task. |
| **Metric** | A numerical measurement of performance. |
| **Gate** | A declared pass/fail requirement. |
| **Baseline** | The reference recipe used for comparison. |
| **Representative recipe** | A technique’s best observed complete recipe under the declared tasks, gates, and budgets. |
| **Counted requirement** | A test with the role `required` in a view. It counts toward the leaderboard total and toward qualification; a test in several views is one requirement per view. |
| **Held test** | A test with the role `held`: still declared, run and shown with its history, but not counted and never blocking progress. Its view stores the reason, a summary, an explanation link and the rule for counting it again. |
| **Removed from the count** | A test that stays assigned as a diagnostic with a note naming why it is not counted: a phase or duplicate of a counted test, a superseded definition, or a test that cannot run as declared. |
| **HELD status** | Reported for a tier with only held tests. A view reports it when every counted tier passes but its highest tier has only held tests, or when it counts nothing; a counted failure still shows as a failure. It is never PASS and never eligible. |
| **Archived view revision** | The exact bytes of an earlier view revision, copied to `configs/forge/view-history/<view>-v<N>.json` before the view changes. Recorded results keep pointing to it. |

**Ownership:** The recipe owns prior updates and regularization. The task defines the initial prior distribution and capacity, and the sampling contract.

**Selection:** Compare one global recipe across the task suite. Keep failures visible; avoid selecting a different recipe for each task.

**Holding:** A test is held when no one global recipe passes it together with every counted test, when no recipe has passed its current declaration, or when it continues a held test. Holding changes what is counted, not any recorded result; see the [2026-10-15 test dispositions](reports/forge/test-dispositions-20261015/README.md).
