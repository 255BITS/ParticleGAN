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

**Ownership:** The recipe owns prior updates and regularization. The task defines the initial prior distribution and capacity, and the sampling contract.

**Selection:** Compare one global recipe across the task suite. Keep failures visible; avoid selecting a different recipe for each task.
