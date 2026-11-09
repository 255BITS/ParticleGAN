# Future execution witnesses

New ordinary Forge runtime attempts attach `observed_contract` version 1 only
when the executed adapter receipt contains the resolved recipe, initializer,
protocol seed and executed sampling law, including measured parameter weights
and output noise. The runtime observes its actual software, hardware, thread
count and deterministic settings, and checks these facts against the complete
frozen execution/evaluation contract. Missing or conflicting facts produce
`observed_contract_gaps`; they do not change the numerical grade.

The scalar `_Run` adapter observes sample calls during evaluation, distinguishing
live, EMA and policy-swapped averaged parameters. Mixed observations and adapters
without this explicit observation cannot claim a live-weight witness. Selected
policy contracts require their own matching weight-source observation; a
`state_selected` label alone does not establish one. No witness changes sampling,
adds a sample, modifies weights or advances an executing RNG stream.

After ordinary queue certification, each new durable attempt directory contains
`request.json`, `result.json`, `evidence.json` and a compact `coverage.json`.
Numerical PASS/FAIL rows with complete witnesses contain a producer pointer to
the original attempt ID and final result hash. Operational, partial or unsupported
outcomes remain gaps. The frozen grader remains the numerical authority.

The automatic export never rewrites `reports/forge/attempt-coverage.json` or an
older attempt. A publisher can use the exact-contract export helper to assign a
display family and combine explicitly selected new rows:

```python
from experiments.forge.contracts import read_json
from experiments.forge.observations import coverage_export

export = coverage_export(
    read_json(attempt_directory / "request.json"),
    read_json(attempt_directory / "result.json"),
    read_json(attempt_directory / "evidence.json"),
    family="declared-display-family",
)
```

The family argument is only a display label. The current coverage verifier still
checks the producer triplet, task/source identity, applied facts, full frozen
contract and numerical grading certificate before accepting a row. This helper
cannot retrofit missing observations onto historical evidence or establish
scientific quality from the presence of a witness.
