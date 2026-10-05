# Independent audit of the original-horizon duration diagnostics

Both paid diagnostics fail every new late observation. The original Gaussian
mean drift largely disappears at 3,000 updates, and ring covariance improves at
1,600 updates, but neither clears its original sustained numerical gates. These
results do not justify increasing either ordinary Tier 1 budget. A separately
bounded optimizer contrast is the next supported hypothesis; unchanged longer
training remains unqualified.

The compact [proof](duration-audit.json) comes from the read-only
[checker](audit_duration.py). It is a diagnostic readout, not another leaderboard
or qualification receipt. Historical results remain unchanged.

| Question | Original attempt | Longer diagnostic attempt | Execution / unchanged schedule horizon | Late verdict |
| --- | --- | --- | --- | --- |
| Gaussian acquisition | `d0638ad5ce5a47e5b2fcb00b369768b2` | `1e5fdd69a2d346a4959b626990b621c8` | 3,000 / 1,000 | FAIL at all five checks |
| Ring16 acquisition | `d8185ca486b54af79ef422395eac8065` | `f3f867963570418a86022b23515bb601` | 1,600 / 400 | FAIL at all five checks |

## Exact prefix and provenance

For **each task**, all 24 original observation dictionaries and all 24 retained
sample tensors match exactly. Every numeric difference and sample difference is
zero. The checker verifies the original request, raw result, grading and scored
samples against the byte hashes in the original
[artifact inventory](../tier1-completion/artifact-inventory.json). It binds the new
samples to their saved-output receipt and recomputes the new prefix observation
and sample digests rather than trusting the adapter's assertion.

The actual recipe, prior, initialization, named RNG bindings, runtime and host
match their original run. All 36 relevant public trainer, RNG, API, architecture,
target and scoring source files retain their original hashes. The vector adapter
adds the larger external execution cap, retains the original 24 observation
steps, hashes the prefix and adds five late checks. No training source, original
schedule horizon, target, sampling law or bound changes. All 53 old/new retained
observations per task are rescored: declared gate metrics match exactly. The
ungated ring SW1 diagnostic differs by at most `1.985e-8` during CPU reduction;
this tolerance does not apply to prefix comparison or any gate metric.

**Checkpoint limit:** both original task declarations had
`produces_state=false`, and their archived attempt directories contain no
checkpoint. The new prefix receipt includes complete-context and named-RNG state
digests, but there is no original state digest to compare them with. Exact
samples, metrics and RNG bindings do not prove full optimizer or final RNG-state
identity. If both checkpoints become available, the checker compares complete
state with only `trainer.max_steps` removed; it permits no other normalization.

## Numerical outcome

Gaussian mean error improves from `0.417016σ` at the original endpoint to
`0.001204σ` at 3,000; final standard-deviation ratio is `1.072398`. Every late
check passes the mean and width bounds but fails KS `<=0.05`:

| Update | 2,600 | 2,700 | 2,800 | 2,900 | 3,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian KS | .116633 | .125455 | .099645 | .076921 | .066045 |

With 4,096 samples per check, the five-check simultaneous 99% DKW radius is
`0.029038`; the first three late checks have lower bounds above `0.05`.
Finite evaluation sampling cannot plausibly explain the entire failed suffix.
The final endpoint alone remains closer to the bound and does not establish
what would happen at an untested later duration.

Ring16 retains all 16 modes; all late checks pass high-quality mass and mass TV.
Full-component covariance improves from `4.457005` to `3.027292`, still over
three times its `0.85` bound. Every late check also fails minimum eigen ratio
`>=0.15`:

| Update | 1,200 | 1,300 | 1,400 | 1,500 | 1,600 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full covariance error | 3.835632 | 3.401831 | 3.284391 | 2.927701 | 3.027292 |
| Minimum eigen ratio | .124930 | .143716 | .128544 | .148982 | .138618 |

Final core covariance error is `0.705066`; using it to replace the full covariance
gate would again hide the tail failure identified by the
[original task audit](task-audit.md). Longer settling repairs part of the behavior,
not the whole declared distribution question. Keep the original budgets and
gates for the next optimizer contrast; any future ordinary budget revision needs
its own frozen, passing evidence.

## Reproduce without training

From the repository root, with the original archive's extracted queue and the new
paid queue available:

```sh
PYTHONPATH=. .venv/bin/python -u reports/forge/bcap-tier1-repair/audit_duration.py \
  --original-queue /mnt/ml7tb/experiments/ParticleGAN/tier1-completion-v1/queue \
  --queue runs/forge/bcap-tier1-repair/queue \
  --output /tmp/bcap-duration-audit.json
cmp reports/forge/bcap-tier1-repair/duration-audit.json /tmp/bcap-duration-audit.json
```

Two complete executions produced byte-identical proof JSON and left global CPU
RNG unchanged. Comparator controls rejected changed checkpoint labels, metric
values and sample tensor values on the archived data. This audit added zero
training updates and zero model-sampling draws. Bulk logs and scored tensors stay
in their existing artifact directories.
