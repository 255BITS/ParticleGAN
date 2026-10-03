# Existing configurations: expanded Tier 1 readout

This bounded refresh evaluates the **15 existing ideas and 32 saved
configurations** against all five required Tier 1 tasks in
`discriminator_stability` revision 3. The
[single current leaderboard](../technique-inventory.md) contains the selected
whole configuration for each family; its companion JSON retains every
configuration alternative. The [compact readout](readout.json) explains each
declaration's measured outcome, first failed requirement and next action.

The [regenerated selection configs](../../../configs/forge/selections/tier1-existing-configs-v1.json)
export the complete frozen recipes selected by the five refreshed studies.
Failed incumbents remain labelled best observed. These projections supply no
qualification input and do not change package defaults. All 32 scientific
configuration cards retain their exact IDs and bytes.

## Results and recommendations

**No declaration passes all five Tier 1 requirements.** The 43 admitted
candidates stop at scientific FAIL; four are blocked before training. All 111
executed attempts completed without execution errors, spending **1,135.233330
paid seconds**. No Tier 2 or Tier 3 task ran. Across the 32 saved configurations,
11 fail movement, 19 fail ring acquisition and two pass ring but fail joint word
acquisition.

| Required Tier 1 task | PASS | FAIL | BLOCKED | UNKNOWN |
| --- | ---: | ---: | ---: | ---: |
| `two_pole` | 22 | 21 | 4 | 0 |
| `unused_token_hold` | 22 | 0 | 4 | 21 |
| `ae_gan_hold` | 22 | 0 | 4 | 21 |
| `ring16_acquisition` | 2 | 20 | 4 | 21 |
| `five_word_joint_acquisition` | 0 | 2 | 4 | 41 |

Unknown cells follow the prerequisite stopping rule. They remain in the
denominator. The five study reports share one campaign accounting ledger;
their repeated campaign totals must not be summed. The paid total above sums
each original attempt once and excludes coordinator overhead.

The current leaderboard selects K3P `0b37e98a` and KA2 `093c6f2b` at **4/5**.
BCap `08689a73`, R1/R2 `302b6baa` and released GAN v3 `1e266b5a` reach **3/5**.
These are deterministic best-observed whole configurations, all unqualified;
the export labels them accordingly. The selector maximizes required PASS count
and breaks ties by configuration hash. It supplies neither a speed ranking nor
a task-by-task composite recipe.

Every measured ring endpoint covers all 16 modes, but 20 candidates fail the
component covariance gate: error ranges from 0.99521 to 14.13970 against the
0.85 ceiling. Some also fail component spread or high-quality fraction. Mode
coverage alone therefore hides the main ring failure. K3P and KA2's selected
configs pass ring with covariance errors 0.62165 and 0.63614, respectively, and
the required five passing terminal observations.

Their word failures differ. K3P finishes with confident outputs but only 3/5
modes, mass TV 0.4 against a 0.1 ceiling, and failed inverse reconstruction.
KA2 finishes with all endpoint bounds passing (5 modes, TV 0.01895,
reconstruction exactness 1.0, minimum correct-token probability 0.94784), but
recurrent reconstruction failures leave only **one of five required terminal
passing observations**. Its good endpoint cannot turn that FAIL into a PASS.
Both word runs complete all 20,001 declared updates. A2 was not activated under
the observed host conditions: eligible/applied updates are zero, and its separately labelled synthetic
component probe is a fidelity check. These failures cannot establish an effect
of active A2 damping. Two R1/R2 movement failures likewise have passing endpoint
metrics but insufficient terminal persistence.

Retain current defaults and stop these exact revisions. First inspect the
archived numerical component moments, spill and reconstruction trajectories to
scope a substantive next hypothesis. Ring component spread and persistent word
inversion are the remaining measured bottlenecks; no seed repeats or unchanged
continuations follow this readout. Calibrate the expanded provisional screen
before using its results for default adoption. Resolve the four explicit host
compatibility blockers before attempting those cohorts.

## Scope and evidence

The [frozen roster](../../../configs/forge/rounds/tier1-existing-configs-v1.json)
binds one shared [campaign](../../../configs/forge/campaigns/tier1-existing-configs-v1.json):
98,700 seconds maximum, 2,100 seconds per declaration. It includes both
previous R1/R2 grids through an exact finite union, with no new settings or
seeds. Only Tier 1 is authorized. Required non-passes stop the remaining work,
including later tasks within Tier 1; unknown and blocked cells retain the full
5/19/2 denominator. The profile remains provisional and supplies no calibrated
default-adoption or robustness claim.

Executed source commit: `2899099048c0a9987eb8720214abfce56d86d92a`.
Scientific source digest: `7306340bac0a4ea67ea7b080513116457b7d72a8cb0d0c1f0727eaae6db38185`.
The current runtime is Python 3.14.7, Torch 2.14.0+cu130, an AMD Ryzen 9
5900X CPU and an NVIDIA RTX A6000 with driver 610.57.04. The three behavioral
tasks run on the bounded CPU lane; acquisition tasks use GPU 0 with one worker.
GPU 1 is excluded. Paid attempt seconds are cost evidence, without a speed
ranking. The initial coordinator's memory-rebuild overhead is excluded from
paid task cost; batch execution subsequently retains the same completed
receipt and frozen requests.

Earlier revision-2 outcomes keep their original 3/19/2 denominator and source,
runtime, prior, initialization, recipe, sampling and budget identities. They
remain in the publication's archived numerical evidence and cannot fill the
new acquisition requirements. Current task gates and budgets are unchanged.

The [archive receipt](archive.json) records the local 43,004,589-byte artifact
at `artifacts/forge/tier1-existing-configs-v1.tar.gz`, its SHA-256, 111 original
attempt certificates and 1,137 byte-exact executed source files. It contains
the campaign's raw results, numerical observation streams and task logs. Verify
its `restore-manifest.json` before hydrating the exact originals into an isolated
checkout; the receipt includes restoration steps. The archive stays outside
Git. Earlier archives remain linked from the
[historical inventory readout](../TECHNIQUE_INVENTORY_READOUT.md).

Four declarations are blocked before training: E22 and Atlas require explicit
policy-aware task contracts; the original released-v0.7 cloud and MoG cards
override fixed host-owned fields. Their exact blockers remain visible, with
zero executed attempts. The word idea's goal-only admission-pin correction
retains its original scientific recipe and archived declaration identity.

Retained question illustrations use actual public-API training GIFs:
[ring acquisition](../../toy_audit/api_contract/ring16/README.md) and
[joint word acquisition](../five-word-joint/README.md). Their separate demo
protocols confer no qualification on these configuration receipts. Scientific
decisions here use the declared numerical gates.

## Reproduce and regenerate

Plan and freeze every declaration before launching any worker:

```sh
python -u reports/forge/prepare_tier1_existing_configs.py plan
python -u reports/forge/prepare_tier1_existing_configs.py enqueue
# Drain the common campaign in batch mode; publish once after it finishes.
python -u - <<'PY' > runs/forge/tier1-existing-configs-v1/worker.log 2>&1
from pathlib import Path
from experiments.forge.queue import Queue, drain
root = Path.cwd()
queue = Queue(root / "runs/forge", report_root=root / "reports/forge")
drain(queue, ["0"], workers_per_gpu=1, campaign="tier1-existing-configs-v1")
PY
tail -F runs/forge/tier1-existing-configs-v1/progress.jsonl
```

Identical scientific requests reuse compatible evidence. Reproduce the
executed checkout and runtime for exact qualification; a later source or
runtime creates a separate cohort. Ordinary failures must remain stopped.

Refresh each completed study with `forge search report STUDY_ID`, using the five
IDs in the roster. This launches no training. Register the new view policy
once from its verified original receipts, then regenerate cached projections:

```sh
python reports/forge/regenerate_technique_inventory.py --device cuda \
  --source-commit 2899099048c0a9987eb8720214abfce56d86d92a --advance-policy
python reports/forge/regenerate_technique_inventory.py
python reports/forge/tier1-refresh/regenerate.py
python -m experiments.forge experiments-by-tier \
  --output reports/forge/EXPERIMENTS_BY_TIER.md
python -m experiments.forge compile --summaries-only
python -m experiments.forge compile --check
```

The first command needs the byte-exact originals. Cached leaderboard,
selection-config and readout regeneration uses committed compact evidence
without artifact hydration, retraining or regrading historical results. Keep
raw stdout, observation streams, checkpoints and complete execution envelopes
under ignored `runs/forge` or in the local artifact archive.
