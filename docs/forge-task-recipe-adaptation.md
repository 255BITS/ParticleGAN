# Explicit task adaptation of reference recipes

`release07-gan-v3-task-adapted-v1` makes the released v0.7 GAN v3 training
settings available across Forge's ordinary toy hosts. Its parent,
`release07-gan-v3-mog-v1`, remains a frozen single-native-host comparison with
its original failures and smoke blockers. The successor does not replace or
qualify that evidence.

The new card retains the parent's complete reference `recipe_overrides` and
adds an opt-in `host_adaptation` contract:

```json
{
  "schema_version": 1,
  "recipe_fields": ["batch_size", "num_particles", "total_steps", "z_dim"]
}
```

This example delegates resources only. The actual released-recipe successor
also lists the objective/model fields already owned by behavioral hosts. Forge
removes a reference value only when both the card explicitly delegates that
field and the executing adapter owns it. Unlisted conflicts remain blocked.
No existing candidate opts in implicitly. The contract enters the candidate
revision, so adding or changing a delegation cannot reuse another candidate's
qualification or continuation state.

The shared binding runs during preflight and construction. Workers revalidate
the frozen declaration and task before reservation and dispatch. Adaptation
cannot delegate penalty settings, optimizer knobs, policy controls, prior laws,
initialization, sampling laws, gates or thresholds. Unsupported extensions
remain blockers.

The released dynamics stay fixed: BCap coefficient 6, cap 1.25, every-update
penalty, betas `(0, .99)`, LR `.00425`, prior multiplier 2, the released cosine
schedule, EMA `.995`, and disabled K3P anchor, guard, A2, direct gain and training
noise. Each task supplies its frozen resource values and schedule horizon.
Scalar trainer hosts retain prior spread `.05`. Behavioral component hosts use
their original objectives instead of importing that scalar spread term. AE keeps
the supported reference routing temperature `.25` and distance reduction `sum`.
Task-declared priors and live sampling stay unchanged; the card is a host
adaptation of the released training recipe, not a full release reproduction.

Every applied receipt contains `host_adaptation`, the reference values delegated
for that specific task, its frozen host definition, the effective recipe and
actual component/mechanism observations. Technique report JSON also records
the per-task delegation contract. Reference values alone are not execution
evidence.

The immutable one-candidate campaign reserves at most 44,100 seconds: 900 for
smoke, 39,600 for quality and 3,600 for grouped endurance. These are reservation
ceilings, not observed costs. The ordinary scheduler stops at the first required
failure, including remaining tasks in that tier; it does not spend to fill a
matrix after rejection. All 3/19/2 required cells remain in the denominator.

```sh
python -m experiments.forge plan release07-gan-v3-task-adapted-v1 --through-tier 3
python -m experiments.forge run release07-gan-v3-task-adapted-v1 --through-tier 3 \
  --campaign configs/forge/campaigns/release07-gan-v3-task-adapted-v1.json --gpus 0,1
python -m experiments.forge logs --follow --campaign release07-gan-v3-task-adapted-v1
```

Record a readout and publish the exact executed source cohort using
`reports/forge/regenerate_technique_inventory.py --source-commit <executed-commit>`.
Use a new output prefix when the live source changes. Its `--compose-original`
option accepts a previous composed inventory as well as a frozen publication;
`--compose-current` and `--append-candidate` append one independently graded row.
Recursive publication hashes, compact receipt proofs and original scientific
row contents are checked before writing. This display regenerates without
training or raw-log hydration and does not pool qualification across cohorts.

Raw execution envelopes, per-update streams and source snapshots remain under
ignored `runs/forge/` or an artifact archive. Commit the compact readout, final
metrics, archive manifest and publication summaries.
