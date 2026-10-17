# Joint-word source binding repair

PR339 changed the shared comparison helper to fixed serialized autograd
scheduling and a new comparison version. The current joint-word task retained
the helper's old SHA, so public preflight rejected every word run before
training. The task now pins the current helper and preserves its previous
source SHA and evaluator revision in nested provenance.

Architecture, vocabulary, target law, five-row prior, initialization, named
streams, 20,001 updates, 24 observations, sampling and numerical bounds are
unchanged. All other task and variant files and all scientific code remain
byte-identical. A model-free audit checked all 52 roster declarations: 47
requests planned, five original declaration refusals stayed unchanged, and all
word source-binding blockers disappeared. Other formulation incompatibilities
remain explicit. The representative DualNorm recipe has clear preflight for
all 26 required Tier 1 and Tier 2 tasks. All 84 evaluator pins for those tasks
match their current files. Four metadata tests, Forge validation and inventory
coverage checks passed; no model, optimizer, training update or CUDA kernel ran.

This metadata correction changes the global source manifest from
`3bf51eab2a80eef3645ca5c7df9fa0583cc7053332bc94b19ec874e2e19f98a0`
to `6269a18ac4f82564cb16ba19afa4b3dd2f836a2b4085fbe5aeb81a35a453e895`.
The catalog's historical selected-policy word variant pins the base task JSON
as evaluator support, which planning captures even outside that variant's view.
Consequently every ordinary candidate revision and job key changes despite
identical numerical code. Source origins differ too. Sharing the prior queue
does not automatically make previous results reusable; prior receipts, costs
and grades retain their original cohort. A new admitted cohort is required for
fresh current qualification. This report grants no scientific pass credit.

The unsupported historical selected-policy variant retains its original
parent and source pins. Rebinding that separately scoped ownership question
would require its own declaration.

[Verification receipt](verification.json) records exact identities and source
equivalence checks. [The model-free audit](verify_metadata.py) reproduces the
checks against the retained V5 worktree and roster:

```sh
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python \
  reports/forge/word-runtime-pins/verify_metadata.py \
  --root "$PWD" --baseline /tmp/particlegan-default-tier-refresh \
  --round /tmp/particlegan-default-tier-refresh/configs/forge/rounds/gaussian-smoke-inventory-v5.json \
  --output runs/software/word-runtime-pins/verification.json
```

Local logs are easy to tail under
`runs/software/word-runtime-pins/verification-2.log`.
