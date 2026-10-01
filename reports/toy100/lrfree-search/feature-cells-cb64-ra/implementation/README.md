# CB64-RA package and config

The actual package is `../pkg-CB64-RA`; the generic recipe overrides are `../configs/overrides-CB64-RA.json`. Source and config are frozen in [READY.json](READY.json), with absolute-path SHA256s, package digest and focused test receipts. Independent acceptance runs use this exact freeze. The reference E22 package and previous studies remain read only.

Eight focused CPU integration tests pass. These verify implementation contracts, not training quality. Nonzero ordinary moves and exact checkpoint continuation are exercised using a saturated learned-feature fixture whose support atoms remain stable through zero-gradient updates. Legacy stationarity diagnostics contain NaNs in that fixture; replay compares tensor bytes, including NaN payloads, while excluding only wall-clock `eval_seconds`. Model, optimizer, private/global RNG, controller, LR tester, FIFO and row state all match exactly.

## Use

Put the candidate package first on Python's module path when running the existing trainer/harness and supply the new overrides JSON. This changes the opt-in backend; ordinary E22 recipes keep `birth_death_backend="knn"` and the original behavior.

```bash
cd /ml2/hypergan/gan-attempts/feature-cells-config-20260929
PYTHONPATH=/ml2/hypergan/gan-attempts/feature-cells-config-20260929/pkg-CB64-RA \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/pr38-default-env/bin/python -u implementation/test_integration.py
```

Applications construct their networks and real stream as usual:

```python
import json
from pathlib import Path
from particlegan import Recipe, GANTrainer

options = json.loads(Path("configs/overrides-CB64-RA.json").read_text())
recipe = Recipe(**options)
# Apply the problem's established particle/dimension/batch settings here.
trainer = GANTrainer(recipe, generator, critic, prior=prior, seed=seed)
trainer.step(real_batch)
```

The config inherits E22's generic N=12/z_dim=4 fields. Existing problem harnesses resolve their established table size/dimension overrides. At N<20, floor(.05*N)=0 ordinary transport and the isolation guard prevents even one flagged row from moving; the config does not invent a larger minimum move. At least six particles are required. Keep the recipe/model/prior dimensions consistent.

New Recipe fields are `birth_death_backend="feature_cells"`, `birth_death_cells=64`, `birth_death_metric_rank=8`, `birth_death_chunk=256`, and `birth_death_parent_policy="real_anchor"`. The backend requires a particle GAN, learned critic head features, reference standardization and DV12 optimizer/controller scaffolding. Invalid combinations are refused. Legacy recipes omit these added fields from `to_dict()`, preserving old checkpoint/config serialization. The original `birth_death.py` is byte-identical to E22.

## Public interfaces and diagnostics

`particlegan.feature_cells` exports:

- `FeatureCellSnapshot.fit(real_features, *, generator, cells=64, rank=8, chunk=256)`; all inputs/results stay on the same device. Metric rank is bounded by active reference units and reference degrees of freedom.
- `transform(features)` and `assign(features)` returning cell IDs and squared distances.
- `support(query_features)` returning flags, p-values and nonconformity scores.
- `select_parents(query_features, flags, *, ordinary_children=None, generator, pvalues=None)` returning child IDs, parent IDs and detail. Detail includes actual bounded `candidate_ids/candidate_mask`, actual even-real `anchor_reference_rows`, `anchor_cell_ids`, `parent_cell_ids`, inaccessible deficit and parent-cell deficits.
- `cell_comparison(fake_features)` and `ordinary_transport(query_features, flags, comparison, *, generator, pvalues=None, max_moves=None)` returning the count comparison or a child/parent/detail transport plan.
- `cache_queries(query_features)` and `refresh_rows(rows, repaired_features)` returning the conservative affected-cell invalidation mask after nonempty copies.
- `FeatureCellBirthDeath`, the actual GANTrainer backend. It reuses inherited FIFO, head capture and row-copy mechanics while replacing the full evaluation/reaction/jitter path.

Snapshot `.work` records distance cells, projection products, count-test terms, parent candidate cells, maximum distance dimensions and retained snapshot tensor-storage bytes. Retained snapshot bytes exclude the raw FIFO, transient q/R/F feature tables, model workspace and interpreter; use process peak memory for complete memory accounting. `last.eval_seconds` is CPU wall time; CUDA callers need synchronization to measure device work.

Actual ordinary mechanism diagnostics are `counters.cell_evals`, `counters.cell_discoveries` (certified cells), and `counters.ordinary_moves` (executed row pairs). Latest values are `last.ordinary_discoveries/ordinary_moves`. Isolation uses `counters.iso_evals/iso_flagged/iso_moves` and `last.iso_moves`. `last.moves` is the total row moves used by existing tester and row-evidence reset hooks. The inherited `counters.moves` retains E22's ordinary-only convention. `S/W/n` do not accumulate kNN evidence on this backend; each rebuilt partition clears them. Their inherited k/s_k diagnostic values have no statistical role.

Checkpoints version the backend/settings, preserve FIFO/sample shape/private stream/counters, and reject incompatible settings, absent/nonfinite/mismatched FIFO tensors or cross-backend resumes. Derived feature snapshots are discarded on load and rebuilt before use; fixed jitter does not require that cache between evaluations. Existing GANTrainer resets the stationarity tester and row evidence for `moved_rows`; direct external `maybe_apply` callers must apply the same caller-side notifications if they own those testers.

## Changed semantics and limits

Ordinary actuation uses exact conditional two-sample categorical tests on odd-real versus iid fake cell counts, Bonferroni .05/K, and integer excess/deficit transport quotas capped at 5%N. It does not reuse the kNN Beta, dimension estimator or sequential excursion law. Reused/adaptive data snapshots have no cumulative calibration guarantee. Clean-center cell assignments can differ from noisy fake-pool counts; this is an approximate count-transport reaction, not an exact density flow.

Isolation retains full-calibration support and the duplicate/5% guard, then anchors to actual real features and searches at most four pools of 64 supported parents. This is local real-feature guidance. It can move mass into a nearby rare component; supported-parent validity and aggregate mass matching remain separate gates. K64 can merge more than 64 modes, rank8 can discard support directions, finite calibration can have zero power, and empty supported cells cannot be restored by cloning. All-zero learned features cause no action.

Training, sampling, fake evaluation and repairs share Gaussian latent std=.025 with norm cap=.05, intentionally replacing DV12's adaptive bandwidth/half-neighbor cap on this config. No all-table nearest-distance sampling pass remains. The fake pool uses the supplied `sigma_out`; GANTrainer supplies the sigma used in the preceding training update. A learned sigma can change during that update, so this is not a claim of identity with the newly updated served output-noise value. BD evaluates the current fast generator and critic in eval mode; serving averages/EMA parameters and training-mode stochastic/BatchNorm layers define distinct model snapshots, as in the existing package. The common latent-noise kernel is aligned; model/noise snapshot differences are disclosed.

G→D capture is chunked at 256, including repaired rows. The raw FIFO remains O(N*outdim); chunking does not solve large-image reservoir storage. Fixed K/rank/parent budgets remove all hidden N² neighbor/parent/stale passes from this backend; reference fitting, count tests, grouping and cache refresh still rebuild at every FIFO turnover. No source/config tuning is permitted after acceptance outcomes. Detailed fixed semantics are in [PROTOCOL.md](PROTOCOL.md).

Before READY, the focused suite corrected one Torch argsort keyword error and tested reviewed rank/FIFO guards. A replay test fixture/comparator correction distinguished discrete support from training-quality claims and identical NaN diagnostics from semantic divergence. These changes are recorded in the frozen run.log. Acceptance results and their commands belong to the independent geometry/validation lanes.
