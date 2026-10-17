# Fixed mechanics and review protocol

Freeze source/config/helper and all raw input hashes before Torch imports,
PT interpretation, constructors or forwards. Both independent source reviews
must pin the same preseal. Preserve failed attempts separately; never retune
the witness, clipping, category conditions, budgets or scoring from results.

run_mechanics.py runs ONE actual new-law reaction per frozen final RA9 grid
and toy input. It supports --device cpu for this owner and --device cuda only
for root's serialized numerical slot. CPU hides CUDA before importing Torch.
CUDA retains root's visibility, caps process memory at .2, disables TF32 and
uses deterministic algorithms. No agent launches the CUDA branch.

Both branches instantiate fresh backend9 GANTrainer with fixed mechanics
seed314159 and device-native dedicated streams. They copy raw G/D/FAST/EMA
weights/table and the original real FIFO, which enters through observe_real.
For meaningful copy accounting, the explicitly frozen prior optimizer row
tensors/scalars, latent history, controller latent bandwidth and learned
log-sigma (when present) are also copied. All other controller/settler/RowEvidence/
lineage/backend semantics start fresh. No backend8/full checkpoint is loaded
or relabeled, and CPU RNG bytes are never passed to a CUDA generator. This is
neither historical GPU replay nor cross-device bit equivalence.
The original tester.begin pre-update anchor boundary runs once; no intrinsic
clock or gradient observation is advanced. Initial own RowEvidence/participation
are zero, which is explicit: hook masks/anchor/reset counters and full row
coverage are checked; inherited historical positive evidence is not fabricated.

One maybe_apply executes the original internal count-sampling draw and noise
law, no quality sample/scorer. The literal unchanged trainer caller rebase,
RowEvidence reset, completed_steps and serving hook are extracted from frozen
training.py and executed without a GAN gradient or optimizer step. Its fresh
mechanics clock is 0->1, within the original budget, with genuine before/after
backend9 checkpoints. Model weights and dedicated training streams must remain
unchanged. Grid must have a firing witness and legal positive actual mean
copies; toy must veto without mean preview/draw. All phase sums, shared budget,
complete reservations/moved hook union and JSON scalar diagnostics must pass.

Output per case:

- before.pt and after.pt: actual fresh backend9 full trainer states.
- provenance.json: original raw input/source maps, fixed construction scope,
  before/after step and output hashes.
- trace.json: old copy calls, ordinary kind0/1/2/3 IDs, novel sources, complete
  reservations, kind4 child/parent IDs, moved union and actual caller hooks.
- hook-state.pt: actual tester/RowEvidence before and after the unchanged hook.
- grid/prepared-packet.pt: exact precommit state/chart/packet/model bytes for
  independent focused controls. It is NOT a resumable trainer checkpoint.
  Pointer/version epoch must be rebound explicitly to reconstructed objects;
  coordinates, parent moments/history and feature bytes are never regenerated.
  Existing precomputed FAST/EMA metric/category/group/eligibility/pvalue views,
  fixed action context, pair/protected-row pools, complete reservations and
  pre-draw stream are exported without an extra generator query. A focused
  independent control may replay that fixed preview, not the full planner.

The mechanics helper independently checks exact committed coordinates/history,
both-view support/inside/category/group retention, source preservation and
actual pre-mean objective progress. Additional focused packet/observer and
typed checkpoint/cold-load/continuation controls are owned by lineage_finish
and state_review in their separate prefrozen helpers. They reuse these exact
reacted fixtures rather than repeat the grid planner. The final mechanics
result has literal status PASS only when all declared checks pass.

Root-only command (output last argument):

    /tmp/pr38-default-env/bin/python run_mechanics.py --device cuda --package-root pkg-MEAN --output /ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/integration/review/ra10-mechanics-gpu/result.json

The final READY expands helper/package paths absolutely. Root's immutable
quality/run_ra10_mechanics.py owns the inherited GPU lock and before/after
guards. This mechanics result provides no quality acceptance; unchanged final
toy25 and full canonical grid100 validation are separate and required.
