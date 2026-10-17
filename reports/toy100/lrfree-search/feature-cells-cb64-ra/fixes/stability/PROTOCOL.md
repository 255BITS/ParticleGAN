# Stability and mass contribution

This is a causal implementation proposal developed from the completed frozen CB64-RA GPU screens and the existing fixed seed-90229 feature fixtures. CPU checks establish contracts and fixture mechanisms. They do not replace the original CUDA quality screens. No seed sweep, quality gate change or training-budget reduction is used.

## Scope and source ownership

Only the private `stability/pkg` copy is edited. The original candidate, original screen/scorers, reference package, old CPU studies, shared integrated candidate and frozen CUDA lane remain untouched. This contribution changes `feature_cells.py` and `training.py`. Geometry owns the active-cell kernel; performance owns fit/count/pool implementation; support owns detector score changes. Root composes these changes and runs every GPU job serially.

## Small populations

The N-row FIFO gives M=floor(N/2) heldout rows. Minimum conformal p is 1/(M+1), so BH at Q=.05 can only flag sets of size at least ceil(N/[Q(M+1)]). The unchanged action guard permits floor(QN). If the minimum exceeds the maximum, the factory returns the unchanged `ParticleBirthDeath` reference law and the original controller sampler. Exact rational arithmetic chooses the route. The present policy is infeasible below 800 rows; 800 is derived, rather than tuned against task outcomes.

Requested backend, actual backend, sampler route and the complete arithmetic are recorded in diagnostics and checkpoints. Reload requires matching metadata. The active class refuses direct construction at an infeasible population. `GANTrainer._generate` dispatches through the selected operator's sampler method; a fallback operator has no feature-cell sampler and uses the reference controller. Six CPU updates on each original mode_hold and blobs4 host verify exact reference models, optimizer state, streams, controller and serving samples. These short checks are contract tests, not quality runs.

## Large-population mass and diversity

The original fine-cell count/support tests and their heldout calibration remain unchanged. Action targets use all reference rows. A real-only bounded K-center minimum spanning tree is split by the deterministic maximum between-class variance of its log squared edge lengths. Derived connected groups coarsen the action mass topology; no fake/query row or oracle label chooses the groups. This topology is an action constraint, with no additional statistical validity claim.

Ordinary moves require the original count certificate and actual clean-cell surplus/vacancy. Within-group moves match within that same group. Transfers between groups additionally require clean group surplus and group vacancy. Counts remain bounded by floor(.05N), eligible pool supply and the existing 64-row reservoir. A real-supported eligible row is protected in each represented cell. Parents are drawn without replacement.

Isolation retains the original BH flags and the .05 population guard. Supported rows and planned ordinary moves reserve group mass first. A child proposes to its nearest real representative with remaining group vacancy and parent supply. Contested local capacity goes first to children with the largest distance penalty for their next accessible group, with stable distance/row ties. Rejected proposals try another nonfull target. The parent is a distinct supported row from that same target cell's bounded pool. Ordinary children and parents are excluded from isolation parent supply. Unmatched holes are reported and left for later snapshots.

The original cross-cell 2×nearest/16th-distance parent ball is replaced by supported membership of the chosen target cell. Keeping that ball with a one-copy limit collapsed supplies to 1–4 rows and forced distant repairs. The new law records its pool and assignment policy explicitly. Geometry's adaptive latent copy/training kernel composes separately.

## Bounded work and continuation

The additional topology uses K×K work with K<=64. Repairs use at most .05N children × K anchors, in query chunks; no N×N or flagged-row×flagged-row matrix is introduced. At most K contested rounds remove accessible cells/groups. These are bounds, not measured throughput guarantees; GPU host synchronization must still be profiled after composition.

The active checkpoint schema is 3, with mass-policy settings and population route metadata. Old unconstrained/fixed-kernel continuation is rejected. Derived topology lives only in rebuilt snapshots. Root must preserve geometry's cache invalidation and kernel metadata while composing the patch.

## Verification and commands

CPU commands use `/tmp/pr38-default-env/bin/python`, `CUDA_VISIBLE_DEVICES=''`, one numerical thread and `PYTHONDONTWRITEBYTECODE=1`:

```bash
/tmp/pr38-default-env/bin/python -u stability/test_stability.py
/tmp/pr38-default-env/bin/python -u stability/diagnose_mass.py
```

The root-only GPU command in `READY.json` checks allocation on frozen CPU partition/score inputs for nominal and rare-hole fixtures. It does not fit a new detector or give a GPU quality verdict. Physical GPU0 UUID, memory fraction .2 and one numerical thread are guarded. Source and input hashes are recorded. Root should run original mode_hold (1200), img_blobs4 (600) and vector_unequal_mass (1200) screens on the composed candidate, with the original fixtures, live noisy gates, serial/public/plain options and saved states. Native and learned-model efficacy require their original full budgets.

## Limits

Coarse reference topology can fragment or merge supports on other critic geometries. Current fixed cost fixtures recover eight supports, but this does not establish all-domain detector or mass validity. The rare-hole intervention erases each corrupted row's original identity; correct aggregate mass cannot establish individual semantic recovery. Exact-copy fixture results exclude jitter and training. CUDA quality, timing and memory of the composed proposal remain pending.
