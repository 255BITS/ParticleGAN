# Forge convergence requirements for 0.10

The owner requests that accurate acquisition, settling, continued-training
accuracy and recovery count toward future qualification even while unsolved.
The current [technique inventory](../technique-inventory.md) is the single
generated leaderboard. Its requirements expand from 81 to 106 view instances,
covering 27 distinct tests; repeated assignments across views remain explicit.
The current selected BISPF evidence projects to 81/106, not a full pass.

| Requirement | Placement | Existing pass condition |
| --- | --- | --- |
| `grid100`, `rotated100`, `staggered100` | Tier 2 in all six ordinary views | 97% precision, all modes, population/centering/shape bounds, five stable final checks and independent holdout |
| `gaussian1d_stability` | Tier 2 in `discriminator_stability` | Correct stationary distribution throughout continued learning, shifted-target re-acquisition and shifted hold |
| `grid100_14k`, `rotated100_14k`, `staggered100_14k` | Tier 3 in `clockfree_continuous` | Continue the actual passing 7k checkpoint to 14k with unchanged numerical quality gates |
| `target_shift_recovery` | Tier 3 in `adaptation` and `clockfree_continuous` | Acquire and hold, then recover by the deadline and remain correct; frozen control fails |
| `clockfree_audit` | Required prerequisite in `clockfree_continuous` | Existing gate used by all three native continuations |

Each changed view archives its exact prior bytes through `write_view_revision`:
adaptation 3→4, clockfree_continuous 4→5, discriminator_stability 11→12,
formulation_comparison 2→3, host_profile_transfer 5→6 and quality_coverage 3→4.
This is an explicit policy revision, not a claim that the old hold re-entry
conditions were met. Other held questions retain their status.

Task declarations, architectures, priors, sampling laws, budgets, numerical
thresholds, recipes, source-bound measurements and historical qualification
packets remain unchanged. A failed acquisition still blocks its checkpoint
continuation; a blocked continuation earns no pass. The current README's
particle-cloud diagnostic supplies no ordinary learned-MoG qualification credit.
Distribution fidelity is required; individual particle IDs need not stop moving.

This change launches no scientific training. The reports project the same
recorded outcomes onto the expanded requirements; frozen qualification retains
its actual original policy. Existing declared campaign reservations still govern
new work. Regenerate without rerunning unchanged scientific experiments:

```bash
python reports/forge/regenerate_technique_inventory.py --refresh-publication
python -m experiments.forge compile --summaries-only
python -m experiments.forge validate
```

No currently selected recipe passes the broadened ordinary requirement set.
The software default remains the selected optimized BISPF research preset,
with its failures visible. Future formulation comparisons must preserve one
global recipe, protocol seed 0, the public initializer, isolated checkpointed
streams and the actual task/sampling contracts.
