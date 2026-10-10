# Named BCAP baseline decision

Select **direction blend** as the named `Recipe.from_preset("bcap")` research baseline. The ordinary, seed-0 matched comparison satisfies the frozen replacement rule: both arms pass all **6 Tier 1** gates; direction blend preserves every incumbent Tier 2 pass and adds **trajectory** and **residual student**, moving from **7/21 to 9/21**. All 27 required Tier 1/2 pairs have verified final consumed-state comparisons. [Results and limitations](README.md), [selection specification](spec.json), [saved audit](audit.json) and [independent audit](independent-audit.json) retain the underlying evidence.

The only preset delta is `constraint_geometry_mode="direction_blend"`. The other 13 explicit BCAP settings remain unchanged. `Recipe()` and all other 14 named presets retain their previous behavior. Generator, encoder and learned-prior updates use the direction wrapper; the critic keeps its normal optimizer. Standard trainers and supported integrated hosts bind their existing protected objectives automatically. Custom loops must follow the [public API contract](../../../docs/api.md).

An explicit causal control remains available:

```python
from particlegan import Recipe

incumbent = Recipe.from_preset("bcap", constraint_geometry_mode="none")
```

Historical Forge API-v1 cards retain their recorded resolver behavior. Old checkpoints restore their saved recipe and optimizer mode; selecting the new preset does not change a saved checkpoint's contract. Compatibility checks include the original failed full-suite run and its narrowly corrected follow-up in [software provenance](software-provenance.json), plus [post-selection checks](post-selection-software.json).

The current family selection changes only the `bcap-dualnorm` whole-row pin. [Selection receipt](selection-change.json) preserves the previous pin verbatim, binds the new pin to its measured source/runtime/recipe/task contracts, and hashes unchanged other-family and historical selections. Qualification is bound to execution commit `d378734f40b09ce223a389e8f54a9783ec6a0c75` and scientific digest `6a225fcdd6922cdad37c9c947e163fb6f164f6ad4390293a3d3091b8f741ce44`; publication and merge commits do not acquire new trained evidence.

This is the owner's **best-observed research default**, following the existing distinction between a named preset and calibrated default adoption. `default_adoption=false` remains explicit. Calibration is provisional; **12/21 Tier 2 gates still fail**, and both requested Tier 3 gates remain **BLOCKED** by the ordinary prerequisite veto. No additional seed cohort, scale-transfer or endurance claim is made. Historical diagnostic and search results keep their original source, seed, initialization and sampling contracts.
