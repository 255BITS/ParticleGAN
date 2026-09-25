# Active research base: dimension_rms_hybrid

Use [the complete selected formulation](../dimension-rms-base/README.md), including
its config, critic mechanism and probe. It passes six GPU regression toys and
fails two_pole; it is not full-22 or stability qualified. Gate on two_pole first.

[Current declaration](../current-research-base.json) · [Search brief](SEARCH.md)

The original scheduled CPU winner and its native16/22 GPU control are historical
references. Old H/epsilon/shared-column continuous-rate bases are also historical.
The active base is explicitly selected by the user; do not revert to older
starting configs or credit their passes to this formulation.
