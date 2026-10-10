# RA11 grid saved-state validity

Artifact evidence VALID; original quality PASS. All 34 observation metadata rows and original native artifact checks pass. The saved final trainer5/backend10 state, RNG placement, fourth phase, bounded lineage and cumulative own evidence reset totals agree.

- Only final-state.pt is saved; earlier 34 observations provide metadata, not historical tensor endpoints.
- Original native JSON replaces lists longer than 32 with length strings. Historical compressed IDs are unavailable; only rendered lengths and scalar balances are validated.
- Own row reset/moment/history/link predicates apply only at an exact saved reaction boundary.
- Isolation row IDs and historical unsaved population participants are unavailable; cumulative reset totals are checked.
- No new scorer/sample/trajectory or historical/cross-device replay. Audit validity is separate from original quality PASS/FAIL.
