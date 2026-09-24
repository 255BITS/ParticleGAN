# Rp pairing erratum

The frozen `continuation/declaration.json.gz` says “antithetic 2N by 2N paired Rp.” The training source actually calls `GANLoss(mode='rp')` with equal-shaped 2N logit vectors. Its subtraction is elementwise, so it averages **2N index-matched pairs**, not every crossed fake/real pair. The original narrative report made the same wording error and has been corrected.

[`pairing-audit.json`](pairing-audit.json) is a later read-only check on the archived PR84 cold snapshot. For a native128 bank, antithetic expansion gives 256 observed real and 256 observed fake points; D and G logits each have shape `(256,)`. Actual D and G scalars equal the 256-pair formulas bitwise and differ from explicitly computed 256 × 256 crossed alternatives. No frozen source, declaration, fit, training update, or original result was changed. This erratum and its receipt are supplemental to the original 14-file manifest.
