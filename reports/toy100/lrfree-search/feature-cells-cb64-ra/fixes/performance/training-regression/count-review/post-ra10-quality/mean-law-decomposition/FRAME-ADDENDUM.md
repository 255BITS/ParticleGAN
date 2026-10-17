# Clean-group frame clarification

This is a premeasurement implementation clarification of the immutable selected
DESIGN.md; the closed source plan remains unchanged.

The paired clean/noisy increment uses g(C) both as the aggregation group and as
the index into the even-fit feature center, scalar scale and radial clipping
frame for BOTH C and Y. Thus it is mean[psi(Y,g(C))-psi(C,g(C))], not a difference
between psi(Y,g(Y)) and psi(C,g(C)). In this same frame the residual identity is
rY_gC = rC - delta_noise.

Natural g(Y) membership/frame, its original-style population energy and the C-to-Y
transition counts are reported separately. Raw EMA anchors A and sampled clean C
are unpaired; their group mean difference is an empirical distribution contrast.
The original native draw/save source establishes genuine C/Y row pairing.

No target, chart law, action, production source, quality gate or numerical result
was changed by this clarification. It was agreed with the independent reviewer
before helper sealing or numerical interpretation.
