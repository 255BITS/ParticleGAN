BASE_INIT = 0.5
REACH = 16.0
CONTROL_INIT = BASE_INIT * REACH
MEANS = [[-16.0, 8.0], [0.0, 16.0], [16.0, -2.0]]
w2 = 1.0
COVS = [[[w2, 0.0], [0.0, w2]] for _ in MEANS]
MASSES = [1.0/3.0] * 3
SPEC_NAME = "vector_boomerang_wide_r16"
FAMILY = "boomerang_wide"
REASON = (
  "Classic 3-mode boomerang: left wing (-16,8), apex (0,16), drooping right (16,-2). "
  "Curved sparse support under absolute RBF lengths; not kite/chevron/trapezoid/arch."
)
LIMITS = "One init; finite particles. Control drops absolute lengths entirely."
