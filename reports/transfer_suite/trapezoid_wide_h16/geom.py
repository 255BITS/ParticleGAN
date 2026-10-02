BASE_INIT = 0.5
REACH = 16.0
CONTROL_INIT = BASE_INIT * REACH
MEANS = [[-8.0, 16.0], [8.0, 16.0], [-16.0, 0.0], [16.0, 0.0]]
w2 = 1.0
COVS = [[[w2, 0.0], [0.0, w2]] for _ in MEANS]
MASSES = [0.25] * 4
SPEC_NAME = "vector_trapezoid_wide_h16"
FAMILY = "trapezoid_wide"
REASON = (
  "Isosceles trapezoid: short top (±8,16), wide base (±16,0). Unequal parallel sides "
  "under absolute RBF lengths; not a kite, spike, square, diamond, or chevron."
)
LIMITS = "One init; finite particles. Control drops absolute lengths entirely."
