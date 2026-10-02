
BASE_INIT = 0.5
REACH = 16.0
CONTROL_INIT = BASE_INIT * REACH
MEANS = [[0.0, 16.0], [-10.0, 0.0], [10.0, 0.0], [0.0, -5.0]]
w2 = 1.0
COVS = [[[w2, 0.0], [0.0, w2]] for _ in MEANS]
MASSES = [0.25] * 4
SPEC_NAME = "vector_kite_wide_h16"
FAMILY = "kite_wide"
REASON = (
  "Asymmetric kite: apex (0,16), wings (±10,0), tail (0,-5). Far apex + near tail "
  "under absolute RBF lengths; not a regular polygon, diamond, or tall spike."
)
LIMITS = "One init; finite particles. Control drops absolute lengths entirely."

