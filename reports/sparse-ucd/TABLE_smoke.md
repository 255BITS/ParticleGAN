# smoke: 4 runs, 4 groups, 0 incomplete

| group | n | bar | bar_step | modes | hq | cond | sep | sym | joint | symKL | sp@1e-2 | sp@1e-3 | rec | zero | smear | core | w1 | ucdF | ppur |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| smoke_ucd | 1 | 0/1 | - | 0 | 0.000 | 0.329 | 0.75 | 0.372 | 0.329 | 0.06 | 0.084 | 0.009 | 0.231 | 0.00 | 0.107 | 1.68 | 0.306 | 0.83 | 0.57 |
| smoke_concat | 1 | 0/1 | - | 0 | 0.000 | 0.216 | 0.55 | 0.277 | 0.125 | 2.15 | 0.108 | 0.011 | 0.053 | 0.00 | 0.107 | 1.86 | 0.345 | - | 0.68 |
| smoke_proj | 1 | 0/1 | - | 0 | 0.000 | 0.141 | 0.54 | 0.140 | 0.123 | 1.62 | 0.102 | 0.011 | 0.071 | 0.00 | 0.106 | 1.65 | 0.339 | - | 0.87 |
| smoke_scalar | 1 | 0/1 | - | 0 | 0.000 | 0.242 | 0.51 | 0.129 | 0.001 | 8.42 | 0.130 | 0.013 | 0.054 | 0.00 | 0.092 | 1.49 | 0.355 | - | 0.59 |

bar = all seeds must hold: modes full & hq>=0.9 & cond>=0.95 & sym>=0.95 & sp@1e-2>=0.95 at the end; bar_step = median first crossing.
cond = P(nearest mode has requested class); sep = per-class cloud separation vs real; sym = symbol matches the mode the real part landed in;
joint = both; symKL = KL(true p(s|c) || emitted); sp@eps = P(|x|<eps on inactive dims); zero = exact 0.0 fraction; core = active-dim width ratio; ppur = particle class purity.
