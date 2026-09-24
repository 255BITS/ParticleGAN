# Independent audit of the 44-check crossfit saved-state filter

The source-frozen saved-state filter reports **44/44 live quality checks**
passing, with eight modes at every check and minimum HQ `.936523`, `.937500`,
and `.987305` on the three branches. The exact original PR84 controls pass
only `9/12`, `14/16`, and `1/16`. This is a short continuation from saved
warm states, **not** a cold acquisition, 200-step warm, or sustained hold
result. The gate marks `shared_gate_eligible=false` and does not itself
promote the method.

| Saved branch | Original quality checks | Crossfit quality checks | Crossfit decisions |
| --- | ---: | ---: | --- |
| 1324–1335 | 9/12 | 12/12, min HQ .936523 | 7 joint fits, 5 rests |
| 1380–1395 | 14/16 | 16/16, min HQ .937500 | 5 joint fits, 11 rests |
| 1530–1545 | 1/16 | 16/16, min HQ .987305 | 7 joint fits, 9 rests |

The exact declaration and completed result are at
`artifacts/continuous-learning/round6/crossfit-saved44/` in the integration
worktree. Their raw SHA-256 hashes are respectively
`1d271e4d8c4c481b1d2fb5c32443607257e5377f85bc902f2a6ac92c54f9be4d`
and
`e45ed1697f3396cff3e4655b32333060d68ceeaf9073e5a012b4536f83fa7234`.
The two crossfit sources have SHA-256
`5d7eacbfd1937a9da999e62ab6156774d4c71f20fe305ff1f3860327278814af`
and
`9cdfa7dada3e1845d2b49d45bad7a3a5840e54d79201a2cdf8b3e8ea428e3514`;
all 20 declared archived source hashes match both their saved copies and the
current integration files at audit time.

The controller splits the **current native D real minibatch** into 64 rows
for discrete donor/sample reallocation and target construction, and 64 for
an additional C+Q acceptance comparison. Across 44 updates it selects 19
converged joint G+prior fits and 25 parameter rests; it selects no native
GAN proposal. All 44 attempted fits converge, including those later
rejected, with maximum final row-target error `2.981e−5` under the declared
scale-adjusted tolerance. Every selected non-rest whole G/prior move
strictly improves clean-support C+Q over the pre-G state **on each half**;
each rest reproduces the pre-G cost on both halves. The receipt records all
two-half costs, 132 native D minibatch identity checks, 44 correction RNG
checks, and 44 D/optimizer ownership checks.

The exact original controls replay saved accepted states, update records,
and observations. Within each branch the corrected and original variants
have matching final RNG hashes and noise receipts. Both retain nominal
D/G learning rates `.00425` and prior rate `.0085` at every update; Adam
moment step counts reach the same final update. The native game still has
three field evaluations and one Adam moment advance per outer update. The
44 corrected updates additionally require **235 joint-output Jacobians**
and **243 nonlinear fit trials**, including fits discarded by the 25 rests.
“Rest” means unchanged G/prior parameters for that update; D and Adam
moments still advance.

The term *crossfit* needs a narrow interpretation. The second half is not
passed directly to the discrete relocation or fixed-target objective, but
the preceding native D and G updates use the **full 128-row real batch**.
The joint numerical fit starts from those updated G/prior parameters and
therefore can depend on the second half through its initial point and
Jacobian. The native GAN comparison also uses that half for its gradient.
Consequently the second-half test is a within-batch acceptance condition,
not a statistically independent holdout. The frozen receipt's sentence
“second half has no role in output target construction or nonlinear
fitting” overstates the latter point; the inherited declaration also omits
the explicit 64/64 split. Neither issue invalidates the recorded two-half
costs, but both limit any generalization claim.

The acceptance objective uses clean G(prior) supports, whereas the host
relativistic GAN loss and quality evaluation include output noise. The
saved-state pass therefore does not prove that C+Q is a Lyapunov function
for the noisy coupled game or that future minibatches will preserve the
observed modes. A strict 200-step warm continuation, then uninterrupted
hold and cold acquisition remain separate gates.
