# Raw-coordinate skip findings

**No sustained rare-mode winner among eight declared critics.** The closest critic passes every final numerical bound but only the last **4/24** observations, so it remains FAIL under the unchanged final-five requirement. No full-six followup was performed in this sealed round.

The near-miss adds a zero-initialized learned raw-coordinate linear skip to the original axis-Fourier2 Softplus(beta5) D64×2: `score(x) = main(x) + Linear(2,1,bias=False)(x)`. This adds just two parameters (4,931 total). The main critic initializes before the skip and exactly matches the existing Softplus5 model at initialization. There is no extra loss or regularizer.

| Observation | Minimum eigenvalue ratio | Covariance error | HQ | Normalized sliced distance |
| ---: | ---: | ---: | ---: | ---: |
| 1000 | .038788 | .627426 | .988525 | .137639 |
| 1050 | .421208 | .580507 | .987549 | .123697 |
| 1100 | .443018 | .548234 | .987549 | .134331 |
| 1150 | .498103 | .676838 | .979492 | .144102 |
| 1200 | .490872 | .557634 | .980225 | .114200 |

Mass TV is .110664 and minimum mass ratio .831410 throughout these five checks. Only the step1,000 eigenvalue ratio fails. We retain the failure rather than extending the budget or relaxing stability. The final residual/Fourier/quadratic card also fails minimum eigenvalue ratio (.105426), despite passing all other final bounds.

The other candidates add fixed coordinate monomials (x,y,x²,xy,y²), separate smooth raw/quadratic MLP paths, or residual smooth layers. Quadratic features feed a learned critic; no covariance-supervision term or target moments are supplied. All added skip outputs initialize to zero with no initialization/seed sweep. There is no normalization, target-derived feature, or Fourier-amplitude sweep. Parameter counts and all failures appear in the [matrix](MATRIX.md).

Every episode retains the original G64×2/z4,256 particles,batch128,1,200 rare-task steps,Adam(0,.99),G/D/prior LRs .001/.0015/.01,Rp logistic,b_cap3/kappa1.25,prior regularization .05,no particle L2,cosine and1:1 updates. All behavioral thresholds and the complete24-check/final-five live rule remain unchanged. EMA is separate.

Exactly **8 GAN episodes**, **192 live observations**, **63.900755 seconds** summed recorded wall time. Static checks verify pointwise behavior, zero initial skip output, original axis-base state/output parity, correct monomials and active-cap backward.

Source base is `981ccbcd6e7e77a1f41f8aac3cc42d1fa1ceab45` plus archived research modules. Exact `skip_critic_research.py` SHA256: `ac27627990b53d3063bd62fde9823f8b4ccd88df9d48730f6695a0003fe5524e`. Every plan, original/effective spec, explicit research_discriminator declaration, full curve/action/update-count/runtime and failed attempt is preserved. Any later refinement is a separate experiment and cannot replace this round's outcomes.
