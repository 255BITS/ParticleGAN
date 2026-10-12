# Optimizer repair proposal: native AMSGrad

Test exactly one recipe delta: `Recipe.amsgrad=false -> true` from the
[constant beta2=.99/gamma=.1/LR=.0085 parent](../../../../configs/forge/configurations/r1r2--169e9539b2dcfaab6b51e17f02907f28ea5b894f77155f7e688ab07a789de2b9.json).
The [candidate](../../../../configs/forge/ideas/r1r2-amsgrad-repair-v1.json) and
[ready study](../../../../configs/forge/studies/r1r2-five-track-optimizer-v1.json)
use native PyTorch Adam with AMSGrad in every applicable player group.
Both input-gradient L2 R1/R2 penalties remain active every update at coefficient
0.1; beta1=0, beta2=.99 and global LR=.0085 remain constant. Learned prior
updates, prior multiplier 1, zero prior regularization, zero additive noise and
zero EMA remain unchanged.

## Mechanism and uncertainty

The previous search separates two failures: all constant policies satisfy the
clock probe but miss Ring16 covariance, while reducing LR also loses sustained
pole movement. Native AMSGrad retains the coordinatewise maximum of past
uncorrected second moments before the existing bias correction. This changes
an adaptive denominator rather than cutting the global LR. The uncorrected
maximum is nondecreasing; the full bias-corrected step need not be. The
[PyTorch 2.14 Adam law](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html)
defines that mechanism. [Reddi, Kale and Kumar](https://research.google/pubs/on-the-convergence-of-adam-and-beyond/)
motivate long memory using Adam counterexamples; their optimization argument
does not guarantee this nonconvex adversarial game will pass its finite gates.

The causal hypothesis is that a declining Adam denominator contributes to late
local stretch or excursions, so retained moment history may reduce covariance
while preserving the initial nominal movement rate. This is an inference,
not a diagnosis established by the saved states. The best earlier low-rate
Ring16 state has tangent variance 2.08–3.88 times target despite all mapped
prior means being within 3 sigma. AMSGrad does not directly penalize a generator
Jacobian, so unchanged anisotropy would favor geometry over denominator decay.

## Recalled evidence and choice

The parent is revision
`35ebed78fe69b8923ead8335ea38bda209eb53ecc910b5e57ce5bac1c4b2f775`
under scientific source
`25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`.
Its Ring16 attempt `27c75f3d768e4b789a488a3fdc5a2e90` has endpoint
covariance error 2.386427 and 0/96 full passes; its pole gate passes and word
acquisition fails. The half-LR child improves Ring16 to 1.409079 but its pole
suffix is 1/5. These preserve their original identities in the
[previous readout](../../r1r2-tier1-search/README.md).

Historical [DV12](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/evidence/amsgrad-dv12.md)
AMSGrad reduced quick-gate counts 12/13 to 10/13 and did not repair allocation.
[RP15](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/evidence/amsgrad-rp15.md)
also lost quick gates and mode capture while improving historical ring/stationary
retention. The RP15 constant-rate comparison still failed multiple gates.
Those are negative priors for a complete repair, preserved as narrative context
rather than transferred qualification: they used secant/tangent/controller or
K3P/KA2 interventions, older initialization/source/task laws and different gates.
Their original source commit is `0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b`;
DV12 evidence blob `425db148298188056bdefbaf2978b0c69b309b2e`, SHA256
`5eea15e28e9ed165f4d0a498f156eb89a6784974c05755ee9cdaf3e94fd2b945`;
RP15 blob `247135e6a8c8ca62e86ee7156dc4c59cececbc34`, SHA256
`076cd9975839a0158750ad2c1bc059bd2c3e2d30c43ed9bdea6407f4f55efc83`.

This single point isolates a supported optimizer-history change on the current
paired R1/R2 law. An early gradient maximum may suppress later transport or
mode capture. A global beta1 change would instead add momentum to every player;
a normalized optimizer would replace more update-law details. Neither is added
to this study.

## Ownership and numerical decision

`Recipe.make_optimizers` supplies AMSGrad to G/E, prior and critic; scalar/vector
and word public trainers consume these optimizers. Behavioral fixtures route
G/E, learned priors and direct generated coordinates through
`make_generator_optimizer`, and D through `make_critic_optimizer`, so the same
flag reaches every group. Fixed identity/zero behavioral fixtures retain their
explicit separate cohorts. Native `max_exp_avg_sq` belongs in saved optimizer
state and must survive the candidate's clock/restart probe. No new optimizer,
source instrumentation or task binding is needed.

The registered numerical prediction is final Ring16 component covariance
error <= 0.85; final error > 0.85 falsifies it. Full scientific success requires
all unchanged Ring16 thresholds and five terminal passes, all six ordinary
Tier 1 gates and the separate clock gate from this one global candidate. Confirmed
Gaussian/word acquisition remains distinct from terminal retention. Compare
actual initial-tensor and consumed-target-batch hashes against the exact parent
cohort; preserve all named RNG checkpoints and report effective optimizer flags.

The study reserves 2,520 seconds for six ordinary tasks plus clock, has one round,
and stops for a readout after all runnable current-tier peers. No seed-only run,
extra update allowance, task-specific recipe or higher-tier work is included.
Root coordinates the shared registration, frozen commit, queue, actual-training
GIF collection and final report; this proposal launches no training.
