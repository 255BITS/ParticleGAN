# First-moment repair proposal

Test exactly one global recipe delta: native Adam `betas=[0, .99]` to
`[.5, .99]`. The [candidate](../../../../configs/forge/ideas/r1r2-five-track-momentum-v1.json)
and [READY study](../../../../configs/forge/studies/r1r2-five-track-momentum-v1.json)
both use ID `r1r2-five-track-momentum-v1`. Constant LR .0085, paired input-gradient
L2 R1/R2 coefficient .1 every update, learned prior, role multipliers 1,
zero prior regularization, zero noise, zero EMA and every other recipe field
remain unchanged. `d_betas` and `prior_betas` remain null, so D/G/E/prior all
inherit the same beta1.

The exact measured parent is
`r1r2--169e9539b2dcfaab6b51e17f02907f28ea5b894f77155f7e688ab07a789de2b9`,
revision `35ebed78fe69b8923ead8335ea38bda209eb53ecc910b5e57ce5bac1c4b2f775`,
scientific source `25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`.
Its Ring16 attempt `27c75f3d768e4b789a488a3fdc5a2e90` ends at covariance error
2.386427, with 0/96 full passes. It passes pole transport and clock, but fails
word acquisition. The lower-rate child improves Ring16 to 1.409079 while losing
sustained pole transport. Preserve those identities in the
[concluded previous search](../../r1r2-tier1-search/README.md).

The [PyTorch 2.14 Adam update](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html)
uses `m_t = beta1*m_(t-1) + (1-beta1)*g_t` and native bias correction.
Beta1=.5 supplies short first-moment averaging without changing the second-moment
law or nominal learning rate. The hypothesis that this reduces alternating
noisy directions and local stretch is an inference. Saved Ring16 anisotropy
and late bursts motivate it; saved states do not establish that direction
variance causes the covariance failure. The
[Adam paper](https://arxiv.org/abs/1412.6980) supports the moment-estimation
mechanism, without guaranteeing this finite adversarial comparison.

Recall is negative as well as positive: prior
[BCAP DualNorm momentum .5/.9](../../configuration-search/bcap-dualnorm-pacing-v2-c-4054018e4d6380ed.json)
failed its complete comparison. The .5 arm is revision
`336582fcac63b2b4efefe0ea3f4e42814c4506e71dff8d3d4977243778ca00b6`,
source `f1755b1b5538901ffd4882f196bfd475030b06df16fd940c9b839eff86dc8226`;
it failed Gaussian, pole, Ring16 and words. Those results weaken confidence
and retain their different normalized update law, prior momentum ownership,
source and task contracts. Here the single point tests native Adam on the
current paired R1/R2 recipe. Momentum may lag a rotating game and worsen
transport or coverage; generator geometry may remain the dominant limitation.

Admission [preflight](preflight.json) is READY under the unchanged scientific
source and runtime. The [constructor-only audit](preflight.py) verifies actual
word G/E/prior/D groups and two-pole direct-coordinate/critic groups all use
(.5, .99). The direct-coordinate fixture's native Adam branch returns before
`direct_particle_betas` is used by the formulation branch, so this delta is
consumed there too. Constructors perform zero training updates or training/eval
sampling draws and restore caller CPU RNG. Fixed identity/zero fixtures remain
separate declared cohorts. Native optimizer moments and counters must be
checkpointed and tested by the candidate's own clock/restart probe.

Prediction: final Ring16 component covariance error <= .85; final error > .85
falsifies that prediction. Scientific success still requires every Ring16 bound
and five terminal passes, all six ordinary Tier 1 gates, and the separate clock
gate under this one recipe. Acquisition and endpoint retention remain distinct.
Hold protocol seed 0, deterministic public initialization, architectures,
target/data laws, actual seen target batches, initial prior distribution/capacity,
sampling, update allowances, evaluation cadence and gates fixed. Require actual
initial-tensor and target-stream hashes to match the measured parent cohort.

Reserve 2,520 seconds for the full six ordinary tasks plus clock, with one round
and no refinement, seed variation or higher-tier work. Root owns the new
user-authorized five-track registration, shared paid ceiling, frozen execution
commit, queue, actual-training GIF collection and readout. This proposal launches
no training and changes no engine or task bindings.
