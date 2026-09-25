# Applying one noisy recipe to the nine custom transfer hosts

The established 19-task result shares the 16 global fields in
`GLOBAL_RECIPE_FIELDS`: optimizer rates and betas, loss, penalty, prior
regularization, EMA, and schedule. Each host keeps its declared generator,
critic, data, initialization, batch, particle count, latent size, and update
budget. Six vector and four image hosts use `GANTrainer`; nine auxiliary hosts
run custom loops through `baseline.run_toy`. Before the legacy adapter work,
the toy100 compatibility runner applied output and discriminator input noise
to `GANTrainer` hosts only. A 22-task claim about the **same full noisy training
mechanism** requires explicit adapters for the custom loops.

| Host | Generated-data insertion | Discriminator input | Existing numerical readout |
| --- | --- | --- | --- |
| `two_pole` | Add fresh noise to learned `particles` when used as fake for D and G; keep particle L2 on base parameters | Ordinary one-argument critic | `mean_abs` and `nearest` inspect learned particle parameters and retain that declared scope; final D slope uses its scheduled σ, which is zero |
| `trajectory` | Noise `_Generator.forward(slow, z)` return (fast arc) | Perturb only `_Critic.forward`'s `fast` argument; `_FastView` then inherits it | `identity_mse` reads generator output |
| `residual_student` | Noise `ResidualHead.forward(slow, z)` return | Perturb only `_Critic.forward`'s `fast` argument; `_FastView` inherits it | `identity_mse` and landing statistics read head output |
| `mode_hold` | Standard `generator(latent)` can use `OutputNoise` | Standard `critic(points)` can use `InputNoise` | Diversity uses generated draws; exact particle-support diagnostic becomes a noisy draw, so its label must change |
| `unipolar` | Noise each row **after** expanding `student.delta(scale)` to `N_ROWS` | `ScaleCritic.score` receives normalized coordinates, including in the gradient penalty; add noise with data-space σ divided by `input_scale` | `score_residual` is a parameter readout of the residual at each scale; retain that scope |
| `ae_gan_hold` | Decoder serves both unconditional generation and reconstruction; wrap both paths consistently | Ordinary critic; its `.features` path needs the same input-noise policy if feature matching is enabled | `evaluate` uses decoder for both reconstruction MSE and generated hold |
| `cover_leftover` | Add noise to each returned cloud in `fake_batch()` | Ordinary `_FourierCritic(points)` | `score_geometry` reads residual parameters directly; retain that scope |
| `unused_token_hold` | Noise each row after expanding the concept-slot embedding | Ordinary `SlotCritic(points)`; `.features` also needs the policy | `score_student` reads the concept and unused slot parameters; retain that scope |
| `mid_scale_identity` | Noise each row after expanding `student.state(scale)` to `N_ROWS` | `ScaleCritic.score` normalized-coordinate path, including gradient penalty, needs σ/`input_scale` | `score_hold` reads one state per scale; retain that scope |

Use one optional `NoisePolicy` passed by the custom-host dispatcher into each
entry point. It should use σ=0.029 in the native data coordinates for every
generated row, and σ=0.5 at the first discriminator update, reduced linearly
to zero at half of **that host's frozen budget**. It should own independent
training and evaluation random streams. Evaluation at each of the 24 frozen
checkpoints must not consume training noise. The policy should record actual
output, D-input, gradient-penalty, and metric call counts per host; a report
must mark any unimplemented route as incomplete.

The four state-readout hosts deserve special care. Their historical metrics
describe learned residual or embedding parameters rather than emitted samples.
Those formulas and thresholds should keep their original parameter scope;
their training GAN fake batches still receive noise on every generated row.
Reports should say which metrics are generated-sample measurements and which
are parameter readouts. A fixed evaluation stream is needed wherever metrics
actually call the noisy generator. No adapter should infer roles from
`requires_grad`, tensor pointers, or perturbed logits; the explicit generated
data and discriminator-input paths above are auditable.
