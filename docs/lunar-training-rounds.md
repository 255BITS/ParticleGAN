# Lunar training rounds

These are development measurements on fixed train seeds `24000:24096` and
validation seeds `34000:34020`. No held-out test seeds were used. All flights
ran in the declared bidirectional Box2D variant. A successful landing means
the simulator settled the lander on its legs; low imitation loss alone does
not count as success.

| Development round | Learned slow validation | Learned fast validation | Paired speedup | Finding |
| --- | ---: | ---: | ---: | --- |
| Initial pilot: 2,500 action-cloning updates, 400 paired RpGAN updates, no engine deadband | 17/20; 205.5 successful-flight steps | 0/20; all timeouts | unavailable | The neural output put small positive values near the expert's exact zero main command. The stock upward engine fires at a minimum power for any positive command. |
| Corrected pilot: final expert code, 8,000 action-cloning updates, 400 paired RpGAN updates, `0.12` main deadband | 20/20; 212.15 steps | 20/20; 183.65 steps | 1.155× | The controller emits an exact zero main command in the deadband. The same projected action enters critic and frozen world-model training through a straight-through gradient path. |
| Pair-filtered check: same corrected slow checkpoint, fast training restricted to 91 successful, faster same-world expert pairs | 20/20; 212.15 steps | 20/20; 182.75 steps | 1.161× | This matches the full pipeline's fast-data extraction rule; 17,053 fast transitions remained. |

The first pilot used an earlier slow expert; its results diagnose the engine
activation problem but are not a matched ablation against the corrected pilot.
Applying the `0.12` deadband to those **same first-pilot checkpoints**, without
retraining, changed slow validation from 17/20 to 20/20 and fast validation
from 0/20 to 20/20, at 211.85 and 184.65 successful-flight steps. This is
direct evidence for the action-semantics fix. A separate direct fast-policy
cloning check on the final expert data landed 11/20 after 2,500 cloning updates
and 20/20 after 8,000. Longer initialization is therefore also necessary for
the chosen compact network.

The corrected pilot used 96 fixed train worlds. The slow expert landed 92/96
and the fast expert 94/96; only successful expert episodes supplied policy
targets. The world model trained on both behaviors from the first 76 train
worlds and used the remaining 20 train worlds for episode-disjoint validation.
Its validation next-state MSE was `0.002106` after 1,800 updates on 31,671
transitions. That split is independent of world-model fitting, although the
policy's expert dataset contains some of the same episodes. The validation
flights above use entirely separate worlds.

Both corrected controllers used a 128-wide neural policy, 8,000 supervised
initialization updates followed by 400 paired conditional RpGAN updates with
the public `get_recipe().make_loss()` and `make_gradient_penalty()` APIs.
Declared recipe overrides were `reg_coeff=1` and `reg_every=4`; the loss
remained paired relativistic logistic. The generator used Adam at `3e-4`, the
critic Adam at `2e-4`, both with betas `(0.5, 0.99)`. The adversarial weight
was `1` and the frozen learned world's successor loss weight was `2.5`.
There was no direct action-cloning loss during the adversarial phase. The
learned world supplied a differentiable next-state target for policy actions;
the action discriminator scored `(state, action)` pairs. These are explicit
task-specific choices, not the repository's unmodified default optimizer or
regularization settings.

The adversarial phase ran and changed the policy, but these pilots do not show
that RpGAN alone improves landings over the cloning initialization. In the
corrected fast pilot, training action MSE rose from `0.000979` before RpGAN to
`0.006170` afterward; normalized world successor MSE fell from `0.41299` to
`0.41038`. Simulator flights establish the final controller's landing and
speed result. The full pipeline uses a stricter extraction of same-world pairs
where both experts landed and the fast flight finished sooner, and evaluates
its selected checkpoint separately.

Source hashes at full-run start (the checkpoint format was revised after the
pilot, without changing its training equations or action projection): `lib/lunar_training.py`
`d558f6a1b49bce552beb3828851874c0929eb86ee9005d6c79c4da16a85e4721`,
`lib/lunar_flight.py`
`dd5370d34554b8304df6c7476e2ea3987d9f56d06b2e06292efce6003fdb12f3`.

## Full single-command release run

The frozen [release report](../reports/lunar_fast/report.json)
records 2,500 world-model updates, 8,000 cloning and 400 RpGAN updates for
each policy round, 96 train worlds, 20 validation worlds, and 30 previously
untouched test worlds (`84000:84030`). The world model's episode-disjoint
validation next-state MSE was `0.002237`. Three fast rounds were trained;
round 1 won the predeclared validation selection rule, and `fast.pt` is
byte-identical to `fast_round_1.pt`. The run took 60.5 seconds on CPU with two
threads.

| Controller | Validation landings | Test landings | Test successful-flight steps |
| --- | ---: | ---: | ---: |
| Learned slow | 19/20 | 28/30 | 220.39 |
| Learned fast | 20/20 | 29/30 | 188.83 |

Both controllers landed on 27 identical test worlds. On that matched set the
fast policy finished sooner on all 27, saving a median of 30 simulator steps
and achieving a 1.172× paired speedup. The fast policy crashed on one other
test world; the slow policy crashed on one and left the bounds on another.
Thus the result supports the declared success and speed gate, not universal
landing success. Checkpoints store the learned weights, explicit `0.12`
main-command deadband, optimizer settings, public RpGAN recipe values, and
update counts. The source hashes above match the release configuration's
recorded hashes.
