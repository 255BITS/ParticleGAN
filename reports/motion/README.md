# Particle DDGAN motion completion

[Round-one results](round1/READOUT.md) ·
[Interactive comparison](round1/confirm_10k/index.html) ·
[Hybrid GIF](round1/confirm_10k/hybrid/futures.gif) ·
[Plan and reproduction](PLAN.md) ·
[Default config](../../configs/motion/default.yaml)

Observe8 frames of HumanAct12 skeleton motion, generate the next16 frames in
four DDGAN calls. Same learned particles, Gaussian step noise, joint UCD,
Rp logistic, bcap/VICReg and constant learning rate as the trajectory experiment.

```sh
.venv/bin/python experiments/prepare_motion.py
.venv/bin/python experiments/train_motion.py
```

No arguments selects the10k hybrid starter. Both hybrid and MLP comparison
configs are preserved. The experiment is functional but still has prediction,
bone consistency and temporal jitter limitations; see the readout.
