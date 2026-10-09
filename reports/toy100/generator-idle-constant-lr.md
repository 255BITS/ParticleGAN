# Generator idle under constant LR

**Not scored. This note has no result.**

The merged `batch_feature_zero` init passes 22/22 with the learning-rate anneal.
With constant LR it scored 13/22, priority 2/8, pre-shift STAY 81/120, and the
long hold did not converge (`reports/toy100/batch-feature-init/no-anneal-a6000.md`
on draft PR #279). Freezing the four unequal-mass critic batch-distance
coefficients was ruled out (they stayed near 0, max |w| 0.005). This note only
records the next opt-in rule and the command that would score it. Nothing here
was run on a GPU.

## Rule

When `generator_idle_se` is set, a generator step is idle if the batch mean of
the critic's paired gap `real_logits - fake_logits` lies within that many
unbiased standard errors of 0. The scoring value is **1**: the critic's own
batch estimate of real/fake separation is then unresolved from a perfect match,
which is that estimate's noise floor. There is no second coefficient and no
moving average. A constant nonzero gap has standard error 0, so it is not idle.
`None` (the default) does not read the gap. An idle step clears the generator
and prior gradients and still calls `optimizer.step()`, so learning-rate hooks
and the trainer phase advance, while parameters and Adam moments do not. The
critic still steps. `constant_lr` defaults off. When it is on, network and prior
learning-rate multipliers are 1. Floors, the horizon cap, widths, salts, seeds,
the cosine function, and the init are unchanged.

## Command

Do not commit a retuned config. The overlay adds only the two switches to the
winner JSON. Repo seeds stay (native `1234` in that file, transfer protocol
seed 0). Gates and task budgets stay the published ones. Pin each process to
one GPU and do not put more than **4** of these processes on one RTX A6000.
Two A6000s are the machine this is written for. The frozen control must finish
before the shift command that reads it.

```bash
python3 - << 'PY'
import json
from pathlib import Path
src = json.loads(Path("configs/toy100/constraints_simple_regularization.json").read_text())
src["constant_lr"] = True
src["generator_idle_se"] = 1.0
Path("/tmp/generator-idle-constant-lr.json").write_text(json.dumps(src, indent=2) + "\n")
PY

CUDA_VISIBLE_DEVICES=0 python3 -u -m benchmarks.toy_suite run \
  --device cuda \
  --init batch_feature_zero \
  --config /tmp/generator-idle-constant-lr.json \
  --output /tmp/generator-idle-suite

CUDA_VISIBLE_DEVICES=1 python3 -u -m benchmarks.toy100.continuous_probe \
  --device cuda --mode constant --init batch_feature_zero \
  --config /tmp/generator-idle-constant-lr.json \
  --steps 2400 --diagnostic-every 10 \
  --output /tmp/generator-idle-hold.json

CUDA_VISIBLE_DEVICES=1 python3 -u -m benchmarks.toy100.continuous_probe \
  --device cuda --mode constant --init batch_feature_zero \
  --config /tmp/generator-idle-constant-lr.json \
  --steps 3600 --shift-step 2400 --freeze-after-shift \
  --diagnostic-every 10 \
  --output /tmp/generator-idle-frozen.json

CUDA_VISIBLE_DEVICES=1 python3 -u -m benchmarks.toy100.continuous_probe \
  --device cuda --mode constant --init batch_feature_zero \
  --config /tmp/generator-idle-constant-lr.json \
  --steps 3600 --shift-step 2400 --diagnostic-every 10 \
  --frozen-control /tmp/generator-idle-frozen.json \
  --output /tmp/generator-idle-shift.json
```

`--mode constant` is the existing probe switch (floor 1, horizon cap removed).
`constant_lr` is what keeps the suite on unit multipliers while the winner
floors and cap remain in the file. Logs are line-delimited JSON; `tail -f` the
suite logs under the output directory and the probe stdout.

**Not scored. No metric, no pass count, and no failure count exists for this rule.**
