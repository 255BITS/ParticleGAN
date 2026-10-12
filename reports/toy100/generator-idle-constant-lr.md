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

## Smoothed gap

`generator_idle_mode` selects `batch` (the default, unchanged se=1 and se=3 rule) or `ema`. The EMA keeps one decay, α = 2/11, the span-10 filter α = 2/(span+1). Its asymptotic effective number of batch-means is (2−α)/α = 10, so the smoothed gap is judged with that sample size and no second coefficient was searched. The threshold stays **se = 1**: idle when the absolute EMA of the batch-mean gap `real_logits - fake_logits` is within one standard error of that EMA. The first step stores the current batch, so it matches the batch rule; a non-finite batch or a batch shorter than 2 does not move the filter and is not idle.

Each run writes `idle_count`, `decisions`, and `idle_fraction` on the probe evidence, the native `summary.json`, and the transfer episode index. The completion line is `idle=<count>/<decisions>`.

```bash
python3 - << 'PY'
import json
from pathlib import Path
src = json.loads(Path("configs/toy100/constraints_simple_regularization.json").read_text())
src["constant_lr"] = True
src["generator_idle_se"] = 1.0
src["generator_idle_mode"] = "ema"
Path("/tmp/generator-idle-ema.json").write_text(json.dumps(src, indent=2) + "\n")
PY

CUDA_VISIBLE_DEVICES=0 python3 -u -m benchmarks.toy_suite run \
  --device cuda \
  --init batch_feature_zero \
  --config /tmp/generator-idle-ema.json \
  --output /tmp/generator-idle-ema-suite

CUDA_VISIBLE_DEVICES=1 python3 -u -m benchmarks.toy100.continuous_probe \
  --device cuda --mode constant --init batch_feature_zero \
  --config /tmp/generator-idle-ema.json \
  --steps 2400 --diagnostic-every 10 \
  --output /tmp/generator-idle-ema-hold.json

CUDA_VISIBLE_DEVICES=1 python3 -u -m benchmarks.toy100.continuous_probe \
  --device cuda --mode constant --init batch_feature_zero \
  --config /tmp/generator-idle-ema.json \
  --steps 3600 --shift-step 2400 --freeze-after-shift \
  --diagnostic-every 10 \
  --output /tmp/generator-idle-ema-frozen.json

CUDA_VISIBLE_DEVICES=1 python3 -u -m benchmarks.toy100.continuous_probe \
  --device cuda --mode constant --init batch_feature_zero \
  --config /tmp/generator-idle-ema.json \
  --steps 3600 --shift-step 2400 --diagnostic-every 10 \
  --frozen-control /tmp/generator-idle-ema-frozen.json \
  --output /tmp/generator-idle-ema-shift.json
```

Native seed `1234` stays in the winner file. Transfer seed stays 0. Gates and budgets stay the published ones. Pin one GPU and keep at most 4 of these processes on one RTX A6000. `tail -f` the suite logs and the probe stdout; `generator_idle` is one JSON line per probe and `idle=<count>/<decisions>` is on the native completion line.

## Notes from the ec00aa4 A6000 run

Replay of the grid100 affine receipt was rebuilding the wrong prior. Training with `--init batch_feature_zero` replaces the prior at Adam construction, and the receipt hashes that replacement. Replay rebuilt only the earlier `uniform_(-5, 5)` draw, and it did so without the recorded device policy, so `ParticlePrior.normal_()` consumed the CPU generator and `uniform_()` then started at a different CUDA offset. The hash and extrema checks were not loosened. `public19/protocol.json` is written only for `--with-default-control`. The command above does not pass that flag, so the missing public v3 control is expected.

`apply_device_policy` is process-global. On CUDA it sets the default device and routes device-less `torch.Generator()` onto that device, so every default-device draw moves onto the CUDA stream, not only the four hosts whose fixtures were pinned back to CPU. That is why overlap, anisotropic, stripes, blobs, bars, and unequal-mass moved while the CPU path of those four hosts stayed byte-identical to `9c9a220`.

