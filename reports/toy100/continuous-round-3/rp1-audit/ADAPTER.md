Observation-only rate adapter is READY. It snapshots the pure policy multipliers before the original step_with_policy; the original function still computes and applies all rates and runs every optimizer update. Only policy_rate_action validates against that same-update snapshot. It rejects changed counts, changed policy arguments, bad optimizer groups and wrong applied rates. No phase_multipliers, model, controller, gradient, EMA, RNG, quality, architecture or budget changes.

Use the unchanged candidate driver with this wrapper instead of executing it directly:

```bash
/tmp/pr38-default-env/bin/python -u /ml2/hypergan/gan-attempts/formulations-20260925T173446Z/k3p_extragradient/20260925T173446Z-3726843/repo/reports/toy100/rp1-independent-audit/run_with_adapter.py /ABS/PATH/TO/UNCHANGED/RP1/probe.py [all original probe arguments unchanged]
/tmp/pr38-default-env/bin/python -u /ml2/hypergan/gan-attempts/formulations-20260925T173446Z/k3p_extragradient/20260925T173446Z-3726843/repo/reports/toy100/rp1-independent-audit/run_with_adapter.py /ABS/PATH/TO/UNCHANGED/RP1/native100.py [all original native arguments unchanged]
```

Use the caller's assigned GPU and original deterministic FP32 environment/one thread. The wrapper adds one observer import/install after the learner import in memory. It does not modify the driver or RP1 files. `observation-wrapper-provenance.json` records original driver, executed in-memory wrapper and adapter hashes; `observation-adapter-receipt.json` records validation counts and actual rates. Install hooks are applied to existing aliases and future imports. The copied RP1 candidate path is `/ml2/hypergan/gan-attempts/formulations-20260925T173446Z/k3p_extragradient/20260925T173446Z-3726843/repo/reports/toy100/rp1-independent-audit/candidate`; its exact mechanism hash is ed49869c6e06e1ac638cf04dba36b123aa7cb3b883650d7e16aa8843aaa22d14.

Seven CUDA FP32 regression properties passed (artifact `runs/observation-adapter-regression.json`): three identical full model/Adam/controller/RNG states with/without observer across genuine close transitions, plus wrong applied rate, wrong completed step, changed policy arguments rejected, and CUDA FP32 optimizer states. This is integration validation, not a toy gate. The original rate-action error remains recorded.

Adapter SHA256: 3b60a1d9e5ab679376ccfd9efe6559c80e5eef23e4813f776349c709577cdf81
Wrapper SHA256: f49720316080578dad000045c621276859f40b25261256c735cad71d42ba7d21
