# Selected H: fixed starting point

Source and candidate: ../critic_signal_attempt/README.md and its batch-h archive.
The current training files reproduce all125 archived source hashes. The exact
checkpoint and all ten cold runs are retained inside this checkout.

Independent verification: ten original cold gates PASS with identical metrics;
three additional vector toys PASS. The remaining older audit finds six FAIL:
two_pole, unipolar, cover_leftover, mid_scale_identity, unused_token_hold,
ae_gan_hold. Native100 tests are SKIPPED until older failures are fixed.
Full evolving verification remains read-only at
/ml2/hypergan/gan-attempts/selected-h-verification/verification.md.

Two-pole spread .03264 versus required .30; unipolar terminal passing suffix1
versus5; mid-scale identity .806 versus .85; cover leftover retention .449/.442.
Unused-token hold .730/movement .540 versus .85; AE reconstruction MSE3.918
versus .05. The final two hosts explicitly test auxiliary-feature objectives
that the pure-GAN policy disables. Do not silently restore those losses and
claim GAN-only qualification.

Executable baseline screen (use fresh OUTPUT and LEDGER paths):

```bash
/tmp/pr38-default-env/bin/python -u reports/toy100/selected_h_remaining.py \
  --declaration reports/toy100/h_stability/selected-h.json \
  --output OUTPUT --ledger LEDGER --workers 1 \
  --tasks two_pole unipolar mid_scale_identity cover_leftover trajectory mode_hold
```

The selected_h_extension.py file supplies conditional callable plumbing and
explicit auxiliary-loss removal. It leaves H's math unchanged. The older
critic_signal_screen.py handles original ten tasks. Both archive source hashes.

The warm runner is already control-validated on all1200 observations and complete
final state. Its paths now resolve inside this worktree. Run:

```bash
bash reports/toy100/h_stability/run-probe.sh control OUTPUT 200
```

Do not rerun the three retained initial probes. All FAIL before200 updates:
critic_refresh2 at1254/HQ.871094; average2 at1228/HQ.705566; extra_adam at1203/
HQ.752197. Raw metrics, final states and work receipts: initial-probes/.
The H control first fails1255/HQ.780518. A later first failure alone is not a
complete stability pass. Code is stability_runner.py; direct fixes/experimental
copies must archive their own actual source and declaration.

Original generator/particles use solely logistic relativistic adversarial
gradients. Actual ring resources are12 particles,z4,batch128,width96, not the
native config's20,000 particles. Preserve frozen host resources and budgets.
Use fixed CPU/AVX2 and one thread per worker. No seeds or threshold changes.
