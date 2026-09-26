# Continuous qualification harness

The ring evaluator can run the pinned stress protocol and the pinned 30,000-update protocol on one uninterrupted host state. This lane did not train a candidate. K3P stays the selected base. RP1 was not retrained.

Harness gates: **10/10 PASS**. Declared 9,000-update and 30,000-update runs, and RP1 qualification, are **NOT_RUN**. The three-update CUDA smoke reports quality **FAIL** (0 modes, HQ 0) and `qualification: false`. That smoke checks the evaluator. It is not a recovery or toy result.

## What the evaluator does

Code: `benchmarks/toy100/continuous_qualification.py`  
sha256 `01c7306aa8f717fcc29047804a81c0bcd931eed81a6daf1d8f524d7a48ccd119`

Guide: `reports/toy100/continuous-round-3/qualification-harness/README.md`

The frozen ring host still owns every gradient and Adam step. The new module patches the checkpoint, the ring tensor, and the optimizer-step counter. It loads the protocol files only after their bytes match:

| Protocol | sha256 |
| --- | --- |
| `reports/toy100/continuous-round-3/stress-protocol.json` | `e5cb3e7cbb48943ab33fcf1cba0d3943f7f65ef40b5843d7d6734d91b76cd798` |
| `reports/toy100/continuous-round-3/long-term-stability-protocol.json` | `8c5f2e27c0e0f9b93b7a1f1a36ff09f1c5fb0d7bc398f6fc520d0a7e8d9dc9fe` |

Stress checks: stable windows **5 + 480 + 60**, deadlines **81 + 81**, **707** checks. The 30,000-update plan keeps those 707 prefix checks and adds stable **1,800 + 180** plus a third deadline **81**, **2,061** extension checks. Prefix step 6,400 and extension step 27,400 are scored apart: a miss in one leaves the other window intact. An incomplete witness forces status **ERROR** even when the windows match.

`absolute_offset` is copied onto the original ring. Update 6,000 still sees `[0, 0]`; update 6,001 sees `[1, 0]`. Update 7,800 still sees `[1, 0]`; the next update sees `[1, 1]` (delta `[0, 1]`). Update 27,001 sees `[0, 1]` (delta `[-1, 0]`). Adding the three protocol vectors would end at `[2, 2]`.

The episode length is recorded and is not given to the controller or the noise policy. Both stay on the canonical **1,200**-update horizon. A candidate `network_lr_horizon_cap` (1,600 on the K3P config) and `total_steps` (7,000 on that config) are left as declared fields. The smoke's 3-update episode still measured controller horizon 1,200 and noise horizon 1,200.

A candidate directory must have `provenance.json` hashes for `config.json`, `mechanism.py`, `response.py`, and `latent.py`, and the import must expose `latent.stats`, `response.prior_ids`, `response.previous`, and `mechanism._state`. A missing file, hash, or attribute stops before training.

## Witness

The capture runs after the named update and before the target copy. Identities are `optimizer/group/index` plus shape, dtype, device, and sha256. Python ids are omitted; controller ids that point at those objects become `{"kind":"reference","path":...}`.

`identity_sha256` is set only when every section is present: model, Adam moments, host EMA, controller `_state`, latent stats, response history and prior membership, CPU/CUDA/host-stream RNG, and the pre-shift ring hash. The bare host smoke is missing controller, latent stats, and response. Its `identity_sha256` is null and its first-shift `partial_sha256` is `e9c048d68f084329ee908387b4b1343d259e4959af4f5cc2614b58643bbe0af0`. The CPU witness test does produce a complete hash, `6d31c1f8eb60b38d865cd8eb86fa738a0276fe7261ee8e0a3bd4826da696a907`, and rejects the same tensors when a CUDA witness is required.

## Measurements

Ledger: `tests.jsonl` (13 rows). Self-test artifact: `smoke/self-test.json`. Host smoke: `smoke/host-result.json` (1.49s). Pytest: `tests/test_continuous_qualification.py`, 1 passed.

| Gate | Status | Evidence |
| --- | --- | --- |
| protocol_windows | PASS | 707 prefix checks, 2,061 extension checks |
| absolute_versus_incremental | PASS | stress final `[1, 1]`; incremental sum `[2, 2]`; update 27,001 sees `[0, 1]` |
| event_ordering | PASS | measure at the change, then shift |
| rng_preservation | PASS | forked global RNG and host stream stay put |
| update_accounting | PASS | stalled or skipped Adam steps are rejected |
| witness_identity | PASS | complete hash set; partial hash left null; CPU rejected for CUDA |
| fail_closed_provenance | PASS | missing manifest and hash mismatch refuse to install |
| schedule_horizon | PASS | 9,000 and 30,000 episode lengths share controller/noise horizon 1,200 |
| prefix_scoring | PASS | step 6,400 and step 27,400 fail separately; partial witness is ERROR |
| smoke_host_mechanics | PASS | 3 CUDA updates; offsets `[0,0]`, `[1,0]`, `[1,1]`; deltas `[1,0]` then `[0,1]`; D and G each 3/3 delegated updates; both post-shift steps moved parameters; RNG preserved |
| stress_protocol_9000 | NOT_RUN | bytes pinned, host run not started |
| long_term_30000 | NOT_RUN | bytes pinned, host run not started |
| rp1_qualification | NOT_RUN | rejected candidate was not retrained |

Smoke quality, kept visible: stable step 1 and deadline steps 2 and 3 are 0 modes, HQ 0, `pass_all` false. Optimizer roles `d` and `g` each advanced 1→2 and 2→3 with nonzero displacement. Device receipt: CUDA FP32, deterministic, TF32 off, one thread, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, RTX A6000 as `cuda:0`.

## Replay

```sh
cd /ml2/hypergan/gan-attempts/formulations-20260925T181158Z/prepare_continuous_qualification/20260925T181158Z-3771196/repo
/tmp/pr38-default-env/bin/python -m pytest tests/test_continuous_qualification.py -q
/tmp/pr38-default-env/bin/python -m benchmarks.toy100.continuous_qualification --self-test \
  --output /tmp/cq-self-test.json
CUDA_VISIBLE_DEVICES=GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/tmp/pr38-default-env/bin/python -u -m benchmarks.toy100.continuous_qualification --smoke-host \
  --config reports/toy100/gap-fill-20260925/sources/k3p/config.json \
  --output /tmp/cq-smoke.json
```

A later qualification run, after a candidate manifest checks out:

```sh
CUDA_VISIBLE_DEVICES=GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/tmp/pr38-default-env/bin/python -u -m benchmarks.toy100.continuous_qualification \
  --protocol stress --candidate CANDIDATE_DIR --output RUN_DIR
```

`--protocol long-term` is the same process through 30,000 updates. `--check-candidate CANDIDATE_DIR` stops at the manifest.

## Still open

The 9,000-update stress run and the 30,000-update continuation have an entry point and have not been executed. A real run still needs its own complete witness, active updates after every change, and separate prefix and extension scores. Noise and the ring controller remain on the 1,200-update horizon; that scheduled piece is unchanged. No formulation was promoted.
