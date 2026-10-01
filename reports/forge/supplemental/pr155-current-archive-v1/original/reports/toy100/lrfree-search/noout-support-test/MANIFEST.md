# Manifest of the packages behind this archive

The packages are not committed (they are copies of the PR checkout's `particlegan/` plus the changes of the lrfree search; the neighbouring differences are in `patches/`). They live in `/ml2/hypergan/gan-attempts/noout-20260928/pkg-*` (base of the chain: `/ml2/hypergan/gan-attempts/seqtest-20260928/pkg-seqC`). `pkg-hash` = sha256 over the sorted `particlegan/*.py` file names and contents.

| package | role | pkg-hash | birth_death.py sha256 |
|---|---|---|---|
| `pkg-E14` | previous candidate E14s (row-evidence gate, anchored release, critic-space birth-death, served average) | `3506765fa3c99b52` | `3cb776571a182066` |
| `pkg-E15` | E14 + support test, uniform parents (0/3) | `8b2e64ff771ac387` | `df3d9b11f8b247f7` |
| `pkg-E16` | E15 with the k nearest unflagged rows as parents | `431dff4a2c8775f3` | `e501ba8f388972a4` |
| `pkg-E17` | E16 with the distance ball as parents (reviewed by two independent reviewers) | `a1ac807bfe9aa7f1` | `e68a0571db69df78` |
| `pkg-E18` | E17 + `birth_death_feature_scale: "std"` | `f1c43ebaf428f957` | `0b4731657510ed24` |
| `pkg-E19` | E18 + rank cap k^2 on parents + chunked stale-site check (the evidence base of E22) | `b0741e772e1ed4a1` | `513aa30517ecd8f9` |
| `pkg-E20` | E19 with p-weighted parents, persistence of two evaluations, duplicate guard (variant) | `784720cdd36647a0` | `b8f2a5858c264f0e` |
| `pkg-E21` | E20 + distance limit on the parents (variant) | `0480f02f7316ff95` | `0cc47ae2377af5b6` |
| `pkg-E22` | **E19 + duplicate guard only: the candidate** | `bc1c43af022370bf` | `447f67de1ed80eec` |

Files of the candidate `pkg-E22/particlegan/` (sha256):

- `__init__.py` 93cf2f09d6e73716405e0ef55dec1a47f3b03c04f3fcaf0ec10ba5d169da4aa7
- `autoencoder.py` f3fbb9184e9f44112e15eccf0c0246dfba5ed490d3c7b81b0f837a5335916402
- `batch_feature_init.py` 7178f3f9653a5ecf22633406f58f262ea2569ddf9c7e090e2ddbe1cd5aefc8c2
- `birth_death.py` 447f67de1ed80eec0e260c88808cd40e9f682bf55e1ef6c39e6dc513594c8b72
- `conditioning.py` 8734fe338868ace4b5851a8fdb1f071e46d47396f66a3765e6b1ef37f0481486
- `continuous.py` 26c71f54cab370cd5bc6b2faa9e860cf01600db780dddfdda6fcaf6081254388
- `det_init_a.py` c22542a3ac83fe9fe3c7a807442028ccd1dd2409a4428c6757beb3f623d81d4b
- `det_init_d.py` 6b2333382e03f7d58956e75dae700b2f6ee48cdb57f61bc3ca1b8ab38e5efaa4
- `det_init_f.py` fb6b0d8d1cc9f61dd0c5b39f6ed606152c0239be944aab47a3de64e83668549c
- `deterministic_init.py` c7fb045cc137187012cdd526b3c62dea068447b1d1a1f07ab06e01fe5c73564f
- `diffusion.py` ec4acf8d6e15c30c5dfbb6fc7b93f10ee1c778214493223c50e08eb2e5047683
- `discriminators.py` 0e2efb125ffb314577612ab7a2eba66b0a1a2d28ad25f403f42c18b6f6ee333f
- `family_e_init.py` 35097639e944f3f97fa07bf5ff23267346059f32e8dfbd615b8f9a802c9a6936
- `gan_loss.py` 1c1019dfe71c583e32a05df0d2f794f9fff6d9ee1ea0332f57f6379ae70cf6b7
- `grad_regularizers.py` a02ec4d50b2782b551927125979dd89622d2b728e4a1aa16312401a058d703c2
- `init_registry.py` 2cc0189f229e4ebaf289485152b3fdcb6eb3c9979f0a4db15ddbaf3fb6d3179c
- `initialization.py` b769638eecc119b22138cd52fa91c56bb0ad3515aab3a567cfef8708dc81a366
- `k3p.py` fcff8a4f7a37ac630d8662ed34d3f9cbf89f2c59a33203cd5f963609842b8fb1
- `ka2.py` a972d7d09ffc8940c962124adea2769d8e7e0b6f02522850fe29fc1a150127a8
- `particle_prior.py` 17e39404cefca5963c82d9981f8582ea6c650c9aae66b51ce11ecf05896bb9a9
- `qr_bz_pq_init.py` 3c7f4677d43f19e769453835b1619a17031cef5a5d3f72e9ecb731f310518ce3
- `recipes.py` 7f4ed2a55b698c8aadf34a6d51ce1683ddafd35307dbce2e3b1c68e83583ef92
- `row_evidence.py` 1e930343a714d7825d517f157683b8dda7d59048781637ba7475efbd0eab9606
- `structured_init.py` b098a033e32e5f535041b713e8a9a4a00992af607f00e45ecf364142fd4c7379
- `training.py` 6ebe9a53d6ed953a2d538eae5f543b10b5c11edd6b33a2570450ee8035d223af
- `vicreg_loss.py` ab1c4dc266dec2c35337f449917240eb38afede7590d2154b63f6f471dc45a36

Configs: `overrides/overrides-E22.json` = `overrides-E13-scaled.json` (the E14s config) + `birth_death_isolation: true` + `birth_death_feature_scale: "std"`. Launch: `analysis/launch.sh pkg-E22 overrides-E22.json <label> <task> <gpu> [native_steps]` (edit `S` in the script; the harness is `/ml2/hypergan/lrfree-20260926/harness`, frozen and untouched; the two scratch copies used for probes differ from it by `harness-probes/*.patch`).

Reproduce the tables: `python3 analysis/build_results00.py` reads `analysis/results00.template.md`, `analysis/probes00.txt` and the run directories (`runs/<label>-<task>`); the receipts of the runs quoted in RESULTS.md section 00 are in `receipts/`.

Determinism: the candidate `pkg-E22` reproduces `pkg-E19` bit for bit while the duplicate guard does not fire (CPU digest test `tests/test_E22.py`; on GPU the three native E22 runs are identical to E19a to the printed digits and at every one of 26-28 live checks per task).
