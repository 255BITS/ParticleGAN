# t22_r2 — R2 full-22 sweep: DQ'd at grid100 (STOP)

Lane R2: exact copy of b3_release2 cands/r2 (mechanism sha 59483b5e57c1bbbcef171f6261c62bf620bf943f3a5ab6d05ab543e91c0bbb4d; config/latent/response K3P-identical).
Verified copied hashes:
- checkpoint.py 0185ba142eb5bffa1e28a47690b3526cab9e3a673a9c0ba0a6cd984c0c9444d9
- config.json a1475108a82f67a93e0cdcd793b920b0cc2b1e1ccf31285974adc3b341b2fca2
- convergence_gate.py a29c8c21083195c2944a2b2509ec7d4b96e6fc1eb7c3fa6df70d51dea6fe1ce0
- hold.py 2ae5a9551739e6bad6190b76bcf252689793181786239238d0b08d2d1cc0c00e
- latent.py 197df6350f5295f7d396f7d3c821808be1d15168d6e5586a89ebfbd403586139
- mechanism.py 59483b5e57c1bbbcef171f6261c62bf620bf943f3a5ab6d05ab543e91c0bbb4d
- native100.py d550b6ed701273381450a3c66f6b264767550921e89ba9690ceafb3bc1c0e997
- probe.py e8653d7e450268310d9c7fc529262279f202a2cb10b3bf1470efc7763d4452bc
- response.py 7e71d60a343f9f47e1c16600279364f0482863ce116c00f4657355638615987d
- shift.py f049e86eac4e1b65212ea2d40c2eee8c9f6a63082a666f6d3f31010f336375a8
- shift_frozen.py 9970dd0195d65aeccd2ffeaf874f2a5ed0df2e06f0f934192201258e41f63fc6

R2 mechanism: LR-decoupled fixed s=0.5 + Adam moment-surprise hysteretic anchor gate (release ratio>3.0, re-anchor<1.75); config/latent/response byte-identical to K3P. Cited prior shift/hold (not re-run): 72/81, 114/120, stable 2890.
Env: 1 worker, GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69 as cuda:0, frozen drivers/fixtures, declared seeds, full 7000-step natives.
Per-toy seconds (this round): mode_hold 100.7, vector_unequal_mass 152.8, vector_unequal_width 109.5, img_stripes2 48.6, ae_gan_hold 18.3, cover_leftover 66.5, img_bars4 45.3, img_blobs4 47.2, img_intensity2 44.4, mid_scale_identity 91.3, residual_student 23.1, trajectory 24.3, two_pole 4.7, unipolar 28.8, unused_token_hold 9.3, vector_two_broad 106.9, vector_anisotropic 147.5, vector_overlap 111.7, vector_spiral 128.9, grid100 467.8 (FAIL). Transfer 19/19 PASS; natives 0/1 (grid100 FAIL → STOP, rotated/staggered NOT_RUN).

## Verdict: R2 DQ'd — grid100 FAIL, STOP per lane rule

grid100 (seed 1234, 7000 steps): 100/100 modes acquired (first full coverage step 750) but 0/5 terminal coverage checks pass; accuracy FAIL (center_rms_sigma 0.241 vs 0.20 limit, cov_trace_bias 0.127 vs 0.10, radial_ks 0.063 vs 0.04; final live HQ 0.9688 vs 0.97 precision floor). Artifact: out/r2-native-grid100/result.json. Replay: cands/r2/native100.py --task grid100 (log logs/r2-native-grid100.log).
Ranking inputs: 19/22 toys PASS, grid100 FAIL, recovery/hold not re-run (cited 72/81, 114/120). Compute total: toy seconds above (sum ≈ 1777.6s incl. grid100) + cited shift/hold runs' tests.jsonl/result.json seconds (not re-measured here).
Negative result narrows mechanism choice: fixed s=0.5 + surprise-gated anchor preserves transfer toys but degrades native precision vs K3P's LR-clocked handover; next mechanism should keep native center accuracy (reg_arm/coeff path) rather than more release-signal tuning alone.
