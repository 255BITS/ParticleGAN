# Deterministic init search B (CPU screening)

CPU only (torch 2.14.0+cpu, 1 thread). These receipts are not the A6000 call.
Default init is unchanged. `--init <name>` selects a variant. `--seed-offset`
moves sample and noise seeds only.

Stay is the shift probe's continued-hold count. The driver's own status stays
FAIL on every variant, including CPU K3P, because shift recovery misses the
deadline. That is the same split the A6000 control recorded.

## CPU K3P baseline, repo seed

| gate | verdict | detail |
|---|---|---|
| ring | PASS | 8 modes, hq 0.9998, suffix 7, 18.2s |
| ring hold 1200 | PASS | 1200 checks, 39.1s |
| stay | 120/120 | driver status FAIL (shift recovery 0), 40.5s |
| unequal mass | PASS | hq 0.9763, suffix 16, min mass ratio 0.871, 42.2s |
| grid100 | FAIL | 100 modes, hq 0.918, coverage FAIL, accuracy FAIL, 188s |
| rotated100 | FAIL | 100 modes, hq 0.947, coverage FAIL, accuracy FAIL, 188s |
| img_stripes2 | PASS | 2 modes, hq 0.969, suffix 23, 8.3s |
| img_blobs4 | PASS | 4 modes, hq 0.938, suffix 11, 8.5s |

CPU K3P passes 6 of 8 priority gates. grid100 and rotated100 fail on CPU even
with the default init (all 100 modes are found; precision and covariance miss
the toy100 bounds).

## Ring + unequal screen, repo seed

Suffix must be at least 5. Both gates must pass to match the baseline.

| variant | ring | unequal | both |
|---|---|---|---|
| baseline | PASS 8 / 0.9998 / suf 7 | PASS suf 16 / mmr 0.871 | yes |
| had_tm_frob_bias | PASS 8 / 0.9998 / suf 7 | PASS suf 11 / mmr 0.967 | yes |
| house_frob | PASS 8 / 0.9990 / suf 6 | PASS suf 7 / mmr 0.901 | yes |
| dst_spec | PASS 8 / 0.9998 / suf 5 | PASS suf 14 / mmr 0.956 | yes |
| dst_frob | PASS 8 / 0.9998 / suf 5 | PASS suf 14 / mmr 0.842 | yes |
| had_tm_spec | PASS 8 / 1.000 / suf 7 | FAIL suf 0 / mmr 0.891 | no |
| dct_spec | PASS 8 / 1.000 / suf 7 | FAIL suf 4 / mmr 0.916 | no |
| dct_frob | PASS 8 / 0.923 / suf 6 | FAIL suf 0 / mmr 0.415 | no |
| house_spec | PASS 8 / 0.922 / suf 8 | FAIL suf 0 / mmr 0.879 | no |
| had_tm_frob | FAIL 8 / 0.995 / suf 2 | PASS suf 21 / mmr 0.937 | no |
| dct_frob_weyl | FAIL 8 / 0.911 / suf 4 | PASS suf 14 / mmr 0.818 | no |
| dct_frob_bias | FAIL 6 / 0.893 / suf 0 | FAIL suf 0 / mmr 0.671 | no |
| had_tm_frob_weyl | FAIL 7 / 0.663 / suf 0 | FAIL suf 0 / mmr 0.146 | no |

Zero-bias Thue-Morse Hadamard (`had_tm_frob`) repeats the earlier GPU signal:
all 8 ring modes, suffix only 2, unequal passes. A DCT bias at the Kaiming
RMS (`had_tm_frob_bias`) is what brings the short ring suffix back to 7.
Replacing the trig-whitened normal prior with a Weyl-normal prior loses the ring.

## Full gates, repo seed

| gate | baseline | had_tm_frob_bias | house_frob | dst_spec | dst_frob |
|---|---|---|---|---|---|
| ring | PASS suf 7 | PASS suf 7 | PASS suf 6 | PASS suf 5 | PASS suf 5 |
| hold 1200 | PASS 1200 | FAIL 549 | FAIL 1042 | FAIL 574 | FAIL 97 |
| stay /120 | 120 | 92 | 120 | 117 | 29 |
| unequal | PASS mmr 0.871 | PASS mmr 0.967 | PASS mmr 0.901 | PASS mmr 0.956 | PASS mmr 0.842 |
| grid100 | FAIL 100 / hq 0.918 | FAIL 100 / hq 0.950 | FAIL 13 / hq 0.158 | not run | not run |
| rotated100 | FAIL 100 / hq 0.947 | FAIL 100 / hq 0.971 | FAIL 5 / hq 0.093 | not run | not run |
| stripes | PASS hq 0.969 | PASS hq 1.000 | PASS hq 1.000 | PASS hq 0.938 | PASS hq 1.000 |
| blobs | PASS hq 0.938 | FAIL hq 0.719 | FAIL hq 0.844 | FAIL hq 0.688 | PASS hq 0.938 |

No structured init reproduced the CPU hold. `had_tm_frob_bias` is the only one
that matched the short ring exactly and still found 100/100 native modes, at a
higher hq than CPU K3P, without clearing coverage. `house_frob` matched stay
120/120 and got the longest failed hold (1042), then collapsed on both natives.

## Seed offsets (samples and noise only)

`had_tm_frob_bias` ring 4/8, unequal 6/8:

| offset | ring | unequal |
|---|---|---|
| 0 | PASS suf 7 | PASS suf 11 mmr 0.967 |
| 101 | PASS suf 8 | PASS suf 19 mmr 0.928 |
| 202 | FAIL 5 modes | FAIL suf 0 mmr 0.867 |
| 303 | PASS suf 11 | FAIL suf 0 mmr 0.932 |
| 404 | FAIL 7 modes | PASS suf 18 mmr 0.953 |
| 505 | PASS suf 9 | PASS suf 21 mmr 0.996 |
| 606 | FAIL 5 modes | PASS suf 5 mmr 0.947 |
| 707 | FAIL 0 modes | PASS suf 7 mmr 0.990 |

`house_frob` ring 5/8, unequal 2/8:

| offset | ring | unequal |
|---|---|---|
| 0 | PASS suf 6 | PASS suf 7 mmr 0.901 |
| 101 | PASS suf 7 | FAIL suf 0 |
| 202 | PASS suf 6 | FAIL suf 0 |
| 303 | PASS suf 6 | FAIL suf 4 |
| 404 | FAIL 7 modes | FAIL suf 0 |
| 505 | FAIL 0 modes | PASS suf 5 mmr 0.952 |
| 606 | FAIL 7 modes | FAIL suf 0 |
| 707 | PASS suf 7 | FAIL suf 0 |

## Determinism

Separate processes, different seed offsets, identical parameter hashes:

| variant | gate | offsets | init sha256 |
|---|---|---|---|
| had_tm_frob_bias | ring | 0 and 101 | `f4ffabe551a393c24afa17585f153cc2b7960f953663378201552beddc08bcfe` |
| had_tm_frob_bias | unequal | 0 and 707 | `449d9e7563282db66aeaa4dbd43713be80d59d867227a607c9417184833bdc00` |
| house_frob | ring | 0 and 707 | `403d3eb609da743dce9e6941a31e80700ff3a7ab9528d039a379595b1221cd54` |
| house_frob | stripes | 0 and 303 | `4b9f7e16f18905b93166e2da631f03b5e0205096e03dc817e23bfb86de843c93` |

Write-log sha256 (the values the hook stored) matches across those pairs too.
Default K3P ring init hashes differ: seed 0
`743729f19046a43b9a3c783e8e560b7bb18b17489d380684efd1fbd089aa9eda`, seed 101
`5067136f4c69e3ccd1ce0549cb5ed1eb9e5c49e66c0bc2b925e6d4d95165bd81`.
`tests/test_structured_init.py::test_two_processes_match_and_default_processes_do_not`
checks the same split in two subprocesses.

## Recommendation

Send `had_tm_frob_bias` to the A6000 first
(`--init had_tm_frob_bias`). It is the structured init that matched CPU K3P on
the short ring and beat unequal mass ratio, and its native runs still covered
100 modes. It already lost ring hold, stay, and blobs on CPU, so an A6000 pass
is not implied. `house_frob` is the second flag: better ring-seed rate (5/8)
and stay 120/120, worse unequal seeds (2/8), and CPU grid/rotated collapsed.
Leave the default init in place.
