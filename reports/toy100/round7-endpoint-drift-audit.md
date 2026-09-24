# Fixed-target endpoint drift audit

This is a read-only comparison of saved **host** states at completed updates 1200 and 2400. The borrowed warm/hold runs inherit the same scheduled 1000-update prefix; their last 1400 updates have constant group rates D/G `0.00425`, prior `0.0085`. “Own” means a cold-acquired candidate state continued with those same constant rates. These saved host states do not contain the memory recorder; the separate complete learner envelope is necessary for an exact restart.

| Candidate and state origin | G parameter L2 displacement | Prior L2 displacement | Clean indexed support RMS move | D trainable L2 displacement | Centered D score RMS change on fixed grid |
| --- | ---: | ---: | ---: | ---: | ---: |
| Post-GAN rest, borrowed | 9.974 | 0.956 | 0.03935 | 4.042 | 0.249 |
| Pre-GAN start, borrowed | 0.110 | 0.0533 | 0.03935 | 4.090 | 0.249 |
| Pre-GAN start, own cold | 0.144 | 0.0828 | 0.05340 | 4.178 | 0.305 |
| Two-bank memory, borrowed | 0.00102 | 0.000729 | 0.001573 | 4.206 | 0.236 |

All optimizer state tensors in these endpoint pairs show step `1200` at the first state and step `2400` at the second, for both D and joint G/prior Adam. In the memory pair, D trainable parameter norm changes from `12.957` to `13.193` while its displacement is `4.206` (32.5% of the starting norm); G norm is `13.26122` to `13.26129` and prior norm `3.57245` to `3.57250`. D Adam first-moment L2 changes `0.1384→0.2413`; second-moment L2 `0.06319→0.04601`. Joint G/prior first-moment L2 changes `0.2924→0.2721`, second-moment L2 `0.1192→0.03776`. Its warm/hold quality is 200/200 and 1200/1200 passing checks, respectively, both with all eight modes. These endpoints show substantial D motion while the clean G support remains almost fixed. They do not measure an asymptotic trend or prove convergence or divergence.

For the function check, I loaded the saved underlying `SimpleMLPDiscriminator(2,96,3,3)` without input noise and evaluated both states on the same 896-point grid: 128 equally spaced angles at radii `0, 1, 2, 2.8, 3, 3.2, 4`. Each critic's grid scores were centered by its own mean before computing RMS difference, removing an arbitrary constant score offset. D parameter norms exclude the immutable Fourier-frequency buffer. The clean G movement uses the same indexed prior particles at each endpoint; permutation-invariant distribution distance may be smaller. Parameter motion can include compensating changes with little functional effect, as the post-GAN rest row illustrates.

The input files and byte hashes, in row order, are:

| Pair | Update 1200 SHA-256 | Update 2400 SHA-256 |
| --- | --- | --- |
| `round6/sample-anchor-rest-{warm,hold}-v2/candidate-final-state.pt` | `58ceab74956da28285efaeab0cb1de42969d9c35892d28a140b4d16fed0d4f6a` | `11dc64acd61f649a8248bc0ada3f7fc57df915d0d602a7a6a6cf36d3bbfe1df1` |
| `round6/sample-anchor-prestart-{warm,hold}-v2/candidate-final-state.pt` | `386b900291f523e8b6a788576c7dd85da094559fa35dd75d6259e02ddee8a841` | `fae94561922ede699f20c6ee6713126e8bb30e776a586fe98ed39137dac0a926` |
| `round6/sample-anchor-prestart-cold-v2/mode_hold-final-state.pt` → `round6/sample-anchor-prestart-own-hold/hold-final-state.pt` | `8a761531fe172147f923fdc2f68609b0e3c45fb8204c549b650f513b9baac163` | `aae43815fd731c9d7d0b1e4019c5457de9f319e53e201ceeedebfef15274d2ea` |
| `round7/sample-anchor-memory-{warm,hold}/candidate-final-state.pt` | `2f2b635fab347e9550498f34188ffe8eaa310293ac15807a3ee3ee469e480c45` | `d624a86f3070917cec4a551ef09363080faaff27d6d08fad4be79632f5ed8315` |

All paths in the final table are relative to the repository's local `artifacts/continuous-learning/` directory. This audit did not perform training or alter saved states.
