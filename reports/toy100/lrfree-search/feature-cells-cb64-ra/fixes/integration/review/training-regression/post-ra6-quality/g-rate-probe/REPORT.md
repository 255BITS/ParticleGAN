# Recorded-gradient generator-rate probe

The root-specified1x versus1/4x virtual G comparison completed on frozen RA6 checkpoints1000 and2000. All original sources, checkpoint tensors, file hashes and global RNG are unchanged. No CUDA, new data, seeds, emissions, birth proposals or training loop. Four plain Adam calls update private clones only; functional candidate weights match those references bitwise, with maximum error zero.

## Measured learned motion

Both candidates use the same saved clean latent table, critic and one fixed current FIFO/CPU partition. Eligibility means the unchanged p>.05 and even-fitted inside boundary. Raw labels do not enter the measurements or candidate choice.

| Checkpoint | Virtual rate | Initially eligible | Eligible retained | Eligible retained in same cell | Changed cells / categories | Projected displacement RMS | Category TV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 1x | 556 | 538 | 519 | 33 /68 | .083786 | .049805 |
| 1000 | 1/4x | 556 | 553 | 548 | 9 /14 | .020993 | .012695 |
| 2000 | 1x | 406 | 392 | 376 | 35 /63 | .112175 | .049805 |
| 2000 | 1/4x | 406 | 403 | 400 | 11 /19 | .028029 | .015625 |

Normalized mean displacement in the previous cell's scale falls .150517 to .037705 at1000, and .208798 to .052192 at2000. Initially eligible losses fall18 to3 and14 to3, respectively. This is a concrete reduction in learned drift and current-cell churn on these fixed inputs.

Total eligible support after the virtual step is556 versus555 at1000 and409 versus408 at2000. The quarter step therefore does not produce more total support on these inputs. Improved retention and smaller displacement cannot be presented as a learned-quality result or accumulated1000-update improvement.

## What the checkpoint permits

The current generator optimizer is Adam/AMSGrad with beta1=0, beta2=.999, zero weight decay and no generator-specific direct-response modifier. Its saved exp_avg is the latest recorded G gradient; second moments, AMSGrad maxima and per-parameter step counters are available. We apply one virtual next step repeating that recorded gradient, with either saved LR or its quarter. Neither the true next gradient nor the previous historical step is reconstructed.

Current effective G rates are .0001328125 at1000 and .00006640625 at2000; virtual quarter rates are .000033203125 and .0000166015625. Each saved CUDA training stream contains16 bytes and is not installed on CPU. The unlogged next D update, latent index/noise draws and role batches are unavailable to this probe. D, prior, EMA, sigma and optimizer histories remain unchanged in the diagnostic.

## Smallest prospective configuration

The proposed root candidate uses existing fields: lr=.0010625, prior_lr_mult=8, d_lr_mult=4. Prior base .0085 and critic base .00425 are preserved exactly. G base and the trainer-owned learned log-sigma group both quarter. The learned-noise formula, floor, mode and live-noisy evaluation remain unchanged. This coupling is explicit; the G-only diagnostic does not simulate the full prospective G+sigma step.

No production field, source or config is edited by this probe. The result supports testing the narrow rate-only hypothesis before adding a generator anchor or transport mechanism. The root must qualify a fresh matched CUDA trajectory and unchanged full canonical gates; this receipt has no quality verdict. Independent API/source reviews qualify the production configuration separately.

Evidence: `result.json`, `probe.py`, `PROTOCOL.md`, `cpu-attempt1.log`. Source/input maps pin the frozen RA6 package, checkpoints, configuration and prior diagnostic. No extra levels, cutoff changes, evaluator changes, seed experiments or serving override were used.
