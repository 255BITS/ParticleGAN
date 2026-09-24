# Fast sustained mode-hold calibration

The frozen 1,200-update `mode_hold` host is a faithful, roughly eight-second
screen for constant-rate instability. It uses the production ring data,
12-particle prior, MLP generator and Fourier critic, adversarial/regularizer
losses, alternating Adam updates, output noise, and 4,096-sample live metric.
The probe adds scoped observation and optimizer-rate receipts to that host; it
does not replace its training body. The 1,200-update observed metrics match the
unmodified host numerically at every one of its 24 checkpoints (wall-clock
seconds excluded).

The default source is the simpler recipe that independently passed 22/22
common tests, `constraints_simple_regularization.json`. The constant sibling
changes only `lr_anneal_start=0`, `lr_floor=1`, and removes the network horizon
cap and floor. All runs use the host's fixed seed 0. Input-noise burn-in ends
after update 120; output-noise warmup ends at update 240. Those clocks stay
bound to 1,200 updates even when training continues longer.

| Arm | Frozen 1,000–1,200 window | 1,200–2,400 hold, checked every 10 updates | Verdict |
| --- | --- | --- | --- |
| Simpler scheduled control | 5/5, eight modes; minimum HQ .9998 | 120/120, eight modes; minimum HQ .9983 | PASS |
| Simpler constant sibling, G/D .00425, prior .0085 | 0/5; minimum four modes/HQ .2029 | Not attempted after frozen failure | FAIL |
| R1+R2 constant .001, coefficient .1, prior .002 | 5/5, eight modes; minimum HQ .9214 | 28/120; 92 failures, minimum zero modes/HQ zero | FAIL |

The R1+R2 row is especially diagnostic: it reached eight modes/HQ .9990 at
update 1,200, then dropped to zero modes/HQ zero at update 1,400. Its last two
dense checks recovered to eight modes/HQ .9988 and 1.0, but the 24-point
terminal window still failed at updates 2,000, 2,100, and 2,200. An endpoint
check would mislabel this run. These observations show non-sustained quality
under this exact constant-rate recipe; they do not by themselves identify a
limit cycle or prove a general impossibility result.

Every G, D, and prior optimizer call was traced. Constant arms had exactly
the declared group rates at all 1,200 or 2,400 calls. The two 2,400-update
runs reproduced their corresponding 1,200-update observation at update 1,200
exactly, excluding elapsed seconds. The scheduled control held every one of
its 120 continuation checks, showing that the extended host and fixed noise
clock can preserve a passing recipe.

For a distribution shift, the probe requires a full pre-shift hold after the
stationary window. The target changes in place after a checkpoint, preserving
model weights, Adam moments, EMA, and all training random streams. Recovery
must occur within 400 updates and **every** diagnostic check from the deadline
through the end must pass. A shifted run can be confirmed only against a
matched frozen/no-update control with the same pre-shift trace; the frozen
control must fail all post-deadline checks. No shifted candidate has been
promoted by the calibration above.

The [evidence manifest](continuous-evidence/manifest.json) inventories five
compressed raw episodes, the exact R1+R2 input config, and the standard
transfer-suite source archive by SHA-256. The standard snapshot excludes
`benchmarks/toy100/models.py`, although the noise adapter imports it. The
manifest therefore also includes a separately hashed byte-for-byte source
copy; that file has the same Git blob at frozen base `983d037` and this probe
revision. New probe episodes record its hash directly. Each episode includes its effective
config and hash, executable-source hashes, Python/PyTorch/CPU profile, all
observations, optimizer-rate ranges, and noise receipts. The source archive
binds the full benchmark and `particlegan` Python code. Logs were emitted as
one JSON record per diagnostic checkpoint so a running probe can be tailed.

## Passing-state local stability screen

A second bounded filter forks the live scheduled host at update 1,000. The
children inherit the exact same generator, discriminator, prior, Adam, EMA,
noise, data-stream, and global RNG state. The identity child reproduced the
separate cold 1,200-update run numerically at all 24 observations and matched
its full final-state SHA-256. This validates the 200-update continuation path;
it does not prove the state is a mathematical equilibrium.

Each child recorded all 200 live checks from update 1,001 to 1,200. The
following eight changed-dynamics rows all used the same cached state;
Lookahead interpolated both players and prior together with α=.5.

| Update after fork | G/D rate | Passing checks / 200 | Minimum modes / HQ | Local verdict |
| --- | ---: | ---: | --- | --- |
| Scheduled identity | Scheduled | 200 | 8 / .9963 | PASS |
| Ordinary Adam | .00425 | 6 | 1 / .0054 | FAIL |
| Lookahead k=2 | .00425 | 1 | 1 / .0110 | FAIL |
| Lookahead k=5 | .00425 | 28 | 1 / .0110 | FAIL |
| Lookahead k=10 | .00425 | 8 | 1 / .0110 | FAIL |
| Ordinary Adam | .001 | 200 | 8 / .9993 | PASS |
| Lookahead k=2 | .001 | 200 | 8 / .9995 | PASS |
| Lookahead k=5 | .001 | 200 | 8 / .9993 | PASS |
| Lookahead k=10 | .001 | 200 | 8 / .9995 | PASS |

The nominal .00425 rate destabilized this passing state with and without
Lookahead. At .001, ordinary Adam already passed every local check, so these
rows provide no evidence that Lookahead added stability. A locally passing
branch still needs cold mode acquisition, a 2,400-update hold, all required
hosts, and a matched shifted/frozen comparison before any continual-learning
claim. The [warm-screen archive](continuous-evidence/warm-lookahead/manifest.json)
contains the raw child episodes, exact Lookahead source, full benchmark source
snapshot, separately hashed noise source, and SHA-256 file inventory.

The matched fork was then extended to update 2,400 for the scheduled identity
and the two ordinary constant rates. The identity child again matched its
separate cold control in all 24 observations and the full final-state hash.
It passed all 200 checks through update 1,200 and all 120 checks every ten
updates afterward. Constant .00425 passed 6/200 and 6/120. Constant .001
passed 200/200 locally but failed one long check at update 1,950: all eight
modes were present, but HQ fell to .7903. It finished at eight modes/HQ .9990.
The strict all-check verdict is therefore FAIL for both constant rates.
Actual post-fork G/D/prior rates were .001/.001/.002 at every one of the
1,400 continuation optimizer calls in the lower-rate branch. The
[extended raw archive](continuous-evidence/warm-extended-2400/manifest.json)
retains the three branches, matched cold control, source snapshot, and hashes.
It also carries the separately hashed noise source omitted by the standard
snapshot.
