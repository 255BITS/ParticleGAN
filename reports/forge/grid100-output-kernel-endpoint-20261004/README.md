# Grid100: contribution of the existing output kernel

The saved 7,000-update Atlas state has all 100 centers and appropriate mode
mass, but its output-noise-off samples are much narrower than the target
Gaussian components. At this one endpoint, enabling the existing learned
public output kernel restores the target width and passes both the unchanged
native bounds and the separate toy100 accuracy bounds.

This is a completed endpoint diagnostic with **zero optimizer updates**. The
original clean-sample FAIL remains unchanged. It establishes neither sustained
acquisition nor a shipping default, convergence-time score or speed winner.

| Recorded 100,000-sample holdout | Output noise off | Public output noise on |
|---|---:|---:|
| Native endpoint bounds | FAIL | PASS |
| Separate accuracy bounds | FAIL | PASS |
| Recovered modes | 100 | 100 |
| Mode mass TV | .02587 | .02587 |
| Minimum covariance eigenvalue / target variance | .00222639 | .75990738 |
| Maximum covariance eigenvalue / target variance | .73856364 | 1.35489246 |
| Radial CDF KS | .85076128 | .01110865 |
| Absolute covariance trace bias | .96419436 | .03215146 |

The public kernel's saved standard deviation is `.028999999165534973`; target
component standard deviation is `.03`. The 20,000-sample primary endpoint shows
the same direction and passes only with the kernel enabled. These are recorded
original scorer outputs, not newly computed grades.

The corrected diagnostic restored the complete original GPU1 public trainer,
including its selected averaged state and RNG owners. Its clean 100,000 rows
match the original retained `live` holdout exactly. Each noisy sample equals
its paired clean sample plus the public Gaussian kernel bit for bit. The live
producer verified the shadow noise stream, full owner state, module modes and
global RNG purity. Root independently checked the retained array identities,
finite values and clean-plus-kernel arithmetic without drawing or rescoring.
The shadow stream's terminal bytes were not separately retained; that check
remains an explicitly identified producer receipt.

The first endpoint attempt remains INVALID: it used the target seed offset
`1601` for the generator. The corrected attempt uses the original mapping
target `1601`, noise `1602`, latent `1603`. Its measured cost is
**6.161060315091163 seconds**, plus the immutable prior engineering charge
**5.461433995049447 seconds**, for **11.62249431014061 / 180 seconds** with zero
reserve or overrun. This engineering scope is separate from the named-family
10,500-second campaign and the original scientific run's 797.9190550409257
seconds. No charges or outcomes are reset.

The scientific source remains commit
`9563dea57bb150f2a0275bbe8d785bf76210fca3`, digest
`db5492df4aa5ef60ce9e7d5869b2b6be8d492f19191a2889967274555f8d0037`.
The corrected producer and supervisor are copied here with their exact
[protocol](protocol.json). [Receipt](receipt.json), [public cost](cost.json),
[input pins](input-index.json) and [root byte review](root-retained-byte-review.json)
keep sample law, endpoint outcome, purity, source and costs distinct. Original
arrays, checkpoint and private lease records remain local; reproducing this
diagnostic requires those pinned inputs and root admission to the original GPU.

The [original training GIF and width view](../atlas-current-gpu-publication-views-20261003/README.md)
illustrate the training question. No training GIF was fabricated for this
zero-update endpoint. A future public-noisy native cohort would need its own
full acquisition and hold evaluation; this endpoint cannot fill those cells.
