# Shared-c6 critic-gradient regularity screen

**No sustained live pass in 12 discriminator-only trials.** The unchanged
`shared_c6` recipe and original `vector_unequal_mass` setup were used throughout:
Rp logistic, b_cap coefficient 6 and κ 1.25, particle spread .05, Adam
betas `(0,.99)`, G/D learning rate .00425, particle learning rate .0085, and
the original hold/cosine schedule. Every trial retained 24 live and 24 EMA
observations, optimizer receipts, action traces, original/effective specs and
its exact source archive. No seed sweep or budget extension was used.

The [read-only diagnostic replay](../shared-rare-diagnostic/README.md) found
that the final variance failure of the previous best discriminator came from
the common 55% and 13% components; the 2% component exceeded the final
variance gate. To test whether bounding the discriminator's score or weight
operator norms stabilized within-mode spread, this screen froze eight spectral
normalization cards and four bounded-score cards before training.

| Discriminator | Final min eigen ratio | Final covariance error | Final HQ | Passing suffix |
| --- | ---: | ---: | ---: | ---: |
| `rare_sn_raw_first` | .03465 | 2.3733 | .9587 | 0 |
| `rare_sn_raw_hidden` | .03957 | .7976 | 1.0000 | 0 |
| `rare_sn_raw_head` | .00261 | .7716 | .9558 | 0 |
| `rare_sn_raw_all` | .00000 | 9.9985 | .3247 | 0 |
| `rare_sn_layer_first` | .00675 | 1.0674 | .9648 | 0 |
| `rare_sn_layer_hidden` | .01589 | 5.2733 | .9624 | 0 |
| `rare_sn_layer_head` | .02508 | .8168 | .9958 | 0 |
| `rare_sn_layer_all` | .00376 | .7206 | .9734 | 0 |
| `rare_bound_layer_1` | .00000 | 3.8178 | .5503 | 0 |
| `rare_bound_layer_2` | .00000 | .7692 | .9824 | 0 |
| `rare_bound_layer_4` | .00116 | .8747 | .9905 | 0 |
| `rare_bound_raw_2` | .00000 | 49.7504 | .0659 | 0 |

The minimum eigen gate is **.15**, covariance error must be at most **.85**,
HQ at least **.85**, and a pass needs five consecutive final observations with
all gates satisfied. All 12 cards have zero passing suffix. Even the best of
these final eigen ratios, .03957, is below the prior LayerNorm/Softplus β4
result, .12525. Bounding the score or spectral norms did not repair this case
under the shared high-rate recipe; in these trials, most variants contracted
the component clouds further. The screen remains negative evidence and leaves
the supported total at **18/19**.

For primary import, add [screen/index.json](screen/index.json) to the existing
`shared_c6` entry in `entries.json`; the index's sibling
[protocol](screen/protocol.json) and [exact source archive](screen/source.tar.gz)
contain every trial. [Frozen plan](plan.json) · [full live/EMA curves](screen/README.md)
· [tail-able log](screen.log). A dry-run import of all existing entries plus
this index validated 527 episodes with zero errors and kept `shared_c6` at
18/19. The dry run wrote only to `/tmp/shared-rare-import`.
