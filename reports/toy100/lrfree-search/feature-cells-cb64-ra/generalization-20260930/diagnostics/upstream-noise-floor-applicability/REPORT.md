# Upstream noise-floor applicability

The upstream change at cabe208 ignores the noise tester when deciding whether
learned output noise may use the controller mobility floor. It changes real
behavior once every retained generator/table scale is at most1/64, even when
the noise tester remains at1. The captured original/new function witness
shows sigma .029/zero derivative versus sigma .01/nonzero derivative at that
boundary. The change is valid and cannot be silently treated as cosmetic.

## Complete-lifetime source proof

Each active tester starts at s=1. An accepted stationary decision alone
reduces s, by a factor .5, and increments the cumulative stationary counter.
Drift increases s, reopen resets it to1, and population expiration uses max
with the previous scale. Restart/rebase/reopen never clear the cumulative
counter. Initial backend resolution occurs before the first update and
installs a fresh tester; later shape/representation changes are rejected.
Therefore for every t in a recorded finite training horizon0..T:

    s(t) >= 2 ** (-C_stationary(T))

This uses lifetime action counts and the actual transition law, rather than
interpolating final scales. Standard and sequential tester ASTs are identical
between the old upstream commit, new upstream commit, and RA16.

All19 original quality sources have a retained table tester with lifetime
stationary count at most2. Its scale was always at least1/4>1/64. Both versions
therefore choose settle=1 at every update, with identical floor arithmetic
and gradients. Some generator testers do cross1/64; the table witness remains
sufficient (stationary G7/table0 and ring G9/table0).

| Fixtures | Maximum table stationary count | Entire-horizon lower bound |
| --- | ---: | ---: |
|13 portable tasks |2 |1/4 |
|3 native tasks |2 |1/4 |
|3 fresh moving tasks |0 |1 |
|Original learned Toy |0 |1 |
|Original learned MNIST |0 |1 |

The two learned2000-update prefixes also prove that original replay windows
1001..1010 for both branches lie inside an unchanged noise-floor horizon.
No original quality fixture requires fresh training because of this noise
change. Latest combined-source replay and full suite still need fresh runs
for current-base compatibility; historical evidence keeps its original source
labels. The proof makes no claim about horizons beyond the recorded budgets
or unsupported callbacks/manual tester mutations.

## Artifacts

receipt.json records the21 exact checkpoints, source/config metadata, hashes,
role owners, lifetime counts, final scales for context, and all-step bounds.
The original and new upstream functions/controllers are captured verbatim.
No frozen file, repository source, GPU state, model, training stream or seed
experiment was changed. All checkpoint mapping and scalar witnesses used CPU;
CUDA remained uninitialized.
