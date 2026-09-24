# Shared-v3 warmup architecture screen

This fixed-seed CPU search compares toy100 architecture resources under the
same optimizer and noise core used in the 19-task transfer replay. Its
[base declaration](../../configs/toy100/accuracy_warmup_v3_base.json) matches
the transfer candidate in every common optimizer/noise field: generator LR
.00425, discriminator multiplier 1, prior multiplier 2, Adam β=(0, .99),
one-sided cap 6 at κ=1.25, prior regularization .05, LR anneal from 40% to
floor .05, output noise peak .029 warmed up over 20% of updates, and input
noise .5 reduced to zero by 10% of updates. Toy100 uses its declared 7,000
updates, 20,000 particles, 20,000 final evaluation draws, and batch 2048 for
every geometry. The seed remains 1234.

The four planned [candidate overrides](../../configs/toy100/accuracy_warmup_arch_search.json)
compare the plain MLP generator and Fourier discriminator. `n_hidden` is the
existing shared depth setting for both generator and discriminator; reducing
it from 3 to 2 therefore changes both networks' depth. The discriminator
width remains 128 in every trial.

| Candidate | Fourier features | Generator width | Shared MLP depth | Batch |
| --- | ---: | ---: | ---: | ---: |
| warm_b2048_f2_g128_d3 | 2 | 128 | 3 | 2048 |
| warm_b2048_f3_g128_d3 | 3 | 128 | 3 | 2048 |
| warm_b2048_f3_g32_d2 | 3 | 32 | 2 | 2048 |
| warm_b2048_f3_g64_d2 | 3 | 64 | 2 | 2048 |

The F2, batch-2048 control runs first. If it fails materially, the search
prioritizes [global optimizer probes](../../configs/toy100/accuracy_warmup_core_search.json)
with the same architecture and noise schedule: generator LR .002125 with the
original β2=.99, or the original LR .00425 with β2=.999. The β2=.999 variant
also has a separately declared anneal-start-.6 comparison because it performs
better in the wider transfer screen. These are distinct common-core proposals
that must each transfer to the other 19 tasks. The
remaining architecture candidates stay declared but deferred until there is
evidence that the common core can support full toy100 coverage.

Every completed candidate is judged by the unchanged five-terminal-check
live-weight gate and the separately frozen final-distribution accuracy limits.
The grid is screened first; only a competitive full-gate candidate is worth
promoting unchanged to rotated and staggered. All failures, configs, logs,
source hashes, and exact final scored draws are retained.

## Completed grid trials

Three plain-MLP F2/batch-2048 candidates ran all 7,000 updates on CPU with the
same 20%-warmup output noise and 10%-cutoff input noise. Each complete
run directory under the local ignored path `artifacts/toy100-accuracy/search-agent/warmup-screen`
retains its plan, config, events, source hashes, progress log, frozen-gate
result, and exact final live draws for the independent accuracy audit.

| Common-core variant | Covered modes | HQ | Mass TV | Center RMS / σ | Abs. cov bias | Radial KS | Accuracy score | Original streak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v3 LR .00425, β2=.99, anneal .4 | **85** | **.98265** | **.20100** | **.273** | .0063 | **.0174** | **1.303** | 0/5 |
| v3 LR .00425, β2=.999, anneal .4 | 54 | .89090 | .38240 | .727 | .0509 | .0810 | 3.135 | 0/5 |
| v3 LR .00425, β2=.999, anneal .6 | 59 | .91940 | .36865 | .632 | .0416 | .0506 | 2.747 | 0/5 |

All three fail both gates. Warmup can give approximately target within-mode
spread on the first candidate, but it does not repair mode allocation. Raising
β2 to .999 markedly worsens coverage in this plain-MLP architecture under
either annealing start. The half-LR candidate and the declared F3/smaller-MLP
architecture candidates were not run: these core settings already fail far
from the grid gate, while separate 19-task transfer checks also reject the
leading exact-schedule v3 candidates. This branch stops here; the distinct
data-space affine architecture is being evaluated separately.
