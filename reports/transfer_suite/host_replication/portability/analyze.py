"""Compare two trace.py outputs: first differing operation and divergence growth.

  python3 -m reports.transfer_suite.host_replication.portability.analyze A.pt B.pt OUT.json
"""
import json
import sys

import torch

STEPS = [1, 2, 5, 10, 20, 50, 100, 150, 200, 250, 300, 400, 600, 800, 1000, 1200]


def main():
    a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
    layers = {name: dict(shape=list(a['first'][name].shape),
                         max_abs_difference=(a['first'][name].double() - b['first'][name].double()).abs().max().item(),
                         differing_elements=int((a['first'][name] != b['first'][name]).sum()))
              for name in a['first'] if name in b['first']}
    g, p = a['sizes']['g'], a['sizes']['p']
    ta, tb = a['trace'].double(), b['trace'].double()
    growth = []
    for step in [s for s in STEPS if s <= min(len(ta), len(tb))]:
        x, y = ta[step - 1], tb[step - 1]
        rel = lambda lo, hi: ((x[lo:hi] - y[lo:hi]).norm() / x[lo:hi].norm()).item()
        growth.append(dict(g_update=step, generator=rel(0, g), particles=rel(g, g + p), discriminator=rel(g + p, len(x))))
    first = next((i + 1 for i in range(min(len(ta), len(tb))) if not torch.equal(ta[i], tb[i])), None)
    report = dict(a=sys.argv[1], b=sys.argv[2], capabilities=[a['capability'], b['capability']], sizes=a['sizes'],
                  step1_outputs_and_first_update=layers, first_g_update_with_parameter_difference=first, relative_parameter_difference=growth)
    with open(sys.argv[3], 'w') as f:
        json.dump(report, f, indent=1)
    for name, row in layers.items():
        print(f"{name:30} {str(row['shape']):12} max|diff|={row['max_abs_difference']:.3e} differing={row['differing_elements']}")
    for row in growth:
        print(f"update {row['g_update']:5d}  G {row['generator']:.2e}  particles {row['particles']:.2e}  D {row['discriminator']:.2e}")


if __name__ == '__main__':
    main()
