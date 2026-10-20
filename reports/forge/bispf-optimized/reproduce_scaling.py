"""Render the resolution comparison from immutable matched performance receipts."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.scaling_publication import _initial_model_for_comparison, hardware_key, validate_receipt

SOURCE = 'd9d72d53761c898bce20102e1995fc3d899c1447'
MANIFESTS = ('bispf-matrix-controls-v1-d0053fc17b21f7a7.json',
             'bispf-matrix-quintic-v1-7c73a339487b7bfb.json')
SERIES = (('bispf-v2', 'Native BISPF', '#a44752'),
          ('bispf-optimized-quintic-v1', 'Optimized BISPF', '#1769aa'),
          ('recipe-prior-refactor-r1r2-v1', 'R1/R2', '#626970'))


def render():
    rows = {}
    for name in MANIFESTS:
        manifest = json.loads((ROOT / 'reports/forge/scaling-history' / name).read_text())
        for entry in manifest['entries']:
            receipt = json.loads((ROOT / entry['receipt']).read_text())
            validate_receipt(receipt)
            assert receipt['identity']['source']['commit'] == SOURCE
            assert receipt['metrics']['finite_update_fraction'] == 1
            assert receipt['metrics']['completed_updates'] == 13
            key = entry['candidate_id'], entry['task_id']
            assert key not in rows
            rows[key] = receipt
    sizes = (32, 64, 128, 256)
    for size in sizes:
        receipts = [rows[c, f'dcgan_performance_r{size}'] for c, _, _ in SERIES]
        identities = [r['identity'] for r in receipts]
        for field in ('batch_sequence_sha256', 'runtime_sha256', 'task_sha256', 'initialization_sha256'):
            assert all(i[field] == identities[0][field] for i in identities)
        assert all(_initial_model_for_comparison(r) == _initial_model_for_comparison(receipts[0])
                   and hardware_key(r) == hardware_key(receipts[0]) for r in receipts)
    with plt.rc_context({'font.size': 10, 'svg.hashsalt': 'bispf-h2-resolution-v1'}):
        fig, ax = plt.subplots(figsize=(7.2, 4.5), layout='constrained')
        for candidate, label, color in SERIES:
            times = [rows[candidate, f'dcgan_performance_r{n}']['metrics']['update_ms_median'] for n in sizes]
            ax.plot(sizes, times, marker='o', markersize=5, linewidth=2, color=color, label=label)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.set_yticks([10, 30, 100, 300, 1000, 3000], ['10', '30', '100', '300', '1,000', '3,000'])
        ax.set_xlabel('Image resolution (pixels per side)')
        ax.set_ylabel('Median complete update (ms; log scale)')
        ax.set_title('DCGAN update latency after BISPF optimization', loc='left', weight='bold', pad=30)
        ax.text(0, 1.025, 'RTX A6000 · batch 16 · base width 64 · FP32 · mock RGB data', transform=ax.transAxes,
                color='#555555', fontsize=9)
        ax.grid(axis='y', which='major', color='#dddddd', linewidth=.7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, loc='upper left')
        fig.supxlabel('3 warmup + 10 measured updates · seed 0 · source d9d72d537\n'
                      'Contention unknown: observed timings, no certified paired ranking.', fontsize=8, color='#555555')
        destination = Path(__file__).parent
        fig.savefig(destination / 'resolution-scaling.svg', metadata={'Date': None})
        fig.savefig(destination / 'resolution-scaling.png', dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    render()
