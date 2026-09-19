#!/usr/bin/env python
"""Wait for certified sweep, select reduced noise, verify forks, then train them."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.cifar_ae_particle_next import write
from experiments.run_grid import code_provenance, has_valid_summary


def main():
    report = ROOT/'reports/cifar-particle-ae/particle_overlap_training'
    sweep_path = ROOT/'reports/cifar-particle-ae/particle_overlap_sweep/results.json'
    write(report/'STATUS.json', {'stage': 'waiting_for_sampling_sweep'})
    deadline = time.monotonic()+3600
    while not sweep_path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Sampling sweep did not finish within an hour; no training launched')
        time.sleep(15)
    rows = json.loads(sweep_path.read_text())
    provenance = code_provenance(str(ROOT/'experiments/probe_cifar_ae_overlap.py'), sys.executable)
    for r in rows:
        assert has_valid_summary(str(ROOT/r['config']['out_dir']), r['config'], provenance)
    # Train from 80k: choose the smaller-noise arm with the least initial FID cost.
    starting = next(r for r in rows if r['results'][0]['step'] == 80000)
    selected = min((r for r in starting['results'] if r['noise_scale'] < 1), key=lambda r: r['fid'])
    scale = selected['noise_scale']
    write(report/'SELECTION.json', {'noise_scale': scale, 'selected_sampling_result': selected,
                                   'reason': 'Lowest FID among reduced-noise choices at the 80k training parent. This tests training prevention even if the sampling-only change is worse than baseline.'})
    cmd = [sys.executable, '-u', 'experiments/cifar_ae_overlap_training.py', '--noise-scale', str(scale)]
    for stage, extra in [('verifying_16_update_forks', ['--smoke']), ('training_80k_to_100k_forks', [])]:
        write(report/'STATUS.json', {'stage': stage, 'noise_scale': scale, 'command': cmd+extra})
        print('STAGE', stage, 'sigma multiplier', scale, flush=True)
        subprocess.run(cmd+extra, cwd=ROOT, check=True)
    write(report/'STATUS.json', {'stage': 'complete', 'noise_scale': scale})
    trained = json.loads((report/'results.json').read_text())
    lines = ['# Overlap intervention results', '', 'All arms start at the same 80k checkpoint; existing unchanged 100k control FID is 16.4609.', '']
    for r in trained:
        best = min(r['curve'], key=lambda v: v['generation']['fid'])
        q = r['endpoint_quality']['results'][0]
        lines.append(f"- {r['name']}: best sampled FID {best['generation']['fid']:.4f} at {best['step']}; final {r['final']['fid']:.4f}; endpoint coverage {q['quality']['coverage']:.2%}, density {q['quality']['density']:.4f}, latent confusion {q['geometry']['wrong_nearest_fraction']:.4%}.")
    winner = min(trained, key=lambda r: r['final']['fid'])
    lines += ['', 'Interpretation: freezing changes center adaptation broadly; a benefit does not isolate overlap as the cause. Reduced noise affects both sampling and learning. Compare its initial sampling result before crediting training.',
              f"Recommendation: review {winner['name']} first (lowest endpoint FID), including coverage and the full curve. " +
              ('It improves on the unchanged 100k control; consider a further checkpoint fork after reviewing whether it also improves on the 80k parent.' if winner['final']['fid'] < 16.4609 else 'Neither endpoint improves on the unchanged 100k control; do not extend either blindly.'),
              'No further training queued beyond 100k.']
    (report/'FINDINGS.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write(ROOT/'reports/cifar-particle-ae/particle_overlap_training/STATUS.json',
              {'stage': 'failed', 'error': repr(exc)})
        raise
