"""Posthoc ring-tangent decomposition of the frozen common-noise cold assay."""

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmarks.locked_shared import mode_hold


def analyze(result):
    means = mode_hold.ring_means()
    target = means[6]
    rows = []
    for bank_id, row in enumerate(result['g_rows']):
        particles = []
        for item in row['particle_geometry_posthoc_only']:
            owner = item['owner']
            radial = means[owner] / means[owner].norm()
            chord = target - means[owner]
            tangent = chord - (chord @ radial)*radial
            if float(tangent.norm()) < 1e-6:
                tangent_projection = None
            else:
                tangent /= tangent.norm()
                tangent_projection = float(torch.tensor(item['accepted_displacement']) @ tangent)
            radial_projection = float(torch.tensor(item['accepted_displacement']) @ radial)
            particle_id = item['particle']
            particles.append({
                'particle': particle_id, 'owner_mode': owner,
                'surplus_donor': item['surplus_donor'],
                'tangential_toward_missing': tangent_projection,
                'outward_radial': radial_projection,
                'accepted_missing_chord_progress': item['accepted_missing_chord_progress'],
                'missing_center_distance_after': item['missing_center_distance_after'],
                'particle_hq_before': row['grade_before']['particle_hq_rate'][particle_id],
                'particle_hq_after': row['grade_after']['particle_hq_rate'][particle_id],
            })
        adjacent = [p for p in particles if p['owner_mode'] in (5, 7)]
        rows.append({
            'bank': bank_id, 'after_modes': row['grade_after']['modes'],
            'after_hq': row['grade_after']['hq'],
            'adjacent_tangent_min': min(p['tangential_toward_missing'] for p in adjacent),
            'adjacent_tangent_max': max(p['tangential_toward_missing'] for p in adjacent),
            'adjacent_tangent_positive': sum(p['tangential_toward_missing'] > 0 for p in adjacent),
            'adjacent_tangent_count': len(adjacent),
            'closest_missing_center_after': min(p['missing_center_distance_after'] for p in particles),
            'largest_particle_hq_loss': min(p['particle_hq_after']-p['particle_hq_before'] for p in particles),
            'particles': particles,
        })
    return {'scope':'posthoc diagnostic only; target means enter no fit or proposal',
            'missing_mode':6,'rows':rows}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=json.loads(args.result.read_text())
    args.output.write_text(json.dumps(analyze(result),indent=2)+'\n')


if __name__=='__main__':
    main()
