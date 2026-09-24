"""Read-only warm/cold Chamfer receipts and fixed-noise target counterfactuals."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from benchmarks.locked_shared import mode_hold
from reports.toy100.coverage_fixed_eval import fixed_draw, score_support


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def stats(records):
    return dict(count=len(records), accepted=sum(r['accepted'] for r in records),
        alpha_counts={str(a): sum(r['alpha'] == a for r in records)
                      for a in sorted({r['alpha'] for r in records})},
        median_latent_move=statistics.median(r['actual_latent_displacement_norm'] for r in records),
        max_latent_move=max(r['actual_latent_displacement_norm'] for r in records),
        median_max_output_move=statistics.median(max(r['actual_output_displacement_norm']) for r in records),
        median_max_target_error=statistics.median(max(r['actual_target_residual_norm']) for r in records),
        median_empty_cells=statistics.median(r['empty_cells'] for r in records),
        all_rows_rank_two=all(all(rank == 2 for rank in r['numerical_rank']) for r in records),
        all_objectives_decrease=all(r['objective_after'] < r['objective_before'] for r in records))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--warm', type=Path, required=True)
    p.add_argument('--cold', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    warm, cold = read(args.warm), read(args.cold)
    records = cold['dynamics']['chamfer_records']
    observations = {r['step']: r for r in cold['result']['observations']}
    means = mode_hold.ring_means()
    rows = []
    for step in (750, 1000, 1050, 1100, 1150, 1200):
        record, observed = records[step - 1], observations[step]
        target = torch.tensor(record['target_points'], dtype=torch.float32)
        index, noise = fixed_draw(step, target)
        ideal = score_support(target, index, noise, means)
        rows.append(dict(step=step, actual={k: observed[k] for k in ('modes', 'hq', 'missing_modes')},
                         ideal_target={k: ideal[k] for k in ('modes', 'hq')},
                         maximum_nonlinear_target_error=max(record['actual_target_residual_norm']),
                         objective_before=record['objective_before'], objective_after=record['objective_after']))
    result = dict(scope='Post-training counterfactual only; target centers never enter the candidate',
                  input_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (args.warm, args.cold)},
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  warm=stats(warm['dynamics_receipt']['chamfer_records']), cold=stats(records),
                  paired_target_checks=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
