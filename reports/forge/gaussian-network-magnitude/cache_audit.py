"""Posthoc read-only utilization of cached spectral motion caps; no gradients."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import torch
from experiments.forge.contracts import atomic_json,file_hash


def inspect(raw):
    rows=[]
    for task,phase in (('gaussian1d_acquisition','stationary'),('ring16_acquisition','stationary'),('gaussian1d_acquisition','shift')):
        path=raw/f'extrapolation_from_past-{task}-{phase}'/'state.pt'
        saved=torch.load(path,map_location='cpu',weights_only=True)
        cache=saved['trainer']['extrapolation']['previous'];parameters=[]
        for name,direction in cache.items():
            if name=='prior.z':continue
            factor=math.sqrt(max(1.,direction.shape[0]/direction.shape[1])) if direction.ndim==2 else 1.
            maximum=factor*math.sqrt(min(direction.shape)) if direction.ndim==2 else 1.
            parameters.append(dict(parameter=name,cached_direction_norm=float(direction.norm()),
                                   dualnorm_motion_cap_norm=maximum,
                                   cap_norm_utilization=float(direction.norm())/maximum))
        prior_norms=cache['prior.z'].norm(dim=1);nonzero=prior_norms[prior_norms>0]
        rows.append(dict(task=task,phase=phase,checkpoint_sha256=file_hash(path),parameters=parameters,
                         prior_nonzero_rows=len(nonzero),prior_direction_min_norm=float(nonzero.min()),
                         prior_direction_max_norm=float(nonzero.max()),
                         implied_mean_prior_row_motion=.03*float(nonzero.mean())))
    return dict(scope='posthoc_saved_direction_diagnostic_not_gate',training_updates=0,new_gradient_evaluations=0,
                model_sampling_draws=0,source_sha256=file_hash(Path(__file__)),results=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=inspect(args.raw);atomic_json(args.output,result);print(json.dumps(result))
