"""Describe counters from full actual training; no training, draws or regrading."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from experiments.forge.contracts import read_json,atomic_json


def main():
    directory=Path(__file__).parent;results=read_json(directory/'results.json');rows=[]
    for task in results['comparison']['candidate']['tasks']:
        packet=task.get('hydraulic')
        if not packet:continue
        summary=packet['summary'];n=summary['updates']
        assert summary['max_shape_bound_violation']==0
        assert summary['max_accepted_radius_ratio']<=1
        rows.append(dict(task_id=task['task_id'],attempt_id=task['attempt_id'],updates=n,
            graph_components_mean=summary['graph_components_sum']/n,
            assigned_local_capacity_mean=summary['local_capacity_sum']/n,
            old_finite_excess_mean=summary['old_excess_sum']/n,
            accepted_finite_excess_mean=summary['accepted_excess_sum']/n,
            old_odd_response_energy_mean=summary['old_odd_energy_sum']/n,
            accepted_odd_response_energy_mean=summary['accepted_odd_energy_sum']/n,
            shape_active_fraction=summary['shape_active']/n,
            corrections_per_update=summary['shape_corrections']/n,
            limited_fraction=summary['limited']/n,rejected_fraction=summary['rejected']/n,
            ray_scale_mean=summary['scale_sum']/n,
            sampled_radius_mean=summary['radius_sum']/n,
            accepted_travel_mean=summary['accepted_rms_sum']/n,
            maximum_mean_linear_rounding_residual=summary['max_mean_linear_residual'],
            maximum_shape_bound_violation=summary['max_shape_bound_violation'],
            maximum_sampled_travel_radius_ratio=summary['max_accepted_radius_ratio']))
    output=dict(schema_version=1,source_digest=results['source_digest'],qualification_input=False,
        optimizer_updates_added=0,random_sampling_draws_added=0,results=rows,
        limitations='Training-batch graph components estimate neighborhoods, not true mixture components. Capacities and assignments change between updates but stay fixed within each proposal. Finite bounds protect consumed jitter only; global first-order mean correction does not guarantee finite mean, component means, adversarial progress or served-law quality.')
    atomic_json(directory/'controller-diagnostics.json',output);print(rows,flush=True)


if __name__=='__main__':main()
