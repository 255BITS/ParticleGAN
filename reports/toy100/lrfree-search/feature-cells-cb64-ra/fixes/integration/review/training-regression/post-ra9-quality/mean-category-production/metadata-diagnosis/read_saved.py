"""CPU metadata-only extraction of the closed failed fresh9 mechanics state."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
OWNER=HERE.parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    seal=json.loads((HERE/'PREPARATION.json').read_text())
    for p,d in seal['source_and_input_sha256'].items():assert sha(p)==d,p
    target=HERE/'result.json'
    assert not target.exists()
    import torch
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('mean_owner',OWNER/'pkg-MEAN/particlegan/__init__.py',
        submodule_search_locations=[str(OWNER/'pkg-MEAN/particlegan')])
    package=importlib.util.module_from_spec(spec);sys.modules['mean_owner']=package;spec.loader.exec_module(package)
    before=torch.load(OWNER/'cpu-contract-attempt1/grid/before.pt',map_location='cpu',weights_only=False)
    after=torch.load(OWNER/'cpu-contract-attempt1/grid/after.pt',map_location='cpu',weights_only=False)
    source=torch.load(ROOT/'validation-cb64-ra9/screens/runs/grid100/final-state.pt',map_location='cpu',weights_only=False)['trainer']
    proto=json.loads((ROOT/'integration/review/training-regression/post-ra9-quality/mean-category-transport/attempt1/grid.json').read_text())
    def tensor_bytes(t):
        return hashlib.sha256(str((t.dtype,tuple(t.shape))).encode()+t.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    b=after['birth_death'];last=b['last']
    models={role:all(tensor_bytes(v)==tensor_bytes(before['models'][role][k]) for k,v in source['models'][role].items())
            for role in ('G','D','prior','ema_G','ema_prior')}
    result=dict(status='METADATA_EXTRACTED',mean_transport=last['mean_transport'],mass_topology=last['mass_topology'],
        actual_cells=last['cells'],actual_rank=last['metric_rank'],paired_chart_valid=b['paired_average']['chart_valid'],
        duplicate_fraction=last['duplicate_fraction'],eligible=last['eligible'],
        phase_moves={k:v for k,v in last.items() if k.endswith('_moves') or k=='moves'},
        mean_counters={k:v for k,v in b['counters'].items() if k.startswith('mean_')},
        fixture_comparison=dict(raw_source_models_tables_exact=models,source_completed_steps=source['completed_steps'],
            constructed_steps_before=before['completed_steps'],constructed_steps_after=after['completed_steps'],
            reference_FIFO_exact=tensor_bytes(source['birth_death']['reservoir'])==tensor_bytes(before['birth_death']['reservoir']),
            bandwidth_exact=tensor_bytes(source['controller']['latent_bandwidth'])==tensor_bytes(before['controller']['latent_bandwidth']),
            dedicated_mechanics_seed=314159,reaction_seed=314165,
            chart_random_context='fresh device-native bd stream manual_seed314165; prototype cloned saved CPU RNG',
            prototype_witness=proto['witness'],prototype_topology=proto['real_topology'],
            prototype_earlier_ordinary=proto['pair_supply']['earlier_ordinary_actions'],
            mechanics_settings=before['birth_death']['settings'],prototype_source_preseal='9818d0439d36b6801ee9c5c21554c5e9b504bf1f1494b1443ebcc89d1ee520ab'),
        source_and_input_sha256=seal['source_and_input_sha256'],
        no_forward_constructor_planner_scoring_or_new_RNG=True,cpu_only=True,cuda_initialized=torch.cuda.is_initialized())
    for p,d in seal['source_and_input_sha256'].items():assert sha(p)==d,p
    assert not torch.cuda.is_initialized()
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],mean=result['mean_transport'],groups=result['mass_topology'].get('groups'),moves=result['phase_moves']),indent=2))

if __name__=='__main__':main()
