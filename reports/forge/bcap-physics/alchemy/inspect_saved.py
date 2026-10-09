"""Zero-training saved-sample witness audit; target-aware controls are diagnostic only."""
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.special import ndtri
import torch
from particlegan.distillation import distillation_loss

ROOT = Path(__file__).resolve().parents[4]
REPORT = Path(__file__).resolve().parent


def teacher(spec, n, masses=None, width=1., shift=0.):
    masses = np.array(spec['masses'] if masses is None else masses)
    group = np.searchsorted(np.cumsum(masses), (np.arange(n) + .5) / n)
    p = (np.arange(n)[:, None] + .5) * np.array([math.sqrt(2), math.sqrt(3)]) % 1
    gaussian = ndtri(np.clip(p, 1e-8, 1-1e-8))
    means, covs = np.array(spec['means']), np.array(spec['covariances'])
    return torch.from_numpy(means[group] + width * np.einsum('nij,nj->ni', np.linalg.cholesky(covs)[group], gaussian) + shift)


def measure(fake, real):
    fake = fake.detach().clone().requires_grad_()
    loss, parts = distillation_loss(fake, real, cells=8, return_parts=True)
    grad, = torch.autograd.grad(loss, fake)
    return dict(loss=float(loss.detach()), **{k:float(v.detach()) for k,v in parts.items()},
                gradient_rms=float(grad.square().mean().sqrt()), finite=bool(torch.isfinite(grad).all()))


def main():
    started=time.monotonic()
    torch.set_num_threads(1)
    before=torch.get_rng_state().clone()
    path=ROOT/'reports/forge/bcap-tier2-search/failure-state-analysis.json'
    original=json.loads(path.read_text())
    result=dict(schema_version=1,scope='zero_training_target_aware_witness_diagnostic_only',
        qualification_credit=False,optimizer_updates=0,random_draws=0,
        original_source=original['source_digest'],original_candidate=original['candidate_id'],
        source_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),tasks={})
    for task in ('vector_unequal_mass','vector_unequal_width'):
        binding=next(x for x in original['artifact_proofs'] if x['task']==task and x['path'].endswith('observed-samples.pt'))
        saved_path=Path(binding['path'])
        digest=hashlib.sha256(saved_path.read_bytes()).hexdigest()
        assert digest==binding['sha256']
        saved=torch.load(saved_path,map_location='cpu',weights_only=True)[-1]['samples'].double()
        spec=json.loads((ROOT/'configs/forge/tasks'/f'{task}.json').read_text())['execution']['host_definition']
        real=teacher(spec,len(saved))
        masses=np.array(spec['masses']); masses[-1]=0; masses/=masses.sum()
        result['tasks'][task]=dict(artifact=binding,step=1200,samples=len(saved),
            teacher_law='deterministic normal quantiles with irrational rotation; separate target-aware diagnostic',
            saved_endpoint=measure(saved,real), empirical_null=measure(real,real),
            collapse=measure(teacher(spec,len(saved),width=0),real),
            broadening=measure(teacher(spec,len(saved),width=2),real),
            shift=measure(teacher(spec,len(saved),shift=.2),real),
            missing_component=measure(teacher(spec,len(saved),masses=masses),real),
            original_center_allocation=original['diagnostics'][task]['latent_center_allocation'])
        print(task,result['tasks'][task]['saved_endpoint'],flush=True)
    result['global_rng_unchanged']=torch.equal(before,torch.get_rng_state())
    result['cpu_wall_seconds']=time.monotonic()-started
    (REPORT/'saved-evidence.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main()
