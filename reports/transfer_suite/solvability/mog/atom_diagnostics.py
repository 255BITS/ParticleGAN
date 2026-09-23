import gzip,hashlib,json
from pathlib import Path
import torch
out=Path('/tmp/pr36-solvability-mog-afe6152')
results=[]
for name in ('vector_unequal_mass','vector_unequal_width'):
    path=out/f'mog_sigma0_std0__{name}.json'
    record=json.loads(path.read_text())
    old_path=Path(f'/tmp/pr36-transfer-vectors-20260922/v2_rescored/{name}__fixed_cosine.json')
    old=json.loads(old_path.read_text())
    (out/f'reference_original__{name}.json.gz').write_bytes(gzip.compress(old_path.read_bytes(),mtime=0))
    before=old['result'];after=record['result']
    strip=lambda row:{k:value for k,value in row.items() if k!='seconds'}
    parity={'live':before['live']==after['live'],'ema':before['ema']==after['ema'],'all_24_observation_values':all(strip(a)==strip(b) for a,b in zip(before['observations'],after['observations'])) and len(before['observations'])==len(after['observations'])==24}
    assert all(parity.values())
    points=torch.tensor(record['prior_diagnostics']['generated_component_centers'])
    spec=record['spec'];means=torch.tensor(spec['means']);cov=torch.tensor(spec['covariances']);index=torch.cdist(points,means).argmin(1)
    components=[]
    for k in range(len(means)):
        p=points[index==k];delta=p-means[k]
        mahal=torch.einsum('ni,ij,nj->n',delta,torch.linalg.inv(cov[k]),delta)
        components.append({'component':k,'actual_distinct_atoms':len(torch.unique(p,dim=0)),'hq_atoms':int((mahal<=9).sum()),'max_squared_mahalanobis':float(mahal.max()),'sampled_covariance_error':after['live']['component_covariance_errors'][k]})
    results.append({'task':name,'zero_noise_control_exact_parity':parity,'reference_sha256':hashlib.sha256(old_path.read_bytes()).hexdigest(),'reference_archived_as':f'reference_original__{name}.json.gz','components':components,'note':'Counts enumerate actual G(prior.z) centers from a zero-noise run. They are not inferred from sampled output counts. This measures the baseline only, not unrecorded previous learned policies.'})
(out/'baseline_atom_diagnostics.json').write_text(json.dumps(results,indent=2)+'\n')
print('Baseline live, EMA and all24 observations exactly match on both tasks; actual-atom diagnostics saved.')
