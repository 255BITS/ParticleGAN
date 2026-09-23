"""Post hoc update attribution and explicit label-aware output-geometry witness.

Neither changes training, creates a GAN success, or changes the suite metrics.
"""
import gzip,json,hashlib
from pathlib import Path
import torch
from benchmarks.transfer_suite import vector_tasks as v
root=Path('/tmp/pr36-rare-forensics');torch.set_num_threads(1)
read=lambda n:json.loads(gzip.decompress((root/n).read_bytes()))
write=lambda n,d:(root/n).write_bytes(gzip.compress((json.dumps(d,sort_keys=True,allow_nan=False)+'\n').encode(),mtime=0))
ref=read('reference.json.gz');spec=ref['spec'];snap=read('snapshots.json.gz')[-1];steps=read('updates.json.gz')
x=torch.tensor(snap['x'],dtype=torch.float64);means=torch.tensor(spec['means'],dtype=torch.float64);cov=torch.tensor(spec['covariances'],dtype=torch.float64)
assign=torch.cdist(x,means).argmin(1);rows=[]
for k in range(4):
 ids=torch.where(assign==k)[0];delta=x[ids]-x[ids].mean(0);emp=delta.T@delta/len(ids);eigen,axes=torch.linalg.eigh(emp);axis=axes[:,0]
 def variance(points):
  a=torch.tensor(points,dtype=torch.float64)[ids]@axis;return float(a.var(correction=0)/cov[k,0,0])
 decomposed=[]
 for step in steps:
  q00=variance(step['old_g_old_z']);q10=variance(step['new_g_old_z']);q01=variance(step['old_g_new_z']);q11=variance(step['new_g_new_z'])
  dg=.5*((q10-q00)+(q11-q01));dp=.5*((q01-q00)+(q11-q10))
  assert abs((dg+dp)-(q11-q00))<1e-12
  decomposed.append(dict(step=step['step'],before=q00,after=q11,g_effect=dg,prior_effect=dp))
 intervals=[]
 for start,end in [(1000,1050),(1050,1100),(1100,1150),(1150,1200),(1100,1200)]:
  s=[o for o in decomposed if start<o['step']<=end]
  for a,b in zip(s,s[1:]):assert abs(a['after']-b['before'])<1e-12
  intervals.append(dict(start=start,end=end,before=s[0]['before'],after=s[-1]['after'],g_effect=sum(o['g_effect'] for o in s),prior_effect=sum(o['prior_effect'] for o in s)))
 rows.append(dict(component=k,ids=ids.tolist(),fixed_axis=axis.tolist(),note='Variance along final minor axis for fixed final-assignment atoms. Exact symmetric two-factor attribution of actual Adam updates; no altered optimizer. Attribute contributions, not independent intervention outcomes.',intervals=intervals,updates=decomposed))
write('update_attribution.json.gz',rows)
# Fixed finite-support geometry witness, target-aware and deliberately not trained.
ids=torch.where(assign==3)[0];delta=x[ids]-x[ids].mean(0);emp=delta.T@delta/len(ids);eigen,axes=torch.linalg.eigh(emp);invsqrt=axes@torch.diag(eigen.rsqrt())@axes.T
new=x.clone();new[ids]=delta@invsqrt@torch.linalg.cholesky(cov[3]).T+means[3]
idx=torch.tensor(snap['eval_indices']);score=v.score_samples(new.float()[idx],spec,1200)
write('geometry_witness.json.gz',dict(label='label-aware output-only representability control; not a GAN run or sustained success',changes='Only affine whitening/recoloring/centering of the same6 rare output atoms; other250 outputs, all atom weights, target, metrics and evaluation multiplicities retained.',before=ref['result']['live'],after=score,all_final_metrics_pass=v.passes(score,spec['thresholds']),changed_atom_ids=ids.tolist(),output_atoms=new.tolist(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
for row in rows:
 print('COMPONENT',row['component'])
 for a in row['intervals']:print(a)
print('WITNESS',v.passes(score,spec['thresholds']),score)
