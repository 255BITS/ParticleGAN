#!/usr/bin/env python
"""Per-mode core covariance distribution, supplementing aggregate shape metrics."""
from pathlib import Path
import argparse
import numpy as np
import json
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--out',required=True);args=p.parse_args()
centers=np.stack(np.meshgrid(np.arange(10)-4.5,np.arange(10)-4.5,indexing='ij'),-1).reshape(-1,2)
rows={}
for path in sorted(Path(args.root).glob('*/final_samples.npz')):
 x=np.load(path)['x'].astype(float)
 distances=((x[:,None,:]-centers[None,:,:])**2).sum(-1);ids=distances.argmin(-1);near=distances.min(-1)<.3**2
 eigs=[];records=[]
 for i in range(100):
  v=x[(ids==i)&near]
  if len(v)<20:continue
  e=np.linalg.eigvalsh(np.cov(v,rowvar=False))/.03**2
  eigs.append(e);records.append({'mode':i,'center':centers[i].tolist(),'core_samples':len(v),'eigenvalue_ratios':e.tolist()})
 if not eigs:continue
 e=np.array(eigs)
 rows[path.parent.name]={'definition':'nearest-center assignment; radius10sigma core; at least20samples/mode; eigenvalues divided by true variance','audited_modes':len(e),'median_min_eigenvalue_ratio':float(np.median(e[:,0])),'median_max_eigenvalue_ratio':float(np.median(e[:,1])),'modes_with_core_min_eigenvalue_below_0_1':int((e[:,0]<.1).sum()),'per_mode':records}
 print(path.parent.name,{k:v for k,v in rows[path.parent.name].items() if k not in ('definition','per_mode')})
Path(args.out).write_text(json.dumps(rows,indent=2)+'\n')
