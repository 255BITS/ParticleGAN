from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path('reports/cifar-ddgan/speed')
rows=json.loads((out/'leaderboard.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
labels={'unet32':'Baseline','cache':'Cache','cache_channels_last':'Cache + channels-last','fused_exact':'Cache + fused','lazy4':'Lazy 4','fd':'FD','fd_lazy4':'FD + lazy 4','batch128':'Batch 128','batch256':'Batch 256', 'ncsnpp128':'Baseline','cache_cl_fused_ncsnpp':'Full bundle','cache_only_ncsnpp':'Cache only','channels_last_only_ncsnpp':'Channels-last only','fused_only_ncsnpp':'Fused only','bundle_lazy4_ncsnpp':'Bundle + lazy 4','cache_fused_ncsnpp':'Cache + fused'}
for ax,arch,title in zip(axes,('unet','ncsnpp'),('U-Net: 64k samples per optimizer','NCSN++: 64k samples per optimizer')):
 for row in rows:
  if row['architecture']!=arch:continue
  ax.scatter(row['steady_samples_s'],row['fid'],s=45)
  offsets={'cache':(-10,-17),'cache_channels_last':(-28,-32),'fused_exact':(4,8),'fd':(4,5),'cache_fused_ncsnpp':(4,15),'bundle_lazy4_ncsnpp':(-65,-17)}
  ax.annotate(labels[row['run']],(row['steady_samples_s'],row['fid']),fontsize=8,xytext=offsets.get(row['run'],(4,5)),textcoords='offset points')
 ax.set(xlabel='Steady training samples/second →',ylabel='FID, 5k generated samples (lower is better)',title=title)
 ax.grid(alpha=.2);ax.margins(.3)
fig.suptitle('1k-step speed scouts — early quality, not convergence')
fig.tight_layout();fig.savefig(out/'scout_quality_speed.png',dpi=160);plt.close(fig)

variants=['exact','fused','lazy4','fd_lazy4']
fig,axes=plt.subplots(2,4,figsize=(13,6))
for j,label in enumerate(variants):
 data=np.load(f'results/speed_100gaussians_10k/{label}/final_samples.npz')['x']
 summary=json.loads(Path(f'results/speed_100gaussians_10k/{label}/summary.json').read_text())['final']
 axes[0,j].scatter(data[:8000,0],data[:8000,1],s=.5,alpha=.35)
 axes[0,j].set(xlim=(-5.5,5.5),ylim=(-5.5,5.5),aspect='equal',title=f"{label}\nHQ {100*summary['hq']:.1f}%, modes {summary['modes']}")
 # Nearest-center cell around (0.5,0.5); all modes use the same fixed zoom.
 local=data[(abs(data[:,0]-.5)<.5)&(abs(data[:,1]-.5)<.5)]
 axes[1,j].scatter(local[:,0]-.5,local[:,1]-.5,s=4,alpha=.5)
 axes[1,j].add_patch(plt.Circle((0,0),.09,fill=False,color='black',linestyle='--',linewidth=1))
 axes[1,j].set(xlim=(-.25,.25),ylim=(-.25,.25),aspect='equal',title='One mode; dashed radius = 3σ')
fig.suptitle('One-shot 100 Gaussians at 10k: counting modes misses shape/tail errors')
fig.tight_layout();fig.savefig(out/'one_shot_10k_shape.png',dpi=160);plt.close(fig)
