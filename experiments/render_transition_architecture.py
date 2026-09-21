#!/usr/bin/env python
"""Export a diagram of the implemented shared-state transition encoder."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path('reports/transition/architecture')
OUT.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(16, 13), dpi=150)
fig.patch.set_facecolor('#f5f7fa')
ax.set(xlim=(0, 16), ylim=(0, 13)); ax.axis('off')
ink, muted = '#192738', '#536579'
blue, green, purple, orange = '#e4efff', '#dff5eb', '#eee5fb', '#ffedd7'

def text(x, y, s, size=12, color=ink, weight='normal', ha='left'):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va='center', linespacing=1.5)

def box(x, y, w, h, label, color='white', size=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                              facecolor=color, edgecolor='#cbd5df', linewidth=1))
    text(x+w/2, y+h/2, label, size, ha='center')

def arrow(x, y, xx, yy, color='#69829b', dashed=False):
    ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=14,
                                linewidth=1.6,color=color,linestyle='--' if dashed else '-'))

def card(y, h, title, subtitle):
    ax.add_patch(FancyBboxPatch((.3,y),15.4,h,boxstyle='round,pad=0.02,rounding_size=0.18',
                              facecolor='white',edgecolor='#dbe2ea'))
    text(.65,y+h-.35,title,14,weight='bold')
    text(.65,y+h-.7,subtitle,10.5,color=muted)

text(.4,12.55,'One transition. Three generators. An encoder that connects them.',22,weight='bold')
text(.4,12.12,'Implemented model • encoder_shared_state • MoG: 1,024 particles • no trajectories',12,color=muted)
card(8.8,2.9,'1  ORIGINAL SAMPLING','All three independent generators receive the same sampled center AND Gaussian noise.')
box(.7,9.55,2,.8,'MoG prior\nz',blue)
for y,label in ((10.45,'G1 → st'),(9.6,'G2 → at'),(8.95,'G3 → st+1')):
    box(4.3,y,2.3,.55,label,blue)
    arrow(2.75,9.95,4.23,y+.275)
    arrow(6.66,y+.275,9.25,9.95)
box(9.3,9.5,5.4,.9,'Original triple\n(st, at, st+1)',blue)
text(9.35,9.13,'This path still determines the leaderboard rank.',10,color=muted)
card(6.1,2.45,'2  SYNTHETIC COMPOSITION','Keep st and at from G1/G2; produce a new next state through the encoder.')
box(.7,6.75,2.3,.7,'Generated st, at',blue)
box(3.65,6.75,1.2,.7,'E',green)
box(5.5,6.75,1.5,.7,'z_hat',green)
box(7.65,6.75,1.25,.7,'G3',blue)
box(9.55,6.75,1.8,.7,'New st+1',blue)
box(12,6.75,2.65,.7,'Composed triple',blue)
for x,xx in ((3.05,3.59),(4.9,5.44),(7.05,7.59),(8.95,9.49),(11.4,11.94)):
    arrow(x,7.1,xx,7.1)
text(.75,6.43,'E selects a particle center + bounded offset. This path receives adversarial feedback and reconstructs generated st / at.',10,color=muted)
card(3.5,2.35,'3  LEARNING FROM REAL TRANSITIONS','E sees real st and at. The paired real st+1 is a training target, never an encoder input.')
box(.7,4.12,2.3,.7,'Real st, at',orange)
box(3.65,4.12,1.2,.7,'E',green)
box(5.5,4.12,1.5,.7,'Encoded z',green)
box(7.65,4.12,2.3,.7,'G1 / G2 / G3',blue)
box(10.6,4.12,4.05,.7,'Reconstruct st, at\nPredict paired st+1',orange,11)
for x,xx in ((3.05,3.59),(4.9,5.44),(7.05,7.59),(10,10.54)):
    arrow(x,4.47,xx,4.47)
text(.75,3.78,'Real normalized triple MSE + generated state/action reconstruction. E and the Gs share parameters across all paths.',10,color=muted)
card(.6,2.65,'4  ADVERSARIAL FEEDBACK','')
text(.75,2.43,'Three networks • four input roles',11,color=muted)
box(.7,1.15,3.1,1,'Real triples\nOriginal / composed triples',orange,11)
for y,label,target in ((2.13,'(st, at, st+1)','D_joint'),(1.43,'at','D_action'),(.73,'st at t  /  st+1 at t+dt','D_state  (shared)')):
    box(5,y,4.4,.5,label,'#f4f6f9',11)
    box(10.5,y,4.15,.5,target,purple,12)
    arrow(3.85,1.65,4.94,y+.25)
    arrow(9.46,y+.25,10.44,y+.25)
text(.4,.24,'Geometry, time and class condition every path. Shared D_state uses common state normalization. Repeated E / G labels mean reused weights.',10,color=muted)
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
for extension in ('png','svg','pdf'):
    fig.savefig(OUT/f'transition_encoder.{extension}',facecolor=fig.get_facecolor())
svg = OUT/'transition_encoder.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
plt.close(fig)
print(OUT/'transition_encoder.png')
