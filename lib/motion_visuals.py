"""Self-contained motion viewer and a small shareable skeleton animation."""
import json
from pathlib import Path

import numpy as np

from lib.motion import CHAINS, ACTIONS


def render(out, samples, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    test = samples['test']
    ids = [int(np.flatnonzero(test['c'] == c)[0]) for c in range(12)]
    p, t = cfg['past_length'], cfg['length']
    past = test['context'][ids].reshape(-1, 72, p).transpose(0, 2, 1).reshape(-1, p, 24, 3)
    real = test['real'][ids].transpose(0, 2, 1).reshape(-1, t, 24, 3)
    x = test['x'][ids, :4].transpose(0, 1, 3, 2).reshape(-1, 4, t, 24, 3)
    truth = np.concatenate([past, real], 1)
    futures = np.concatenate([np.repeat(past[:, None], 4, axis=1), x], 2)
    payload = dict(truth=truth.round(4).tolist(), samples=futures.round(4).tolist(),
                   names=ACTIONS, chains=CHAINS, past=p, frames=p+t, fps=cfg['display_fps'],
                   architecture=cfg['d_architecture'])
    html = '''<!doctype html><html><meta charset="utf-8"><title>Particle DDGAN: motion completion</title>
<style>body{background:#101923;color:#e6edf3;font:16px system-ui;margin:24px auto;max-width:1500px}canvas{width:100%;background:#172330;border-radius:12px}button,select,input{margin:8px;padding:8px}p{color:#b9c7d6}a{color:#84c9ff}</style>
<h1>One observed movement, four possible futures</h1>
<p>HumanAct12 · held-out subjects · <span id="architecture"></span> discriminator. Gray: recorded motion. Blue: generated motion. Each panel uses the same scale and camera.</p>
<select id="action"></select><button id="play">Pause</button><input id="frame" type="range" min="0" value="0"><span id="status"></span>
<label>Rotate view <input id="yaw" type="range" min="-180" max="180" value="25"></label>
<canvas id="canvas" width="1500" height="500"></canvas>
<p>The first 8 frames are observed. Four diffusion calls generate the entire 16-frame future before playback. These are the first four random draws for the first evaluated clip of each action; no sample selection by quality.</p>
<p>Different outputs alone do not demonstrate valid diversity. Inspect bone shape, boundary continuity and motion along with the numerical report. Playback uses approximate source timing.</p>
<script>const data=PAYLOAD;
const el=id=>document.getElementById(id),canvas=el('canvas'),ctx=canvas.getContext('2d');
el('architecture').textContent=data.architecture;el('frame').max=data.frames-1;
data.names.forEach((n,i)=>{let o=document.createElement('option');o.value=i;o.textContent=n;el('action').append(o)});el('action').value=1;
let playing=true,f=0,last=0;el('play').onclick=()=>{playing=!playing;el('play').textContent=playing?'Pause':'Play'};
el('frame').oninput=()=>{f=+el('frame').value;draw()};el('action').onchange=()=>draw();el('yaw').oninput=()=>draw();
function draw(){let a=+el('action').value,theta=+el('yaw').value*Math.PI/180;ctx.clearRect(0,0,1500,500);
let all=[data.truth[a],...data.samples[a]],pts=all.flat(2),project=q=>[Math.cos(theta)*q[0]+Math.sin(theta)*q[2],q[1]];
let xy=pts.map(project),xs=xy.map(q=>q[0]),ys=xy.map(q=>q[1]);let lo=[Math.min(...xs),Math.min(...ys)],hi=[Math.max(...xs),Math.max(...ys)];
let scale=Math.min(240/Math.max(hi[0]-lo[0],1),360/Math.max(hi[1]-lo[1],1));
function skeleton(pose,panel,color,width){ctx.strokeStyle=color;ctx.lineWidth=width;data.chains.forEach(chain=>{ctx.beginPath();chain.forEach((j,i)=>{let q=project(pose[j]),xx=panel*300+150+(q[0]-(lo[0]+hi[0])/2)*scale,yy=260+(q[1]-(lo[1]+hi[1])/2)*scale;i?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy)});ctx.stroke()})}
all.forEach((seq,i)=>{ctx.fillStyle='#e6edf3';ctx.font='18px system-ui';ctx.fillText(i===0?'Recorded future':'Sample '+i,i*300+60,35);if(i>0&&f>=data.past)skeleton(data.truth[a][f],i,'#657383',2);skeleton(seq[f],i,i===0||f<data.past?'#c3cbd5':'#52beff',4)});
el('frame').value=f;el('status').textContent=(f<data.past?'Observed prefix':'Generated future')+' · frame '+(f+1)+'/'+data.frames;}
function tick(ms){if(playing&&ms-last>1000/data.fps){f=(f+1)%data.frames;last=ms;draw()}requestAnimationFrame(tick)}draw();requestAnimationFrame(tick);
</script></html>'''.replace('PAYLOAD', json.dumps(payload, separators=(',', ':'), allow_nan=False))
    (Path(out)/'viewer.html').write_text(html)
    # Camera coordinates in released HumanAct12 have positive Y downward.
    truth = truth.copy(); futures = futures.copy()
    truth[..., 1] *= -1; futures[..., 1] *= -1
    # Two fixed actions, first three random futures. Gray reference overlay.
    fig, axes = plt.subplots(2, 3, figsize=(10, 7), subplot_kw={'projection': '3d'})
    lines = []
    for row, action in enumerate([1, 10]):
        values = np.concatenate([truth[action].reshape(-1, 3), futures[action, :3].reshape(-1, 3)])
        center = (values.min(0)+values.max(0))/2
        radius = max(float(np.ptp(values, axis=0).max()/2), .5)*1.1
        for col in range(3):
            ax = axes[row, col]
            ax.set(xlim=(center[0]-radius, center[0]+radius),
                   ylim=(center[2]-radius, center[2]+radius), zlim=(center[1]-radius, center[1]+radius))
            ax.set_box_aspect((1,1,1)); ax.view_init(elev=12, azim=-65); ax.set_axis_off()
            ax.set_title(f'{ACTIONS[action]} · sample {col+1}')
            for sequence, color, width in [(truth[action], '#aaaaaa', 2), (futures[action,col], '#138dcc', 2.5)]:
                for chain in CHAINS:
                    line, = ax.plot([], [], [], color=color, lw=width)
                    lines.append((line, sequence, chain))
    title = fig.suptitle('')
    def update(frame):
        title.set_text(f'{cfg["d_architecture"]} D · '+('observed prefix' if frame<p else 'generated future')+' · gray = recorded')
        for line, seq, chain in lines:
            q=seq[frame, chain]; line.set_data_3d(q[:,0], q[:,2], q[:,1])
        return [title]+[v[0] for v in lines]
    fig.tight_layout()
    FuncAnimation(fig, update, frames=p+t, interval=1000/cfg['display_fps']).save(Path(out)/'futures.gif', writer=PillowWriter(fps=cfg['display_fps']))
    plt.close(fig)
