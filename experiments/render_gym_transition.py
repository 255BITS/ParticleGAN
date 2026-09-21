#!/usr/bin/env python
"""Build an offline, code-native Lunar Lander replay from evaluated checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lunar Lander · three-generator world model</title><style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#141d32;--text:#ecf1ff;--muted:#aebbd3;--blue:#65c9ff;--gold:#ffc974}
*{box-sizing:border-box}body{margin:0;background:var(--bg);font:15px/1.5 system-ui,sans-serif;color:var(--text)}
main{max-width:1260px;margin:auto;padding:32px 24px 48px}h1{font-size:30px;line-height:1.2;margin:8px 0 12px}h2{font-size:19px;margin:0 0 10px}
p{margin:8px 0;color:var(--muted)}.tag{font-size:12px;letter-spacing:1px;text-transform:uppercase;color:var(--blue)}
.panel{background:var(--panel);border:1px solid #27334d;border-radius:14px;padding:18px;margin-top:20px}
.controls{display:flex;gap:12px;align-items:end;flex-wrap:wrap}.controls label{display:grid;gap:5px;color:var(--muted);font-size:12px}
select,button{font:inherit;color:var(--text);border:1px solid #3b4d6b;background:#1c2a43;border-radius:7px;padding:8px 12px}
button{cursor:pointer}button:hover{background:#2a3a56}input[type=range]{accent-color:var(--blue);min-width:180px;flex:1}
.timeline{display:flex;gap:14px;align-items:center;margin-top:12px}canvas{display:block;width:100%;height:auto;border-radius:9px;background:#0b1426}
.canvases{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:18px}.canvas-label{font-weight:600;margin-bottom:8px}.real{color:var(--blue)}.pred{color:var(--gold)}
.metrics{display:flex;gap:25px;flex-wrap:wrap;margin:14px 0;color:var(--muted)}.metrics b{display:block;color:var(--text);font-size:20px;font-variant-numeric:tabular-nums}
.small{font-size:12px}.graph{display:grid;grid-template-columns:1fr 1fr;gap:14px}.graph pre{margin:0;white-space:pre-wrap;color:#d0dcf4;font:14px/1.7 ui-monospace,monospace}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:8px;border-bottom:1px solid #30405a}th{font-size:12px;color:var(--muted)}
.tablewrap{overflow-x:auto}.disabled{display:none!important}#error{color:#ff9c9c}a{color:var(--blue)}
@media(max-width:720px){main{padding:20px 12px}.canvases,.graph{grid-template-columns:1fr}h1{font-size:25px}.controls select{max-width:95vw}}
</style></head><body><main>
<div class="tag">LunarLander-v3 · finite individual transitions · MoG1024</div>
<h1>Can three generators learn a world together?</h1>
<p>G1 generates a state, G2 an action, and G3 a successor from the same latent draw. An encoder routes observed state/action pairs back through G3. Compare its predictions with the simulator and a direct supervised baseline.</p>
<div class="panel graph"><pre>Shared z + observed terrain
G1 -> st
G2 -> at
G3 -> st+1

E(st, at, terrain) -> z_hat -> G3</pre><pre>D_joint(st, at, st+1, terrain)
D_action(at, terrain)
D_state(st, terrain, current_role)
D_state(st+1, terrain, next_role)

3 generators · 3 critic networks · 4 roles</pre></div>
<div class="controls" style="margin-top:20px"><label>Model shown throughout<select id="model"></select></label></div>
<div class="panel" id="actionPanel"><h2>Engine what-if at the same recorded state</h2>
<p>Choose one of 11 cached commands at a replayed anchor. Every simulator branch starts from the same physical world and pre-step RNG state. Compare the command’s effect relative to engines off.</p>
<div class="controls"><label>Recorded anchor<select id="actionScene"></select></label><label>Cached engine command<select id="command"></select></label></div>
<div class="canvases"><div><div class="canvas-label real">Simulator successor · selected command</div><canvas id="actionReal" width="720" height="440"></canvas></div>
<div><div class="canvas-label pred">Predicted successor · selected command</div><canvas id="actionPred" width="720" height="440"></canvas></div></div>
<p class="small" id="actionNote"></p><div class="tablewrap"><table><thead><tr><th>Effect relative to engines off</th><th>Simulator</th><th>Model</th></tr></thead><tbody id="effects"></tbody></table></div>
<p class="small">Solid body: selected-command successor. Faint body: engines-off successor. Connecting arrows show actual position change relative to off; the table exposes velocity and rotation effects that are too small to see. Close-up axes cover all 11 saved command outcomes. This selector replays cached results; it does not simulate arbitrary new controls.</p></div>
<div class="panel"><h2>Real versus learned physics</h2>
<p id="modeNote"></p>
<div class="controls"><label>Fixed scene<select id="scene"></select></label>
<label>Prediction mode<select id="mode"><option value="recursive">Recursive · no state refresh</option><option value="one_step">One step · real state each step</option></select></label>
<button id="play">Play</button><button id="back">−1 step</button><button id="forward">+1 step</button><button id="reset">Reset</button></div>
<div class="timeline"><span id="stepLabel"></span><input id="step" type="range" min="0" value="0" step="1"><span class="small">¼ speed</span></div>
<div class="canvases"><div><div class="canvas-label real">Simulator observation</div><canvas id="real" width="720" height="490"></canvas></div>
<div><div class="canvas-label pred" id="predLabel">Model prediction</div><canvas id="pred" width="720" height="490"></canvas></div></div>
<div class="metrics"><div>Position distance <b id="position"></b></div><div>Velocity distance <b id="velocity"></b></div><div>Main command <b id="main"></b></div><div>Lateral command <b id="side"></b></div></div>
<canvas id="trace" width="1450" height="150"></canvas>
<p class="small" id="sceneNote"></p><p class="small">Recorded commands drive both displays. Playback ends at the saved reference limit; this is not autonomous model control. Predicted contacts feed back as p ≥ 0.5 in recursive mode. Body and leg drawings are schematic; eight observations do not recover the articulated legs. Velocity arrows show 0.2 seconds of linear extrapolation. Axes fit the entire selected scene, including divergence. No ground snapping or state repair.</p>
</div>
<div class="panel"><h2>Held-out numerical results</h2><div class="tablewrap"><table><thead><tr><th>Model / checkpoint</th><th>Next-state MSE ↓</th><th>vs persistence</th><th>Contact Brier ↓</th><th>Action-effect MSE ↓</th></tr></thead><tbody id="scores"></tbody></table></div>
<p class="small">Continuous errors use six coordinates and training-only scales. These aggregate metrics determine model comparisons; individual animations illustrate behavior. The dataset includes 11 terrain heights as privileged scene context. Checkpoints are selected on validation error; test scores are held out.</p></div>
<div class="panel" id="galleryPanel"><h2>Three-generator transition gallery</h2>
<p>Independent prior samples on a held-out terrain. G2 proposes commands from the data distribution. These samples are not a learned policy or a rollout.</p>
<div class="controls"><label>Terrain and path<select id="gallery"></select></label><label>Fixed saved sample<select id="sample"></select></label></div>
<canvas id="galleryCanvas" width="1450" height="580" style="margin-top:15px"></canvas>
<p class="small" id="galleryNote"></p></div>
<div class="panel"><h2>What this demonstration establishes</h2><p>The models train on shuffled single transitions, using paired successor supervision. Collection and evaluation retain episode prefixes for replay; histories are never model inputs. The three-generator model jointly learns observations and actions, then predicts through E → G3.</p>
<p>Engine dispersion and hidden Box2D state remain. The deterministic encoder produces continuous point predictions; a learned prior distribution does not establish calibrated conditional uncertainty. Animation is prerecorded checkpoint inference, not a live counterfactual simulator. A model-based landing controller is a separate future experiment.</p>
<p class="small">This HTML works offline. Scene selection uses the evaluator’s fixed episode/phase rules and includes failures. Model selection, replay controls, and gallery sample selection do not generate new predictions.</p></div>
<p id="error"></p></main>
<script id="payload" type="application/json">__DATA__</script><script>
'use strict';
const DATA=JSON.parse(document.getElementById('payload').textContent),$=id=>document.getElementById(id);
let frame=0,playing=false,last=0,selectedSceneKey=null;
const modelNames=DATA.rows.map(r=>r.name).filter(n=>DATA.models[n]);
const fmt=(x,n=4)=>Number.isFinite(x)?x.toFixed(n):'nonfinite';
function option(select,value,label){const o=document.createElement('option');o.value=value;o.textContent=label;select.append(o)}
for(const name of modelNames)option($('model'),name,name);
for(const row of DATA.rows){const tr=document.createElement('tr');for(const text of [row.name,fmt(row.mse,6),fmt(row.improvement,1)+'%',fmt(row.brier,5),fmt(row.action_mse,6)]){const td=document.createElement('td');td.textContent=text;tr.append(td)}$('scores').append(tr)}
function current(){return DATA.models[$('model').value].scenes[Number($('scene').value)]}
function path(scene){return $('mode').value==='one_step'?scene.one_step:scene.predicted}
function sceneKey(s){return s.episode_id+':'+s.anchor+':'+s.phase}
function populateScenes(oldKey){$('scene').replaceChildren();const scenes=DATA.models[$('model').value].scenes;scenes.forEach((s,i)=>option($('scene'),i,'Episode '+s.episode_id+' · '+s.phase+' · t='+s.anchor));const ix=scenes.findIndex(s=>sceneKey(s)===oldKey);if(ix>=0)$('scene').value=ix;populateGallery();populateActions();reset(false)}
function terrainPoints(terrain){return terrain.map((y,i)=>[-1+2*i/(terrain.length-1),y])}
function bounds(scene,prediction){const points=[scene.initial_state,...scene.real,...prediction,...terrainPoints(scene.terrain)];const xs=points.map(p=>p[0]).filter(Number.isFinite),ys=points.map(p=>p[1]).filter(Number.isFinite);const xmin=Math.min(-1.1,...xs),xmax=Math.max(1.1,...xs),ymin=Math.min(-.28,...ys),ymax=Math.max(.8,...ys);return [xmin-.05*(xmax-xmin),xmax+.05*(xmax-xmin),ymin-.08*(ymax-ymin),ymax+.08*(ymax-ymin)]}
function world(canvas,terrain,box){const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);const pad=42;
// Expand limits to preserve physical aspect (observation x and y have different scales).
box=box.slice();const ratio=(w-2*pad)/(h-2*pad)*(2/3),dx=box[1]-box[0],dy=box[3]-box[2];if(dx/dy<ratio){const grow=(dy*ratio-dx)/2;box[0]-=grow;box[1]+=grow}else{const grow=(dx/ratio-dy)/2;box[2]-=grow;box[3]+=grow}
const p=(x,y)=>[pad+(x-box[0])/(box[1]-box[0])*(w-2*pad),h-pad-(y-box[2])/(box[3]-box[2])*(h-2*pad)];
ctx.fillStyle='#0b1426';ctx.fillRect(0,0,w,h);ctx.strokeStyle='#22324b';ctx.lineWidth=1;ctx.font='14px system-ui';ctx.fillStyle='#8094b4';
for(let i=0;i<=4;i++){const x=box[0]+i*(box[1]-box[0])/4,y=box[2]+i*(box[3]-box[2])/4;ctx.beginPath();ctx.moveTo(...p(x,box[2]));ctx.lineTo(...p(x,box[3]));ctx.stroke();ctx.beginPath();ctx.moveTo(...p(box[0],y));ctx.lineTo(...p(box[1],y));ctx.stroke();ctx.fillText(fmt(x,2),p(x,box[2])[0]-12,h-22);ctx.fillText(fmt(y,2),4,p(box[0],y)[1]+4)}
const pts=terrainPoints(terrain),groundBase=Math.min(box[2],...terrain)-1;ctx.save();ctx.beginPath();ctx.rect(pad,pad,w-2*pad,h-2*pad);ctx.clip();ctx.beginPath();pts.forEach(([x,y],i)=>i?ctx.lineTo(...p(x,y)):ctx.moveTo(...p(x,y)));ctx.lineTo(...p(1,groundBase));ctx.lineTo(...p(-1,groundBase));ctx.closePath();ctx.fillStyle='#293444';ctx.fill();ctx.beginPath();pts.forEach(([x,y],i)=>i?ctx.lineTo(...p(x,y)):ctx.moveTo(...p(x,y)));ctx.strokeStyle='#95a4b7';ctx.lineWidth=2;ctx.stroke();ctx.restore();
ctx.fillStyle='#aebbd3';ctx.fillText('x (observation coordinates)',w/2-80,h-4);return {ctx,p};}
function line(ctx,p,points,color,width=2){ctx.beginPath();points.forEach((v,i)=>i?ctx.lineTo(...p(v[0],v[1])):ctx.moveTo(...p(v[0],v[1])));ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()}
function arrow(ctx,p,a,b,color){line(ctx,p,[a,b],color,2);const u=p(a[0],a[1]),v=p(b[0],b[1]),angle=Math.atan2(v[1]-u[1],v[0]-u[0]);if(Math.hypot(v[0]-u[0],v[1]-u[1])<2)return;ctx.beginPath();ctx.moveTo(...v);ctx.lineTo(v[0]-7*Math.cos(angle-.45),v[1]-7*Math.sin(angle-.45));ctx.moveTo(...v);ctx.lineTo(v[0]-7*Math.cos(angle+.45),v[1]-7*Math.sin(angle+.45));ctx.stroke()}
function lander(ctx,p,s,action,color,alpha=1){if(!s.every(Number.isFinite)){ctx.fillStyle='#ff8a8a';ctx.fillText('Nonfinite predicted state',70,60);return}
ctx.save();ctx.globalAlpha=alpha;const angle=s[4],co=Math.cos(angle),si=Math.sin(angle);const local=(x,y)=>[s[0]+(x*co-y*si)/10,s[1]+(x*si+y*co)/(20/3)];
const body=[[-.45,-.15],[-.45,.25],[0,.55],[.45,.25],[.45,-.15],[-.45,-.15]].map(([x,y])=>local(x,y));line(ctx,p,body,color,3);ctx.fillStyle=color+'25';ctx.fill();
for(const side of [-1,1]){line(ctx,p,[local(side*.35,-.08),local(side*.7,-.65),local(side*.9,-.65)],color,2);const foot=p(...local(side*.78,-.65));ctx.beginPath();ctx.arc(...foot,4,0,Math.PI*2);ctx.fillStyle=s[side<0?6:7]>=.5?'#8cf0a9':'#344863';ctx.fill()}
if(action&&action[0]>0){const flame=[local(-.18,-.15),local(0,-.4-.5*(action[0]+1)/2),local(.18,-.15)];line(ctx,p,flame,'#ff954b',3)}
if(action&&Math.abs(action[1])>.5){const sign=Math.sign(action[1]);line(ctx,p,[local(sign*.45,.2),local(sign*(.75+.3*Math.abs(action[1])),.2)],'#ff954b',3)}
// Observation vx = physical vx*10/50, vy = physical vy*(20/3)/50.
// Convert to displacement over 0.2 seconds in the observation coordinate frame.
const center=p(s[0],s[1]),tip=p(s[0]+.1*s[2],s[1]+.225*s[3]);ctx.beginPath();ctx.moveTo(...center);ctx.lineTo(...tip);ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();ctx.beginPath();ctx.arc(...center,3,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();ctx.restore();}
function drawTrace(scene,prediction){const ctx=$('trace').getContext('2d'),w=$('trace').width,h=$('trace').height;ctx.clearRect(0,0,w,h);const errors=prediction.map((s,i)=>Math.hypot(s[0]-scene.real[i][0],s[1]-scene.real[i][1]));const max=Math.max(.001,...errors.filter(Number.isFinite));ctx.font='17px system-ui';ctx.fillStyle='#aebbd3';ctx.fillText('Position error across the saved segment · max '+fmt(max),18,25);ctx.strokeStyle='#ffc974';ctx.lineWidth=2;ctx.beginPath();errors.forEach((e,i)=>{const x=20+(w-40)*i/Math.max(1,errors.length-1),y=h-20-(h-60)*e/max;if(Number.isFinite(y)){if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y)}});ctx.stroke();const x=20+(w-40)*Math.max(0,frame-1)/Math.max(1,errors.length-1);ctx.strokeStyle='#65c9ff';ctx.beginPath();ctx.moveTo(x,36);ctx.lineTo(x,h-12);ctx.stroke()}
function draw(){try{const scene=current();if(!scene)return;selectedSceneKey=sceneKey(scene);const prediction=path(scene);frame=Math.max(0,Math.min(frame,prediction.length));$('step').max=prediction.length;$('step').value=frame;$('stepLabel').textContent='Step '+frame+' / '+prediction.length;
const real=frame?scene.real[frame-1]:scene.initial_state,pred=frame?prediction[frame-1]:scene.initial_state,action=frame?scene.actions[frame-1]:[0,0],box=bounds(scene,prediction);
for(const [id,state,color,trajectory] of [['real',real,'#65c9ff',scene.real],['pred',pred,'#ffc974',prediction]]){const {ctx,p}=world($(id),scene.terrain,box);line(ctx,p,[scene.initial_state,...trajectory.slice(0,frame)],color+'80',1.5);lander(ctx,p,state,action,color);ctx.fillStyle='#b6c3dc';ctx.font='15px system-ui';ctx.fillText('angle '+fmt(state[4],3)+' rad · contacts '+fmt(state[6],2)+' / '+fmt(state[7],2),55,27)}
$('position').textContent=fmt(Math.hypot(real[0]-pred[0],real[1]-pred[1]));$('velocity').textContent=fmt(Math.hypot(real[2]-pred[2],real[3]-pred[3]));$('main').textContent=fmt(action[0],3);$('side').textContent=fmt(action[1],3);$('predLabel').textContent=$('model').value+' · '+($('mode').value==='recursive'?'recursive':'one step');
$('modeNote').textContent=$('mode').value==='recursive'?'Start once from the real observation, then feed predictions back under the same recorded commands.':'Predict each successor from its real preceding observation. Errors do not accumulate through feedback in this view.';
$('sceneNote').textContent='Episode '+scene.episode_id+' · anchor '+scene.anchor+' · '+scene.phase+' · '+(prediction.length/50).toFixed(2)+' seconds of simulated time, replayed at quarter speed. Position and velocity distances use observation units.';drawTrace(scene,prediction);$('error').textContent='';window.DEMO_READY=true;}catch(err){$('error').textContent=err.stack;window.DEMO_ERROR=String(err)}}
function reset(stop=true){frame=0;if(stop){playing=false;$('play').textContent='Play'}draw()}
function populateGallery(){const list=DATA.models[$('model').value].gallery||[];$('galleryPanel').classList.toggle('disabled',!list.length);$('gallery').replaceChildren();list.forEach((g,i)=>option($('gallery'),i,'Episode '+g.episode_id+' · '+g.mode));populateSamples()}
function populateSamples(){const g=(DATA.models[$('model').value].gallery||[])[Number($('gallery').value)];$('sample').replaceChildren();if(!g)return;g.transitions.forEach((_,i)=>option($('sample'),i,'Sample '+(i+1)));drawGallery()}
function drawGallery(){const g=(DATA.models[$('model').value].gallery||[])[Number($('gallery').value)];if(!g)return;const x=g.transitions[Number($('sample').value)],state=x.slice(0,8),action=x.slice(8,10),next=x.slice(10);const box=bounds({initial_state:state,real:[next],terrain:g.terrain},[]),{ctx,p}=world($('galleryCanvas'),g.terrain,box);lander(ctx,p,state,action,'#65c9ff');lander(ctx,p,next,null,'#ffc974',.85);line(ctx,p,[state,next],'#b8c5dc',2);ctx.fillStyle='#65c9ff';ctx.font='21px system-ui';ctx.fillText('G1 -> st',65,35);ctx.fillStyle='#ffc974';ctx.fillText(g.mode==='composed'?'E(G1,G2) -> z_hat -> G3':'G3 -> st+1',255,35);$('galleryNote').textContent='Blue: generated state and G2 command. Gold: '+(g.mode==='composed'?'composed':'joint prior')+' successor. Main '+fmt(action[0],3)+', lateral '+fmt(action[1],3)+'. The connecting line shows displacement, not an integrated physics path. No exact Box2D state is identified by an arbitrary generated observation.'}
function populateActions(){const scenes=DATA.models[$('model').value].action_scenes||[];$('actionPanel').classList.toggle('disabled',!scenes.length);$('actionScene').replaceChildren();scenes.forEach((s,i)=>option($('actionScene'),i,'Episode '+s.episode_id+' · '+s.phase+' · t='+s.anchor));populateCommands()}
function populateCommands(){const s=(DATA.models[$('model').value].action_scenes||[])[Number($('actionScene').value)];$('command').replaceChildren();if(!s)return;s.command_names.forEach((n,i)=>option($('command'),i,n));const main=s.command_names.findIndex(n=>n.includes('main')&&!n.includes('boundary'));if(main>=0)$('command').value=main;drawAction()}
function drawAction(){const s=(DATA.models[$('model').value].action_scenes||[])[Number($('actionScene').value)];if(!s)return;const i=Number($('command').value),off=s.command_names.indexOf('off'),points=[s.initial_state,...s.real,...s.predicted];const box=[Math.min(...points.map(v=>v[0]))-.12,Math.max(...points.map(v=>v[0]))+.12,Math.min(...points.map(v=>v[1]))-.16,Math.max(...points.map(v=>v[1]))+.16];
for(const [id,series,color] of [['actionReal',s.real,'#65c9ff'],['actionPred',s.predicted,'#ffc974']]){const {ctx,p}=world($(id),s.terrain,box);lander(ctx,p,series[off],null,color,.25);lander(ctx,p,series[i],s.actions[i],color);arrow(ctx,p,series[off],series[i],color)}
$('effects').replaceChildren();['x','y','vx','vy','angle (rad)','angular velocity'].forEach((name,j)=>{const tr=document.createElement('tr');for(const text of [name,(s.real[i][j]-s.real[off][j]).toExponential(4),(s.predicted[i][j]-s.predicted[off][j]).toExponential(4)]){const td=document.createElement('td');td.textContent=text;tr.append(td)}$('effects').append(tr)});$('actionNote').textContent='Main '+fmt(s.actions[i][0],6)+' · lateral '+fmt(s.actions[i][1],6)+' · all effects in observation coordinates.'}
$('model').onchange=()=>populateScenes(selectedSceneKey);
$('scene').onchange=()=>reset();$('mode').onchange=()=>reset();$('step').oninput=()=>{frame=Number($('step').value);draw()};$('reset').onclick=()=>reset();$('back').onclick=()=>{frame--;draw()};$('forward').onclick=()=>{frame++;draw()};$('play').onclick=()=>{playing=!playing;if(playing&&frame>=current().real.length)frame=0;$('play').textContent=playing?'Pause':'Play'};
$('gallery').onchange=populateSamples;$('sample').onchange=drawGallery;
$('actionScene').onchange=populateCommands;$('command').onchange=drawAction;
function tick(t){if(playing&&t-last>=80){last=t;frame++;if(frame>=current().real.length){playing=false;$('play').textContent='Play'}draw()}requestAnimationFrame(tick)}
populateScenes();requestAnimationFrame(tick);
</script></body></html>'''


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_one_step(models, rows, device):
    """Compute cached teacher-forced predictions, without changing recursive records."""
    from experiments.train_gym_transition import load_checkpoint, predict
    import torch
    torch.set_num_threads(1)
    for row in rows:
        name = row["name"]
        if name not in models:
            continue
        checkpoint = row.get("checkpoint")
        bundle = load_checkpoint(checkpoint, device=device) if checkpoint else None
        for scene in models[name]["scenes"]:
            inputs = np.asarray([scene["initial_state"]] + scene["real"][:-1], np.float32)
            if bundle is None:
                prediction = inputs.copy()
            else:
                action = np.asarray(scene["actions"], np.float32)
                terrain = np.repeat(np.asarray(scene["terrain"], np.float32)[None], len(inputs), 0)
                prediction = predict(bundle, inputs, action, terrain).cpu().numpy()
            scene["one_step"] = prediction.tolist()
        print(f"Cached one-step replay: {name} ({len(models[name]['scenes'])} fixed scenes)", flush=True)


def render_gif(models, name, destination):
    """First fixed scene, unmodified recursive outputs, code-native schematic bodies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Polygon

    scene = models[name]["scenes"][0]
    real = np.asarray([scene["initial_state"]] + scene["real"])
    pred = np.asarray([scene["initial_state"]] + scene["predicted"])
    terrain = np.asarray(scene["terrain"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), facecolor="#0b1020")
    body = np.array([[-.45, -.15], [-.45, .25], [0, .55], [.45, .25], [.45, -.15]])
    xmin, xmax = min(-1.1, real[:, 0].min(), pred[:, 0].min()), max(1.1, real[:, 0].max(), pred[:, 0].max())
    ymin, ymax = min(-.3, terrain.min(), real[:, 1].min(), pred[:, 1].min()), max(.8, real[:, 1].max(), pred[:, 1].max())
    patches, trails = [], []
    for ax, label, color in zip(axes, ("Simulator", f"{name} · recursive"), ("#65c9ff", "#ffc974")):
        ax.set_facecolor("#0b1426")
        ax.set_xlim(xmin-.1, xmax+.1)
        ax.set_ylim(ymin-.1, ymax+.1)
        ax.fill_between(np.linspace(-1, 1, len(terrain)), ymin-.1, terrain, color="#293444")
        ax.plot(np.linspace(-1, 1, len(terrain)), terrain, color="#95a4b7", lw=1)
        ax.tick_params(colors="#aebbd3", labelsize=8)
        ax.set_title(label, color=color, fontsize=11)
        patch = Polygon(body, closed=True, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(patch)
        patches.append(patch)
        trails.append(ax.plot([], [], color=color, lw=1, alpha=.6)[0])
    label = fig.text(.5, .04, "", ha="center", color="#d5e1f5", fontsize=9)
    fig.suptitle("Recorded actions · ¼ speed · schematic bodies · no state refresh", color="white", fontsize=11)
    fig.subplots_adjust(bottom=.17, top=.82, wspace=.22)

    def update(frame):
        for series, patch, trail in zip((real, pred), patches, trails):
            state = series[frame]
            co, si = np.cos(state[4]), np.sin(state[4])
            xy = body @ np.array([[co, si], [-si, co]])
            xy /= np.array([10, 20/3])
            xy += state[:2]
            patch.set_xy(xy)
            trail.set_data(series[:frame+1, 0], series[:frame+1, 1])
        action = scene["actions"][frame-1] if frame else [0, 0]
        label.set_text(f"Episode {scene['episode_id']} · {scene['phase']} · step {frame}/{len(real)-1}   "
                       f"main {action[0]:+.2f} · lateral {action[1]:+.2f}")
        return patches + trails + [label]

    animation = FuncAnimation(fig, update, frames=len(real), interval=80, blit=False)
    animation.save(destination, writer=PillowWriter(fps=12.5))
    plt.close(fig)


def render(evaluation_dir, out_dir, device="cpu", gif=False):
    evaluation = Path(evaluation_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "index.html").exists():
        raise FileExistsError("Use a fresh rendering directory")
    board = json.loads((evaluation / "leaderboard.json").read_text())
    models = json.loads((evaluation / "demo_data.json").read_text())
    rows = board["rows"]
    add_one_step(models, rows, device)
    compact_rows = [dict(name=r["name"], mse=r["conditional"]["continuous_mse"],
        improvement=r["improvement_over_persistence_percent"], brier=r["conditional"]["contact_brier"],
        action_mse=r["action_response"]["all"]["effect_mse"]) for r in rows]
    payload = json.dumps(dict(models=models, rows=compact_rows), allow_nan=False, separators=(",", ":"))
    (out / "index.html").write_text(HTML.replace("__DATA__", payload.replace("<", "\\u003c")))
    provenance = dict(renderer_sha256=sha256(__file__),
        inputs={name: sha256(evaluation / name) for name in ("leaderboard.json", "demo_data.json")},
        checkpoints={r["name"]: sha256(r["checkpoint"]) for r in rows if r.get("checkpoint")},
        scenes="Evaluator's fixed episode/phase selection; first saved gallery samples",
        one_step="Checkpoint inference from each real preceding state; cached offline",
        recursive="Evaluator's existing predicted arrays without modification",
        html_sha256=sha256(out / "index.html"))
    if gif:
        name = next((r["name"] for r in rows if r.get("checkpoint") and "adversarial" in r["name"]), rows[0]["name"])
        render_gif(models, name, out / "lunar_lander.gif")
        provenance["gif"] = dict(model=name, scene_index=0, sha256=sha256(out / "lunar_lander.gif"))
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (out / "README.md").write_text("# Lunar Lander world-model replay\n\nOpen `index.html` locally; no server or dependencies required. "
        "The viewer replays fixed checkpoint outputs, with one-step and recursive modes. "
        "It does not run a live simulator or a learned policy.\n\n"
        "The three-generator model uses `G1 -> st`, `G2 -> at`, `G3 -> st+1` from a shared learned MoG1024 draw. "
        "`E(st,at,terrain) -> z_hat -> G3` supplies the conditional prediction path. "
        "Every model receives privileged terrain context. Numerical results appear in the viewer.\n")
    with zipfile.ZipFile(out / "lunar_lander_demo.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("index.html", "README.md", "provenance.json", "lunar_lander.gif"):
            if (out / name).exists():
                archive.write(out / name, name)
    print(f"Offline demo: {out / 'index.html'}", flush=True)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", default="reports/gym/lunar_lander/baseline")
    parser.add_argument("--out-dir", default="reports/gym/lunar_lander/demo")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()
    render(args.evaluation_dir, args.out_dir, args.device, args.gif)


if __name__ == "__main__":
    main()
