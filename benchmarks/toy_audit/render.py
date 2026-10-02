"""Render real evaluator checkpoints as training GIFs, never interpolated clouds.

Use --artifacts for the local, untracked capture directory. Only GIFs, PNG
posters and compact media receipts are written into --output.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from PIL import Image
import torch

from benchmarks.transfer_suite.vector_tasks import sample_target, target_scale
from .capture import write

COLORS = ["#b94352", "#137d69", "#3274aa"]
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                     "figure.facecolor":"#fafafa","axes.facecolor":"#fafafa"})


def read(path):
    return json.loads(Path(path).read_text())


def frame(fig):
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()).quantize(colors=128)


def save(frames, path):
    path.parent.mkdir(parents=True,exist_ok=True)
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=[220]*(len(frames)-1)+[1800],loop=0,optimize=True)
    frames[-1].convert("RGB").save(path.with_suffix(".png"))
    return dict(gif=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,
                frames=len(frames),interpolation=False,poster=path.with_suffix(".png").name)


def tiles(images, columns=8):
    images=np.asarray(images).reshape(-1,8,8)
    rows=(len(images)+columns-1)//columns
    out=np.full((rows*9+1,columns*9+1),.16,dtype=np.float32)
    for k,im in enumerate(images):
        y,x=divmod(k,columns)
        out[y*9+1:y*9+9,x*9+1:x*9+9]=im
    return out


def curves(ax, result, keys, upto, *, color, thresholds=()):
    rows=result.get("observations",result.get("curve",[]))[:upto+1]
    for k,key in enumerate(keys):
        values=[r.get(key,float("nan")) for r in rows]
        ax.plot([r["step"] for r in rows],values,color=color,ls="-" if k==0 else "--",label=key)
    for bound in thresholds:
        ax.axhline(bound,color="#777777",ls=":",lw=1)
    ax.set_xlim(0,result.get("spec",{}).get("steps",result.get("curve",result.get("observations"))[-1]["step"]))
    ax.grid(alpha=.2)


def transfer(records, title, path):
    records=records[:2]
    data=[np.load(Path(r["artifact"])/"observations.npz") for r in records]
    results=[read(Path(r["artifact"])/"result.json") for r in records]
    kind=records[0]["kind"]
    n=min(len(a["steps"]) for a in data)
    limits=None
    if kind=="vector":
        target=sample_target(records[0]["spec"],1600,torch.Generator().manual_seed(771),int(data[0]["steps"][-1])).numpy()
        combined=np.concatenate([target]+[a["live"].reshape(-1,2)[::4] for a in data])
        lo,hi=np.quantile(combined,[.001,.999],axis=0)
        span=np.maximum(hi-lo,1.)
        limits=(lo-.08*span,hi+.08*span)
    fig=plt.figure(figsize=(9.2,5.7),dpi=90)
    grid=fig.add_gridspec(len(records),3,width_ratios=[1.3,1.0,1.0],left=.06,right=.98,bottom=.15,top=.82,wspace=.34,hspace=.7)
    axes=[[fig.add_subplot(grid[i,j]) for j in range(3)] for i in range(len(records))]
    frames=[]
    for t in range(n):
        for i,(record,a,result,row) in enumerate(zip(records,data,results,axes)):
            for ax in row: ax.clear()
            step=int(a["steps"][t]); spec=record["spec"]
            metrics=result["observations"][t]
            label=(("transpose12" if spec["architecture"]=="transpose" and spec["width"]==12 else
                   "residual16" if spec["architecture"]=="residual_upsample" else spec["architecture"]) if kind=="image" else
                   ("published lengths" if i==0 and len(records)>1 else "published control" if len(records)>1 else "frozen reference"))
            cloud=a["live"][t]
            if kind=="image":
                # All particles in fixed table order; the two target tiles are
                # appended, clearly separated by their own caption.
                row[0].imshow(tiles(np.concatenate([cloud,a["templates"]])),cmap="gray",vmin=0,vmax=1,interpolation="nearest")
                row[0].set_axis_off()
                row[0].set_title(f"{label} · step {step}\nall {len(cloud)} particles; target tiles in last row",fontsize=9)
                curves(row[1],result,["hq"],t,color=COLORS[i],thresholds=[spec["thresholds"]["hq_min"]])
                row[1].set_ylim(-.03,1.05); row[1].set_title(f"HQ {metrics['hq']:.3f} · modes {metrics['modes']}/{spec['modes']}")
                curves(row[2],result,["mean_rmse"],t,color=COLORS[i],thresholds=[spec["thresholds"]["quality_rmse"]])
                row[2].set_ylim(0,max(.12,max(o["mean_rmse"] for o in result["observations"])*1.07))
                row[2].set_title("Mean nearest-template RMSE")
            else:
                reference=sample_target(spec,1600,torch.Generator().manual_seed(771),step).numpy()
                row[0].scatter(reference[:,0],reference[:,1],s=1,c="#777777",alpha=.18)
                row[0].scatter(cloud[::2,0],cloud[::2,1],s=2,c=COLORS[i],alpha=.28)
                lo,hi=limits
                row[0].set_xlim(lo[0],hi[0]);row[0].set_ylim(lo[1],hi[1])
                row[0].set_aspect("equal",adjustable="box")
                row[0].set_title(f"{label} · step {step}\ngray: target; color: clean live evaluator draw")
                curves(row[1],result,["sw1_normalized"],t,color=COLORS[i],thresholds=[.18])
                row[1].set_title(f"Normalized sliced W1 {metrics['sw1_normalized']:.3f}")
                max_sw=max(o["sw1_normalized"] for o in result["observations"])
                row[1].set_ylim(0,max(.25,max_sw*1.06))
                shape="resolved_core_min_eigen_ratio" if "resolved_core_min_eigen_ratio" in metrics else "component_min_eigen_ratio" if "component_min_eigen_ratio" in metrics else "covariance_error"
                curves(row[2],result,[shape],t,color=COLORS[i])
                row[2].set_title(shape.replace("component_","").replace("resolved_core_","").replace("_"," "))
            for ax in row[1:]: ax.set_xlabel("actual training update")
        fig.suptitle(title+"\n"+" / ".join(r["verdict"]["status"] for r in records)+" at full budget · live clean law · 24 observed checkpoints",y=.98,fontsize=12)
        fig.text(.06,.025,"Dotted lines are evaluation bounds. Gates also require a five-observation passing suffix. No clouds are interpolated.",fontsize=8)
        frames.append(frame(fig))
        fig.texts[-1].remove()
    plt.close(fig)
    return save(frames,path)


def behavior(record, path):
    result=read(Path(record["artifact"])/"result.json")
    rows=result.get("observations",result.get("curve",[]))
    bounds=record["spec"]["thresholds"]
    columns=min(3,len(bounds));row_count=(len(bounds)+columns-1)//columns
    fig,axes=plt.subplots(row_count,columns,figsize=(8.8,4 if row_count==1 else 6.4),dpi=90,squeeze=False)
    frames=[]
    for t in range(len(rows)):
        for ax in axes.flat:ax.clear()
        for ax,(key,op,bound) in zip(axes.flat,bounds):
            ax.clear();curves(ax,result,[key],t,color=COLORS[2],thresholds=[bound])
            ax.set_title(f"{key}\n{op} {bound:g}",fontsize=9);ax.set_xlabel("training update")
        for ax in list(axes.flat)[len(bounds):]:ax.set_axis_off()
        fig.suptitle(record["name"]+f" · {record['verdict']['status']} · step {rows[t]['step']}")
        fig.text(.06,.03,"Actual host metrics over training; this host has no recorded sample clouds. Live law; no interpolated measurements.",fontsize=8)
        fig.subplots_adjust(bottom=.17,top=.8,wspace=.35,hspace=.7)
        frames.append(frame(fig))
        fig.texts[-1].remove()
    plt.close(fig)
    return save(frames,path)


def native(artifact,path):
    a=np.load(artifact/"observations.npz");summary=read(artifact/"summary.json")
    rows=read(artifact/"observations.json");gates=read(artifact/"gates.json")
    fig,axes=plt.subplots(2,2,figsize=(9.2,6.2),dpi=90)
    fig.subplots_adjust(left=.075,right=.97,bottom=.16,top=.8,hspace=.6,wspace=.28)
    # The mode is chosen by index, never by which learned mode looks best.
    mode=0;frames=[]
    for t in range(len(a["steps"])):
        for ax in axes.flat:ax.clear()
        theta=a["angles"][t];c,s=np.cos(theta),np.sin(theta);rotation=np.array([[c,-s],[s,c]])
        centers=a["centers"]@rotation.T;step=int(a["steps"][t])
        axes[0,0].scatter(a["live"][t,:,0],a["live"][t,:,1],s=1,c="#3274aa",alpha=.35)
        axes[0,0].scatter(centers[:,0],centers[:,1],s=8,c="#222222",marker="+")
        axes[0,0].set_xlim(-7,7);axes[0,0].set_ylim(-7,7);axes[0,0].set_aspect("equal")
        axes[0,0].set_title("Served samples with learned output noise")
        for key,color in [("live","#3274aa"),("clean","#b94352")]:
            points=(a[key][t]-centers[mode])@rotation
            keep=np.max(np.abs(points),axis=1)<.15
            axes[0,1].scatter(points[keep,0],points[keep,1],s=8,c=color,alpha=.5,label=key)
        for sig,ls in [(1,"--"),(3,":")]:
            axes[0,1].add_patch(Ellipse((0,0),2*sig*.03,2*sig*.03,fill=False,ec="#222222",ls=ls))
        axes[0,1].set_xlim(-.12,.12);axes[0,1].set_ylim(-.12,.12);axes[0,1].set_aspect("equal")
        axes[0,1].set_title("Fixed mode 0 · width audit (target σ=.03)");axes[0,1].legend(loc="upper right",fontsize=7)
        visible=rows[:t+1]
        axes[1,0].plot([r["step"] for r in visible],[r["hq"] for r in visible],c="#3274aa",label="4k diagnostic HQ")
        hq_bound=.9*next(g["noisy"]["hq"] for g in gates if g["step"]==500) if "moving_original_status" in summary else .97
        axes[1,0].axhline(hq_bound,ls=":",c="#777777");axes[1,0].set_ylim(0,1.04);axes[1,0].set_title(f"HQ {rows[t]['hq']:.3f}; covered {rows[t]['modes']}/100 (4k)")
        passed=[g for g in gates if g["step"]<=step]
        for key,color in [("noisy","#3274aa"),("clean","#b94352")]:
            axes[1,1].plot([g["step"] for g in passed],[g[key]["min_cov_eig_ratio"] for g in passed],c=color,label=key)
        axes[1,1].axhline(.40,ls=":",c="#777777");axes[1,1].set_ylim(0,1.2)
        axes[1,1].set_title("Worst component variance / target variance (20k)");axes[1,1].legend(fontsize=7)
        for ax in axes[1]:ax.set_xlim(0,summary["steps"]);ax.set_xlabel("actual training update");ax.grid(alpha=.2)
        status=("moving gate "+summary["moving_original_status"]+" / strict native "+summary["native_noisy_status"]+" / accuracy "+summary["accuracy_status"] if "moving_original_status" in summary else
                "noisy "+summary["native_noisy_status"]+" / clean "+summary["native_clean_status"]+" / accuracy "+summary["accuracy_status"])
        fig.suptitle(f"Atlas · {summary['name']} · step {step} · target {np.degrees(theta):.0f}°\n{status} at full budget",fontsize=12)
        fig.text(.05,.035,"Actual saved observations; fixed diagnostic latent/noise draws. Target-jump frames keep the model fixed. Full gates use separate 20k draws.",fontsize=8)
        frames.append(frame(fig))
        fig.texts[-1].remove()
    plt.close(fig)
    return save(frames,path)


def stiff(artifact,path):
    rows=read(artifact/"observations.json");fig,axes=plt.subplots(1,3,figsize=(10,4.2),dpi=90)
    frames=[];indices=sorted(set(range(0,96,3))|{46,47,48,95})
    for t in indices:
        for ax in axes:ax.clear()
        for k,(label,values) in enumerate(rows.items()):
            x=[r["step"] for r in values[:t+1]]
            axes[0].plot(x,[max(r["game_after"]-np.log(2),1e-14) for r in values[:t+1]],c=COLORS[k],label=label)
            axes[1].plot(x,[r["next_scale"] for r in values[:t+1]],c=COLORS[k])
            axes[2].plot(x,[abs(r["residual"][0]) for r in values[:t+1]],c=COLORS[k])
        axes[0].set_yscale("log");axes[0].set_ylim(1e-14,1e4);axes[0].set_title("RpGAN game excess over log(2)");axes[0].legend(fontsize=7)
        axes[1].set_ylim(.35,1.1);axes[1].set_title("Native proposed LR scale")
        axes[2].set_yscale("log");axes[2].set_ylim(1e-18,1.);axes[2].set_title("Stiff coordinate |residual|")
        for ax in axes:ax.axvline(48,ls=":",c="#777777");ax.set_xlim(0,96);ax.set_xlabel("actual native generator update");ax.grid(alpha=.2)
        fig.suptitle(f"PR224 · constructed SettleTest unit fixture · step {t+1}\nUnsafe geometry escapes; cancel-one-release and safe geometry stay bounded",fontsize=11)
        fig.text(.06,.025,"Fixed constructed critic + specified settled Adam memory. This is a controller unit counterexample, not an end-to-end Atlas training failure.",fontsize=8)
        fig.subplots_adjust(left=.08,right=.98,bottom=.2,top=.76,wspace=.36)
        frames.append(frame(fig))
        fig.texts[-1].remove()
    plt.close(fig)
    return save(frames,path)


def misgan(artifact,path):
    a=np.load(artifact/"observations.npz");summary=read(artifact/"run/summary.json")
    history=summary["history"]
    fig,axes=plt.subplots(2,2,figsize=(9.2,6.2),dpi=90)
    fig.subplots_adjust(left=.075,right=.97,bottom=.16,top=.8,hspace=.6,wspace=.3)
    centers=np.stack(np.meshgrid(np.arange(-4.5,5),np.arange(-4.5,5)),axis=-1).reshape(-1,2)
    frames=[]
    for t in range(min(len(history),len(a["live"]))):
        for ax in axes.flat:ax.clear()
        metrics=history[t];visible=history[:t+1]
        cloud=a["live"][t]
        axes[0,0].scatter(cloud[::3,0],cloud[::3,1],s=1,c="#3274aa",alpha=.4)
        axes[0,0].scatter(centers[:,0],centers[:,1],s=8,c="#222222",marker="+")
        axes[0,0].set_xlim(-6,6);axes[0,0].set_ylim(-6,6);axes[0,0].set_aspect("equal")
        axes[0,0].set_title(f"EMA data generator: modes {metrics['modes']}/100; HQ {metrics['hq']:.1f}%")
        if len(a["imputed"]):
            imp=a["imputed"][t]
            for i in range(imp.shape[1]):axes[0,1].scatter(imp[:,i,0],imp[:,i,1],s=15,alpha=.6)
        axes[0,1].set_xlim(-6,6);axes[0,1].set_ylim(-6,6);axes[0,1].set_aspect("equal")
        axes[0,1].set_title("16 conditional draws for first 16 ambiguous rows\ncolor is a fixed test-row index")
        for key in ["acc","itv"]:axes[1,0].plot([r["step"] for r in visible],[r[key] for r in visible],label=key)
        axes[1,0].set_ylim(0,1.03);axes[1,0].set_title("Conditional mode accuracy / posterior TV");axes[1,0].legend(fontsize=7)
        axes[1,1].plot([r["step"] for r in visible],[r["istd"] for r in visible],label="imputation std",c="#b94352")
        axes[1,1].set_title("Does the imputer use its stochastic input?");axes[1,1].set_ylim(0,max(.1,max(r["istd"] for r in history)*1.08))
        for ax in axes[1]:ax.set_xlim(0,7000);ax.set_xlabel("actual training update");ax.grid(alpha=.2)
        fig.suptitle(f"PR196 · {summary['config']['mechanism']} · step {metrics['step']}\nCurrent develop recipe; EMA evaluation; distribution and conditional diversity are separate questions",fontsize=11)
        fig.text(.06,.035,"Unchanged proposal loop and evaluator. No scalar pass gate was declared by this PR; high global accuracy does not establish posterior calibration.",fontsize=8)
        frames.append(frame(fig))
        fig.texts[-1].remove()
    plt.close(fig)
    return save(frames,path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--cohort",choices=["proposals","base","native","special","all"],default="all")
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    receipts=read(args.output/"media.json") if (args.output/"media.json").exists() else {}
    if args.cohort in ("proposals","all"):
        for capture in sorted(args.artifacts.glob("pr*/capture-index.json")):
            records=read(capture);name=capture.parent.name
            if not records:continue
            path=args.output/(name+".gif")
            receipts[name]=transfer(records,f"{name.replace('-adapted','')} · {records[0]['name']}"+ (" · archived recipe adapter" if name.endswith("adapted") else ""),path)
            print(json.dumps(dict(event="GIF_DONE",name=name,bytes=path.stat().st_size)),flush=True)
    if args.cohort in ("base","all"):
        for record in read(args.artifacts/"develop-frozen-reference/index.json"):
            name="develop-"+record["name"];path=args.output/(name+".gif")
            receipts[name]=behavior(record,path) if record["kind"]=="behavior" else transfer([record],"Develop frozen host · "+record["name"],path)
            print(json.dumps(dict(event="GIF_DONE",name=name,bytes=path.stat().st_size)),flush=True)
    if args.cohort in ("native","all"):
        for summary in sorted(args.artifacts.glob("native-*/summary.json")):
            record=read(summary);name="atlas-"+record["name"];receipts[name]=native(summary.parent,args.output/(name+".gif"))
            print(json.dumps(dict(event="GIF_DONE",name=name)),flush=True)
    if args.cohort in ("special","all"):
        receipts["pr224"]=stiff(args.artifacts/"pr224-v2",args.output/"pr224.gif")
        for artifact in sorted(args.artifacts.glob("pr196-*")):
            if (artifact/"observations.npz").exists():
                receipts[artifact.name]=misgan(artifact,args.output/(artifact.name+".gif"))
                print(json.dumps(dict(event="GIF_DONE",name=artifact.name)),flush=True)
    write(args.output/"media.json",receipts)


if __name__=="__main__":main()
