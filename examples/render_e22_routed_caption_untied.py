"""CPU60s render of saved actual target and three-arm observations.

python -m examples.render_e22_routed_caption_untied --run-directory RUN
Optional NumPy/Pillow>=10.1; no models, native updates or GPU forwards.
"""
import time
STARTED=time.monotonic()
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

ARMS=("ordinary_BF16","particle_BF16","particle_untied_BF16")
STEPS=(0,64,128,256,384,512)
LIMIT=60


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def budget():
    if time.monotonic()-STARTED>LIMIT:raise TimeoutError("CPU render60s exceeded")


def token_maps(value):
    if value.ndim!=3 or len(value)<6 or not torch.isfinite(value).all():raise ValueError("six finite source maps required")
    side=math.isqrt(value.shape[1])
    if side*side!=value.shape[1]:raise ValueError("square patch grid required")
    return value[:6].double().square().mean(-1).sqrt().reshape(6,side,side).numpy()


def goal_frame(target,ordinary,shared,untied,*,step,vmax,terminal):
    if not math.isfinite(vmax) or vmax<=0:raise ValueError("positive fixed initial scale required")
    maps=[token_maps(v) for v in (target,ordinary,shared,untied)]
    image=Image.new("RGB",(1230,860),"white");draw=ImageDraw.Draw(image)
    font=ImageFont.load_default(size=17);small=ImageFont.load_default(size=14)
    draw.text((20,15),f"Caption edits | native update {step}/512 per arm",fill="#111111",font=font)
    draw.text((20,44),"Observed patch RMS velocity error; desired residual is zero",fill="#333333",font=small)
    for row,(name,values) in enumerate(zip(("Target = 0","Ordinary LoRA","Shared Up","Untied Ups"),maps)):
        y=94+row*174;draw.text((15,y+65),name,fill="#111111",font=small)
        for source,value in enumerate(values):
            x=175+source*170;a=np.clip(value/vmax,0,1)
            rgb=np.stack((np.full_like(a,255),255*(1-a),255*(1-a)),-1).astype(np.uint8)
            image.paste(Image.fromarray(rgb).resize((155,155),Image.Resampling.NEAREST),(x,y));draw.rectangle((x,y,x+155,y+155),outline="#dddddd")
            if row==0:draw.text((x+44,72),f"Source {source}",fill="#333333",font=small)
    draw.text((20,806),f"All frames: white=0, red={vmax:.6f} physical error. Six fixed TEST queries; formal score uses48.",fill="#333333",font=small)
    label=(f"Terminal {terminal['scientific_status']}: ordinary {terminal['ordinary']:.8f}, shared {terminal['shared']:.8f}, untied {terminal['untied']:.8f}"
           if step==512 else "Visual observations only; numerical PASS/FAIL uses the fixed512 endpoint.")
    draw.text((20,832),label,fill="#111111",font=small)
    return image


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run-directory",type=Path,required=True);args=parser.parse_args()
    run=args.run_directory;source=sha(__file__);before=torch.get_rng_state().clone();threads=torch.get_num_threads();torch.set_num_threads(1)
    writable=False;completion=None;error=None;code=2
    try:
        if any((run/name).exists() for name in ("goal.gif","goal-final.png","media-completion.json")):raise ValueError("preserve existing media; fresh unrendered run required")
        if not run.is_dir():raise ValueError("existing completed run required")
        writable=True
        rp,cp,mp=run/"report.json",run/"completion.json",run/"observed-media.pt"
        report=json.loads(rp.read_text());receipt=json.loads(cp.read_text())
        bound={"report_sha256":sha(rp),"completion_sha256":sha(cp),"observed_media_sha256":sha(mp)}
        if (not receipt["complete"] or receipt["report_sha256"]!=bound["report_sha256"] or not report["complete"]
            or report["task"]!="routed_caption_untied_up_v1" or report["scientific_status"] not in ("PASS","FAIL") or report["observed_media_sha256"]!=bound["observed_media_sha256"]):raise ValueError("bound completed three-arm campaign required")
        media=torch.load(mp,map_location="cpu",weights_only=True)
        if tuple(media["steps"])!=STEPS or list(media["source_ids"][:6])!=list(range(6)) or not media["capture_native_state_rng_diagnostics_unchanged"] or media["target_residual"].count_nonzero() or set(media["actual_residuals"])!=set(ARMS):raise ValueError("fixed native target/source/media law differs")
        for arm in ARMS:
            if set(media["actual_residuals"][arm])!={str(s) for s in STEPS}:raise ValueError("every media step mandatory")
            for value in media["actual_residuals"][arm].values():
                if tuple(value.shape)!=(8,256,16):raise ValueError("full observed media shape differs")
        vmax=max(float(token_maps(media["actual_residuals"][a]["0"]).max()) for a in ARMS)
        terminal={"scientific_status":report["scientific_status"],**{k:report["accuracy"][a]["rmse"] for k,a in zip(("ordinary","shared","untied"),ARMS)}}
        frames=[]
        for step in STEPS:
            frames.append(goal_frame(media["target_residual"],*(media["actual_residuals"][a][str(step)] for a in ARMS),step=step,vmax=vmax,terminal=terminal));budget()
        gif,png=run/"goal.gif",run/"goal-final.png"
        with gif.open("xb") as handle:frames[0].save(handle,format="GIF",save_all=True,append_images=frames[1:],duration=900,loop=0,disposal=2)
        with png.open("xb") as handle:frames[-1].save(handle,format="PNG")
        budget()
        if sha(__file__)!=source or any(sha(p)!=expected for p,expected in ((rp,bound["report_sha256"]),(cp,bound["completion_sha256"]),(mp,bound["observed_media_sha256"]))):raise ValueError("source/observations changed")
        if torch.cuda.is_initialized():raise AssertionError("CPU renderer initialized CUDA")
        completion={"complete":True,**bound,"renderer_source_sha256":source,"gif_sha256":sha(gif),"png_sha256":sha(png),"media_steps":list(STEPS),"fixed_initial_only_color_limit":vmax,"native_updates":0,"host_forwards":0,"scientific_status_unchanged":report["scientific_status"],"seconds":time.monotonic()-STARTED,"limit_seconds":LIMIT};code=0
    except BaseException as caught:error={"type":type(caught).__name__,"message":str(caught)}
    finally:
        torch.set_num_threads(threads)
        if not torch.equal(before,torch.get_rng_state()):error={"type":"AssertionError","message":"CPU renderer changed caller RNG"}
        if time.monotonic()-STARTED>LIMIT:error={"type":"TimeoutError","message":"render cleanup60s exceeded"}
        if error is not None:completion={"complete":False,"error":error,"seconds":time.monotonic()-STARTED,"limit_seconds":LIMIT};code=2
        if writable:
            path=run/"media-completion.json"
            try:
                with path.open("x") as handle:handle.write(json.dumps(completion,indent=2,allow_nan=False)+"\n")
            except FileExistsError:return 2
            if time.monotonic()-STARTED>LIMIT:
                completion.update(complete=False,error={"type":"TimeoutError","message":"final media receipt60s exceeded"},seconds=time.monotonic()-STARTED)
                path.write_text(json.dumps(completion,indent=2,allow_nan=False)+"\n");code=2
        print(json.dumps(completion,allow_nan=False),flush=True)
    return code


if __name__=="__main__":raise SystemExit(main())
