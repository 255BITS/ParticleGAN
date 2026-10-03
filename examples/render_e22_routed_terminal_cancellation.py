"""CPU60 saved-data render; optional NumPy/Pillow>=10.1, no native/model imports.

python -m examples.render_e22_routed_terminal_cancellation --run-directory RUN
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

ARMS=("ordinary_BF16","ordinary_FP32","particle_BF16","particle_FP32")
STEPS=(0,64,128,256,384,512)
CARD=Path(__file__).resolve().parents[1]/"docs/e22_routed_terminal_cancellation_v1.json"


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def budget():
    if time.monotonic()-STARTED>60:raise TimeoutError("CPU60 render exceeded")


def maps(value):
    if value.shape!=(8,16,16) or not bool(torch.isfinite(value).all()):raise ValueError("eight actual finite token arrays required")
    return value[:6].double().square().mean(-1).sqrt().reshape(6,4,4).numpy()


def frame(target,values,step,vmax,report):
    image=Image.new("RGB",(1230,1060),"white");draw=ImageDraw.Draw(image)
    font=ImageFont.load_default(size=17);small=ImageFont.load_default(size=14)
    draw.text((20,12),f"Constructed paired-column terminal cancellation | native update {step}/512",fill="black",font=font)
    draw.text((20,42),"Actual physical residual RMS; fixed analytic stress case, not measured Supra geometry",fill="#333333",font=small)
    for row,(label,array) in enumerate(zip(("Target = 0",*ARMS),(target,*values))):
        y=95+row*174;draw.text((10,y+65),label,fill="black",font=small)
        for source,value in enumerate(maps(array)):
            a=np.clip(value/vmax,0,1);rgb=np.stack((np.full_like(a,255),255*(1-a),255*(1-a)),-1).astype(np.uint8)
            x=180+source*170;image.paste(Image.fromarray(rgb).resize((155,155),Image.Resampling.NEAREST),(x,y))
            draw.rectangle((x,y,x+155,y+155),outline="#dddddd")
            if row==0:draw.text((x+38,72),f"Source {source}",fill="black",font=small)
    draw.text((20,980),f"Fixed initial colors: white=0, red={vmax:.7f}. Six cameras; formal score uses all48 TEST rows.",fill="black",font=small)
    label=f"Terminal {report['scientific_status']}: particleFP32 {report['accuracy']['particle_FP32']['rmse']:.8f}; ordinaryFP32 {report['accuracy']['ordinary_FP32']['rmse']:.8f}" if step==512 else "Visual-only fixed schedule; terminal convergence metric is independent of precision witness."
    draw.text((20,1014),label,fill="black",font=small);return image


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run-directory",type=Path,required=True);args=parser.parse_args()
    run=args.run_directory;writable=False;receipt=None;error=None;code=2;source=sha(__file__)
    rng=torch.get_rng_state().clone();threads=torch.get_num_threads()
    try:
        if any((run/n).exists() for n in ("goal.gif","goal-final.png","media-completion.json")):raise ValueError("existing media must remain unchanged")
        rp,cp,mp=(run/n for n in ("report.json","completion.json","observed-media.pt"))
        report=json.loads(rp.read_text());completion=json.loads(cp.read_text());bound={"report_sha256":sha(rp),"completion_sha256":sha(cp),"observed_media_sha256":sha(mp)}
        if not completion["complete"] or completion["report_sha256"]!=bound["report_sha256"] or not report["complete"] or report["task"]!="routed_terminal_cancellation_v1" or report["observed_media_sha256"]!=bound["observed_media_sha256"] or report["scientific_status"]!=("PASS" if report["gate"]["pass"] else "FAIL"):raise ValueError("bound complete PASS/FAIL campaign required")
        card=json.loads(CARD.read_text());card_sha=sha(CARD)
        if report["protocol_sha256"]!=card_sha or report["source_identity"]!=card["sources"] or card["renderer"]["sha256"]!=source:raise ValueError("shipped card/source/renderer bindings differ")
        if not all(math.isfinite(report["accuracy"][a]["rmse"]) and report["accuracy"][a]["rmse"]>=0 for a in (*ARMS,"zero_code")):raise ValueError("finite nonnegative terminal metrics required")
        media=torch.load(mp,map_location="cpu",weights_only=True)
        if media["task"]!=report["task"] or tuple(media["steps"])!=STEPS or media["indices"]!=[0,8,16,24,32,40,1,9] or media["source_ids"]!=[0,1,2,3,4,5,0,1] or not media["captures_immutable"] or media["target_residual"].count_nonzero():raise ValueError("actual fixed target/camera scope differs")
        if set(media["actual_residuals"])!=set(ARMS):raise ValueError("allfourarms mandatory")
        for arm in ARMS:
            if set(media["actual_residuals"][arm])!={str(s) for s in STEPS}:raise ValueError("allfixedframes mandatory")
            for value in media["actual_residuals"][arm].values():maps(value)
        vmax=max(float(maps(media["actual_residuals"][a]["0"]).max()) for a in ARMS)
        if not math.isfinite(vmax) or vmax<=0:raise ValueError("positive initial-only color scale")
        writable=True;torch.set_num_threads(1);frames=[]
        for step in STEPS:
            frames.append(frame(media["target_residual"],[media["actual_residuals"][a][str(step)] for a in ARMS],step,vmax,report));budget()
        with (run/"goal.gif").open("xb") as handle:frames[0].save(handle,format="GIF",save_all=True,append_images=frames[1:],duration=900,loop=0,disposal=2)
        with (run/"goal-final.png").open("xb") as handle:frames[-1].save(handle,format="PNG")
        if sha(__file__)!=source or sha(CARD)!=card_sha or any(sha(p)!=bound[k] for p,k in ((rp,"report_sha256"),(cp,"completion_sha256"),(mp,"observed_media_sha256"))):raise AssertionError("inputs/source changed")
        if torch.cuda.is_initialized():raise AssertionError("CPU renderer initializedCUDA")
        receipt={"complete":True,**bound,"protocol_sha256":card_sha,"renderer_sha256":source,"gif_sha256":sha(run/"goal.gif"),"png_sha256":sha(run/"goal-final.png"),"initial_only_vmax":vmax,"native_updates":0,"model_forwards":0,"scientific_status_unchanged":report["scientific_status"]};code=0
    except BaseException as exc:error={"type":type(exc).__name__,"message":str(exc)}
    finally:
        torch.set_num_threads(threads)
        if not torch.equal(rng,torch.get_rng_state()):error={"type":"AssertionError","message":"renderer callerRNG changed"}
        if time.monotonic()-STARTED>60:error={"type":"TimeoutError","message":"CPU cleanup60s"}
        if error is not None:receipt={"complete":False,"error":error};code=2
        if writable:
            receipt.update(seconds=time.monotonic()-STARTED,limit_seconds=60)
            cp=run/"media-completion.json"
            with cp.open("x") as handle:handle.write(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
            if time.monotonic()-STARTED>60:receipt.update(complete=False,error="post-write60s overrun");cp.write_text(json.dumps(receipt,indent=2)+"\n");code=2
        print(json.dumps(receipt or {"complete":False,"error":error},allow_nan=False),flush=True)
    return code


if __name__=="__main__":raise SystemExit(main())
