"""Zero-update latent-code line from already retained E-only movements."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import diagnose as common


def main():
    parser=argparse.ArgumentParser()
    for name in ("runtime-root","raw","unsmoothed-root","smoothed-root","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Refuse to overwrite supplement")
    source_receipt=common.runtime_receipt(args.runtime_root)
    sys.path.insert(0,str(args.runtime_root))
    import torch
    from benchmarks.toy_audit.api_images import WordFixture,_join_words as join,score_words
    from experiments.forge.state import state_digest
    import particlegan.optim.dualnorm as dualnorm
    common.torch,common.WordFixture,common.join,common.score_words,common.state_digest,common.dualnorm = torch,WordFixture,join,score_words,state_digest,dualnorm
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("CUDA physicalGPU1 required, noCPU fallback")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    protocol=json.loads((HERE/"supplement-protocol.json").read_text())
    raw=json.loads((args.raw/"readout.json").read_text())
    started=time.monotonic()
    records=[]
    for endpoint in raw["endpoints"]:
        if time.monotonic()-started>protocol["maximum_runtime_seconds"]:
            raise RuntimeError("Supplement exhausted bound")
        cohort,arm=endpoint["cohort"],endpoint["arm"]["id"]
        source=(args.unsmoothed_root/arm if cohort=="unsmoothed" else args.smoothed_root/("five_word_joint_acquisition--"+arm))
        with common.arm_runtime(endpoint["arm"]):
            fixture,packet,context,checks=common.load_fixture(args.runtime_root,source)
            before=state_digest(context.streams.state_dict())
            first=endpoint["role_probes"]["E"]["first"]
            base=endpoint["exact_uniform_endpoint"]
            e0=torch.tensor(base["encoded_latents"],device="cuda:0",dtype=torch.float32)
            e1=torch.tensor(first["encoded_latents"],device="cuda:0",dtype=torch.float32)
            with torch.no_grad():
                generated=fixture.G(fixture.prior.z)
            points=[]
            for t in protocol["line_parameters"]:
                codes=e0*(1-t)+e1*t
                reconstruction=fixture.G(codes)
                metrics=score_words(generated.repeat_interleave(205,0).detach().cpu().numpy(),reconstruction.detach().cpu().numpy())
                words=fixture.words.repeat_interleave(5,0)
                real_joint=join(words,codes.repeat_interleave(5,0))
                fake_joint=join(generated.repeat(5,1,1),fixture.prior.z.repeat(5,1))
                real_logits,fake_logits=fixture.D(real_joint),fixture.D(fake_joint)
                joint_loss=fixture.loss.joint_g_loss(fake_logits,real_logits)
                d_adv=fixture.loss.d_loss(real_logits,fake_logits)
                penalty=fixture.penalty(fixture.D,real_joint.detach(),fake_joint.detach())
                points.append(dict(t=t,joint_loss=float(joint_loss.detach()),d_adversarial=float(d_adv.detach()),
                    d_penalty=float(penalty.detach()),
                    reconstruction_exact=metrics["metrics"]["reconstruction_exact"],
                    minimum_reconstruction_token_probability=metrics["metrics"]["minimum_reconstruction_token_probability"],
                    full_exact_law_pass=metrics["passed"]))
            errors={}
            for label,point,expected in (("t0",points[0],base),("t1",points[-1],first)):
                errors[label]=abs(point["minimum_reconstruction_token_probability"]-expected["metrics"]["minimum_reconstruction_token_probability"])
                if errors[label]>1e-5 or point["reconstruction_exact"]!=expected["metrics"]["reconstruction_exact"]:
                    raise RuntimeError(f"Retained reconstruction endpoint mismatch:{cohort}/{arm}/{label}")
            if state_digest(context.streams.state_dict())!=before:
                raise RuntimeError("Supplement consumed RNGstream")
            row=dict(cohort=cohort,arm=arm,restore_checks=checks,endpoint_probability_errors=errors,
                points=points,joint_loss_change_t1_minus_t0=points[-1]["joint_loss"]-points[0]["joint_loss"],
                original_exact_goal_pass=base["passed"],first_E_only_exact_goal_pass=first["passed"])
            records.append(row)
            print(json.dumps(dict(event="supplement_endpoint",cohort=cohort,arm=arm,
                joint_loss_change=row["joint_loss_change_t1_minus_t0"],endpoint_match=True)),flush=True)
    common.write(args.output,dict(protocol_id=protocol["id"],protocol_sha256=common.sha(HERE/"supplement-protocol.json"),
        source_sha256=common.sha(__file__),original_raw_readout_sha256=common.sha(args.raw/"readout.json"),
        frozen_runtime_files=source_receipt,records=records,model_updates=0,sampling_draws=0,
        forward_points=len(records)*len(protocol["line_parameters"]),wall_seconds=time.monotonic()-started,
        script_commit=__import__('subprocess').check_output(["git","rev-parse","HEAD"],cwd=HERE,text=True).strip(),
        qualification_input=False))


if __name__=="__main__":
    main()
