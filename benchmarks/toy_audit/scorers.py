"""Probe what declared evaluators accept, using explicit synthetic controls.

These are evaluator checks, never trained models or convergence evidence.
They expose density/topology claims that need stronger independent metrics.
"""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from benchmarks.transfer_suite import image_tasks, vector_tasks
from benchmarks.transfer_suite.protocol import requirements
from .capture import write


def accepted(spec, metrics):
    return all(metrics.get(k) is not None and (metrics[k]>=v if op==">=" else metrics[k]<=v)
               for k,op,v in requirements(spec))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args();torch.set_num_threads(1);rows=[]
    for path in sorted(args.artifacts.glob("pr*/capture-index.json")):
        import json
        records=json.loads(path.read_text())
        if not records or records[0]["kind"]!="image":continue
        record=records[0];spec=record["spec"]
        centers=torch.from_numpy(np.load(Path(record["artifact"])/"observations.npz")["templates"])
        cases={"exact_uniform_templates":centers.repeat_interleave(16,0),
               "collapsed_one_template":centers[:1].repeat_interleave(32,0),
               "mean_of_templates":centers.mean(0,keepdim=True).repeat_interleave(32,0),
               "exact_templates_25_75_mass":torch.cat([centers[:1].repeat_interleave(8,0),centers[1:].repeat_interleave(24,0)])}
        # A full corner-pixel flip is a simple negative control. Its rejection
        # does not turn a spatial RMSE into an independent topology/count oracle.
        changed=cases["exact_uniform_templates"].clone()
        flat=changed.flatten(1)
        flat[:,0]=1-flat[:,0]
        cases["one_corner_pixel_flipped"]=changed
        controls={}
        for name,images in cases.items():
            metrics=image_tasks.image_metrics(images,centers,spec["thresholds"])
            controls[name]=dict(accepted=accepted(spec,metrics),metrics=metrics)
        rows.append(dict(name=spec["name"],pr=int(path.parent.name.removeprefix("pr").removesuffix("-adapted")),
                         threshold=spec["thresholds"],controls=controls))
    vectors=[]
    for original in vector_tasks.TASKS+vector_tasks.RESERVED:
        spec=vector_tasks.resolve(deepcopy(original),allow_reserved=True)
        real=vector_tasks.sample_target(spec,4096,torch.Generator().manual_seed(1931),spec["steps"])
        cases={"independent_target_draw":real,"point_at_global_mean":real.mean(0,keepdim=True).expand_as(real)}
        if spec["kind"]=="gaussian_mixture":
            centers=torch.tensor(spec["means"])*vector_tasks.target_scale(spec,spec["steps"])
            ids=torch.multinomial(torch.tensor(spec["masses"]),4096,True,generator=torch.Generator().manual_seed(991))
            cases["centers_only_no_width"]=centers[ids]
            if spec["identifiable"]:
                first=real[(real[:,None]-centers[None]).square().sum(2).argmin(1)==0]
                cases["one_component_only"]=first[torch.arange(4096)%len(first)]
        controls={}
        for name,points in cases.items():
            metrics=vector_tasks.score_samples(points,spec,spec["steps"])
            controls[name]=dict(accepted=accepted(spec,metrics),metrics=metrics)
        vectors.append(dict(name=spec["name"],requirements=requirements(spec),controls=controls))
    write(args.output,dict(scope="synthetic evaluator controls, not training runs",images=rows,vectors=vectors))


if __name__=="__main__":main()
