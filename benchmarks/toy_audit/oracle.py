"""Independent analytic/mean controls for PR196's four fixed missingness laws."""
import argparse
import importlib.util
from pathlib import Path

import torch

from .capture import write


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--device",default="cpu")
    args=ap.parse_args();torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location("misgan_oracle",args.source/"lib/misgan.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    controls={}
    for mechanism in mod.MECHANISMS:
        problem=mod.Problem(mechanism,20000,10000,0,torch.device(args.device))
        post,draws=mod.bayes_posterior(problem,problem.x_test,problem.m_test,draws=16,
                                      generator=torch.Generator(device=args.device).manual_seed(11))
        controls[mechanism]=dict(ambiguous_rows=int((problem.m_test.sum(1)<=1).sum()),test_rows=10000,
                                  bayes=mod.imputation_metrics(problem,draws,post),
                                  mean=mod.imputation_metrics(problem,mod.mean_impute(problem,problem.x_test,problem.m_test),post))
        print(mechanism,controls[mechanism],flush=True)
    write(args.output,dict(scope="analytic Bayes and deterministic mean controls, no training",draws=16,
                           data_seed=0,draw_seed=11,controls=controls))


if __name__=="__main__":main()
