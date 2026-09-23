"""Check whether float32 sqrt, as used in Adam's denominator, is correctly rounded on this CPU path.

  python3 -m reports.transfer_suite.host_replication.portability.kernels LABEL
"""
import hashlib
import json
import os
import sys

import torch


def main():
    torch.manual_seed(0)
    x = (torch.randn(10467) * torch.logspace(-9, 0, 10467)).square() * .01
    s = x.sqrt()
    p = torch.randn(10467)
    p.grad = x.sqrt().clone()
    torch.optim.Adam([p], lr=.0015, betas=(0., .99)).step()
    digest = lambda t: hashlib.sha256(bytes(t.detach().contiguous().untyped_storage())).hexdigest()[:12]
    print(json.dumps(dict(label=sys.argv[1], capability=torch.backends.cpu.get_cpu_capability(), mkl_cbwr=os.environ.get('MKL_CBWR'),
                          sqrt_not_correctly_rounded=int((s != x.double().sqrt().float()).sum()), elements=len(x),
                          sqrt_digest=digest(s), adam_step_digest=digest(p))), flush=True)


if __name__ == '__main__':
    main()
