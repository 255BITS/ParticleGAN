"""Record CPU vendor and the MKL vendor-dispatch bit. Does not train."""

import ctypes
import json
import os
from pathlib import Path
import sys

import torch


def cpuinfo():
    fields = {}
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        fields.setdefault(key, value)
    return fields


def mkl_intel_cpu():
    torch.zeros(1)  # load MKL before resolving the symbol
    try:
        fn = ctypes.CDLL(None).mkl_serv_intel_cpu_true
    except AttributeError as exc:
        return dict(error=str(exc))
    fn.restype = ctypes.c_int
    return dict(mkl_serv_intel_cpu_true=int(fn()))


def main():
    info = cpuinfo()
    row = dict(
        event="DISPATCH",
        cpu_vendor=info.get("vendor_id"),
        cpu_model=info.get("model name"),
        cpu_family=info.get("cpu family"),
        cpu_model_id=info.get("model"),
        cpu_stepping=info.get("stepping"),
        torch=torch.__version__,
        aten=torch.backends.cpu.get_cpu_capability(),
        audit_seed=os.environ.get("AUDIT_SEED"),
        audit_repo=os.environ.get("AUDIT_REPO"),
        ld_preload=os.environ.get("LD_PRELOAD"),
        mkl_enable_instructions=os.environ.get("MKL_ENABLE_INSTRUCTIONS"),
        aten_cpu_capability=os.environ.get("ATEN_CPU_CAPABILITY"),
        onednn_max_cpu_isa=os.environ.get("ONEDNN_MAX_CPU_ISA"),
        dnnl_max_cpu_isa=os.environ.get("DNNL_MAX_CPU_ISA"),
        threads=dict(torch=torch.get_num_threads(), interop=torch.get_num_interop_threads(),
                     omp=os.environ.get("OMP_NUM_THREADS"), mkl=os.environ.get("MKL_NUM_THREADS")),
        **mkl_intel_cpu(),
    )
    text = json.dumps(row)
    print(text, flush=True)
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")


if __name__ == "__main__":
    main()
